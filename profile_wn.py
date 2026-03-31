"""Profile eager / compile / fused 3L WN MLP with torch.profiler.

Single Chrome-trace JSON with all 3 variants back-to-back at N=4096.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from fused_mlp import FusedMLPSoftplusWN

D, H, O = 128, 128, 128
M = 4096


class EagerMLPWN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = nn.Parameter(torch.randn(H, D) * 0.1)
        self.g1 = nn.Parameter(torch.ones(H, 1))
        self.v2 = nn.Parameter(torch.randn(H, H) * 0.1)
        self.g2 = nn.Parameter(torch.ones(H, 1))
        self.v3 = nn.Parameter(torch.randn(O, H) * 0.1)
        self.g3 = nn.Parameter(torch.ones(O, 1))

    def _wn(self, v, g):
        return g * (v / v.norm(dim=1, keepdim=True))

    def forward(self, x):
        w1, w2, w3 = self._wn(self.v1, self.g1), self._wn(self.v2, self.g2), self._wn(self.v3, self.g3)
        h1 = F.softplus(x @ w1.t())
        h2 = F.softplus(h1 @ w2.t())
        return h2 @ w3.t()


def train_step(model, x):
    model.zero_grad(set_to_none=True)
    out = model(x)
    out.sum().backward()


def main():
    import os
    os.makedirs("torch_compile_debug", exist_ok=True)

    eager = EagerMLPWN3().cuda().half()
    compile_model = torch.compile(EagerMLPWN3().cuda().half(), mode="max-autotune")
    fused = FusedMLPSoftplusWN(D, H, O).cuda().half()

    x = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.1

    # Trigger JIT compilation outside profiling
    train_step(eager, x)
    train_step(compile_model, x)
    train_step(fused, x)
    torch.cuda.synchronize()
    print("JIT compilation done\n")

    out_path = "torch_compile_debug/profile_wn_3layer.json"

    # Run compile outside the trace so CUDAGraph is already captured
    train_step(compile_model, x)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("EAGER"):
            train_step(eager, x)
            torch.cuda.synchronize()

        with torch.profiler.record_function("COMPILE"):
            train_step(compile_model, x)
            torch.cuda.synchronize()

        with torch.profiler.record_function("FUSED"):
            train_step(fused, x)
            torch.cuda.synchronize()

    prof.export_chrome_trace(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
