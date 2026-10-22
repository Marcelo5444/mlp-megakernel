"""PyTorch manual backward-only vs cuTile split-K backward (bwd-only, fair)."""
import sys
import torch
import torch.nn.functional as F
from triton.testing import do_bench

sys.path.insert(0, "/work")
import cutile_mlp as cm

MS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DEV = "cuda"


def make(M, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w1 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w2 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w3 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    return x, w1, w2, w3


def torch_bwd_only(dout, x, w1, w2, w3):
    h1_pre = x @ w1
    h1 = F.softplus(h1_pre)
    h2_pre = h1 @ w2
    h2 = F.softplus(h2_pre)
    dw3 = h2.t() @ dout
    dh2 = (dout @ w3.t()) * torch.sigmoid(h2_pre)
    dw2 = h1.t() @ dh2
    dh1 = (dh2 @ w2.t()) * torch.sigmoid(h1_pre)
    dw1 = x.t() @ dh1
    dx = dh1 @ w1.t()
    return dx, dw1, dw2, dw3


print(f"{'M':>7} {'torch bwd':>10} {'splitk bwd':>11} {'torch/splitk':>12}")
for M in MS:
    x, w1, w2, w3 = make(M)
    out, H0, H1PRE = cm._fwd_train(x, w1, w2, w3)
    torch.cuda.synchronize()
    g = torch.Generator(device=DEV).manual_seed(42 + M)
    dout = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g)

    t_torch = do_bench(lambda: torch_bwd_only(dout, x, w1, w2, w3), warmup=10, rep=50) * 1e3
    t_splitk = do_bench(lambda: cm.fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1PRE),
                        warmup=10, rep=50) * 1e3
    print(f"{M:7d} {t_torch:9.1f}u {t_splitk:10.1f}u {t_torch/t_splitk:11.2f}x", flush=True)
