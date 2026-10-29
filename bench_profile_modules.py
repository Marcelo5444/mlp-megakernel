"""torch.profiler: 5 training steps, real nn.Module objects on BOTH sides.

- MLPPyTorch: nn.Linear x3 + Softplus, standard autograd backward.
- MLPCutile:  nn.Module wrapping GraphedBucketedMLP with an autograd.Function
              (forward() calls gm.fwd, backward() calls gm.bwd) — a real
              .backward() graph, no CUDA graphs (eager bucket path).

M varies per step (IDR-style): [60000, 47040, 36864, 28913, 22676].
"""
import os, sys
os.environ["MLP_FAST_AUTOTUNE"] = "1"
sys.path.insert(0, "/work")
import torch, torch.nn as nn, torch.nn.functional as F
import cutile_mlp as cm

torch.manual_seed(0)
DEV, DT, D = "cuda", torch.float16, 128
STEPS_M = [60000, 47040, 36864, 28913, 22676]


class MLPPyTorch(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.fc3 = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            for fc in (self.fc1, self.fc2, self.fc3):
                fc.weight.mul_(0.02 / fc.weight.std())

    def forward(self, x):
        return self.fc3(F.softplus(self.fc2(F.softplus(self.fc1(x)))))


class _CutileFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gm):
        ctx.gm = gm
        ctx.save_for_backward(x)
        ctx.set_materialize_grads(False)
        out = gm.fwd(x).clone()  # own the storage; gm view dies on next call
        return out

    @staticmethod
    def backward(ctx, dout):
        dx, dw1, dw2, dw3 = ctx.gm.bwd(dout.contiguous())
        for p, g in zip(ctx.gm._grad_params, (dw1, dw2, dw3)):
            p.grad = g.clone()
        return dx.clone(), None


class MLPCutile(nn.Module):
    def __init__(self, torch_model):
        super().__init__()
        # share the exact same weights as the torch module
        self.w1 = nn.Parameter(torch_model.fc1.weight.t().contiguous().clone())
        self.w2 = nn.Parameter(torch_model.fc2.weight.t().contiguous().clone())
        self.w3 = nn.Parameter(torch_model.fc3.weight.t().contiguous().clone())
        self._gm = cm.GraphedBucketedMLP(self.w1.detach(), self.w2.detach(),
                                         self.w3.detach(), graph_max_bucket=0)
        self._gm._grad_params = (self.w1, self.w2, self.w3)

    def forward(self, x):
        return _CutileFn.apply(x, self._gm)


model_t = MLPPyTorch(D).to(DEV).half()
model_c = MLPCutile(model_t).to(DEV)

xs = {m: torch.randn(m, D, dtype=DT, device=DEV) * 0.02 for m in STEPS_M}
torch.cuda.synchronize()


def torch_steps():
    for m in STEPS_M:
        x = xs[m].detach().requires_grad_(True)
        out = model_t(x)
        loss = out.sum()
        loss.backward()
        model_t.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def cutile_steps():
    for m in STEPS_M:
        x = xs[m].detach().requires_grad_(True)
        out = model_c(x)
        loss = out.sum()
        loss.backward()
        model_c.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


# sanity: fwd outputs match (grads checked in the repo's own gate)
with torch.no_grad():
    x = xs[STEPS_M[2]]
    ot = model_t(x)
    oc = model_c(x)
print("fwd cos:",
      F.cosine_similarity(ot.flatten().float(), oc.flatten().float(), dim=0).item())

for name, fn in [("MLPPyTorch", torch_steps), ("MLPCutile", cutile_steps)]:
    fn(); fn()  # warm
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        fn()
    path = f"/work/trace_{name}_5steps.json"
    prof.export_chrome_trace(path)
    print(f"wrote {path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print()
