"""torch.profiler: 5 training steps using a real torch.nn.Module + loss.backward().

PyTorch side: nn.Linear layers + autograd engine (as it actually happens in a
training loop — autograd engine nodes, backward kernel scheduling).
cuTile side: GraphedBucketedMLP eager path (no CUDA graphs).

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


class MLPSoftplus(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.fc3 = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            for fc in (self.fc1, self.fc2, self.fc3):
                fc.weight.mul_(0.02 / fc.weight.std())

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        return self.fc3(x)


model = MLPSoftplus(D).to(DEV).half()
w1 = model.fc1.weight.t().contiguous().clone()  # nn.Linear stores (out,in); our kernel wants (in,out)
w2 = model.fc2.weight.t().contiguous().clone()
w3 = model.fc3.weight.t().contiguous().clone()

gm_eager = cm.GraphedBucketedMLP(w1, w2, w3, graph_max_bucket=0)

xs = {m: torch.randn(m, D, dtype=DT, device=DEV) * 0.02 for m in STEPS_M}
torch.cuda.synchronize()


def torch_steps():
    for m in STEPS_M:
        out = model(xs[m])
        loss = out.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def cutile_steps():
    for m in STEPS_M:
        gm_eager.fwd(xs[m])
        gm_eager.bwd(torch.ones_like(xs[m]))  # d(out.sum())/dout = 1
    torch.cuda.synchronize()


for name, fn in [("pytorch_module", torch_steps), ("cutile_bucketed", cutile_steps)]:
    fn(); fn()  # warm
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn()
    path = f"/work/trace_{name}_5steps.json"
    prof.export_chrome_trace(path)
    print(f"wrote {path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14))
    print()
