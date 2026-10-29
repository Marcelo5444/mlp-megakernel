"""torch.profiler traces of 5 full training steps (fwd+bwd each), no CUDA graphs.

Uses M values from the IDR decay trace so each step has a different shape:
  M = [60000, 47040, 36864, 28913, 22676]
Exports one chrome trace per variant: cutile bucketed eager vs pytorch eager.
"""
import os, sys
os.environ["MLP_FAST_AUTOTUNE"] = "1"
sys.path.insert(0, "/work")
import torch, torch.nn.functional as F
import cutile_mlp as cm

torch.manual_seed(0)
DEV, DT, D = "cuda", torch.float16, 128
STEPS_M = [60000, 47040, 36864, 28913, 22676]

w1, w2, w3 = [torch.randn(D, D, dtype=DT, device=DEV) * 0.02 for _ in range(3)]

gm_eager = cm.GraphedBucketedMLP(w1, w2, w3, graph_max_bucket=0)

xs = {m: torch.randn(m, D, dtype=DT, device=DEV) * 0.02 for m in STEPS_M}
douts = {m: torch.randn(m, D, dtype=DT, device=DEV) * 0.02 for m in STEPS_M}
torch.cuda.synchronize()


def cutile_steps():
    for m in STEPS_M:
        gm_eager.fwd(xs[m])
        gm_eager.bwd(douts[m])
    torch.cuda.synchronize()


def torch_steps():
    for m in STEPS_M:
        xr = xs[m].clone().requires_grad_(True)
        wr = [w.clone().requires_grad_(True) for w in (w1, w2, w3)]
        o = F.softplus(F.softplus(xr @ wr[0]) @ wr[1]) @ wr[2]
        o.backward(douts[m])
    torch.cuda.synchronize()


for name, fn in [("cutile_bucketed", cutile_steps), ("pytorch_eager", torch_steps)]:
    fn(); fn()  # warm (bucket first-touch + cudnn/cublas handles)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn()
    path = f"/work/trace_{name}_5steps.json"
    prof.export_chrome_trace(path)
    print(f"wrote {path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print()
