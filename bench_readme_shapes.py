"""README benchmark: warmed bucketed path at non-multiple M shapes (fwd + bwd).

Also exports torch.profiler traces (one full training iteration, NO CUDA
graphs): cutile bucketed eager vs pytorch eager, for GUI inspection.
"""
import os, sys
# exact-M column uses heuristic configs (no exhaustive autotune per ad-hoc M);
# bucketed columns use the committed exhaustive disk-cache configs.
os.environ["MLP_FAST_AUTOTUNE"] = "1"
sys.path.insert(0, "/work")
import torch, torch.nn.functional as F
from triton.testing import do_bench
import cutile_mlp as cm

torch.manual_seed(0)
DEV, DT, D = "cuda", torch.float16, 128
SHAPES = [300, 1000, 2500, 5000, 10000, 20000, 40000, 60000]

w1, w2, w3 = [torch.randn(D, D, dtype=DT, device=DEV) * 0.02 for _ in range(3)]

print("warming up (kernels from disk cache + graph capture)...", flush=True)
gm_eager = cm.GraphedBucketedMLP(w1, w2, w3, graph_max_bucket=0)
gm_graph = cm.warmup_graphs(w1, w2, w3, graph_max_bucket=4096, verbose=False)
torch.cuda.synchronize()
print("warm. benchmarking.", flush=True)

print(f"{'M_real':>7} {'bucket':>7} {'pad':>5} | {'exact fwd':>9} {'bktE fwd':>9} {'bktG fwd':>9} | {'exact bwd':>9} {'bktE bwd':>9} {'bktG bwd':>9} | {'torch f+b':>9}")
rows = []
for m in SHAPES:
    x = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    dout = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    b = cm.bucket_for(m)

    out, H0, H1 = cm._fwd_train(x, w1, w2, w3)
    t_ex_f = do_bench(lambda: cm._fwd_train(x, w1, w2, w3), warmup=25, rep=100) * 1e3
    t_ex_b = do_bench(lambda: cm.fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1), warmup=10, rep=50) * 1e3

    gm_eager.fwd(x); gm_eager.bwd(dout)
    t_be_f = do_bench(lambda: gm_eager.fwd(x), warmup=25, rep=100) * 1e3
    t_be_b = do_bench(lambda: gm_eager.bwd(dout), warmup=10, rep=50) * 1e3

    if b <= 4096:
        gm_graph.fwd(x); gm_graph.bwd(dout)
        t_bg_f = do_bench(lambda: gm_graph.fwd(x), warmup=25, rep=100) * 1e3
        t_bg_b = do_bench(lambda: gm_graph.bwd(dout), warmup=10, rep=50) * 1e3
    else:
        t_bg_f = t_bg_b = float("nan")

    def torch_fb():
        xr = x.clone().requires_grad_(True)
        wr = [w.clone().requires_grad_(True) for w in (w1, w2, w3)]
        o = F.softplus(F.softplus(xr @ wr[0]) @ wr[1]) @ wr[2]
        o.backward(dout)
    t_torch = do_bench(torch_fb, warmup=10, rep=50) * 1e3

    rows.append((m, b, b / m, t_ex_f, t_be_f, t_bg_f, t_ex_b, t_be_b, t_bg_b, t_torch))
    print(f"{m:7d} {b:7d} {b/m:4.2f}x | {t_ex_f:8.1f}u {t_be_f:8.1f}u {t_bg_f:8.1f}u | {t_ex_b:8.1f}u {t_be_b:8.1f}u {t_bg_b:8.1f}u | {t_torch:8.1f}u", flush=True)

# correctness spot-check at the awkward shapes
print("\ncorrectness (cos vs eager reference, fwd out / bwd dx,dW1):")
for m in [300, 2500, 10000, 60000]:
    x = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    dout = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    xr = x.clone().requires_grad_(True)
    wr = [w.clone().requires_grad_(True) for w in (w1, w2, w3)]
    ref = F.softplus(F.softplus(xr @ wr[0]) @ wr[1]) @ wr[2]
    ref.backward(dout)
    out = gm_graph.fwd(x) if cm.bucket_for(m) <= 4096 else gm_eager.fwd(x)
    cos_f = F.cosine_similarity(out[:m].flatten().float(), ref.flatten().float(), dim=0).item()
    dx, dw1, dw2, dw3 = (gm_graph.bwd(dout) if cm.bucket_for(m) <= 4096 else gm_eager.bwd(dout))
    cos_dx = F.cosine_similarity(dx[:m].flatten().float(), xr.grad.flatten().float(), dim=0).item()
    cos_dw1 = F.cosine_similarity(dw1.flatten().float(), wr[0].grad.flatten().float(), dim=0).item()
    print(f"  M={m}: fwd={cos_f:.5f} dx={cos_dx:.5f} dW1={cos_dw1:.5f}", flush=True)

# ---- torch.profiler: one training iteration each, NO cuda graphs ----------
M_PROF = 5000
x = torch.randn(M_PROF, D, dtype=DT, device=DEV) * 0.02
dout = torch.randn(M_PROF, D, dtype=DT, device=DEV) * 0.02


def cutile_iter():
    gm_eager.fwd(x)
    gm_eager.bwd(dout)
    torch.cuda.synchronize()


def torch_iter():
    xr = x.clone().requires_grad_(True)
    wr = [w.clone().requires_grad_(True) for w in (w1, w2, w3)]
    o = F.softplus(F.softplus(xr @ wr[0]) @ wr[1]) @ wr[2]
    o.backward(dout)
    torch.cuda.synchronize()


for name, fn in [("cutile_bucketed", cutile_iter), ("pytorch_eager", torch_iter)]:
    fn(); fn()  # warm
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn()
    path = f"/work/trace_{name}_M{M_PROF}.json"
    prof.export_chrome_trace(path)
    print(f"wrote {path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
