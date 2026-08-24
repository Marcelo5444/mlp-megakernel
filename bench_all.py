"""Benchmark: PyTorch eager vs Triton megakernel vs cuTile (full autotune), across M shapes."""
import os, sys, time
os.environ.pop("MLP_FAST_AUTOTUNE", None)  # force full exhaustive autotune
sys.path.insert(0, "/home/marcelo/mlp-megakernel")
import torch
import torch.nn.functional as F

torch.manual_seed(0)
D, H, N3 = 128, 128, 128

def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters  # us

def pytorch_ref(x, w1, w2, w3):
    return F.softplus(F.softplus(x @ w1) @ w2) @ w3

from kernel import ModelNew as TritonMLP
from cutile_mlp import fused_mlp_fwd_cutile

triton_model = TritonMLP()

Ms = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
rows = []
for M in Ms:
    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02

    ref = pytorch_ref(x, w1, w2, w3)

    # correctness
    out_t = triton_model(x, w1, w2, w3)
    out_c = fused_mlp_fwd_cutile(x, w1, w2, w3)
    torch.cuda.synchronize()
    ok_t = torch.allclose(out_t, ref, atol=1e-2, rtol=1e-2)
    ok_c = torch.allclose(out_c, ref, atol=1e-2, rtol=1e-2)

    iters = max(20, min(200, int(2e8 / (M * 128 * 128 * 3))))  # scale down for big M
    t_p = bench(lambda: pytorch_ref(x, w1, w2, w3), iters=iters)
    t_t = bench(lambda: triton_model(x, w1, w2, w3), iters=iters)
    t_c = bench(lambda: fused_mlp_fwd_cutile(x, w1, w2, w3), iters=iters)

    rows.append((M, t_p, t_t, t_c, ok_t, ok_c))
    print(f"M={M:6d}  pytorch={t_p:8.1f}us  triton={t_t:8.1f}us  cutile={t_c:8.1f}us"
          f"  triton_speedup={t_p/t_t:.2f}x  cutile_speedup={t_p/t_c:.2f}x  ok={ok_t}/{ok_c}", flush=True)

print("\n=== SUMMARY ===")
print(f"{'M':>7} {'pytorch':>10} {'triton':>10} {'cutile':>10} {'T spd':>6} {'C spd':>6} {'C vs T':>7}")
for M, tp, tt, tc, okt, okc in rows:
    print(f"{M:7d} {tp:9.1f}u {tt:9.1f}u {tc:9.1f}u {tp/tt:5.2f}x {tp/tc:5.2f}x {tt/tc:6.2f}x"
          + ("" if (okt and okc) else "  !!CORRECTNESS"))
