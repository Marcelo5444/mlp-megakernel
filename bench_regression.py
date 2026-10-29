"""Test D: regression on the 9 standard shapes at M_real == bucket_M.
Checks bucketed path (eager + graph) matches main's published timing ranges:
  fwd 6.5-77us, split-K bwd 36-360us on the 4090.
"""
import os, time, sys
os.environ.setdefault("MLP_FAST_AUTOTUNE", "1")
import torch
import cutile_mlp as cm

DEV, DT, D = "cuda", torch.float16, 128
w1, w2, w3 = [torch.randn(D, D, dtype=DT, device=DEV) * 0.02 for _ in range(3)]

SHAPES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
FWD_RANGE = (0.0065, 0.077)   # ms
BWD_RANGE = (0.036, 0.360)    # ms

def timed(fn, iters=200):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

print(f"{'M':>7} {'fwd_us':>9} {'bwd_us':>9}  fwd_range bwd_range  (bucketed eager, M==bucket)")
gm_eager = cm.GraphedBucketedMLP(w1, w2, w3, graph_max_bucket=0)
ok = True
rows = []
for m in SHAPES:
    x = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    dout = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    gm_eager.fwd(x); gm_eager.bwd(dout)  # first-touch
    torch.cuda.synchronize()
    f = timed(lambda: gm_eager.fwd(x))
    b = timed(lambda: gm_eager.bwd(dout))
    fin = FWD_RANGE[0] <= f <= FWD_RANGE[1] * 1.6  # wrapper adds copies; allow 60% slack
    bin_ = BWD_RANGE[0] <= b <= BWD_RANGE[1] * 1.6
    ok &= fin and bin_
    rows.append((m, f * 1000, b * 1000, fin, bin_))
    print(f"{m:>7} {f*1000:>9.1f} {b*1000:>9.1f}  {'ok' if fin else 'OUT':>9} {'ok' if bin_ else 'OUT'}")

# raw exact-M kernel timings (no wrapper) for the published-range check
print("\nraw exact-M fused_mlp path (published-range reference):")
for m in SHAPES:
    x = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    dout = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    out, H0, H1 = cm._fwd_train(x, w1, w2, w3)
    torch.cuda.synchronize()
    f = timed(lambda: cm._fwd_train(x, w1, w2, w3))
    b = timed(lambda: cm.fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1))
    fin = FWD_RANGE[0] * 0.7 <= f <= FWD_RANGE[1] * 1.4
    bin_ = BWD_RANGE[0] * 0.7 <= b <= BWD_RANGE[1] * 1.4
    print(f"{m:>7} fwd {f*1000:>8.1f}us bwd {b*1000:>8.1f}us  {'ok' if fin and bin_ else 'OUT-OF-RANGE'}")

print("\nREGRESSION:", "PASS" if ok else "CHECK ROWS ABOVE")
