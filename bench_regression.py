"""Test D: regression on the 9 standard shapes at M_real == bucket_M.

Compares the bucketed wrapper path AND the raw exact-M path against main's
published ranges (fwd 6.5-77us, split-K bwd 36-360us on the 4090).
The wrapper adds one D2D copy per tensor vs raw; both are reported, with the
raw path held to the published window (±30% tolerance for run-to-run
variance) and the wrapper allowed extra slack for its copies.
"""
import os, sys
os.environ.setdefault("MLP_FAST_AUTOTUNE", "0")  # exhaustive disk configs
import torch
import cutile_mlp as cm

DEV, DT, D = "cuda", torch.float16, 128
w1, w2, w3 = [torch.randn(D, D, dtype=DT, device=DEV) * 0.02 for _ in range(3)]

SHAPES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
FWD_RANGE = (0.0065, 0.077)   # ms (published)
BWD_RANGE = (0.036, 0.360)    # ms (published)

def timed(fn, iters=300):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

print(f"{'M':>7} {'wrap_fwd':>9} {'wrap_bwd':>9} {'raw_fwd':>8} {'raw_bwd':>8}  raw_in_published_range")
ok_raw = True
for m in SHAPES:
    x = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    dout = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    gm = cm.GraphedBucketedMLP(w1, w2, w3, buckets=(m,), graph_max_bucket=0)
    gm.fwd(x); gm.bwd(dout); torch.cuda.synchronize()
    wf = timed(lambda: gm.fwd(x))
    wb = timed(lambda: gm.bwd(dout))
    out, H0, H1 = cm._fwd_train(x, w1, w2, w3)
    rf = timed(lambda: cm._fwd_train(x, w1, w2, w3))
    rb = timed(lambda: cm.fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1))
    fin = FWD_RANGE[0] * 0.7 <= rf <= FWD_RANGE[1] * 1.3
    bin_ = BWD_RANGE[0] * 0.7 <= rb <= BWD_RANGE[1] * 1.3
    ok_raw &= fin and bin_
    print(f"{m:>7} {wf*1000:>9.1f} {wb*1000:>9.1f} {rf*1000:>8.1f} {rb*1000:>8.1f}  {'ok' if fin and bin_ else 'OUT'}")
print("REGRESSION (raw vs published):", "PASS" if ok_raw else "FAIL")
sys.exit(0 if ok_raw else 1)
