"""Test C: IDR-style training simulation — M shrinks every iteration.

Variants:
  (i)   current main eager exact-M (fast autotune = heuristic configs;
        exhaustive autotune per new M would add ~15-25 min per size and is
        reported separately as the cold-start compile-storm cost)
  (ii)  bucketed eager
  (iii) bucketed + CUDA graphs for buckets <= graph_max
"""
import os, sys, time, random
os.environ.setdefault("MLP_FAST_AUTOTUNE", "1")
import torch, torch.nn.functional as F
import cutile_mlp as cm

torch.manual_seed(0)
random.seed(0)
DEV, DT, D = "cuda", torch.float16, 128

w1, w2, w3 = [torch.randn(D, D, dtype=DT, device=DEV) * 0.02 for _ in range(3)]

# ---- M trace: 32768 decaying to ~512, 500 steps, with jitter ------------
STEPS = 500
trace = []
for t in range(STEPS):
    base = max(512, round(32768 * 0.98 ** t))
    jitter = random.randint(-int(base * 0.05), int(base * 0.05))
    m = max(512, min(65536, base + jitter))
    trace.append(m)
xs = {m: torch.randn(m, D, dtype=DT, device=DEV) * 0.02 for m in sorted(set(trace))}
douts = {m: torch.randn(m, D, dtype=DT, device=DEV) * 0.02 for m in sorted(set(trace))}
torch.cuda.synchronize()

def exact_step(m):
    """Variant (i): main's eager path at exact M."""
    x, dout = xs[m], douts[m]
    out, H0, H1 = cm._fwd_train(x, w1, w2, w3)
    cm.fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1)

gm_eager = cm.GraphedBucketedMLP(w1, w2, w3, graph_max_bucket=0)  # (ii) eager only
gm_graph = cm.warmup_graphs(w1, w2, w3, graph_max_bucket=4096, verbose=False)  # (iii) preflighted

def bucketed_step(m, gm):
    x, dout = xs[m], douts[m]
    gm.fwd(x)
    gm.bwd(dout)

import collections
def run_inst(name, fn, steps=STEPS):
    """Run the sim; report steady-state total vs first-touch overhead."""
    first_ms = {}
    t_first = 0.0
    t0 = time.perf_counter()
    for m in trace[:steps]:
        ta = time.perf_counter()
        fn(m)
        tb = time.perf_counter()
        b = cm.bucket_for(m)
        if b not in first_ms:
            first_ms[b] = (tb - ta) * 1000
            t_first += tb - ta
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{name}: {dt:.3f}s total ({dt/steps*1000:.2f} ms/step); "
          f"first-touch total {t_first:.2f}s across {len(first_ms)} buckets")
    for b in sorted(first_ms):
        print(f"    first-touch bucket {b}: {first_ms[b]:.1f} ms")
    return dt

def run(name, fn, steps=STEPS, warm=5):
    # warmup: touch every distinct size once (excl. from timing) to keep the
    # measurement about steady-state stepping, like a real training loop after
    # a few iterations. Compile/autotune cost is measured separately.
    for m in list(dict.fromkeys(trace[:8])):
        fn(m)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for m in trace[:steps]:
        fn(m)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{name}: {dt:.3f}s total ({dt/steps*1000:.2f} ms/step)")
    return dt

# (i) exact-M eager — sizes appear once each mostly; fast autotune pays a
# heuristic pick + replace_hints compile on FIRST use of every new M.
# Measure WITH those first-use storms to show the problem IDR hits.
i_storms = 0
seen = set()
t0 = time.perf_counter()
for m in trace:
    first = m not in seen
    if first:
        seen.add(m)
        i_storms += 1
    exact_step(m)
torch.cuda.synchronize()
t_i = time.perf_counter() - t0
print(f"(i) exact-M eager (incl. {i_storms} first-size compile storms): {t_i:.3f}s")

# steady-state re-run (all sizes hot now)
t_iss = run("(i) exact-M eager steady-state", exact_step)

t_ii = run_inst("(ii) bucketed eager", lambda m: bucketed_step(m, gm_eager))
t_iii = run_inst("(iii) bucketed + graphs (<=4096)", lambda m: bucketed_step(m, gm_graph))

# ---- per-phase breakdown at representative M ---------------------------
print("\n--- per-M breakdown (fwd / bwd, ms) ---")

def timed(fn, iters=200):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e2.record(); torch.cuda.synchronize()
    return s.elapsed_time(e2) / iters

for m in (65536, 4096, 1024, 512, 256):
    x, dout = xs.get(m), douts.get(m)
    if x is None:
        x = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
        dout = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    row = [f"M={m}"]
    # exact eager
    f = timed(lambda: cm._fwd_train(x, w1, w2, w3))
    out, H0, H1 = cm._fwd_train(x, w1, w2, w3)
    b = timed(lambda: cm.fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1))
    row.append(f"exact fwd {f:.3f} bwd {b:.3f}")
    # bucketed eager == exact here (m aligned or same loop); use gm wrappers:
    bkt = cm.bucket_for(m)
    gm0 = cm.GraphedBucketedMLP(w1, w2, w3, buckets=(bkt,), graph_max_bucket=0)
    f2 = timed(lambda: gm0.fwd(x))
    b2 = timed(lambda: gm0.bwd(dout))
    row.append(f"bkt-eager fwd {f2:.3f} bwd {b2:.3f}")
    if m <= 8192:
        gm1 = cm.GraphedBucketedMLP(w1, w2, w3, buckets=(bkt,), graph_max_bucket=8192)
        f3 = timed(lambda: gm1.fwd(x))
        b3 = timed(lambda: gm1.bwd(dout))
        row.append(f"bkt-graph fwd {f3:.3f} bwd {b3:.3f}")
    print("  " + " | ".join(row))

print(f"\nSUMMARY steps={STEPS}: (i) {t_i:.2f}s [steady {t_iss:.2f}s], (ii) {t_ii:.2f}s, (iii) {t_iii:.2f}s")
