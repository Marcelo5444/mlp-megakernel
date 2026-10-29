"""Time full warmup including graph preflight for buckets <= 4096."""
import os, time
os.environ.setdefault("MLP_FAST_AUTOTUNE", "1")
import torch
import cutile_mlp as cm

D, DEV, DT = 128, "cuda", torch.float16
w1, w2, w3 = [torch.randn(D, D, dtype=DT, device=DEV) * 0.02 for _ in range(3)]

t0 = time.time()
times = cm.warmup_buckets(verbose=False)
t1 = time.time()
gm = cm.warmup_graphs(w1, w2, w3, verbose=True)
t2 = time.time()
print(f"CONFIG_WARMUP {t1-t0:.1f}s; GRAPH_PREFLIGHT {t2-t1:.1f}s; TOTAL {t2-t0:.1f}s")

# verify a training-style step has no first-touch cost now
x = torch.randn(700, D, dtype=DT, device=DEV) * 0.02
dout = torch.randn(700, D, dtype=DT, device=DEV) * 0.02
t3 = time.time()
gm.fwd(x); gm.bwd(dout)
torch.cuda.synchronize()
print(f"first training step after warmup: {(time.time()-t3)*1000:.2f} ms")
