"""Triton bucketed correctness + cold/warm startup timing."""
import os, time, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel as km

torch.manual_seed(0)
DEV, DT, D = "cuda", torch.float16, 128

w1, w2, w3 = [torch.randn(D, D, dtype=DT, device=DEV) * 0.02 for _ in range(3)]

# ---- warmup timing (cold or warm depending on TRITON_CACHE_DIR state) ----
t0 = time.time()
km.warmup_triton_buckets(verbose=False)
t1 = time.time()
print(f"TRITON_WARMUP_TOTAL {t1-t0:.1f}s for {len(km.TRITON_BUCKETS)} buckets")

# ---- correctness: bucketed graph path vs eager exact ----
gm = km.GraphedBucketedTritonMLP(w1, w2, w3, graph_max_bucket=4096)
all_ok = True
for m in (257, 1000, 5000, 31000, 65535, 512, 4096):
    x = torch.randn(m, D, dtype=DT, device=DEV) * 0.02
    ref = F.softplus(F.softplus(x @ w1) @ w2) @ w3
    out_eager = km._fused_mlp_fwd(x, w1, w2, w3)
    out_b = gm.fwd(x)
    torch.cuda.synchronize()
    c1 = F.cosine_similarity(out_b.flatten().float(), ref.flatten().float(), dim=0).item()
    c2 = F.cosine_similarity(out_b.flatten().float(), out_eager.flatten().float(), dim=0).item()
    ok = c1 >= 0.99 and c2 >= 0.99
    all_ok &= ok
    b = km.triton_bucket_for(m)
    mode = "graph" if b <= 4096 else "eager"
    print(f"  M={m} (bucket {b}, {mode}): cos_vs_eager_torch={c1:.5f} cos_vs_kernel={c2:.5f} {'PASS' if ok else 'FAIL'}")

print("TRITON OVERALL:", "ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
