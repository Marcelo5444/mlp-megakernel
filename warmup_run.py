"""Test B: cold vs warm startup for bucket warmup."""
import os, sys, time, json
import torch

mode = os.environ.get("MLP_FAST_AUTOTUNE", "1")
buckets_env = os.environ.get("MLP_BUCKETS", "")
import cutile_mlp as cm
buckets = tuple(int(x) for x in buckets_env.split(",") if x) or cm.DEFAULT_BUCKETS

t1 = time.time()
times = cm.warmup_buckets(buckets=buckets, verbose=True)
torch.cuda.synchronize()
t2 = time.time()

# verify every bucket now has a cache entry
cc = torch.cuda.get_device_capability()
missing = [b for b in buckets
           if cm._disk_key((cc[0], cc[1], b, 128, 128, 128, 128)) not in cm._disk_cache]
print(f"WARMUP_TOTAL {t2-t1:.1f}s for {len(buckets)} buckets; missing_cache={missing}")
