"""Per-kernel timing of the split-K backward path + SPLIT sweep."""
import sys
import torch
import torch.nn.functional as F
from triton.testing import do_bench
import cuda.tile as ct

sys.path.insert(0, "/work")
import cutile_mlp as cm

M = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
DEV = "cuda"

g = torch.Generator(device=DEV).manual_seed(0)
x = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
w1 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
w2 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
w3 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
out, H0, H1PRE = cm._fwd_train(x, w1, w2, w3)
dout = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g)
torch.cuda.synchronize()

TN = 128
num_sms = cm._get_num_sms()
stream = torch.cuda.current_stream()
ws = cm._get_bwd_ws(M, TN, x.dtype, x.device)
dh1, dh2, dx16 = ws["dh1"], ws["dh2"], ws["dx16"]

TM = 64 if M >= 4096 else 16
k_dgrad = cm._bwd_splitk_hinted(cm.bwd_dgrad_only_kernel, 1, 8)
grid_d = (min(num_sms, (M + TM - 1) // TM),)

def dgrad():
    ct.launch(stream, grid_d, k_dgrad, (
        x, w1, w2, w3, dout, H0, H1PRE, dx16, dh1, dh2, M,
        TN, TN, TN, TN, TM, TN, 4))

print("dgrad occ1 nww8:", round(do_bench(dgrad, warmup=10, rep=50)*1e3, 1), "us")

for occ, nww in [(1, 4), (2, 4), (2, 8), (4, 4)]:
    k = cm._bwd_splitk_hinted(cm.bwd_dgrad_only_kernel, occ, nww)
    print(f"dgrad occ{occ} nww{nww}:", round(do_bench(lambda: ct.launch(stream, grid_d, k, (
        x, w1, w2, w3, dout, H0, H1PRE, dx16, dh1, dh2, M,
        TN, TN, TN, TN, TM, TN, 4)), warmup=10, rep=50)*1e3, 1), "us")

# wgrad sweep: SPLIT x TM x (occ, nww)
def make_wg(split, tm, occ, nww, act):
    k = cm._bwd_splitk_hinted(
        cm.bwd_wgrad_splitk_act_kernel if act else cm.bwd_wgrad_splitk_kernel, occ, nww)
    st = split * TN
    P = torch.empty((st, TN), device=DEV, dtype=torch.float32)
    grid_w = (split,)
    if act:
        return lambda: ct.launch(stream, grid_w, k, (H0, dh1, P, M, TN, TN, tm, split, 4))
    return lambda: ct.launch(stream, grid_w, k, (x, dh2, P, M, TN, TN, tm, split, 4))

print("\nwgrad dW1 (X^T@dh2, fp16 A):")
for split in (16, 32, 42, 64, 128):
    for tm in (32, 64, 128):
        if tm > max(16, M // split):
            continue
        try:
            t = do_bench(make_wg(split, tm, 1, 4, False), warmup=5, rep=20)
            print(f"  split={split:>3} tm={tm:>3}: {round(t*1e3,1)} us")
        except Exception as e:
            print(f"  split={split:>3} tm={tm:>3}: FAIL {type(e).__name__}")

print("\nwgrad dW2 (softplus(H0)^T@dh1, fp32 PRE load):")
for split in (32, 42, 64):
    for tm in (32, 64, 128):
        try:
            t = do_bench(make_wg(split, tm, 1, 4, True), warmup=5, rep=20)
            print(f"  split={split:>3} tm={tm:>3}: {round(t*1e3,1)} us")
        except Exception as e:
            print(f"  split={split:>3} tm={tm:>3}: FAIL {type(e).__name__}")

# hint sweep on a good config
print("\nwgrad hints (split=42 tm=64):")
for occ, nww in [(1, None), (1, 8), (2, 4), (2, 8), (4, 4)]:
    try:
        t = do_bench(make_wg(42, 64, occ, nww, False), warmup=5, rep=20)
        print(f"  occ={occ} nww={nww}: {round(t*1e3,1)} us")
    except Exception as e:
        print(f"  occ={occ} nww={nww}: FAIL {type(e).__name__}")

# reduce kernel + allocation costs
k_red = cm._bwd_splitk_hinted(cm.bwd_wgrad_reduce_kernel, 1, None)
P = torch.empty((42*TN, TN), device=DEV, dtype=torch.float32)
dw = torch.empty((3*TN, TN), device=DEV, dtype=x.dtype)
print("\nreduce (SPLIT=42):", round(do_bench(lambda: ct.launch(stream, (3,), k_red, (P, dw, 42, TN)), warmup=10, rep=50)*1e3, 1), "us")
print("P alloc:", round(do_bench(lambda: torch.empty((3*42*TN, TN), device=DEV, dtype=torch.float32), warmup=10, rep=50)*1e3, 1), "us")
