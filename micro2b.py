"""Sweep the fused dgrad+wgrad-partial kernel: TM x (occ,nww) x latency."""
import sys
import torch
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
dx16 = torch.empty((M, TN), device=DEV, dtype=torch.float16)

def bench(tm, occ, nww, lat):
    grid = (min(num_sms, (M + tm - 1) // tm),)
    nb = grid[0]
    P = torch.empty((3 * nb * TN, TN), device=DEV, dtype=torch.float32)
    P1, P2, P3 = P[0:nb*TN], P[nb*TN:2*nb*TN], P[2*nb*TN:3*nb*TN]
    k = cm._bwd_splitk_hinted(cm.bwd_dgrad_only_kernel, occ, nww)
    def f():
        ct.launch(stream, grid, k, (
            x, w1, w2, w3, dout, H0, H1PRE, dx16, P1, P2, P3, M,
            TN, TN, TN, TN, tm, TN, lat))
    return do_bench(f, warmup=5, rep=20) * 1e3

# reduce bench at nb=128
k_red = cm._bwd_splitk_hinted(cm.bwd_wgrad_reduce_kernel, 1, None)
P = torch.empty((128 * TN, TN), device=DEV, dtype=torch.float32)
dw32 = torch.empty((TN, TN), device=DEV, dtype=torch.float32)
print("reduce nb=128 1 launch:", round(do_bench(lambda: ct.launch(stream, (1,), k_red, (P, dw32, 128, TN)), warmup=10, rep=50)*1e3, 1), "us")
for nb in (16, 32, 64):
    print(f"reduce nb={nb}:", round(do_bench(lambda: ct.launch(stream, (1,), k_red, (P, dw32, nb, TN)), warmup=10, rep=50)*1e3, 1), "us")

print("\nfused dgrad+wgrad kernel sweep (M=%d):" % M)
for tm in (16, 32, 64):
    for occ, nww in [(1, 8), (1, 4), (1, None), (2, 8), (2, 4), (4, 4), (4, None)]:
        try:
            t = bench(tm, occ, nww, 4)
            print(f"  TM={tm:>3} occ={occ} nww={str(nww):>4}: {t:7.1f} us")
        except Exception as e:
            print(f"  TM={tm:>3} occ={occ} nww={str(nww):>4}: FAIL {type(e).__name__}")
