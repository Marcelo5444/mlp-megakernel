"""Micro round 5: dgrad(TM/lat) sweep, wgrad sweep, reduce tree, launch overhead."""
import sys
import torch
from triton.testing import do_bench
import cuda.tile as ct

sys.path.insert(0, "/work")
import cutile_mlp as cm

DEV = "cuda"
TN = 128
num_sms = cm._get_num_sms()
stream = torch.cuda.current_stream()

M = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
g = torch.Generator(device=DEV).manual_seed(0)
x = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
w1 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
w2 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
w3 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
out, H0, H1PRE = cm._fwd_train(x, w1, w2, w3)
dout = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g)
torch.cuda.synchronize()
dx16 = torch.empty((M, TN), device=DEV, dtype=torch.float16)
dh1 = torch.empty((M, TN), device=DEV, dtype=torch.float16)
nb = min(num_sms, (M + 31) // 32)
P = torch.empty((3 * nb * TN, TN), device=DEV, dtype=torch.float32)
P1, P2, P3 = P[0:nb*TN], P[nb*TN:2*nb*TN], P[2*nb*TN:3*nb*TN]

print(f"M={M} nb={nb}")
print("== dgrad kernel (TM x lat), occ1 nww8 ==")
for tm in (16, 32):
    nb2 = min(num_sms, (M + tm - 1) // tm)
    for lat in (2, 4):
        k = cm._bwd_splitk_hinted(cm.bwd_dgrad_only_kernel, 1, 8)
        def f():
            ct.launch(stream, (nb2,), k, (
                x, w1, w2, w3, dout, H0, H1PRE, dx16, dh1, P1, M,
                TN, TN, TN, TN, tm, TN, lat))
        print(f"  TM={tm} lat={lat} nb={nb2}: {do_bench(f, warmup=10, rep=50)*1e3:7.1f} us")

print("== wgrad act kernel (TM x SPLIT) occ1 nww8 ==")
for tm in (16, 32, 64):
    for split in (64, 128, 256):
        if split > (M + tm - 1) // tm:
            continue
        k = cm._bwd_splitk_hinted(cm.bwd_wgrad_splitk_act_kernel, 1, 8)
        def f():
            ct.launch(stream, (split,), k, (H0, dh1, P2, M, TN, TN, tm, split, 4))
        print(f"  TM={tm:>3} SPLIT={split:>3}: {do_bench(f, warmup=10, rep=50)*1e3:7.1f} us")

print("== wgrad hints (TM=32 SPLIT=128) ==")
for occ, nww in [(1, 4), (1, None), (2, 8), (2, 4)]:
    k = cm._bwd_splitk_hinted(cm.bwd_wgrad_splitk_act_kernel, occ, nww)
    def f():
        ct.launch(stream, (128,), k, (H0, dh1, P2, M, TN, TN, 32, 128, 4))
    print(f"  occ={occ} nww={str(nww):>4}: {do_bench(f, warmup=10, rep=50)*1e3:7.1f} us")

print("== launch overhead: 7 sequential tiny launches ==")
k_red = cm._bwd_splitk_hinted(cm.bwd_wgrad_reduce_kernel, 1, None)
dw32 = torch.empty((TN, TN), device=DEV, dtype=torch.float32)
def seven():
    for _ in range(7):
        ct.launch(stream, (1,), k_red, (P1, dw32, 1, 128, 128))
print(f"  7 launches: {do_bench(seven, warmup=10, rep=50)*1e3:7.1f} us")
