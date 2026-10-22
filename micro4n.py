"""Latency sweep on the dgrad+wgrad kernel + old fused kernel same-session."""
import sys
import torch
from triton.testing import do_bench
import cuda.tile as ct
from cuda.tile import ByTarget

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

TM = 32
num_m = (M + TM - 1) // TM
nb = min(num_sms, num_m)
P = torch.empty((3 * nb * TN, TN), device=DEV, dtype=torch.float32)
P1, P2, P3 = P[0:nb*TN], P[nb*TN:2*nb*TN], P[2*nb*TN:3*nb*TN]

print(f"M={M}, dgrad kernel nb={nb} TM={TM}")
for lat in (1, 2, 4, 8):
    k = cm._bwd_splitk_hinted(cm.bwd_dgrad_only_kernel, 1, 8)
    def f():
        ct.launch(stream, (nb,), k, (
            x, w1, w2, w3, dout, H0, H1PRE, dx16, P1, P2, P3, M,
            TN, TN, TN, TN, TM, TN, lat))
    print(f"  lat={lat}: {do_bench(f, warmup=10, rep=50)*1e3:7.1f} us")

dx32 = torch.zeros((M, TN), device=DEV, dtype=torch.float32)
dw = [torch.zeros((TN, TN), device=DEV, dtype=torch.float32) for _ in range(3)]
k_old = cm.bwd_fused_kernel.replace_hints(occupancy=1, num_worker_warps=8,
    num_ctas=ByTarget(sm_90=1, sm_100=1, sm_120=1, sm_121=1, sm_89=1))
for tm_old in (16, 32):
    grid = (min(num_sms, (M + tm_old - 1) // tm_old),)
    def fo():
        ct.launch(stream, grid, k_old, (
            x, w1, w2, w3, dout, H0, H1PRE, dx32, dw[0], dw[1], dw[2], M,
            TN, TN, TN, TN, tm_old, TN, TN))
    print(f"OLD fused kernel TM={tm_old}: {do_bench(fo, warmup=10, rep=50)*1e3:7.1f} us")
