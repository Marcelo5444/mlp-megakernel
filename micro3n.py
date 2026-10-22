"""Micro: where does the split-K path time go? M=65536 / 8192."""
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

for M in (8192, 65536):
    g = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w1 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w2 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w3 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    out, H0, H1PRE = cm._fwd_train(x, w1, w2, w3)
    dout = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g)
    torch.cuda.synchronize()
    dx16 = torch.empty((M, TN), device=DEV, dtype=torch.float16)

    print(f"\n===== M={M} =====")
    TM = 16 if M <= 2048 else 32
    num_m = (M + TM - 1) // TM
    k_dgrad = cm._bwd_splitk_hinted(cm.bwd_dgrad_only_kernel, 1, 8)

    for nb_mult in (1, 2, 3, 4):
        nb = min(nb_mult * num_sms, num_m)
        P = torch.empty((3 * nb * TN, TN), device=DEV, dtype=torch.float32)
        P1, P2, P3 = P[0:nb*TN], P[nb*TN:2*nb*TN], P[2*nb*TN:3*nb*TN]
        def f():
            ct.launch(stream, (nb,), k_dgrad, (
                x, w1, w2, w3, dout, H0, H1PRE, dx16, P1, P2, P3, M,
                TN, TN, TN, TN, TM, TN, 4))
        t = do_bench(f, warmup=10, rep=50) * 1e3
        print(f"dgrad kernel nb={nb} ({nb_mult}xSMs): {t:7.1f} us")

    # reduce sweep over tile shapes at nb=256
    nb = min(2 * num_sms, num_m)
    P = torch.empty((3 * nb * TN, TN), device=DEV, dtype=torch.float32)
    dw32 = torch.empty((TN, TN), device=DEV, dtype=torch.float32)
    k_red = cm._bwd_splitk_hinted(cm.bwd_wgrad_reduce_kernel, 1, None)
    for TR, TC in ((16, 16), (32, 16), (16, 32), (8, 32), (32, 32), (64, 16), (16, 64)):
        grid_r = ((TN // TR) * (TN // TC),)
        t = do_bench(lambda: ct.launch(stream, grid_r, k_red, (P[0:nb*TN], dw32, nb, TR, TC)),
                     warmup=10, rep=50) * 1e3
        print(f"reduce TR={TR} TC={TC} (nb={nb}): {t:7.1f} us")

    # other overheads
    print("P alloc:", round(do_bench(lambda: torch.empty((3 * nb * TN, TN), device=DEV, dtype=torch.float32), warmup=10, rep=50)*1e3, 1), "us")
    a = torch.empty((3 * TN, TN), device=DEV, dtype=torch.float32)
    print("dw32->fp16 cast:", round(do_bench(lambda: a.to(torch.float16), warmup=10, rep=50)*1e3, 1), "us")
