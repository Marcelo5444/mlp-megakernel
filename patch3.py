"""Patch round 3: tiled reduce kernel, grid=2*SMs for dgrad kernel, latency sweep."""
p = "/home/marcelo/mlp-megakernel-wt-splitk/cutile_mlp.py"
s = open(p).read()

# ---- tiled reduce kernel ----
old_red = s[s.index("@ct.kernel\ndef bwd_wgrad_reduce_kernel("):s.index("_BWD_SPLITK_VARIANTS = {}")]
new_red = '''@ct.kernel
def bwd_wgrad_reduce_kernel(
    P, DW,
    SPLIT,
    TR: ct.Constant[int], TC: ct.Constant[int],
):
    """Deterministic tiled reduction: DW = sum_s P[s] (fp32 in, fp32 out).

    P: (SPLIT*T, T) fp32 partials; DW: (T, T) fp32 out for one weight.
    Grid: (T/TR) * (T/TC) blocks, each reduces a (TR, TC) tile.
    """
    pid = ct.bid(0)
    nc = 128 // TC
    pr = pid // nc
    pc = pid % nc
    acc = ct.zeros((TR, TC), ct.float32)
    for s in range(SPLIT):
        acc += ct.load(P, (s * (128 // TR) + pr, pc), shape=(TR, TC))
    ct.store(DW, (pr, pc), acc)


'''
s = s.replace(old_red, new_red)

# ---- wrapper: grid=2*SMs, 2 reduce launches of (8,8) tiles ----
old = '''    # ---- dgrad + wgrad-partial kernel ----
    TM = 64 if M >= 4096 else (16 if M <= 1024 else 32)
    if M < TM:
        TM = max(16, _next_pow2(M))
    grid_d = (min(num_sms, (M + TM - 1) // TM),)
    nb = grid_d[0]
    P = torch.empty((3 * nb * TN, TN), device=x.device, dtype=torch.float32)
    P1, P2, P3 = P[0:nb*TN], P[nb*TN:2*nb*TN], P[2*nb*TN:3*nb*TN]
    k_dgrad = _bwd_splitk_hinted(bwd_dgrad_only_kernel, 1, 8)
    ct.launch(stream, grid_d, k_dgrad, (
        x, w1, w2, w3, dout, H0, H1PRE, dx16, P1, P2, P3, M,
        K1, N1, N2, N3, TM, TN, 4,
    ))

    # ---- reduce ----
    dw32 = torch.empty((3 * TN, TN), device=x.device, dtype=torch.float32)
    k_red = _bwd_splitk_hinted(bwd_wgrad_reduce_kernel, 1, None)
    ct.launch(stream, (1,), k_red, (P1, dw32[0:TN], nb, TN))
    ct.launch(stream, (1,), k_red, (P2, dw32[TN:2*TN], nb, TN))
    ct.launch(stream, (1,), k_red, (P3, dw32[2*TN:3*TN], nb, TN))

    dw = dw32.to(x.dtype)
    return dx16, dw[0:TN], dw[TN:2*TN], dw[2*TN:3*TN]'''
new = '''    # ---- dgrad + wgrad-partial kernel ----
    TM = 16 if M <= 2048 else 32
    if M < TM:
        TM = max(16, _next_pow2(M))
    num_m = (M + TM - 1) // TM
    nb = min(2 * num_sms, num_m)  # 2 blocks/SM halve the reduce pass
    grid_d = (nb,)
    P = torch.empty((3 * nb * TN, TN), device=x.device, dtype=torch.float32)
    P1, P2, P3 = P[0:nb*TN], P[nb*TN:2*nb*TN], P[2*nb*TN:3*nb*TN]
    k_dgrad = _bwd_splitk_hinted(bwd_dgrad_only_kernel, 1, 8)
    ct.launch(stream, grid_d, k_dgrad, (
        x, w1, w2, w3, dout, H0, H1PRE, dx16, P1, P2, P3, M,
        K1, N1, N2, N3, TM, TN, 4,
    ))

    # ---- reduce (tiled, 128 blocks over 2 launches) ----
    dw32 = torch.empty((3 * TN, TN), device=x.device, dtype=torch.float32)
    k_red = _bwd_splitk_hinted(bwd_wgrad_reduce_kernel, 1, None)
    TR, TC = 16, 16
    grid_r = ((TN // TR) * (TN // TC),)
    ct.launch(stream, grid_r, k_red, (P1, dw32[0:TN], nb, TR, TC))
    ct.launch(stream, grid_r, k_red, (P2, dw32[TN:2*TN], nb, TR, TC))
    ct.launch(stream, grid_r, k_red, (P3, dw32[2*TN:3*TN], nb, TR, TC))

    dw = dw32.to(x.dtype)
    return dx16, dw[0:TN], dw[TN:2*TN], dw[2*TN:3*TN]'''
assert old in s
s = s.replace(old, new)

open(p, "w").write(s)
print("patched, lines:", s.count("\n") + 1)
