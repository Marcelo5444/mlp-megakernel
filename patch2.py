"""Patch round 2: fused dgrad+wgrad-epilogue dgrad kernel, fp32 reduce output,
persistent wgrad kernels, config table."""
p = "/home/marcelo/mlp-megakernel-wt-splitk/cutile_mlp.py"
s = open(p).read()

# ---- 1. dgrad kernel: add per-block wgrad epilogue (like bwd_fused but dW
# ----    partials stored per block, deterministic reduce after) ----
old_dgrad_kernel = s[s.index("@ct.kernel\ndef bwd_dgrad_only_kernel("):s.index("@ct.kernel\ndef bwd_wgrad_splitk_kernel(")]
new_dgrad_kernel = '''@ct.kernel
def bwd_dgrad_only_kernel(
    X, W1, W2, W3, DOUT, H0, H1PRE, DX, P1, P2, P3,
    M,
    K1: ct.Constant[int], N1: ct.Constant[int], N2: ct.Constant[int], N3: ct.Constant[int],
    TM: ct.Constant[int], TN: ct.Constant[int],
    LATENCY: ct.Constant[int],
):
    """Persistent kernel: dh1/dh2/dx per M-tile + per-block wgrad partials.

    The 3 wgrad operands (h2=softplus(H1_PRE), h1act=softplus(H0)) are already
    computed here for the dgrad chain, so the wgrad mma's ride along nearly
    free. Each block stores its own 128x128 partial per weight to
    P1/P2/P3 (num_blocks x 128 x 128 fp32) — a later reduce kernel sums them
    deterministically (no atomics, no register-resident-across-kernel state
    beyond this block's own partials).
    """
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_m = ct.cdiv(M, TM)

    dw3_acc = ct.zeros((TN, TN), ct.float32)
    dw2_acc = ct.zeros((TN, TN), ct.float32)
    dw1_acc = ct.zeros((TN, TN), ct.float32)

    for pm in range(raw_pid, num_m, num_pids):
        # dh1 = (dout @ W3^T) * sigmoid(H1_PRE)
        dout_t = ct.load(DOUT, (pm, 0), shape=(TM, TN), latency=LATENCY)
        w3_t = ct.load(W3, (0, 0), shape=(TN, TN), latency=LATENCY)
        w3_t = ct.transpose(w3_t)
        dh1_acc = ct.zeros((TM, TN), ct.float32)
        dh1_acc = ct.mma(dout_t, w3_t, dh1_acc)
        h1pre_t = ct.load(H1PRE, (pm, 0), shape=(TM, TN), latency=LATENCY)
        dh1 = dh1_acc * softplus_grad_tile(h1pre_t)
        dh1_f16 = ct.astype(dh1, ct.float16)

        # dW3 += softplus(H1_PRE)^T @ dout
        h2 = softplus_tile(h1pre_t)
        h2_t = ct.transpose(ct.astype(h2, ct.float16))
        dw3_acc = ct.mma(h2_t, dout_t, dw3_acc)

        # dh2 = (dh1 @ W2^T) * sigmoid(H0)
        w2_t = ct.load(W2, (0, 0), shape=(TN, TN), latency=LATENCY)
        w2_t = ct.transpose(w2_t)
        dh2_acc = ct.zeros((TM, TN), ct.float32)
        dh2_acc = ct.mma(dh1_f16, w2_t, dh2_acc)
        h0_t = ct.load(H0, (pm, 0), shape=(TM, TN), latency=LATENCY)
        dh2 = dh2_acc * softplus_grad_tile(h0_t)
        dh2_f16 = ct.astype(dh2, ct.float16)

        # dW2 += softplus(H0)^T @ dh1
        h1a = softplus_tile(h0_t)
        h1a_t = ct.transpose(ct.astype(h1a, ct.float16))
        dw2_acc = ct.mma(h1a_t, dh1_f16, dw2_acc)

        # dx = dh2 @ W1^T ; dW1 += X^T @ dh2
        x_t = ct.load(X, (pm, 0), shape=(TM, TN), latency=LATENCY)
        w1_t = ct.load(W1, (0, 0), shape=(TN, TN), latency=LATENCY)
        w1_t = ct.transpose(w1_t)
        dx_acc = ct.zeros((TM, TN), ct.float32)
        dx_acc = ct.mma(dh2_f16, w1_t, dx_acc)
        ct.store(DX, (pm, 0), ct.astype(dx_acc, ct.float16))
        x_tT = ct.transpose(x_t)
        dw1_acc = ct.mma(x_tT, dh2_f16, dw1_acc)

    # flush this block's partials (plain stores; block raw_pid owns row raw_pid)
    ct.store(P3, (raw_pid, 0), dw3_acc)
    ct.store(P2, (raw_pid, 0), dw2_acc)
    ct.store(P1, (raw_pid, 0), dw1_acc)


'''
s = s.replace(old_dgrad_kernel, new_dgrad_kernel)

# ---- 2. persistent wgrad kernels (strided pm loop) ----
old_wg = s[s.index("@ct.kernel\ndef bwd_wgrad_splitk_kernel("):s.index("@ct.kernel\ndef bwd_wgrad_reduce_kernel(")]
new_wg = '''@ct.kernel
def bwd_wgrad_splitk_kernel(
    A, B, P,
    M,
    KA: ct.Constant[int], NB: ct.Constant[int],
    TM: ct.Constant[int], SPLIT: ct.Constant[int],
    LATENCY: ct.Constant[int],
):
    """Split-K weight-grad partial: P[sid] = A_chunk^T @ B_chunk.

    A: (M, KA) fp16 activation source; B: (M, NB) fp16 gradient source.
    P: (SPLIT*KA, NB) fp32 partials buffer (block sid owns row-block sid).
    Persistent: block sid processes M-tiles sid, sid+SPLIT, ... so large
    SPLIT values give fine-grained K parallelism without idle tail blocks.
    """
    sid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_m = ct.cdiv(M, TM)

    acc = ct.zeros((KA, NB), ct.float32)
    for pm in range(sid, num_m, num_pids):
        a_t = ct.load(A, (pm, 0), shape=(TM, KA), latency=LATENCY)
        a_t = ct.transpose(a_t)
        b_t = ct.load(B, (pm, 0), shape=(TM, NB), latency=LATENCY)
        acc = ct.mma(a_t, b_t, acc)
    ct.store(P, (sid, 0), acc)


@ct.kernel
def bwd_wgrad_splitk_act_kernel(
    PRE, B, P,
    M,
    KA: ct.Constant[int], NB: ct.Constant[int],
    TM: ct.Constant[int], SPLIT: ct.Constant[int],
    LATENCY: ct.Constant[int],
):
    """Same as bwd_wgrad_splitk_kernel but A = fp16(softplus(PRE)) computed
    on the fly from the fp32 pre-activation tensor (H0 / H1_PRE)."""
    sid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_m = ct.cdiv(M, TM)

    acc = ct.zeros((KA, NB), ct.float32)
    for pm in range(sid, num_m, num_pids):
        pre_t = ct.load(PRE, (pm, 0), shape=(TM, KA), latency=LATENCY)
        a = softplus_tile(pre_t)
        a_f16 = ct.astype(a, ct.float16)
        a_t = ct.transpose(a_f16)
        b_t = ct.load(B, (pm, 0), shape=(TM, NB), latency=LATENCY)
        acc = ct.mma(a_t, b_t, acc)
    ct.store(P, (sid, 0), acc)


'''
s = s.replace(old_wg, new_wg)

# ---- 3. reduce kernel: fp32 output (accuracy; cast once in wrapper) ----
old_red = s[s.index("@ct.kernel\ndef bwd_wgrad_reduce_kernel("):s.index("_BWD_SPLITK_VARIANTS = {}")]
new_red = '''@ct.kernel
def bwd_wgrad_reduce_kernel(
    P, DW,
    SPLIT: ct.Constant[int],
    T: ct.Constant[int],
):
    """Deterministic reduction: DW[pid] = sum_s P[s] (fp32 in, fp32 out).

    P: (SPLIT*T, T) fp32 partials; DW: (n*T, T) fp32 out, row block pid.
    """
    pid = ct.bid(0)
    acc = ct.zeros((T, T), ct.float32)
    for s in range(SPLIT):
        acc += ct.load(P, (s, 0), shape=(T, T))
    ct.store(DW, (pid, 0), acc)


'''
s = s.replace(old_red, new_red)

# ---- 4. new wrapper ----
old_wrap = s[s.index("def fused_mlp_bwd_splitk("):s.index("@ct.kernel\ndef bwd_fused_kernel(")]
new_wrap = '''def fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1PRE, SPLIT=None):
    """Split-K backward.

    dgrad kernel: dh1/dh2/dx per M-tile, fused per-block wgrad partials
    (P1/P2/P3 fp32, one 128x128 partial per block per weight).
    reduce kernel: deterministic sum of partials -> fp32 dW -> cast fp16.

    Deterministic (no atomics). Returns (dx, dW1, dW2, dW3) in x.dtype.
    """
    M, K1 = x.shape
    N1 = w1.shape[1]
    N2 = w2.shape[1]
    N3 = w3.shape[1]
    assert K1 == N1 == N2 == N3
    TN = K1

    dout = dout.contiguous()
    x = x.contiguous()
    H0 = H0.contiguous()
    H1PRE = H1PRE.contiguous()

    num_sms = _get_num_sms()
    stream = torch.cuda.current_stream()
    ws = _get_bwd_ws(M, K1, x.dtype, x.device)
    dx16 = ws["dx16"]

    # ---- dgrad + wgrad-partial kernel ----
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
    return dx16, dw[0:TN], dw[TN:2*TN], dw[2*TN:3*TN]


'''
s = s.replace(old_wrap, new_wrap)

open(p, "w").write(s)
print("patched, lines:", s.count("\n") + 1)
