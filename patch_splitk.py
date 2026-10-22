"""Apply split-K wgrad patch to cutile_mlp.py (run on builder)."""
import re

p = "/home/marcelo/mlp-megakernel-wt-splitk/cutile_mlp.py"
src = open(p).read()

# ------------------------------------------------------------------
# 1. New kernels + reduction, inserted right before bwd_fused_kernel
# ------------------------------------------------------------------
new_kernels = '''

# ============================================================
# Split-K backward: dgrad-only kernel + 3 split-K wgrad kernels
# + deterministic reduction kernel.
#
# The 3 wgrad GEMMs (dW = act^T @ grad, contraction over K=M) are
# tiled across the batch dim: each block accumulates a partial
# 128x128 dW in registers over its K-chunk, stores the partial to
# a fp32 partials buffer, and a small reduce kernel sums the
# partials (deterministic, no 128-way atomics).
# The 3 dgrad GEMMs (+dx) stay in a fwd-shaped persistent kernel
# (no register-resident wgrad accumulators -> normal occupancy).
# ============================================================

@ct.kernel
def bwd_dgrad_only_kernel(
    X, W1, W2, W3, DOUT, H0, H1PRE, DX, DH1, DH2,
    M,
    K1: ct.Constant[int], N1: ct.Constant[int], N2: ct.Constant[int], N3: ct.Constant[int],
    TM: ct.Constant[int], TN: ct.Constant[int],
    LATENCY: ct.Constant[int],
):
    """Persistent kernel, fwd-shaped: dh1, dh2, dx per M-tile.

    Also stores DH1/DH2 (fp16, Mx128) for the wgrad kernels.
    dx written with plain stores (each pm owned by one block).
    """
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_m = ct.cdiv(M, TM)

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
        ct.store(DH1, (pm, 0), dh1_f16)

        # dh2 = (dh1 @ W2^T) * sigmoid(H0)
        w2_t = ct.load(W2, (0, 0), shape=(TN, TN), latency=LATENCY)
        w2_t = ct.transpose(w2_t)
        dh2_acc = ct.zeros((TM, TN), ct.float32)
        dh2_acc = ct.mma(dh1_f16, w2_t, dh2_acc)
        h0_t = ct.load(H0, (pm, 0), shape=(TM, TN), latency=LATENCY)
        dh2 = dh2_acc * softplus_grad_tile(h0_t)
        dh2_f16 = ct.astype(dh2, ct.float16)
        ct.store(DH2, (pm, 0), dh2_f16)

        # dx = dh2 @ W1^T
        w1_t = ct.load(W1, (0, 0), shape=(TN, TN), latency=LATENCY)
        w1_t = ct.transpose(w1_t)
        dx_acc = ct.zeros((TM, TN), ct.float32)
        dx_acc = ct.mma(dh2_f16, w1_t, dx_acc)
        ct.store(DX, (pm, 0), dx_acc)


@ct.kernel
def bwd_wgrad_splitk_kernel(
    A, B, P,
    M,
    KA: ct.Constant[int], NB: ct.Constant[int],
    TM: ct.Constant[int], SPLIT: ct.Constant[int],
    LATENCY: ct.Constant[int],
):
    """Split-K weight-grad partial: P[s] = A_chunk^T @ B_chunk.

    A: (M, KA) fp16 activation tile source (x, softplus(H0), softplus(H1_PRE))
    B: (M, NB) fp16 gradient tile source (dh2, dh1, dout)
    P: (SPLIT, KA, NB) fp32 partials buffer

    Grid is 1-D over split ids; block s accumulates rows
    [s*chunk, (s+1)*chunk) in TM-sized K-chunks.
    """
    sid = ct.bid(0)
    num_m = ct.cdiv(M, TM)
    per = ct.cdiv(num_m, SPLIT)
    start = sid * per
    end = min(start + per, num_m)

    acc = ct.zeros((KA, NB), ct.float32)
    for pm in range(start, end):
        a_t = ct.load(A, (pm, 0), shape=(TM, KA), latency=LATENCY)
        a_t = ct.transpose(a_t)
        b_t = ct.load(B, (pm, 0), shape=(TM, NB), latency=LATENCY)
        acc = ct.mma(a_t, b_t, acc)
    ct.store(P, (sid, 0, 0), acc)


@ct.kernel
def bwd_wgrad_splitk_act_kernel(
    PRE, B, P,
    M,
    KA: ct.Constant[int], NB: ct.Constant[int],
    TM: ct.Constant[int], SPLIT: ct.Constant[int],
    LATENCY: ct.Constant[int],
):
    """Same as bwd_wgrad_splitk_kernel but A = softplus(PRE) computed on the
    fly from an fp32 pre-activation tensor (H0 / H1_PRE)."""
    sid = ct.bid(0)
    num_m = ct.cdiv(M, TM)
    per = ct.cdiv(num_m, SPLIT)
    start = sid * per
    end = min(start + per, num_m)

    acc = ct.zeros((KA, NB), ct.float32)
    for pm in range(start, end):
        pre_t = ct.load(PRE, (pm, 0), shape=(TM, KA), latency=LATENCY)
        a = softplus_tile(pre_t)
        a_f16 = ct.astype(a, ct.float16)
        a_t = ct.transpose(a_f16)
        b_t = ct.load(B, (pm, 0), shape=(TM, NB), latency=LATENCY)
        acc = ct.mma(a_t, b_t, acc)
    ct.store(P, (sid, 0, 0), acc)


@ct.kernel
def bwd_wgrad_reduce_kernel(
    P, DW,
    SPLIT: ct.Constant[int],
    T: ct.Constant[int],
):
    """Deterministic reduction: DW[pid] = sum_s P[s, pid] (fp32 -> fp16)."""
    pid = ct.bid(0)
    acc = ct.zeros((T, T), ct.float32)
    for s in range(SPLIT):
        acc += ct.load(P, (s, pid, 0), shape=(T, T))
    ct.store(DW, (pid, 0, 0), ct.astype(acc, ct.float16))


_BWD_SPLITK_VARIANTS = {}


def _bwd_splitk_hinted(kernel, occ, nww):
    key = (kernel, occ, nww)
    k = _BWD_SPLITK_VARIANTS.get(key)
    if k is None:
        hints = {"occupancy": occ}
        if nww is not None:
            hints["num_worker_warps"] = nww
        k = kernel.replace_hints(**hints)
        _BWD_SPLITK_VARIANTS[key] = k
    return k


# Cached gradient workspaces keyed by (M, dtype, device) — kills the per-call
# torch.zeros + fp32->fp16 cast overhead of the old wrapper.
_BWD_WS = {}


def _get_bwd_ws(M, K, dtype, device):
    key = (M, K, dtype, str(device))
    ws = _BWD_WS.get(key)
    if ws is None:
        ws = {
            "dh1": torch.empty((M, K), device=device, dtype=torch.float16),
            "dh2": torch.empty((M, K), device=device, dtype=torch.float16),
            "dx16": torch.empty((M, K), device=device, dtype=torch.float16),
            "dx32": torch.empty((M, K), device=device, dtype=torch.float32),
        }
        _BWD_WS[key] = ws
    return ws


def fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1PRE, SPLIT=None):
    """Split-K backward: dgrad kernel + 3 split-K wgrad kernels + reduce.

    Returns (dx, dW1, dW2, dW3) in x.dtype. Deterministic (partials buffer +
    reduce kernel, no atomics).
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
    dh1, dh2, dx16 = ws["dh1"], ws["dh2"], ws["dx16"]

    # ---- dgrad kernel ----
    TM = 64 if M >= 4096 else (16 if M <= 1024 else 32)
    if M < TM:
        TM = max(16, _next_pow2(M))
    grid_d = (min(num_sms, (M + TM - 1) // TM),)
    k_dgrad = _bwd_splitk_hinted(bwd_dgrad_only_kernel, 1, 8)
    ct.launch(stream, grid_d, k_dgrad, (
        x, w1, w2, w3, dout, H0, H1PRE, dx16, dh1, dh2, M,
        K1, N1, N2, N3, TM, TN, 4,
    ))

    # ---- 3 split-K wgrad kernels (shared 3*SPLIT partials buffer) ----
    if SPLIT is None:
        SPLIT = max(1, min(num_sms // 3, (M + TM - 1) // TM))
    P = torch.empty((3, SPLIT, TN, TN), device=x.device, dtype=torch.float32)
    grid_w = (SPLIT,)
    k_wg = _bwd_splitk_hinted(bwd_wgrad_splitk_kernel, 1, 4)
    k_wga = _bwd_splitk_hinted(bwd_wgrad_splitk_act_kernel, 1, 4)
    # dW1: X^T @ dh2
    ct.launch(stream, grid_w, k_wg, (x, dh2, P[0], M, K1, N1, TM, SPLIT, 4))
    # dW2: softplus(H0)^T @ dh1
    ct.launch(stream, grid_w, k_wga, (H0, dh1, P[1], M, N1, N2, TM, SPLIT, 4))
    # dW3: softplus(H1_PRE)^T @ dout
    ct.launch(stream, grid_w, k_wga, (H1PRE, dout, P[2], M, N2, N3, TM, SPLIT, 4))

    # ---- reduce ----
    dw = torch.empty((3, TN, TN), device=x.device, dtype=x.dtype)
    k_red = _bwd_splitk_hinted(bwd_wgrad_reduce_kernel, 1, None)
    ct.launch(stream, (3,), k_red, (P, dw, SPLIT, TN))

    return dx16, dw[0], dw[1], dw[2]

'''

marker = "@ct.kernel\ndef bwd_fused_kernel("
assert marker in src
src = src.replace(marker, new_kernels.strip("\n") + "\n\n\n" + marker, 1)

# ------------------------------------------------------------------
# 2. autograd backward: use split-K when dims are 128
# ------------------------------------------------------------------
old_bw = """    @staticmethod
    def backward(ctx, dout):
        x, w1, w2, w3, H0, H1PRE = ctx.saved_tensors
        dx, dw1, dw2, dw3 = fused_mlp_bwd_cutile(dout, x, w1, w2, w3, H0, H1PRE)
        return dx, dw1, dw2, dw3"""
new_bw = """    @staticmethod
    def backward(ctx, dout):
        x, w1, w2, w3, H0, H1PRE = ctx.saved_tensors
        if w1.shape[0] == w1.shape[1] == w2.shape[1] == w3.shape[1] == 128:
            dx, dw1, dw2, dw3 = fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0, H1PRE)
        else:
            dx, dw1, dw2, dw3 = fused_mlp_bwd_cutile(dout, x, w1, w2, w3, H0, H1PRE)
        return dx, dw1, dw2, dw3"""
assert old_bw in src
src = src.replace(old_bw, new_bw, 1)

open(p, "w").write(src)
print("patched OK, lines:", src.count("\n") + 1)
