"""
Fused 5-layer MLP with Softplus: register-tiled megakernel for both
forward (inference & training) and backward passes.

Architecture:
    h1 = softplus(x @ W1)
    h2 = softplus(h1 @ W2)
    h3 = softplus(h2 @ W3)
    h4 = softplus(h3 @ W4)
    out = h4 @ W5

Forward keeps all intermediate activations (h1-h4) in registers via
5-level nested loop fusion. Backward uses a mirrored megakernel for
the data-gradient chain (dx, dz4, dz3, dz2, dz1) + cuBLAS for weight
gradients.

Key constraint vs 3-layer: 5 accumulator tiles in registers caps BM
at 64 (vs 128) to stay within the 255-register-per-thread limit.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fused_mlp import _sigmoid_h_mul, _NoCast

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

_TRITON_DTYPE = {torch.float16: tl.float16, torch.float32: tl.float32}


# ===========================================================================
# Forward megakernel (inference + training via constexpr TRAINING flag)
# ===========================================================================

@triton.jit
def _fused_mlp5_kernel(
    X, W1, W2, W3, W4, W5, OUT,
    H1_OUT, H2_OUT, H3_OUT, H4_OUT,
    M,
    K1: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
    N3: tl.constexpr, N4: tl.constexpr,
    N5,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w2k, stride_w2n,
    stride_w3k, stride_w3n,
    stride_w4k, stride_w4n,
    stride_w5k, stride_w5n,
    stride_om, stride_on,
    stride_h1m, stride_h1n,
    stride_h2m, stride_h2n,
    stride_h3m, stride_h3n,
    stride_h4m, stride_h4n,
    num_tiles, num_n5,
    TRAINING: tl.constexpr,
    BM: tl.constexpr, BN5: tl.constexpr,
    BK1: tl.constexpr, BK2: tl.constexpr, BK3: tl.constexpr,
    BK4: tl.constexpr, BK5: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.float16,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_m = tl.cdiv(M, BM)

    for tile_id in range(raw_pid, num_tiles, num_pids):
        group_id = tile_id // (GROUP_M * num_n5)
        group_size = min(num_m - group_id * GROUP_M, GROUP_M)
        pm = group_id * GROUP_M + (tile_id % group_size)
        pn5 = (tile_id % (GROUP_M * num_n5)) // group_size

        offs_m = pm * BM + tl.arange(0, BM)
        offs_n5 = pn5 * BN5 + tl.arange(0, BN5)
        m_mask = offs_m < M
        n5_mask = offs_n5 < N5
        is_first_n5 = (pn5 == 0)

        out_acc = tl.zeros((BM, BN5), dtype=tl.float32)

        for k5_start in tl.static_range(0, N4, BK5):
            offs_k5 = k5_start + tl.arange(0, BK5)
            h4_chunk = tl.zeros((BM, BK5), dtype=tl.float32)

            for k4_start in tl.static_range(0, N3, BK4):
                offs_k4 = k4_start + tl.arange(0, BK4)
                h3_chunk = tl.zeros((BM, BK4), dtype=tl.float32)

                for k3_start in tl.static_range(0, N2, BK3):
                    offs_k3 = k3_start + tl.arange(0, BK3)
                    h2_chunk = tl.zeros((BM, BK3), dtype=tl.float32)

                    for k2_start in tl.static_range(0, N1, BK2):
                        offs_k2 = k2_start + tl.arange(0, BK2)
                        h1_chunk = tl.zeros((BM, BK2), dtype=tl.float32)

                        for k1_start in tl.static_range(0, K1, BK1):
                            offs_k1 = k1_start + tl.arange(0, BK1)
                            a = tl.load(
                                X + offs_m[:, None] * stride_xm + offs_k1[None, :] * stride_xk,
                                mask=m_mask[:, None], other=0.0,
                            )
                            b = tl.load(W1 + offs_k1[:, None] * stride_w1k + offs_k2[None, :] * stride_w1n)
                            h1_chunk = tl.dot(a, b, acc=h1_chunk, out_dtype=tl.float32)

                        h1_chunk = tl.where(h1_chunk > 20.0, h1_chunk, tl.log(tl.exp(h1_chunk) + 1.0))
                        if TRAINING:
                            if k3_start == 0:
                                if k4_start == 0:
                                    if k5_start == 0:
                                        tl.store(
                                            H1_OUT + offs_m[:, None] * stride_h1m + offs_k2[None, :] * stride_h1n,
                                            h1_chunk,
                                            mask=m_mask[:, None] & is_first_n5,
                                        )

                        w2_tile = tl.load(W2 + offs_k2[:, None] * stride_w2k + offs_k3[None, :] * stride_w2n)
                        h2_chunk = tl.dot(h1_chunk.to(INPUT_DTYPE), w2_tile, acc=h2_chunk, out_dtype=tl.float32)

                    h2_chunk = tl.where(h2_chunk > 20.0, h2_chunk, tl.log(tl.exp(h2_chunk) + 1.0))
                    if TRAINING:
                        if k4_start == 0:
                            if k5_start == 0:
                                tl.store(
                                    H2_OUT + offs_m[:, None] * stride_h2m + offs_k3[None, :] * stride_h2n,
                                    h2_chunk,
                                    mask=m_mask[:, None] & is_first_n5,
                                )

                    w3_tile = tl.load(W3 + offs_k3[:, None] * stride_w3k + offs_k4[None, :] * stride_w3n)
                    h3_chunk = tl.dot(h2_chunk.to(INPUT_DTYPE), w3_tile, acc=h3_chunk, out_dtype=tl.float32)

                h3_chunk = tl.where(h3_chunk > 20.0, h3_chunk, tl.log(tl.exp(h3_chunk) + 1.0))
                if TRAINING:
                    if k5_start == 0:
                        tl.store(
                            H3_OUT + offs_m[:, None] * stride_h3m + offs_k4[None, :] * stride_h3n,
                            h3_chunk,
                            mask=m_mask[:, None] & is_first_n5,
                        )

                w4_tile = tl.load(W4 + offs_k4[:, None] * stride_w4k + offs_k5[None, :] * stride_w4n)
                h4_chunk = tl.dot(h3_chunk.to(INPUT_DTYPE), w4_tile, acc=h4_chunk, out_dtype=tl.float32)

            h4_chunk = tl.where(h4_chunk > 20.0, h4_chunk, tl.log(tl.exp(h4_chunk) + 1.0))
            if TRAINING:
                tl.store(
                    H4_OUT + offs_m[:, None] * stride_h4m + offs_k5[None, :] * stride_h4n,
                    h4_chunk,
                    mask=m_mask[:, None] & is_first_n5,
                )

            w5_tile = tl.load(
                W5 + offs_k5[:, None] * stride_w5k + offs_n5[None, :] * stride_w5n,
                mask=n5_mask[None, :], other=0.0,
            )
            out_acc = tl.dot(h4_chunk.to(INPUT_DTYPE), w5_tile, acc=out_acc, out_dtype=tl.float32)

        tl.store(
            OUT + offs_m[:, None] * stride_om + offs_n5[None, :] * stride_on,
            out_acc.to(INPUT_DTYPE),
            mask=m_mask[:, None] & n5_mask[None, :],
        )


# ===========================================================================
# Forward heuristic
# ===========================================================================

def _pick_config_5(M, K1, N1, N2, N3, N4, N5, fp32=False):
    """Forward heuristic for 5-layer MLP.

    BM capped at 64 (not 128 like 3-layer) because 5 accumulator tiles
    in registers approaches the 255-register-per-thread limit at BM=128.

    fp32: BM caps at 32 (5 fp32 operand tiles is tight), BK stays 128.
    """
    BK1 = min(64, K1)
    BK2 = min(128, N1)
    BK3 = min(128, N2)
    BK4 = min(128, N3)
    BK5 = min(128, N4)
    BN5 = min(128, N5)
    num_n5 = (N5 + BN5 - 1) // BN5
    target_bm = max(1, M * num_n5 // NUM_SMS)
    if fp32:
        if target_bm <= 24:
            BM, warps = 16, 4
        else:
            BM, warps = 32, 4
    else:
        if target_bm <= 24:
            BM, warps = 16, 4
        elif target_bm <= 48:
            BM, warps = 32, 8
        else:
            BM, warps = 64, 8
    if BM > M:
        BM, warps = 16, 4
    return BM, BN5, BK1, BK2, BK3, BK4, BK5, warps


# ===========================================================================
# Forward launch functions
# ===========================================================================

def _fused_mlp5_fwd(x, w1, w2, w3, w4, w5):
    """Inference-only forward pass (no intermediate stores)."""
    M, K1 = x.shape
    N1, N2, N3, N4, N5 = w1.shape[1], w2.shape[1], w3.shape[1], w4.shape[1], w5.shape[1]
    OUT = torch.empty((M, N5), device=x.device, dtype=x.dtype)

    BM, BN5, BK1, BK2, BK3, BK4, BK5, warps = _pick_config_5(M, K1, N1, N2, N3, N4, N5, fp32=(x.dtype == torch.float32))
    num_n5 = (N5 + BN5 - 1) // BN5
    num_m = (M + BM - 1) // BM
    num_tiles = num_m * num_n5
    grid = (min(NUM_SMS, num_tiles),)

    _fused_mlp5_kernel[grid](
        x, w1, w2, w3, w4, w5, OUT,
        OUT, OUT, OUT, OUT,
        M, K1, N1, N2, N3, N4, N5,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1), w2.stride(0), w2.stride(1),
        w3.stride(0), w3.stride(1), w4.stride(0), w4.stride(1),
        w5.stride(0), w5.stride(1), OUT.stride(0), OUT.stride(1),
        OUT.stride(0), OUT.stride(1), OUT.stride(0), OUT.stride(1),
        OUT.stride(0), OUT.stride(1), OUT.stride(0), OUT.stride(1),
        num_tiles, num_n5,
        TRAINING=False,
        BM=BM, BN5=BN5, BK1=BK1, BK2=BK2, BK3=BK3, BK4=BK4, BK5=BK5,
        GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[x.dtype],
        num_warps=warps, num_stages=1,
    )
    return OUT


def _fused_mlp5_training_fwd(x, w1, w2, w3, w4, w5):
    """Training forward: returns (out, h1, h2, h3, h4)."""
    M, K1 = x.shape
    N1, N2, N3, N4, N5 = w1.shape[1], w2.shape[1], w3.shape[1], w4.shape[1], w5.shape[1]
    OUT = torch.empty((M, N5), device=x.device, dtype=x.dtype)
    H1 = torch.empty((M, N1), device=x.device, dtype=torch.float32)
    H2 = torch.empty((M, N2), device=x.device, dtype=torch.float32)
    H3 = torch.empty((M, N3), device=x.device, dtype=torch.float32)
    H4 = torch.empty((M, N4), device=x.device, dtype=torch.float32)

    BM, BN5, BK1, BK2, BK3, BK4, BK5, warps = _pick_config_5(M, K1, N1, N2, N3, N4, N5, fp32=(x.dtype == torch.float32))
    num_n5 = (N5 + BN5 - 1) // BN5
    num_m = (M + BM - 1) // BM
    num_tiles = num_m * num_n5
    grid = (min(NUM_SMS, num_tiles),)

    _fused_mlp5_kernel[grid](
        x, w1, w2, w3, w4, w5, OUT,
        H1, H2, H3, H4,
        M, K1, N1, N2, N3, N4, N5,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1), w2.stride(0), w2.stride(1),
        w3.stride(0), w3.stride(1), w4.stride(0), w4.stride(1),
        w5.stride(0), w5.stride(1), OUT.stride(0), OUT.stride(1),
        H1.stride(0), H1.stride(1), H2.stride(0), H2.stride(1),
        H3.stride(0), H3.stride(1), H4.stride(0), H4.stride(1),
        num_tiles, num_n5,
        TRAINING=True,
        BM=BM, BN5=BN5, BK1=BK1, BK2=BK2, BK3=BK3, BK4=BK4, BK5=BK5,
        GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[x.dtype],
        num_warps=warps, num_stages=1,
    )
    return OUT, H1, H2, H3, H4


# ===========================================================================
# Backward data-gradient megakernel
# Computes: dx = ((((grad @ W5.T * sig4) @ W4.T * sig3) @ W3.T * sig2) @ W2.T * sig1) @ W1.T
# Also stores dz4, dz3, dz2, dz1 for weight-gradient GEMMs.
# Isomorphic to forward: transposed weights, sigmoid*mul instead of softplus.
# ===========================================================================

@triton.jit
def _fused_mlp5_bwd_full_kernel(
    GRAD, W5, H4, W4, H3, W3, H2, W2, H1, W1, X,
    DX, DW5, DW4, DW3, DW2, DW1,
    M,
    N5: tl.constexpr, N4: tl.constexpr, N3: tl.constexpr,
    N2: tl.constexpr, N1: tl.constexpr, K1: tl.constexpr,
    stride_gm, stride_gn,
    stride_w5k, stride_w5n, stride_h4m, stride_h4n,
    stride_w4k, stride_w4n, stride_h3m, stride_h3n,
    stride_w3k, stride_w3n, stride_h2m, stride_h2n,
    stride_w2k, stride_w2n, stride_h1m, stride_h1n,
    stride_w1k, stride_w1n, stride_xm, stride_xk,
    stride_dxm, stride_dxn,
    stride_dw5k, stride_dw5n, stride_dw4k, stride_dw4n,
    stride_dw3k, stride_dw3n, stride_dw2k, stride_dw2n,
    stride_dw1k, stride_dw1n,
    num_tiles, num_out_cols,
    BM: tl.constexpr, BOUT: tl.constexpr,
    BK1: tl.constexpr, BK2: tl.constexpr, BK3: tl.constexpr,
    BK4: tl.constexpr, BK5: tl.constexpr, BKx: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.float16,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_m = tl.cdiv(M, BM)

    for tile_id in range(raw_pid, num_tiles, num_pids):
        group_id = tile_id // (GROUP_M * num_out_cols)
        group_size = min(num_m - group_id * GROUP_M, GROUP_M)
        pm = group_id * GROUP_M + (tile_id % group_size)
        pn = (tile_id % (GROUP_M * num_out_cols)) // group_size

        offs_m = pm * BM + tl.arange(0, BM)
        offs_out = pn * BOUT + tl.arange(0, BOUT)
        m_mask = offs_m < M
        out_mask = offs_out < K1
        is_first_out = (pn == 0)

        dx_acc = tl.zeros((BM, BOUT), dtype=tl.float32)

        for k5_start in tl.static_range(0, N1, BK5):
            offs_k5 = k5_start + tl.arange(0, BK5)
            h1_f32 = tl.load(
                H1 + offs_m[:, None] * stride_h1m + offs_k5[None, :] * stride_h1n,
                mask=m_mask[:, None], other=0.0,
            )
            h1_inp = h1_f32.to(INPUT_DTYPE)
            dh1_chunk = tl.zeros((BM, BK5), dtype=tl.float32)

            for k4_start in tl.static_range(0, N2, BK4):
                offs_k4 = k4_start + tl.arange(0, BK4)
                h2_f32 = tl.load(
                    H2 + offs_m[:, None] * stride_h2m + offs_k4[None, :] * stride_h2n,
                    mask=m_mask[:, None], other=0.0,
                )
                h2_inp = h2_f32.to(INPUT_DTYPE)
                dh2_chunk = tl.zeros((BM, BK4), dtype=tl.float32)

                for k3_start in tl.static_range(0, N3, BK3):
                    offs_k3 = k3_start + tl.arange(0, BK3)
                    h3_f32 = tl.load(
                        H3 + offs_m[:, None] * stride_h3m + offs_k3[None, :] * stride_h3n,
                        mask=m_mask[:, None], other=0.0,
                    )
                    h3_inp = h3_f32.to(INPUT_DTYPE)
                    dh3_chunk = tl.zeros((BM, BK3), dtype=tl.float32)

                    for k2_start in tl.static_range(0, N4, BK2):
                        offs_k2 = k2_start + tl.arange(0, BK2)
                        h4_f32 = tl.load(
                            H4 + offs_m[:, None] * stride_h4m + offs_k2[None, :] * stride_h4n,
                            mask=m_mask[:, None], other=0.0,
                        )
                        h4_inp = h4_f32.to(INPUT_DTYPE)
                        dh4_chunk = tl.zeros((BM, BK2), dtype=tl.float32)

                        for k1_start in tl.static_range(0, N5, BK1):
                            offs_k1 = k1_start + tl.arange(0, BK1)
                            g = tl.load(
                                GRAD + offs_m[:, None] * stride_gm + offs_k1[None, :] * stride_gn,
                                mask=m_mask[:, None], other=0.0,
                            )
                            w5t = tl.load(W5 + offs_k1[:, None] * stride_w5n + offs_k2[None, :] * stride_w5k)
                            dh4_chunk = tl.dot(g, w5t, acc=dh4_chunk, out_dtype=tl.float32)

                            # dW5[k2, k1] += h4.T @ grad
                            if is_first_out and k3_start == 0 and k4_start == 0 and k5_start == 0:
                                tl.atomic_add(
                                    DW5 + offs_k2[:, None] * stride_dw5k + offs_k1[None, :] * stride_dw5n,
                                    tl.dot(tl.trans(h4_inp), g, out_dtype=tl.float32),
                                )

                        dz4_chunk = dh4_chunk * (1.0 - tl.exp(-h4_f32))
                        dz4_inp = dz4_chunk.to(INPUT_DTYPE)

                        # dW4[k3, k2] += h3.T @ dz4
                        if is_first_out and k4_start == 0 and k5_start == 0:
                            tl.atomic_add(
                                DW4 + offs_k3[:, None] * stride_dw4k + offs_k2[None, :] * stride_dw4n,
                                tl.dot(tl.trans(h3_inp), dz4_inp, out_dtype=tl.float32),
                            )

                        w4t = tl.load(W4 + offs_k2[:, None] * stride_w4n + offs_k3[None, :] * stride_w4k)
                        dh3_chunk = tl.dot(dz4_inp, w4t, acc=dh3_chunk, out_dtype=tl.float32)

                    dz3_chunk = dh3_chunk * (1.0 - tl.exp(-h3_f32))
                    dz3_inp = dz3_chunk.to(INPUT_DTYPE)

                    # dW3[k4, k3] += h2.T @ dz3
                    if is_first_out and k5_start == 0:
                        tl.atomic_add(
                            DW3 + offs_k4[:, None] * stride_dw3k + offs_k3[None, :] * stride_dw3n,
                            tl.dot(tl.trans(h2_inp), dz3_inp, out_dtype=tl.float32),
                        )

                    w3t = tl.load(W3 + offs_k3[:, None] * stride_w3n + offs_k4[None, :] * stride_w3k)
                    dh2_chunk = tl.dot(dz3_inp, w3t, acc=dh2_chunk, out_dtype=tl.float32)

                dz2_chunk = dh2_chunk * (1.0 - tl.exp(-h2_f32))
                dz2_inp = dz2_chunk.to(INPUT_DTYPE)

                # dW2[k5, k4] += h1.T @ dz2
                if is_first_out:
                    tl.atomic_add(
                        DW2 + offs_k5[:, None] * stride_dw2k + offs_k4[None, :] * stride_dw2n,
                        tl.dot(tl.trans(h1_inp), dz2_inp, out_dtype=tl.float32),
                    )

                w2t = tl.load(W2 + offs_k4[:, None] * stride_w2n + offs_k5[None, :] * stride_w2k)
                dh1_chunk = tl.dot(dz2_inp, w2t, acc=dh1_chunk, out_dtype=tl.float32)

            dz1_chunk = dh1_chunk * (1.0 - tl.exp(-h1_f32))
            dz1_inp = dz1_chunk.to(INPUT_DTYPE)

            # dW1[:, k5] += x.T @ dz1  (chunked over K1)
            if is_first_out:
                for kx_start in tl.static_range(0, K1, BKx):
                    offs_kx = kx_start + tl.arange(0, BKx)
                    x_chunk = tl.load(
                        X + offs_m[:, None] * stride_xm + offs_kx[None, :] * stride_xk,
                        mask=m_mask[:, None], other=0.0,
                    )
                    tl.atomic_add(
                        DW1 + offs_kx[:, None] * stride_dw1k + offs_k5[None, :] * stride_dw1n,
                        tl.dot(tl.trans(x_chunk), dz1_inp, out_dtype=tl.float32),
                    )

            w1t = tl.load(
                W1 + offs_k5[:, None] * stride_w1n + offs_out[None, :] * stride_w1k,
                mask=out_mask[None, :], other=0.0,
            )
            dx_acc = tl.dot(dz1_inp, w1t, acc=dx_acc, out_dtype=tl.float32)

        tl.store(
            DX + offs_m[:, None] * stride_dxm + offs_out[None, :] * stride_dxn,
            dx_acc.to(INPUT_DTYPE),
            mask=m_mask[:, None] & out_mask[None, :],
        )


# ===========================================================================
# Backward heuristic
# ===========================================================================

def _pick_config_bwd_5(M, N5, N4, N3, N2, N1, K1, fp32=False):
    """Backward heuristic for 5-layer MLP.

    Same register-pressure cap as forward (BM <= 64) plus extra
    loads (H1-H4) and stores (DZ1-DZ4) add slight pressure.

    fp32: BM=32, warps=4, BK stays 128. BOUT=128 always optimal.
    """
    BK1 = min(64, N5)
    BK2 = min(128, N4)
    BK3 = min(128, N3)
    BK4 = min(128, N2)
    BK5 = min(128, N1)
    BOUT = min(128, K1)
    num_out = (K1 + BOUT - 1) // BOUT
    target = max(1, M * num_out // NUM_SMS)
    if fp32:
        if target <= 24:
            BM, warps = 16, 4
        else:
            BM, warps = 32, 4
    else:
        if target <= 24:
            BM, warps = 16, 4
        elif target <= 48:
            BM, warps = 32, 8
        else:
            BM, warps = 64, 8
    if BM > M:
        BM, warps = 16, 4
    return BM, BOUT, BK1, BK2, BK3, BK4, BK5, warps


# ===========================================================================
# Backward launch
# ===========================================================================

_ATOMIC_M_THRESHOLD = 16384


def _fused_mlp5_bwd(grad_output, x, w1, w2, w3, w4, w5, h1, h2, h3, h4,
                     return_fp32_dw=False):
    """Backward: fused kernel (atomic dW) for small M, PyTorch for large M.
    Fallback interleaves weight grads for L2 cache reuse.
    When return_fp32_dw=True, skip fp32→fp16 cast on atomic DW (for WN backward)."""
    M = grad_output.shape[0]

    if M > _ATOMIC_M_THRESHOLD:
        dh4 = grad_output @ w5.t()
        dz4 = _sigmoid_h_mul(h4, dh4)
        dh3 = dz4 @ w4.t()
        dz3 = _sigmoid_h_mul(h3, dh3)
        dh2 = dz3 @ w3.t()
        dz2 = _sigmoid_h_mul(h2, dh2)
        dh1 = dz2 @ w2.t()
        dz1 = _sigmoid_h_mul(h1, dh1)
        dx = dz1 @ w1.t()
        if return_fp32_dw:
            dw5 = h4.float().t() @ grad_output.float()
            dw4 = h3.float().t() @ dz4.float()
            dw3 = h2.float().t() @ dz3.float()
            dw2 = h1.float().t() @ dz2.float()
            dw1 = x.float().t() @ dz1.float()
        else:
            inp_dt = grad_output.dtype
            h4_hp = h4.to(inp_dt) if h4.dtype != inp_dt else h4
            h3_hp = h3.to(inp_dt) if h3.dtype != inp_dt else h3
            h2_hp = h2.to(inp_dt) if h2.dtype != inp_dt else h2
            h1_hp = h1.to(inp_dt) if h1.dtype != inp_dt else h1
            dw5 = h4_hp.t() @ grad_output
            dw4 = h3_hp.t() @ dz4
            dw3 = h2_hp.t() @ dz3
            dw2 = h1_hp.t() @ dz2
            dw1 = x.t() @ dz1
        return (dx, dw1, dw2, dw3, dw4, dw5)

    N5, N4, N3 = w5.shape[1], w4.shape[1], w3.shape[1]
    N2, N1, K1 = w2.shape[1], w1.shape[1], w1.shape[0]

    DX = torch.empty((M, K1), device=x.device, dtype=x.dtype)
    DW5 = torch.zeros((N4, N5), device=x.device, dtype=torch.float32)
    DW4 = torch.zeros((N3, N4), device=x.device, dtype=torch.float32)
    DW3 = torch.zeros((N2, N3), device=x.device, dtype=torch.float32)
    DW2 = torch.zeros((N1, N2), device=x.device, dtype=torch.float32)
    DW1 = torch.zeros((K1, N1), device=x.device, dtype=torch.float32)

    BM, BOUT, BK1b, BK2b, BK3b, BK4b, BK5b, warps = _pick_config_bwd_5(
        M, N5, N4, N3, N2, N1, K1, fp32=(grad_output.dtype == torch.float32),
    )
    BKx = min(64, K1)
    num_out_cols = (K1 + BOUT - 1) // BOUT
    num_m = (M + BM - 1) // BM
    num_tiles = num_m * num_out_cols
    grid = (min(NUM_SMS, num_tiles),)

    _fused_mlp5_bwd_full_kernel[grid](
        grad_output, w5, h4, w4, h3, w3, h2, w2, h1, w1, x,
        DX, DW5, DW4, DW3, DW2, DW1,
        M, N5, N4, N3, N2, N1, K1,
        grad_output.stride(0), grad_output.stride(1),
        w5.stride(0), w5.stride(1), h4.stride(0), h4.stride(1),
        w4.stride(0), w4.stride(1), h3.stride(0), h3.stride(1),
        w3.stride(0), w3.stride(1), h2.stride(0), h2.stride(1),
        w2.stride(0), w2.stride(1), h1.stride(0), h1.stride(1),
        w1.stride(0), w1.stride(1), x.stride(0), x.stride(1),
        DX.stride(0), DX.stride(1),
        DW5.stride(0), DW5.stride(1), DW4.stride(0), DW4.stride(1),
        DW3.stride(0), DW3.stride(1), DW2.stride(0), DW2.stride(1),
        DW1.stride(0), DW1.stride(1),
        num_tiles, num_out_cols,
        BM=BM, BOUT=BOUT,
        BK1=BK1b, BK2=BK2b, BK3=BK3b, BK4=BK4b, BK5=BK5b, BKx=BKx,
        GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[grad_output.dtype],
        num_warps=warps, num_stages=1,
    )

    if return_fp32_dw:
        return DX, DW1, DW2, DW3, DW4, DW5
    dt = x.dtype
    return DX, DW1.to(dt), DW2.to(dt), DW3.to(dt), DW4.to(dt), DW5.to(dt)


# ===========================================================================
# Autograd Function
# ===========================================================================

class FusedMLP5SoftplusFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, x, w1, w2, w3, w4, w5):
        out, h1, h2, h3, h4 = _fused_mlp5_training_fwd(x, w1, w2, w3, w4, w5)
        ctx.save_for_backward(x, w1, w2, w3, w4, w5, h1, h2, h3, h4)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, w1, w2, w3, w4, w5, h1, h2, h3, h4 = ctx.saved_tensors
        return _fused_mlp5_bwd(grad_output, x, w1, w2, w3, w4, w5, h1, h2, h3, h4)


# ===========================================================================
# nn.Module with learnable weights
# ===========================================================================

class FusedMLP5Softplus(nn.Module):
    """Drop-in 5-layer MLP: 4 softplus hidden layers + linear output.

    Forward uses a single Triton megakernel (register-tiled fusion).
    Backward flows through all five weight matrices via autograd.
    Uses standard nn.Linear layers (bias=False) so weight init, serialization,
    and tooling work out of the box.  The kernel expects (in, out) layout;
    nn.Linear stores (out, in), so we pass weight.t() (a zero-copy view).
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear2 = nn.Linear(hidden_features, hidden_features, bias=False)
        self.linear3 = nn.Linear(hidden_features, hidden_features, bias=False)
        self.linear4 = nn.Linear(hidden_features, hidden_features, bias=False)
        self.linear5 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        return FusedMLP5SoftplusFunction.apply(
            x,
            self.linear1.weight.t(),
            self.linear2.weight.t(),
            self.linear3.weight.t(),
            self.linear4.weight.t(),
            self.linear5.weight.t(),
        )


# ===========================================================================
# Weight-norm backward (reuse kernel from fused_mlp.py)
# ===========================================================================

from fused_mlp import (_wn_fwd_contiguous, _wn_bwd_contiguous,
                       _wn_fwd_batch, _wn_bwd_batch)


@triton.jit
def _wn_bwd_from_dw_T5_kernel(
    DW1, DW2, DW3, DW4, DW5,
    V, G, INVNORM, DV, DG,
    split0, split1, split2, split3, total_rows,
    cols: tl.constexpr,
    stride_dw1_k, stride_dw1_n,
    stride_dw2_k, stride_dw2_n,
    stride_dw3_k, stride_dw3_n,
    stride_dw4_k, stride_dw4_n,
    stride_dw5_k, stride_dw5_n,
    stride_vm, stride_vn,
    BLOCK_N: tl.constexpr,
):
    """5-layer WN backward reading DW in (K, N) layout — no .t().contiguous()."""
    row = tl.program_id(0)
    if row >= total_rows:
        return
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < cols

    if row < split0:
        lr = row
        dw = tl.load(DW1 + offs_n * stride_dw1_k + lr * stride_dw1_n,
                      mask=mask, other=0.0).to(tl.float32)
    elif row < split1:
        lr = row - split0
        dw = tl.load(DW2 + offs_n * stride_dw2_k + lr * stride_dw2_n,
                      mask=mask, other=0.0).to(tl.float32)
    elif row < split2:
        lr = row - split1
        dw = tl.load(DW3 + offs_n * stride_dw3_k + lr * stride_dw3_n,
                      mask=mask, other=0.0).to(tl.float32)
    elif row < split3:
        lr = row - split2
        dw = tl.load(DW4 + offs_n * stride_dw4_k + lr * stride_dw4_n,
                      mask=mask, other=0.0).to(tl.float32)
    else:
        lr = row - split3
        dw = tl.load(DW5 + offs_n * stride_dw5_k + lr * stride_dw5_n,
                      mask=mask, other=0.0).to(tl.float32)

    v = tl.load(V + row * stride_vm + offs_n * stride_vn, mask=mask, other=0.0).to(tl.float32)
    inv_n = tl.load(INVNORM + row).to(tl.float32)
    g_val = tl.load(G + row).to(tl.float32)

    v_hat = v * inv_n
    dg_val = tl.sum(dw * v_hat, axis=0)
    dv = (g_val * inv_n) * (dw - v_hat * dg_val)

    tl.store(DV + row * stride_vm + offs_n * stride_vn,
             dv, mask=mask)
    tl.store(DG + row, dg_val)


def _wn_bwd_from_dw_transposed_5(dw1, dw2, dw3, dw4, dw5,
                                  v_all, g_all, invn_all,
                                  splits, dv_buf, dg_buf):
    """5-layer WN backward that reads DW in (K, N) layout — zero copies."""
    total_rows, cols = v_all.shape
    BLOCK_N = triton.next_power_of_2(cols)
    s0 = splits[0]
    s1 = s0 + splits[1]
    s2 = s1 + splits[2]
    s3 = s2 + splits[3]
    _wn_bwd_from_dw_T5_kernel[(total_rows,)](
        dw1, dw2, dw3, dw4, dw5,
        v_all, g_all.view(-1), invn_all.view(-1),
        dv_buf, dg_buf.view(-1),
        s0, s1, s2, s3, total_rows, cols,
        dw1.stride(0), dw1.stride(1),
        dw2.stride(0), dw2.stride(1),
        dw3.stride(0), dw3.stride(1),
        dw4.stride(0), dw4.stride(1),
        dw5.stride(0), dw5.stride(1),
        v_all.stride(0), v_all.stride(1),
        BLOCK_N=BLOCK_N,
    )


# ===========================================================================
# 5-layer Weight-norm fused megakernel
# ===========================================================================

@triton.jit
def _fused_mlp5_wn_kernel(
    X, V_ALL, G_ALL, W_BUF, INVN_BUF, OUT,
    H1_OUT, H2_OUT, H3_OUT, H4_OUT,
    M,
    K1: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
    N3: tl.constexpr, N4: tl.constexpr,
    N5,
    stride_xm, stride_xk,
    stride_vm, stride_vn,
    stride_wm, stride_wn,
    stride_om, stride_on,
    stride_h1m, stride_h1n,
    stride_h2m, stride_h2n,
    stride_h3m, stride_h3n,
    stride_h4m, stride_h4n,
    num_tiles, num_n5,
    BM: tl.constexpr, BN5: tl.constexpr,
    BK1: tl.constexpr, BK2: tl.constexpr, BK3: tl.constexpr,
    BK4: tl.constexpr, BK5: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.float16,
):
    V1_T = V_ALL
    V2_T = V_ALL + N1 * stride_vm
    V3_T = V_ALL + (N1 + N2) * stride_vm
    V4_T = V_ALL + (N1 + N2 + N3) * stride_vm
    V5_T = V_ALL + (N1 + N2 + N3 + N4) * stride_vm
    G1 = G_ALL
    G2 = G_ALL + N1
    G3 = G_ALL + (N1 + N2)
    G4 = G_ALL + (N1 + N2 + N3)
    G5 = G_ALL + (N1 + N2 + N3 + N4)

    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_m = tl.cdiv(M, BM)

    for tile_id in range(raw_pid, num_tiles, num_pids):
        group_id = tile_id // (GROUP_M * num_n5)
        group_size = min(num_m - group_id * GROUP_M, GROUP_M)
        pm = group_id * GROUP_M + (tile_id % group_size)
        pn5 = (tile_id % (GROUP_M * num_n5)) // group_size

        offs_m = pm * BM + tl.arange(0, BM)
        offs_n5 = pn5 * BN5 + tl.arange(0, BN5)
        m_mask = offs_m < M
        n5_mask = offs_n5 < N5
        is_first_n5 = (pn5 == 0)
        write_wn = (pm == 0) & is_first_n5

        out_acc = tl.zeros((BM, BN5), dtype=tl.float32)

        for k5_start in tl.static_range(0, N4, BK5):
            offs_k5 = k5_start + tl.arange(0, BK5)
            h4_chunk = tl.zeros((BM, BK5), dtype=tl.float32)

            for k4_start in tl.static_range(0, N3, BK4):
                offs_k4 = k4_start + tl.arange(0, BK4)
                h3_chunk = tl.zeros((BM, BK4), dtype=tl.float32)

                for k3_start in tl.static_range(0, N2, BK3):
                    offs_k3 = k3_start + tl.arange(0, BK3)
                    h2_chunk = tl.zeros((BM, BK3), dtype=tl.float32)

                    for k2_start in tl.static_range(0, N1, BK2):
                        offs_k2 = k2_start + tl.arange(0, BK2)

                        # ── W1 norm: two-pass (BK1 < K1) ──
                        norm1_sq = tl.zeros((BK2,), dtype=tl.float32)
                        for k1_start in tl.static_range(0, K1, BK1):
                            offs_k1 = k1_start + tl.arange(0, BK1)
                            v1t = tl.load(V1_T + offs_k1[:, None] * stride_vn + offs_k2[None, :] * stride_vm)
                            norm1_sq += tl.sum(v1t.to(tl.float32) * v1t.to(tl.float32), axis=0)
                        inv_n1 = 1.0 / tl.sqrt(norm1_sq + 1e-12)
                        scale1 = tl.load(G1 + offs_k2).to(tl.float32) * inv_n1

                        if write_wn:
                            tl.store(INVN_BUF + offs_k2, inv_n1.to(INPUT_DTYPE))

                        h1_chunk = tl.zeros((BM, BK2), dtype=tl.float32)
                        for k1_start in tl.static_range(0, K1, BK1):
                            offs_k1 = k1_start + tl.arange(0, BK1)
                            a = tl.load(
                                X + offs_m[:, None] * stride_xm + offs_k1[None, :] * stride_xk,
                                mask=m_mask[:, None], other=0.0,
                            )
                            v1t = tl.load(V1_T + offs_k1[:, None] * stride_vn + offs_k2[None, :] * stride_vm)
                            w1_tile = (v1t.to(tl.float32) * scale1[None, :]).to(INPUT_DTYPE)
                            h1_chunk = tl.dot(a, w1_tile, acc=h1_chunk, out_dtype=tl.float32)

                            if write_wn:
                                tl.store(
                                    W_BUF + offs_k2[None, :] * stride_wm + offs_k1[:, None] * stride_wn,
                                    w1_tile,
                                )

                        h1_chunk = tl.where(h1_chunk > 20.0, h1_chunk, tl.log(tl.exp(h1_chunk) + 1.0))
                        if k3_start == 0:
                            if k4_start == 0:
                                if k5_start == 0:
                                    tl.store(
                                        H1_OUT + offs_m[:, None] * stride_h1m + offs_k2[None, :] * stride_h1n,
                                        h1_chunk,
                                        mask=m_mask[:, None] & is_first_n5,
                                    )

                        # ── W2: single-pass norm ──
                        v2t = tl.load(V2_T + offs_k2[:, None] * stride_vn + offs_k3[None, :] * stride_vm)
                        norm2_sq = tl.sum(v2t.to(tl.float32) * v2t.to(tl.float32), axis=0)
                        inv_n2 = 1.0 / tl.sqrt(norm2_sq + 1e-12)
                        scale2 = tl.load(G2 + offs_k3).to(tl.float32) * inv_n2
                        w2_tile = (v2t.to(tl.float32) * scale2[None, :]).to(INPUT_DTYPE)

                        if write_wn:
                            tl.store(INVN_BUF + N1 + offs_k3, inv_n2.to(INPUT_DTYPE))
                            tl.store(
                                W_BUF + (N1 + offs_k3[None, :]) * stride_wm + offs_k2[:, None] * stride_wn,
                                w2_tile,
                            )

                        h2_chunk = tl.dot(h1_chunk.to(INPUT_DTYPE), w2_tile, acc=h2_chunk, out_dtype=tl.float32)

                    h2_chunk = tl.where(h2_chunk > 20.0, h2_chunk, tl.log(tl.exp(h2_chunk) + 1.0))
                    if k4_start == 0:
                        if k5_start == 0:
                            tl.store(
                                H2_OUT + offs_m[:, None] * stride_h2m + offs_k3[None, :] * stride_h2n,
                                h2_chunk,
                                mask=m_mask[:, None] & is_first_n5,
                            )

                    # ── W3: single-pass norm ──
                    v3t = tl.load(V3_T + offs_k3[:, None] * stride_vn + offs_k4[None, :] * stride_vm)
                    norm3_sq = tl.sum(v3t.to(tl.float32) * v3t.to(tl.float32), axis=0)
                    inv_n3 = 1.0 / tl.sqrt(norm3_sq + 1e-12)
                    scale3 = tl.load(G3 + offs_k4).to(tl.float32) * inv_n3
                    w3_tile = (v3t.to(tl.float32) * scale3[None, :]).to(INPUT_DTYPE)

                    if write_wn:
                        tl.store(INVN_BUF + N1 + N2 + offs_k4, inv_n3.to(INPUT_DTYPE))
                        tl.store(
                            W_BUF + (N1 + N2 + offs_k4[None, :]) * stride_wm + offs_k3[:, None] * stride_wn,
                            w3_tile,
                        )

                    h3_chunk = tl.dot(h2_chunk.to(INPUT_DTYPE), w3_tile, acc=h3_chunk, out_dtype=tl.float32)

                h3_chunk = tl.where(h3_chunk > 20.0, h3_chunk, tl.log(tl.exp(h3_chunk) + 1.0))
                if k5_start == 0:
                    tl.store(
                        H3_OUT + offs_m[:, None] * stride_h3m + offs_k4[None, :] * stride_h3n,
                        h3_chunk,
                        mask=m_mask[:, None] & is_first_n5,
                    )

                # ── W4: single-pass norm ──
                v4t = tl.load(V4_T + offs_k4[:, None] * stride_vn + offs_k5[None, :] * stride_vm)
                norm4_sq = tl.sum(v4t.to(tl.float32) * v4t.to(tl.float32), axis=0)
                inv_n4 = 1.0 / tl.sqrt(norm4_sq + 1e-12)
                scale4 = tl.load(G4 + offs_k5).to(tl.float32) * inv_n4
                w4_tile = (v4t.to(tl.float32) * scale4[None, :]).to(INPUT_DTYPE)

                if write_wn:
                    tl.store(INVN_BUF + N1 + N2 + N3 + offs_k5, inv_n4.to(INPUT_DTYPE))
                    tl.store(
                        W_BUF + (N1 + N2 + N3 + offs_k5[None, :]) * stride_wm + offs_k4[:, None] * stride_wn,
                        w4_tile,
                    )

                h4_chunk = tl.dot(h3_chunk.to(INPUT_DTYPE), w4_tile, acc=h4_chunk, out_dtype=tl.float32)

            h4_chunk = tl.where(h4_chunk > 20.0, h4_chunk, tl.log(tl.exp(h4_chunk) + 1.0))
            tl.store(
                H4_OUT + offs_m[:, None] * stride_h4m + offs_k5[None, :] * stride_h4n,
                h4_chunk,
                mask=m_mask[:, None] & is_first_n5,
            )

            # ── W5: single-pass norm ──
            v5t = tl.load(
                V5_T + offs_k5[:, None] * stride_vn + offs_n5[None, :] * stride_vm,
                mask=n5_mask[None, :], other=0.0,
            )
            norm5_sq = tl.sum(v5t.to(tl.float32) * v5t.to(tl.float32), axis=0)
            inv_n5 = 1.0 / tl.sqrt(norm5_sq + 1e-12)
            scale5 = tl.load(G5 + offs_n5, mask=n5_mask, other=0.0).to(tl.float32) * inv_n5
            w5_tile = (v5t.to(tl.float32) * scale5[None, :]).to(INPUT_DTYPE)

            if write_wn:
                tl.store(INVN_BUF + N1 + N2 + N3 + N4 + offs_n5, inv_n5.to(INPUT_DTYPE), mask=n5_mask)
                tl.store(
                    W_BUF + (N1 + N2 + N3 + N4 + offs_n5[None, :]) * stride_wm + offs_k5[:, None] * stride_wn,
                    w5_tile, mask=n5_mask[None, :],
                )

            out_acc = tl.dot(h4_chunk.to(INPUT_DTYPE), w5_tile, acc=out_acc, out_dtype=tl.float32)

        tl.store(
            OUT + offs_m[:, None] * stride_om + offs_n5[None, :] * stride_on,
            out_acc.to(INPUT_DTYPE),
            mask=m_mask[:, None] & n5_mask[None, :],
        )


def _fused_mlp5_wn_training_fwd(x, v_all, g_all, w_buf, invn_buf, splits):
    """Single-kernel forward for 5-layer WN MLP. Returns (out, h1, h2, h3, h4)."""
    s = splits
    M, K1 = x.shape
    N1, N2, N3, N4, N5 = s

    OUT = torch.empty((M, N5), device=x.device, dtype=x.dtype)
    H1 = torch.empty((M, N1), device=x.device, dtype=torch.float32)
    H2 = torch.empty((M, N2), device=x.device, dtype=torch.float32)
    H3 = torch.empty((M, N3), device=x.device, dtype=torch.float32)
    H4 = torch.empty((M, N4), device=x.device, dtype=torch.float32)

    BM, BN5, BK1, BK2, BK3, BK4, BK5, warps = _pick_config_5(
        M, K1, N1, N2, N3, N4, N5, fp32=(x.dtype == torch.float32),
    )
    num_n5 = (N5 + BN5 - 1) // BN5
    num_m = (M + BM - 1) // BM
    num_tiles = num_m * num_n5
    grid = (min(NUM_SMS, num_tiles),)

    _fused_mlp5_wn_kernel[grid](
        x, v_all, g_all.view(-1), w_buf, invn_buf.view(-1), OUT,
        H1, H2, H3, H4,
        M, K1, N1, N2, N3, N4, N5,
        x.stride(0), x.stride(1),
        v_all.stride(0), v_all.stride(1),
        w_buf.stride(0), w_buf.stride(1),
        OUT.stride(0), OUT.stride(1),
        H1.stride(0), H1.stride(1),
        H2.stride(0), H2.stride(1),
        H3.stride(0), H3.stride(1),
        H4.stride(0), H4.stride(1),
        num_tiles, num_n5,
        BM=BM, BN5=BN5, BK1=BK1, BK2=BK2, BK3=BK3, BK4=BK4, BK5=BK5,
        GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[x.dtype],
        num_warps=warps, num_stages=1,
    )
    return OUT, H1, H2, H3, H4


# ===========================================================================
# Weight-normed autograd Function for 5-layer (contiguous storage)
# ===========================================================================

_WN_FUSED_M_THRESHOLD_5 = 40960


class FusedMLP5SoftplusWNFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, x, v_all, g_all, w_buf, invn_buf, splits, dw_buf_nc):
        dw_buf = dw_buf_nc.value
        M = x.shape[0]
        s = splits
        offsets = [0]
        for si in s:
            offsets.append(offsets[-1] + si)

        if M <= _WN_FUSED_M_THRESHOLD_5:
            out, h1, h2, h3, h4 = _fused_mlp5_wn_training_fwd(
                x, v_all, g_all, w_buf, invn_buf, splits,
            )
        else:
            _wn_fwd_contiguous(v_all, g_all, w_buf, invn_buf)
            ws = [w_buf[offsets[i]:offsets[i + 1]].t() for i in range(5)]
            out, h1, h2, h3, h4 = _fused_mlp5_training_fwd(x, *ws)

        ws = [w_buf[offsets[i]:offsets[i + 1]].t() for i in range(5)]
        ctx.save_for_backward(x, *ws, h1, h2, h3, h4, v_all, g_all, invn_buf)
        ctx._splits = splits
        ctx._dw_buf = dw_buf
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x = saved[0]
        w1, w2, w3, w4, w5 = saved[1:6]
        h1, h2, h3, h4 = saved[6:10]
        v_all, g_all, invn_buf = saved[10:13]
        splits = ctx._splits
        M = grad_output.shape[0]

        if M > _ATOMIC_M_THRESHOLD:
            dx, dw1, dw2, dw3, dw4, dw5 = _fused_mlp5_bwd(
                grad_output, x, w1, w2, w3, w4, w5, h1, h2, h3, h4,
                return_fp32_dw=True,
            )
        else:
            N5, N4, N3 = w5.shape[1], w4.shape[1], w3.shape[1]
            N2, N1, K1 = w2.shape[1], w1.shape[1], w1.shape[0]
            dw_buf = ctx._dw_buf
            dw_buf.zero_()
            o = 0
            dw1 = dw_buf[o:o + K1 * N1].view(K1, N1); o += K1 * N1
            dw2 = dw_buf[o:o + N1 * N2].view(N1, N2); o += N1 * N2
            dw3 = dw_buf[o:o + N2 * N3].view(N2, N3); o += N2 * N3
            dw4 = dw_buf[o:o + N3 * N4].view(N3, N4); o += N3 * N4
            dw5 = dw_buf[o:o + N4 * N5].view(N4, N5)

            DX = torch.empty((M, K1), device=x.device, dtype=x.dtype)
            BM, BOUT, BK1b, BK2b, BK3b, BK4b, BK5b, warps = _pick_config_bwd_5(
                M, N5, N4, N3, N2, N1, K1,
                fp32=(grad_output.dtype == torch.float32),
            )
            BKx = min(64, K1)
            num_out_cols = (K1 + BOUT - 1) // BOUT
            num_m = (M + BM - 1) // BM
            num_tiles = num_m * num_out_cols
            grid = (min(NUM_SMS, num_tiles),)

            _fused_mlp5_bwd_full_kernel[grid](
                grad_output, w5, h4, w4, h3, w3, h2, w2, h1, w1, x,
                DX, dw5, dw4, dw3, dw2, dw1,
                M, N5, N4, N3, N2, N1, K1,
                grad_output.stride(0), grad_output.stride(1),
                w5.stride(0), w5.stride(1), h4.stride(0), h4.stride(1),
                w4.stride(0), w4.stride(1), h3.stride(0), h3.stride(1),
                w3.stride(0), w3.stride(1), h2.stride(0), h2.stride(1),
                w2.stride(0), w2.stride(1), h1.stride(0), h1.stride(1),
                w1.stride(0), w1.stride(1), x.stride(0), x.stride(1),
                DX.stride(0), DX.stride(1),
                dw5.stride(0), dw5.stride(1), dw4.stride(0), dw4.stride(1),
                dw3.stride(0), dw3.stride(1), dw2.stride(0), dw2.stride(1),
                dw1.stride(0), dw1.stride(1),
                num_tiles, num_out_cols,
                BM=BM, BOUT=BOUT,
                BK1=BK1b, BK2=BK2b, BK3=BK3b, BK4=BK4b, BK5=BK5b, BKx=BKx,
                GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[grad_output.dtype],
                num_warps=warps, num_stages=1,
            )
            dx = DX

        dv_buf = torch.empty(v_all.shape, device=v_all.device, dtype=torch.float32)
        dg_buf = torch.empty(g_all.shape, device=g_all.device, dtype=torch.float32)
        _wn_bwd_from_dw_transposed_5(
            dw1, dw2, dw3, dw4, dw5,
            v_all, g_all, invn_buf, splits, dv_buf, dg_buf,
        )

        return (dx, dv_buf, dg_buf, None, None, None, None)


# ===========================================================================
# 5-layer Module with weight normalization (contiguous storage)
# ===========================================================================

class FusedMLP5SoftplusWN(nn.Module):
    """5-layer fused MLP with weight normalization on all layers.

    All v's and g's stored as contiguous blocks to eliminate torch.cat overhead.
    Pre-allocated DW buffer avoids per-call torch.zeros overhead in backward.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        D, H, O = in_features, hidden_features, out_features
        r1 = H
        r2 = H
        r3 = H
        r4 = H
        r5 = O
        c = max(D, H)
        self._splits = (r1, r2, r3, r4, r5)
        self._total_rows = r1 + r2 + r3 + r4 + r5

        self.v_all = nn.Parameter(torch.randn(self._total_rows, c) * 0.1)
        self.g_all = nn.Parameter(torch.ones(self._total_rows, 1))

        self.register_buffer('_w_buf', torch.empty(self._total_rows, c, dtype=torch.float16))
        self.register_buffer('_invn_buf', torch.empty(self._total_rows, 1, dtype=torch.float16))

        total_dw = D * H + H * H + H * H + H * H + H * O
        self.register_buffer('_dw_buf', torch.zeros(total_dw, dtype=torch.float32))

    def forward(self, x):
        return FusedMLP5SoftplusWNFunction.apply(
            x, self.v_all, self.g_all, self._w_buf, self._invn_buf, self._splits,
            _NoCast(self._dw_buf),
        )
