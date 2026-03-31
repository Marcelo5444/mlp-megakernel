"""
Fused 3-layer MLP with Softplus: training-capable autograd Function + Module.

Forward uses a Triton megakernel that keeps h1/h2 in registers. The training
variant adds 2 stores to save h1/h2 for backward. Backward uses a mirrored
megakernel for the data-gradient chain + cuBLAS for weight gradients.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from kernel import NUM_SMS, _pick_config


class _NoCast:
    """Wrapper to prevent torch.amp.custom_fwd(cast_inputs=...) from casting a tensor.
    PyTorch's _cast only traverses pytree-registered types (list, tuple, dict)."""
    __slots__ = ('value',)
    def __init__(self, v):
        self.value = v


# ---------------------------------------------------------------------------
# Training-mode forward kernel: same as inference but also writes H1, H2
# ---------------------------------------------------------------------------

@triton.jit
def _fused_mlp_training_kernel(
    X, W1, W2, W3, OUT, H1_OUT, H2_OUT,
    M, K1: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, N3,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w2k, stride_w2n,
    stride_w3k, stride_w3n,
    stride_om, stride_on,
    stride_h1m, stride_h1n,
    stride_h2m, stride_h2n,
    num_tiles, num_n3,
    BM: tl.constexpr, BN3: tl.constexpr,
    BK1: tl.constexpr, BK2: tl.constexpr, BK3: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.float16,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_m = tl.cdiv(M, BM)

    for tile_id in range(raw_pid, num_tiles, num_pids):
        group_id = tile_id // (GROUP_M * num_n3)
        group_size = min(num_m - group_id * GROUP_M, GROUP_M)
        pm = group_id * GROUP_M + (tile_id % group_size)
        pn3 = (tile_id % (GROUP_M * num_n3)) // group_size

        offs_m = pm * BM + tl.arange(0, BM)
        offs_n3 = pn3 * BN3 + tl.arange(0, BN3)
        m_mask = offs_m < M
        n3_mask = offs_n3 < N3

        out_acc = tl.zeros((BM, BN3), dtype=tl.float32)
        is_first_n3 = (pn3 == 0)

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

                if k3_start == 0:
                    tl.store(
                        H1_OUT + offs_m[:, None] * stride_h1m + offs_k2[None, :] * stride_h1n,
                        h1_chunk,
                        mask=m_mask[:, None] & is_first_n3,
                    )

                w2_tile = tl.load(W2 + offs_k2[:, None] * stride_w2k + offs_k3[None, :] * stride_w2n)
                h2_chunk = tl.dot(h1_chunk.to(INPUT_DTYPE), w2_tile, acc=h2_chunk, out_dtype=tl.float32)

            h2_chunk = tl.where(h2_chunk > 20.0, h2_chunk, tl.log(tl.exp(h2_chunk) + 1.0))

            tl.store(
                H2_OUT + offs_m[:, None] * stride_h2m + offs_k3[None, :] * stride_h2n,
                h2_chunk,
                mask=m_mask[:, None] & is_first_n3,
            )

            w3_tile = tl.load(
                W3 + offs_k3[:, None] * stride_w3k + offs_n3[None, :] * stride_w3n,
                mask=n3_mask[None, :], other=0.0,
            )
            out_acc = tl.dot(h2_chunk.to(INPUT_DTYPE), w3_tile, acc=out_acc, out_dtype=tl.float32)

        tl.store(
            OUT + offs_m[:, None] * stride_om + offs_n3[None, :] * stride_on,
            out_acc.to(INPUT_DTYPE),
            mask=m_mask[:, None] & n3_mask[None, :],
        )


_TRITON_DTYPE = {torch.float16: tl.float16, torch.float32: tl.float32}


# ---------------------------------------------------------------------------
# Weight-norm fused megakernel: computes W = g*v/||v|| ON THE FLY inside
# the MLP forward, eliminating the separate WN kernel launch entirely.
# For W2/W3 (BK==hidden_dim): norm computed from a single full tile.
# For W1 (BK1 < K1): two-pass — first accumulate norm, then GEMM with
# scaling. V1 tiles are served from L1 cache on the second pass.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_mlp_wn_kernel(
    X, V_ALL, G_ALL, W_BUF, INVN_BUF, OUT, H1_OUT, H2_OUT,
    M, K1: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, N3,
    stride_xm, stride_xk,
    stride_vm, stride_vn,
    stride_wm, stride_wn,
    stride_om, stride_on,
    stride_h1m, stride_h1n,
    stride_h2m, stride_h2n,
    num_tiles, num_n3,
    BM: tl.constexpr, BN3: tl.constexpr,
    BK1: tl.constexpr, BK2: tl.constexpr, BK3: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.float16,
):
    V1_T = V_ALL
    V2_T = V_ALL + N1 * stride_vm
    V3_T = V_ALL + (N1 + N2) * stride_vm
    G1 = G_ALL
    G2 = G_ALL + N1
    G3 = G_ALL + (N1 + N2)

    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_m = tl.cdiv(M, BM)

    for tile_id in range(raw_pid, num_tiles, num_pids):
        group_id = tile_id // (GROUP_M * num_n3)
        group_size = min(num_m - group_id * GROUP_M, GROUP_M)
        pm = group_id * GROUP_M + (tile_id % group_size)
        pn3 = (tile_id % (GROUP_M * num_n3)) // group_size

        offs_m = pm * BM + tl.arange(0, BM)
        offs_n3 = pn3 * BN3 + tl.arange(0, BN3)
        m_mask = offs_m < M
        n3_mask = offs_n3 < N3

        out_acc = tl.zeros((BM, BN3), dtype=tl.float32)
        is_first_n3 = (pn3 == 0)
        write_wn = (pm == 0) & is_first_n3

        for k3_start in tl.static_range(0, N2, BK3):
            offs_k3 = k3_start + tl.arange(0, BK3)
            h2_chunk = tl.zeros((BM, BK3), dtype=tl.float32)

            for k2_start in tl.static_range(0, N1, BK2):
                offs_k2 = k2_start + tl.arange(0, BK2)

                # ── W1 norm: accumulate over k1 tiles ──
                norm1_sq = tl.zeros((BK2,), dtype=tl.float32)
                for k1_start in tl.static_range(0, K1, BK1):
                    offs_k1 = k1_start + tl.arange(0, BK1)
                    v1t = tl.load(V1_T + offs_k1[:, None] * stride_vn + offs_k2[None, :] * stride_vm)
                    norm1_sq += tl.sum(v1t.to(tl.float32) * v1t.to(tl.float32), axis=0)
                inv_n1 = 1.0 / tl.sqrt(norm1_sq + 1e-12)
                scale1 = tl.load(G1 + offs_k2).to(tl.float32) * inv_n1

                if write_wn:
                    tl.store(INVN_BUF + offs_k2, inv_n1.to(INPUT_DTYPE))

                # ── W1 GEMM: reload V1 tiles (L1-cached), apply scale ──
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
                    tl.store(
                        H1_OUT + offs_m[:, None] * stride_h1m + offs_k2[None, :] * stride_h1n,
                        h1_chunk,
                        mask=m_mask[:, None] & is_first_n3,
                    )

                # ── W2: single-pass norm (BK2 covers full input dim) ──
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

            tl.store(
                H2_OUT + offs_m[:, None] * stride_h2m + offs_k3[None, :] * stride_h2n,
                h2_chunk,
                mask=m_mask[:, None] & is_first_n3,
            )

            # ── W3: single-pass norm (BK3 covers full input dim) ──
            v3t = tl.load(
                V3_T + offs_k3[:, None] * stride_vn + offs_n3[None, :] * stride_vm,
                mask=n3_mask[None, :], other=0.0,
            )
            norm3_sq = tl.sum(v3t.to(tl.float32) * v3t.to(tl.float32), axis=0)
            inv_n3 = 1.0 / tl.sqrt(norm3_sq + 1e-12)
            scale3 = tl.load(G3 + offs_n3, mask=n3_mask, other=0.0).to(tl.float32) * inv_n3
            w3_tile = (v3t.to(tl.float32) * scale3[None, :]).to(INPUT_DTYPE)

            if write_wn:
                tl.store(INVN_BUF + N1 + N2 + offs_n3, inv_n3.to(INPUT_DTYPE), mask=n3_mask)
                tl.store(
                    W_BUF + (N1 + N2 + offs_n3[None, :]) * stride_wm + offs_k3[:, None] * stride_wn,
                    w3_tile, mask=n3_mask[None, :],
                )

            out_acc = tl.dot(h2_chunk.to(INPUT_DTYPE), w3_tile, acc=out_acc, out_dtype=tl.float32)

        tl.store(
            OUT + offs_m[:, None] * stride_om + offs_n3[None, :] * stride_on,
            out_acc.to(INPUT_DTYPE),
            mask=m_mask[:, None] & n3_mask[None, :],
        )


def _fused_mlp_wn_training_fwd(x, v_all, g_all, w_buf, invn_buf, splits):
    """Single-kernel forward: weight-norm + MLP fused. Returns (out, h1, h2)."""
    s0, s1, s2 = splits
    M, K1 = x.shape
    N1, N2, N3 = s0, s1, s2
    cols = v_all.shape[1]

    OUT = torch.empty((M, N3), device=x.device, dtype=x.dtype)
    H1 = torch.empty((M, N1), device=x.device, dtype=torch.float32)
    H2 = torch.empty((M, N2), device=x.device, dtype=torch.float32)

    BM, BN3, BK1, BK2, BK3, warps = _pick_config(M, K1, N1, N2, N3, fp32=(x.dtype == torch.float32))
    num_n3 = (N3 + BN3 - 1) // BN3
    num_m = (M + BM - 1) // BM
    num_tiles = num_m * num_n3
    grid = (min(NUM_SMS, num_tiles),)

    _fused_mlp_wn_kernel[grid](
        x, v_all, g_all.view(-1), w_buf, invn_buf.view(-1), OUT, H1, H2,
        M, K1, N1, N2, N3,
        x.stride(0), x.stride(1),
        v_all.stride(0), v_all.stride(1),
        w_buf.stride(0), w_buf.stride(1),
        OUT.stride(0), OUT.stride(1),
        H1.stride(0), H1.stride(1),
        H2.stride(0), H2.stride(1),
        num_tiles, num_n3,
        BM=BM, BN3=BN3, BK1=BK1, BK2=BK2, BK3=BK3,
        GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[x.dtype],
        num_warps=warps, num_stages=1,
    )
    return OUT, H1, H2


def _fused_mlp_training_fwd(x, w1, w2, w3):
    """Forward pass that returns (out, h1, h2) for backward."""
    M, K1 = x.shape
    N1, N2, N3 = w1.shape[1], w2.shape[1], w3.shape[1]

    OUT = torch.empty((M, N3), device=x.device, dtype=x.dtype)
    H1 = torch.empty((M, N1), device=x.device, dtype=torch.float32)
    H2 = torch.empty((M, N2), device=x.device, dtype=torch.float32)

    BM, BN3, BK1, BK2, BK3, warps = _pick_config(M, K1, N1, N2, N3, fp32=(x.dtype == torch.float32))
    num_n3 = (N3 + BN3 - 1) // BN3
    num_m = (M + BM - 1) // BM
    num_tiles = num_m * num_n3
    grid = (min(NUM_SMS, num_tiles),)

    _fused_mlp_training_kernel[grid](
        x, w1, w2, w3, OUT, H1, H2,
        M, K1, N1, N2, N3,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        w3.stride(0), w3.stride(1),
        OUT.stride(0), OUT.stride(1),
        H1.stride(0), H1.stride(1),
        H2.stride(0), H2.stride(1),
        num_tiles, num_n3,
        BM=BM, BN3=BN3, BK1=BK1, BK2=BK2, BK3=BK3,
        GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[x.dtype],
        num_warps=warps, num_stages=1,
    )
    return OUT, H1, H2


# ---------------------------------------------------------------------------
# Helper: fused sigmoid_from_h * dh kernel (single pass, avoids 3 kernels)
# ---------------------------------------------------------------------------

@triton.jit
def _sigmoid_h_mul_kernel(H, DH, DZ, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    h = tl.load(H + offs, mask=mask).to(tl.float32)
    dh = tl.load(DH + offs, mask=mask).to(tl.float32)
    sig = 1.0 - tl.exp(-h)
    dz = dh * sig
    tl.store(DZ + offs, dz.to(DH.dtype.element_ty), mask=mask)


def _sigmoid_h_mul(h, dh):
    """Compute (1 - exp(-h)) * dh in a single fused kernel launch.
    Output dtype matches dh (typically fp16) to keep downstream GEMMs on Tensor Core."""
    assert h.shape == dh.shape and h.is_contiguous() and dh.is_contiguous()
    dz = torch.empty_like(dh)
    n = h.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _sigmoid_h_mul_kernel[grid](h, dh, dz, n, BLOCK=BLOCK)
    return dz


# ---------------------------------------------------------------------------
# Fully-fused backward megakernel (small M, atomic weight grads)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_mlp_bwd_full_kernel(
    GRAD, W3, H2, W2, H1, W1, X,
    DX, DW3, DW2, DW1,
    M, N3: tl.constexpr, N2: tl.constexpr, N1: tl.constexpr, K1: tl.constexpr,
    stride_gm, stride_gn,
    stride_w3k, stride_w3n,
    stride_h2m, stride_h2n,
    stride_w2k, stride_w2n,
    stride_h1m, stride_h1n,
    stride_w1k, stride_w1n,
    stride_xm, stride_xk,
    stride_dxm, stride_dxn,
    stride_dw3k, stride_dw3n,
    stride_dw2k, stride_dw2n,
    stride_dw1k, stride_dw1n,
    num_tiles, num_out_cols,
    BM: tl.constexpr, BOUT: tl.constexpr,
    BK1: tl.constexpr, BK2: tl.constexpr, BK3: tl.constexpr,
    BKx: tl.constexpr,
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

        for k3_start in tl.static_range(0, N1, BK3):
            offs_k3 = k3_start + tl.arange(0, BK3)

            h1_f32 = tl.load(
                H1 + offs_m[:, None] * stride_h1m + offs_k3[None, :] * stride_h1n,
                mask=m_mask[:, None], other=0.0,
            )
            h1_inp = h1_f32.to(INPUT_DTYPE)
            dh1_chunk = tl.zeros((BM, BK3), dtype=tl.float32)

            for k2_start in tl.static_range(0, N2, BK2):
                offs_k2 = k2_start + tl.arange(0, BK2)

                h2_f32 = tl.load(
                    H2 + offs_m[:, None] * stride_h2m + offs_k2[None, :] * stride_h2n,
                    mask=m_mask[:, None], other=0.0,
                )
                h2_inp = h2_f32.to(INPUT_DTYPE)
                dh2_chunk = tl.zeros((BM, BK2), dtype=tl.float32)

                for k1_start in tl.static_range(0, N3, BK1):
                    offs_k1 = k1_start + tl.arange(0, BK1)
                    g = tl.load(
                        GRAD + offs_m[:, None] * stride_gm + offs_k1[None, :] * stride_gn,
                        mask=m_mask[:, None], other=0.0,
                    )
                    w3t = tl.load(W3 + offs_k1[:, None] * stride_w3n + offs_k2[None, :] * stride_w3k)
                    dh2_chunk = tl.dot(g, w3t, acc=dh2_chunk, out_dtype=tl.float32)

                    if is_first_out and k3_start == 0:
                        dw3_partial = tl.dot(
                            tl.trans(h2_inp), g, out_dtype=tl.float32,
                        )
                        tl.atomic_add(
                            DW3 + offs_k2[:, None] * stride_dw3k + offs_k1[None, :] * stride_dw3n,
                            dw3_partial,
                        )

                dz2_chunk = dh2_chunk * (1.0 - tl.exp(-h2_f32))
                dz2_inp = dz2_chunk.to(INPUT_DTYPE)

                if is_first_out:
                    dw2_partial = tl.dot(
                        tl.trans(h1_inp), dz2_inp, out_dtype=tl.float32,
                    )
                    tl.atomic_add(
                        DW2 + offs_k3[:, None] * stride_dw2k + offs_k2[None, :] * stride_dw2n,
                        dw2_partial,
                    )

                w2t = tl.load(W2 + offs_k2[:, None] * stride_w2n + offs_k3[None, :] * stride_w2k)
                dh1_chunk = tl.dot(dz2_inp, w2t, acc=dh1_chunk, out_dtype=tl.float32)

            dz1_chunk = dh1_chunk * (1.0 - tl.exp(-h1_f32))
            dz1_inp = dz1_chunk.to(INPUT_DTYPE)

            if is_first_out:
                for kx_start in tl.static_range(0, K1, BKx):
                    offs_kx = kx_start + tl.arange(0, BKx)
                    x_chunk = tl.load(
                        X + offs_m[:, None] * stride_xm + offs_kx[None, :] * stride_xk,
                        mask=m_mask[:, None], other=0.0,
                    )
                    dw1_partial = tl.dot(
                        tl.trans(x_chunk), dz1_inp, out_dtype=tl.float32,
                    )
                    tl.atomic_add(
                        DW1 + offs_kx[:, None] * stride_dw1k + offs_k3[None, :] * stride_dw1n,
                        dw1_partial,
                    )

            w1t = tl.load(
                W1 + offs_k3[:, None] * stride_w1n + offs_out[None, :] * stride_w1k,
                mask=out_mask[None, :], other=0.0,
            )
            dx_acc = tl.dot(dz1_inp, w1t, acc=dx_acc, out_dtype=tl.float32)

        tl.store(
            DX + offs_m[:, None] * stride_dxm + offs_out[None, :] * stride_dxn,
            dx_acc.to(INPUT_DTYPE),
            mask=m_mask[:, None] & out_mask[None, :],
        )


def _pick_config_bwd(M, N3, N2, N1, K1, fp32=False):
    BK1 = min(64, N3)
    BK2 = min(128, N2)
    BK3 = min(128, N1)
    BOUT = min(128, K1)
    num_out = (K1 + BOUT - 1) // BOUT
    target = max(1, M * num_out // NUM_SMS)
    if fp32:
        if target <= 24:
            BM, warps = 16, 4
        elif target <= 48:
            BM, warps = 32, 4
        else:
            BM, warps = 64, 4
    else:
        if target <= 24:
            BM, warps = 16, 4
        elif target <= 48:
            BM, warps = 32, 8
        else:
            BM, warps = 64, 8
    if BM > M:
        BM, warps = 16, 4
    return BM, BOUT, BK1, BK2, BK3, warps


_ATOMIC_M_THRESHOLD = 32768


def _fused_mlp_bwd(grad_output, x, w1, w2, w3, h1, h2, return_fp32_dw=False):
    """Backward: fused kernel (atomic dW) for small M, PyTorch for large M.
    Fallback interleaves weight grads with data grads for L2 cache reuse.
    When return_fp32_dw=True, skip fp32→fp16 cast on atomic DW (for WN backward)."""
    M = grad_output.shape[0]

    if M > _ATOMIC_M_THRESHOLD:
        dh2 = grad_output @ w3.t()
        dz2 = _sigmoid_h_mul(h2, dh2)
        dh1 = dz2 @ w2.t()
        dz1 = _sigmoid_h_mul(h1, dh1)
        dx = dz1 @ w1.t()
        if return_fp32_dw:
            dw3 = h2.float().t() @ grad_output.float()
            dw2 = h1.float().t() @ dz2.float()
            dw1 = x.float().t() @ dz1.float()
        else:
            inp_dt = grad_output.dtype
            h1_hp = h1.to(inp_dt) if h1.dtype != inp_dt else h1
            h2_hp = h2.to(inp_dt) if h2.dtype != inp_dt else h2
            dw3 = h2_hp.t() @ grad_output
            dw2 = h1_hp.t() @ dz2
            dw1 = x.t() @ dz1
        return dx, dw1, dw2, dw3

    N3, N2, N1, K1 = w3.shape[1], w2.shape[1], w1.shape[1], w1.shape[0]

    DX = torch.empty((M, K1), device=x.device, dtype=x.dtype)
    DW3 = torch.zeros((N2, N3), device=x.device, dtype=torch.float32)
    DW2 = torch.zeros((N1, N2), device=x.device, dtype=torch.float32)
    DW1 = torch.zeros((K1, N1), device=x.device, dtype=torch.float32)

    BM, BOUT, BK1b, BK2b, BK3b, warps = _pick_config_bwd(M, N3, N2, N1, K1, fp32=(grad_output.dtype == torch.float32))
    BKx = min(64, K1)
    num_out_cols = (K1 + BOUT - 1) // BOUT
    num_m = (M + BM - 1) // BM
    num_tiles = num_m * num_out_cols
    grid = (min(NUM_SMS, num_tiles),)

    _fused_mlp_bwd_full_kernel[grid](
        grad_output, w3, h2, w2, h1, w1, x,
        DX, DW3, DW2, DW1,
        M, N3, N2, N1, K1,
        grad_output.stride(0), grad_output.stride(1),
        w3.stride(0), w3.stride(1),
        h2.stride(0), h2.stride(1),
        w2.stride(0), w2.stride(1),
        h1.stride(0), h1.stride(1),
        w1.stride(0), w1.stride(1),
        x.stride(0), x.stride(1),
        DX.stride(0), DX.stride(1),
        DW3.stride(0), DW3.stride(1),
        DW2.stride(0), DW2.stride(1),
        DW1.stride(0), DW1.stride(1),
        num_tiles, num_out_cols,
        BM=BM, BOUT=BOUT, BK1=BK1b, BK2=BK2b, BK3=BK3b,
        BKx=BKx,
        GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[grad_output.dtype],
        num_warps=warps, num_stages=1,
    )

    if return_fp32_dw:
        return DX, DW1, DW2, DW3
    return DX, DW1.to(x.dtype), DW2.to(x.dtype), DW3.to(x.dtype)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class FusedMLPSoftplusFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, x, w1, w2, w3):
        out, h1, h2 = _fused_mlp_training_fwd(x, w1, w2, w3)
        ctx.save_for_backward(x, w1, w2, w3, h1, h2)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, w1, w2, w3, h1, h2 = ctx.saved_tensors
        return _fused_mlp_bwd(grad_output, x, w1, w2, w3, h1, h2)


# ---------------------------------------------------------------------------
# nn.Module with learnable weights
# ---------------------------------------------------------------------------

class FusedMLPSoftplus(nn.Module):
    """Drop-in 3-layer MLP: softplus(x@W1) -> softplus(h1@W2) -> h2@W3.

    Forward uses a single Triton megakernel (register-tiled fusion).
    Backward flows through all three weight matrices via autograd.
    Uses standard nn.Linear layers (bias=False) so weight init, serialization,
    and tooling work out of the box.  The kernel expects (in, out) layout;
    nn.Linear stores (out, in), so we pass weight.t() (a zero-copy view).
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear2 = nn.Linear(hidden_features, hidden_features, bias=False)
        self.linear3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        return FusedMLPSoftplusFunction.apply(
            x,
            self.linear1.weight.t(),
            self.linear2.weight.t(),
            self.linear3.weight.t(),
        )


# ---------------------------------------------------------------------------
# Weight-norm forward: normalize all weights in one kernel launch
# ---------------------------------------------------------------------------

@triton.jit
def _wn_fwd_kernel(
    V, G, W, INVNORM,
    rows, cols: tl.constexpr,
    stride_vm, stride_vn,
    stride_wm, stride_wn,
    BLOCK_N: tl.constexpr,
):
    """Compute W[row] = g[row] * v[row] / ||v[row]|| for one row.
    Also stores inv_norm[row] = 1/||v[row]|| for backward.
    Multiple weight matrices are stacked contiguously in the row dimension.
    """
    row = tl.program_id(0)
    if row >= rows:
        return
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < cols

    v = tl.load(V + row * stride_vm + offs_n * stride_vn, mask=mask, other=0.0).to(tl.float32)
    g_val = tl.load(G + row).to(tl.float32)

    norm_sq = tl.sum(v * v, axis=0)
    inv_n = 1.0 / tl.sqrt(norm_sq + 1e-12)
    w = g_val * v * inv_n

    tl.store(W + row * stride_wm + offs_n * stride_wn, w.to(V.dtype.element_ty), mask=mask)
    tl.store(INVNORM + row, inv_n.to(V.dtype.element_ty))


def _wn_fwd_batch(vs, gs):
    """Normalize multiple weight matrices in ONE kernel launch.

    All matrices must have the same number of columns (hidden dim).
    Returns list of (W, inv_norm) tuples.
    """
    cols = vs[0].shape[1]
    total_rows = sum(v.shape[0] for v in vs)

    v_cat = torch.cat(vs, dim=0)
    g_cat = torch.cat([g.view(-1) for g in gs], dim=0)
    w_cat = torch.empty_like(v_cat)
    invn_cat = torch.empty(total_rows, device=vs[0].device, dtype=vs[0].dtype)

    BLOCK_N = triton.next_power_of_2(cols)
    _wn_fwd_kernel[(total_rows,)](
        v_cat, g_cat, w_cat, invn_cat,
        total_rows, cols,
        v_cat.stride(0), v_cat.stride(1),
        w_cat.stride(0), w_cat.stride(1),
        BLOCK_N=BLOCK_N,
    )

    results = []
    offset = 0
    for v in vs:
        r = v.shape[0]
        w_i = w_cat[offset:offset + r]
        invn_i = invn_cat[offset:offset + r].unsqueeze(1)
        results.append((w_i, invn_i))
        offset += r
    return results


def _wn_fwd_contiguous(v_all, g_all, w_buf, invn_buf):
    """Normalize pre-concatenated weight block in ONE kernel launch.

    v_all: (total_rows, cols) contiguous
    g_all: (total_rows, 1)   contiguous
    w_buf, invn_buf: pre-allocated output buffers (same shapes).
    No torch.cat or allocation needed.
    """
    total_rows, cols = v_all.shape
    BLOCK_N = triton.next_power_of_2(cols)
    _wn_fwd_kernel[(total_rows,)](
        v_all, g_all.view(-1), w_buf, invn_buf.view(-1),
        total_rows, cols,
        v_all.stride(0), v_all.stride(1),
        w_buf.stride(0), w_buf.stride(1),
        BLOCK_N=BLOCK_N,
    )


# ---------------------------------------------------------------------------
# Weight-norm backward: fused dW → dv, dg conversion
# ---------------------------------------------------------------------------

@triton.jit
def _wn_bwd_kernel(
    DW, V, G, INVNORM, DV, DG,
    rows, cols: tl.constexpr,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    """Convert dW → dv, dg for one weight matrix.

    W = g * v / ||v||, so v_hat = v * inv_norm.
    dg_i = sum_j(dW_ij * v_hat_ij)
    dv_i = (g_i * inv_norm_i) * (dW_i - v_hat_i * dg_i)
    """
    row = tl.program_id(0)
    if row >= rows:
        return
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < cols

    dw = tl.load(DW + row * stride_m + offs_n * stride_n, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V + row * stride_m + offs_n * stride_n, mask=mask, other=0.0).to(tl.float32)
    inv_n = tl.load(INVNORM + row).to(tl.float32)
    g_val = tl.load(G + row).to(tl.float32)

    v_hat = v * inv_n
    dg_val = tl.sum(dw * v_hat, axis=0)
    dv = (g_val * inv_n) * (dw - v_hat * dg_val)

    tl.store(DV + row * stride_m + offs_n * stride_n, dv, mask=mask)
    tl.store(DG + row, dg_val)


def _wn_bwd_batch(dWs, vs, gs, inv_norms):
    """Convert dW → (dv, dg) for multiple weight matrices in ONE launch.

    All matrices must have the same number of columns (hidden dim).
    dWs may be non-contiguous (.t() views); they are concatenated contiguously.
    """
    cols = vs[0].shape[1]
    total_rows = sum(v.shape[0] for v in vs)

    dw_cat = torch.cat([dw.contiguous() for dw in dWs], dim=0)
    v_cat = torch.cat(vs, dim=0)
    g_cat = torch.cat([g.view(-1) for g in gs], dim=0)
    invn_cat = torch.cat([inv.view(-1) for inv in inv_norms], dim=0)

    dv_cat = torch.empty_like(v_cat)
    dg_cat = torch.empty(total_rows, device=vs[0].device, dtype=vs[0].dtype)

    BLOCK_N = triton.next_power_of_2(cols)
    _wn_bwd_kernel[(total_rows,)](
        dw_cat, v_cat, g_cat, invn_cat, dv_cat, dg_cat,
        total_rows, cols, v_cat.stride(0), v_cat.stride(1), BLOCK_N=BLOCK_N,
    )

    results = []
    offset = 0
    for v in vs:
        r = v.shape[0]
        dv_i = dv_cat[offset:offset + r]
        dg_i = dg_cat[offset:offset + r].unsqueeze(1)
        results.append((dv_i, dg_i))
        offset += r
    return results


def _wn_bwd_contiguous(dw_all, v_all, g_all, invn_all, dv_buf, dg_buf):
    """Convert dW → (dv, dg) for pre-concatenated weight block in ONE launch.

    dw_all must be contiguous (caller ensures this).
    No torch.cat or allocation needed.
    """
    total_rows, cols = v_all.shape
    BLOCK_N = triton.next_power_of_2(cols)
    _wn_bwd_kernel[(total_rows,)](
        dw_all, v_all, g_all.view(-1), invn_all.view(-1),
        dv_buf, dg_buf.view(-1),
        total_rows, cols, v_all.stride(0), v_all.stride(1), BLOCK_N=BLOCK_N,
    )


@triton.jit
def _wn_bwd_from_dw_T_kernel(
    DW1, DW2, DW3,
    V, G, INVNORM, DV, DG,
    split0, split1, total_rows,
    cols: tl.constexpr,
    stride_dw1_k, stride_dw1_n,
    stride_dw2_k, stride_dw2_n,
    stride_dw3_k, stride_dw3_n,
    stride_vm, stride_vn,
    BLOCK_N: tl.constexpr,
):
    """WN backward reading DW in (K, N) layout — no .t().contiguous() needed.

    Each program handles one output neuron row. Reads column `row` of the
    appropriate DW matrix (stride access, but 128×128 fits in L1).
    """
    row = tl.program_id(0)
    if row >= total_rows:
        return
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < cols

    if row < split0:
        local_row = row
        dw = tl.load(DW1 + offs_n * stride_dw1_k + local_row * stride_dw1_n,
                      mask=mask, other=0.0).to(tl.float32)
    elif row < split1:
        local_row = row - split0
        dw = tl.load(DW2 + offs_n * stride_dw2_k + local_row * stride_dw2_n,
                      mask=mask, other=0.0).to(tl.float32)
    else:
        local_row = row - split1
        dw = tl.load(DW3 + offs_n * stride_dw3_k + local_row * stride_dw3_n,
                      mask=mask, other=0.0).to(tl.float32)

    v = tl.load(V + row * stride_vm + offs_n * stride_vn, mask=mask, other=0.0).to(tl.float32)
    inv_n = tl.load(INVNORM + row).to(tl.float32)
    g_val = tl.load(G + row).to(tl.float32)

    v_hat = v * inv_n
    dg_val = tl.sum(dw * v_hat, axis=0)
    dv = (g_val * inv_n) * (dw - v_hat * dg_val)

    tl.store(DV + row * stride_vm + offs_n * stride_vn, dv, mask=mask)
    tl.store(DG + row, dg_val)


def _wn_bwd_from_dw_transposed(dw1, dw2, dw3, v_all, g_all, invn_all,
                                splits, dv_buf, dg_buf):
    """WN backward that reads DW in original (K, N) layout — zero copies.

    Replaces: 3× .to() + 3× .t().contiguous() + torch.cat + _wn_bwd_contiguous
    with a single kernel launch.
    """
    total_rows, cols = v_all.shape
    BLOCK_N = triton.next_power_of_2(cols)
    s0, s1 = splits[0], splits[0] + splits[1]
    _wn_bwd_from_dw_T_kernel[(total_rows,)](
        dw1, dw2, dw3,
        v_all, g_all.view(-1), invn_all.view(-1),
        dv_buf, dg_buf.view(-1),
        s0, s1, total_rows, cols,
        dw1.stride(0), dw1.stride(1),
        dw2.stride(0), dw2.stride(1),
        dw3.stride(0), dw3.stride(1),
        v_all.stride(0), v_all.stride(1),
        BLOCK_N=BLOCK_N,
    )


# ---------------------------------------------------------------------------
# Weight-normed autograd Function (fused forward + backward)
# ---------------------------------------------------------------------------

_WN_FUSED_M_THRESHOLD = 40960


class FusedMLPSoftplusWNFunction(torch.autograd.Function):
    """Autograd for weight-normed 3-layer MLP with contiguous parameter storage.

    Forward: at small M, single fused megakernel (1 launch). At large M,
    separate WN + megakernel (2 launches).
    Backward: megakernel (atomic DW at small M, cuBLAS at large M) +
    zero-copy WN backward (reads fp32 DW column-wise). 2 kernel launches.
    Pre-allocated DW buffer eliminates per-call torch.zeros overhead.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, x, v_all, g_all, w_buf, invn_buf, splits, dw_buf_nc):
        dw_buf = dw_buf_nc.value
        M = x.shape[0]
        if M <= _WN_FUSED_M_THRESHOLD:
            out, h1, h2 = _fused_mlp_wn_training_fwd(x, v_all, g_all, w_buf, invn_buf, splits)
        else:
            _wn_fwd_contiguous(v_all, g_all, w_buf, invn_buf)
            s0, s1, s2 = splits
            out, h1, h2 = _fused_mlp_training_fwd(
                x, w_buf[:s0].t(), w_buf[s0:s0 + s1].t(), w_buf[s0 + s1:].t(),
            )
        s0, s1, s2 = splits
        w1 = w_buf[:s0].t()
        w2 = w_buf[s0:s0 + s1].t()
        w3 = w_buf[s0 + s1:].t()
        ctx.save_for_backward(x, w1, w2, w3, h1, h2, v_all, g_all, invn_buf)
        ctx._splits = splits
        ctx._dw_buf = dw_buf
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, w1, w2, w3, h1, h2, v_all, g_all, invn_buf = ctx.saved_tensors
        splits = ctx._splits
        M = grad_output.shape[0]

        if M > _ATOMIC_M_THRESHOLD:
            dx, dw1, dw2, dw3 = _fused_mlp_bwd(
                grad_output, x, w1, w2, w3, h1, h2, return_fp32_dw=True,
            )
        else:
            N3, N2, N1, K1 = w3.shape[1], w2.shape[1], w1.shape[1], w1.shape[0]
            dw_buf = ctx._dw_buf
            dw_buf.zero_()
            dw1 = dw_buf[:K1 * N1].view(K1, N1)
            dw2 = dw_buf[K1 * N1:K1 * N1 + N1 * N2].view(N1, N2)
            dw3 = dw_buf[K1 * N1 + N1 * N2:].view(N2, N3)

            DX = torch.empty((M, K1), device=x.device, dtype=x.dtype)
            BM, BOUT, BK1b, BK2b, BK3b, warps = _pick_config_bwd(
                M, N3, N2, N1, K1, fp32=(grad_output.dtype == torch.float32),
            )
            BKx = min(64, K1)
            num_out_cols = (K1 + BOUT - 1) // BOUT
            num_m = (M + BM - 1) // BM
            num_tiles = num_m * num_out_cols
            grid = (min(NUM_SMS, num_tiles),)

            _fused_mlp_bwd_full_kernel[grid](
                grad_output, w3, h2, w2, h1, w1, x,
                DX, dw3, dw2, dw1,
                M, N3, N2, N1, K1,
                grad_output.stride(0), grad_output.stride(1),
                w3.stride(0), w3.stride(1),
                h2.stride(0), h2.stride(1),
                w2.stride(0), w2.stride(1),
                h1.stride(0), h1.stride(1),
                w1.stride(0), w1.stride(1),
                x.stride(0), x.stride(1),
                DX.stride(0), DX.stride(1),
                dw3.stride(0), dw3.stride(1),
                dw2.stride(0), dw2.stride(1),
                dw1.stride(0), dw1.stride(1),
                num_tiles, num_out_cols,
                BM=BM, BOUT=BOUT, BK1=BK1b, BK2=BK2b, BK3=BK3b,
                BKx=BKx, GROUP_M=16, INPUT_DTYPE=_TRITON_DTYPE[grad_output.dtype],
                num_warps=warps, num_stages=1,
            )
            dx = DX

        dv_buf = torch.empty(v_all.shape, device=v_all.device, dtype=torch.float32)
        dg_buf = torch.empty(g_all.shape, device=g_all.device, dtype=torch.float32)
        _wn_bwd_from_dw_transposed(
            dw1, dw2, dw3, v_all, g_all, invn_buf, splits, dv_buf, dg_buf,
        )

        return (dx,
                dv_buf, dg_buf,
                None, None, None, None)


# ---------------------------------------------------------------------------
# nn.Module with weight normalization
# ---------------------------------------------------------------------------

class FusedMLPSoftplusWN(nn.Module):
    """3-layer fused MLP with weight normalization on all layers.

    Each weight W_i is parameterized as g_i * v_i / ||v_i|| (per output row).
    All v's and g's are stored as contiguous blocks to eliminate torch.cat
    overhead in forward (0 allocations). Backward uses a zero-copy Triton
    kernel for dW → dv, dg conversion. Pre-allocated DW buffer avoids
    per-call torch.zeros overhead in the atomic backward path.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        H, O = hidden_features, out_features
        D = in_features
        r1, r2, r3 = H, H, O
        c = max(D, H)
        self._splits = (r1, r2, r3)
        self._total_rows = r1 + r2 + r3

        v_init = torch.randn(self._total_rows, c) * 0.1
        g_init = torch.ones(self._total_rows, 1)
        self.v_all = nn.Parameter(v_init)
        self.g_all = nn.Parameter(g_init)

        self.register_buffer('_w_buf', torch.empty(self._total_rows, c, dtype=torch.float16))
        self.register_buffer('_invn_buf', torch.empty(self._total_rows, 1, dtype=torch.float16))

        total_dw = D * H + H * H + H * O
        self.register_buffer('_dw_buf', torch.zeros(total_dw, dtype=torch.float32))

    def forward(self, x):
        return FusedMLPSoftplusWNFunction.apply(
            x, self.v_all, self.g_all, self._w_buf, self._invn_buf, self._splits,
            _NoCast(self._dw_buf),
        )
