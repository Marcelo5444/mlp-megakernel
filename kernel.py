"""
Fused 3-layer MLP with Softplus: register-tiled recompute megakernel.

Architecture: out = softplus(softplus(x @ W1) @ W2) @ W3

All GEMMs + activations execute in ONE persistent kernel launch.
Intermediate activations (h1, h2) live entirely in registers — never
written to global memory. Grouped tile ordering for L2 cache locality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


# ---------------------------------------------------------------------------
# Heuristic config selection (no warmup benchmarking needed)
# ---------------------------------------------------------------------------

def _pick_config(M, K1, N1, N2, N3, fp32=False):
    """Runtime heuristic for forward 3-layer config.

    Selects BM/warps based on occupancy target (tiles per SM).
    BK values match hidden dims to avoid recomputation redundancy.
    """
    BK1 = min(64, K1)
    BK2 = min(128, N1)
    BK3 = min(128, N2)
    BN3 = min(128, N3)
    num_n3 = (N3 + BN3 - 1) // BN3
    target_bm = max(1, M * num_n3 // NUM_SMS)
    if fp32:
        if target_bm <= 24:
            BM, warps = 16, 4
        elif target_bm <= 48:
            BM, warps = 32, 4
        else:
            BM, warps = 64, 4
    else:
        if target_bm <= 24:
            BM, warps = 16, 4
        elif target_bm <= 48:
            BM, warps = 32, 8
        else:
            BM, warps = 64, 8
    if BM > M:
        BM, warps = 16, 4
    return BM, BN3, BK1, BK2, BK3, warps


# ---------------------------------------------------------------------------
# Triton megakernel: register-tiled recompute fusion
# ---------------------------------------------------------------------------

@triton.jit
def _fused_mlp_recompute(
    X, W1, W2, W3, OUT,
    M, K1: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, N3,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w2k, stride_w2n,
    stride_w3k, stride_w3n,
    stride_om, stride_on,
    num_tiles, num_n3,
    BM: tl.constexpr, BN3: tl.constexpr,
    BK1: tl.constexpr, BK2: tl.constexpr, BK3: tl.constexpr,
    GROUP_M: tl.constexpr,
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
                w2_tile = tl.load(W2 + offs_k2[:, None] * stride_w2k + offs_k3[None, :] * stride_w2n)
                h2_chunk = tl.dot(h1_chunk.to(tl.float16), w2_tile, acc=h2_chunk, out_dtype=tl.float32)

            h2_chunk = tl.where(h2_chunk > 20.0, h2_chunk, tl.log(tl.exp(h2_chunk) + 1.0))
            w3_tile = tl.load(
                W3 + offs_k3[:, None] * stride_w3k + offs_n3[None, :] * stride_w3n,
                mask=n3_mask[None, :], other=0.0,
            )
            out_acc = tl.dot(h2_chunk.to(tl.float16), w3_tile, acc=out_acc, out_dtype=tl.float32)

        tl.store(
            OUT + offs_m[:, None] * stride_om + offs_n3[None, :] * stride_on,
            out_acc.to(tl.float16),
            mask=m_mask[:, None] & n3_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Python launch wrapper
# ---------------------------------------------------------------------------

def _fused_mlp_fwd(x, w1, w2, w3):
    """Inference forward: single kernel launch, no intermediate global memory."""
    x = x.contiguous()
    w1, w2, w3 = w1.contiguous(), w2.contiguous(), w3.contiguous()
    M, K1 = x.shape
    N1, N2, N3 = w1.shape[1], w2.shape[1], w3.shape[1]
    OUT = torch.empty((M, N3), device=x.device, dtype=x.dtype)

    BM, BN3, BK1, BK2, BK3, warps = _pick_config(M, K1, N1, N2, N3, fp32=(x.dtype == torch.float32))
    num_n3 = triton.cdiv(N3, BN3)
    num_m = triton.cdiv(M, BM)
    num_tiles = num_m * num_n3
    grid = (min(NUM_SMS, num_tiles),)

    _fused_mlp_recompute[grid](
        x, w1, w2, w3, OUT,
        M, K1, N1, N2, N3,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        w3.stride(0), w3.stride(1),
        OUT.stride(0), OUT.stride(1),
        num_tiles, num_n3,
        BM=BM, BN3=BN3, BK1=BK1, BK2=BK2, BK3=BK3,
        GROUP_M=16, num_warps=warps, num_stages=1,
    )
    return OUT


# ---------------------------------------------------------------------------
# Reference (for correctness testing)
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """PyTorch reference: 3-layer MLP with softplus."""
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2, w3):
        h1 = F.softplus(x @ w1)
        h2 = F.softplus(h1 @ w2)
        return h2 @ w3


class ModelNew(nn.Module):
    """Fused Triton megakernel: 3-layer MLP with softplus."""
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2, w3):
        return _fused_mlp_fwd(x, w1, w2, w3)
