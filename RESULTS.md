# Backward Strategy B: single fused backward megakernel (fp32 activation storage)

Branch: `bwd-strategy-B`. GB10 (sm_121, 48 SMs), cuTile 1.7, CUDA 13.4,
dims K1=N1=N2=N3=128, fp16 I/O with fp32 accumulators.

## Design

- `fwd_store_acts_kernel`: same as `fused_mlp_kernel` but additionally stores
  pre-activations H0=(X@W1) (M,N1) and H1_PRE=(softplus(H0)@W2) (M,N2) in
  **fp32** global memory.
- `bwd_fused_kernel`: ONE persistent kernel launch computing all six gradients:
  - `dh1 = (dout @ W3^T) * sigmoid(H1_PRE)` — H1_PRE loaded from global, no recompute
  - `dh2 = (dh1 @ W2^T) * sigmoid(H0)`
  - `dx = dh2 @ W1^T` — plain store (pm tiles are block-partitioned, no atomics needed)
  - `dW3 += softplus(H1_PRE)^T @ dout`, `dW2 += softplus(H0)^T @ dh1`,
    `dW1 += X^T @ dh2` — fp32 register accumulators held across all pm tiles of
    the block, flushed with one `atomic_add` each at kernel end
- `FusedMLPCutile`: forward = `fused_mlp_fwd_store_acts`, backward = single
  `ct.launch` of `bwd_fused_kernel` (TM=16, occupancy=1, num_worker_warps=8).

## Correctness (vs PyTorch eager autograd, cos >= 0.99 gate)

| M | fwd cos | dx | dW1 | dW2 | dW3 |
|---|---|---|---|---|---|
| 256 | 1.0000 | 1.0000 PASS | 1.0000 PASS | 1.0000 PASS | 1.0000 PASS |
| 1024 | 1.0000 | 1.0000 PASS | 1.0000 PASS | 1.0000 PASS | 1.0000 PASS |
| 4096 | 1.0000 | 1.0000 PASS | 1.0000 PASS | 1.0000 PASS | 1.0000 PASS |

(fp32 activation storage keeps all four grads at cos ~1.0, better than the
fp16-DH-scratch recompute baseline's ~0.997 on dW2.)

## Fwd+bwd timing (do_bench, L2 flushed, us)

| M | pytorch eager | cutile strategy-B | speedup |
|---|---|---|---|
| 256 | 46.2 | 126.3 | 0.37x |
| 1024 | 121.3 | 170.9 | 0.71x |
| 4096 | 134.3 | 244.8 | 0.55x |
| 16384 | 439.1 | 570.6 | 0.77x |
| 65536 | 2086.2 | 2090.6 | 1.00x |

Baseline context (recompute backward): M=4096 fwd+bwd = 1551.6us; strategy B
achieves 244.8us (6.3x faster than recompute baseline) but still 1.8x slower
than PyTorch eager at M=4096. Strategy B reaches parity only at M=65536.

## Notes / tuning history

- Default hints: backward kernel ~490us at M=4096 (dominated by MMA latency
  with few resident warps). Sweeping occupancy x num_worker_warps found
  `nww=8, occ=1` gives 2-3x (255us at M=4096). nww=8 passes correctness at
  ALL M here (unlike the forward kernel where nww=8 fails at M<=256).
- TM=16 beats TM=32/64 at every M (109/156/197/390/1234us for
  M=256..65536, kernel-only incl. zero-init).
- `replace_hints` recompiles on every call — cache the variant in a module
  global (`_BWD_FUSED_K8`) or do_bench shows ~35ms/call.
- dx needs no atomics: each pm row-tile belongs to exactly one block.
- Contract: square dims K1=N1=N2=N3<=128 (single weight tile per weight);
  asserted in `fused_mlp_bwd_cutile`.
- Remaining gap vs PyTorch: 5 chained MMAs per pm tile in one kernel +
  fp32 activation stores in fwd (fwd_store ~2.7x cost of inference-only fwd at
  M=65536: 497us vs 187us). PyTorch's cuBLAS GEMMs are simply better tuned per
  GEMM; fusion eliminates launch overhead (~4 launches) but not the GEMM
  efficiency gap.
