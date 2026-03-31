# MLP Megakernel

Triton megakernels that fuse entire multi-layer MLPs with Softplus activations into single GPU kernel launches. Intermediate activations live entirely in registers — zero global memory traffic between layers.

## Architecture

**3-layer:** `out = softplus(softplus(x @ W1) @ W2) @ W3`

**5-layer:** `out = softplus(softplus(softplus(softplus(x @ W1) @ W2) @ W3) @ W4) @ W5`

## What's here

| File | Description |
|------|-------------|
| `kernel.py` | Inference-only 3-layer megakernel (register-tiled recompute fusion) |
| `fused_mlp.py` | Full training 3-layer: forward + backward megakernels, `torch.autograd.Function`, weight normalization |
| `fused_mlp5.py` | Full training 5-layer: same architecture extended to 5 layers |
| `bench_weightnorm.py` | Benchmark harness comparing Eager, `torch.compile`, and Fused variants |

## Key ideas

- **Register-tiled recompute fusion**: all GEMMs + activations in ONE kernel. Nested loops tile each layer's GEMM, apply softplus in-register, and feed directly into the next layer's GEMM. No intermediate tensors ever hit global memory.

- **Persistent kernel**: launch `min(NUM_SMs, num_tiles)` programs that iterate tiles round-robin, amortizing launch overhead.

- **FP32 accumulation, FP16 Tensor Cores**: all `tl.dot` accumulate in fp32; conversion to fp16 only happens at tile boundaries for Tensor Core GEMMs.

- **FP32 activation storage for training**: forward pass stores activations in fp32 to avoid numerical loss in the backward sigmoid derivative. Backward loads fp32 activations and casts to fp16 only for GEMM operands.

- **Atomic backward for small batches**: below 32K batch size, weight gradients are accumulated via `tl.atomic_add` in a single fused kernel. Above that threshold, falls back to cuBLAS GEMMs interleaved with data-gradient computation for L2 cache reuse.

- **Fused weight normalization**: single Triton kernel normalizes all weight matrices (`W = g * v / ||v||`) in one launch, eliminating Python dispatch overhead that otherwise dominates at small matrix sizes.

## Reports

Detailed optimization journey and benchmark results:

- [`mlp_fused.md`](mlp_fused.md) — Main optimization report
- [`weight_norm.md`](weight_norm.md) — Weight normalization integration
- [`split_k_atomic_backward.md`](split_k_atomic_backward.md) — Atomic backward kernel design
- [`generalizing_to_n_layers.md`](generalizing_to_n_layers.md) — Scaling to N layers
- [`comparison_tinycudann.md`](comparison_tinycudann.md) — Comparison with tiny-cuda-nn
- [`deep_analysis_tinycudann.md`](deep_analysis_tinycudann.md) — Deep analysis of tiny-cuda-nn internals

## Requirements

- PyTorch >= 2.0
- Triton >= 2.1
- CUDA GPU with Tensor Core support (sm_80+)

## Quick start

```python
from fused_mlp import FusedMLPSoftplus, FusedMLPSoftplusWN

# Inference + training, 3-layer
model = FusedMLPSoftplus(D=128, H=128, out_features=128).cuda().half()
out = model(x)  # single kernel launch
loss = out.sum()
loss.backward()  # fused backward

# With weight normalization
model_wn = FusedMLPSoftplusWN(D=128, H=128, out_features=128).cuda().half()
out = model_wn(x)
```

## License

MIT
