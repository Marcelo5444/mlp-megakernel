# Fused MLP with Softplus: Complete Optimization Report

## Overview

Custom Triton megakernels that fuse multi-layer MLPs with softplus activations
into single GPU kernel launches, eliminating all intermediate memory traffic.
Implementations for both **3-layer** and **5-layer** variants, supporting fp16
and fp32, with full training (forward + backward) support.

Architecture (3-layer): `out = softplus(softplus(x @ W1) @ W2) @ W3`
Architecture (5-layer): `out = softplus(softplus(softplus(softplus(x @ W1) @ W2) @ W3) @ W4) @ W5`

---

## 1. Forward Pass: Register-Tiled Recompute Fusion

### Core Idea

All GEMMs + activations in ONE kernel launch. Intermediate activations (h1, h2,
...) never exist in global memory — they live entirely in registers.

Each thread block computes an output tile via nested loops. For the 3-layer:

```
for k3 in static_range(0, N2, BK3):       # h2 columns
    h2_chunk = zeros[BM, BK3]
    for k2 in static_range(0, N1, BK2):   # h1 columns
        h1_chunk = zeros[BM, BK2]
        for k1 in static_range(0, K1, BK1):
            h1_chunk += X_tile @ W1_tile   # tl.dot accumulate
        h1_chunk = softplus(h1_chunk)
        h2_chunk += h1_chunk @ W2_tile
    h2_chunk = softplus(h2_chunk)
    out_acc += h2_chunk @ W3_tile
store(OUT, out_acc)
```

The 5-layer extends this to 5-level nesting (h1 → h2 → h3 → h4 → out).

### Key Design Decisions

- **BK = hidden_dim (128)**: When block sizes match hidden dimensions, each
  intermediate activation is computed exactly once per tile. No recomputation
  redundancy. For D=H=128: BK2=128, BK3=128 → 4 tl.dot calls per tile.
- **Persistent kernel**: Launch `min(NUM_SMS, num_tiles)` programs. Each iterates
  tiles round-robin, amortizing launch overhead.
- **Grouped tile ordering**: GROUP_M=16 swizzle improves L2 cache locality for
  weight loads shared across row-groups.
- **FP32 accumulation**: All `tl.dot` accumulate in fp32 regardless of input
  dtype, with `.to(INPUT_DTYPE)` conversion only at tile boundaries.

### Training Forward Variant

Same megakernel as inference + extra `tl.store` calls to write intermediate
activations (h1, h2 for 3-layer; h1-h4 for 5-layer) to global memory. These
are needed by the backward pass for sigmoid derivative computation:
`softplus'(z) = sigmoid(z) = 1 - exp(-h)` where `h = softplus(z)`.

Saving post-activation `h` (rather than pre-activation `z`) avoids extra
register computation in the fused backward kernel, where the simple
`1 - exp(-h)` is faster than reconstructing sigmoid and softplus from `z`.

### Forward Heuristic

Pure integer math, no autotuning. Selects (BM, warps) based on tile occupancy:

```python
target = M * num_out_cols // NUM_SMS
if target <= 24:  BM, warps = 16, 4
elif target <= 48: BM, warps = 32, 8
elif target <= 384: BM, warps = 64, 8
else: BM, warps = 128, 8
```

For fp32: BM capped at 32 (register pressure from fp32 operand tiles).
For 5-layer fp16: BM capped at 64 (5 intermediate tiles in registers).

---

## 2. Backward Pass: Fully-Fused Megakernel

### Mathematical Derivation

Key identity: `softplus'(z) = sigmoid(z) = 1 - exp(-h)` — derivative computed
from the saved post-activation value, no need to save pre-activation.

3-layer backward:
```
dz2 = (grad @ W3.T) * (1 - exp(-h2))     # data gradient chain
dz1 = (dz2 @ W2.T) * (1 - exp(-h1))
dx  = dz1 @ W1.T
dW3 = h2.T @ grad                          # weight gradients
dW2 = h1.T @ dz2
dW1 = x.T @ dz1
```

### Fully-Fused Architecture

The backward kernel computes **all outputs** (dx + all dW) in a single launch:

1. **Data-gradient chain** (dx): Uses the same register-tiled fusion as forward,
   with transposed weight loads and `sigmoid * multiply` instead of softplus.
   dh and dz intermediates stay in registers.

2. **Weight gradients** (dW1-dW3/dW5): Accumulated atomically within the same
   kernel using `tl.atomic_add`. Each tile computes its local contribution
   `tl.dot(tl.trans(h_raw), dz)` and atomically adds to the shared dW matrix.
   All accumulation is in fp32 for precision.

3. **Hoisted H loads**: Activation tensors (h1, h2, ...) are loaded *before*
   their inner loops, making the raw INPUT_DTYPE version available for both the
   sigmoid derivative AND the weight gradient dot product.

### Hybrid Dispatch

Atomic contention scales with the number of tiles competing for the same weight
matrix entries. For small 128×128 weight matrices with many tiles, this becomes
a bottleneck. The solution:

```python
_ATOMIC_M_THRESHOLD = 32768  # 3-layer
_ATOMIC_M_THRESHOLD = 16384  # 5-layer

if M > threshold:
    # Fall back to optimized PyTorch ops
else:
    # Single fused Triton kernel with atomic weight grads
```

### Fallback Backward Optimizations

The fallback path uses two key optimizations to minimize overhead vs autograd:

1. **Fused `sigmoid_h_mul` Triton kernel**: Computes `(1 - exp(-h)) * dh → dz`
   in a single kernel launch, replacing 3 separate PyTorch ops (negate, exp,
   subtract, multiply). Reduces the element-wise kernel count from ~8 to 2
   for the sigmoid computations, matching autograd's kernel count.

2. **Interleaved weight gradients**: Weight gradient matmuls are computed
   immediately after the corresponding activation data is used, matching
   autograd's backward traversal order for better L2 cache locality:
   ```
   dh2 = grad @ W3.T     → dw3 = h2.T @ grad     (h2, grad still in L2)
   dz2 = sigmoid(h2)*dh2 → dh1 = dz2 @ W2.T
   dw2 = h1.T @ dz2      (h1, dz2 still in L2)
   ```

### Backward Heuristic

Same structure as forward heuristic with tighter BM cap due to higher register
pressure (extra H loads, extra dz stores, dW atomic accumulation):

- 3-layer: BM ≤ 64
- 5-layer fp16: BM ≤ 64
- 5-layer fp32: BM ≤ 32, warps ≤ 4

---

## 3. Dtype Support

Both kernels accept an `INPUT_DTYPE: tl.constexpr` parameter, making them fully
dtype-agnostic. All `.to(tl.float16)` calls were replaced with `.to(INPUT_DTYPE)`.
Accumulation is always fp32 regardless of input dtype.

Heuristics are dtype-aware: fp32 operand tiles use 2× the registers of fp16,
so BM caps are tighter for fp32.

### AMP Integration

Both modules use `@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)`
and `@torch.amp.custom_bwd(device_type="cuda")` for seamless integration with
PyTorch's Automatic Mixed Precision.

---

## 4. Benchmark Results

Hardware: RTX 4090 (128 SMs), D=H=128. PyTorch 2.11, TF32 enabled.
Three approaches compared: **Eager** (PyTorch), **Compile** (`torch.compile`
with `mode="max-autotune"`, properly warmed up), and **Fused** (our Triton
megakernel). All times are median of 5 rounds × 500 iterations.

### 4.1 Forward-Only Inference (fp16)

The fused kernel dominates at every batch size and layer count. torch.compile
is compiled and autotuned at the **exact benchmark shape** (not a proxy) with
200 extra warmup iterations after compilation, then measured as median of
5 rounds × 1000 iterations.

#### 3-Layer MLP

torch.compile is **slower than eager** — it has a flat ~0.081ms dispatch floor
from the compiled graph overhead. Our fused kernel's floor is ~0.029ms (single
kernel launch).

```
     N     Eager      Compile    Fused      F/Eager  C/Eager  F/Compile
   512     0.059ms    0.081ms    0.029ms     2.08x     0.74x     2.82x
  2048     0.061ms    0.083ms    0.033ms     1.86x     0.73x     2.54x
  4096     0.061ms    0.080ms    0.029ms     2.13x     0.75x     2.82x
 16384     0.055ms    0.083ms    0.029ms     1.89x     0.66x     2.85x
 32768     0.058ms    0.081ms    0.035ms     1.68x     0.72x     2.34x
 65536     0.084ms    0.083ms    0.054ms     1.55x     1.01x     1.54x
131072     0.203ms    0.210ms    0.105ms     1.93x     0.97x     1.99x
```

#### 5-Layer MLP

torch.compile IS **faster than eager** for 5L — eager's 5-op overhead
(~0.104ms) exceeds compile's dispatch floor (~0.081ms). But our fused kernel
still beats both by 1.3-2.2×.

```
     N     Eager      Compile    Fused      F/Eager  C/Eager  F/Compile
   512     0.104ms    0.081ms    0.038ms     2.78x     1.29x     2.16x
  2048     0.105ms    0.081ms    0.038ms     2.78x     1.30x     2.14x
  4096     0.103ms    0.082ms    0.037ms     2.74x     1.26x     2.17x
 16384     0.097ms    0.081ms    0.037ms     2.60x     1.20x     2.17x
 32768     0.097ms    0.083ms    0.057ms     1.70x     1.16x     1.46x
 65536     0.157ms    0.140ms    0.111ms     1.42x     1.13x     1.26x
131072     0.466ms    0.316ms    0.216ms     2.16x     1.48x     1.46x
```

### 4.2 Training: Forward + Backward (fp16)

Training includes the fused forward + backward pass. The fused backward uses
atomic weight gradients for small M or `_sigmoid_h_mul` + cuBLAS fallback for
large M.

#### 3-Layer MLP

```
     N     Eager      Compile    Fused      F/Eager  C/Eager
   512     0.453ms    0.370ms    0.361ms     1.25x     1.23x
  4096     0.538ms    0.475ms    0.549ms     0.98x     1.13x
 16384     0.431ms    0.440ms    0.429ms     1.00x     0.98x
 65536     0.337ms    0.422ms    0.614ms     0.55x     0.80x
131072     0.706ms    0.809ms    0.717ms     0.99x     0.87x
```

#### 5-Layer MLP

```
     N     Eager      Compile    Fused      F/Eager  C/Eager
   512     0.589ms    0.502ms    0.568ms     1.04x     1.17x
  4096     0.581ms    0.410ms    0.564ms     1.03x     1.42x
 16384     0.554ms    0.439ms    0.583ms     0.95x     1.26x
 65536     0.580ms    0.588ms    0.744ms     0.78x     0.99x
131072     1.274ms    1.393ms    1.288ms     0.99x     0.91x
```

### 4.3 Why torch.compile Behaves Differently for 3L vs 5L

torch.compile has a fixed dispatch floor (~0.081ms) — the cost of entering the
compiled graph and dispatching its internal Triton kernels, regardless of N.

**3L forward (compile slower):** Eager's 3-op overhead is only ~0.059ms, which
is *below* compile's 0.081ms floor. So compile's graph dispatch overhead exceeds
the compute savings from Inductor's optimized Triton kernels.

**5L forward (compile faster):** Eager's 5-op overhead is ~0.104ms, which is
*above* compile's floor. Now the graph dispatch overhead is amortized across
more fused operations, and Inductor's kernel selection wins.

**Training backward (compile competitive):** Inductor CAN fuse element-wise
backward ops (e.g., `softplus_backward` = `sigmoid(z) * grad` → single kernel),
reducing the backward kernel count. Since backward has more element-wise ops to
fuse, torch.compile's advantage materializes there (1.1-1.4× vs eager).

### 4.4 Why Fused Backward Is Slower at Large N

The fused backward at large N uses the PyTorch fallback path. Each tensor
operation dispatches through Python (~35μs overhead per op), vs autograd's
C++ backend which dispatches backward ops entirely in C++ without entering
the Python interpreter. For 8 ops in the backward, this adds ~0.28ms of
overhead — enough to negate the forward savings at large N.

This is a fundamental limitation of `torch.autograd.Function`. The only way
to eliminate it would be implementing the backward in C++/CUDA, avoiding
the Python interpreter entirely.

### 4.5 Summary

| Workload | Fused vs Eager | Fused vs Compile | Compile vs Eager |
|---|---|---|---|
| **3L fp16 inference** | **1.6–2.1×** | **1.5–2.9×** | 0.66–1.01× (slower!) |
| **5L fp16 inference** | **1.4–2.8×** | **1.3–2.2×** | 1.13–1.48× (faster) |
| **Training (N ≤ 4K)** | 1.0–1.25× | 0.7–1.0× | 1.1–1.4× |
| **Training (N ≥ 64K)** | 0.5–1.0× | 0.7–1.5× | 0.8–1.0× |

**Key observations:**
- **Forward inference: fused kernel wins at every batch size** — 1.4–2.8× vs
  eager, 1.3–2.9× vs torch.compile
- **torch.compile has a ~0.081ms dispatch floor** — slower than eager for 3L
  (eager floor ~0.059ms) but faster for 5L (eager floor ~0.104ms)
- **torch.compile is best for training backward** thanks to Inductor's
  element-wise fusion (sigmoid × grad → single compiled kernel)
- **Our fused backward is limited by Python dispatch** from autograd.Function —
  each of the 8 fallback ops has ~35μs Python overhead vs C++ autograd's ~0μs
- The fused kernel's sweet spot is **inference at any N** and **training at
  small N** where the fused backward kernel (with atomics) runs

---

## 5. Why Forward Wins More Than Backward

### Structural asymmetry

```
Forward:  L GEMMs + (L-1) activations → 1 kernel  (100% fused)
Backward: 2L GEMMs + (L-1) sig*mul   → complex fusion story
```

The weight gradients `dW = input.T @ dz` reduce over M (sum all rows), while
the data gradient chain works row-by-row. This means weight gradients either
require atomic accumulation (contention at large M) or separate cuBLAS calls.

### Memory traffic ratio

Forward megakernel reads X + weights, writes OUT only. All intermediates stay
in registers. Backward must additionally read saved activations (h1, h2, ...)
and write weight gradients. Total backward memory ≈ 6× forward.

### The backward dx kernel itself is ~2× slower than forward

Extra loads (saved activations for sigmoid), extra stores (or atomics for dW),
and tighter register pressure (BM cap at 64 instead of 128).

---

## 6. FP16 Precision Considerations

For deep networks (5-layer) with small initialization scales, gradients at early
layers can drop below the FP16 minimum normal (~6e-5), entering the subnormal
range where precision degrades catastrophically. This manifests as backward
correctness failures for dx and dW1.

**Mitigation**: Use initialization scales ≥ 0.1 (not 0.02) to keep gradients
within the normal FP16 range. In production, gradient scaling (as in AMP)
handles this automatically.

---

## 7. File Structure

Core files:
```
kernel.py       3-layer inference-only megakernel (KernelBench submission)
fused_mlp.py    3-layer training: fwd/bwd megakernels + autograd + nn.Module
fused_mlp5.py   5-layer training: fwd/bwd megakernels + autograd + nn.Module
reference.py    PyTorch reference model (KernelBench)
```

Reports:
```
mlp_fused.md                  This file — full optimization report
comparison_tinycudann.md      Source-level comparison with tiny-cuda-nn
deep_analysis_tinycudann.md   Blog-post: tiny-cuda-nn optimizations & Triton gaps
split_k_atomic_backward.md   Blog-post: split-K atomic backward optimization
generalizing_to_n_layers.md  Guide: extending to arbitrary layer count
weight_norm.md               Weight normalization integration & benchmarks
```

---

## 8. Optimization History

1. **Naive fusion** → single kernel with global memory intermediates
2. **Register-tiled recompute** → intermediates in registers, zero global traffic
3. **Persistent kernel + grouped tiles** → better SM utilization and L2 locality
4. **Dynamic heuristic** → replaces autotuning, adapts to any batch size
5. **Backward megakernel** → fused data-gradient chain (dx), DZ intermediates saved
6. **Fully-fused backward** → atomic dW accumulation eliminates separate cuBLAS calls
7. **Hybrid dispatch** → atomic kernel for small M, PyTorch fallback for large M
8. **Dtype-agnostic kernels** → INPUT_DTYPE constexpr for fp16/fp32 support
9. **5-layer generalization** → same pattern extended to deeper networks
10. **AMP integration** → custom_fwd/custom_bwd decorators
11. **Fused sigmoid_h_mul kernel** → reduces fallback backward from 14 to 8 kernel
    launches by computing `(1-exp(-h)) * dh` in a single Triton pass
12. **Interleaved weight gradients** → compute dW immediately after dh for L2 reuse
13. **Threshold tuning** → increased 3L threshold to 32K, 5L to 16K based on sweeps
14. **Weight normalization integration** → fused WN megakernel (1-launch fwd),
    zero-copy WN backward (reads fp32 DW column-wise, no .t().contiguous()),
    contiguous parameter storage. See `weight_norm.md`.
15. **tiny-cuda-nn optimizations attempted**:
    - N_ITERS=2 weight reuse: 0-6% gain, 50% regression at large N (L2 suffices)
    - num_stages>1 pipelining: OOM (shared memory limit)
    - Multi-stream backward: 0-9% slower (Python stream overhead)
    - Always-fallback backward: 10-232% faster standalone, but 100-114μs slower
      in full training due to Python dispatch (8 launches × 15μs > 1 atomic launch)

---

## 9. Comparison with tiny-cuda-nn (NVIDIA)

See `comparison_tinycudann.md` for the full source-level analysis. Key differences:

| | tiny-cuda-nn | Our Fused Triton |
|---|---|---|
| **Language** | Raw CUDA + WMMA intrinsics | Triton (Python DSL) |
| **Intermediates** | Shared memory + `__syncthreads` | Registers only (no sync) |
| **Weight reuse** | WMMA fragments: 128 batch elems per load | L2 cached via GROUP_M swizzle |
| **Weight grads** | CUTLASS split-K on C++ streams | Atomic fused (small M) or cuBLAS (large M) |
| **FP32 support** | No (fp16 only for fused path) | Yes (dtype-agnostic) |
| **Batch constraint** | Multiple of 128 required | Any size (masked loads) |
| **JIT fusion** | Inline MLP into user CUDA kernels (1.5-2.5×) | Not available |
| **Depth** | Any number of layers (runtime loop) | Fixed 3 or 5 (compile-time unroll) |

### Attempted tiny-cuda-nn Optimizations (Experimental Results)

We implemented and benchmarked four tiny-cuda-nn optimizations in Triton:

**1. N_ITERS=2 weight reuse** (load weights once, process 2 BM chunks):
```
N=256-16384: 0-5% improvement (within noise, L2 cache already effective)
N=32768:     6% improvement (sweet spot where L2 misses occasionally)
N=65536:     50% REGRESSION (doubled effective BM causes register spilling)
```

**2. num_stages>1 software pipelining**: OOM at all sizes. The kernel already
uses ~96KB shared memory; pipelining doubles that past the 101KB hardware limit.

**3. Multi-stream backward** (overlap weight grads on separate CUDA stream):
```
N=65536:  9% SLOWER (Python stream overhead > overlap benefit)
N=131072: ~same (overhead ≈ benefit, wash)
```

**4. Always-fallback backward** (remove atomic kernel entirely):
```
Standalone: cuBLAS fallback is 10-232% faster than atomic at all N
Full training: atomic kernel wins by 100-114μs because 1 launch has less
              Python overhead than 8 separate launches (~15μs × 8 = 120μs)
```

**Conclusion**: The performance gap with tiny-cuda-nn is NOT algorithmic — it's
caused by three abstraction layers: (1) Triton's L2 cache is ~6% slower than
WMMA fragment persistence, (2) no inter-warp shared-memory cooperation, and
(3) Python dispatch overhead in backward dominates all other factors (~120μs
added to every backward pass through `torch.autograd.Function`).

---

## 10. Possible Further Work

- **C++ backward extension**: Implementing the backward pass as a C++ CUDA
  extension would eliminate the ~120μs Python dispatch overhead from
  `torch.autograd.Function`, which is the single largest performance gap.
- **torch.compile for backward**: Inductor can fuse element-wise backward ops
  in C++ and already achieves 1.1-1.4× over eager for training.
- **Activation recomputation in backward**: Skip saving H in forward, recompute
  from X+weights in backward. Trades compute for memory at very large M.
- **Scaling to larger hidden dims**: Current heuristics optimized for H=128.
  Larger H would benefit from different block size ratios.
- **Triton shared memory API**: Future Triton versions may expose explicit
  shared memory control, enabling the warp-cooperative pattern used by
  tiny-cuda-nn without leaving the Python ecosystem.

---

## 11. Weight Normalization Integration

See `weight_norm.md` for the full analysis. Summary of the approach and results:

### Approach

Weight normalization reparameterizes each weight matrix as `W = g * v / ||v||`
where `g` is a per-output-neuron scalar magnitude and `v` is the direction
vector. This adds normalization overhead to every forward/backward call.

We implemented four optimizations to minimize this overhead:

1. **Custom Triton kernels**: `_wn_fwd_kernel` computes `W = g*v/||v||` and
   `inv_norm = 1/||v||` per row. `_wn_bwd_kernel` converts `dW → dv, dg`.
2. **Batched execution**: All weight matrices are processed in a single kernel
   launch per direction (forward: 1 launch for all 3/5 weights, backward: 1
   launch for all 3/5 dW→dv,dg conversions).
3. **Contiguous parameter storage**: All `v` vectors and `g` scalars stored as
   single contiguous parameter blocks (`v_all`, `g_all`), with pre-allocated
   output buffers. Eliminates all `torch.cat` and `torch.empty` allocations.
4. **Zero-copy backward**: `_wn_bwd_from_dw_T_kernel` reads DW directly in
   fp32 (K,N) layout using column-stride access, eliminating 7 kernel launches
   (3 casts + 3 contiguous copies + 1 concat) per backward call.

Total kernel launches: 1–2 forward (1 WN+megakernel at M≤40K, 2 at M>40K),
2 backward (1 megakernel + 1 WN). Zero dynamic allocations in forward.

### Results (fp16, D=H=128, N up to 300K)

| Workload | N range | vs Eager | vs Compile |
|---|---|---|---|
| **3L inference** | 256–300K | **1.16–2.30×** | 0.74–**1.87×** |
| **5L inference** | 256–300K | **1.50–2.38×** | 0.80–**1.61×** |
| **3L training** | 256–300K | **1.10–2.94×** | **1.05–3.82×** |
| **5L training** | 256–300K | **1.15–3.48×** | 0.81–**3.67×** |

Crossover point where Fused beats torch.compile:
- Inference: **N ≈ 512** (3-layer), **N ≈ 65K** (5-layer)
- 3L Training: **always wins** (worst: 1.05× at N=1024)
- 5L Training: **N ≈ 256** (1.09×), except N=8K dip (0.81×)

At large batch sizes, torch.compile's CUDAGraph overhead scales poorly with
distinct input sizes, while the fused kernel maintains near-constant overhead.
At N=131K, 3L training: Fused=0.65ms vs Compile=2.49ms (**3.82× faster**).
At N=131K, 5L training: Fused=1.13ms vs Compile=4.14ms (**3.67× faster**).
