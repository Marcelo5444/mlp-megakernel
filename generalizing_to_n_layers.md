# Generalizing the Fused MLP to N Layers

A practical guide to extending the register-tiled Triton megakernel from the
existing 3-layer and 5-layer implementations to arbitrary depth, based on
patterns observed, what worked, what didn't, and where the hard limits are.

---

## The Pattern: What 3-Layer and 5-Layer Have in Common

Both implementations follow the exact same structural recipe. Looking at them
side by side reveals a mechanical pattern that scales to any depth.

### Forward Kernel Structure

An L-layer MLP (`L-1` hidden layers with softplus + 1 output layer) produces
an `L`-level nested loop:

```
3-layer (L=3):                           5-layer (L=5):

for k3 in range(N2, BK3):               for k5 in range(N4, BK5):
  h2 = zeros                              h4 = zeros
  for k2 in range(N1, BK2):               for k4 in range(N3, BK4):
    h1 = zeros                               h3 = zeros
    for k1 in range(K1, BK1):                 for k3 in range(N2, BK3):
      h1 += X_tile @ W1_tile                    h2 = zeros
    h1 = softplus(h1)                            for k2 in range(N1, BK2):
    h2 += h1 @ W2_tile                            h1 = zeros
  h2 = softplus(h2)                                for k1 in range(K1, BK1):
  out += h2 @ W3_tile                                h1 += X_tile @ W1_tile
store(out)                                         h1 = softplus(h1)
                                                   h2 += h1 @ W2_tile
                                                 h2 = softplus(h2)
                                                 h3 += h2 @ W3_tile
                                               h3 = softplus(h3)
                                               h4 += h3 @ W4_tile
                                             h4 = softplus(h4)
                                             out += h4 @ W5_tile
                                           store(out)
```

**The recipe for L layers:**
1. The outermost loop iterates over columns of layer `L-1`'s output dimension
2. Each inner loop `i` (from outer to inner) iterates over layer `L-i`'s columns
3. The innermost loop iterates over the input dimension `K1`
4. After each inner loop completes, apply `softplus` to the accumulator
5. Feed the activated result into the next matmul via `tl.dot`
6. The final output accumulator has no activation

### Training Forward: Where to Store Intermediates

The training variant adds `tl.store` calls to save post-activation intermediates
(`h1`, `h2`, ..., `h_{L-1}`) needed by the backward pass. The critical pattern
for WHEN to store each intermediate is governed by **guard conditions**:

```python
# 3-layer: h1 is stored when k3_start == 0 (outermost hidden loop is at first iter)
if k3_start == 0:
    tl.store(H1_OUT + ..., h1_chunk)

# 5-layer: h1 is stored when ALL outer loops are at their first iteration
if k3_start == 0 and k4_start == 0 and k5_start == 0:
    tl.store(H1_OUT + ..., h1_chunk)

# 5-layer: h2 is stored when outer loops k4 and k5 are at first iteration
if k4_start == 0 and k5_start == 0:
    tl.store(H2_OUT + ..., h2_chunk)
```

**General rule for L layers:** Intermediate `h_i` (layer i's post-activation
output) is stored when ALL loop variables from layer `i+1` to layer `L-1` are
at their first iteration. This ensures each row of `h_i` is written exactly once
(not overwritten by redundant recomputations from outer loops).

Additionally, all `h_i` stores are gated by `is_first_n_out = (pn_out == 0)` to
prevent duplicate writes when there are multiple output-column tiles.

### Backward Kernel Structure

The backward mirrors the forward with two key differences:
1. Weight matrices are transposed (`W_i.T` instead of `W_i`)
2. Softplus is replaced by `sigmoid_from_h * dh` (activation derivative)
3. Weight gradient `tl.atomic_add` calls are interleaved at each layer

```
3-layer backward:                        5-layer backward:

for k3 in range(N1, BK3):               for k5 in range(N1, BK5):
  load h1, dh1 = zeros                    load h1, dh1 = zeros
  for k2 in range(N2, BK2):               for k4 in range(N2, BK4):
    load h2, dh2 = zeros                     load h2, dh2 = zeros
    for k1 in range(N3, BK1):                 for k3 in range(N3, BK3):
      dh2 += grad @ W3.T                        load h3, dh3 = zeros
      atomic_add(DW3, h2.T @ grad)               for k2 in range(N4, BK2):
    dz2 = dh2 * sigmoid(h2)                         load h4, dh4 = zeros
    atomic_add(DW2, h1.T @ dz2)                     for k1 in range(N5, BK1):
    dh1 += dz2 @ W2.T                                 dh4 += grad @ W5.T
  dz1 = dh1 * sigmoid(h1)                             atomic_add(DW5, ...)
  atomic_add(DW1, x.T @ dz1)                        dz4 = dh4 * sig(h4)
  dx += dz1 @ W1.T                                  atomic_add(DW4, ...)
store(dx)                                            dh3 += dz4 @ W4.T
                                                   dz3 = dh3 * sig(h3)
                                                   atomic_add(DW3, ...)
                                                   dh2 += dz3 @ W3.T
                                                 dz2 = dh2 * sig(h2)
                                                 atomic_add(DW2, ...)
                                                 dh1 += dz2 @ W2.T
                                               dz1 = dh1 * sig(h1)
                                               atomic_add(DW1, ...)
                                               dx += dz1 @ W1.T
                                             store(dx)
```

**The guard conditions for `dW_i` atomics** follow the same pattern as forward
stores: `dW_i` is computed only when all OUTER loop variables (layers closer to
the output) are at their first iteration, AND `is_first_out == True`.

### Heuristic Structure

Both implementations use the same heuristic formula:

```python
def _pick_config(M, ..., fp32=False):
    BK_i = min(128, N_i)     # for each hidden dim, max out BK
    BK_1 = min(64, K1)       # input dim uses smaller BK (wider input)
    BN_out = min(128, N_out)  # output tile width

    num_out_tiles = ceil(N_out / BN_out)
    target_bm = max(1, M * num_out_tiles // NUM_SMS)

    if fp32:
        BM ∈ {16, 32}        # tighter due to fp32 register pressure
    else:
        BM ∈ {16, 32, 64, 128}  # 3-layer
        BM ∈ {16, 32, 64}       # 5-layer (lower cap)
```

The key difference between 3-layer and 5-layer is **BM_max**:

| Layers | Forward BM_max (fp16) | Forward BM_max (fp32) | Backward BM_max |
|---|---|---|---|
| 3 | 128 | 32 | 64 |
| 5 | 64 | 32 | 64 |

### Autograd Function and Module

The pattern is identical — only the number of weight arguments changes:

```python
class FusedMLP_L_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, ..., wL):
        out, h1, h2, ..., h_{L-1} = _training_fwd(x, w1, ..., wL)
        ctx.save_for_backward(x, w1, ..., wL, h1, ..., h_{L-1})
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, ..., wL, h1, ..., h_{L-1} = ctx.saved_tensors
        return _bwd(grad_output, x, w1, ..., wL, h1, ..., h_{L-1})

class FusedMLP_L(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear2 = ... # L-2 hidden layers
        self.linearL = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        return FusedMLP_L_Function.apply(x, self.linear1.weight.t(), ...)
```

### Fallback Backward

The PyTorch fallback for large M is a simple chain. For L layers:

```python
if M > threshold:
    dh_{L-1} = grad @ wL.t()
    dwL = h_{L-1}.t() @ grad
    for i in range(L-1, 1, -1):
        dz_i = _sigmoid_h_mul(h_i, dh_i)
        dh_{i-1} = dz_i @ w_i.t()
        dw_i = h_{i-1}.t() @ dz_i
    dz_1 = _sigmoid_h_mul(h_1, dh_1)
    dx = dz_1 @ w1.t()
    dw1 = x.t() @ dz_1
```

This produces `2L + (L-1)` kernel launches (L data-grad GEMMs, L weight-grad
GEMMs, L-1 sigmoid_h_mul). For 3 layers: 8 launches. For 5 layers: 14 launches.

---

## What Worked

### 1. Setting BK equal to the hidden dimension (BK = H)

**Impact: Eliminated recomputation entirely.**

When the block size for each hidden dimension equals the hidden dimension itself
(BK2 = N1 = 128, BK3 = N2 = 128), each inner loop runs for exactly 1 iteration.
This means each intermediate activation is computed once and used immediately —
no redundant recomputation.

This was the single biggest performance win. For 128-wide hidden layers:
- Before (BK=64): each h1 tile computed 2× per h2 tile → 100% recompute overhead
- After (BK=128): each h1 tile computed 1× → zero recompute

**Generalizes to N layers**: Always set `BK_i = min(128, hidden_dim_i)`. If your
hidden dimension is 128 or smaller, every inner loop is a single iteration.

### 2. Persistent kernel with grouped tiles (GROUP_M=16)

**Impact: 5-15% improvement from better L2 cache locality.**

Instead of launching one thread block per tile, launch `min(NUM_SMS, num_tiles)`
programs and iterate round-robin. The GROUP_M swizzle ensures that adjacent
row-groups of tiles execute consecutively, keeping shared weight tiles warm in L2.

**Generalizes to N layers**: The weight matrices for all layers are the same size
(H×H for hidden layers), so L2 caching benefits equally regardless of depth.
GROUP_M=16 worked for both 3 and 5 layers without tuning.

### 3. FP32 accumulation with dtype-cast at tile boundaries

**Impact: Correctness with minimal performance cost.**

All `tl.dot` calls use `out_dtype=tl.float32`. The `.to(INPUT_DTYPE)` cast
happens once per tile boundary (before feeding into the next layer's matmul).
This gives fp32-accurate intermediate values while still using fp16 tensor cores.

**Generalizes to N layers**: Same pattern at every layer. No depth dependency.

### 4. Softplus numerical guard (`tl.where(h > 20.0, h, log(exp(h) + 1))`)

**Impact: Prevents exp overflow for large pre-activations.**

For `h > 20`, `softplus(h) ≈ h` to float32 precision. The guard avoids computing
`exp(20+)` which can overflow in fp16.

**Generalizes to N layers**: Applied identically at every activation layer.

### 5. Saving post-activation values for backward

**Impact: Simpler backward math, fewer registers.**

Instead of saving pre-activation `z` and recomputing `softplus(z)` in backward,
we save `h = softplus(z)` and use the identity:
`softplus'(z) = sigmoid(z) = 1 - exp(-softplus(z)) = 1 - exp(-h)`

This avoids needing to store or recompute `z`, and the backward derivative is
a single `1 - exp(-h)` applied element-wise.

**Generalizes to N layers**: Same identity at every layer. The number of saved
tensors scales linearly: an L-layer MLP saves `L-1` intermediates.

### 6. Hybrid backward dispatch (atomic for small M, fallback for large M)

**Impact: Best-of-both-worlds across batch sizes.**

The atomic kernel wins at small M (low contention, 1 launch) while the PyTorch
fallback wins at large M (no contention, but many launches). The threshold is
tunable per architecture:
- 3-layer: `_ATOMIC_M_THRESHOLD = 32768`
- 5-layer: `_ATOMIC_M_THRESHOLD = 16384`

**Generalizes to N layers**: Deeper networks have MORE atomic operations per tile
(one `tl.atomic_add` per weight matrix per tile), so contention grows faster.
The threshold should DECREASE with depth. Empirically:
- 3-layer: threshold ~ 32K
- 5-layer: threshold ~ 16K
- Estimated L layers: threshold ~ `64K / L` (rough heuristic, needs validation)

---

## What Didn't Work

### 1. N_ITERS=2 weight reuse (loading weights once, processing 2 BM chunks)

**Idea**: Mimic tiny-cuda-nn's fragment persistence by manually loading weight
tiles once and using them for 2 consecutive batch chunks.

**Result**: 0-6% gain at small N, 50% regression at N=65536.

**Why it failed**: Doubling the effective BM doubles the number of register tiles
that must coexist (2× h_i chunks per layer). The Triton compiler spills registers
to local memory, which is slower than just reloading weights from L2 cache.

**Implication for N layers**: The regression would be WORSE with more layers
because there are more intermediate tiles competing for registers. At L=5 with
N_ITERS=2, you'd have 10 accumulator tiles simultaneously — guaranteed spilling.

### 2. Software pipelining (num_stages > 1)

**Idea**: Overlap global memory loads with computation by using multiple pipeline
stages.

**Result**: `OutOfResources: shared memory 106496 > hardware limit 101376`.

**Why it failed**: The kernel already uses ~96KB of shared memory for staging the
large BK=128 tiles. Adding a second pipeline stage doubles the staging buffer.

**Implication for N layers**: More layers means more weight tiles loaded per tile,
which means MORE shared memory pressure. Software pipelining becomes LESS feasible
with depth, not more.

### 3. Multi-stream backward for weight gradients

**Idea**: Launch weight gradient GEMMs on separate CUDA streams to overlap with
the data gradient chain (like tiny-cuda-nn does in C++).

**Result**: 0-9% slower.

**Why it failed**: Each Python `torch.cuda.stream()`, `record_event()`, and
`wait_event()` adds ~10-20μs of overhead. For 128-wide GEMMs that take ~20-50μs,
the overhead exceeds the overlap benefit.

**Implication for N layers**: More layers means more weight gradient GEMMs, which
means more stream management overhead. At L=5 with 5 weight gradients, you'd need
5 stream switches (~75-100μs overhead) to overlap ~150μs of compute. Net negative.

### 4. BM=128 for 5-layer forward

**Idea**: Use the same maximum BM as the 3-layer kernel.

**Result**: Register spilling, 20-40% slower than BM=64.

**Why it failed**: 5 accumulator tiles (h1 through h4 plus out_acc) at BM=128
requires `5 × 128 × 128 = 81,920` fp32 values in registers simultaneously.
That's 320KB — far beyond the 256KB register file per SM on RTX 4090.

**Implication for N layers**: BM_max should decrease as depth increases:

| L (layers) | Accumulators in registers | BM_max (fp16) | BM_max (fp32) |
|---|---|---|---|
| 3 | 3 (h1, h2, out) | 128 | 32 |
| 5 | 5 (h1, h2, h3, h4, out) | 64 | 32 |
| 7 | 7 | 32 | 16 |
| 9+ | 9+ | 16 | 16 |

Beyond ~9 layers, even BM=16 may not fit all accumulators in registers for
H=128. At that point, the megakernel approach breaks down.

---

## The Hard Limits: Where Megakernel Fusion Stops Working

### Limit 1: Register pressure scales with depth

Each layer adds one `[BM, BK]` accumulator tile that must live in registers
simultaneously. For fp16 with BK=128:

```
Registers per accumulator tile = BM × BK × 4 bytes (fp32 accum) / 4 bytes per reg
  = BM × 128 registers

Total registers = L × BM × 128

Hardware limit (RTX 4090): 65536 registers per SM, 255 registers per thread
At 8 warps × 32 threads = 256 threads per SM:
  Available per thread = 65536 / 256 = 256 registers
```

For BM=64, each tile uses `64 × 128 / 256_threads × 4B/4B ≈ 32 registers`
(simplified — actual Triton allocation is more complex). With overhead for
offsets, masks, and temporaries, ~L=7-9 is the practical limit for BM ≥ 16.

**Workaround**: For L > 8, you could chunk the MLP into blocks of 4-5 layers,
fuse each block into a megakernel, and chain them with global memory intermediates.
This trades some memory traffic for feasibility.

### Limit 2: Compile-time unrolling via `tl.static_range`

Triton's `tl.static_range` unrolls loops at compile time. For an L-layer MLP,
the kernel has L levels of nesting, each unrolled. If BK < hidden_dim (requiring
multiple iterations), the unrolled code size grows exponentially.

For H=128 with BK=128, each loop is 1 iteration, so unrolling is free. But for
H=256 with BK=128, each inner loop is 2 iterations, and the total unrolled code is
`2^L` copies of the inner body. At L=10, that's 1024 copies — the kernel compile
time and instruction cache pressure become prohibitive.

**Workaround**: Use runtime loops (`range()` instead of `tl.static_range()`) for
the innermost dimension. This prevents full unrolling but doesn't affect the layer
nesting structure.

### Limit 3: Triton JIT compilation time

Each unique kernel configuration (L, BM, BK, dtype) produces a separate compiled
kernel. For L=10 with 4 BM variants × 2 dtypes, that's 8 kernel variants, each
taking 5-30 seconds to compile on first use.

**Workaround**: Pre-warm the cache by calling each variant once at startup, or
use `@triton.jit(do_not_specialize=...)` to reduce the number of specializations.

### Limit 4: Backward kernel complexity

The backward kernel has TWICE the nesting depth of the forward (data-gradient
chain + interleaved weight gradient atomics). At L=5, the backward kernel already
has ~160 lines of core loop logic. At L=10, it would be ~350 lines with deeply
nested guard conditions for the atomic writes.

The guard conditions for `dW_i` also become harder to reason about:

```python
# For layer i's weight gradient dW_i:
# Must be gated by ALL outer loops (layers closer to output) being at first iter

# 3-layer dW3:  if is_first_out and k3_start == 0
# 5-layer dW5:  if is_first_out and k3==0 and k4==0 and k5==0
# 10-layer dW10: if is_first_out and k3==0 and k4==0 and ... and k9==0
```

**Workaround**: For L > 5, consider generating the kernel code programmatically
(see next section).

---

## How to Implement N-Layer: A Code Generation Approach

Given that the 3-layer and 5-layer kernels follow an identical mechanical pattern,
the most practical path to arbitrary depth is **code generation** — a Python
function that produces the Triton kernel source string for any L.

### The Generator Structure

```python
def generate_fused_mlp_kernel(L: int, direction: str = "forward") -> str:
    """Generate Triton kernel source for an L-layer fused MLP.

    L: number of weight matrices (L-1 hidden layers + 1 output layer)
    direction: "forward", "forward_training", or "backward"
    """
    lines = []

    # 1. Generate function signature
    #    Weights: W1, W2, ..., WL
    #    Hidden dims: N1, N2, ..., N_{L-1}, N_L (output)
    #    Block sizes: BK1, BK2, ..., BK_L
    #    If training: H1_OUT, H2_OUT, ..., H_{L-1}_OUT
    weight_args = ", ".join(f"W{i}" for i in range(1, L+1))
    dim_args = ", ".join(f"N{i}: tl.constexpr" for i in range(1, L))
    # ... etc

    # 2. Generate nested loop structure
    for layer in range(L, 1, -1):  # outer to inner
        lines.append(f"for k{layer}_start in tl.static_range(0, N{layer-1}, BK{layer}):")
        lines.append(f"    h{layer-1}_chunk = tl.zeros((BM, BK{layer}), dtype=tl.float32)")

    # 3. Innermost loop (input layer)
    lines.append(f"for k1_start in tl.static_range(0, K1, BK1):")
    lines.append(f"    h1_chunk += tl.dot(X_tile, W1_tile, ...)")

    # 4. Close loops with activation + next matmul
    for layer in range(1, L):
        lines.append(f"h{layer}_chunk = softplus(h{layer}_chunk)")
        if training:
            guard = " and ".join(f"k{j}_start == 0" for j in range(layer+1, L))
            if guard:
                lines.append(f"if {guard}:")
            lines.append(f"    tl.store(H{layer}_OUT + ..., h{layer}_chunk)")
        lines.append(f"h{layer+1}_chunk += tl.dot(h{layer}_chunk, W{layer+1}_tile, ...)")

    return "\n".join(lines)
```

This is a sketch — the actual generator would need to handle strides, offsets,
masks, and the backward direction with atomic weight gradients. But the point
is that the logic is completely mechanical and can be generated from a loop count.

### The Heuristic Generator

```python
def generate_pick_config(L: int):
    """Generate heuristic function for L-layer MLP."""

    # BM_max decreases with depth due to register pressure
    if L <= 3:
        bm_choices_fp16 = [16, 32, 64, 128]
    elif L <= 5:
        bm_choices_fp16 = [16, 32, 64]
    elif L <= 7:
        bm_choices_fp16 = [16, 32]
    else:
        bm_choices_fp16 = [16]

    bm_choices_fp32 = [16, 32] if L <= 5 else [16]

    # BK always maxed to hidden dim
    # Thresholds for target_bm remain the same (24, 48, 384)
    # num_warps: 4 for BM=16, 8 for BM≥32
```

### The Atomic Threshold Generator

```python
def estimate_atomic_threshold(L: int) -> int:
    """Estimate the M threshold for switching from atomic to fallback.

    More layers = more atomic ops per tile = lower threshold.
    The fallback has 2L + (L-1) launches, so its Python overhead also grows.
    """
    atomic_ops_per_tile = L  # one dW_i per layer
    fallback_launches = 3 * L - 2  # L data GEMMs + L weight GEMMs + (L-1) sigmoid

    # Sweet spot: where atomic contention cost = fallback Python overhead
    # Empirical: ~32K for L=3, ~16K for L=5
    # Linear fit: threshold ≈ 96K / L
    return max(4096, 96_000 // L)
```

---

## Practical Recommendations

### For L ≤ 5: Write the kernel by hand

The 3-layer and 5-layer kernels are 100-160 lines of core logic each. Manually
writing them ensures correctness and allows per-layer tuning. The existing code
can serve as a direct template.

### For L = 6-8: Use code generation

The pattern is mechanical enough that a generator is more reliable than hand-writing
300+ lines of nested loops with correct guard conditions. Generate the kernel string,
use `exec()` to compile it, and verify against a PyTorch reference.

### For L > 8: Chunk into blocks

Fuse layers in groups of 4-5 (e.g., a 12-layer MLP as three 4-layer fused blocks).
Each block is a manageable megakernel. The inter-block intermediates go through
global memory, but the intra-block intermediates stay in registers.

This "block fusion" approach trades ~2MB of intermediate traffic per block boundary
for feasible register pressure. For H=128 and M=4096, each intermediate is
4096 × 128 × 2 = 1MB — at 1TB/s bandwidth, that's ~1μs per boundary. For a
12-layer MLP with 2 boundaries, the overhead is ~2μs — negligible compared to
the ~10μs saved by fusing within each block.

### For variable depth at runtime: Use the fallback

If the number of layers is not known at compile time, the megakernel approach
doesn't apply (Triton requires compile-time loop bounds via `tl.static_range`).
Use the forward megakernel for inference (compile a kernel per depth on first use)
and the PyTorch fallback for training.

---

## Summary Table

| Aspect | 3-Layer | 5-Layer | N-Layer (Predicted) |
|---|---|---|---|
| Forward nesting depth | 3 levels | 5 levels | L levels |
| Register tiles alive | 3 | 5 | L |
| BM_max (fp16) | 128 | 64 | ~512/L |
| BM_max (fp32) | 32 | 32 | ~128/L |
| Atomic threshold | 32K | 16K | ~96K/L |
| Fallback launches | 8 | 14 | 3L-2 |
| Backward BM_max | 64 | 64 | ~256/L |
| Practical depth limit | — | — | ~8-9 (megakernel) |
| Code lines (kernel) | ~120 | ~160 | ~40L |
| Compilation time | ~5s | ~10s | ~2L seconds |

The fused megakernel approach is most effective for shallow-to-medium depth
networks (L ≤ 8) with moderate hidden widths (H ≤ 256). Beyond that, block
fusion or alternative strategies (shared memory pipeline like tiny-cuda-nn)
become necessary.
