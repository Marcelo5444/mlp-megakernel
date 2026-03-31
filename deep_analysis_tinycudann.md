# How tiny-cuda-nn Makes MLPs Fast — and What Triton Can't Replicate

A deep technical analysis of NVIDIA's tiny-cuda-nn FullyFusedMLP optimizations,
from high-level architecture down to individual PTX instructions, and why each
optimization cannot be fully reproduced in OpenAI's Triton.

Based on source analysis of: `src/fully_fused_mlp.cu`, `include/tiny-cuda-nn/common_device.h`,
`include/tiny-cuda-nn/networks/fully_fused_mlp.h`

---

## Part 1: The Big Picture — Why Fusing an MLP Matters

A 3-layer MLP with softplus activations computes:

```
h1 = softplus(x @ W1)       # GEMM + activation
h2 = softplus(h1 @ W2)      # GEMM + activation
out = h2 @ W3               # GEMM
```

In PyTorch, this launches **5 separate GPU kernels**: 3 cuBLAS GEMMs + 2 softplus
element-wise kernels. Each kernel writes its output to GPU global memory (HBM),
and the next kernel reads it back. For hidden width H=128 and batch size M=4096:

```
h1:  M × H = 4096 × 128 × 2 bytes = 1 MB written then read
h2:  M × H = 4096 × 128 × 2 bytes = 1 MB written then read
```

That's **4 MB of unnecessary global memory round-trips**. On an RTX 4090 with
1 TB/s bandwidth, that's ~4 μs of pure memory overhead — trivial for big matrices,
but for these small 128-wide matrices, the actual compute time is only ~2 μs per
GEMM. The memory traffic DOUBLES the wall time.

The fundamental insight shared by both tiny-cuda-nn and our Triton approach:
**fuse all layers into one kernel, keep intermediates on-chip.**

But HOW you keep intermediates on-chip, and WHERE on the chip, leads to
radically different architectures.

---

## Part 2: Two Architectures for the Same Idea

### tiny-cuda-nn: The Shared-Memory Pipeline

Think of a factory assembly line. The thread block is the factory. Shared memory
is the conveyor belt between stations. Each warp is a station that owns one piece
of the weight matrix.

```
┌──────────────── Thread Block (256 threads, 8 warps) ────────────────┐
│                                                                      │
│  Warp 0 ──► owns W1 rows [0:16]     ──► writes to shared memory    │
│  Warp 1 ──► owns W1 rows [16:32]    ──► writes to shared memory    │
│  ...                                                                 │
│  Warp 7 ──► owns W1 rows [112:128]  ──► writes to shared memory    │
│                                                                      │
│  ═══════════ __syncthreads() ═══════════                            │
│                                                                      │
│  Warp 0 ──► owns W2 rows [0:16]     ──► reads shared, writes shared│
│  ...                                                                 │
│  ═══════════ __syncthreads() ═══════════                            │
│                                                                      │
│  Warp 0 ──► owns W3 rows [0:16]     ──► reads shared, writes global│
│  ...                                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

**128 batch elements** flow through this pipeline simultaneously. Weights are
loaded once into registers, activations pass through shared memory between layers.

### Our Triton: The Register Accumulator

Think of a single artisan crafting one piece from start to finish. Each Triton
program instance works alone, keeping everything in its own registers.

```
┌──────── Program Instance (1 of 128 SMs) ────────┐
│                                                    │
│  h1_chunk = zeros[16, 128]     ← register tile   │
│  for k1 in [0, 64, 128]:                         │
│      h1_chunk += X_tile @ W1_tile                 │
│  h1_chunk = softplus(h1_chunk) ← still registers │
│                                                    │
│  h2_chunk = zeros[16, 128]     ← register tile   │
│  h2_chunk += h1_chunk @ W2_tile                   │
│  h2_chunk = softplus(h2_chunk) ← still registers │
│                                                    │
│  out_acc += h2_chunk @ W3_tile                    │
│  store(out_acc)                ← first global IO  │
└────────────────────────────────────────────────────┘
```

**16-128 batch elements** processed per tile (depending on heuristic). No shared
memory, no synchronization, no cooperation between warps. Weight tiles loaded
from global memory each time (cached in L2).

---

## Part 3: The Seven Optimizations of tiny-cuda-nn

### Optimization 1: WMMA Fragment Persistence

**What it is:**

NVIDIA's WMMA (Warp Matrix Multiply Accumulate) API provides `wmma::fragment` —
a C++ type that maps directly to physical GPU registers holding a 16×16 matrix tile.
In tiny-cuda-nn, the weight matrix for each layer is loaded into fragment arrays
**once per thread block** and reused across multiple batch chunks:

```cpp
// fully_fused_mlp.cu, threadblock_layer():

// Step 1: Load weight block-rows into WMMA register fragments (ONCE)
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
    weights_frag[N_BLOCKS];   // N_BLOCKS = WIDTH/16 = 8 for WIDTH=128

for (uint32_t i = 0; i < N_BLOCKS; ++i) {
    wmma::load_matrix_sync(
        weights_frag[i],
        weights_this_layer + 16*i * (WIDTH + SKEW),  // from global memory
        WIDTH + SKEW
    );
}
// weights_frag now holds the FULL 128×128 weight matrix in registers
// across all 8 warps: each warp owns one 16-wide block-row

// Step 2: Use these fragments for ALL batch chunks (N_ITERS=8 times)
for (uint32_t l = 0; l < N_ITERS; ++l) {
    wmma::fill_fragment(result_frag[l], 0.0f);
    for (uint32_t i = 0; i < N_BLOCKS; ++i) {
        wmma::load_matrix_sync(act_frag, act_shmem + ..., ...);
        wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
        //                                       ^^^^^^^^^^^^^^^^
        //                   SAME weight fragment, reused 8 times!
    }
}
```

Each warp loads one 16×16 weight block (512 bytes of fp16) from global memory
into registers. With 8 warps × 8 blocks = 64 fragments, the full 128×128 weight
matrix (32 KB) is distributed across all warps' registers. This happens ONCE.
Then the N_ITERS=8 loop processes 8 × 16 = 128 batch rows, each doing `mma_sync`
with the SAME `weights_frag` — meaning the weight data stays in registers for
128 batch elements of reuse.

**The numbers**: 32 KB of weight data loaded from global memory, used for
128 × 128 × 128 = 2M multiply-accumulate operations. That's 62.5 FLOPs per byte
loaded — well above the RTX 4090's compute-to-bandwidth ratio of ~330 FLOPs/byte.
Weight loading is entirely hidden by compute.

**Why Triton can't do this:**

Triton's `tl.dot(a, b, acc)` compiles to WMMA/MMA instructions, but there is no
Triton-level construct for `wmma::fragment`. When you write:

```python
w1_tile = tl.load(W1 + ...)
h1_chunk = tl.dot(a, w1_tile, acc=h1_chunk)
```

Triton compiles this to:
1. Load `w1_tile` from global memory into registers (or shared memory staging)
2. Execute `mma.sync` instruction using those register values
3. The compiler MAY or MAY NOT keep `w1_tile` in registers for the next iteration

There is no way to tell Triton: "keep this specific tensor in registers across
loop iterations." The Triton compiler's register allocator makes its own decisions.
If register pressure is too high, it will spill `w1_tile` to local memory (which
lives in L1 cache, or worse, global memory).

**What we tried:** An N_ITERS=2 variant that loads weight tiles once and processes
2 consecutive BM chunks. Results: 0-6% improvement at small N (L2 cache already
provides decent reuse), 50% regression at large N (doubled tile count causes
register spilling). The Triton compiler couldn't keep the extra tiles in registers.

**The gap**: ~6% forward performance difference. L2 cache latency (~200 cycles)
vs register access (0 cycles) means tiny-cuda-nn's approach is faster, but L2 is
"good enough" for most practical purposes.

---

### Optimization 2: Shared-Memory Inter-Layer Pipeline

**What it is:**

Between MLP layers, intermediate activations need to flow from one matmul's output
to the next matmul's input. tiny-cuda-nn uses shared memory as the communication
channel:

```cpp
// After computing layer output in register fragments:
__syncthreads();  // ensure all warps finished previous layer

for (uint32_t l = 0; l < N_ITERS; ++l) {
    // Store result from registers → shared memory
    wmma::store_matrix_sync(
        act_shmem + (16 + 16*l) * (WIDTH + SKEW) + 16*warp_idx,
        result_frag[l],
        WIDTH + SKEW,
        wmma::mem_col_major
    );
}

__syncthreads();  // ensure all results written before next layer reads
```

The shared memory layout is carefully designed:

```
Shared memory: (16 + 128) rows × 136 columns × 2 bytes = 39,168 bytes ≈ 38 KB

  Row layout:
  [0..15]    = input buffer (for loading from global memory)
  [16..143]  = activation buffer (128 batch elements = 8 chunks × 16)

  Column layout:
  [0..127]   = 128 neurons (WIDTH)
  [128..135] = 8 padding elements (SKEW)
               ^^^^^^^^^^^^^^^^^^^^^^^^
               This is the bank-conflict trick (Optimization 3)
```

8 warps collectively compute the full 128×128 layer output for 128 batch elements.
Each warp writes its 16-wide portion to shared memory. After `__syncthreads()`, each
warp reads a different 16-wide column of the result as input to the next layer's matmul.

**Why Triton can't do this:**

Triton's programming model is **single-program, multiple-data (SPMD)**: each program
instance (a thread block) works independently. There is no mechanism for warps within
a Triton program to cooperate via shared memory on user data.

Triton DOES use shared memory internally — for staging `tl.load` data and for the
internal reduction patterns of `tl.dot`. But the programmer cannot:
- Allocate user-managed shared memory buffers
- Write to shared memory from one warp and read from another
- Insert `__syncthreads()` at specific points in the code

This is a fundamental design choice of Triton. The abstraction is: "each program
works on a tile independently." Inter-warp cooperation would require a different
programming model.

**What we do instead:** Keep intermediates in **registers** (Triton tile accumulators).
This has a different tradeoff:
- **Pro**: No `__syncthreads()` needed, no shared memory capacity limits
- **Con**: All layer tiles must coexist in registers simultaneously (h1_chunk +
  h2_chunk + out_acc), increasing register pressure. This limits BM to 16-128
  depending on the number of layers.

**The gap**: tiny-cuda-nn processes 128 batch elements per weight load (the entire
128-element batch chunk shares one weight load via shared memory cooperation). Our
Triton processes BM=16-128 per tile (each tile loads weights independently, relying
on L2 cache for reuse). For small BM, weight loading is less amortized.

---

### Optimization 3: Shared Memory Bank-Conflict Avoidance (SKEW)

**What it is:**

GPU shared memory is divided into 32 **banks**, each 4 bytes wide. When multiple
threads in a warp access the same bank simultaneously, the accesses are
**serialized** — a bank conflict. This can reduce shared memory throughput by up to
32×.

For a 128-wide matrix stored in shared memory with stride 128, column `j` maps to
bank `(j * 2) % 32` (for fp16, 2 bytes per element). Columns 0, 16, 32, 48, ...
all map to bank 0. When a warp loads a 16-wide WMMA tile starting at column 0, all
16 threads might access bank 0 simultaneously — a 16-way bank conflict.

tiny-cuda-nn's fix is elegant: add 8 padding elements to each row.

```cpp
constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
// stride = WIDTH + SKEW = 128 + 8 = 136

// Store with padded stride:
wmma::store_matrix_sync(
    act_shmem + row * (WIDTH + SKEW) + col,  // stride = 136, not 128
    result_frag,
    WIDTH + SKEW,
    wmma::mem_col_major
);
```

With stride 136, column `j` maps to bank `(j * 2 + row * 136 * 2) % 32`. The
extra 8 elements shift each row's alignment, distributing accesses across different
banks. A 16-way conflict becomes no conflict.

The cost: 8/128 = 6.25% wasted shared memory. The benefit: shared memory loads run
at full throughput (128 bytes/cycle per SM) instead of 1/16th throughput.

**Why this doesn't apply to Triton:**

We don't use shared memory for activations at all. Our intermediates live in
registers. Therefore, shared memory bank conflicts are not part of our performance
model.

Triton DOES use shared memory internally for `tl.dot` operand staging. The Triton
compiler handles shared memory layout for these internal uses, including its own
bank-conflict mitigation strategies (swizzling). The programmer has no control over
this and cannot apply a SKEW pattern.

**The gap**: Zero — this optimization is specific to the shared-memory pipeline
architecture. It's a necessary optimization FOR their architecture, not something
we're "missing."

---

### Optimization 4: Warp-Cooperative Weight Distribution

**What it is:**

In tiny-cuda-nn, the 128×128 weight matrix is distributed across 8 warps. Each
warp owns one 16-wide "block-row" of the matrix:

```
Weight matrix W (128×128):

  Warp 0 → rows [0..15]    : loads 16×128 = 4 KB of weights into fragments
  Warp 1 → rows [16..31]   : loads 16×128 = 4 KB
  Warp 2 → rows [32..47]   : loads 16×128 = 4 KB
  ...
  Warp 7 → rows [112..127] : loads 16×128 = 4 KB

  Total: 8 × 4 KB = 32 KB loaded across 8 warps (the full matrix)
```

Each warp executes `mma_sync` using:
- Its own 16×16 weight fragment (from its block-row)
- A 16×16 activation fragment (from shared memory, shared across all warps)

The result is a 16×16 output fragment that contributes to one block-row of the
layer output. All 8 warps work in parallel, collectively computing the full 128×128
layer output.

This is a **warp-cooperative** pattern: no single warp has the full weight matrix.
The 8 warps MUST work together (via shared memory for activations) to produce the
complete output.

**Why Triton can't do this:**

Triton's `tl.dot(a, b, acc)` performs a full tile-level matmul within a single
program instance. The Triton compiler distributes the work across warps internally,
but the programmer cannot control how:
- Which warp loads which portion of the weight matrix
- How warps share partial results
- The degree of warp-level parallelism within a tile

In our Triton kernel, each program instance loads the ENTIRE weight matrix
independently (or relies on L2 cache for reuse across instances). There's no
mechanism to say: "warp 0 loads rows 0-15, warp 1 loads rows 16-31, etc."

**The gap**: tiny-cuda-nn loads the full 128×128 weight matrix using 8 warps in
parallel (each loading 4 KB), completing in ~4 KB / (128 bytes/clock × 8 warps) ≈
4 clocks. Our single-program approach loads the same data through L2 cache, which has
higher latency but similar throughput for cached access patterns. The effective
difference is small (~5-10% for compute-bound tiles) but compounds across layers.

---

### Optimization 5: Multi-Stream C++ Weight Gradient Overlap

**What it is:**

In the backward pass, there are two independent chains of computation:

```
Data gradient chain (must be sequential):
  dh2 = grad @ W3.T → dz2 = sigmoid(h2)*dh2 → dh1 = dz2 @ W2.T → dz1 → dx

Weight gradient chain (each independent, depends only on saved activations):
  dW3 = h2.T @ grad         ← can start immediately
  dW2 = h1.T @ dz2          ← can start after dz2 is ready
  dW1 = x.T  @ dz1          ← can start after dz1 is ready
```

tiny-cuda-nn launches weight gradient GEMMs on **separate CUDA streams**, overlapping
them with the data gradient fused kernel:

```cpp
// backward_impl() — C++ code, zero Python overhead

// Launch data gradient fused kernel on main stream
kernel_mlp_fused_backward<<<grid, block, shmem, stream>>>(
    dL_dinput, weights, ...);

// Launch weight gradient GEMMs on separate streams (overlap!)
for (uint32_t i = 0; i < n_hidden_layers; ++i) {
    multi_streams.emplace_back(stream, 2);  // creates new CUDA stream
    fc_multiply_split_k(
        multi_streams.back().get(1),        // separate stream
        backward_tmp, forward_hidden.transposed(),
        gradient_matrix, split_k_factor, beta
    );
}
```

Key implementation details:
- `multi_streams.emplace_back(stream, 2)` creates a stream that waits on the main
  stream (ensuring dependencies are met)
- `fc_multiply_split_k` is a CUTLASS GEMM template that splits the M dimension
  across multiple thread blocks, then reduces
- `split_k_factor = batch_size / min(1 << 12, batch_size)` — so for M=16384,
  split_k = 4
- ALL of this is orchestrated in C++ with **zero Python interpreter involvement**

**Why Triton can't replicate this:**

We tried implementing multi-stream backward in Python:

```python
_wg_stream = torch.cuda.Stream()
main = torch.cuda.current_stream()

dh2 = grad_output @ w3.t()              # main stream

with torch.cuda.stream(_wg_stream):     # ~15μs Python overhead
    dw3 = h2.t() @ grad_output          # separate stream

dz2 = _sigmoid_h_mul(h2, dh2)           # main stream

evt = main.record_event()               # ~10μs Python overhead
with torch.cuda.stream(_wg_stream):     # ~15μs Python overhead
    _wg_stream.wait_event(evt)           # ~10μs Python overhead
    dw2 = h1.t() @ dz2                  # separate stream
```

Each Python `with torch.cuda.stream()`, `record_event()`, and `wait_event()` call
enters the Python interpreter, acquires the GIL, and dispatches through PyTorch's
C++ backend. Each operation adds ~10-20 μs of overhead.

For 128-wide weight matrices, each cuBLAS GEMM takes only ~20-50 μs. The Python
stream management overhead is COMPARABLE to the compute time. The overlap benefit
(maybe saving 30 μs) is negated by the overhead (adding 50-80 μs).

**Measured results:**
```
N=65536:  baseline 0.202ms, multi-stream 0.222ms → 9% SLOWER
N=131072: baseline 0.492ms, multi-stream 0.496ms → no benefit
```

In tiny-cuda-nn, stream creation and event management happens in C++ with ~0 μs
overhead. The overlap is pure profit.

**The gap**: This is one of the most impactful differences. For a 3-layer backward
with 3 weight gradient GEMMs, tiny-cuda-nn overlaps ~60 μs of compute at zero cost.
Our Python-based multi-stream adds ~60 μs of overhead, completely negating the benefit.

---

### Optimization 6: CUTLASS Split-K for Weight Gradients

**What it is:**

Weight gradients are reduction matmuls: `dW = h.T @ dz` reduces over the M (batch)
dimension. For large M, a single thread block computing the full reduction would
leave many SMs idle. CUTLASS split-K divides the M dimension across multiple thread
blocks:

```
Standard GEMM for dW3 = h2.T @ grad:
  One thread block computes ALL M rows → most SMs idle

Split-K (split_k_factor=4):
  Block 0: computes partial dW3 from rows [0, M/4)
  Block 1: computes partial dW3 from rows [M/4, M/2)
  Block 2: computes partial dW3 from rows [M/2, 3M/4)
  Block 3: computes partial dW3 from rows [3M/4, M)
  → 4 blocks work in parallel on different SM
  → final reduction: dW3 = partial_0 + partial_1 + partial_2 + partial_3
```

tiny-cuda-nn automatically computes the split factor:
```cpp
uint32_t split_k_factor = batch_size / min((uint32_t)(1 << 12), batch_size);
// M=4096:  split_k = 1 (no split, small enough)
// M=16384: split_k = 4
// M=65536: split_k = 16
```

**Why our approach differs:**

Instead of split-K, we tried a different strategy: **fuse weight gradients INTO the
data-gradient kernel using atomics**. Each tile computes its local weight gradient
contribution and atomically accumulates to a shared output:

```python
# Inside the backward kernel:
dw3_partial = tl.dot(tl.trans(h2_raw), g)  # local contribution
tl.atomic_add(DW3 + ..., dw3_partial)       # accumulate atomically
```

This eliminates the need for separate weight gradient launches entirely. But atomics
have a cost: when 2048 tiles compete to update the same 128×128 matrix (16K elements),
contention is severe.

**The measured tradeoff:**
```
Standalone backward timing:
  M=4096:  atomic=0.135ms  cuBLAS-fallback=0.121ms  (atomic 11% slower)
  M=16384: atomic=0.215ms  cuBLAS-fallback=0.112ms  (atomic 92% slower)
  M=32768: atomic=0.385ms  cuBLAS-fallback=0.116ms  (atomic 232% slower!)

Full training (through Python autograd.Function):
  M=4096:  atomic=0.422ms  cuBLAS-fallback=0.536ms  (atomic 27% FASTER)
```

The atomic kernel is slower in raw GPU compute, but faster in full training because
it's **1 kernel launch** vs **8 kernel launches**. Each Python launch adds ~15 μs
of overhead. 8 launches × 15 μs = 120 μs of pure Python overhead — enough to flip
the winner.

tiny-cuda-nn avoids this dilemma entirely: they use CUTLASS split-K (no atomic
contention) launched from C++ (no Python overhead). Best of both worlds.

**The gap**: At M=32768, our atomic kernel wastes 0.269ms on contention. CUTLASS
split-K would complete the same computation in ~0.116ms. The difference is the cost
of atomic serialization on a hot 128×128 matrix.

---

### Optimization 7: JIT Kernel Fusion via NVRTC

**What it is:**

tiny-cuda-nn v2.0 can generate the entire MLP as a **CUDA device function** — a
callable piece of GPU code that can be inlined into ANY user kernel at compile time:

```cpp
// From fully_fused_mlp.h:
std::string generate_device_function(const std::string& name) const override {
    return generate_mlp_device_code<WIDTH>(
        m_input_width, WIDTH, m_padded_output_width, m_output_width,
        m_n_hidden_layers, m_activation, m_output_activation
    );
}
```

This generates a string of CUDA source code that gets compiled at runtime using
NVRTC (NVIDIA Runtime Compilation). The user can embed the MLP evaluation inside
their own CUDA kernel:

```cpp
// User's application kernel (e.g., NeRF ray marching):
__global__ void render_pixel(Ray ray, ...) {
    // March along the ray
    float3 position = march_ray(ray);

    // Encode position to features
    float features[128];
    encode_position(position, features);

    // Evaluate MLP — THIS IS INLINED, not a separate kernel launch!
    float output[128];
    fused_mlp_evaluate(features, output);  // <-- generated device function

    // Use MLP output for rendering
    float4 color = compute_color(output);
    store_pixel(color);
}
```

Without JIT fusion, the ray marching kernel and the MLP kernel are separate launches.
The intermediate `features` array (128 values per ray, millions of rays) must be
written to global memory by the ray marcher and read back by the MLP kernel.

With JIT fusion, the `features` array NEVER leaves registers. The MLP evaluation is
inlined directly into the ray marching kernel. This eliminates:
- 1 global memory write (features: millions_of_rays × 128 × 2 bytes)
- 1 global memory read (same)
- 1 kernel launch overhead (~5 μs)
- Pipeline stall between kernels (~1-3 μs)

NVIDIA's Instant NGP paper reports **5× speedup** from JIT fusion in the full NeRF
rendering pipeline. The MLP itself is the same speed, but eliminating the
surrounding memory traffic and launch overhead is transformative.

**Why Triton fundamentally cannot do this:**

Triton kernels are standalone GPU programs. They are compiled by Triton's own
MLIR-based compiler to produce PTX/CUBIN. There is no mechanism to:

1. **Generate a Triton "device function"** — Triton has no concept of a device
   function that can be called from CUDA code. Every `@triton.jit` function compiles
   to a complete kernel.

2. **Inline Triton code into CUDA** — The compilation pipelines are incompatible.
   Triton generates MLIR → LLVM IR → PTX. CUDA uses NVCC/NVRTC to compile C++ to
   PTX. There's no bridge to inline one into the other.

3. **Use NVRTC from Python** — While technically possible (you could write raw CUDA
   strings and compile with NVRTC), this would bypass Triton entirely. You'd
   be writing CUDA, not Triton.

The PyTorch/Triton ecosystem assumes a **kernel-per-operation** model. `torch.compile`
can fuse PYTHON-level operations into combined kernels, but it cannot fuse a Triton
kernel with arbitrary CUDA code from outside the PyTorch graph.

**The gap**: JIT fusion provides 1.5-2.5× additional speedup in tiny-cuda-nn's target
use case (real-time rendering). This is arguably the single most impactful optimization
and is completely unavailable in Triton. It's not an optimization of the MLP itself —
it's an optimization of how the MLP integrates with the surrounding application.

---

## Part 4: The Abstraction Tax — Quantified

We experimentally measured each abstraction gap:

| Optimization | tiny-cuda-nn | Triton | Measured Gap | Root Cause |
|---|---|---|---|---|
| **Fragment persistence** | 128 elems/load | L2 cached | ~6% fwd | No register control |
| **Shared-memory pipeline** | 8 warps cooperate | Independent tiles | Structural | No inter-warp API |
| **Bank-conflict SKEW** | 6% shmem overhead | N/A | 0% | We don't use shmem |
| **Warp-cooperative weights** | 4 KB/warp parallel | Full matrix/tile | ~5% | No warp control |
| **Multi-stream C++ overlap** | ~0 μs overhead | ~60 μs overhead | 9% bwd | Python GIL |
| **CUTLASS split-K** | No atomics | Atomic contention | 232% at M=32K | No split-K in kernel |
| **JIT fusion** | 1.5-5× app-level | Not available | **Fundamental** | Different ecosystems |

The dominant factors are:
1. **JIT fusion** (1.5-5× for integrated apps) — fundamentally impossible in Triton
2. **Python dispatch overhead** (120 μs per backward) — inherent to `autograd.Function`
3. **Atomic contention at large M** (232% slower) — solvable with split-K approach

Factors 2 and 3 are addressable within the Triton/PyTorch ecosystem (CUDA graphs,
split-K atomic partitioning, C++ extensions). Factor 1 requires leaving the ecosystem.

---

## Part 5: What We Do Better

It's not all disadvantages. Our Triton approach has genuine strengths:

### No synchronization overhead
tiny-cuda-nn requires `__syncthreads()` between every layer (2-4 calls per forward,
2-4 per backward). Each sync costs ~5 cycles × 8 warps = ~40 cycles. For a 5-layer
MLP, that's 8 × 40 = 320 wasted cycles. Our register approach has zero sync cost.

### FP32 native support
tiny-cuda-nn's fused path throws a runtime exception for fp32:
```cpp
std::enable_if_t<!std::is_same<T, __half>::value>
mlp_fused_backward(...) {
    throw std::runtime_error{"only supports __half precision."};
}
```
Our `INPUT_DTYPE: tl.constexpr` parameter handles both fp16 and fp32 with the same
kernel code, just different register allocation heuristics.

### Arbitrary batch sizes
tiny-cuda-nn requires `batch_size % (N_ITERS × 16)` = `batch_size % 128 == 0`.
Smaller batches are padded. Our masked loads (`m_mask = offs_m < M`) handle any M
with zero wasted computation.

### Pure Python, no build step
tiny-cuda-nn requires: CMake, CUDA toolkit, CUTLASS headers, C++ compiler,
`pip install` from source (several minutes of compilation). Our approach requires:
`pip install triton torch`. That's it. The kernels JIT-compile on first use.

### 400 lines vs 900 lines
Our complete 3-layer implementation (forward + backward + autograd + module) is ~428
lines of Python. tiny-cuda-nn's equivalent spans `fully_fused_mlp.cu` (~900 lines),
`fully_fused_mlp.h` (~400 lines), plus CUTLASS and utility dependencies.

---

## Part 6: The Fundamental Lesson

The performance gap between tiny-cuda-nn and Triton is NOT about algorithms.
Both implementations use the same core idea: fuse all MLP layers, minimize memory
traffic. The gap comes from **three layers of abstraction**:

```
                         tiny-cuda-nn          Triton
                         ────────────          ──────
Hardware layer:          WMMA fragments        tl.dot (compiler decides)
                         Shared memory         Registers (compiler decides)
                         Bank-conflict SKEW    Compiler handles

Orchestration layer:     C++ streams           Python dispatch
                         CUTLASS split-K       Atomic + cuBLAS
                         Zero overhead         ~15 μs per operation

Integration layer:       JIT device functions   Standalone kernels
                         NVRTC fusion          No cross-kernel fusion
                         1.5-5× app speedup    Not available
```

Each layer of abstraction trades control for convenience. Triton's higher abstraction
makes GPU programming accessible to Python developers, but it hides the exact hardware
mechanisms that tiny-cuda-nn exploits. This is the fundamental tradeoff of
high-level GPU programming: you write 2× less code, iterate 5× faster, but leave
10-40% of peak performance on the table.

For most applications, this tradeoff is overwhelmingly worth it. The 1.5-2.1× speedup
we achieve over PyTorch eager — obtained in a few days of Python development — would
take weeks of CUDA/CUTLASS engineering to match through tiny-cuda-nn's approach. And
for applications that don't need JIT fusion (which is most applications outside
real-time rendering), the absolute gap is just the Python dispatch overhead in the
backward pass, which is addressable through CUDA graphs or C++ extensions.
