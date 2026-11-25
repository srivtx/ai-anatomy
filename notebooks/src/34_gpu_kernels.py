# %% [markdown]
# # 34 · GPU kernels step by step
# Write the same matrix multiply four ways in CUDA (naive, coalesced, shared-memory tiled) and Triton
# (autotuned), time each against `torch.matmul`, and measure a shared-memory bank conflict. Every step is one
# idea from the “GPU kernels from first principles” chapter.
#
# **Hardware:** Colab T4 GPU. CUDA kernels compile with `load_inline` (≈1 minute the first time). Without a GPU,
# the CUDA part is skipped and the Triton kernel is checked for correctness in the interpreter.
#
# **You're done when** you have a GFLOPS table for every version and the bank-conflict slowdown with and
# without padding.

# %%
import os
import torch

GPU = torch.cuda.is_available()

if not GPU:
    os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl

dev = "cuda" if GPU else "cpu"

# %% [markdown]
# ## Three CUDA matmuls and a transpose
# * **naive**: consecutive threads compute consecutive *rows* of C, so in one warp the reads of A are N floats
#   apart: every load is a separate memory transaction.
# * **coalesced**: consecutive threads compute consecutive *columns*: one warp reads 32 adjacent floats of B in
#   a single transaction.
# * **tiled**: a 32×32 block of threads loads tiles of A and B into shared memory once and reuses each value 32
#   times, cutting global-memory traffic by ~32×.
# * **transpose**: reading a shared-memory tile by column makes all 32 threads of a warp hit the same bank (32-way
#   conflict). Padding each row to 33 floats shifts columns across banks.

# %%
CUDA_SRC = r"""
#include <torch/extension.h>
#define TILE 32
__global__ void naive(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < N && col < N) { float s = 0; for (int k = 0; k < N; ++k) s += A[row * N + k] * B[k * N + col]; C[row * N + col] = s; }
}
__global__ void coalesced(const float* A, const float* B, float* C, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < N && col < N) { float s = 0; for (int k = 0; k < N; ++k) s += A[row * N + k] * B[k * N + col]; C[row * N + col] = s; }
}
__global__ void tiled(const float* A, const float* B, float* C, int N) {
  __shared__ float As[TILE][TILE], Bs[TILE][TILE];
  int tx = threadIdx.x, ty = threadIdx.y, row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
  float s = 0;
  for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
    As[ty][tx] = (row < N && t * TILE + tx < N) ? A[row * N + t * TILE + tx] : 0.f;
    Bs[ty][tx] = (col < N && t * TILE + ty < N) ? B[(t * TILE + ty) * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) s += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < N && col < N) C[row * N + col] = s;
}
template <int PAD> __global__ void transpose(const float* in, float* out, int N) {
  __shared__ float tile[32][32 + PAD];
  int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8) if (x < N && y + j < N) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x; y = blockIdx.x * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8) if (x < N && y + j < N) out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];  // column read
}
torch::Tensor mm(torch::Tensor A, torch::Tensor B, int which) {
  int N = A.size(0); auto C = torch::zeros_like(A);
  const float *a = A.data_ptr<float>(), *b = B.data_ptr<float>(); float* c = C.data_ptr<float>();
  if (which == 0) { dim3 bl(16, 16), g((N + 15) / 16, (N + 15) / 16); naive<<<g, bl>>>(a, b, c, N); }
  else if (which == 1) { dim3 bl(16, 16), g((N + 15) / 16, (N + 15) / 16); coalesced<<<g, bl>>>(a, b, c, N); }
  else { dim3 bl(TILE, TILE), g((N + TILE - 1) / TILE, (N + TILE - 1) / TILE); tiled<<<g, bl>>>(a, b, c, N); }
  return C;
}
torch::Tensor tr(torch::Tensor A, int pad) {
  int N = A.size(0); auto O = torch::empty_like(A); dim3 bl(32, 8), g((N + 31) / 32, (N + 31) / 32);
  if (pad) transpose<1><<<g, bl>>>(A.data_ptr<float>(), O.data_ptr<float>(), N);
  else transpose<0><<<g, bl>>>(A.data_ptr<float>(), O.data_ptr<float>(), N);
  return O;
}
"""

if GPU:
    from torch.utils.cpp_extension import load_inline

    ext = load_inline(
        "kernels34",
        cpp_sources="torch::Tensor mm(torch::Tensor A, torch::Tensor B, int which);\ntorch::Tensor tr(torch::Tensor A, int pad);",
        cuda_sources=CUDA_SRC,
        functions=["mm", "tr"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

# %% [markdown]
# ## Triton, autotuned
# The same tiling idea, but Triton picks the tile sizes and pipelining from a list of candidates by timing them.

# %%
configs = (
    [
        triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
        for bm, bn, bk, w, s in [
            (64, 64, 32, 4, 3),
            (128, 64, 32, 4, 3),
            (128, 128, 32, 8, 3),
            (64, 128, 32, 4, 4),
        ]
    ]
    if GPU
    else [triton.Config({"BM": 32, "BN": 32, "BK": 32})]
)


@triton.autotune(configs=configs, key=["M", "N", "K"])
@triton.jit
def tmm(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        A = tl.load(
            a + rm[:, None] * K + (k + rk)[None, :],
            mask=(rm[:, None] < M) & ((k + rk)[None, :] < K),
            other=0.0,
        )
        B = tl.load(
            b + (k + rk)[:, None] * N + rn[None, :],
            mask=((k + rk)[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(A, B)
    tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_mm(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)
    tmm[lambda m: (triton.cdiv(M, m["BM"]), triton.cdiv(N, m["BN"]))](A, B, C, M, N, K)
    return C


n = 64 if not GPU else 4096
A, B = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
print("Triton matmul max error:", (triton_mm(A, B) - A @ B).abs().max().item())

# %% [markdown]
# ## Time everything


# %%
def time_ms(f, reps=10):
    f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


if GPU:
    flops = 2 * n**3
    rows = []
    ref = A @ B
    for name, f in [
        ("CUDA naive", lambda: ext.mm(A, B, 0)),
        ("CUDA coalesced", lambda: ext.mm(A, B, 1)),
        ("CUDA shared-memory tiled", lambda: ext.mm(A, B, 2)),
        ("Triton autotuned (fp32 in, tf32 dot)", lambda: triton_mm(A, B)),
        ("torch.matmul fp32", lambda: A @ B),
        ("torch.matmul fp16 (tensor cores)", lambda: A.half() @ B.half()),
    ]:
        err = (f().float() - ref).abs().max().item()
        ms = time_ms(f, 3 if "naive" in name else 10)
        rows.append((name, ms, flops / ms / 1e9, err))
    print(f"{'kernel':40s} {'ms':>9s} {'GFLOPS':>9s} {'max err':>9s}")
    for r in rows:
        print(f"{r[0]:40s} {r[1]:9.2f} {r[2]:9.0f} {r[3]:9.1e}")
    X = torch.randn(8192, 8192, device=dev)
    t0, t1 = time_ms(lambda: ext.tr(X, 0)), time_ms(lambda: ext.tr(X, 1))
    assert torch.equal(ext.tr(X, 1), X.T)
    print(
        f"transpose 8192²: tile[32][32] {t0:.2f} ms, padded tile[32][33] {t1:.2f} ms → bank conflicts cost {t0 / t1:.1f}×"
    )
else:
    print("No GPU: CUDA kernels skipped. On a T4, expect (fp32, 4096²) roughly:")
    for name, g in [
        ("CUDA naive", "300–600"),
        ("CUDA coalesced", "1,000–2,000"),
        ("CUDA shared-memory tiled", "2,000–3,000"),
        ("Triton autotuned", "60–80% of torch"),
        ("torch.matmul fp32", "4,000–5,000"),
        ("torch.matmul fp16", "25,000–40,000"),
    ]:
        print(f"   {name:28s} {g} GFLOPS")

# %% [markdown]
# **What to expect on a T4** (my estimates; record yours). Naive is slowest by an order of magnitude; simply
# swapping which index the threads walk (coalescing) gives a several-fold speed-up with the same arithmetic.
# Tiling adds another 1.5–2×. cuBLAS (`torch.matmul`) is still ~2× faster than our tiled kernel because it also
# tiles in registers, double-buffers loads and is tuned per GPU. Tensor cores in fp16 are another ~6–8× on top.
# The unpadded transpose is ~1.5–3× slower than the padded one: that's the 32-way bank conflict.
