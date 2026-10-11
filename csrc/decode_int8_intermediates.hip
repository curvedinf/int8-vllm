// SPDX-License-Identifier: Apache-2.0
// Decode-only TP4: INT8 G128 on every peer read, fused residual/RMSNorm.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

namespace {
constexpr int MaxRows = 64;
constexpr int Hidden = 5120;
constexpr int MaxElements = MaxRows * Hidden;
struct alignas(128) Signal {
  uint32_t start[MaxRows][4];
  uint32_t end[MaxRows][4];
  uint32_t epoch[MaxRows];
};
static_assert(sizeof(Signal) <= 4096);
struct Peers { char* data[4]; Signal* signals[4]; };

__device__ void barrier(Peers p, int rank, bool end) {
  __syncthreads();
  auto* self = p.signals[rank];
  uint32_t epoch = self->epoch[blockIdx.x] + 1;
  if (threadIdx.x < 4) {
    int peer = threadIdx.x;
    auto* dst = end ? &p.signals[peer]->end[blockIdx.x][rank]
                    : &p.signals[peer]->start[blockIdx.x][rank];
    auto* src = end ? &self->end[blockIdx.x][peer]
                    : &self->start[blockIdx.x][peer];
    __scoped_atomic_store_n(dst, epoch, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(src, __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) != epoch) {}
  }
  __syncthreads();
  if (threadIdx.x == 0) self->epoch[blockIdx.x] = epoch;
}

__device__ float bf16(float x) { return float(__hip_bfloat16(x)); }

__device__ float peer_sum(Peers p, int row, int col) {
  float sum = 0;
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    float scale = reinterpret_cast<const float*>(p.data[r] + MaxElements)[
        row * (Hidden / 128) + col / 128];
    int code = reinterpret_cast<const int8_t*>(p.data[r])[row * Hidden + col];
    sum += float(code) * scale;
  }
  // Preserve the existing all-reduce output rounding, in registers only.
  return bf16(sum);
}

__device__ float conv_value(Peers p, int row, int col,
                            const __hip_bfloat16* delta,
                            const __hip_bfloat16* base, int block, int group) {
  float x = peer_sum(p, row, col);
  if (!delta) return x;
  int groups = Hidden / group;
  // DFlash2's deployed convolution has exactly two taps. Each elementary
  // BF16 multiply/add retains the original convolution's rounding points.
  float c0 = bf16(float(base[col]) + float(delta[(row * 2) * groups + col / group]));
  float y = bf16(c0 * x);
  if (row > 0 && row % block > 0) {
    float c1 = bf16(float(base[Hidden + col]) +
                    float(delta[(row * 2 + 1) * groups + col / group]));
    y = bf16(y + bf16(c1 * peer_sum(p, row - 1, col)));
  }
  return y;
}

__global__ void decode_int8_g128_car_rmsnorm(
    Peers p, const __hip_bfloat16* residual, const __hip_bfloat16* weight,
    __hip_bfloat16* output, __hip_bfloat16* residual_out,
    float epsilon, int rank, bool gemma, bool normalize,
    const __hip_bfloat16* delta, const __hip_bfloat16* base,
    int block, int group) {
  barrier(p, rank, false);
  int row = blockIdx.x;
  float values[Hidden / 256];
  float sq = 0;
  #pragma unroll
  for (int j = 0; j < Hidden / 256; ++j) {
    int col = threadIdx.x + j * 256;
    float value = conv_value(p, row, col, delta, base, block, group)
                    + float(residual[row * Hidden + col]);
    values[j] = value;
    sq += value * value;
    residual_out[row * Hidden + col] = __hip_bfloat16(value);
  }
  if (normalize) {
    #pragma unroll
    for (int d = 32; d; d /= 2) sq += __shfl_down(sq, d, 64);
    __shared__ float reduction[4];
    if (threadIdx.x % 64 == 0) reduction[threadIdx.x / 64] = sq;
    __syncthreads();
    if (threadIdx.x == 0)
      reduction[0] = rsqrtf((reduction[0] + reduction[1] +
                            reduction[2] + reduction[3]) / Hidden + epsilon);
    __syncthreads();
    float inv = reduction[0];
    #pragma unroll
    for (int j = 0; j < Hidden / 256; ++j) {
      int col = threadIdx.x + j * 256;
      float w = float(weight[col]);
      float value = values[j] * inv;
      // Gemma keeps the weight multiply in FP32. Ordinary RMSNorm's
      // native implementation rounds the normalized activation first.
      value = gemma ? value * (1.0f + w) : bf16(value) * w;
      output[row * Hidden + col] = __hip_bfloat16(value);
    }
  }
  // Includes completion of all convolution peer reads before reuse.
  barrier(p, rank, true);
}
}  // namespace

extern "C" int decode_int8_zero(void* ptr, void* stream) {
  return int(hipMemsetAsync(ptr, 0, 4096, static_cast<hipStream_t>(stream)));
}

extern "C" int decode_int8_norm(
    const void* q, const void* scales, const void* residual, const void* weight,
    void* output, void* residual_out, int rows, float epsilon, int rank,
    int gemma, int normalize, const void* delta, const void* base,
    int block, int group, void** buffers, void** signals, void* stream) {
  if (rows < 1 || rows > MaxRows || rank < 0 || rank >= 4) return -1;
  if (delta && (!base || block <= 0 || group <= 0 || Hidden % group)) return -2;
  auto st = static_cast<hipStream_t>(stream);
  Peers p;
  for (int r = 0; r < 4; ++r) {
    p.data[r] = static_cast<char*>(buffers[r]);
    p.signals[r] = static_cast<Signal*>(signals[r]);
  }
  auto err = hipMemcpyAsync(p.data[rank], q, rows * Hidden,
                            hipMemcpyDeviceToDevice, st);
  if (err != hipSuccess) return int(err);
  err = hipMemcpyAsync(p.data[rank] + MaxElements, scales,
      rows * (Hidden / 128) * sizeof(float), hipMemcpyDeviceToDevice, st);
  if (err != hipSuccess) return int(err);
  decode_int8_g128_car_rmsnorm<<<rows, 256, 0, st>>>(
      p, static_cast<const __hip_bfloat16*>(residual),
      static_cast<const __hip_bfloat16*>(weight),
      static_cast<__hip_bfloat16*>(output),
      static_cast<__hip_bfloat16*>(residual_out), epsilon, rank,
      gemma, normalize, static_cast<const __hip_bfloat16*>(delta),
      static_cast<const __hip_bfloat16*>(base), block, group);
  return int(hipGetLastError());
}
