// SPDX-License-Identifier: Apache-2.0
// TP4 GS128 transport using the native custom-AR IPC allocation mechanism.
// Codes and scales are copied as-is; only the reduction epilogue requantizes.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {
// Up to 64 CTAs; the Python side allocates 16 KiB, enough for sizeof(Signal).
constexpr int kMaxBlocks = 64;
struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][16];
  alignas(128) uint32_t end[kMaxBlocks][16];
  alignas(128) uint32_t epoch[kMaxBlocks];
};
struct Peers { char* data[4]; Signal* signals[4]; };
__device__ __forceinline__ void barrier(Peers p, int rank, bool end) {
  __syncthreads();
  Signal* self = p.signals[rank];
  uint32_t epoch = self->epoch[blockIdx.x] + 1;
  if (threadIdx.x < 4) {
    int peer = threadIdx.x;
    auto* dst = end ? &p.signals[peer]->end[blockIdx.x][rank]
                    : &p.signals[peer]->start[blockIdx.x][rank];
    auto* src = end ? &self->end[blockIdx.x][peer]
                    : &self->start[blockIdx.x][peer];
    __scoped_atomic_store_n(dst, epoch, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(src, __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < epoch) {}
  }
  __syncthreads();
  if (threadIdx.x == 0) self->epoch[blockIdx.x] = epoch;
}
__global__ void reduce(Peers p, int8_t* out, float* scales, int n, int rank) {
  barrier(p, rank, false);
  const int lane = threadIdx.x % 64;
  const int wave = threadIdx.x / 64;
  for (int g = blockIdx.x * 4 + wave; g < n / 128; g += gridDim.x * 4) {
    float a = 0.0f, b = 0.0f;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const auto* q = reinterpret_cast<const int8_t*>(p.data[r]);
      float s = reinterpret_cast<const float*>(p.data[r] + n)[g];
      a += float(q[g * 128 + lane]) * s;
      b += float(q[g * 128 + lane + 64]) * s;
    }
    float amax = fmaxf(fabsf(a), fabsf(b));
    int bad = !isfinite(a) || !isfinite(b);
    for (int d = 32; d; d /= 2) {
      amax = fmaxf(amax, __shfl_xor(amax, d, 64));
      bad |= __shfl_xor(bad, d, 64);
    }
    float s = bad ? NAN : fmaxf(amax / 127.0f, 1.0e-30f);
    if (lane == 0) scales[g] = s;
    out[g * 128 + lane] = bad ? 0 : int8_t(nearbyintf(a / s));
    out[g * 128 + lane + 64] = bad ? 0 : int8_t(nearbyintf(b / s));
  }
  barrier(p, rank, true);
}
}  // namespace
extern "C" int residual_g128_zero(void* signal, void* stream) {
  return int(hipMemsetAsync(signal, 0, sizeof(Signal),
                            static_cast<hipStream_t>(stream)));
}
extern "C" int residual_g128_reduce(
    const void* q, const void* s, void* out, void* scales, int n, int rank,
    void** buffers, void** signals, void* stream) {
  if (n <= 0 || n % 128 || rank < 0 || rank >= 4) return -1;
  Peers p;
  for (int r = 0; r < 4; ++r) {
    p.data[r] = static_cast<char*>(buffers[r]);
    p.signals[r] = static_cast<Signal*>(signals[r]);
  }
  auto st = static_cast<hipStream_t>(stream);
  auto rc = hipMemcpyAsync(p.data[rank], q, n, hipMemcpyDeviceToDevice, st);
  if (rc != hipSuccess) return int(rc);
  rc = hipMemcpyAsync(p.data[rank] + n, s, n / 128 * sizeof(float),
                      hipMemcpyDeviceToDevice, st);
  if (rc != hipSuccess) return int(rc);
  static const int env_blocks = [] {
    const char* b = std::getenv("VLLM_GFX908_G128_AR_BLOCKS");
    return b ? std::atoi(b) : 0;
  }();
  int cap = env_blocks > 0 ? env_blocks : 16;
  cap = cap > kMaxBlocks ? kMaxBlocks : cap;
  int blocks = (n / 128 + 3) / 4;
  if (blocks > cap) blocks = cap;
  reduce<<<blocks, 256, 0, st>>>(
      p, static_cast<int8_t*>(out), static_cast<float*>(scales), n, rank);
  return int(hipGetLastError());
}
