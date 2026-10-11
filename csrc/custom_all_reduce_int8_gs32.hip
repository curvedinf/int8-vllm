// SPDX-License-Identifier: Apache-2.0
// Experimental TP4 MLP-only transport. Build separately with hipcc; no change
// to the native CAR ABI. All scratch/signals are private to this transport.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

namespace {
struct Signal {
  alignas(128) uint32_t start[36][16];
  alignas(128) uint32_t end[36][16];
  alignas(128) uint32_t flag[36];
};
struct Peers { char* data[4]; Signal* signals[4]; };
static_assert(sizeof(Signal) == 4864);

__device__ __forceinline__ void barrier(Peers p, int rank, bool end) {
  __syncthreads();
  Signal* self = p.signals[rank];
  uint32_t epoch = self->flag[blockIdx.x] + 1;
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
  if (threadIdx.x == 0) self->flag[blockIdx.x] = epoch;
}

template<typename T>
__global__ void mlp_car_gs32_pack(const T* input, char* stage, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;  // The public entry point requires whole 256-value blocks.
  float x = float(input[i]);
  float amax = fabsf(x);
  int bad = !isfinite(x);
  for (int d = 16; d; d /= 2) {
    amax = fmaxf(amax, __shfl_xor(amax, d, 32));
    bad |= __shfl_xor(bad, d, 32);
  }
  float scale = bad ? NAN : (amax == 0 ? 1.0f : amax / 127.0f);
  if (i % 32 == 0) reinterpret_cast<float*>(stage + n)[i / 32] = scale;
  int q = bad ? 0 : min(127, max(-127, int(nearbyintf(x / scale))));
  reinterpret_cast<int8_t*>(stage)[i] = int8_t(q);
}

template<typename T, bool TwoStage>
__global__ void mlp_car_gs32_reduce(Peers p, T* out, int n, int rank) {
  barrier(p, rank, false);
  int part = TwoStage ? n / 4 : n;
  int start = TwoStage ? rank * part : 0;
  for (int local = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
       local < part; local += gridDim.x * blockDim.x * 16) {
    int i = start + local;
    float sums[16] = {};
    // Fixed rank order for every size, unlike rotating native two-stage.
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      float scale = 0;
      if (threadIdx.x % 2 == 0)
        scale = reinterpret_cast<const float*>(p.data[r] + n)[i / 32];
      scale = __shfl(scale, 0, 2);
      uint4 codes = *reinterpret_cast<const uint4*>(p.data[r] + i);
      uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        float value = int8_t(words[j / 4] >> (8 * (j % 4))) * scale;
        sums[j] += value;
      }
    }
    struct alignas(16) Output { T values[16]; } packed;
    #pragma unroll
    for (int j = 0; j < 16; ++j) packed.values[j] = T(sums[j]);
    T* dest = TwoStage ? reinterpret_cast<T*>(p.signals[rank] + 1) : out;
    *reinterpret_cast<Output*>(dest + (TwoStage ? local : i)) = packed;
  }
  barrier(p, rank, true);
  if constexpr (TwoStage) {
    for (int local = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
         local < part; local += gridDim.x * blockDim.x * 16) {
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const uint4* src = reinterpret_cast<const uint4*>(
            reinterpret_cast<const T*>(p.signals[r] + 1) + local);
        uint4* dst = reinterpret_cast<uint4*>(out + r * part + local);
        dst[0] = src[0]; dst[1] = src[1];
      }
    }
    // Do not permit the next invocation to overwrite an unfinished gather.
    barrier(p, rank, true);
  }
}

template<typename T>
void launch(Peers p, const void* input, void* output, int n, int rank,
            hipStream_t stream) {
  mlp_car_gs32_pack<T><<<(n + 255) / 256, 256, 0, stream>>>(
      static_cast<const T*>(input), p.data[rank], n);
  if (n * sizeof(T) < 512 * 1024)
    mlp_car_gs32_reduce<T, false><<<min(16, (n + 4095) / 4096), 256, 0, stream>>>(
        p, static_cast<T*>(output), n, rank);
  else
    mlp_car_gs32_reduce<T, true><<<min(16, (n / 4 + 4095) / 4096), 256, 0, stream>>>(
        p, static_cast<T*>(output), n, rank);
}
}  // namespace

extern "C" int mlp_car_int8_zero_meta(void* signals, void* stream) {
  return int(hipMemsetAsync(signals, 0, sizeof(Signal),
                            static_cast<hipStream_t>(stream)));
}

extern "C" int mlp_car_int8_gs32(int fp16, const void* input, void* output,
                                 int n, int rank, void** buffers,
                                 void** signals, void* stream) {
  if (n <= 0 || n % 256 || rank < 0 || rank >= 4) return -1;
  Peers p;
  for (int r = 0; r < 4; ++r) {
    p.data[r] = static_cast<char*>(buffers[r]);
    p.signals[r] = static_cast<Signal*>(signals[r]);
  }
  auto st = static_cast<hipStream_t>(stream);
  if (fp16) launch<__half>(p, input, output, n, rank, st);
  else launch<__hip_bfloat16>(p, input, output, n, rank, st);
  return int(hipGetLastError());
}
