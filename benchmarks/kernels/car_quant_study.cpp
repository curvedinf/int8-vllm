// SPDX-License-Identifier: Apache-2.0
// Model-free experimental TP4 quantized CAR; NOT a serving implementation.
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx908 -shared -fPIC ...
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

namespace {
// Match csrc/custom_collective_common.cuh's IPC signal ABI and SYSTEM scope.
struct Signal {
  alignas(128) uint32_t start[36][16];
  alignas(128) uint32_t end[36][16];
  alignas(128) uint32_t flag[36];
};
struct Peers { void* data[4]; Signal* signals[4]; };
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
    __scoped_atomic_store_n(dst, epoch, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(src, __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < epoch) {}
  }
  __syncthreads();
  if (threadIdx.x == 0) self->flag[blockIdx.x] = epoch;
}

// Modes: 0=FP32 scale, 1=FP16 scale, 2=BF16 scale, 3=FP16 sqrt,
// 4=FP16 cbrt, 5=adaptive log FP16(offset,step), 6=log FP32(offset,step),
// 7=log FP16 with decoded magnitudes bounded to the finite BF16 domain.
template<int Mode> constexpr int meta_bytes() {
  if constexpr (Mode == 0) return 4;
  if constexpr (Mode == 5 || Mode == 7) return 4;
  if constexpr (Mode == 6) return 8;
  return 2;
}
template<int Mode>
__device__ __forceinline__ void store_meta(char* ptr, float a, float b) {
  if constexpr (Mode == 0 || Mode == 6) {
    reinterpret_cast<float*>(ptr)[0] = a;
    if constexpr (Mode == 6) reinterpret_cast<float*>(ptr)[1] = b;
  } else if constexpr (Mode == 2) {
    *reinterpret_cast<__hip_bfloat16*>(ptr) = __hip_bfloat16(a);
  } else {
    reinterpret_cast<__half*>(ptr)[0] = __float2half_rn(a);
    if constexpr (Mode == 5 || Mode == 7)
      reinterpret_cast<__half*>(ptr)[1] = __float2half_rn(b);
  }
}
template<int Mode>
__device__ __forceinline__ void load_meta(const char* ptr, float& a, float& b) {
  b = 0;
  if constexpr (Mode == 0 || Mode == 6) {
    a = reinterpret_cast<const float*>(ptr)[0];
    if constexpr (Mode == 6) b = reinterpret_cast<const float*>(ptr)[1];
  } else if constexpr (Mode == 2) {
    a = float(*reinterpret_cast<const __hip_bfloat16*>(ptr));
  } else {
    a = __half2float(reinterpret_cast<const __half*>(ptr)[0]);
    if constexpr (Mode == 5 || Mode == 7)
      b = __half2float(reinterpret_cast<const __half*>(ptr)[1]);
  }
  if constexpr (Mode == 3) a = a * a;
  if constexpr (Mode == 4) a = (a * a) * a;
}

template<typename T, int Group, int Mode>
__global__ void pack(const T* in, char* stage, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;  // n and block size are whole groups.
  int lane = i % Group;
  float x = float(in[i]), maximum = fabsf(x);
  float minimum = maximum > 0 ? maximum : INFINITY;
  for (int d = Group / 2; d; d /= 2) {
    maximum = fmaxf(maximum, __shfl_xor(maximum, d, Group));
    if constexpr (Mode >= 5)
      minimum = fminf(minimum, __shfl_xor(minimum, d, Group));
  }
  char* meta = stage + n + (i / Group) * meta_bytes<Mode>();
  float a = 0, b = 0;
  if (lane == 0) {
    if constexpr (Mode >= 5) {
      // sign(q)*2^(a + (abs(q)-1)*b), q=0 is exact zero.
      a = maximum == 0 ? 0 : floorf(log2f(minimum));
      b = maximum == 0 ? 1 : fmaxf((log2f(maximum) - a) / 126, 1e-4f);
    } else {
      a = maximum == 0 ? 1 : maximum / 127;
      if constexpr (Mode == 3) a = sqrtf(a);
      if constexpr (Mode == 4) a = cbrtf(a);
    }
    store_meta<Mode>(meta, a, b);
    // Quantize using the rounded, actually transmitted scale/parameters.
    load_meta<Mode>(meta, a, b);
  }
  a = __shfl(a, 0, Group);
  b = __shfl(b, 0, Group);
  int code = 0;
  if constexpr (Mode >= 5) {
    if (x != 0) {
      int k = int(nearbyintf((log2f(fabsf(x)) - a) / b));
      code = min(126, max(0, k)) + 1;
      if (x < 0) code = -code;
    }
  } else {
    // Invalid FP16/BF16 scales stay detectable (decoded NaN), no fallback.
    if (a > 0 && isfinite(a))
      code = min(127, max(-127, int(nearbyintf(x / a))));
  }
  reinterpret_cast<int8_t*>(stage)[i] = int8_t(code);
}

template<int Mode>
__device__ __forceinline__ float decode(int q, float a, float b) {
  if constexpr (Mode >= 5) {
    float value = q == 0 ? 0 : exp2f(a + (abs(q) - 1) * b);
    if constexpr (Mode == 7) value = fminf(value, 0x1.fep127f);
    return copysignf(value, float(q));
  }
  return a > 0 && isfinite(a) ? q * a : NAN;
}

template<typename T, int Group, int Mode, bool TwoStage = false>
__global__ void reduce(Peers p, T* out, int n, int rank) {
  barrier(p, rank, false);
  int part = TwoStage ? n / 4 : n;
  int start = TwoStage ? rank * part : 0;
  // Sixteen codes per thread: ld.128 peer reads, one scale per GS16.
  // Compare to native CAR's ld.128 BF16 reads, not a scalar-byte strawman.
  for (int local = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
       local < part; local += gridDim.x * blockDim.x * 16) {
    int i = start + local;
    float sums[16] = {};
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const char* stage = static_cast<const char*>(p.data[r]);
      float a = 0, b = 0;
      if (threadIdx.x % (Group / 16) == 0)
        load_meta<Mode>(stage + n + (i / Group) * meta_bytes<Mode>(), a, b);
      if constexpr (Group == 32) {
        a = __shfl(a, 0, 2);
        b = __shfl(b, 0, 2);
      }
      uint4 codes = *reinterpret_cast<const uint4*>(stage + i);
      uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
      #pragma unroll
      for (int j = 0; j < 16; ++j)
        sums[j] += decode<Mode>(int8_t(words[j / 4] >> (8 * (j % 4))), a, b);
    }
    struct alignas(16) Output { T values[16]; } packed;
    #pragma unroll
    for (int j = 0; j < 16; ++j) packed.values[j] = T(sums[j]);
    if constexpr (TwoStage) {
      T* tmp = reinterpret_cast<T*>(p.signals[rank] + 1);
      *reinterpret_cast<Output*>(tmp + local) = packed;
    } else {
      *reinterpret_cast<Output*>(out + i) = packed;
    }
  }
  barrier(p, rank, true);
  if constexpr (TwoStage) {
    // Keep the producing and consuming thread/block identical, as in native
    // CAR. Gather is uncompressed T: isolate input quantization, avoid a
    // second independent quantization of the already cancellation-sensitive sum.
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
    barrier(p, rank, true);
  }
}

template<int Group, int Mode>
__global__ void unpack(const char* stage, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a, b;
  load_meta<Mode>(stage + n + (i / Group) * meta_bytes<Mode>(), a, b);
  out[i] = decode<Mode>(reinterpret_cast<const int8_t*>(stage)[i], a, b);
}

template<typename T, int Group, int Mode>
void launch(int phase, Peers p, const void* in, void* out, int n, int rank,
            hipStream_t stream) {
  if (phase == 0 || phase == 1 || phase == 4)
    pack<T, Group, Mode><<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<const T*>(in), static_cast<char*>(p.data[rank]), n);
  if (phase == 0 || phase == 2)
    reduce<T, Group, Mode><<<min(16, (n + 4095) / 4096), 256, 0, stream>>>(
        p, static_cast<T*>(out), n, rank);
  if (phase == 4 || phase == 5)
    reduce<T, Group, Mode, true><<<min(16, (n / 4 + 4095) / 4096), 256, 0, stream>>>(
        p, static_cast<T*>(out), n, rank);
  if (phase == 3)
    unpack<Group, Mode><<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<const char*>(p.data[rank]), static_cast<float*>(out), n);
}

template<typename T, int Group>
void dispatch(int mode, int phase, Peers p, const void* in, void* out,
              int n, int rank, hipStream_t stream) {
  #define CASE(M) case M: launch<T, Group, M>(phase,p,in,out,n,rank,stream); break
  switch (mode) {
    CASE(0); CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7);
  }
  #undef CASE
}
}  // namespace

extern "C" int car_quant(int phase, int mode, int group, int fp16,
                         void* input, void* output, int n, int rank,
                         void** buffers, void** signals, void* stream) {
  if (n <= 0 || n % 256 || rank < 0 || rank >= 4 || mode < 0 || mode > 7 ||
      (group != 16 && group != 32) || phase < 0 || phase > 5) return -1;
  Peers peers;
  for (int r = 0; r < 4; ++r) {
    peers.data[r] = buffers[r];
    peers.signals[r] = static_cast<Signal*>(signals[r]);
  }
  auto st = static_cast<hipStream_t>(stream);
  #define DISPATCH(T,G) dispatch<T,G>(mode,phase,peers,input,output,n,rank,st)
  if (fp16) {
    if (group == 16) DISPATCH(__half,16); else DISPATCH(__half,32);
  } else {
    if (group == 16) DISPATCH(__hip_bfloat16,16);
    else DISPATCH(__hip_bfloat16,32);
  }
  #undef DISPATCH
  return int(hipGetLastError());
}
