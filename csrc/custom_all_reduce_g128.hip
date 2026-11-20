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
// Fixed register operands keep the four peer pointers out of per-thread scratch.
template <typename T>
__device__ __forceinline__ T* select_peer(int i, T* a, T* b, T* c, T* d) {
  return i < 2 ? (i == 0 ? a : b) : (i == 2 ? c : d);
}

template <bool RegisterPeers>
__device__ __forceinline__ Signal* signal_at(Peers p, int index) {
  if (RegisterPeers)
    return select_peer(index, p.signals[0], p.signals[1],
                       p.signals[2], p.signals[3]);
  return p.signals[index];
}

template <bool RegisterPeers = false>
__device__ __forceinline__ void barrier(Peers p, int rank, bool end) {
  __syncthreads();
  Signal* self = signal_at<RegisterPeers>(p, rank);
  uint32_t epoch = self->epoch[blockIdx.x] + 1;
  if (threadIdx.x < 4) {
    int peer = threadIdx.x;
    Signal* remote = signal_at<RegisterPeers>(p, peer);
    auto* dst = end ? &remote->end[blockIdx.x][rank]
                    : &remote->start[blockIdx.x][rank];
    auto* src = end ? &self->end[blockIdx.x][peer]
                    : &self->start[blockIdx.x][peer];
    __scoped_atomic_store_n(dst, epoch, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    // A peer cannot reuse this signal until we enter the opposite barrier.
    // Equality also works when the 32-bit epoch wraps during long uptime.
    while (__scoped_atomic_load_n(src, __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) != epoch) {}
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
template <typename Word>
__device__ __forceinline__ Word load_codes(const int8_t* src) {
  if (reinterpret_cast<uintptr_t>(src) % alignof(Word) == 0)
    return *reinterpret_cast<const Word*>(src);
  Word value;
  __builtin_memcpy(&value, src, sizeof(value));
  return value;
}

template <typename Word>
__device__ __forceinline__ void store_codes(int8_t* dst, Word value) {
  if (reinterpret_cast<uintptr_t>(dst) % alignof(Word) == 0)
    *reinterpret_cast<Word*>(dst) = value;
  else
    __builtin_memcpy(dst, &value, sizeof(value));
}

// An owner reads all four contributions in rank order, requantizes once,
// and distributes the resulting codes and scale. The two IPC barriers
// protect exactly the groups staged and consumed by the corresponding CTA.
using CodeWord = unsigned __int128;
constexpr int kGroupSize = 128;
constexpr int kLanes = kGroupSize / sizeof(CodeWord);
constexpr int kThreads = 256;
constexpr int kGroupsPerBlock = kThreads / kLanes;
__global__ void reduce_scatter_gather(Peers p, const int8_t* codes,
                                      const float* scales, int8_t* out,
                                      float* out_scales, int n, int rank) {
  const int lane = threadIdx.x % kLanes;
  const int first = blockIdx.x * kGroupsPerBlock + threadIdx.x / kLanes;
  const int stride = gridDim.x * kGroupsPerBlock;
  const int groups = n / kGroupSize;
  char* local = select_peer(rank, p.data[0], p.data[1], p.data[2], p.data[3]);
  auto* send = reinterpret_cast<CodeWord*>(local);
  auto* send_scales = reinterpret_cast<float*>(local + n);
  for (int g = first; g < groups; g += stride) {
    int i = g * kLanes + lane;
    send[i] = load_codes<CodeWord>(codes + i * sizeof(CodeWord));
    if (lane == 0) send_scales[g] = scales[g];
  }
  __threadfence_system();
  barrier<true>(p, rank, false);

  const int first_owned = (rank + 4 - int(blockIdx.x % 4)) % 4;
  for (int g = first + first_owned * stride; g < groups; g += 4 * stride) {
    int i = g * kLanes + lane;
    float acc[sizeof(CodeWord)] = {};
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      CodeWord value = reinterpret_cast<const CodeWord*>(p.data[r])[i];
      float scale = reinterpret_cast<const float*>(p.data[r] + n)[g];
      #pragma unroll
      for (int j = 0; j < sizeof(CodeWord); ++j)
        acc[j] += float(static_cast<int8_t>(value >> (8 * j))) * scale;
    }
    float amax = 0.0f;
    int bad = 0;
    #pragma unroll
    for (int j = 0; j < sizeof(CodeWord); ++j) {
      amax = fmaxf(amax, fabsf(acc[j]));
      bad |= !isfinite(acc[j]);
    }
    for (int delta = kLanes / 2; delta; delta /= 2) {
      amax = fmaxf(amax, __shfl_xor(amax, delta, kLanes));
      bad |= __shfl_xor(bad, delta, kLanes);
    }
    float scale = bad ? NAN : fmaxf(amax / 127.0f, 1.0e-30f);
    CodeWord result = 0;
    #pragma unroll
    for (int j = 0; j < sizeof(CodeWord); ++j) {
      int8_t code = bad ? 0 : static_cast<int8_t>(nearbyintf(acc[j] / scale));
      result |= static_cast<CodeWord>(static_cast<uint8_t>(code)) << (8 * j);
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      reinterpret_cast<CodeWord*>(p.data[r])[i] = result;
      if (lane == 0) reinterpret_cast<float*>(p.data[r] + n)[g] = scale;
    }
  }
  __threadfence_system();
  barrier<true>(p, rank, true);
  for (int g = first; g < groups; g += stride) {
    int i = g * kLanes + lane;
    store_codes<CodeWord>(out + i * sizeof(CodeWord), send[i]);
    if (lane == 0) out_scales[g] = send_scales[g];
  }
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
  static const int env_blocks = [] {
    const char* b = std::getenv("VLLM_GFX908_G128_AR_BLOCKS");
    return b ? std::atoi(b) : 0;
  }();
  static const bool use_exchange = [] {
    const char* value = std::getenv("VLLM_GFX908_G128_REDUCE_SCATTER");
    return value && value[0] == '1' && value[1] == 0;
  }();
  if (use_exchange) {
    int limit = env_blocks > 0 && env_blocks < kMaxBlocks ? env_blocks : kMaxBlocks;
    int blocks = (n / kGroupSize + kGroupsPerBlock - 1) / kGroupsPerBlock;
    if (blocks > limit) blocks = limit;
    reduce_scatter_gather<<<blocks, kThreads, 0, st>>>(
        p, static_cast<const int8_t*>(q), static_cast<const float*>(s),
        static_cast<int8_t*>(out), static_cast<float*>(scales), n, rank);
    return int(hipGetLastError());
  }
  auto rc = hipMemcpyAsync(p.data[rank], q, n, hipMemcpyDeviceToDevice, st);
  if (rc != hipSuccess) return int(rc);
  rc = hipMemcpyAsync(p.data[rank] + n, s, n / 128 * sizeof(float),
                      hipMemcpyDeviceToDevice, st);
  if (rc != hipSuccess) return int(rc);
  int cap = env_blocks > 0 ? env_blocks : 16;
  cap = cap > kMaxBlocks ? kMaxBlocks : cap;
  int blocks = (n / 128 + 3) / 4;
  if (blocks > cap) blocks = cap;
  reduce<<<blocks, 256, 0, st>>>(
      p, static_cast<int8_t*>(out), static_cast<float*>(scales), n, rank);
  return int(hipGetLastError());
}
