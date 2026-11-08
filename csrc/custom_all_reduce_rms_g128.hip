// SPDX-License-Identifier: Apache-2.0
// TP4 fused collective: AllReduce(partials) + residual add + RMSNorm +
// per-128-group int8 quantization, over the native custom-AR IPC buffers.
//
// Replaces the unfused chain  custom_ar(partials) -> add_residual ->
// rmsnorm -> group_quant  with one launch and one HBM round trip of the
// hidden state. Numerics mirror the unfused path: the AR sum is rounded to
// bf16, the residual is added in fp32 and rounded to bf16, the norm is
// computed from the bf16 residual in fp32 (fp32 mean), and the group quant
// uses amax/127 with round-half-even.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {
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
    __scoped_atomic_store_n(dst, epoch, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(src, __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < epoch) {}
  }
  __syncthreads();
  if (threadIdx.x == 0) self->epoch[blockIdx.x] = epoch;
}

// gfx908 wavefronts are 64 lanes wide.
__device__ __forceinline__ float block_sum(float v, float* smem, int tid) {
  #pragma unroll
  for (int d = 32; d; d /= 2) v += __shfl_xor(v, d);
  const int wave = tid / 64;
  if ((tid & 63) == 0) smem[wave] = v;
  __syncthreads();
  v = (tid < 4) ? smem[tid] : 0.0f;
  if (tid < 64) {
    #pragma unroll
    for (int d = 2; d; d /= 2) v += __shfl_xor(v, d);
  }
  return v;
}

// One block per row of the payload; rows are strided across blocks.
__global__ void ar_rms_g128(Peers p, const __hip_bfloat16* in,
                            const __hip_bfloat16* residual,
                            const __hip_bfloat16* w, __hip_bfloat16* normed,
                            __hip_bfloat16* res_out, int8_t* q_out,
                            float* s_out, int m, int k, float eps, int rank,
                            int row_stride) {
  __shared__ float smem[32];
  __shared__ float inv_s;
  const int tid = threadIdx.x;
  const int per = k / 256;  // elements per thread (k=2048 -> 8)
  // Publish this block's rows into its own IPC buffer BEFORE the barrier.
  // Rows are block-strided, so the per-block barrier protects exactly the
  // rows this block copies and every peer's matching block has copied: the
  // host-side whole-buffer memcpy could overwrite rows a lagging peer's
  // other blocks were still reading (measured at m=8).
  {
    __hip_bfloat16* dst = reinterpret_cast<__hip_bfloat16*>(p.data[rank]);
    for (int row = blockIdx.x; row < m; row += row_stride) {
      const int base = tid * per;
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i < per) {
          dst[row * k + base + i] = in[row * k + base + i];
        }
      }
    }
    __threadfence_system();
  }
  barrier(p, rank, false);
  for (int row = blockIdx.x; row < m; row += row_stride) {
    float acc[16];
    float resf[16];
    const int base = tid * per;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i < per) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          sum += __bfloat162float(
              reinterpret_cast<const __hip_bfloat16*>(p.data[r])[row * k +
                                                              base + i]);
        }
        const float sum_bf16 = __bfloat162float(__float2bfloat16(sum));
        resf[i] = sum_bf16 + __bfloat162float(residual[row * k + base + i]);
        const __hip_bfloat16 rb = __float2bfloat16(resf[i]);
        res_out[row * k + base + i] = rb;
        resf[i] = __bfloat162float(rb);
      } else {
        resf[i] = 0.0f;
      }
      acc[i] = resf[i] * resf[i];
    }
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) ss += acc[i];
    ss = block_sum(ss, smem, tid);
    if (tid == 0) inv_s = 1.0f / sqrtf(ss / float(k) + eps);
    __syncthreads();
    const float inv = inv_s;

    float nf[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i < per) {
        // Quantize the bf16-rounded normed values (the unfused chain stores
        // the norm and re-reads it for the quant).
        const __hip_bfloat16 nb =
            __float2bfloat16(resf[i] * inv * __bfloat162float(w[base + i]));
        normed[row * k + base + i] = nb;
        nf[i] = __bfloat162float(nb);
      } else {
        nf[i] = 0.0f;
      }
    }
    // Per-128-group quant: 16 threads per group (8 elements each).
    const int g = base / 128;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(nf[i]));
    #pragma unroll
    for (int d = 8; d >= 1; d /= 2) amax = fmaxf(amax, __shfl_xor(amax, d));
    const float s = fmaxf(amax, 1.0e-8f) / 127.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i < per) {
        float v = nearbyintf(nf[i] / s);
        v = fminf(fmaxf(v, -127.0f), 127.0f);
        q_out[row * k + base + i] = int8_t(v);
      }
    }
    if ((tid & 15) == 0) s_out[row * (k / 128) + g] = s;
    __syncthreads();
  }
}
}  // namespace

extern "C" int ar_rms_g128_zero(void* signal, void* stream) {
  return int(hipMemsetAsync(signal, 0, sizeof(Signal),
                            static_cast<hipStream_t>(stream)));
}

extern "C" int ar_rms_g128_launch(
    const void* in, const void* residual, const void* w, void* normed,
    void* res_out, void* q_out, void* s_out, int m, int k, float eps, int rank,
    void** buffers, void** signals, void* stream) {
  if (m <= 0 || k <= 0 || k % 128 || k % 256 || rank < 0 || rank >= 4) {
    return -1;
  }
  Peers p;
  for (int r = 0; r < 4; ++r) {
    p.data[r] = static_cast<char*>(buffers[r]);
    p.signals[r] = static_cast<Signal*>(signals[r]);
  }
  auto st = static_cast<hipStream_t>(stream);
  int blocks = m < kMaxBlocks ? m : kMaxBlocks;
  const char* b = std::getenv("VLLM_GFX908_G128_AR_BLOCKS");
  if (b) {
    int cap = std::atoi(b);
    if (cap > 0 && cap < blocks) blocks = cap;
  }
  ar_rms_g128<<<blocks, 256, 0, st>>>(
      p, static_cast<const __hip_bfloat16*>(in),
      static_cast<const __hip_bfloat16*>(residual),
      static_cast<const __hip_bfloat16*>(w), static_cast<__hip_bfloat16*>(normed),
      static_cast<__hip_bfloat16*>(res_out), static_cast<int8_t*>(q_out),
      static_cast<float*>(s_out), m, k, eps, rank, blocks);
  return int(hipGetLastError());
}
