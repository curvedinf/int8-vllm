// Fork-delta op registrations for the int8-vllm fork (gfx908).
//
// Upstream v0.33.0 consolidated csrc/libtorch_stable/* into csrc/* and its
// torch_bindings.cpp registers the shared op surface. This file registers
// ONLY the fork-local ops not present upstream (plus the 3-arg
// gptq_shuffle overload used by the repacked-weight W8A16 path), so both
// binding files can be linked into _C_stable_libtorch via
// STABLE_TORCH_LIBRARY_FRAGMENT without duplicate-registration errors.
#include "ops.h"
#include "cuda_utils.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "gptq_w8a16_repacked_gemm(Tensor a, Tensor b_q_weight, "
      "Tensor b_gptq_qzeros, Tensor b_gptq_scales, bool use_v2_format) "
      "-> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("gptq_w8a16_repacked_gemm",
           TORCH_BOX(&gptq_w8a16_repacked_gemm));
}
