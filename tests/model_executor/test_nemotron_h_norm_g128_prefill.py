# SPDX-License-Identifier: Apache-2.0
"""Native norm payload and real opaque prefill MoE protocol checks.

Routed expert math is isolated as identity; norm and all dense kernels are real.
Full routed computation and TP collectives are exercised by the serving gate.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.runner import moe_runner as runner_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.modelopt_gfx908 import (
    PackedG128LinearMethod,
    PackedG128MoEMethod,
)
from vllm.model_executor.layers.quantization.utils import packed_g128_int8 as ops
from vllm.model_executor.models import nemotron_h_prefill as prefill
from vllm.model_executor.models.nemotron_h import (
    NemotronHMLP,
    NemotronHMoE,
    NemotronHMoEDecoderLayer,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="gfx908 required"
)


class _Linear(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.quant_method = PackedG128LinearMethod("iq2_xxs")
        self.quant_method.k = k
        self.quant_method.start = 0
        self.bias = None
        self.tp_size = 1
        self.reduce_results = False
        self.register_buffer(
            "weight", torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        )
        self.register_buffer(
            "weight_gs128", torch.rand((1, n, k // 128), device="cuda") * 0.01 + 0.001
        )

    def forward(self, x):
        return self.quant_method.apply(self, x), None


class _Gate(nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 4), device=x.device), None


class _Router:
    def select_experts(self, hidden_states, **kwargs):
        shape = (hidden_states.shape[0], 1)
        return (
            torch.ones(shape, device=hidden_states.device),
            torch.zeros(shape, device=hidden_states.device, dtype=torch.int32),
        )


class _Routed(nn.Module):
    """Unchanged routed computation is isolated; dense projections use real kernels."""

    def __init__(self):
        super().__init__()
        self.quant_method = PackedG128MoEMethod(
            SimpleNamespace(
                activation=MoEActivation.RELU2_NO_MUL,
                moe_parallel_config=SimpleNamespace(ep_size=1),
            )
        )

    def _ensure_moe_quant_config_init(self):
        pass

    def forward_modular(self, x, **kwargs):
        return x.clone()


def _blank(cls):
    obj = object.__new__(cls)
    nn.Module.__init__(obj)
    return obj


def _model(monkeypatch):
    from vllm.platforms.rocm import on_gfx908

    if not on_gfx908():
        pytest.skip("requires gfx908")
    torch.manual_seed(629)
    hidden, latent_size, intermediate = 4096, 1024, 256
    shared = _blank(NemotronHMLP)
    shared.packed_relu2 = True
    shared.up_proj = _Linear(intermediate, hidden)
    shared.down_proj = _Linear(hidden, intermediate)
    shared.act_fn = lambda x: x.relu().square()
    first, last = _Linear(latent_size, hidden), _Linear(hidden, latent_size)
    parallel = SimpleNamespace(
        enable_eplb=False, all2all_backend="naive", use_fi_nvl_two_sided_kernels=False
    )
    cfg = SimpleNamespace(
        hidden_dim=latent_size,
        hidden_dim_unpadded=latent_size,
        tp_size=4,
        ep_size=1,
        dp_size=1,
        pcp_size=1,
        sp_size=1,
        is_sequence_parallel=False,
        is_lora_enabled=False,
        skip_final_all_reduce=False,
        moe_parallel_config=parallel,
    )
    wrapper = _blank(SharedExperts)
    wrapper.enable_dbo = False
    wrapper._output = [None, None]
    wrapper._layer = shared
    wrapper._moe_config = cfg
    wrapper._mk_can_overlap_shared_experts = lambda: False
    wrapper._stream = None
    runner = _blank(MoERunner)
    runner.moe_config = cfg
    runner.layer_name = "test.quant_reuse"
    runner.routed_input_transform = first
    runner.routed_output_transform = last
    runner.routed_scaling_factor = 1.0
    runner.gate = None
    runner.shared_expert_gate = None
    runner.routed_experts = _Routed()
    runner.router = _Router()
    runner.enable_dbo = False
    runner._shared_experts = wrapper
    runner._forward_entry = torch.ops.vllm.moe_forward_shared
    runner._sequence_parallel_context = lambda: nullcontext()
    model = _blank(NemotronHMoE)
    model.use_latent_moe = True
    model.is_sequence_parallel = False
    model.enable_eplb = False
    model.fc1_latent_proj = first
    model.fc2_latent_proj = last
    model.shared_experts = shared
    model.experts = runner
    model.gate = _Gate()
    monkeypatch.setattr(runner_module, "get_layer_from_name", lambda name: runner)
    norm = _blank(RMSNorm)
    norm.hidden_size = hidden
    norm.variance_epsilon = 1e-5
    norm.variance_size_override = None
    norm.has_weight = norm.pass_weight = norm.pass_weight_add = True
    norm.weight = nn.Parameter(torch.ones(hidden, device="cuda", dtype=torch.bfloat16))
    norm._forward_method = norm.forward_hip
    decoder = _blank(NemotronHMoEDecoderLayer)
    decoder.norm, decoder.mixer = norm, model
    return decoder


def bits(a):
    dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
    return a.contiguous().view(dtype)


@pytest.mark.parametrize("m", [256, 257, 4096, 4160])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rdtype", [None, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_real_norm_dense_and_opaque_protocol(monkeypatch, m, dtype, rdtype):
    decoder = _model(monkeypatch)
    decoder.mixer.experts.routed_scaling_factor = 1.25
    x = torch.randn((m, 4096), device="cuda", dtype=dtype)
    r = torch.randn_like(x, dtype=rdtype) if rdtype is not None else None
    saved = x.clone()
    r_saved = r.clone() if r is not None else None
    reductions, quant_calls = [], []
    original_quant = ops.quantize

    def reduce(value):
        reductions.append((value.shape, value.dtype))
        return value.clone()

    def quant(value):
        quant_calls.append(tuple(value.shape))
        return original_quant(value)

    monkeypatch.setattr(runner_module, "tensor_model_parallel_all_reduce", reduce)
    monkeypatch.setattr(ops, "quantize", quant)
    assert prefill.can_use_prefill(decoder, x, r)
    with monkeypatch.context() as original:
        original.setattr(prefill, "can_use_prefill", lambda *args: False)
        expected, expected_r = decoder(x, r)
    old_reductions = list(reductions)
    assert len(quant_calls) == 3
    reductions.clear()
    quant_calls.clear()
    actual, actual_r = decoder(x, r)
    assert torch.equal(bits(actual), bits(expected))
    assert torch.equal(bits(actual_r), bits(expected_r))
    assert (
        reductions
        == old_reductions
        == [(torch.Size([m, 1024]), dtype), (torch.Size([m, 4096]), dtype)]
    )
    assert quant_calls == [(m, 1024)]
    assert torch.equal(bits(x), bits(saved))
    if r is not None:
        assert torch.equal(bits(r), bits(r_saved))
    assert decoder.mixer.experts.shared_experts._output == [None, None]


@pytest.mark.parametrize("m", [8, 64, 255])
def test_native_decoder_does_not_import_or_call_prefill(monkeypatch, m):
    decoder = _blank(NemotronHMoEDecoderLayer)
    decoder.norm = nn.Identity()
    decoder.mixer = nn.Identity()
    x = torch.ones((m, 16))

    def forbidden(*args):
        pytest.fail("Native decode entered prefill protocol")

    monkeypatch.setattr(prefill, "try_prefill", forbidden)
    out, residual = decoder(x, None)
    assert out is x and residual is x


@pytest.mark.parametrize(
    "condition",
    [
        "sp",
        "dp",
        "ep",
        "pcp",
        "tp",
        "eplb",
        "dbo",
        "lora",
        "bias",
        "norm_hook",
        "latent_hook",
        "shared_hook",
        "gate_hook",
        "runner_hook",
        "legacy_quant",
        "native_norm",
        "variance_override",
        "pending_shared",
        "alternate_transform",
        "fp16",
        "strided",
        "small",
    ],
)
@torch.inference_mode()
def test_unsupported_modes_keep_original_path(monkeypatch, condition):
    decoder = _model(monkeypatch)
    x = torch.randn((256, 4096), device="cuda", dtype=torch.bfloat16)
    mixer = decoder.mixer
    runner = mixer.experts
    cfg = runner.moe_config
    if condition in ("sp", "dp", "ep", "pcp", "tp"):
        if condition == "sp":
            cfg.is_sequence_parallel = True
        elif condition == "tp":
            cfg.tp_size = 2
        else:
            setattr(cfg, condition + "_size", 2)
    elif condition == "eplb":
        mixer.enable_eplb = True
    elif condition == "dbo":
        runner.enable_dbo = True
    elif condition == "lora":
        cfg.is_lora_enabled = True
    elif condition == "bias":
        mixer.fc1_latent_proj.bias = torch.ones(1024, device="cuda")
    elif condition.endswith("_hook"):
        module = {
            "norm_hook": decoder.norm,
            "latent_hook": mixer.fc1_latent_proj,
            "shared_hook": mixer.shared_experts,
            "gate_hook": mixer.gate,
            "runner_hook": runner,
        }[condition]
        module.register_forward_hook(lambda *args: None)
    elif condition == "legacy_quant":
        monkeypatch.setattr(prefill.envs, "VLLM_GFX908_FUSED_NORM_QUANT", True)
    elif condition == "native_norm":
        decoder.norm._forward_method = decoder.norm.forward_native
    elif condition == "variance_override":
        decoder.norm.variance_size_override = 2048
    elif condition == "pending_shared":
        runner._shared_experts._output[0] = x
    elif condition == "alternate_transform":
        runner.routed_input_transform = nn.Identity()
    elif condition == "fp16":
        x = x.to(torch.float16)
    elif condition == "strided":
        x = x.T.contiguous().T
    elif condition == "small":
        x = x[:255]
    assert not prefill.can_use_prefill(decoder, x, None)
    assert prefill.try_prefill(decoder, x, None) is None


@torch.inference_mode()
def test_prefill_graph_replay(monkeypatch):
    decoder = _model(monkeypatch)
    monkeypatch.setattr(
        runner_module, "tensor_model_parallel_all_reduce", lambda value: value.clone()
    )
    x = torch.randn((256, 4096), device="cuda", dtype=torch.bfloat16)
    r = torch.randn((256, 4096), device="cuda", dtype=torch.float32)
    decoder(x, r)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_r = decoder(x, r)
    x.mul_(0.5)
    graph.replay()
    with monkeypatch.context() as original:
        original.setattr(prefill, "can_use_prefill", lambda *args: False)
        expected, expected_r = decoder(x, r)
    assert torch.equal(bits(actual), bits(expected))
    assert torch.equal(bits(actual_r), bits(expected_r))


def test_fake_operator_is_a_pure_shape_function(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    decoder = _model(monkeypatch)
    layer_name = decoder.mixer.experts._encode_layer_name()
    with FakeTensorMode():
        x = torch.empty((256, 4096), device="cuda", dtype=torch.bfloat16)
        q = torch.empty_like(x, dtype=torch.int8)
        s = torch.empty((256, 32), device="cuda", dtype=torch.float32)
        logits = torch.empty((256, 4), device="cuda")
        shared, routed = torch.ops.vllm.gfx908_moe_prefill_quant(
            x, q, s, logits, layer_name, 1024
        )
        assert shared.shape == (256, 4096) and routed.shape == (256, 1024)
        assert shared.dtype == routed.dtype == x.dtype
