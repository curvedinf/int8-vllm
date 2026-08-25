# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 graph pass: fuse RMSNorm + aiter per-token int8 quant.

The W8A8 CK stack quantizes every post-norm activation through the
opaque custom op ``vllm.rocm_aiter_pertoken_quant_int8``. Under
torch.compile the norm lowers to ``vllm_ir.rms_norm`` /
``vllm_ir.fused_add_rms_norm`` and the quant stays a separate node, so
each norm pays a full activation round-trip plus the eager quant
kernels. This pass rewrites

    rms_norm -> (to fp16) -> pertoken_quant_int8
    fused_add_rms_norm[0] -> (to fp16) -> pertoken_quant_int8

into the single fused custom op ``vllm.rocm_rms_norm_int8_quant`` (or
the ``_add_`` variant, which also returns the new residual), gated on
VLLM_GFX908_FUSED_NORM_QUANT.

The rewrite is a targeted graph walk instead of a traced pattern so the
fp16 model (no convert between norm and quant), bf16 models (an
aten._to_copy), and 2D-flatten view/reshape/clone boilerplate all match
uniformly. A norm whose 16-bit output feeds anything besides the quant
chain is left untouched (the fused op does not materialize it).
"""

import operator

import torch
from torch import fx

import vllm.ir.ops  # noqa: F401 (registers vllm_ir op namespace)
from vllm import envs
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.logger import init_logger
from vllm.model_executor.layers.rms_norm_int8_quant import (  # noqa: F401 (op register)
    rms_norm_int8_quant,
)

logger = init_logger(__name__)

_QUANT_OP = torch.ops.vllm.rocm_aiter_pertoken_quant_int8.default
_RMS_NORM_OP = torch.ops.vllm_ir.rms_norm.default
_FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
_FUSED_OP = torch.ops.vllm.rocm_rms_norm_int8_quant.default
_FUSED_ADD_OP = torch.ops.vllm.rocm_rms_norm_add_int8_quant.default

# Value-neutral plumbing the quant input may sit behind after the norm.
_UNWRAP_TARGETS = {
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.clone.default,
}


def _convert_dtype(node: fx.Node) -> torch.dtype | None:
    for arg in (*node.args[1:], *node.kwargs.values()):
        if isinstance(arg, torch.dtype):
            return arg
    return None


def _unwrap(node: fx.Node) -> tuple[fx.Node, list[fx.Node]] | None:
    """Walk from the quant input back to a norm call.

    Returns (norm_call, chain) where chain holds the nodes strictly
    between the norm-output accessor and the quant (chain[-1] consumes
    the norm output), or None if the path is anything else.
    """
    chain: list[fx.Node] = []
    cur = node
    while True:
        if cur.target == _RMS_NORM_OP:
            return cur, chain
        if cur.target in _UNWRAP_TARGETS and isinstance(cur.args[0], fx.Node):
            if _convert_dtype(cur) not in (None, torch.float16):
                return None
            chain.append(cur)
            cur = cur.args[0]
            continue
        if (
            cur.target == operator.getitem
            and cur.args[1] == 0
            and isinstance(cur.args[0], fx.Node)
            and cur.args[0].target == _FUSED_ADD_RMS_NORM_OP
        ):
            chain.append(cur)
            return cur.args[0], chain
        return None


class GFX908RMSNormInt8QuantFusionPass(VllmInductorPass):
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        matched = 0
        for node in list(graph.nodes):
            if node.target != _QUANT_OP:
                continue
            unwrapped = _unwrap(node.args[0])
            if unwrapped is None or not self._eligible(node, *unwrapped):
                continue
            self._rewrite(graph, node, *unwrapped)
            matched += 1
        self.matched_count = matched
        if matched:
            logger.info("%s fused %d norm+quant pairs", self.pass_name, matched)

    def _eligible(
        self, quant_node: fx.Node, norm_node: fx.Node, chain: list[fx.Node]
    ) -> bool:
        # ir op signatures: rms_norm(x, weight, eps, variance_size?),
        # fused_add_rms_norm(x, residual, weight, eps, variance_size?)
        if norm_node.target == _RMS_NORM_OP:
            x, weight = norm_node.args[0], norm_node.args[1]
            extra = norm_node.args[3:] if len(norm_node.args) > 3 else ()
            accessor = norm_node  # single-output op
            nxt = chain[-1] if chain else quant_node
        else:
            x, weight = norm_node.args[0], norm_node.args[2]
            extra = norm_node.args[4:] if len(norm_node.args) > 4 else ()
            accessor = chain[-1]  # getitem(norm, 0), guaranteed by _unwrap
            nxt = chain[-2] if len(chain) > 1 else quant_node
        if not isinstance(weight, fx.Node):
            return False  # weightless norms are not covered by the fused op
        if any(a is not None for a in extra):
            return False  # variance_size override unsupported
        val = x.meta.get("val", None)
        if not isinstance(val, torch.Tensor) or val.ndim != 2:
            return False
        if val.dtype not in (torch.float16, torch.bfloat16):
            return False
        if val.shape[-1] * val.dtype.itemsize > 65536:
            return False  # single-block kernel limit
        # The (q, scale) tuple must be consumed only through getitem.
        if not all(
            u.target == operator.getitem and u.args[1] in (0, 1)
            for u in quant_node.users
        ):
            return False
        # The norm's 16-bit output must feed nothing but this chain: the
        # fused op does not materialize it, so any other user would keep
        # the standalone norm alive and duplicate the launch.
        return all(user is nxt for user in accessor.users)

    def _rewrite(
        self, graph: fx.Graph, quant_node: fx.Node, norm_node: fx.Node, chain: list
    ) -> None:
        with graph.inserting_before(quant_node):
            if norm_node.target == _RMS_NORM_OP:
                x, weight, epsilon = norm_node.args[:3]
                fused = graph.call_function(_FUSED_OP, (x, weight, epsilon))
                res_out = None
            else:
                x, residual, weight, epsilon = norm_node.args[:4]
                fused = graph.call_function(
                    _FUSED_ADD_OP, (x, residual, weight, epsilon)
                )
                res_out = graph.call_function(operator.getitem, (fused, 2))
            q_out = graph.call_function(operator.getitem, (fused, 0))
            s_out = graph.call_function(operator.getitem, (fused, 1))

        for user in list(quant_node.users):
            if user.args[1] == 0:
                user.replace_all_uses_with(q_out)
            else:
                user.replace_all_uses_with(s_out)
            graph.erase_node(user)
        graph.erase_node(quant_node)

        if res_out is not None:
            # residual_out users move to the fused op's output
            for user in list(norm_node.users):
                if user.target == operator.getitem and user.args[1] == 1:
                    user.replace_all_uses_with(res_out)
                    graph.erase_node(user)
        # dead norm/chain nodes are removed by the post-cleanup DCE

    def uuid(self) -> str:
        return self.hash_source(self, envs.VLLM_GFX908_FUSED_NORM_QUANT)
