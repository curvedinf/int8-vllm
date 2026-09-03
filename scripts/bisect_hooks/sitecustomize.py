"""Env-gated forward-capture injection for offline bisect experiments.

Loaded via PYTHONPATH in vLLM TP worker processes only when
BISECT_CAPTURE_DIR is set (see scripts/test_row0_layer_bisect.py, which
self-configures the env and re-execs). When the worker imports
vllm.v1.worker.gpu_worker, this patches Worker.load_model: after the model
loads on rank 0 it registers forward hooks on every decoder layer (+ the
layer-0 input embedding stream and the final RMSNorm) and writes one small
pickle per model forward into BISECT_CAPTURE_DIR:

    fwd_{pid}_{counter:06d}.pt = {
        "fwd": int, "n": int,            # forward index, rows in the batch
        "ids": [...], "pos": [...],      # head-16 input ids / positions
        "layers": {i: tensor[<=16, H]},  # only when n <= BISECT_CAPTURE_MAXN
        "embed": tensor, "norm": tensor,
    }

plus a one-time meta.json with the per-layer attention type. Nothing runs
unless BISECT_CAPTURE_DIR is set, so production boots are unaffected.
"""

import importlib.util
import os
import sys

_CAPTURE_DIR = os.environ.get("BISECT_CAPTURE_DIR")
_MAXN = int(os.environ.get("BISECT_CAPTURE_MAXN", "32"))
_MAXROWS = 16
_TARGET = "vllm.v1.worker.gpu_worker"


class _LoaderProxy:
    """Delegate everything to the real loader, wrap exec_module."""

    def __init__(self, loader, exec_module):
        self._loader = loader
        self._exec_module = exec_module

    def exec_module(self, module):
        self._exec_module(module)

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET or not _CAPTURE_DIR:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _patch(module)
            except Exception:
                import traceback

                traceback.print_exc()

        spec.loader = _LoaderProxy(spec.loader, exec_module)
        return spec


def _patch(mod):
    Worker = mod.Worker
    orig_load_model = Worker.load_model

    def load_model(self, *, load_dummy_weights: bool = False):
        orig_load_model(self, load_dummy_weights=load_dummy_weights)
        try:
            _install(self)
        except Exception:
            import traceback

            traceback.print_exc()

    Worker.load_model = load_model


def _install(worker):
    rank = int(getattr(worker, "rank", 0))
    model = worker.model_runner.get_model()
    lm = getattr(model, "language_model", None)
    inner = getattr(lm, "model", model)
    layers = list(inner.layers)
    if rank != 0:
        return

    import json

    import torch

    os.makedirs(_CAPTURE_DIR, exist_ok=True)

    layer_types = {}
    for i, layer in enumerate(layers):
        lt = getattr(layer, "layer_type", None)
        if lt is None:
            lt = "linear_attention" if hasattr(layer, "linear_attn") else "full_attention"
        layer_types[i] = lt
    with open(os.path.join(_CAPTURE_DIR, "meta.json"), "w") as f:
        json.dump({"num_layers": len(layers), "layer_types": layer_types}, f)

    state = {"fwd": 0, "layers": {}, "embed": None, "norm": None, "ids": None, "pos": None, "n": 0}

    def _rows(t):
        if isinstance(t, (tuple, list)):
            t = t[0] if t else None
        if t is None or not torch.is_tensor(t):
            return None
        return t

    def layer_hook(i):
        def h(module, args, output):
            t = _rows(output)
            if t is None or t.dim() != 2 or t.shape[0] > _MAXN:
                return
            state["layers"][i] = t[:_MAXROWS].detach().to(torch.float32).cpu()

        return h

    for i, layer in enumerate(layers):
        layer.register_forward_hook(layer_hook(i))

    def embed_pre_hook(module, args, kwargs):
        t = _rows(args)
        if t is None:
            t = kwargs.get("hidden_states")
        if t is None or not torch.is_tensor(t) or t.dim() != 2 or t.shape[0] > _MAXN:
            return
        state["embed"] = t[:_MAXROWS].detach().to(torch.float32).cpu()

    layers[0].register_forward_pre_hook(embed_pre_hook, with_kwargs=True)

    norm = getattr(inner, "norm", None)
    if norm is not None and hasattr(norm, "register_forward_hook"):

        def norm_hook(module, args, output):
            t = _rows(output)
            if t is None or t.dim() != 2 or t.shape[0] > _MAXN:
                return
            state["norm"] = t[:_MAXROWS].detach().to(torch.float32).cpu()

        norm.register_forward_hook(norm_hook)

    def head(t):
        if t is None:
            return None
        if t.dim() > 1:
            t = t[0]
        return t[:_MAXROWS].detach().to("cpu").tolist()

    def model_pre_hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and len(args) >= 1:
            ids = args[0]
        pos = kwargs.get("positions")
        if pos is None and len(args) >= 2:
            pos = args[1]
        state["ids"] = head(ids)
        state["pos"] = head(pos)
        t = ids if torch.is_tensor(ids) else pos
        state["n"] = int(t.shape[0]) if t is not None else 0

    def model_post_hook(module, args, output):
        nonlocal state
        n = state["n"]
        payload = {"fwd": state["fwd"], "n": n, "ids": state["ids"], "pos": state["pos"]}
        if n <= _MAXN and state["layers"]:
            payload["layers"] = dict(state["layers"])
            payload["embed"] = state["embed"]
            payload["norm"] = state["norm"]
        path = os.path.join(_CAPTURE_DIR, f"fwd_{os.getpid()}_{state['fwd']:06d}.pt")
        torch.save(payload, path)
        state = {
            "fwd": state["fwd"] + 1,
            "layers": {},
            "embed": None,
            "norm": None,
            "ids": None,
            "pos": None,
            "n": 0,
        }

    model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)
    model.register_forward_hook(model_post_hook)


if _CAPTURE_DIR:
    sys.meta_path.insert(0, _Finder())
