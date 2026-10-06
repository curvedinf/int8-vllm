# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-plan recorder for per-rank weight byte ranges.

Runs the *real* ``model.load_weights`` consumer chain against recorder
tensors whose ``copy_`` is intercepted (no-op) and whose consumed views'
geometry is logged. Every consumer slice op used by the fork's parameter
loaders (``narrow`` / ``slice`` / ``view`` / ... — see parameter.py and
linear.py) produces storage-sharing views, so the byte footprint of each
copied view maps directly onto the checkpoint tensor.

Two safety invariants make recording unable to silently miss bytes:
1. Any op outside the view allowlist (compute, ``contiguous``, indexing,
   ...) taints the weight -> full read.
2. An allowlisted op that returns storage *not* shared with its inputs
   (``reshape`` copying a non-contiguous view, ``.to`` moving devices)
   also taints -> full read.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch

_VIEW_OPS = {
    "narrow",
    "slice",
    "select",
    "view",
    "reshape",
    "detach",
    "t",
    "transpose",
    "expand",
    "alias",
    "permute",
    "unsqueeze",
    "squeeze",
    "as_strided",
    "split",
    "split_with_sizes",
    "unbind",
    "chunk",
    "flatten",
    "unflatten",
    "to",
}

MAX_FRAGMENTS_PER_WEIGHT = 262144

# Attribute-style calls that carry no data and must never taint (the
# recorder itself calls these; they dispatch through __torch_function__).
_META_OPS = {
    "untyped_storage",
    "data_ptr",
    "storage",
    "storage_offset",
    "stride",
    "size",
    "shape",
    "numel",
    "element_size",
    "dim",
    "ndim",
    "is_contiguous",
    "is_cuda",
    "device",
    "dtype",
    "as_subclass",
    "type",
    # descriptor/attribute machinery — no data moves
    "__get__",
    "__getattr__",
    "__repr__",
    "__str__",
    "__deepcopy__",
    "__reduce__",
    "__format__",
}


@dataclasses.dataclass
class WeightPlan:
    """Planned reads for one checkpoint tensor (offsets are tensor-relative)."""

    ranges: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    full: bool = False
    copies: int = 0

    def mark_unsupported(self) -> None:
        self.full = True

    def total_bytes(self) -> int:
        return sum(n for _, n in self.ranges)


class ReadPlan:
    """Aggregate of WeightPlans keyed by (prefixed) weight name."""

    def __init__(self) -> None:
        self.weights: dict[str, WeightPlan] = {}
        self._by_storage: dict[int, WeightPlan] = {}

    def new_recorder(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        entry = WeightPlan()
        self.weights[name] = entry
        base = torch.empty(shape, dtype=dtype, device=device)
        rec = base.as_subclass(_RecorderTensor)
        rec._plan_entry = entry
        self._by_storage[base.untyped_storage().data_ptr()] = entry
        return rec

    def entry_for_storage(self, ptr: int) -> WeightPlan | None:
        return self._by_storage.get(ptr)

    def get(self, name: str) -> WeightPlan | None:
        return self.weights.get(name)

    def summary(self) -> str:
        n_full = sum(1 for w in self.weights.values() if w.full)
        planned = sum(w.total_bytes() for w in self.weights.values() if not w.full)
        frags = sum(len(w.ranges) for w in self.weights.values() if not w.full)
        return (
            f"read plan: {len(self.weights)} weights, {n_full} full-fallback, "
            f"{planned / (1 << 20):.0f} MiB in {frags} ranges"
        )


class _RecorderTensor(torch.Tensor):
    """Tensor subclass that records what the weight-loader chain copies.

    Uses ``__torch_function__`` (not ``__torch_dispatch__``): the
    torch-function machinery preserves the subclass for storage-sharing
    view ops and returns plain tensors for compute ops, which maps exactly
    onto the record/taint split we need.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        args = args or ()
        name = getattr(func, "__name__", str(func))
        plan = _ACTIVE_PLAN[0]
        if name not in _META_OPS and name not in _VIEW_OPS and name != "copy_":
            import os

            if os.getenv("FSGPU_RECORDER_DEBUG"):
                print(
                    f"[recorder] non-view op: {name} "
                    f"args={[type(a).__name__ for a in args if isinstance(a, torch.Tensor)]}"
                )

        if name == "copy_":
            # t.copy_(src, ...) -> args = (t, src, ...)
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if plan is not None and isinstance(src, torch.Tensor):
                entry = _entry_for(plan, src)
                if entry is not None:
                    _record_view(entry, src)
            return args[0]  # no-op: recorder bytes never reach the params

        if name in _META_OPS:
            return super().__torch_function__(func, types, args, kwargs)

        rec_inputs = [
            t
            for t in _iter_tensors(args) + _iter_tensors(list(kwargs.values()))
            if isinstance(t, cls)
        ]

        out = super().__torch_function__(func, types, args, kwargs)

        if rec_inputs:
            if name not in _VIEW_OPS:
                for t in rec_inputs:
                    _entry_for(plan, t).mark_unsupported()
            else:
                # A view-family op must stay a recorder sharing one of its
                # inputs' storage; anything else (reshape-copy, .to)
                # taints.
                outs = [t for t in _iter_tensors(_as_seq(out))]
                out_ptrs = {t.untyped_storage().data_ptr() for t in outs}
                in_ptrs = {t.untyped_storage().data_ptr() for t in rec_inputs}
                if not outs or not any(isinstance(t, cls) for t in outs) or not (
                    out_ptrs & in_ptrs
                ):
                    for t in rec_inputs:
                        _entry_for(plan, t).mark_unsupported()
        return out


_ACTIVE_PLAN: list[ReadPlan | None] = [None]


def _as_seq(x) -> Iterable:
    if isinstance(x, (tuple, list)):
        return x
    return (x,)


def _iter_tensors(xs) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for x in xs:
        if isinstance(x, torch.Tensor):
            out.append(x)
        elif isinstance(x, (tuple, list)):
            out.extend(_iter_tensors(x))
    return out


def _entry_for(plan: ReadPlan | None, t: torch.Tensor) -> WeightPlan | None:
    if plan is None:
        return _NULL_ENTRY
    try:
        entry = plan.entry_for_storage(t.untyped_storage().data_ptr())
    except Exception:
        return _NULL_ENTRY
    if entry is None:  # tensor from a tainted/copied lineage
        return _NULL_ENTRY
    return entry


class _NullEntry:
    def mark_unsupported(self) -> None:
        pass


_NULL_ENTRY = _NullEntry()


def with_active_plan(plan: ReadPlan | None):
    class _Ctx:
        def __enter__(self):
            _ACTIVE_PLAN[0] = plan

        def __exit__(self, *exc):
            _ACTIVE_PLAN[0] = None

    return _Ctx()


def _record_view(entry: WeightPlan, src: torch.Tensor) -> None:
    if isinstance(entry, _NullEntry):
        return
    entry.copies += 1
    if entry.full:
        return
    sizes = tuple(src.shape)
    strides = tuple(src.stride())
    off = src.storage_offset()
    es = src.element_size()
    if len(sizes) == 0:
        entry.ranges.append((off * es, es))
        return
    if strides[-1] != 1:
        # e.g. a transposed view: the stride-1 dim may not be last. Rotate
        # it last — _flat_runs only needs *a* unit-stride inner dim, and the
        # (sizes, strides) multiset still covers exactly the same elements.
        try:
            d = strides.index(1)
        except ValueError:
            entry.mark_unsupported()
            return
        order = [i for i in range(len(sizes)) if i != d] + [d]
        sizes = tuple(sizes[i] for i in order)
        strides = tuple(strides[i] for i in order)
    runs = _flat_runs(0, off, sizes, strides)
    if runs is None or len(runs) > MAX_FRAGMENTS_PER_WEIGHT:
        entry.mark_unsupported()
        return
    for start, length in runs:
        if length <= 0:
            continue
        o, n = start * es, length * es
        if entry.ranges and entry.ranges[-1][0] + entry.ranges[-1][1] == o:
            entry.ranges[-1] = (entry.ranges[-1][0], entry.ranges[-1][1] + n)
        else:
            entry.ranges.append((o, n))


def _flat_runs(dim: int, offset: int, sizes: tuple, strides: tuple):
    """Decompose a strided view into contiguous element runs, or None."""
    if dim == len(sizes) - 1:
        return [(offset, sizes[dim])]
    inner = _flat_runs(dim + 1, offset, sizes, strides)
    if inner is None:
        return None
    if len(inner) == 1 and strides[dim] == inner[0][1]:
        # this dim concatenates with the single inner run
        return [(inner[0][0], inner[0][1] * sizes[dim])]
    out: list[tuple[int, int]] = []
    for i in range(sizes[dim]):
        delta = i * strides[dim]
        if i == 0:
            out.extend(inner)
        else:
            out.extend((s + delta, n) for s, n in inner)
    return out
