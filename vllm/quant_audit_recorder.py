"""gfx908 quant-audit boundary recorder (record-once / replay-offline).

Env VLLM_QUANT_AUDIT=<dir> enables capture. Zero overhead when unset.

Hot-path safety: captures copy into a pre-allocated pinned-CPU ring with
non-blocking copies and return immediately — no disk I/O, no sync, no
collective stalls (the synchronous torch.save design deadlocked TP workers
mid-decode twice). A background flusher drains the ring to disk.

Arming: workers start disarmed; the API server touches <dir>/ARMED on the
first real request. Boot/profile/warmup runs record nothing.

Ring: per-process, bounded. When full, later captures are dropped (counted).
"""
import json
import os
import queue
import threading

import torch

_LOCK = threading.Lock()
_STATE = {
    "dir": None,
    "index": [],
    "counts": {},
    "max_per_key": int(os.environ.get("VLLM_QUANT_AUDIT_MAX", "3")),
    "armed": False,
    "arm_file": None,
    "queue": None,
    "flusher": None,
    "dropped": 0,
}

# Pinned staging buffers reused per capture (bounded count, freed on flush).
_POOL = {}
_POOL_LIMIT = 64


def init():
    d = os.environ.get("VLLM_QUANT_AUDIT")
    if d:
        os.makedirs(d, exist_ok=True)
        _STATE["dir"] = d
        _STATE["arm_file"] = os.path.join(d, "ARMED")
        _STATE["queue"] = queue.Queue(maxsize=4096)
        _STATE["flusher"] = threading.Thread(
            target=_flush_loop, daemon=True, name="qa-flush"
        )
        _STATE["flusher"].start()


def arm():
    """Enable capture (API-server side). Workers pick it up via the flag file."""
    d = os.environ.get("VLLM_QUANT_AUDIT")
    if d:
        os.makedirs(d, exist_ok=True)
        af = os.path.join(d, "ARMED")
        if not os.path.exists(af):
            with open(af, "w") as f:
                f.write("1\n")


def _enabled() -> bool:
    if _STATE["dir"] is None:
        return False
    if not _STATE["armed"]:
        af = _STATE.get("arm_file")
        if af and os.path.exists(af):
            _STATE["armed"] = True
        else:
            return False
    return True


def _flush_loop():
    import time

    while True:
        item = _STATE["queue"].get()
        if item is None:
            break
        kind, key, n, name, cpu, meta = item
        try:
            d = _STATE["dir"]
            base = f"{kind}_{key}_{n}"
            fn = f"{base}__{name}.pt"
            torch.save(cpu, os.path.join(d, fn))
            with _LOCK:
                _POOL.pop(id(cpu), None)
                _STATE["index"].append(
                    {
                        "kind": kind,
                        "key": key,
                        "n": n,
                        "tensors": [{"file": fn, **meta}],
                    }
                )
        except Exception:
            pass


def _record(kind: str, key: str, tensors: dict):
    if not _enabled():
        return
    with _LOCK:
        n = _STATE["counts"].get((kind, key), 0)
        if n >= _STATE["max_per_key"]:
            return
        _STATE["counts"][(kind, key)] = n + 1
    import inspect

    frame = inspect.currentframe()
    q = _STATE["queue"]
    if q is None:
        return
    staged = []
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        if t.is_cuda:
            cpu = torch.empty(
                t.shape, dtype=torch.float16 if t.is_floating_point() else t.dtype, pin_memory=True
            )
            cpu.copy_(t.to(cpu.dtype) if t.is_floating_point() else t, non_blocking=True)
            # Hold a reference so the async copy lands before the flusher saves.
            staged.append((name, cpu, {"shape": list(t.shape), "dtype": str(cpu.dtype)}))
        else:
            staged.append((name, t, {"shape": list(t.shape), "dtype": str(t.dtype)}))
    for name, cpu, meta in staged:
        try:
            q.put_nowait((kind, key, n, name, cpu, meta))
        except queue.Full:
            with _LOCK:
                _STATE["dropped"] += 1
            return


def record_gemm(layer_name: str, x_2d: torch.Tensor, x_q, x_s, N: int, K: int):
    if not _enabled():
        return
    key = f"{layer_name}_N{N}_K{K}"
    _record("gemm", key, {"x": x_2d, "xq": x_q, "xs": x_s})


def record_kv(layer_name: str, k: torch.Tensor, v: torch.Tensor, k_q, v_q, k_s, v_s):
    if not _enabled():
        return
    _record("kv", layer_name, {"k": k, "v": v, "ks": k_s, "vs": v_s})


def record_gdn_state(layer_name: str, h: torch.Tensor, step: int):
    if not _enabled():
        return
    with _LOCK:
        n = _STATE["counts"].get(("gdn", layer_name), 0)
        if n >= 16:
            return
        if n > 0 and step % 64 != 0:
            return
        _STATE["counts"][("gdn", layer_name)] = n + 1
    _record("gdn", f"{layer_name}_s{step}", {"h": h})


def record_ar(group_size: int, numel: int, partial: torch.Tensor, out: torch.Tensor):
    if not _enabled():
        return
    key = f"g{group_size}_n{numel}"
    _record("ar", key, {"partial": partial, "out": out})


def record_draft(hidden_states: torch.Tensor, draft_tokens: torch.Tensor, step: int):
    if not _enabled():
        return
    _record("draft", f"s{step}", {"hs": hidden_states, "draft": draft_tokens})


def flush():
    if not _enabled():
        return
    q = _STATE["queue"]
    if q is not None:
        import time

        deadline = time.time() + 60
        while not q.empty() and time.time() < deadline:
            time.sleep(0.2)
    with _LOCK:
        with open(os.path.join(_STATE["dir"], "index.json"), "w") as f:
            json.dump(
                {"entries": _STATE["index"], "dropped": _STATE["dropped"]}, f, indent=1
            )
