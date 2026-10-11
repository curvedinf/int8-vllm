# SPDX-License-Identifier: Apache-2.0
"""Opt-in host lifecycle spans and idle-only GPU capture controls.

Set VLLM_LIFECYCLE_DIR and VLLM_LIFECYCLE_GATE before boot. The gate file
enables host recording only during a bounded test. Host spans do not read or
synchronize GPUs; explicit GPU capture boundaries synchronize idle workers.
Disabled decorators return the original function (no production wrapper).
Durations are HOST spans, not GPU durations. Use kernel timestamps for GPU work.
"""

import functools
import gc
import gzip
import inspect
import json
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_DIRECTORY = os.environ.get("VLLM_LIFECYCLE_DIR", "")
_GATE = os.environ.get("VLLM_LIFECYCLE_GATE", "")
_stream = None
_gpu_profiler = None
_gpu_tag = None
_gpu_clock = None
_gpu_base_tag = None
_gpu_part = 0
_gpu_steps = 0
_gpu_chunks = []
_gpu_ranks = {0}


def _clock_offset():
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return {"wall_minus_monotonic_ns": wall - (before + after) // 2,
            "uncertainty_ns": after - before}


def _start_gpu_chunk(tag):
    """Idle-only diagnostic RPC. Never enable this on an exposed server."""
    global _gpu_profiler, _gpu_tag, _gpu_clock
    if not _DIRECTORY or os.environ.get("VLLM_LIFECYCLE_KINETO") != "1":
        raise RuntimeError("Lifecycle GPU profiling is disabled")
    if _gpu_profiler is not None or not re.fullmatch(r"[a-zA-Z0-9_-]+", tag):
        raise ValueError("Invalid tag or profiler already active")
    import torch
    import torch.distributed as dist

    if dist.get_rank() not in _gpu_ranks:
        return {"pid": os.getpid(), "tag": tag, "skipped": True}

    # Explicit capture/chunk boundaries only. Ordinary CPU-only lifecycle spans
    # never synchronize; GPU capture intentionally adds diagnostic pauses.
    torch.cuda.synchronize()
    _gpu_clock = _clock_offset()
    _gpu_tag = tag
    _gpu_profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False, profile_memory=False,
    )
    _gpu_profiler.start()
    return {"pid": os.getpid(), "tag": tag, "clock": _gpu_clock}


def _stop_gpu_chunk():
    """Export on disk, never through torch's RAM-backed /tmp gzip staging."""
    global _gpu_profiler, _gpu_tag, _gpu_clock
    if _gpu_profiler is None:
        if _gpu_base_tag is not None:
            return {"pid": os.getpid(), "skipped": True}
        raise RuntimeError("No lifecycle GPU profile is active")
    import torch

    torch.cuda.synchronize()
    _gpu_profiler.stop()
    directory = Path(_DIRECTORY) / "gpu"
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"{_gpu_tag}-{os.getpid()}"
    raw_path = str(stem) + ".json"
    path = raw_path + ".gz"
    _gpu_profiler.export_chrome_trace(raw_path)
    # torch.profiler.profile.action_map contains bound methods pointing back
    # to the profile. Refcount alone does not release its native event buffers.
    # Explicit collection is essential for bounded long-running captures.
    _gpu_profiler = None
    gc.collect()
    # Return freed native event arenas to the OS on this Linux diagnostic path.
    import ctypes

    libc = ctypes.CDLL(None)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
    with open(raw_path, "rb") as source, gzip.open(path, "wb", compresslevel=1) as dest:
        shutil.copyfileobj(source, dest, length=1024 * 1024)
    Path(raw_path).unlink()
    Path(str(stem) + ".clock.json").write_text(json.dumps({
        "pid": os.getpid(), "start": _gpu_clock, "end": _clock_offset()}))
    _gpu_profiler = _gpu_tag = _gpu_clock = None
    return {"pid": os.getpid(), "path": path}


def start_gpu_profile(tag, ranks="0"):
    global _gpu_base_tag, _gpu_part, _gpu_steps, _gpu_chunks, _gpu_ranks
    if _gpu_base_tag is not None or not re.fullmatch(r"[a-zA-Z0-9_-]+", tag):
        raise ValueError("Invalid tag or profiler already active")
    _gpu_ranks = {int(rank) for rank in ranks.split(",")}
    if not _gpu_ranks or not _gpu_ranks <= {0, 1, 2, 3}:
        raise ValueError("Capture ranks must be a subset of 0,1,2,3")
    _gpu_base_tag, _gpu_part, _gpu_steps, _gpu_chunks = tag, 0, 0, []
    return _start_gpu_chunk(f"{tag}-part0000")


def stop_gpu_profile():
    global _gpu_base_tag
    _gpu_chunks.append(_stop_gpu_chunk())
    _gpu_base_tag = None
    return {"pid": os.getpid(), "chunks": list(_gpu_chunks)}


def _gpu_chunk_barrier():
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_tp_group

    dist.barrier(group=get_tp_group().cpu_group)


def _rotate_gpu_profile(name):
    """All TP workers reach this boundary after the same completed step.

    Diagnostic-only: synchronization and export pauses perturb latency. The
    separate unprofiled run is the timing baseline. No serving kernels execute
    between chunks; the next execute_model call begins after capture restarts.
    """
    global _gpu_part, _gpu_steps
    if name != "worker.sample_tokens" or _gpu_base_tag is None:
        return
    _gpu_steps += 1
    limit = int(os.environ.get("VLLM_LIFECYCLE_GPU_CHUNK_STEPS", "64"))
    if limit > 0 and _gpu_steps >= limit:
        begin = time.monotonic_ns()
        _gpu_chunks.append(_stop_gpu_chunk())
        _gpu_part += 1
        _gpu_steps = 0
        _start_gpu_chunk(f"{_gpu_base_tag}-part{_gpu_part:04d}")
        # Do not launch a collective while another rank is still exporting.
        # The CPU-group barrier itself produces no GPU kernel work.
        _gpu_chunk_barrier()
        if _stream is not None:
            _stream.write(json.dumps({"name": "profiler.rotate", "pid": os.getpid(),
                "tid": threading.get_native_id(), "start_ns": begin,
                "end_ns": time.monotonic_ns()}) + "\n")


def _enabled():
    return bool(_GATE and os.path.isfile(_GATE))


def _metadata(name, args, result=None):
    data = {}
    if len(args) > 1 and name in (
        "worker.execute_model", "scheduler.update_from_output",
        "scheduler.add_request", "frontend.render_chat_request", "cache.lookup",
    ):
        arg = args[1]
        req_id = getattr(arg, "request_id", None)
        if req_id is not None:
            data["request_id"] = req_id
        scheduled = getattr(arg, "num_scheduled_tokens", None)
        if isinstance(scheduled, dict):
            data["scheduled"] = scheduled
    if name == "cache.lookup" and result is not None:
        data["matched_tokens"], data["load_async"] = result
    if name in ("cache.fs_store", "cache.fs_load") and len(args) >= 4:
        data["blocks"] = len(args[0])
        data["bytes"] = len(args[0]) * args[3]
    if name == "scheduler.schedule" and result is not None:
        scheduler = args[0]
        data["scheduled"] = result.num_scheduled_tokens
        data["ready_waiting"] = len(scheduler.waiting)
        data["deferred_waiting"] = len(scheduler.skipped_waiting)
        data["waiting"] = len(scheduler.waiting) + len(scheduler.skipped_waiting)
        data["running"] = len(scheduler.running)
        data["waiting_states"] = {
            req.request_id: req.status.name
            for req in (*scheduler.waiting, *scheduler.skipped_waiting)}
        data["requests"] = {
            key: {"prompt": req.num_prompt_tokens,
                  "computed_after_schedule": req.num_computed_tokens,
                  "output": req.num_output_tokens,
                  "cache": vars(req.prefill_stats) if req.prefill_stats else None}
            for key in result.num_scheduled_tokens
            if (req := scheduler.requests.get(key)) is not None}
    return data


@contextmanager
def _span(name, args):
    global _stream
    if not _enabled():
        yield None
        return
    if _stream is None:
        Path(_DIRECTORY).mkdir(parents=True, exist_ok=True)
        _stream = open(Path(_DIRECTORY) / f"{os.getpid()}.jsonl", "a", buffering=1)
    data = {"name": name, "pid": os.getpid(), "tid": threading.get_native_id(),
            "start_ns": time.monotonic_ns(), **_metadata(name, args)}
    try:
        yield data
    finally:
        data["end_ns"] = time.monotonic_ns()
        _stream.write(json.dumps(data) + "\n")
        _rotate_gpu_profile(name)


def lifecycle(name):
    def decorate(fn):
        if not _DIRECTORY:
            return fn
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapped_async(*args, **kwargs):
                with _span(name, args):
                    return await fn(*args, **kwargs)
            return wrapped_async

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with _span(name, args) as data:
                result = fn(*args, **kwargs)
                if data is not None:
                    data.update(_metadata(name, args, result))
                return result
        return wrapped
    return decorate
