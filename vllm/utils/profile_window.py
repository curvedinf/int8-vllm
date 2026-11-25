# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cProfile a fixed window of steps in a live process (diagnostic).

With ``VLLM_CP_PROFILE=<dir>`` set, each scope counts its own steps;
``VLLM_CP_PROFILE_START`` (default 300) skips warm-up and the next
``VLLM_CP_PROFILE_COUNT`` (default 40) steps are profiled; ``_<SCOPE>`` suffixes
(``VLLM_CP_PROFILE_START_ENGINE``) override either bound per scope. pstats sorted by
tottime and by cumulative time are then written to ``<dir>/<scope>.<pid>.txt``.
Without the variable every call is a no-op.
"""

import os

_DIR = os.environ.get("VLLM_CP_PROFILE")


def _bound(scope: str, name: str, default: int) -> int:
    """``VLLM_CP_PROFILE_<name>_<SCOPE>`` overrides ``VLLM_CP_PROFILE_<name>``."""
    return int(
        os.environ.get(
            f"VLLM_CP_PROFILE_{name}_{scope.upper()}",
            os.environ.get(f"VLLM_CP_PROFILE_{name}", str(default)),
        )
    )


def _start(scope: str) -> int:
    return _bound(scope, "START", 300)


def _count(scope: str) -> int:
    return _bound(scope, "COUNT", 40)


ENABLED = bool(_DIR)
_calls: dict[str, int] = {}
_profs: dict[str, object] = {}
_DONE = -(10**9)


def step(scope: str) -> object | None:
    """Count one step of `scope`; return the profile to enable, or None."""
    n = _calls[scope] = _calls.get(scope, 0) + 1
    prof = _profs.get(scope)
    if prof is None:
        if n != _start(scope):
            return None
        import cProfile

        prof = _profs[scope] = cProfile.Profile()
    if n >= _start(scope) + _count(scope):
        _dump(scope, prof)
        del _profs[scope]
        _calls[scope] = _DONE
        return None
    return prof


def _dump(scope: str, prof: object) -> None:
    import io
    import pstats

    try:
        out = io.StringIO()
        stats = pstats.Stats(prof, stream=out)
        stats.sort_stats("tottime").print_stats(70)
        stats.sort_stats("cumulative").print_stats(90)
        with open(f"{_DIR}/{scope}.{os.getpid()}.txt", "w") as f:
            f.write(out.getvalue())
    except Exception:  # diagnostic only: never fail the profiled call
        pass
