# SPDX-License-Identifier: Apache-2.0
"""Propagate launcher-only Bolt env flags into engine/worker processes.

The launcher exports fork-local switches that are not registered in
``vllm.envs`` (dense dispatch, page MMA, G128 adaptive spec length, AR
library/cap, ...). EngineCore and worker processes do not reliably inherit
those variables, so the launcher writes the current ``VLLM_*`` set to a
fixed shm file and both process entry points apply it before reading any
flag.
"""

from __future__ import annotations

import json
import os

BOLT_RUNTIME_ENV_PATH = "/dev/shm/vllm_bolt_runtime_env.json"


def apply_bolt_runtime_env() -> int:
    """Apply the launcher's env file; returns the number of keys applied."""
    try:
        with open(BOLT_RUNTIME_ENV_PATH) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return 0
    if not isinstance(data, dict):
        return 0
    applied = 0
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("VLLM_"):
            os.environ[key] = str(value)
            applied += 1
    return applied
