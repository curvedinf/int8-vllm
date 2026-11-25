# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead named-phase wall accumulator for a live process.

With ``VLLM_PPHASE_FILE=<prefix>`` set, a PhaseTrace writes one summary line
per ``every`` ticks to ``<prefix>.<pid>.<name>``; without it every instance is
a disabled no-op.
"""

import os


class PhaseTrace:
    __slots__ = ("every", "name", "path", "sums", "n")

    def __init__(self, name: str, every: int = 200) -> None:
        prefix = os.environ.get("VLLM_PPHASE_FILE")
        self.name = name
        self.every = every
        self.path = f"{prefix}.{os.getpid()}.{name}" if prefix else None
        self.sums: dict[str, float] = {}
        self.n = 0

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def add(self, key: str, seconds: float) -> None:
        if self.path is not None:
            self.sums[key] = self.sums.get(key, 0.0) + seconds

    def tick(self) -> None:
        if self.path is None:
            return
        self.n += 1
        if self.n % self.every:
            return
        parts = " | ".join(
            f"{k} {1000 * v / self.every:6.2f}ms" for k, v in self.sums.items()
        )
        try:
            with open(self.path, "a") as f:
                f.write(f"[{self.name}] n={self.n} | {parts}\n")
        except OSError:
            pass
        self.sums.clear()
