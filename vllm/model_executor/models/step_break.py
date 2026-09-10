#!/usr/bin/env python3
"""VLLM_STEPBREAK (env-gated): per-layer-type GPU-time split in the engine's
decoder loop. Safe under CUDA graphs: only records CUDA events (no sync)
during capture; accumulates elapsed on the post-step sync. In eager mode
(--enforce-eager) events are exact.
"""
import os
import torch


class StepBreak:
    def __init__(self, layer_types: list[str]):
        self.on = os.environ.get("VLLM_STEPBREAK") is not None
        self.acc = {t: 0.0 for t in set(layer_types)}
        self.n = 0
        self._pend = []

    def layer(self, layer_type: str):
        if not self.on:
            return None, None
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        return e0, e1

    def end_layer(self, layer_type: str, e0, e1):
        if not self.on or e0 is None:
            return
        e1.record()
        self._pend.append((layer_type, e0, e1))

    def end_step(self):
        if not self.on:
            return
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
            for t, e0, e1 in self._pend:
                self.acc[t] += e0.elapsed_time(e1) / 1000.0
        self._pend.clear()
        self.n += 1
        if self.n % 25 == 0:
            parts = " | ".join(
                f"{t} {1000*v/self.n:7.2f}ms" for t, v in self.acc.items())
            print(f"[stepbreak] steps {self.n} | {parts}", flush=True)
