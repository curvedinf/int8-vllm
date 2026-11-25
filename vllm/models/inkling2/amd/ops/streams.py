# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide side streams for independent work inside a captured graph.

The caching allocator keeps a pool per stream, so every extra stream costs
memory in the profile run that sizes the KV cache. Three shared streams serve
all users; callers fork with `wait_stream` and join before their consumer, and
users of one stream never overlap in time.
"""

import torch

_STREAMS: list[torch.cuda.Stream] | None = None


def side_stream(index: int = 0) -> torch.cuda.Stream:
    global _STREAMS
    if _STREAMS is None:
        _STREAMS = [torch.cuda.Stream() for _ in range(3)]
    return _STREAMS[index]
