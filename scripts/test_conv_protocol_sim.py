#!/usr/bin/env python3
"""Protocol referee for the spec-decode conv rolling buffer — CORRECTED
to the engine's real allocation geometry.

Geometry facts (verified in-tree):
- Rolling width L = conv_kernel - 1 + num_spec  (shape funcs pass
  num_spec = num_speculative_tokens = K)  =>  L = M + K, cols 0..L-1.
- Spec forward rows N = 1 + K (dflash.py num_query_per_req; row 0 = anchor).
- Kernel write mask splits the buffer at VAL = L - N = M - 1: cols
  [0, M-1) = copied overlap, cols [M-1, L) = x = [anchor, d_1..d_K].
- num_accepted a = valid count incl. correction, in [1..K+1]; neutral = 1.
- Kernel READ (STEP 1): window = buf[a-1 .. a+M-2]  (offset a-1).
- Kernel WRITE source: buf[a + j] for j in [0, M-1).
- Migrations: conv shift = a-1; temporal column = a-1; post-shift a := 1.

Ground truth: the window for row 0 must be the M stream tokens immediately
before x[0] = anchor, i.e. ending at the previous round's x[a-1] = d_{a-1}
(col M-1 + (a-1) = a + M - 2). The engine read window [a-1, a+M-2] ends
exactly there => CONSISTENT under the real geometry.

The earlier "conviction" assumed head width M (L = M+1+K) — wrong for this
engine. Run both geometries to make the dependence explicit.
"""

import random

W, M, K = 4, 3, 6
N = 1 + K
ROUNDS = 500


def simulate(head_width: int, read_delta: int, reset_val: int, mig_delta: int,
             cross_every: int):
    """head_width: overlap-head width the allocation provides (engine: M-1;
    the earlier wrong assumption: M). read_delta added to offset a-1
    (0 = engine, 1 = proposed-but-wrong fix). reset_val/mig_delta as engine."""
    L = head_width + N
    rng = random.Random(7)
    stream = [5000 + i for i in range(60)]   # processed truth
    anchor = rng.randint(0, 4999)            # prefill bonus (unprocessed)
    # prefill wrote the last M processed tokens at cols [0, M-1)
    buf = list(stream[-M:]) + [None] * (L - M)
    a = 1
    bad = []
    for r in range(ROUNDS):
        x = [anchor] + [rng.randint(0, 4999) for _ in range(K)]

        truth = stream[-M:]                  # M tokens before x[0]=anchor
        off = a - 1 + read_delta
        window = buf[off : off + M]
        if window != truth:
            bad.append((r, a, window, truth))
            if len(bad) >= 3:
                break

        # engine WRITE: overlap-head = buf[a + j] for j < head_width,
        # x at cols [head_width, head_width + N)
        new_buf = [None] * L
        for j in range(head_width):
            src = a + j
            new_buf[j] = buf[src] if src < L else None
        for i in range(N):
            new_buf[head_width + i] = x[i]
        buf = new_buf

        a_next = rng.randint(1, K + 1)
        stream.extend(x[:a_next])            # anchor + accepted drafts
        correction = rng.randint(0, 4999)
        anchor = correction
        a = a_next

        if cross_every and (r + 1) % cross_every == 0:
            b = max(a - 1 + mig_delta, 0)
            buf = buf[b:] + [None] * b
            a = reset_val
            if buf[:M] != stream[-M:]:
                bad.append((r, "mig", buf[:M], stream[-M:]))
                if len(bad) >= 3:
                    break
    return bad


for head, name in ((M - 1, "REAL (L=M+K, head=M-1)"), (M, "assumed-wrong (head=M)")):
    for cross in (0, 7):
        label = f"crossing/{cross}" if cross else "no-cross "
        bad = simulate(head, 0, 1, 0, cross)
        print(f"{name:28s} {label} engine protocol: "
              f"{'CONSISTENT' if not bad else 'MISMATCH ' + repr(bad[:1])}")
