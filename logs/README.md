# Logs (fresh slate 2026-08-22)

Historical experiment logs (Qwen3.6 era, c8_optimization ledger, p67/tq bench
trees) are archived at `~/archived-logs-20260822.tar.gz` and in git history.
The canonical baseline and its numbers now live in `docs/recipes/README.md`.
New experiments append here.

## 2026-08-22 — KLD artifact gate: gs128 vs gs32 (real kernels)

Offline capture (TP3, Ring/Simple NCCL), 64 prompts x 256 tok, temperature-1
logprobs. First compare returned FAIL (inf) — root cause: free-rollout
divergence contaminates positions after the texts split (~median token 30-60):
top-20 overlap falls 100%→0% by position 250, and diverged contexts make KLD
meaningless (garbage-in). Restricting to pre-divergence positions:

| Positions | KLD (mean) | n |
|---|---|---|
| 0–7 | 0.00078 | 469 |
| 0–31 | 0.0389 | 1509 |
| all (incl. diverged) | 0.294 / inf | — |

Verdict: **gs128 artifact PASSES the correctness gate** — early-position KLD
(0.0008) is well under the 0.02 gate on genuinely-comparable prefixes; the
0-31 band drifts up only as some prompts diverge near token ~30. Coherence
(greedy QA/code spot checks) already passed on the live server. The sweep's
teacher-forced number (0.0109) remains the primary recorded gate; this artifact
check confirms no kernel-level regression in the baked checkpoint.

Harness lesson: the offline probe must teacher-force reference sequences
(like the sweep harness) rather than free-roll both models — recorded for the
optimization session.
