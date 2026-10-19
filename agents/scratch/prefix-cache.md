# Prefix cache publication pending

The fix is deployed on vllm-bolt with 8 GB KV, hash unit 8 and checkpoints
64/128/232. nginx restored; both services active. All live gates passed:
C8 replay hits 232 on each request, long retrieval repeat hits 28056, quality
52/52; CPU/core and GPU gates recorded in the new experiment-ledger rows.
Raw private requests/token IDs/responses and duplicate credential removed;
sanitized measurements and test logs remain under ignored agents/tmp/prefix-cache.

All implementation/docs/tests/ledger files are staged. Awaiting explicit
security disposition for two pre-existing local account paths in the cache
manager (lines 1765/1903) and configured personal Git author email. Async
question offers neutral attribution + relative paths, retain for this commit,
or leave deployed changes uncommitted. Do not publish before answer. Apply
choice, recheck exact staged candidate, commit/push main as owner curved.
Remove this note and its index entry after publication; canonical evidence is
in docs/recipes/nemotron_bolt.md and the experiment ledger.
