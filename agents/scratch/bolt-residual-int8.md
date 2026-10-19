# Decoder INT8 experiment artifacts

The deployed contract, user acceptance override and reproducible measurements
are recorded in `docs/recipes/nemotron_bolt.md` and ledger rows
`BOLT_DECODER_INT8_G128_V1_20261019` / `BOLT_DECODER_INT8_G128_V2_20261019`.

Raw artifacts remain in ignored `agents/tmp/residual-int8/` (mode 0700):
`baseline-bench.json`, `direct-v1-bench.json`, `candidate_v2-bench.json`,
`candidate-v2-quality.json`, `candidate-v2-long/`, `direct-v2-candidate.log`,
`direct-v2-source.json`, and `direct-v2-{micro,battery,gpu,tp4-varied}.log`.
The raw V2 one-point acceptance assertion failure is retained; the user
explicitly accepted that result. The private probe credential was deleted.

Source snapshot predates equivalent test formatting and fail-closed validation
guards; kernel arithmetic is unchanged. Live validation exercised the guards.
The production collective is the exact tested binary copied to
`vllm/_residual_g128.so`; rebuild it on fresh checkouts with
`scripts/build_residual_g128.sh`. Both sibling checkouts are required.

Do not run GPU tests beside serving. An early prototype test batch did so and
OOMed; that failed batch is not validation evidence. The superseded
storage-only prototype's 66 GPU passes also do not validate the direct path.
