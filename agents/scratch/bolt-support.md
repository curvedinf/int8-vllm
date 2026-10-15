# Bolt production activation — deferred by user

User authorized the official packed-weight implementation, peer recipe,
commits/pushes and making Bolt active, then explicitly deferred the OS
installation/activation step. Do not activate services until the user resumes
that work. Separate distributable Qwen3.8 and Bolt units are saved in scripts/;
installation and selection are documented in docs/recipes/services.md.
Implementation, checkpoint revisions, validation and scheduler tradeoffs are
in docs/recipes/nemotron_bolt.md. Do not ask for deployment permission again.

Preserve unrelated user edits in AGENTS.md, scripts/serve_recipe_qwen38.sh,
deleted logs/README.md, and untracked docs/recipes/qwen38_flash_next_support.md.
Both forks use main. vLLM remote is origin; AITER remote is curved (not origin).
Implementation and validation are published on both main branches.
Only production service activation and its final endpoint check remain.

## Current process and remaining work

Qwen post-loader regression PASSED and its supervisor/workers are stopped.
Bolt is running from the verified AIS caches on localhost8021 with the
private existing regression key, HF offline, and VLLM_SERVER_DEV_MODE=0.
Use logs/serve_recipe_bolt/supervisor.pid and server.log; do not use stale
boot.py PIDs. The updated stop command signals its private setsid group.

The cache-hit serving gate PASSED: all8 AIS cache hits, C8 acceptance4.559,
reasoning/tools/image/stream, authentication and disabled dev endpoint. Bolt
is healthy on8021 under the diagnostic key; the production OS step remains.
The final serving ledger row is in the canonical JSONL; both remote main
refs were verified. Root installation was attempted and sudo returned
"interactive authentication is required". The user has been asked to run
the concrete commands below and chose to do this later. Unit is currently
not installed. Keep the
healthy diagnostic process until the unit stops it through ExecStartPre.
Once activated, authenticate on8020 and verify persistent service ownership.

## Costly evidence

The profiling investigation is documented in docs/recipes/bolt_profile_20261015.md.
Raw request/response timings, rank-0 Kineto traces, all-rank host spans and
analysis are retained under ignored agents/tmp/bolt-profile/. The normal
diagnostic boot must keep VLLM_SERVER_DEV_MODE=0 and lifecycle tracing disabled.

All raw files are under ignored agents/tmp/bolt-eval/. regression.key is0600;
never print or commit it. Public scripts/probe_bolt_prefix.py reproduces the
original long prefix exactly, and scripts/bench_bolt_recipe.py retains inputs
and output counters. Use canonical docs/ledger for numerical results.

- candidate-{cold,warm,long}-c8.json, candidate-prefix/, candidate-cpu-restore.json
  and candidate-before-reset.metrics/candidate-after-restore.metrics: all gates
  pass. CPU restoration has positive load bytes and external-prefix hits.
- graph-{cold,warm,long}-c8.json: no-policy graph baseline. Single paired runs,
  variable response lengths; no equal-work throughput claim.
- cache-byte-gates.log: all8 rank caches pass source byte checks.
- candidate-server-complete.log: complete successful first-cache build/serve.
- ais-small-write-stall.log: initial hipFile cache writes stalled in journal
  waits. Preallocation did not fix it. Buffered CPU writes solve construction;
  AIS source/cache reads and packed GPU storage remain unchanged.
- final-loader-micro.log / final-loader-battery.log: PASS /4of4.
- cpu-repack-test.log: short writes, alignment, publication, invalidation pass.
- qwen-post-loader-{test.log,server.log,greedy.json}: final unchanged Qwen
  TP4/C6/NS6, AIS, graphs, custom AR, INT8G128, FP32 state,12GiB RAM tier pass.
- functional*, video*, stream*: earlier functional gates; cache-hit versions
  are being retained separately by the current coordinator.

Full tests/test_config.py cannot collect due to pre-existing missing
huggingface_hub.ResolvedRevision. Do not upgrade the serving venv. Ruff on the
existing repack module reports5 pre-existing findings; new test/probe files
pass. No new numerical kernel changes have occurred since the prior gates.

## Required OS access

Prepared scripts/vllm-bolt@.service uses existing /etc/llama/llama-api.env.
That directory is root0700; credentials are inaccessible here. No production
API key is in the current environment. sudo -n true fails; sudo -n -l permits
only an obsolete Qwen unit that is not installed, not the required operations.
The user deferred this concrete OS step; do not request it again during
the service-file preparation task. For later activation from the serving account:

    sudo install -m0644 scripts/vllm-bolt@.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl disable --now vllm.service
    sudo systemctl enable --now "vllm-bolt@$USER.service"

The unit's ExecStartPre stops the diagnostic Bolt supervisor automatically.
Once production runs as the same account, privately use its API-key environment
for authenticated checks without printing credentials. Verify active+enabled,
port8020, nemotron-bolt, no dev mode, actual generation and startup contract.
Then update active-service selection in recipe README, commit/push final state,
and remove this obsolete active note. Diagnostic8021 is not final deployment.
