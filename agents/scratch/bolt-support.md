# Bolt deployment — active

User authorized official-checkpoint download, packed in-memory weights with
in-kernel INT8 GS128 execution, a peer recipe, commits/pushes and switching the
active service to Bolt. Continue through production hosting; no new deployment
permission is needed. Preserve unrelated user edits in AGENTS.md,
scripts/serve_recipe_qwen38.sh, deleted logs/README.md, and untracked
 docs/recipes/qwen38_flash_next_support.md.

Implementation and checkpoint revisions are in docs/recipes/nemotron_bolt.md.
Both forks have local commits not yet pushed. vLLM remote is origin; AITER
remote is curved (not origin). Work directly on authoritative main.

## Costly evidence and resumption

All raw artifacts below are under ignored agents/tmp/bolt-eval/. The private
regression.key is mode0600; never print or commit its contents. Diagnostic API
is localhost8021. The peer launcher's PID/log live in logs/serve_recipe_bolt.
Do not use old boot.py PID files to stop the current supervisor.

- Official target/draft verified with hf cache verify: 118/6 files.
- Packed component oracle, routed experts, boundary slices, distinct fused
  NVFP4 scales, graph replay pass; generated ISA uses INT8 MFMA. Requantization
  introduces ~0.8–1.7% random-activation relative error, not a model quality gate.
- Loader suite5, Mamba recurrence/chunk suite8, post-context-fix LiLiCorr suite32
  pass. Full tests/test_config.py cannot collect due to pre-existing missing
  huggingface_hub.ResolvedRevision. Do not upgrade the serving venv for it.
- INT8 micro PASS/battery4/4 after shared-code changes. Independent HD128
  noncausal SW1024+sinks attention probe passes at lengths31/1091/4271.
- DFlash NVFP4 context projection initially cast features to uint8 weight dtype,
  collapsing acceptance. Fixed to draft compute dtype, with signed/fractional
  regression. Integer activation input now rejected by AITER quantizer.
- Eager post-fix C8:627 tokens/14.425s, acceptance length4.606.
- Graph baseline (no mixed scheduler policy): cold638/13.125s, warm649/13.533s,
  acceptance4.483/4.353. graph-cold-c8.json and graph-warm-c8.json.
- Reasoning, tool calls, image/video and streaming pass (functional*, video*,
  stream*). 28,067-token retrieval cold98.84s/repeat27.54s, same phrase.
  long-prefix-results.json and long-prefix-request.json retain baseline data.
- Long C8 baseline569 tokens/228.630s, acceptance4.445. graph-long-c8.json.
  long-bench-prefix.txt retains common prefix. Reproduce via benchmark script
  --prefix-file that file --max-tokens128 (separate flag/value). Warm the original
  long_prefix.py cold/repeat pair first for comparable cache state.
- Final Qwen regression passed unchanged TP4/C6/NS6, PTQR target/draft,
  INT8G128, FP32 state, graphs, AIS, custom AR, existing12GiB RAM tier.
  Six concurrent queries sane: qwen-final-greedy.json/qwen-final-server.log.
  Qwen is stopped; redo only if new shared-code edits require it.

## Current boot and next actions

Bolt is booting with VLLM_FS_GPU=weights, HF_HUB_OFFLINE=1 and
VLLM_SERVER_DEV_MODE=1 (diagnostic reset endpoint only). First AIS load builds
per-rank repack caches; allow time for 90 shards. Validate cache bytes against
checkpoint using scripts/fs_gpu_repack_check.py; it accepts rank0 only, so use
ignored temporary symlink directories to inspect other ranks.

Current boot adds mixed prefill4256/interval16 plus fair admission. Qwen's
HD256/GQA6-specific attention switches do not cover Bolt HD128/GQA8; removed
no-op exports from launcher, with no shared attention changes. Candidate tests:
short C8 cold/warm, original long-prefix pair (save baseline results first),
long C8, GPU-prefix reset preserving CPU followed by retrieval and restore
metric inspection. Do not claim CPU restore until demonstrated.

Then finish ledger/docs, security-gated commit, push both main refs and verify.
Do not claim performance gain from unpaired or variable-output smoke runs.

## Production OS step

The reviewed scripts/vllm-bolt@.service uses existing /etc/llama/llama-api.env.
Directory is root0700 and credentials are inaccessible here. No API-key env
is present. sudo -n true fails. sudo -n -l only allows an obsolete Qwen unit
which is not installed, so does not authorize the needed systemd operations.
After all independent validation, ask for the concrete OS installation step:

    sudo install -m0644 scripts/vllm-bolt@.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl disable --now vllm.service
    sudo systemctl enable --now "vllm-bolt@$USER.service"

Stop the diagnostic launcher first to release GPUs. Once production runs as
the same account, use its API-key environment privately for endpoint checks,
without printing credentials. Required outcome: authenticated Bolt hosting
on8020 with existing production credentials and systemd restart ownership.
A diagnostic8021 process is not completed deployment.
