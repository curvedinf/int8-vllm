#!/usr/bin/env bash
# Boot the production recipe, capture the 52-prompt KLD probe, compare vs the
# BF16 reference, stop the server. One argument: the result tag.
#
#   scripts/kld_gate_boot.sh <tag>            # e.g. fp32state_round
#
# Extra env (MAMBADT, NS, CAR, UA, ...) passes straight through to
# scripts/serve_direwolf_qwen38.sh. Artifacts land in
# ~/models/kld/quant_audit/<tag> and ~/models/kld/quant_audit/<tag>.npz.
# Reference tag: R0_bf16_ref (bf16 weights, auto KV, fp32 mamba, no spec).
set -euo pipefail

TAG="${1:?usage: kld_gate_boot.sh <tag>}"
ROOT="${HOME}/vllm-gfx908"
AUDIT_DIR="${HOME}/models/kld/quant_audit/${TAG}"
REF_TAG="${REF_TAG:-R0_bf16_ref}"

if curl -fsS --max-time 2 http://127.0.0.1:8020/v1/models >/dev/null 2>&1; then
  echo "ERROR: something is already serving on 8020 — stop it first" >&2
  exit 1
fi

cd "${ROOT}"
VLLM_QUANT_AUDIT="${AUDIT_DIR}" VLLM_API_KEY="${VLLM_API_KEY:-test-key-local-only}" \
  scripts/serve_direwolf_qwen38.sh start
trap 'scripts/serve_direwolf_qwen38.sh stop >/dev/null 2>&1 || true' EXIT

for _ in $(seq 1 180); do
  curl -fsS --max-time 2 http://127.0.0.1:8020/v1/models >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS --max-time 2 http://127.0.0.1:8020/v1/models >/dev/null 2>&1 || {
  echo "ERROR: server did not come up" >&2
  tail -40 logs/serve_direwolf_qwen38/server.log >&2 || true
  exit 1
}

.venv/bin/python scripts/kld_probe_v2.py capture --tag "${TAG}"
.venv/bin/python scripts/kld_probe_v2.py compare --tag "${TAG}" --ref-tag "${REF_TAG}"
