#!/usr/bin/env bash
# One experiment leg: boot the production recipe with EXP_ENV overrides,
# measure speed (bench_quick) + quality (KLD gate + chat soak), append a
# ledger row. Usage:
#   scripts/surface_ab.sh <exp_id> [repeats]
# Env contract:
#   EXP_ENV  — extra env for the server leg, e.g. "VLLM_GFX908_DF2_FC_W8A8=1"
#   DRAFT_MODEL_DIR — optional draft checkpoint override (E1/E4/E5/E8)
#   SKIP_KLD / SKIP_SOAK / REPEATS — optional controls
set -euo pipefail

EXP_ID="${1:?exp id}"
REPEATS="${2:-1}"
ROOT="${HOME}/vllm-gfx908"
LEDGER="${ROOT}/logs/surface_experiments/ledger.jsonl"
KEY="${VLLM_API_KEY:-test-key-local-only}"
mkdir -p "$(dirname "${LEDGER}")"

cd "${ROOT}"

# 1) boot with overrides
if curl -fsS --max-time 2 -H "Authorization: Bearer ${KEY}" http://127.0.0.1:8020/v1/models >/dev/null 2>&1; then
  echo "ERROR: server already up on 8020" >&2; exit 1
fi
env ${EXP_ENV:-} VLLM_API_KEY="${KEY}" scripts/serve_direwolf_qwen38.sh start
trap 'scripts/serve_direwolf_qwen38.sh stop >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 90); do
  curl -fsS --max-time 2 -H "Authorization: Bearer ${KEY}" http://127.0.0.1:8020/v1/models >/dev/null 2>&1 && break
  sleep 10
done
curl -fsS --max-time 2 -H "Authorization: Bearer ${KEY}" http://127.0.0.1:8020/v1/models >/dev/null 2>&1 || {
  echo "ERROR: server did not come up"; tail -30 logs/serve_direwolf_qwen38/server.log >&2; exit 1; }

# 2) speed: bench legs
TPOTS=(); TGS=(); TTFTS=()
for r in $(seq 1 "${REPEATS}"); do
  scripts/bench_quick.sh "${HOME}/models/Qwen3.8-27B-GPTQ-8bit-gs128" "sx_${EXP_ID}_r${r}" "${KEY}" >/dev/null 2>&1 || true
  M="logs/c8_optimization/sx_${EXP_ID}_r${r}/c8_metrics.txt"
  TPOTS+=("$(grep -oP 'Mean TPOT \(ms\):\s+\K[0-9.]+' "${M}" 2>/dev/null || echo null)")
  TGS+=("$(  grep -oP 'Output token throughput \(tok/s\):\s+\K[0-9.]+' "${M}" 2>/dev/null || echo null)")
  TTFTS+=("$(grep -oP 'Mean TTFT \(ms\):\s+\K[0-9.]+' "${M}" 2>/dev/null || echo null)")
done
echo "${EXP_ID}: TPOT=[${TPOTS[*]}] TG=[${TGS[*]}] TTFT=[${TTFTS[*]}]"

# 3) quality: KLD gate needs its own boot (recorder-armed, deterministic stop);
#    reuse this server only for the soak.
SOAK_N="${SOAK_N:-100}"
SOAK=$(.venv/bin/python - "$SOAK_N" <<'PY'
import json, sys, urllib.request, concurrent.futures as cf
N = int(sys.argv[1])
def one(i):
    try:
        p = {"model": "qwen3.8-27b-gptq8",
             "messages": [{"role": "user", "content": f"In two sentences, explain why the sky is blue at noon but red at sunset. Variation {i%17}."}],
             "max_tokens": 96, "temperature": 0.7, "top_p": 0.95}
        r = urllib.request.Request("http://127.0.0.1:8020/v1/chat/completions",
            json.dumps(p).encode(), {"Authorization": "Bearer test-key-local-only", "Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(r, timeout=300))
        m = out["choices"][0]["message"]
        txt = (m.get("content") or "") + (m.get("reasoning") or "")
        return len(txt.strip()) > 20
    except Exception:
        return False
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, range(N)))
print(f"{sum(res)}/{N}")
PY
)
echo "${EXP_ID}: soak ${SOAK} non-empty"

# ledger row (speed + soak; KLD filled by kld_gate_boot into exp_<id> npz and merged below)
.venv/bin/python - "$EXP_ID" "$SOAK" "${TPOTS[@]}" "${TGS[@]}" "${TTFTS[@]}" <<'PY'
import json, sys
exp, soak = sys.argv[1], sys.argv[2]
n_rep = (len(sys.argv) - 3) // 3
tpot = sys.argv[3:3+n_rep]; tg = sys.argv[3+n_rep:3+2*n_rep]; ttft = sys.argv[3+2*n_rep:]
row = {"exp": exp, "tpot_ms": tpot, "tg_toks": tg, "ttft_ms": ttft, "soak_nonempty": soak}
try:
    row["env"] = __import__("os").environ.get("EXP_ENV", "")
except Exception:
    pass
with open("logs/surface_experiments/ledger.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print("ledger:", row)
PY
