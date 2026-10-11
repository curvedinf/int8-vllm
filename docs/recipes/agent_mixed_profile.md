# Long-context agent workload and lifecycle profile

This test targets the production complaint: long conversations slowing down
while other conversations prefill. It does not change the numerical recipe.
Nginx **must be stopped** during all measurements. The client refuses to start
and aborts if any Nginx process appears; use the loopback backend directly.

## Workload

`scripts/bench_agent_mixed.py` freezes the documentation corpus into a manifest.
Eight independent approximately 40k-context conversations start at
0, 7, 19, 29, 43, 61, 83, and 109 seconds. Each has up to three responses:
1536-token release notes, a 256-token tool-call turn, then a 768-token analysis
after a mocked file result. Tool calls use the model's actual streamed IDs and
arguments; no tool executes on the host. If a model does not call the tool, the
diagnostic result is supplied as a user message, explicitly visible in the
saved request. Natural EOS is allowed. Waits between turns are 2, 5, or 8 s.

Sampling matches the deployed defaults: temperature 1, top-p .95, top-k 20,
thinking enabled, low reasoning effort. Seeds are fixed per conversation/turn.
Eight client slots are bounded independently of the unchanged C6 engine.
There is no infinite generator, daemon, or periodic background job.

Two modes must not be conflated:

- **Agent mode:** fixed initial arrivals, response-dependent follow-ups.
- **Replay mode:** `--replay-from <run>` uses the exact saved request bodies and
  original send offsets. It is open-loop, including frozen assistant/tool
  history, and is suitable for comparing timing/profile legs. New outputs can
  differ, particularly if batching changes. Prefix-hit parity must be checked.
  Profiling overhead can also delay sends at the bounded client semaphore;
  `ready_ns` and `sent_ns` retain that delay. TTFT starts at actual send time.
  Frozen follow-ups may overlap their preceding request in replay mode; only
  agent mode preserves response dependencies.

Prepare the manifest once. Do not rebuild it between comparison legs:

```bash
.venv/bin/python scripts/bench_agent_mixed.py --prepare \
  --manifest logs/agent_mixed/manifest.json --prompt-namespace unique-campaign
```

## Isolated execution

As root, stop Nginx first, then the production service. Temporarily mask
`nginx.service` with `systemctl mask --runtime nginx.service` if automation can
restart it; remove the mask when restoring production. Run the recipe and
client as the service owner, not root. Supply credentials through the inherited
environment, **not command-line arguments**; never print the key or a process
command containing it.

For the test boot, set these additional environment variables:

```text
LIFECYCLE_METRICS=1
VLLM_LIFECYCLE_DIR=/absolute/path/to/run/host
VLLM_LIFECYCLE_GATE=/absolute/path/to/run/active
DISKROOT=/new/isolated/cache/namespace
```

CPU-only instrumentation records spans and scheduler request/token metadata.
It performs no tensor reads or GPU synchronization. Without
`VLLM_LIFECYCLE_DIR`, its decorators return the original functions. The metrics
lever enables per-request timing and detailed speculative acceptance; it does
not alter model parameters, numerical flags, batching policy, or sampling.

After a real completion smoke test and confirmation of zero running/waiting
requests:

```bash
.venv/bin/python scripts/bench_agent_mixed.py \
  --manifest logs/agent_mixed/manifest.json \
  --output logs/agent_mixed/baseline --duration 900 --request-timeout 600 \
  --trace-gate /absolute/path/to/run/active
```

The gate exists only during the test, including a five-second cache-drain
interval. Cancellation closes all streaming connections and retains unfinished
and unstarted requests in the summary. HTTP 429/503 get at most two retries.
The first SSE event's large prompt-ID array requires a larger client read buffer;
this is covered by the 40k-event regression test.
The client opens a fresh loopback TCP connection per HTTP request. Agent waits
include five seconds, which can race the server's keep-alive timeout when
reusing pooled sockets. Transport failures are retained, not silently replayed.
The connection policy is recorded in each run's metadata.

## Kernel and CPU tracing

Validate profiler compatibility with a bounded micro-test including graph
replay first. Do not replace the patched 7.2 HIP/HSA runtime. Do not attach an
unbounded PyTorch profiler to production.

For a separate fresh-cache recipe boot, retain the lifecycle settings and add:

```text
VLLM_LIFECYCLE_KINETO=1
VLLM_LIFECYCLE_GPU_CHUNK_STEPS=64
VLLM_SERVER_DEV_MODE=1
```

Use an isolated, time- and memory-limited systemd unit, as the service owner,
with the production EnvironmentFile, recipe launcher, and these overrides.
The continuous-capture unit used MemoryMax=50G, MemorySwapMax=8G, OOMPolicy=stop
and RuntimeMaxSec=3600. Do not expose the development RPC endpoint through Nginx.
Do not set PROFILER_WRAPPER. Native rocprofv3 passed a single-GPU smoke but
failed the TP4 collective preflight with excessive memory growth on this stack;
it is not the validated capture method here.

The validated method is GPU-only Kineto (no CPU activities, shape recording,
stacks, or tensor-memory tracking). First run
`scripts/profile_mi100_preflight.py --backend kineto` under four-worker
`torch.distributed.run`, with the recipe ROCm/collective environment and a
memory/time limit. This checks graph replay and custom/RCCL collectives.
Then verify a short full-model capture before the complete replay:

```bash
.venv/bin/python scripts/agent_profile_control.py start --tag cold --ranks 0
.venv/bin/python scripts/bench_agent_mixed.py \
  --manifest logs/agent_mixed/manifest.json \
  --replay-from logs/agent_mixed/baseline \
  --output logs/agent_mixed/profile-cold --duration 1200 --request-timeout 900 \
  --trace-gate /absolute/path/to/run/active
.venv/bin/python scripts/agent_profile_control.py stop \
  --output logs/agent_mixed/cold-capture-manifest.json
.venv/bin/python scripts/kineto_to_agent_csv.py \
  /absolute/path/to/run/host/gpu/cold-*.json.gz \
  --output logs/agent_mixed/cold-csv
```

The controller refuses start/final-stop transitions unless Nginx is down and
the backend has zero running/waiting requests. During recording, all TP workers
rotate after 64 completed sampling steps, synchronizing and exporting before
the next worker call. This intentionally adds pauses: use the separate baseline
for latency. All four workers acknowledge control and rotate together; only
selected ranks record GPU events. Use `--ranks 0` for continuous long runs and
`--ranks 0,1,2,3` only for bounded cross-rank samples. Native recorder memory
still grows across captures after explicit Python-cycle collection; long
all-rank recording exceeded this machine's memory budget. Do not claim all-rank
continuous coverage from a rank-0 trace.
For continuous coverage of every rank, repeat the isolated replay on separate
boots with ranks 0, 1, 2, and 3, keeping the manifest and cache preparation fixed.
Exports stage JSON beside the output on disk, not in RAM-backed `/tmp`.
A monolithic full-workload export exceeded the test service's memory limit;
do not disable chunking for long captures. Keep the unit alive through the final
export. Raw compressed Chrome traces retain kernel, HIP
runtime, and copy events, including graph replay kernels. Clock calibration
maps them to client/host monotonic time. Use separate, new CSV directories per
tag. Conversion appends all chunks per PID and namespaces their correlation IDs.
A CPU-group barrier after rotation prevents another rank from entering the
next GPU collective during export. The analyzer also reports phase totals
excluding each chunk's first step as a conservative sensitivity check;
earlier all-rank samples predate that barrier.
Stop the isolated server after export. Keep Nginx stopped until the normal
production service has restarted and generated a sane reply.

A separate bounded `py-spy record` capture can sample the API/engine/worker
process tree. Include its rate/options and overhead in the report. Host span
durations are **not** GPU durations: launches are asynchronous, spans nest, and
TP rank times must never be added together and called elapsed time.

## Artifacts and interpretation

Each run retains request bodies, token-ID burst timestamps, reasoning/content,
tool-call assembly, usage/cache counts, per-request timing/acceptance, failures,
Prometheus snapshots, process I/O/RSS, and GPU clocks/power. Manifests and traces
are local ignored artifacts; they can contain project documentation and model
outputs. Do not publish them without reviewing their contents.

```bash
.venv/bin/python scripts/analyze_agent_mixed.py logs/agent_mixed/baseline \
  --output logs/agent_mixed/baseline-analysis.json
.venv/bin/python scripts/analyze_agent_profile.py \
  --rocprof logs/agent_mixed/cold-csv --host /absolute/path/to/run/host \
  --run logs/agent_mixed/profile-cold \
  --output logs/agent_mixed/kernel-analysis.json
.venv/bin/python -m unittest discover -s scripts -p test_bench_agent_mixed.py
```

Decode rate excludes the first burst and TTFT. Stream-gap percentiles describe
network-visible speculative bursts, not hypothetical individual-token latency.
Rates count reasoning as well as content tokens. `first_content_ns` separately
records the first visible-content event; some budget-limited answers can remain
entirely in reasoning. Transport completion is not a claim that the agent task
finished successfully; inspect `finish_reason`, usage, and saved messages.
Client-side contention rates count periods when another request awaits its
first token; these include queue wait as well as prefill. Kernel phase labels
are attributed separately using HIP launch correlation IDs and worker/scheduler
spans (GPU-start fallback is labeled in the analyzer). GPU busy fractions use
interval unions per rank/device, not summed kernel durations. Keep the full
CSV names/correlation IDs for detailed attribution; category names alone are
not enough to identify an optimization.

No performance improvement is claimed by adding this test. Baseline and
profiled timing must be reported separately, with cache state and completion
counts, before proposing changes to the serving stack.
