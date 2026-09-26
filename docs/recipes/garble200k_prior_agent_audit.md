# Audit of the retained 200k-investigation changes

The September 26 handoff left two behavioral changes and three groups of
tracing additions uncommitted. They did not fix the corruption. The proven
fix is [64-bit KV address arithmetic](bug_gfx908_g128_kv_offset_overflow.md),
committed as `64fcef760e`.

That necessary fix remains. Its five-pair safe-address kernel comparison
measured 812.13 to 812.54 us (+0.05%); it provides no evidence of a material
kernel slowdown.

| Change | Finding | Disposition |
|---|---|---|
| `VLLM_OFFLOAD_NO_LOADS`: `None` to `0` | Real diagnostic bug. `None` tells admission that lookup is pending indefinitely; `0` allows recomputation. The environment check already existed, so the change adds no default-path work. | Retain with a regression test. |
| Per-block negative-refcount assertion | Additional detection, not a repair. It never triggered in the incident. Existing allocation already asserts `ref_cnt == 0`. | Remove the uncommitted assertion. |
| GDN WINHASH/PREMASK and COREHASH extensions | Temporary state hashing and CPU readbacks, gated by environment variables. | Archive patch; restore the file to its committed state. |
| ATTNHASH/ATTNDUMP and BTSCHECK readback extensions | Temporary attention dumps and hashes with incomplete or misleading coverage. | Archive patch; restore the file to its committed state. |
| SLOTTRACE | Temporary slot-mapping dumps, gated by an environment variable. | Archive patch; restore the file to its committed state. |

The raw patch is preserved at
`logs/garble/prior_agent_audit_20260926/worktree_before.diff`. The untracked
repro scripts remain available as investigation evidence.

## The block assertion's actual behavior and cost

The previous comment claimed a double-free would insert a block twice in
the free list. The existing method only inserts on the transition to zero.
A second release changes the refcount to -1 without another insertion.
Allocating that block then fails the existing assertion. The added check
moves detection earlier; it does not implement a new ownership correction.

A direct probe of the original method confirms that the free-list count
remains two after both releases and that subsequent allocation asserts.

Seven interleaved pairs on one pinned CPU measure complete allocation/free
cycles through the actual `BlockPool` code, with prefix caching enabled and
4,669 pool blocks:

| Blocks per cycle | Original median | With assertion | Added time |
|---|---:|---:|---:|
| 1 | 1.260 us | 1.275 us | 0.015 us (+1.19%) |
| 64 | 28.153 us | 29.181 us | 1.029 us (+3.65%) |
| 512 | 216.425 us | 222.156 us | 5.731 us (+2.65%) |
| 2048 | 873.097 us | 896.318 us | 23.221 us (+2.66%) |

This establishes a small CPU cost. It does not establish an end-to-end
throughput regression. Raw pairs, the exact guarded source, and the
replayable benchmark are preserved in
`logs/garble/prior_agent_audit_20260926/`.

```bash
PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python \
  logs/garble/prior_agent_audit_20260926/block_guard_benchmark.py
```

## Diagnostic coverage and performance

When the new tracing flags are absent, their bodies do not execute GPU
copies, reductions, synchronizations, or file writes. The added environment
lookups still run where their Python paths execute. Decode graph replay
does not rerun the captured model's Python forward. Enabled instruments do
perform synchronizing CPU readbacks and must not be used for performance
claims. The final overflow-fix serving measurement had those flags disabled.

Two details weaken the attention-hash instrument's claimed coverage:

- `raw[s // 1664, :520]` ignores the token's offset within its page. It
  hashes the page's first token rather than each requested token's bytes.
- `_ah_prev_out` clones the output before the attention call writes the
  current result. Its logged hash is not a measurement of that call's output.

The BTSCHECK extension also compares every inspected request's block-table
row with `first_slots[0]`, which belongs to the first request. That comparison
does not establish slot/table consistency for subsequent requests.

More fundamentally, Python readbacks use 64-bit indexing. These hashes
cannot prove what the overflowing kernel actually read. The earlier claim
that identical hashes eliminated all software explanations was incorrect.

## Other proposed fixes and the dossier

The incident record shows that classic H2D loads, Mamba lag/chunk changes,
full-row block-table clearing, and Mamba-store suppression were falsified
and already reverted. They have no retained production cost from this
investigation. Active launcher files contain only `G128_GLUON=1`; the
diagnostic bypass files are absent.

The earlier draft of the memcpy dossier's section 12 still blamed the Mamba
store/CoW chain and claimed store suppression was the production default.
Those statements contradicted its later falsification legs and the actual
launcher. Section 12 now points to the proven address-overflow resolution.

The classic D2H store executor was committed on September 14, before the
September 24 Gluon readers, and was not changed by the handoff. Its earlier
investigation is separate; the new overflow finding is not sufficient
evidence to revert it.

## Validation

The retained NO_LOADS regression passes in both normal and diagnostic modes.
Loading the old lookup method makes the diagnostic case fail with
`(None, False)` instead of `(0, False)`. The scheduler and single-type block
manager suites pass **133/133**, the INT8 KV micro test passes, the battery
passes **4/4**, and the large-offset addressing regressions pass **25/25**.

The cleaned production recipe boots on all four GPUs at TP4/C6/NS6 with
grouped INT8 target/draft KV, FP32 Mamba, AITER W8A8/unified attention,
CUSTOM all-reduce, graphs, prefix caching, and CPU offload enabled. The
32-request greedy C6 serving check completes without failures: 32,000
output tokens in 98.742 seconds (**324.08 output tok/s**), speculative
acceptance length **4.904**, acceptance rate **65.06%**.
After that run, all six fresh greedy probes retain the exact healthy
` Paris` logprob of `-0.5463467240333557`; chat correctly answers Paris and
counts from one to five.

This is one serving validation run, not a paired throughput comparison.
The earlier overflow-fix run had different acceptance, so its 286.25 tok/s
result cannot establish a cleanup speedup. The three >200k conversations
and old-prefix continuation from that fix remain the deep-context gate;
this cleanup does not change the kernels or allocation order. Raw checks
and measurements are indexed by ledger row
`GARBLE200K_PRIOR_AGENT_AUDIT_20260926`.
