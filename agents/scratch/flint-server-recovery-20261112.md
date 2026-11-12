# Flint server recovery — 2026-11-12 01:15 UTC

State: server down. `serve_recipe_flint.sh start` boots the API server but
worker init fails; three attempts (clean stop, pinned arch env, 90 s cooldown)
all ended with:

    EngineCore ERROR ... WorkerProc initialization failed due to an
    exception in a background process
    RuntimeError: Engine core initialization failed.

Origin: at 00:45:53 during on-policy corpus generation, Worker_TP1 raised

    int8 payload chain failed (no bf16 fallback):
    RuntimeError('Triton Error [HIP]: Code: 209, no kernel image is available
    for execution on the device')

then the engine shut down at 00:50:52 (clean SIGTERM path).

Known contributing facts:

- A bare `serve_recipe_flint.sh start` uses `KV_DTYPE=int8_per_token_head`.
  The recipe contract is `KV_DTYPE=int8_block_g128` (the earlier working
  server log shows int8_block_g128). Always export it before `start`.
- Boot logs also show `cannot import name 'is_hip_gfx1250' from
  'triton.language.target_info` (fork Triton kernels vs installed Triton
  3.6). It was non-fatal on the previously working boot but may now be part
  of the kernel-image failure.
- No vLLM processes remain; /dev/shm is clean (452 K); host has free RAM.

RESOLVED 2026-11-12 01:15: the real cause was **/tmp exhaustion**, not the
GPUs. The tmpfs was at 25G/31G (old draft-training smoke checkpoints,
rocprof output, comgr dirs), and the boot workers died with
``LLVM ERROR: IO failure on output stream: Disk quota exceeded`` — which is
also the likely source of the original "no kernel image" runtime error
(Triton kernel writes failing mid-compile). Freeing `/tmp`
(removed 18G of my smoke artifacts) and booting with
``KV_DTYPE=int8_block_g128`` brought the server up healthy; the log shows
``kv_cache_dtype': 'int8_block_g128'``.

Pitfall confirmed: a bare ``serve_recipe_flint.sh start`` defaults
``KV_DTYPE`` to ``int8_per_token_head`` (forbidden by the recipe). Always
export ``KV_DTYPE=int8_block_g128``.

Watch `/tmp` before boots: comgr dirs and training smoke checkpoints are
the usual consumers.

Dataset state while blocked:

- `runs/onpolicy_ready/` holds the alpaca tier: 1,791,488 train /
  18,039 val tokens, 2,752 documents, boundaries validated as chat turns.
- Specialized agent/doc/chat tiers were not generated (all 800 requests got
  500 when the engine shut down mid-run).
- `runs/onpolicy_teacher.pt` holds a validated 4,000-window teacher cache
  (top-k 20 from the served cap, zero-padded to 32).
