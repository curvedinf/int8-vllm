# Bolt packed-weight support — active

User authorized downloading and implementing Bolt as a peer recipe to Qwen3.8,
then making Bolt the active service after validation. No further deployment
permission is needed within that scope. Do not claim completion before the
production acceptance gates pass.

## Contract

Keep official IQ2-XXS, Q8_0, and NVFP4 weights packed in GPU memory. Decode and
requantize tiles inside GEMM to INT8 GS128; carry eligible activations as INT8
GS128. No persistent expanded INT8/BF16 weight copy. Preserve floating-point
residual/routing/norm operations, FP32 Mamba state, and sensitive draft surfaces
until independently gated. TP4, INT8-G128 target/draft KV, custom all-reduce,
CPU prefix tier, disk tier explicitly off. LiLiCorr uses NS7 (trained block8).

Artifacts downloaded to the existing models directory, under their repository
basenames, and verified with `hf cache verify --fail-on-missing-files`:

- `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` at
  `a326822af111290f2af5ac59ef4b5040347dabc4`: 118 checked files.
- `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` at
  `4c6a5613b6d887194a3a5493e3a18a5b3930bb86`: 6 checked files.

## Integration and evidence

Starting source is `33505c684`. Existing user edits in AGENTS.md,
scripts/serve_recipe_qwen38.sh, deleted logs/README.md, and untracked
docs/recipes/qwen38_flash_next_support.md must remain untouched.

Referenced upstream snapshot `77154ca6116ae34cbfe314beef1016bde9b63989`
has 2727 commits absent from this fork. Relevant feature commits:

- `73a78e6f1f38e280986b81e0f2a9aa5e1ee6fe47`, LiLiCorr PR #57934:
  ported against existing ModelOpt classes and DFlash interfaces, retaining
  NaN handling, TP sampling synchronization, lookup override, and diagnostic
  cache bypass. Adaptive verification is not ported.
- `6849ca696a`, ModelOpt IQ2/Q8 PR #59604: inspected, not integrated. Its
  backend requires B12X/SM12x and must be replaced for gfx908.

New packed GEMM component supports all three codecs, arbitrary logical K
slices, masked output tails, fused ReLU-squared and GS128 quantized output.
It is not wired into model dispatch. GPU weights remain packed; only the
reference probe expands full matrices. GGML IQ2 codebook retains MIT notice.

Component gate (`scripts/probe_bolt_packed_int8.py --checkpoint <target>`):
synthetic M1/M8/M17 and misaligned K576/K640 slices; actual first-expert up/down,
129 LM-head rows (Q8), and 129 Mamba in-projection rows (NVFP4). All matched
explicit group-scaled reference (relative error zero in this run), including
graph replay. Random-activation relative L2 vs decoded official weights:
IQ2 up 0.0086234, IQ2 down 0.0088799, Q8 LM head 0.0173478, NVFP4 0.0096044.
These are component numerics, not model fidelity or throughput evidence.
INT8 MFMA instructions were observed in generated `_gemm.amdgcn`.

Quantizer reference originally differed by one code at a half-integer tie:
Torch scalar division used reciprocal multiplication, changing scale by one
ULP. Kernel now specifies IEEE division and nearest-even rounding; independent
NumPy reference uses the same contract. Do not weaken exact code checks.

LiLiCorr/DFlash tests: 35 passed on MI100, including GPU head/sampler and graph
tests. Full tests/test_config.py cannot collect: installed huggingface_hub
lacks ResolvedRevision, already imported by that file before this work.
Candidate runner selection was separately exercised without that dependency.
Serving venv torch/triton/runtime unchanged; only standalone ruff 0.17.1 was
installed after a dry-run showing no dependency changes.

Pre-change and final-candidate INT8 KV micro PASS and battery 4/4. Post-port Qwen recipe booted TP4/C6/NS6 (current canonical launcher) with
INT8-G128 target/draft KV, FP32 Mamba and captured graphs. Six concurrent short
queries produced sane factual/code answers; test server was stopped afterward.
This is a smoke regression, not a long-context acceptance or performance gate.
The full Bolt recipe is not running and production activation is pending.

Raw logs and failed merge evidence are in ignored `agents/tmp/bolt-eval/`.

## Remaining work

1. Packed ModelOpt loader/linear methods; preserve original byte blocks and
   implement TP slices plus scalar-scale handling for merged projections.
2. Routed/shared-expert dispatch with latent projections, top9 routing,
   ReLU-squared, INT8 intermediate chaining, and weighted FP32 combination.
   Existing AITER MoE does not advertise non-gated ReLU-squared.
3. Optimize the proven component kernels in sibling AITER; benchmark actual
   expert token-count distributions. No performance claim exists yet.
4. LiLiCorr packed NVFP4 loading and exclusions; integration with shared head,
   INT8 KV, and target hidden-state taps; Mamba2 speculative rollback/prefix
   caching. Qwen GDN kernels do not implement Mamba2.
5. Peer launcher and model/cache namespace; text, image/video, reasoning, and
   tool-parser gates; aligned fidelity/acceptance and performance ledger.
6. Full ladder, Qwen regression, commit/security gate, push main, then activate
   Bolt. Production credential directory is inaccessible to this user and
   passwordless sudo is unavailable; temporary localhost regression can use
   VLLM_API_KEY from a private ignored file. Reconcile service privileges at
   deployment time without disclosing credentials.
