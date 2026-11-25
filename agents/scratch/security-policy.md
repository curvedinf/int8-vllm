# Project disclosure policy

The pre-commit security gate pauses on uncovered findings for an explicit
user disposition; existing occurrence is not consent.

Record user-approved permissions by information type, purpose, and conditions,
not by exempting files, paths, globs, or scanner rules; every file remains
inspectable. Do not retain sensitive values here.

On 2026-10-15, the user directed redaction of the local account paths identified
in four historical experiment-ledger entries during the Bolt profiling report
gate. Those entries use generic model/checkout placeholders instead. This
decision grants no permission to disclose local account paths elsewhere.

On 2026-10-16, the user directed replacing the existing account-specific
offload flag-directory path with a checkout-relative path. This is a removal
decision, not permission to disclose account paths. Existing Git history is
unchanged.

On 2026-10-19, the user explicitly approved using their configured Git name
and email as author and committer attribution for the decoder INT8 changes
being committed and pushed in int8-vllm and the sibling int8-aiter repository.
This approval covers these commits only, not contact details in file content
or other disclosures. No identifying values are recorded here.

On 2026-11-25, the user directed that the empty-API-key change for issue #3
be committed and pushed to main when complete. This approves the configured
Git name and email as author and committer attribution for that commit only.

On 2026-11-25, the user directed that Inkling be configured with the best
configuration, its Dockerfile tested, and the result pushed ("push it"). This
approves the configured Git name and email as author and committer attribution
for the commits carrying that work, in int8-vllm and in the sibling int8-aiter
repository (the routed-expert and tabulated-SwiGLU kernels the recipe loads),
only. It does not approve contact details in file content, local account paths,
or any other disclosure.
