# P5 checkpoint and runtime state acceptance

The baseline is produced from the real SHA-pinned checkpoint by CPU
`weights_only=True` deserialization, strict model construction/load and eval/
frozen inspection. There is no neural forward, training, CUDA execution,
environment, or future label access. It records all 26 parameter shapes, float32
byte hashes, 93,842 parameter count, module names/classes/training flags,
normalization tensor hashes, local precision flags and pinned source identities.
`BASELINE02` is the current baseline; `BASELINE01` is the earlier
source-incomplete intermediate, retained unchanged and not accepted by this
version of the tool.
The baseline CPU precision is descriptive; it is not evidence of future CUDA
precision. Its zero CUDA-call count refers to CUDA device work; the baseline
does inspect PyTorch's CUDA-related precision flags and autocast setting.

`capture(model, torch)` is a separately callable observation helper for a future
instrumented execution: it reads module flags, actual tensor device/dtype,
requires-grad and tensor bytes, normalization and precision settings without
running a forward. It copies weights to CPU for hashing. It must be called
immediately before the first model call and immediately after the final model
call by a separately reviewed implementation, and its output preserved in a
runtime bundle. The current pinned runner and supervisor **do not call this
helper or produce that bundle**. No command in this document authorizes or
starts them. Existing production journal alone therefore cannot satisfy this
acceptance. A separately reviewed integration is still required before launch.

The offline consumer requires externally SHA-pinned baseline, bundle, supervisor
log and prediction journal. It checks the real baseline again, complete fixed
164/80 strict journal, separate 82/82 off/on completion, supervisor admission,
single launch, return status and PID, before/after equality, complete module
identity, checkpoint parameter bytes, CUDA placement/frozen flags,
normalization and precision. A changed or missing snapshot fails closed.
The bundle links its two snapshots to the completed journal and supervisor
hashes and records a capture-producer source hash; this hash is provenance
only, not an independently approved producer identity. The current acceptance
does not verify capture timing, kernel dtype, context inside model calls, GPU
identity from hardware, supervisor truth, or runtime source immutability. The
result is only self-reported consistency, never scientific admission.

Baseline and future acceptance output use exclusive new files; existing
artifacts are not modified. Baseline must be generated and later reconstructed
with `CUDA_VISIBLE_DEVICES=''`. The validation depends on the CPU PyTorch
version and baseline precision settings remaining the same. No on/off results
are pooled, and renderer/training-distribution equivalence is not established.
