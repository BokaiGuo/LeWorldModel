# P2 exploratory host-to-physical bridge: incomplete, physical blocked

`codex_p2_host_physical_bridge_20261030_v1.host_plan` is an independent
wrapper around the new common-cap frontier entry. It requires an externally
constructed frozen eval CUDA model and an externally pinned tensor-state
identity; it does not load or initialize the model. It checks parameters and
buffers are on the supplied CUDA device, records before/after model state,
elapsed wall time, process CUDA peak counters, source/input bindings supplied
by the caller, actual attempted hook counts, and saved planner/bank hashes.
It synchronizes before and after planning. CUDA peak counters are process
high-water marks and are not reset here; wall time omits model loading and
external setup. Failed work stays in the exclusive directory, and bridge
failure reporting is best effort. The delegated runner separately records its
candidate and operation events.

After planning the wrapper calls the existing strict frontier saved-result
validator for **every candidate** in both arms, checks the reported winner and
raw action witness, and requires the provenance candidate-bank digest to equal
the actual bank plus arm and original frontier contract. It exports two raw10
plans only after both arms validate. No physical outcome is used in selection.

## Critical unresolved bank binding

The delegated `run_native` currently takes `provenance_by_arm` *before* it
compiles the runtime bank, whereas correct frontier provenance includes the
digest of that bank. The wrapper verifies the binding after execution but
cannot generally construct the correct pre-run provenance for a newly
generated bank. Unless an independently authorized, correctly prebound bank
identity is available, this path will fail validation after planning. Do not
run it merely to discover the bank. A future independently reviewed entry
must bind the generated bank after compilation and before either arm begins,
with durable source and stage seals. Do not change the old contract or claim a
pre-outcome seal after inspecting results.

## Physical boundary

`physical_dispatch` always raises before calling any callback. The old physical
consumer only accepts its own sealed historical planner report; this new
report cannot be substituted. A usable bridge still needs independently
reviewed fresh-plan authority, exact physical root/goal and warmup replay,
raw10 selected-action binding, guard/stop and resource instrumentation,
exclusive cross-entry claims, saved-result verification, and handling for
partial or failed branches. Therefore this delivery does **not** implement or
authorize a real physical run.

Before even exploratory host work, main must resolve the bank dependency,
verify complete immutable source/checkpoint/input seals and canonical identities,
confirm GPU exclusivity and numeric settings, capture model loading and full
resource coverage, provide disk/time limits and cross-entry attempt protection,
and review the end-to-end entry. The wrapper has no CLI or complete manifest
loader, and source bindings are not a proven complete closure. Its `caps` are
common upper bounds, not proof of equal use; FLOPs and total budget remain
unknown. Neither P1 nor P2 admission is changed. Candidate roots are the
existing known roots, not a new independent cohort.

Only synthetic metadata/callback tests were run; no GPU, neural inference,
simulator or physics was started. Tests do not prove native CUDA success.
