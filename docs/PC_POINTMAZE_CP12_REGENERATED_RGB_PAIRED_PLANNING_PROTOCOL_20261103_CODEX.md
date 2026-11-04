# Development protocol: regenerated-RGB cp12 single/three-frame planning

Status: **protocol only; not implemented or executed**. Host NVIDIA rendering
matches 52/52 sampled old frames, but the old renderer identity is not
authenticated and 28 other branches were not checked. This contract is
therefore explicitly a new, current-executor observation stream, not a claim
that the historical rendering environment has been recovered.

## Fixed sources and decision clock

Use all 32 ordered branches (four roots 8343, 8345, 8349, 8351; raw/spectral
arms, candidates 2–5) in the existing sealed
`pc_causal_mixed_rejected_physical_20261103_DEV01` receipt and NPZ. Pin their
hashes, fixed checkpoint, config, dataset action/proprio normalizers, camera,
scene, source code, compiled MuJoCo extension, runtime and renderer identity.
The reset state is supplied only to the environment. Replay each branch's
saved raw actions[0:12] using one command per bridge call. Capture and seal
the executor-generated RGB[0:13] and actions before optimizing; record a
separate comparison to old images but never substitute old images as new
feedback. Generate the single and three contexts from the same sealed
executor-generated images; for t=2,7,12 use causal RGB position and the
preceding 0.05s image difference. At t=12, one arm uses only t=12; the
other uses t=2,7,12 and actual normalized outgoing packets actions[2:7] and
actions[7:12]. The candidate packet is actions[12:17], bound to t=12.
Both predict t=17. RGB goal and its proprio projection use the same fixed
goal image, camera and scoring kernel as the old cp12 refresh.

## Optimization and execution

Two independent CEM instances per branch use identical seeds, prior,
sampling, score, bounds and budget. The initial prior is raw zero for all
five commands, normalized once; raw domain is [-1,1] for each coordinate.
Use three populations of 16, four elites, then compare prior and final mean:
50 scored packet samples per arm/branch. Both arms use the same pre-generated
Gaussian noise indexed by branch, iteration, sample and coordinate; model
feedback differs only in context history. Update mean and sample standard
deviation from the *bounded scored candidates*, with std floor 1e-5; final
ties choose prior. This explicitly avoids the old kernel's projected-score /
unprojected-update ambiguity. Save noise, projected candidates, scores,
elite indices, means/stds, chosen normalized and raw actions. Require
normalized/raw round trips and model parameter hashes before/after.

Seal all 64 decisions and their provenance before opening any tail outcome.
For each arm reset and replay the fixed 12-action prefix through the same
renderer; require all 13 RGB frames to match the sealed new-stream context
exactly, then execute that arm's five chosen raw commands and save RGB[13:18].
No physical state, distance, old cp17 outcomes or candidate-specific future
RGB may enter the optimizer or determine whether/how to choose a tail.
After all decisions and actions are fixed, an isolated evaluation stage may
replay the complete sequences with the audited position-only backend, report
position distance to the fixed evaluation goal [1.8,2.8], paired outcomes,
and separately label any wrapper threshold. Physical values remain strictly
evaluation-only; no adaptive reselection.

## Mandatory preflight, fail-closed gates and budget

Before planning, verify actual active renderer from a process in the same
environment; record GL vendor/renderer/version and device, not merely the
compiled GPU extension. Require current host NVIDIA path, fixed expected
source hashes, all 32 source rows and valid finite action domains. Require
repeat replay exactness within the newly generated stream, action/context
clock tests with batch size 16 and 2, and one-frame parity with the existing
wrapper. Check historical frames as a separate diagnostic; if any differ,
do not relabel this stream as a historical reproduction. If the current
new-stream replay itself mismatches, the renderer cannot be identified, a
source/model hash or context/action binding fails, the budget differs, the
model returns nonfinite outputs, an output already exists, or execution
fails, stop and mark the run incomplete; never silently use saved frames,
skip branches, retry outcomes selectively, or change tolerance.

Counts to record, including failed attempts: planning-stage 32 prefix
replays = 384 environment commands; 64 execution replays = 1,088 commands
including 320 tail commands; optional separate endpoint replays = 1,088
commands; total new online RGB commands 1,472. Each arm scores 1,600
candidate packets (3,200 total), plus shared goal/context encoding and any
fixed parity checks, which must be counted separately. Encoding counts and
wall time, host/GPU synchronization and failed calls must be recorded, not
only candidate counts. Three-frame context has additional encodes and
longer-sequence compute, so equal CEM samples do not imply equal total
compute. If live implementation counts differ, fix and predeclare the
revised protocol before running, rather than retrospectively relabeling.

This is a 32-branch, four exposed-root development comparison with shared
prefixes and prior adaptive exposure, not an independent estimate of
generalization or closed-loop benchmark success. Single-frame parity is
not an independent real-model reference for the three-frame path. Checkpoint
training provenance, 0.05s finite-difference approximation, rendering
clock and position-backend equivalence remain explicit limits.
