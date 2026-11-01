# Archive action discrimination v1 — prospective fixed calibration

Only this source, corresponding tests and this innerdoc are the delivery write-set.
No upstream source/result is edited. MAIN audits and runs HOST CUDA; worker tests
are RAM/saved metadata only. Tests passing does not establish scientific success.
The pre-existing blocked personal goal is not completed or replaced by this task.

## Fixed authority and inputs

`p1_archive_target_overlay_fidelity_cuda_20261027_v1_cuda_main01/report.json`
has SHA `d8cbd4b6e8ff1497421e6cdaf8644f4b39ed2cd2b695bb10cd15b57fbae121f0`.
The source constants also bind its exact original prepare, source and MAIN independent
receipt. `validate_old()` re-verifies producer prepare/report/journal/physical PNG
identity via the old saved-only verifier, every old artifact, 30 image/pixel/latent
receipts, 24 requests and their normalization, all 156 actual native module entries
and full tensor edges, all original NPY records and original analysis/journal.
Model/config/normalizer loaded-asset source bindings are inherited unchanged.
There is no claim of fresh roots or verified training lineage.

Episodes `[1,6229]`, conditions `[archive,nativeoff,nativeon]`, windows
`[H1_10_to_15,teacher_forced_15_to_20]` are fixed in this order: 12 bases.
Each has ordered arms `[factual,zero_outgoing,reverse_time_outgoing,swap_xy_outgoing]`.
All 48 remain in the report, including failed/not-started rows. No fresh-H2 replay,
deduplication, padding, retries, outcome selection or new visual encoding.

Original context is respectively z0/z5/z10 or z5/z10/actual-z15, full float32
`(3,192)`; raw is `(3,5,2)`. Only the last raw5 packet is changed: literal float32
zeros, reverse the five time rows, or exchange x/y columns. Earlier raw10 and all
latent history remain bitwise identical. Reconstruct all three normalized tokens
once using the original float32 mean/std and reshape `(3,10)`; no /100, clipping,
tanh or new normalizer. Reconstructed historical tokens must also match bitwise.

The backend receives only identity plus these three arrays and `arm`; no goal,
state, successor label or coverage. Every factual output, all three projected
tokens `(3,192)`, must bitwise equal its original saved output. A single differing
bit aborts the job, retaining output and all 48 statuses; no tolerance or retry.

## Prediction seal and arithmetic

All 48 original-precision full3 output NPYs are fsynced before the durable
`all48_predictions_sealed_before_labels` event. Only then does the scoring label
loader run. Provenance validation necessarily decodes old labels to verify the
sealed source; this is distinct from scoring and labels are never passed into
the predictor. No literal claim that provenance checking never reads label bytes.

Same-condition actual z15/z20 is the only scoring target for each arm. Save all
48 float64 mean192 successor MSEs and persistence MSEs; all 36 nonfactual−factual
MSE gaps, full3 and last-token output-change MSEs; all 24 native-minus-archive gap
contrasts (2 episodes × 2 windows × 3 nonfactual arms × 2 native conditions).
Positive gap means factual predicts its factual label closer than that altered
input; no winner selected. These are successor-discrimination diagnostics, NOT
counterfactual physical truth, controllability, control gain or a fitted gate.
Conditions are paired, not independent samples. No significance test or P1 admission.

## Actual dispatch accounting / resources

New job: 48 `predict` API and 48 each action_encoder/predictor/pred_proj actual
calls and batch-one samples; 144 projected token samples. API and submodules are
separate axes, not 192 independent predictions. Encoder/projector/AD/physics=0.
Old 30 encoder/projector and 24 prediction calls are inherited and separately
reported, not new work or free new evidence.

Subclass the original `NativeTrace` with own 48-call predispatch counters and
prediction-only scopes; attach module hooks and the same instance-level predict
observer, restoring both on exit. No old source/global/contract monkeypatch.
Save every module full input/output tensor leaf plus kwargs and nested durable
intent/input/return events. Consume checks all edges, native call order, complete
attempt-return census and hash-chain chronology. Full NPYs are losslessly gzipped;
compressed file, original NPY and raw tensor SHAs are independently checked.

HOST CUDA only, original frozen model hash, inference/no-AD guards, MATH SDPA,
matmul/cudnn TF32 off, offline loading; no CPU fallback or training. Output cap
32 MiB includes final report/journal, with 2 MiB final reserve; disk floor 200 MiB,
initial free space at least 232 MiB. Prepare/strict JSON each cap 2 MiB. Runtime
wall 600 s from Store creation (after prepare verification), including model load,
prediction/scoring and final source verification. A +5 s allowance permits only
final failure preservation, never a successful >600 s run. Peak CUDA allocated
3 GiB/reserved4 GiB. All failures/partial rows kept, no automatic restart.

The old captured image hidden state is `(1,257,192)`, not197; decode actual saved
shape and take the original CLS edge only for provenance. No new image call or
hidden-token shape assumption enters the prediction backend. Approximate native
prediction tensor payload is <3 MiB uncompressed for all192 ledger entries plus
request arrays; actual cap remains enforced on all writes, no precision reduction.

## MAIN commands (inner repository root)

```bash
.venv/bin/python -m pytest -q tests/test_agent_archive_action_discrimination_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_archive_action_discrimination_20261028_v1 prepare --prepared results/agent_archive_action_discrimination_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_archive_action_discrimination_20261028_v1 verify --prepared results/agent_archive_action_discrimination_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_archive_action_discrimination_20261028_v1 run --prepared results/agent_archive_action_discrimination_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_archive_action_discrimination_20261028_v1_cuda_MAIN01 --confirm-host-cuda
.venv/bin/python -m experiments.spectral_reachability.agent_archive_action_discrimination_20261028_v1 consume --prepared results/agent_archive_action_discrimination_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --report results/agent_archive_action_discrimination_20261028_v1_cuda_MAIN01/report.json --report-sha ACTUAL_REPORT_SHA --output results/agent_archive_action_discrimination_20261028_v1_consume_MAIN01.json
```

Use actual generated SHAs, never these placeholder strings as credentials. NEW
exclusive paths, no overwrite. `consume()` is also a pure saved-return Python API
with `(prepared, prepare_sha, report, report_sha)`; CLI persists its strict receipt.
Do not reuse old consume statuses as new success. Pending actual execution evidence:
MAIN prepare/verify, HOST CUDA all48/factual bit gate and actual saved consume.

## RAM regression coverage

Mutation chronology, literal-vs-normalized zero, illegal fields/windows/dtypes,
bounded single-member gzip, all48 seal-before-labels, full192-entry nested IO graph,
independent scalar recomputation, factual one-bit rejection and partial bytes,
predispatch caps, trace cleanup, lossless NPY/SHA/trailing guards, duplicate JSON,
disk/cap/no-overwrite and complete semantic prepare binding. Numpy protocol fixtures
are explicitly synthetic, never substitute production results or run neural code.
