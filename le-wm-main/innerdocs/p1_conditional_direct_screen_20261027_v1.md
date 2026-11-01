# P1 finite full-H2 direct screen core v1

Engineering core and mock tests only. No CUDA, neural CPU forward, reset,
render, physics, source generation, training, outcome reads or download was
performed for this implementation. P1 admission and model-task eligibility
remain false. Main owns a NEW independently sealed driver and actual CUDA.
The existing admission core/procedural source and all sealed P2 v2 files are
unchanged by this work.

## Read-only interface findings and remaining prerequisites

1. `p1_conditional_procedural_source_20261027_v1` provides finite saved RGB,
   raw zero-padding and reset/source receipts, not native latents. This core
   implements `encode_source` and `native_image_encoder` to bridge actual RGB
   to the existing native projected CLS192. State/physical goal vectors are
   used ONLY inside the old loader's source-root identity validation; neither
   reaches encoder/planner. The task goal is encoded goal RGB, not pose.
2. Old `to_admission_scan` needs true encoded task IDs. Validate plan/exclusions
   and raw SHA before encoding; afterward main must call `to_admission_scan`
   with the COMPLETE root→task mapping to validate original sourcepack journal,
   census, all attempts, first-legal-root selection and exact raw bindings.
   A bridge receipt sets `census_validated=false`; it cannot certify freshness.
3. The currently prepared source purpose is `calibration_exploration`, NOT
   fresh confirmation. The bridge deliberately preserves the old plan's
   readiness semantics: exploration can remain historically incomplete, which
   MUST NOT be promoted to fresh confirmation by an external driver. Three
   repeated current RGBs are `causal_padding_not_observed_transitions`, one
   observed frame, zero public warmup steps; not measured dynamics history.
   Main must retain purpose/exclusion-readiness/reset-internal-operation costs
   from the original sourcepack in its final report, separately from neural work.
4. The old core checks saved direct trajectories but cannot execute them. This
   new core executes genuine incumbent-conditioned continuous optimization;
   it does not inspect any saved candidate bank or transplant historical roots.
5. Independent witness producer and its budget/replay remain required. This
   output is a DIRECT COMPONENT, not an admission completed-row. Do not invent
   producer work, witness actions or producer/witness receipts to fill the row.
6. Independent pre-calibration recipe/threshold authority and confirmation
   chronology are not supplied here. No tau/epsilon/default plateau threshold
   exists in configuration. Algorithm/budgets/normalizer/Q/goal must freeze
   before execution; thresholds cannot be chosen from current physical results.

## Fixed execution semantics (explicit configuration, no chosen hyperparameters)

`validate_configuration` accepts exactly method, B1, B2, initial_logits,
step_scale, damping, extra_line_search_trials. B2>B1>=1; finite predeclared
float32-representable restarts. Both methods form the same full192×20 terminal
Jacobian at EACH incumbent; no prefix detach, H1 decomposition or recompile.

Let r=Q(z-goal), A=QJ. Gradient is 2 A.T r/192. GN uses ONE NumPy thin SVD
of A/sqrt(192) and delta=-V diag(s/(s²+damping)) U.T r/sqrt(192), i.e. ridge
for the MSE linearization plus damping||delta||². Gradient delta=-gradient
uses no SVD/solve. Both use one explicit fixed step_scale and project each
10-dimensional packet into the ORIGINAL task's per-packet logit ball, then
float32 tanh into raw[-1,1]. This is NOT a joint20 ball or a library search.
The tiny inward float32 feasibility rounding is numerical, not a cost threshold.

Every iteration evaluates exactly ONE proposal. Extra line-search trials=0;
there is no retry/backtracking/early stop, not even for rejected or identical
proposals. Strictly lower original Q cost replaces incumbent, otherwise exact
incumbent survives. B1 is a checkpoint in a continuous B2 run, not a restart.
Initial/proposal/derivative/incumbent and all replay contexts are saved.

### Exact direct resource ceiling per task

For R restarts and B=B2, all attempted work is nonrefundable:

| Resource | Exact completed direct work |
| --- | --- |
| predict + action_encoder (each) | R[6(B+1)+2B] |
| full H2 Jacobian (AD forward included above) | RB |
| mathematical VJP directions | 192 RB |
| NumPy thin SVD / ridge solve | RB each for GN, 0 for gradient |
| actual proposal / incumbent selection | RB each |
| extra line-search / candidate-bank evaluation | 0 / 0 |

Each saved prediction pays nominalH2+wholeH2+segmentedH2=6 calls of each.
Each full-J pays 2 more AD-forward calls; 192 scalar reverse directions are
explicitly journaled, not equated to FLOPs or kernel launches. Original core
events retain exact allowed fields/resources; a separate extended ledger pays
VJP/solve/proposal/selection. This is not evidence of equal total compute.

Native image encoding overhead is 4 images (3 padded histories+goal), each
with one encoder and original projector entry. No action encoding in image
encode. Source reset/render (including reset internal work), Q construction,
model/checkpoint loading, source/model hashing, CPU normalization comparisons,
host-device transfers, fsync, synchronizations, JSON serialization and final
driver verification are OUTSIDE direct ceilings and must remain explicit
driver overhead. `pred_proj` is native to `JEPA.predict` (one call on its
flattened predicted sequence per predict); it is not an image projector or
an independently FLOP-matched operation. No measured timing/FLOPs claimed.

## Native backend checks / exact arithmetic boundary

`NativeCuda` rejects non-CUDA before torch import, requires frozen eval CUDA
parameters/buffers, and pins the full task_id. Main must bind checkpoint,
model-config, actual `jepa.py`, image transform, and source closure hashes;
this API does NOT authenticate externally loaded weights from a caller string.
Main must check frozen/eval/unchanged weights before/after, synchronize and
durably finalize exceptions in its NEW driver.

The entry hooks charge actual `model.predictor` and `model.action_encoder`
module entries. Read-only `jepa.py:JEPA.predict` calls predictor once then
pred_proj; `_step` also checks exactly one entry of each per model.predict
invocation. A nonstandard bypass/repeated predictor is rejected, never silently
treated as one predict. Encoder/projector entries are forbidden during direct
optimization. Caps apply BEFORE module dispatch; missing completions are failed
attempts. Hook cleanup never replaces a primary exception.

CUDA torch.tanh is NOT assumed bitwise identical to NumPy float32 tanh. The
admission contract defines raw commands with NumPy float32 tanh. Native nominal,
whole/segmented and derivative forward ALL use these identical commands.
The full joint H2 graph differentiates with respect to these raw20 commands,
with normalization in-graph, then multiplies columns ONCE by (1-command²).
Thus it is a logit Jacobian including the tanh+normalizer chain, not a raw-control
Jacobian relabeled as logit. No detached H2 prefix or straight-through trick.
Float32 normalizer tokens are compared exactly to the old core's NumPy result
BEFORE each neural packet call. Receipt-only detach/copy does not replace the
graph token. AD terminal must equal incumbent terminal exactly; mismatch fails
without tolerance relaxation or extra nominal calls. CUDA determinism/rounding
compatibility is still an unexecuted MAIN verification, not proven by mocks.

## Main-only callable wiring (NOT executed here)

1. Independently freeze all code/task/source/configuration and actual finite
   configuration, budget and storage limit. No numeric algorithm or tau defaults
   are supplied by this doc. Preflight source census hashes, disk headroom and
   model/config/metric/normalizer/image-transform bindings in NEW driver.
2. Main loads existing frozen native JEPA on explicit CUDA; construct
   `DiskJournal(NEW_OUTPUT, byte_limit=EXPLICIT_BYTES)` (>=200MiB reserve).
3. `encode = native_image_encoder(model, device, native_transform, journal)`;
   call `encode_source(raw_npz, source_plan, exclusions, EXPECTED_RAW_SHA,
   task_template, encode)` for all original completed source slots. Template
   contains exactly old task fields EXCEPT root_context/goal/root_raw_action_history.
   `goal_source_sha256` binds contiguous uint8 goal RGB bytes; metric_source SHA
   binds the existing frozen Q producer, not a newly fitted metric.
4. Validate COMPLETE true root/task mapping through old `to_admission_scan`;
   preserve exploration versus confirmation semantics. Only after source/task
   checks, call `run_direct(receipt['task'], configuration,
   NativeCuda(model, device, receipt['task']), journal)` for each allowed task.
   No GPU loading or source generation is triggered merely by importing core.
5. Main checks returned status, primary/secondary errors, ALL operation events,
   strict group budgets and final model/source checks, then writes its actual
   report. Failed source/encode/direct/close/synchronize cannot become completed.
   Assemble old admission row only when separate genuine witness/authority
   prerequisites exist. Even model eligibility does not establish physical P1.

`DiskJournal` exclusively creates a NEW directory and fsyncs each event,
including arrays before later work. AD forward packets/terminal survive later
VJP/SVD failure; completed trajectory steps survive later failures. A write
failure stops work; returned partial report does not promise durability after
storage failure. Main must reserve space for its own finalization and report.

## Pure mock verification command

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_p1_conditional_direct_screen_20261027_v1.py tests/test_p1_conditional_task_admission_20261027_v1.py
```

Tests exercise both genuine direct methods, continuous B1/B2, both-packet chain,
Q/goal/action bindings, exact old direct-budget grouping, reject/incumbent,
predispatch caps, fault partials/first errors, small durable journals and real
raw-source serialization with MOCK source readiness/encoder. These fixtures are
engineering evidence only; no actual fresh source/model efficacy is asserted.
