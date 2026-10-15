# LeWorldModel execution report — 2026-10-14

This report records the execution state after applying the review criteria. It
does not promote an experiment to scientific support merely because a command
started or produced partial artifacts.

## GPU preflight and failure boundary

- Host device: NVIDIA GeForce RTX 5060 Ti was previously visible to PyTorch
  (`torch 2.12.0+cu130`, CUDA tensor probe passed).
- Current host state: `nvidia-smi` fails with `Driver/library version mismatch`;
  kernel module is `595.84` while the visible user-space NVML library is
  `595.99`.
- Five consecutive fresh-process CUDA allocations subsequently failed with
  `RuntimeError: No CUDA GPUs are available`, despite the stale
  `torch.cuda.is_available()` result. The current learned GPU protocol is
  therefore **environment-blocked**, not CPU-completed.
- No learned GPU experiment was silently rerouted to CPU.

### Unified decision-transfer rerun on 2026-10-14

The required fresh GPU-first probe was repeated before the unified P0/P1
post-hoc audit. In the current process environment, PyTorch reports
`torch 2.12.0+cu130`, `cuda_available=false`, and `device_count=0`; a real
CUDA allocation is therefore environment-blocked. The post-hoc audits below
do not run learned inference and were allowed to use CPU. This state is
recorded in
`le-wm-main/results/unified_decision_transfer_p0p1_20261014/unified_manifest.json`.

The fresh PointMaze task-contrast audit reproduces `0.4077976818` inversion
over 960 within-root pairs and a calibration bound of `0.4282840914`. The
HTSC preview remains fail-closed because support and realized RGB successor
data are unavailable. The fresh ATM audit reproduces the inverse-risk matrix
and explicitly reports ranking as unavailable without candidate/root/task-cost
identifiers. The learned-panel audit remains gated to one identifiable
stratum/one environment, and the existing two-environment quality-matched
audit remains positive-point-estimate-only with its preregistered gate not
established.

This rerun closes the unified baseline bookkeeping; it does not open GDD-WM,
HTSC release, frozen RGB online planning, or any universal-model claim.

The follow-up D0/D1 implementation and full regression completed after this
baseline: `671 passed, 3 skipped, 26 warnings`. The new D1 audit is covered by
dedicated tests and does not alter the earlier evidence boundary.

### Host-permission recovery on 2026-10-14

The restricted process continued to fail CUDA initialization, but a
host-permission probe confirmed that the device node and runtime are usable
from the host: a real tensor allocation succeeded on an NVIDIA GeForce RTX
5060 Ti, and the compatibility loader placed `pusht/lewm` on `cuda:0`. A fresh
rich export then completed 1,536 rows with the declared multi-horizon and
counterfactual fields.

The capped recovery protocol is running in a fresh directory:
`le-wm-main/results/full_spectral_transfer_cuda_recovery_20261014/`. It is not
yet scientific evidence until all stages and the final manifest complete. The
earlier blocked directories remain blocked artifacts.

## Code repair applied

`le-wm-main/experiments/io_utils.py` now loads named legacy LeWM checkpoints
through the compatibility key-remap path instead of first invoking the
incompatible upstream loader. CUDA initialization is kept GPU-first and raises
on a genuine device failure. The compatibility loader was observed placing the
model on `cuda:0`. The fail-closed probe is implemented in
`le-wm-main/experiments/gpu_preflight.py`; targeted tests and the
host-permission CUDA probe pass. Learned entry points now default to explicit
`cuda` and do not silently fall back to CPU.

The repair is a runtime robustness fix; it is not evidence that the learned
model has better dynamics or planning.

## Main protocol runs

| Run | Execution state | Evidence state |
| --- | --- | --- |
| `results/full_spectral_transfer_cuda_review_20261014` | stopped after 6,160 rich-export batches; resumable chunks retained | partial / interrupted |
| `results/full_spectral_transfer_cuda_review_capped200k_20261014` | theory stages completed; latent export blocked by CUDA initialization | blocked |
| `results/full_spectral_transfer_cuda_review_capped200k_loaderfix_20261014` | theory stages completed; latent export blocked after the same host failure | blocked |
| `results/gpu_batch_probe_256_rich_20261014` | 10-batch CUDA rich export completed | smoke/protocol validation only |
| `results/gpu_recovery_probe_20261014` | host-permission CUDA rich export completed, 1,536 rows / 1,000 saved | smoke/protocol validation only |
| `results/full_spectral_transfer_cuda_recovery_20261014` | capped GPU protocol running; chunks retained | running / not yet evidence |

The successful probe verified the intended rich schema, including
`dynamics_h3/h5`, predicted and residual horizons, physical state, goals,
counterfactual fields, episode identifiers, and timestep. It is not a full
dataset result.

### Latest GPU-first check and GDD-WM v5 continuation

Before the v5 post-hoc provenance audit, `nvidia-smi` was re-run in a fresh
process. It exited with code 18 (`Failed to initialize NVML: Driver/library
version mismatch`; visible NVML `595.99`). This continuation did not run
learned inference and used CPU only for JSON/NPZ validation. No learned GPU
experiment was silently rerouted to CPU.

The v5 continuation is at
`results/gdd_wm_d0_d1_audit_20261014_v5/`. The new join utility copied the
candidate bank, verified exact report-to-input provenance, and joined all
640/640 realized RGB projection references without changing teacher scores,
real costs, splits, actions, or the original bank. D0 passed. D1 has zero
missing runtime references but remains `rejected_fail_closed` because (i) no
root/candidate pair has repeated candidate-set contexts and (ii) the declared
offline teacher score is exactly negative real cost. This is a repaired
support gate, not a valid teacher or distillation result.

After the join, non-leaking teacher-control audit, and fixed feature-family
panel, the full project regression completed with `677 passed, 3 skipped, 26
warnings`.

### Relative action-effect controls and current negative-result diagnosis

After a fresh GPU-first check, no learned inference was run because NVML still
reported the driver/library mismatch. CPU was used only for the train-only
JSON/NPZ audit. The relative-effect panel was rerun over five frozen banks with
the original root-disjoint splits and 5,000 paired root-bootstrap draws. The
implementation adds full source/delta-action interactions, train-only
interaction PCA ranks 2/4/8/16, and within-root pairwise ridge ranking.

The full 188-dimensional interaction lowers train fit error but does not yield
a stable held-out outcome gain. Low-rank controls reduce capacity without
producing a uniformly positive consistency/regret interval. Pairwise ranking,
which directly targets candidate order rather than absolute task cost, also
has bank-dependent point estimates and CIs crossing zero. The evidence
therefore rejects three immediate repairs: more unconstrained interactions,
PCA compression alone, and a regression-to-ranking objective swap.

The current cause is narrowed to inadequate cross-root action-effect support
or action identifiability in the frozen latent/candidate bank; it is not
legitimate to claim that the interaction model solved transfer. The artifact
is `results/gdd_wm_relative_action_effect_panel_20261014/panel.json`.
Because each bank has only eight outcome roots, this remains a diagnostic
boundary with wide uncertainty. No D-JEPA teacher, D2 distillation, adapter,
or planning claim is opened by this run. The complete regression after these
changes is `681 passed, 3 skipped, 26 warnings`. After adding the
cross-case aggregators and their regression tests, the complete project suite
is `687 passed, 3 skipped, 26 warnings`.

### Root-dependent action-bank repair

The new cross-root audit found that the existing bank's metadata claim of a
shared candidate bank was false at the raw action-sequence level: the same
candidate id had root-specific actions. The raw pair flip fraction was
`0.6917` on outcome roots for the final MLP bank and `0.9750` for the history-4
bank. This explains why raw action summaries did not transfer consistently.

The relative-effect runner now canonicalizes actions into each root's
goal-relative frame. On the existing records, three non-history banks show
strictly positive outcome bootstrap intervals for both consistency gain and
regret reduction; the history-4 and history-near-goal banks remain CI-crossing.
The candidate generator also has a `--canonical-action-bank` mode for a clean
future export. No learned export was rerun because the required GPU-first
preflight remains blocked by the NVML mismatch. The current positive result is
therefore an offline protocol repair with privileged source-state/goal
canonicalization, not RGB planning or a universal transfer claim.

### Fresh CUDA canonical-bank replication

Host-permission GPU preflight passed (`torch 2.12.0+cu130`, real allocation on
RTX 5060 Ti), so learned inference was run on CUDA for two matched frozen
checkpoints. The new canonical banks each contain 640 records over 40 roots
with 24/8/8 train/calibration/outcome roots. A generator repair bounds the
rotated controls; the final artifacts have one canonical bank hash and
maximum raw control magnitude `0.99865`.

The corrected cross-root audit canonicalizes raw simulator actions before
checking sharedness: both banks pass `shared_action_bank=true`, while real-cost
pair order still flips for `0.675` of outcome pairs. Using `source+delta` as
the fixed baseline, fresh goal-frame action features improve outcome
consistency by `+0.381` `[+0.240,+0.513]` for the MLP checkpoint and `+0.350`
`[+0.223,+0.466]` for the standardized checkpoint. Paired regret reductions
are `+0.122` `[+0.051,+0.188]` and `+0.149` `[+0.082,+0.204]`. These are
held-out offline action-effect diagnostics; D1 remains closed because the
teacher is an offline cost oracle and no independent D-JEPA preference
contexts or realized RGB successor references are present.

The protocol was extended to seed `8412` for both MLP variants. The four-case
aggregate uses a separate case (`architecture x seed`) uncertainty layer and
does not count every root-by-case cell as an independent model. The
goal-frame variant is positive in all four cases: case-macro consistency gain
`+0.2784`, two-level case/root bootstrap 95% CI `[+0.2110,+0.3442]`; regret
reduction `+0.0739`, CI `[+0.0415,+0.1110]`. The aggregate artifact is
`results/gdd_wm_relative_action_effect_panel_20261014/canonical_2arch_2seed_aggregate.json`.

This resolves the first reproducibility question for the repaired offline
diagnostic: the gain is not confined to seed `8411` or one MLP parameterization.
The scientific boundary is unchanged. The benchmark uses privileged
goal-frame/source-state canonicalization and simulator task costs, so it is
not yet D-JEPA validation, causal-dynamics evidence, frozen RGB online
planning, or a universal-world-model result.

The 10 canonical architecture-by-seed cases were also evaluated on independent
outcome roots with the frozen RGB successor verifier (`8 roots x 16 candidates`
per case). All renderer, realized-RGB-latent, and PCA-comparison gates passed.
The aggregate action-conditioned successor error improvement over the
source-latent baseline is `+14.21%`, with two-level case/root bootstrap 95% CI
`[+8.47%,+19.69%]`; the predicted successor is better for `66.56%` of rows,
CI `[59.77%,73.05%]`. Artifact:
`results/decision_benchmark_candidate_bank_20261014/rgb_successor_canonical_2arch_5seed_aggregate.json`.
This is realized RGB successor evidence, not closed-loop planning or causal
world-model evidence.

To address the remaining raw-action architecture condition, the feature-family
panel now includes a train-only latent-to-source-XY probe and an estimated
goal-frame action family. Relative to `source_plus_delta`, this family gains
`+0.0985` held-out consistency (two-level case/root CI `[+0.0534,+0.1449]`)
and reduces candidate regret by `+0.0546` (CI `[+0.0176,+0.0888]`). It is
positive on 9/10 architecture-by-seed cases; standardized seed `8413` remains
negative (`-0.0249`). This is a partial representation repair, not a stable
universal certificate or closed-loop planning result.
Notably, standardized seed `8413` has positive realized RGB successor
improvement (about `14.39%`) but negative estimated-frame candidate regret
reduction (`-0.0249`), directly preserving the prediction-versus-decision
failure boundary.

A train-only successor-calibration negative control was then run on this
failure case. It reduced train RGB successor RMSE to `0.1628`, but increased
outcome `source_plus_delta` regret from `0.0872` to `0.1503`; calibrated raw
action and estimated-frame readouts also failed to recover the decision
result. Generic prediction calibration therefore does not solve the task
geometry bottleneck.

The same protocol was expanded to all five available seeds (`8411`--`8415`)
for both MLP variants. Across 10 architecture-by-seed cases, every case has
40 roots and the goal-frame variant remains positive. The case-macro
consistency gain is `+0.2777` with two-level case/root bootstrap 95% CI
`[+0.2356,+0.3212]`; regret reduction is `+0.0751`, CI
`[+0.0455,+0.1073]`. The weakest case has regret reduction `+0.0182`.
Artifacts:
`results/gdd_wm_relative_action_effect_panel_20261014/canonical_2arch_5seed.json`
and `canonical_2arch_5seed_aggregate.json`. This is still an offline
action-effect diagnostic with privileged goal/source canonicalization, not
independent RGB planning or causal-dynamics evidence.

## Existing evidence carried forward

The review matrix remains the scientific source of truth:

- matched transfer diagnostic: positive point estimate but preregistered
  sign-flip gate not passed;
- RGB successor prediction: improved prediction, no closed-loop planning proof;
- PushT contact/action intervention: within-mode action-effect identification
  remains the bottleneck;
- strict visual pose closure and universal certificate claims remain open or
  negative;
- GDD-WM remains proposed/not validated until teacher provenance,
  geometrizability, student-generated candidates, and independent planning
  consequences are executed.

## Latest PushT loss-support repair and matched pilots

The current GPU preflight was performed first. `nvidia-smi` reported
`Failed to initialize NVML: Driver/library version mismatch`, and the project
environment reported `torch.cuda.is_available() == false` with zero CUDA
devices. The following visual pilots therefore used explicit `--device cpu`;
this was a recorded environment fallback, not an untested CPU preference.

The repair in
`le-wm-main/experiments/run_causal_visual_pusht_world_model.py` masks
`block_effect`, `orientation_transition`, `orientation_effect`, and the
explicit wrapped-angle effect loss by `contact_mode > 0`. A unit test verifies
that changing a non-contact target cannot change the orientation loss. Targeted
compatibility/serialization regression is `18 passed`.

The same matched protocol was run for all pilot variants in
`le-wm-main/results/causal_visual_pusht_*_cpu_pilot_20261014/`. The strongest
intermediate result is the contact-masked angle-effect-only variant:
paired action-motion top-1 `0.6875` versus prior torque baseline `0.5625`,
contact action-motion top-1 `0.875`, paired effect error `13.0741`, but paired
orientation top-1 `0.0` and visual strict success `0/2`. Thus the repair
improves translation/effect ranking without solving within-contact pose
closure; it cannot be promoted as a stable positive planning result.

The 75% contact-root and 25% terminal-root interventions were both negative,
so increasing contact quantity or adding terminal roots is not sufficient.
The mode-conditioned all-loss variant reached contact orientation top-1
`0.75` but reduced overall motion top-1 to `0.1875`, identifying shared
mode-routing/gradient interference as a second failure mechanism. The RGB
geometry-servo control from the best checkpoint ran for 16 episodes and had
strict success `0/16`; it is an interface diagnostic only.

The privileged boundary-contact short-pulse benchmark with observed-sign
calibration was stopped after a long CPU candidate-scoring loop before any
result was written. It is excluded from all claims and remains a queued
performance optimization target, not a negative scientific result.

## Newly isolated artifact failure in PushT repair/MPC

The earlier `metrics: null` files from the contact-predictor/MPC family are
not interpretable as model failures.  `predict_latent_sequence` was decorated
with `torch.enable_grad()`, so it overrode the caller's `torch.no_grad()` in
evaluation.  The MPC runner then attempted to convert a graph-attached tensor
to NumPy and exited after creating only its initial protocol artifact.  The
function now respects the caller's autograd context, and the regression test
checks both no-grad evaluation and differentiable training paths.  The MPC
runner also writes atomic progress checkpoints after predictor repair and each
policy, so a future interruption is represented as an incomplete run rather
than a null-metric result.  This repairs the experiment protocol; it does not
yet establish positive pose closure or online planning.

## CPU-safe validation completed during the GPU block

- The initial repository suite recorded during the GPU block passed: `629 passed, 3 skipped`;
  the post-governance regression is recorded below as `632 passed, 3 skipped`.
- The refreshed oral-level learned-panel aggregator completed with
  `panel_gate=blocked`, `strict_matched_status=gated_single_environment`, and
  only one identifiable matched stratum. It does not establish the required
  multi-architecture, multi-environment conditional effect.

## Resume condition

After the NVIDIA user/kernel libraries are made version-consistent (or the host
is rebooted into the matching driver), the restricted-shell CUDA path should be
rechecked. The host-permission recovery run may continue in the meantime. Do not reuse the blocked
run as positive evidence unless its input lock and all stages validate.

## Governance repair validation

## Literature-driven protocol implementation

The predictive/action/decision sufficiency framework is now implemented in
`experiments/empirical_spectral_audit/decision_sufficiency.py` and documented
in `docs/DECISION_SUFFICIENCY_RESEARCH_PLAN_20261014.md`. Its hard gates reject
the two recurring false positives: prediction improvement without action-
ranking improvement, and mode-accuracy improvement without pose closure.
The focused regression across this gate, horizon memory, enriched schema, and
RGB adapter paths passes `51 tests`. This does not promote any learned-model
artifact to positive planning evidence.

The next ATM-to-HTSC layer is implemented in
`experiments/decision_benchmark/task_contrast.py`: it audits pairwise task
contrasts, calibrates a root-level maximum contrast envelope, and returns
`permit`, `probe`, or `abstain`. Its focused ATM/PDTC/sufficiency regression
passes `19 tests`. No coverage or closed-loop claim is promoted until fresh
calibration roots and RGB-only planning are run.

The ATM-style post-hoc audit was also run on the current 20,000-row export.
Its inverse-risk matrix is `[[0.5660, 0.5703], [0.6499, 0.6124]]`, with
effect-only matrix `[[0.4154, 0.4265], [0.4630, 0.4456]]`. The export lacks
root/candidate/task-cost identifiers, so finite-horizon ranking is explicitly
`unavailable_missing_candidate_root_identifiers`; these numbers are not a
candidate-wise planning certificate.

After the review criteria were converted into runtime contracts, the following
checks passed:

- explicit claim-matrix evidence manifest, checksum rejection, unified manifest
  and runtime-state tests: `11 passed`;
- full repository regression suite: `632 passed, 3 skipped`;
- Bash syntax check and dry-run side-effect check: passed; dry-run did not create
  its requested output directory.

These are software/reproducibility results only. They do not promote the
blocked GPU run, the PushT pose-closure result, or any universal planning claim.

## Latest CUDA PushT mechanism runs

The host-permission PyTorch preflight was rerun before the new runs and passed
on `NVIDIA GeForce RTX 5060 Ti` (`torch 2.12.0+cu130`, `cuda_available=True`,
real CUDA tensor allocation). The NVML query still reports a user/kernel
library mismatch, but no run below was routed to CPU.

The local observed-effect neighborhood smoke completed with mode accuracy
`0.9837`, adaptive final XY `79.45`, pose success `0/8`, and online identified
use `4.17%`. The paired local-contrast smoke completed with final XY `35.26`,
predicted/observed effect error `0.286`, pose success `0/8`, and final angle
`1.545 rad`. A terminal one-step pulse rerun left the adaptive metrics
unchanged. These are privileged PushT mechanism results: the translation
improvement is real but does not pass the pose-closure gate.

The first neighborhood run computed its metrics but failed final JSON writing
because a NumPy array remained in a trace payload. The serializer was repaired
and the v2 artifact is the only version used above.

The independent orientation-response audit then completed on the same host
GPU. The expanded artifact is
`le-wm-main/results/negative_cause_diagnosis_20261014/orientation_response_full_32root_cuda_20261014/`.
It passed CUDA analysis, equal-norm intervention, opposite-moment, and shared
edge-chart checks for all 32 roots. The paired angular separation was `0.09297
rad` (95% CI `[0.06867, 0.11803]`), while sign consistency was `0.75`
(`[0.59375, 0.875]`) and successor-mode agreement was `0.625`
(`[0.4375, 0.78125]`). This is a partial orientation-effect mechanism result;
the calibration/support gate remains closed and no pose or RGB-planning claim
is promoted.

The calibration-only orientation gate was then run on the frozen 32-root
artifact with a root-disjoint split. It authorized one edge/fraction stratum,
releasing 2 of 14 evaluation roots. The released roots had sign consistency
and successor-mode agreement `1.0`, with paired separation `0.1076 rad` (95%
CI `[0.0718, 0.1434]`). The low coverage keeps this as a partial local
mechanism result; it does not promote pose closure or RGB planning.

### Edge-stratified independent orientation replication

The host-permission project virtualenv CUDA preflight passed on an RTX 5060 Ti
(`torch 2.12.0+cu130`; real CUDA tensor allocation). The NVML command still
reports a driver/library mismatch, but all three audits completed on CUDA and
record `cuda_analysis_completed=true`.

The earlier random-boundary protocol was edge-biased, and its center-distance
mode filter rejected valid stem contacts. The new `edge_boundary` runs cover
all eight polygon edges and have zero exact root-state digest overlap across
seeds:

| seed | response roots | sign consistency | separation (rad) | mode agreement | authorized strata | released roots | released sign/mode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261015 | 103 | 0.592 | 0.06513 | 0.893 | 6 | 12 | 0.917 / 1.000 |
| 20261016 | 100 | 0.640 | 0.06958 | 0.890 | 6 | 12 | 1.000 / 1.000 |
| 20261017 | 99 | 0.646 | 0.06514 | 0.869 | 8 | 15 | 1.000 / 1.000 |

Gate thresholds were frozen before evaluation (`n_calibration >= 2`, sign and
mode agreement `>=0.75`, separation `>=0.02 rad`, antisymmetry RMSE `<=0.06
rad`). This repairs the previous protocol confound and establishes a
reproducible local angular response, but still fails global orientation
transfer. It does not open pose closure, learned RGB planning, or
universal-world-model claims.

## Indexed canonical-action repair

The train-only latent-to-source-XY chart was tested as a direct explanation of
the standardized seed `8413` failure. The linear chart had fit-root RMSE
`0.4156`; a quadratic chart reduced training error to `0.0022` but increased
outcome regret from `0.1121` to `0.2092`, so the higher-capacity chart is
recorded as an overfit negative control.

The implemented repair is
`experiments/gdd_wm/audit_indexed_canonical_action_geometry.py`: reconstruct
the fixed goal-relative action summary once from train-root representatives,
index it by the shared `candidate_id`, and never estimate a test-root action
frame from the test latent. The panel runner is
`experiments/gdd_wm/aggregate_indexed_canonical_action_panel.py`.

The 10-case artifact is
`le-wm-main/results/decision_benchmark_candidate_bank_20261014/indexed_canonical_action_2arch_5seed_aggregate.json`.
All 10 cases are positive against `source+delta`: outcome consistency gain
`+0.2845`, two-level case/root bootstrap 95% CI `[+0.2425,+0.3275]`; regret
reduction `+0.1090`, CI `[+0.0820,+0.1372]`. The former standardized-8413
negative case is repaired (`+0.0708` regret reduction). The train-derived bank
reconstructs all 16 candidate ids and the all-root shared-bank diagnostic has
maximum deviation `6.9e-16`.

This closes the specific offline raw-action/latent-frame reconstruction
failure. It is not yet an independent RGB-only planning result: the task
readout still uses simulator-labeled costs, and no closed-loop planner has
been released. D-JEPA preference, causal dynamics, pose closure, and
universal-world-model claims remain gated.

## Independent RGB-compatible task audit

The follow-up aggregate is
`results/decision_benchmark_candidate_bank_20261014/rgb_indexed_canonical_task_2arch_5seed_aggregate.json`.
It joins the frozen RGB successor reports to the same 2-architecture x
5-seed candidate panel. The RGB readout and indexed canonical action bank are
fit on train roots only; calibration and outcome roots are root-disjoint and
use only the immutable candidate-id lookup.

The outcome consequence is positive in all 10 cases. Indexed canonical action
versus RGB source-plus-delta gives execution-consistency gain `+0.2845`
(two-level case/root bootstrap 95% CI `[+0.2425,+0.3275]`) and candidate
regret reduction `+0.1090` (CI `[+0.0820,+0.1372]`). Standardized seed 8413
is positive at `+0.0708` regret reduction. The `standardized-8413` train and
calibration reports were resolved through their canonical-named artifact
aliases because the indexed-named aliases were not emitted; the report keeps
this path substitution explicit.

Regression coverage now includes a synthetic RGB task-selection fixture that
checks root-disjoint train-only action-bank transfer and rejects failed RGB
successor gates. The result is still an offline RGB-compatible task readout
with simulator labels, not closed-loop visual online planning or causal
physics evidence.

## RGB-only visual world-model continuation

The next visual branch was executed after the GPU preflight failed closed
(`nvidia-smi` could not communicate with the driver and PyTorch reported
`cuda_available=False`), so these pilots explicitly use CPU rather than
silently claiming CUDA execution.

The first end-to-end CPU smoke of
`experiments/run_causal_visual_pusht_world_model.py` completed data generation,
training, held-out counterfactual evaluation, and one visual-only planning
episode after repairing a JSON serialization bug caused by NumPy values in the
planning trace. The smoke artifact is
`results/causal_visual_pusht_rgb_only_smoke_20261014/`; it reports paired
orientation top-1 `0.0`, angle-delta correlation `0.0543`, and strict success
`0/1`.

On the same train/validation roots, the contact-frame/torque pilot adds
temporal context, contact-frame dynamics, signed torque features, explicit
angle heads, and visual contact gating. It obtains action-motion top-1
`0.5625` and angle-delta correlation `0.1610`, but orientation top-1 remains
`0.0` and strict success is `0/2`.

The matched all-loss/soft-mode intervention is a negative control rather than
a selected winner: action-motion top-1 drops to `0.0625`, angle-delta
correlation is `-0.1717`, and mode top-1 is `0.1420`. This identifies a
support/interface conflict in the current contact-mode supervision and keeps
the orientation-loss combination out of the formal protocol.

The JSON conversion fix is covered by
`tests/test_causal_visual_world_model_serialization.py`; the existing PushT
MPC tests also pass. None of these pilots establishes frozen-LeWM pose closure,
universal causal dynamics, or a successful RGB online planner.

### PushT goal-pose label repair and CUDA verification

The subsequent host-permission CUDA run used the project virtual environment
(`torch 2.12.0+cu130`, RTX 5060 Ti) and exposed a target-label mismatch in the
visual PushT runner. `info["goal"]` is rendered from `info["goal_pose"]`, but
the dataset builder previously derived target orientation from
`info["goal_state"]`, whose angle is the sampled reset state rather than the
rendered target angle. This made the goal-angle readout task intrinsically
unidentifiable.

The runner now carries an explicit `goal_pose_vector` through source roots,
terminal roots, bootstrap roots, counterfactual labels, manifests, and the
planning evaluator. Current and goal explicit angle heads are separate; the
new `--explicit-goal-angle-readout` flag controls only the goal image head.
The project-environment targeted tests pass (`5 passed`).

The matched repaired pilot improved goal-pose readout RMSE to `9.334` and
goal-angle readout RMSE to `0.245`, confirming the diagnosis. It still failed
the downstream mechanism gate: angle-delta correlation `-0.571`, contact
action-motion top-1 `0.000`, paired orientation top-1 `0.000`, and strict
planning `0/2`. The protocol defect is fixed, but the learned visual
within-contact dynamics and pose closure remain open; this result is not
promoted to a planning claim.
## Goal-pose label audit and CUDA continuation

The host CUDA runtime was verified (`RTX 5060 Ti`; PyTorch CUDA available),
so the next visual pilots were run with the project virtual environment and
`--device cuda`. A source audit found that the runner used
`info["goal_state"]` as the supervision target for `info["goal"]`. In PushT,
the rendered target is defined by `info["goal_pose"]`; `goal_state` is the
simulator reset/evaluation state and can carry a different sampled block
angle. This made goal-angle readout statistically unidentifiable.

`goal_pose_vector()` now propagates the rendered target through source roots,
counterfactual labels, split manifests, and the planning evaluator. The
current and goal explicit angle heads are also independently configurable.
The targeted regression suite passes (`5 passed`).

The corrected pilot artifact is
`le-wm-main/results/causal_visual_pusht_gpu_goalposefix_20261014/`. Goal-pose
RMSE fell to `9.334` and goal-angle RMSE to `0.245`, compared with the prior
`43--58` and `0.54--1.00` ranges. This confirms a genuine protocol bug, but
the same run still has contact orientation top-1 `0.0` and strict planning
`0/2`.

Two matched CUDA follow-ups,
`causal_visual_pusht_gpu_goalposefix_currentgoalreadout_20261014/` and
`causal_visual_pusht_gpu_goalposefix_angleeffectonly_20261014/`, reduce
goal-angle RMSE further to `0.143` and `0.059`, respectively. They still have
contact orientation top-1 `0.0` and strict success `0/2`. The repaired target
protocol therefore removes one cause of the negative results but does not
close current-pose/within-contact action identifiability.

## Contact-support and runtime-interface continuation

Host-permission GPU preflight was rechecked before each learned pilot: RTX
5060 Ti, 16 GiB, driver `595.99.02`, and a real CUDA allocation under
`torch 2.12.0+cu130` succeeded. No learned run in this continuation was
silently rerouted to CPU.

The rich temporal routing pilot (`...richtemporal_routed_20261014`) did not
close planning: contact action-motion/orientation top-1 was `0.25/0.25`,
angle-delta correlation `-0.589`, and strict success `0/2`. A separate fixed
RGB mask-moment angle-head control showed that the signal is present in the
input: held-out contact angle RMSE was `0.184 rad`, with `81.25%` within 15
degrees.

The runner consequently adds contact-root angle weighting, angle-head
warmup/freeze, and block-only invariant angle features. The weighted and
warmup small pilots remained negative. With `256/128` roots, contact
action-motion top-1 reached `0.3125`, but orientation top-1 was `0.125` and
strict success `0/4`. The formal `1600/240` invariant run reached contact
action-motion top-1 `0.375`, orientation top-1 `0.0417`, and strict success
`0/8`.

The same formal checkpoint under frozen RGB geometry state readout plus visual
contact gating produced `1/8` on evaluation seed `20261014` and `0/8` on an
independent seed `20261015`. Task-space ranking reproduced `1/8`; a separate
phase-utility formal checkpoint produced `0/8`. These are interface/mechanism
diagnostics, not stable planning gains. The current claim boundary remains:
larger contact support improves offline translation/effect transfer, but
action-conditioned orientation transfer and closed-loop contact retention are
unresolved.

## Contact-interface continuation (2026-10-15)

Host-permission CUDA was verified before the new learned runs: RTX 5060 Ti,
PyTorch `2.12.0+cu130`, and a real CUDA allocation succeeded. The new RGB
interface chart was implemented without changing the old pose-chart dimensions;
old-checkpoint compatibility remained `21 passed`.

The full interface-context run
(`causal_visual_pusht_gpu_contactinterface_full_20261015`) reported contact
action-motion/orientation top-1 `0.275/0.0583`, angle-delta correlation
`0.0402`, and visual strict success `0/8`. An explicit lever/action
interaction pilot was lower (`0.1875/0.03125`). The isolated current-only
explicit angle-head pilot was also negative (`0.09375/0.0`, angle-delta
correlation `-0.266`, no planning episodes).

These results reject three simple repairs: adding boundary context alone,
letting an MLP learn the lever/action product, and removing temporal
contamination from the angle head. They sharpen the diagnosis to a
deployment-support problem: RGB contains pose, but the within-contact
action-effect map is not stable across candidate contact interfaces. The next
experiment must use observed short-pulse effect calibration with a predeclared
support/fail-closed gate. These artifacts remain mechanism diagnostics; no
learned causal-dynamics, pose-closure, or stable RGB-planning claim is opened.

## RGB analytic pose and re-observation continuation (2026-10-15)

宿主权限 GPU 预检确认 RTX 5060 Ti（16,311 MiB）可用；本批次所有张量分析和
runtime observer 均使用 `--device cuda`，没有静默回退 CPU。

新增入口：`experiments/audit_rgb_analytic_pose.py`、
`experiments/audit_rgb_reobserved_action_effect.py`，以及
`run_visual_geometry_servo.py --analytic-pose-calibration`。同时修复了
`MaskMomentGeometryEncoder._interface_features` 中 `[y,x]` 与 `[x,y]` 的
坐标顺序错误。

正式 2,048-root pose audit 的 current-scene test 为 83.0% within 15°，contact
test 为 85.1%，但 current calibration 直接迁移到 goal image 仅 4.9%。独立
goal calibration 在本 PushT 固定目标角协议上达到 100%，所以 goal-domain
错配是一个真实接口原因，但不能代表多目标泛化。双校准器接入 geometry
servo 后仍为 0/8；修复 support 坐标后仍为 0/8，说明 action-to-effect
identifiability 仍是主瓶颈。

256 contact roots、8-action fan 的 RGB re-observation 得到 angle effect
88.1% within 15°，但 re-observed task top-1 23.05%，held-out global effect
adapter 18.75%。短脉冲观测提供了局部 verification signal，却没有证明
全局 action-effect predictor 或 online planning 成功。

## Mode-aware verification-gate continuation (2026-10-15)

The complete privileged PushT held-out run completed on CUDA after batching
candidate-bank forest calls and root-grouped effect-gate calibration. The
artifact is
`le-wm-main/results/negative_cause_diagnosis_20261015/mode_aware_verification_gate_64x32_seed20261015_final/`.
It uses 64 train roots, 32 held-out roots, eight shared actions, 128 trees,
root-disjoint collection, paired effect probes, observed-sign and
contact-geometry feedback, uncertainty gating, and terminal handoff.

Held-out mode accuracy is `0.969832`. Mode-aware pose success is `0.09375`
with bootstrap CI `[0, 0.21875]`; ranking regret is `0.096452`, not lower than
the global one-step baseline `0.095099` or contact-chart ranking `0.095207`.
Adaptive online correction has `0.125` pose success, effect error `0.085970`,
and contact loss `0.71875`; uncertainty gating has `0.09375` pose success,
contact loss `0.59375`, and root fallback `0.8125`. The safe fallback has
`0.0625` pose success and `0.5` contact loss.

The result separates the mechanisms: successor-mode identification is already
high, while within-contact action-effect transfer and contact retention remain
insufficient. The predicted/observed effect improvement is not an independent
planning consequence. Learned physical causality, stable adapter pose
closure, frozen RGB online planning, universal world model, and universal
commutator certificate remain closed.

The source repair is execution-only: batched forest inference for candidate
banks and root-grouped effect-gate calibration. No feature, label, split,
estimator, threshold, or controller rule changed. Pilot and final artifacts
are kept separately so the 8-root pilot is not treated as the main estimate.

## Distance-aware online support pilot (2026-10-15)

The online observed-effect cache was tightened to include pusher-to-boundary
distance in both its exact key and local-neighborhood distance. This prevents
reusing a contact pulse after geometric separation. The CUDA pilot is at
`le-wm-main/results/negative_cause_diagnosis_20261015/mode_aware_distance_support_64x8_seed20261015_pilot/`.

It did not repair the mechanism: mode-aware pose success is `0/8`, adaptive
online remains `1/8` with contact loss `0.875`, and uncertainty-gated pose
success is `0/8`. Ranking is unchanged from the previous pilot. The cache
alias was therefore a real implementation risk but not the dominant cause;
nonstationary within-contact action effects and insufficient action support
remain the active diagnosis.

## Orientation-gate cross-seed continuation (2026-10-18)

Added `experiments/aggregate_pusht_orientation_gate.py` to aggregate the
calibration-only orientation response across seeds with CUDA root/seed
bootstrap. It reads mechanism artifacts only. The three-seed intersection is
4/10 strata and release coverage is 24.5%--31.9%; released sign consistency
is 0.972 and successor-mode agreement is 1.0.

Added a fail-closed `--orientation-gate-artifact` path to
`run_pusht_mode_aware_shared_candidate.py`. Unauthorized edge/fraction strata
execute the declared inward fallback and are reported separately. The CUDA
smoke control at
`le-wm-main/results/negative_cause_diagnosis_20261015/orientation_gated_control_smoke_20261018/`
used root-disjoint train/held-out roots and the same candidate bank for all
selectors. Orientation-gated, mode-aware, and contact-chart pose success were
all `0/8`; the orientation gate blocked 89.1% of contact steps and every root
fell back at least once. The gate reduces unsupported action release but does
not solve within-contact action-effect/angle transfer.

## Paired local-effect probe smoke (2026-10-18)

The predeclared paired opposite-moment online probe was run separately on
CUDA, using re-observed transitions only. The artifact is
`le-wm-main/results/negative_cause_diagnosis_20261015/paired_effect_probe_smoke_20261018/`.
The adaptive-online branch reached pose success `2/8`, predicted/observed
effect error `0.123`, and wrong-sign events `1.0`, but contact loss remained
`0.50` and uncertainty-gated pose success was `1/8`. This is a local
mechanism improvement over uncalibrated transfer, not a stable planning
result; a larger multi-seed gate remains required.

## Contact-retention metric repair (2026-10-24)

The formal paired-probe rerun found that the earlier contact-loss number used the wrong geometry for the `boundary_contact` root protocol. A legal edge contact can have pusher-to-block-centre distance above 80 px; the correct runtime contact predicate is polygon boundary gap above 70 px. The runner and CUDA panel aggregator now report this protocol metric and retain the centre-distance statistic only as a compatibility diagnostic. Existing artifacts remain unchanged.

The new three-seed panel is at `le-wm-main/results/negative_cause_diagnosis_20261015/contact_metric_repair_panel_20261024/`. Among 48 held-out roots, all variants have physical contact loss `0/48` with bootstrap CI `[0,0]`, whereas the centre-distance diagnostic remains nonzero. Adaptive paired probing has effect-error delta `-0.0696` (`[-0.0884,-0.0541]`) and wrong-sign delta `-3.9583` (`[-5.5625,-2.4583]`) versus global one-step, but pose-success delta is only `+0.0417` with CI `[0,0.1042]` and final-angle delta crosses zero. The measurement bug is fixed; stable pose closure and frozen online planning remain unverified.

## Terminal orientation handoff and execution-order repair (2026-10-24)

The first terminal-handoff artifact was invalid because the adaptive reranker
overwrote the terminal action after selection. The runner now gives terminal
handoff priority and exposes `--terminal-max-moment`. In the corrected CUDA
artifact `terminal_max_moment_metric_repair_seed20261020_v2/`, physical
contact loss is `0/16`, effect error is `0.0433`, but pose success remains
`2/16`, final angle is `1.748` rad, and the stronger pulse causes frequent
fail-closed probing. The execution bug is fixed; stronger terminal magnitude
alone is not a pose-closure repair and was not promoted to three seeds.

## Frozen contact-frame interface replication (2026-10-15)

The next predeclared experiment changed the action-effect support/interface,
not the held-out selector or terminal threshold. It used the real polygon
boundary-contact protocol, a contact-frame physical chart, paired opposite-
moment probes, observed sign calibration, online local-effect correction,
uncertainty gating, and terminal handoff. All three runs used the host CUDA
environment (`torch 2.12.0+cu130`, NVIDIA GeForce RTX 5060 Ti); the simulator
itself is CPU-bound, while Ridge fitting and aggregation use CUDA.

The three-seed artifacts are under
`le-wm-main/results/negative_cause_diagnosis_20261015/contact_frame_boundary_panel_seed20261020/`,
`...seed20261021/`, and `...seed20261022/`. The frozen aggregation is
`le-wm-main/results/negative_cause_diagnosis_20261015/contact_frame_boundary_panel_20261015/`.

The adaptive online arm reduces predicted/observed effect error by `0.04497`
(95% CI `[-0.05848,-0.03224]`) and wrong-sign events by `3.0625` (95% CI
`[-5.4167,-0.9375]`) relative to the global one-step arm over 48 paired
held-out roots. Physical polygon-boundary contact loss is `0/48` for every
reported arm. This confirms a stable local identification benefit and fixes
the earlier metric confound between legal edge contact and centre distance.

The downstream pose gate still fails: adaptive pooled pose success is only
`0.0833` (95% CI `[0.0208,0.1667]`), the paired success delta is `+0.0417`
with CI `[0,0.1042]`, and final-angle delta is `+0.0292` rad with CI
`[-0.0576,0.1158]`. Thus this continuation does not establish stable pose
closure, frozen RGB planning, or physical causality. It localizes the negative
result to the transfer from locally identified one-step effect to a
multi-step within-contact angle policy; further static feature stacking or
terminal-pulse tuning is not justified by this panel.

## Horizon-conditioned short-pulse visual MPC (2026-10-15)

To test the horizon explanation directly, I ran a frozen three-seed panel
with prediction horizon 4 and one-step action blocks. The protocol kept the
visual state source, stable-contact bootstrap, staged 64-candidate proposals,
96/24 train/held-out cases, 8 held-out episodes per seed, and identical
base/repaired candidate proposals. CUDA was verified before execution and
used by the frozen encoder/predictor and aggregator.

The aggregate artifact is
`le-wm-main/results/pusht/lewm/h4_shortpulse_visual_mpc_panel_20261015/`.
Neither base nor repaired visual MPC achieved task success (`0/24`). The
repaired-minus-base paired consequence was `0` success gain, `+16.26 px`
block XY distance (95% CI `[-16.86,50.63]`), and `+383.68` final distance
(95% CI `[100.84,738.25]`). Thus h=4 short-pulse control is a negative
intervention: shorter prediction horizon alone does not transfer predictor
repair into visual planning. The evidence points to candidate action-support
and contact-interface mismatch, not merely autoregressive horizon length.
