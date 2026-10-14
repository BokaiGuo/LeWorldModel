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
