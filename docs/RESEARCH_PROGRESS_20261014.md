# LeWorldModel research progress — 2026-10-14

This is the current evidence-bounded progress record for the public repository.
It distinguishes implementation, protocol validation, partial evidence, negative
results, and blocked claims. Generated checkpoints and raw datasets are not
committed; each result below points to the reproducible runner or compact
result artifact in the repository.

The latest execution audit is recorded in
`docs/EXECUTION_REPORT_20261014.md`. The restricted shell still reports the
host NVIDIA user/kernel library mismatch, but a host-permission CUDA probe and
loader test now pass. A fresh capped rich CUDA protocol is running; no GPU
experiment was rerouted to CPU or promoted from a partial artifact.

## Current scientific object

The project is organized around finite-horizon **decision-transfer sufficiency**:

\[
G_h^\Phi = D_\Phi A_\Phi^{h-1} B_\Phi,
\]

with action identifiability, action-support coverage, task readout, mode
observability, and an independently calibrated error envelope. Alignment,
commutator, Schur geometry, resolvent, and transient amplification are
diagnostic components; none is presented as a universal transfer certificate
on its own.

### Horizon-conditioned memory repair

The fresh five-split audit in
`le-wm-main/results/horizon_conditioned_repair_20261014/summary.json` now
separates memory-family selection from deployment. At `h=1`, the fit-only
selector falls back on all five splits; stability ranges from `0.375` to
`0.5625`, so fixed Fourier is the declared policy. At `h=4`, the frozen
`Fourier+phase` policy beats the fixed-Fourier carrier on all five splits, with
mean paired RMSE difference `-7.0122` and bootstrap 95% CI
`[-8.5417,-5.7109]`. The selector remains unstable on two of five splits and
is never deployed. This is a positive horizon-conditioned transfer-sufficiency
diagnostic, not evidence for action ranking, pose closure, or online planning.

## Completed implementation and evidence

### PointMaze decision-transfer benchmark

The P0/P1 benchmark is implemented under
`le-wm-main/experiments/decision_benchmark/` and includes train-only affine
task-readout centering, root-disjoint candidate banks, independent
`+axis/-axis/zero` action-identifiability audits, finite-difference action
Jacobian diagnostics, shared-bank counterfactual regret, causal history
features, and RGB temporal successor verification.

The corrected RGB successor bridge reports 640 candidate pairs over 40 roots;
realized RGB successor prediction RMSE is `0.1631` versus `0.2011` for the
source-latent baseline (relative improvement `17.8%`). This is successor
prediction evidence, not yet closed-loop online planning.

The paired history audit is deliberately mixed: direct four-frame latent
concatenation worsened successor RMSE by `+0.00834` with root-bootstrap 95% CI
`[+0.00175,+0.01505]`, while the causal history task-effect adapter reduced
held-out selected regret from `0.1897` to `0.0798`. Selected success remained
`0.125` in the small panel. History is therefore useful as a calibrated
action-effect/task feature, not a universally better successor representation.

### Cross-environment matched transfer diagnostic

The final quality-matched PointMaze/PushT audit reports 2 environments, 4
architectures, 5 seeds, and 7 identifiable matched strata. Mean
`Delta R (MSE-only - full)` is `+0.05336`, with paired bootstrap 95% CI
`[+0.01150,+0.11661]`; the stratum sign-flip p-value is `0.0630`, so the
predeclared positive gate is **not established**.

Artifact:
`le-wm-main/results/cross_environment_transfer_diagnostic_final_quality_20261014/`

The point estimate is encouraging, but alignment/geometry is not yet a
generally validated planner-risk predictor.

### PushT contact-mode intervention

The contact-stratified paired intervention uses equal-norm `+a/-a/zero`
actions, root-disjoint splits, four explicit strata, and horizons `1,2,4`:

`le-wm-main/results/pusht_contact_stratified_paired_intervention_20261014/`

Contact geometry helps mainly in free-space and approach states. In
normal-contact and tangential-contact states, action-effect errors and
`+a/-a` separation remain large and do not improve consistently. The main
boundary is within-mode action-effect identification, not merely missing
contact-mode features.

### PushT signed-torque audit and protocol repair

The first radial-contact audit produced an apparently strong direction score
but failed closed-loop pose success. Inspection showed that its center-radius
root and center/vertex normal were not a valid contact chart for the asymmetric
T-shaped object. The audit was repaired to use the transformed polygon-boundary
contact point and boundary inward normal without changing the train/test seed
split or labels.

The corrected boundary-contact audit reports signed direction accuracy `0.609`
(`[0.453,0.750]`), classifier angular regret `0.0151`
(`[0.0045,0.0307`), geometric angular regret `0.0265`
(`[0.0155,0.0398]`), closed-loop angle improvement `+0.124 rad`
(`[−0.0779,+0.3237]`), and strict pose success `0/32`.

Artifact:
`le-wm-main/results/pusht_signed_torque_audit_boundary_corrected_20261014/`

This is a privileged simulator-state signed-effect diagnostic, not visual
LeWM planning. It shows an action-ranking repair but not stable pose closure.
The separate terminal-pose root construction still yields zero accepted roots
under its declared RGB support gate; this remains a protocol boundary rather
than being replaced by radial roots.

## Data and reproducibility boundary

The PushT HDF5 dataset was recovered from the declared Hugging Face mirror
after the previous local compressed file failed Zstandard integrity checking.
The decompressed dataset is local-only and intentionally not committed.
`REPRODUCIBILITY.md` remains the source of dataset/checkpoint requirements.

## Six-goal status

| goal | current status | supported wording |
|---|---|---|
| alignment predicts planning universally | `partial / gated` | conditional matched diagnostic with a positive but not gate-passing cross-environment point estimate |
| adapter stably closes pose | `negative / partial` | torque/action ranking can improve under a corrected privileged contact chart; stable pose closure is not established |
| latent has physical causal dynamics | `partial` | bounded action-associated intervention structure; complete physical causality is not established |
| frozen LeWM supports RGB online planning | `blocked` | RGB successor verification is implemented; pure closed-loop planning remains incomplete |
| universal world model | `not established` | the current target is a universal diagnostic protocol, not a universal learned model |
| commutator is a universal certificate | `rejected` | commutator is one gauge-aware diagnostic term in a joint, calibrated, fail-closed certificate |

The interrupted visual MPC smoke run is not counted as an experiment result.
No result in this document uses privileged simulator successors at runtime or
uses held-out planning outcomes for metric selection.

## Lightweight verification

The newly added decision-benchmark, GDD-WM, RGB-verification, and PointMaze
protocol tests pass together: `20 passed`. The complete data-dependent suite
remains dependent on the local checkpoints and PushT dataset described above.

### Decision-sufficiency gate implementation

The literature-driven three-layer protocol is now executable through
`experiments/empirical_spectral_audit/decision_sufficiency.py`. It requires
predictive error, action-ranking regret, and an independent planning
consequence in one paired record. It fail-closes when prediction improves
without ranking improvement, when mode accuracy improves without pose closure,
or when a panel contains any failed record. The focused regression covering
this interface and adjacent horizon/schema/adapter paths passes `51 tests`.
This is a protocol result, not a positive learned-model result.

### ATM-to-HTSC task-contrast implementation

`experiments/decision_benchmark/task_contrast.py` now implements the next
decision-level object rather than another global score: root-wise pairwise
task contrast error, pairwise ranking inversion, calibration-root maximum
contrast envelopes, and conditional `permit`/`probe`/`abstain` decisions.
The certificate is only for a declared fixed candidate bank and exchangeable
calibration roots; support failure or missing bounds abstains. The focused
regression for ATM, IAEI, PDTC, and the existing sufficiency protocols passes
`19 tests`. This validates the implementation contract, not the empirical
coverage or closed-loop benefit of the certificate.

The first existing PointMaze shared-bank run now supplies an offline task
contrast artifact. Across 8 outcome roots and 960 within-root candidate pairs,
the predicted-vs-real task contrast inversion rate is `0.4078` and mean
contrast error is `0.1081`. Eight independent calibration roots give a
root-level maximum contrast bound of `0.4283`. Because the bank lacks RGB
realized latents and independent support-coverage calibration, the operational
certificate abstains on all roots (`released_root_count=0`). This is evidence
of the ATM-to-task gap and a fail-closed protocol result, not an online
planning gain.

### ATM-style cross-domain audit

The new post-hoc runner
`experiments/empirical_spectral_audit/run_atm_cross_domain_audit.py` was applied
to the current 20,000-row export at
`results/full_spectral_transfer_cuda_downstream_final2_20261014/atm_cross_domain_audit.json`.
The inverse-risk matrix (probe train domain × test domain) is

```text
[[0.5660, 0.5703],
 [0.6499, 0.6124]]
```

The effect-only matrix is

```text
[[0.4154, 0.4265],
 [0.4630, 0.4456]]
```

The state-only action-prior risk is `1.7577`, so this export does not show the
specific state-only shortcut counterexample. The predicted-probe transfer gap
is `0.0375`, while the true-probe gap is `0.0043`; this is a model-level
diagnostic only. The export has no root/candidate/task-cost identifiers, so the
runner correctly reports
`unavailable_missing_candidate_root_identifiers` for finite-horizon action
ranking. No candidate-wise planning claim is made from this matrix.

### RGB-to-task decision bridge: positive successor prediction, negative decision release

The new runner
`le-wm-main/experiments/decision_benchmark/audit_rgb_task_contrast.py` joins the
same 640-candidate bank with the realized RGB successor report, fits the task
readout on train roots only, calibrates a simultaneous root-level contrast
bound on eight disjoint calibration roots, and reports selective
risk--coverage--regret on the eight outcome roots. The artifact is
`le-wm-main/results/decision_benchmark_candidate_bank_20261014/rgb_task_transfer_sufficiency_audit.json`.

The RGB successor gate is complete for all 640 rows and all eight outcome roots
pass the fixed train-only coordinate support gate. Nevertheless, outcome
pairwise inversion remains `0.4258` against realized RGB task costs and
`0.4235` against the privileged post-hoc physical task costs. The calibration
root-level contrast bound is `6.4728`, so no outcome root has a separated
predicted margin and the release count is `0/8` (coverage `0.0`).

This isolates the current negative result: the earlier `17.7%` RGB successor
error improvement does not survive the task readout and candidate ranking
operation. The task readout amplifies residual latent error and the resulting
ranking is not decision-sufficient. The repair is therefore a real fail-closed
policy (`probe`/`abstain` when the bound is not separated), not a relaxed
certificate threshold. Frozen RGB online planning remains blocked until a
fresh, independently calibrated short-pulse/probe protocol produces positive
risk--coverage--regret on held-out roots.

### Candidate-collapse diagnosis and stage/parameterization intervention

The RGB/task audit now records within-root candidate spread, spread ratio,
pairwise predicted/realized variation correlation, and action-to-variation
correlations. The old coordinatewise action gate remains visible as a
diagnostic, but release uses a train-root-only standardized joint nearest-
neighbour support radius over the complete action-block sequence.

On the task-effect checkpoint, all 40 roots exhibited candidate collapse:
predicted/realized spread ratios were approximately `0.03--0.08`, with mostly
near-zero predicted/realized variation correlation. This explains why lower
aggregate RGB successor error did not yield a useful candidate ranking. The
audit is
`le-wm-main/results/decision_benchmark_candidate_bank_20261014/rgb_task_transfer_sufficiency_audit_joint_support.json`.

The matched frozen-backbone intervention gives:

| checkpoint | held-out RGB successor vs source | physical inversion | outcome spread ratio | collapse gate |
| --- | ---: | ---: | ---: | ---: |
| task-effect stage 005 | prior positive gain | `0.411` | `0.050` | `0/8` |
| ordinary MLP stage 005 | `+13.7%` | `0.239` | `0.175` | `0/8` |
| ordinary MLP stage 075 | `-6.2%` | `0.230` | `0.889` | `5/8` |
| standardized MLP stage 005 | `+14.7%` | `0.217` | `0.054` | `0/8` |

The stage-075 result is a useful boundary: restoring candidate variation while
degrading successor accuracy is not a planning repair. Candidate diversity
and candidate correctness are separate gates, and no current checkpoint is
released because calibrated contrast coverage remains `0/8`.

The new
`le-wm-main/experiments/decision_benchmark/train_rgb_candidate_successor_adapter.py`
fits a fixed five-seed residual adapter using train-root realized RGB
successors only. On the ordinary stage-005 bank its held-out outcome
projection RMSE is `0.2065`, but physical inversion is `0.442`; direct
counterfactual RGB supervision alone does not recover task ranking. The
current fail-closed chain is therefore:

```text
joint action support -> candidate variation -> task-contrast calibration
-> independent physical consequence
```

Failure at any gate produces `abstain`/`probe`; no threshold is relaxed using
outcome roots.

### History-4 temporal-interface repair and controlled RGB adapter comparison

The previous history-4 RGB verifier still encoded a two-frame context even
though the candidate bank declared four causal windows. The verifier now
derives its temporal contract from the bank metadata: the ordinary bank keeps
the pair [source_frame-5, source_frame], while history-4 uses
[source_frame-15, source_frame-10, source_frame-5, source_frame]. The
terminal readout is now the explicit index
context_length + predicted_horizon - 1, and the actual indices are recorded
per root. The candidate-bank generator also records the corresponding
history-dependent context length.

The repaired GPU artifact is
le-wm-main/results/decision_benchmark_candidate_bank_20261014/rgb_successor_history4_temporal_repaired/.
All 40 roots and 640 candidate pairs have realized RGB successors; the
renderer/source consistency gates pass for every root. On the eight
predeclared outcome roots, the RGB successor PCA RMSE is 0.14927 versus
0.19879 for the source baseline, a 20.83% relative improvement. This is
successor-interface evidence only, not a planning release.

To test whether the extracted history is usable rather than merely
reconstructive, train_rgb_candidate_successor_adapter.py now accepts
--history-windows. A five-seed train-root-only history-4 adapter obtains
outcome RMSE 0.1810, compared with 0.1915 for the same-bank single-source
adapter (a 5.5% relative improvement). The paired task-level audit gives
RGB contrast inversion 0.4473 for history-4 versus 0.4656 for single-source,
but privileged physical inversion is 0.5178 versus 0.4801. Candidate-collapse
gates pass on 32/40 roots for history-4 versus 27/40 for single-source, yet
calibrated selective release remains 0/8 because the root contrast bound is
much larger than every decision margin.

This is a useful positive-but-bounded result: causal history improves visual
successor prediction and candidate diversity, but the gain does not yet
transfer to physical action ranking. The immediate bottleneck is now the
RGB-to-task readout/action-effect geometry, not temporal context extraction.
The release rule therefore remains fail-closed.

### PushT local observed-effect repair and paired probing (2026-10-14)

The privileged PushT controller previously reused an observed action effect
only when the next pulse landed in the same exact local action bin.  This
made online identification nearly unusable as the contact point moved.  The
implementation now adds a fixed local descriptor (polygon edge, edge
fraction, normal/tangent action components, action norm, and moment sign), a
bounded same-edge neighborhood aggregate, and an observed task-cost-gain term
for adaptive re-ranking.  It remains fail-closed outside the declared chart
and does not read held-out terminal outcomes.

The CUDA smoke artifact is
`le-wm-main/results/negative_cause_diagnosis_20261014/local_neighborhood_smoke_v2_8x8_seed20261014/`.
It preserved the frozen strict protocol and produced mode accuracy `0.9837`,
but adaptive online pose success was `0/8`, final XY `79.45`, and online
identified use was only `4.17%`.  Thus local neighborhood reuse alone did not
solve the sparse/support and common-drift problem.

The next preregistered repair alternates opposite moment-sign probes and uses
their local contrast to remove common angular drift.  The CUDA artifact is
`le-wm-main/results/negative_cause_diagnosis_20261014/paired_contrast_smoke_8x8_seed20261014/`.
It reduces adaptive final XY to `35.26` versus `79.45` for the neighborhood
arm, lowers predicted/observed effect error to `0.286`, and raises online
identified use to `10.07%`.  However pose success remains `0/8` and final
angle is `1.545 rad`; a terminal one-step pulse rerun gives the same adaptive
result in
`paired_contrast_terminal_short_smoke_8x8_seed20261014/`.

This is a genuine partial positive consequence: paired short probes improve
translation while contact boundary loss remains `0/8`, but they do not close
orientation.  The evidence therefore narrows the remaining failure to
within-mode orientation/action identifiability under hybrid contact, not
successor-mode classification or candidate count.  The privileged controller
still cannot be promoted to stable pose closure or frozen RGB planning.

The first neighborhood rerun
`local_neighborhood_smoke_8x8_seed20261014/` is retained as an incomplete
artifact because the numerical run finished but JSON serialization failed on
an ndarray; it is not used as evidence.  The serializer and regression tests
were repaired before the v2 artifact.

### Independent orientation-response audit

The new `experiments/audit_pusht_orientation_response.py` isolates the
remaining orientation interface with a same-root zero/+moment/-moment probe.
The corrected 32-root CUDA artifact is
`results/negative_cause_diagnosis_20261014/orientation_response_full_32root_cuda_20261014/`.
It has 32/32 valid equal-norm opposite-moment pairs and a paired angular
separation of `0.09297 rad` (95% CI `[0.06867, 0.11803]`). However, sign
consistency is `0.75` (CI `[0.59375, 0.875]`) and successor-mode agreement is
`0.625` (CI `[0.4375, 0.78125]`). The result is therefore a partial
mechanistic positive: local orientation response exists, but it is not a
stable within-mode transfer law across edge/fraction strata. The orientation
controller gate remains closed; the next experiment must use calibration-only
support gating with independent roots.

The calibration-only gate is implemented in
`experiments/calibrate_pusht_orientation_gate.py` and evaluated on the frozen
orientation artifact with a root-disjoint split. It authorized only one
edge/fraction stratum: 18 calibration roots produced 14 evaluation roots, but
only 2 evaluation roots were released. Those two roots had evaluation sign
consistency `1.0`, successor-mode agreement `1.0`, and paired separation
`0.1076 rad` (95% CI `[0.0718, 0.1434]`). This is a local mechanism positive
with low coverage; the gate is not a general orientation certificate and does
not open pose-closure or RGB-planning claims.

### Edge-stratified independent replication (2026-10-15)

The `random_boundary` sampler overrepresented edge `0:0`, and the coarse
center-distance mode label rejected valid stem contacts. The audit now includes
an `edge_boundary` protocol: edge and fraction are sampled first, the accepted
nearest polygon edge is checked after reset, and continuous gap—not the coarse
mode label—defines geometric validity. The mode label is retained only for
successor analysis.

Three CUDA runs use four roots per edge/fraction target, root-disjoint
calibration/evaluation splits, and fixed thresholds. They cover all eight
polygon edges with zero pairwise root-state digest overlap:

| seed | roots | sign consistency | separation (rad) | mode agreement | authorized strata | released evaluation roots | released sign/mode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261015 | 103 | 0.592 | 0.06513 | 0.893 | 6 | 12 | 0.917 / 1.000 |
| 20261016 | 100 | 0.640 | 0.06958 | 0.890 | 6 | 12 | 1.000 / 1.000 |
| 20261017 | 99 | 0.646 | 0.06514 | 0.869 | 8 | 15 | 1.000 / 1.000 |

This resolves the earlier root-construction/seed-overlap confound, but does not
turn the mechanism into a universal law. Pooled sign consistency is below the
fixed `0.75` release threshold and only 6–8 strata per seed are authorized.
Stable cross-mode orientation transfer, learned pose closure, frozen RGB online
planning, and universal world-model claims remain unestablished.
