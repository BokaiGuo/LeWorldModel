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
