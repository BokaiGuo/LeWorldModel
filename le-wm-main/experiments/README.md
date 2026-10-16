# LeWM Directional Latent-Space Experiments

This directory implements the PushT-first experiment suite for auditing whether
LeWM latent eigenspaces carry controllable, predictable, and plannable
directions.

The nonlinear planner metric and training ablations are synthetic proxy
experiments.  The fixed-weight and curriculum extensions are stored at
`results/nonlinear_planner_metric_ablation_20261003/multihorizon_weight_sweep_curriculum.md`;
they improve some ID/long-horizon cells but worsen OOD h=2/h=4, so a fixed
schedule does not remove the trade-off.  The deeper state-conditioned extension
is stored at
`results/nonlinear_planner_metric_ablation_20261003/uncertainty_conditioned_horizon_ablation.md`.
It uses a frozen training-only bootstrap disagreement probe plus a permuted
score control; any gains remain synthetic training evidence rather than online
LeWM/PushT control evidence.  In the completed 10-seed run, the probe reduced
OOD rollout MSE slightly relative to its permutation control at h=2, but the
corresponding planning-regret contrasts were not significant after Holm
correction; the direct evidence does not support a unique state-conditioned
dynamics advantage.

The uncertainty-subspace follow-up
`experiments/empirical_spectral_audit/run_uncertainty_subspace_conflict_audit.py`
cross-fits residual-direction energy against covariance, dynamics, predicted
dynamics, planning, nuisance, action-residualized dynamics, and residual
covariance bases.  Its canonical artifact is
`results/pusht/lewm/uncertainty_subspace_conflict_20261003.md`.  It uses
training-only high/low residual thresholds and both label-permutation and
random-basis nulls; the residual norm is an error proxy, not an epistemic
uncertainty estimator.  In the three-split frozen audit, the k=8
nuisance-minus-planning contrast is positive while the direct dynamics and
planning contrasts are near zero; this supports a bounded error-geometry
diagnosis, not a causal nuisance or online-planning claim.

The residual-aware planner follow-up
`experiments/empirical_spectral_audit/run_nonlinear_residual_aware_planner_ablation.py`
turns that diagnosis into a synthetic candidate-ranking test.  It compares
residual precision, residual covariance, and an equal-spectrum random residual
basis against Euclidean, covariance, reachability, and oracle metrics.  Its
artifact is
`results/nonlinear_planner_metric_ablation_20261003/residual_aware_planner_metric.json`;
all residual metrics remain training-only and synthetic.  The completed
10-seed run shows a small OOD long-horizon gain for residual precision, but its
direct comparison with the equal-spectrum random residual basis is not robust
after Holm correction; residual-aware weighting is therefore retained as a
diagnostic lead, not a confirmed planner improvement.

The deeper strength/calibration audit
`experiments/empirical_spectral_audit/run_nonlinear_residual_strength_calibration_audit.py`
adds four residual-weight strengths, an independently randomized basis with
matched trace/spectrum/condition number, and an independent heldout residual
calibration split.  Its artifact is
`results/nonlinear_planner_metric_ablation_20261003/residual_strength_calibration_audit.md`.
For the none model at the OOD shift and h=8, the heldout strength-1
aligned-minus-random regret contrast is -0.00207 with 9/10 negative seed
deltas (focused-family Holm p=0.0234; global all-cell Holm p=1).  This is a
mechanism-consistent but exploratory signal: it is stronger than a mere
condition-number explanation, yet it does not justify a universal planner
claim.  Independent calibration is close to in-sample calibration, so the
signal is not explained by reusing the predictor's training residuals alone.

The candidate-count follow-up
`experiments/empirical_spectral_audit/run_nonlinear_residual_candidate_sensitivity.py`
tests the same residual-aligned/random contrast at 16, 48, and 96 candidates.
Its artifact is
`results/nonlinear_planner_metric_ablation_20261003/residual_candidate_sensitivity.md`.
For heldout calibration at the OOD shift and h=8, strength-1
aligned-minus-random regret is -0.00274, -0.00149, and +0.00035 at candidate
counts 16, 48, and 96, respectively.  The sign reversal at the largest search
set is decisive negative evidence against treating the earlier local signal as
a candidate-count-invariant planner improvement.

The cross-audit evidence summary
`experiments/empirical_spectral_audit/summarize_dynamics_residual_evidence.py`
joins the split-robust frozen readouts, external rollout interventions,
physical-motion probes, residual transport, and synthetic planner controls
without pooling their samples or intervals.  Its canonical artifact is
`results/pusht/lewm/dynamics_residual_planner_evidence_summary_20261004.md`.
The resulting claim matrix supports a frozen latent dynamics-sensitive
subspace and frozen-rollout sensitivity, but does not support added physical
motion predictability beyond actions or a candidate-count-invariant residual
planner gain.

The cross-horizon follow-up
`experiments/empirical_spectral_audit/run_external_dynamics_horizon_transport.py`
transfers the same source-fitted bases to encoded future embeddings from the
external public PushT windows. It measures projection energy for
$z_2-z_1$, $z_4-z_2$, and $z_8-z_4$ with covariance and dimension-matched
random controls. The canonical artifact is
`results/pusht/lewm/external_public_pusht_temporal_20261003/external_dynamics_horizon_transport.md`.
The dynamics basis is positive against both controls for all three source
fits and all three horizon spans, supporting conditional latent-trajectory
transport. This remains an encoded-latent diagnostic: it is not physical
causal dynamics, online planning, task success, or normalization-invariant
evidence.

The action-conditioned follow-up
`experiments/empirical_spectral_audit/run_external_action_residual_horizon_transport.py`
fits a compact action-history readout leave-one-episode-out, subtracts its
prediction from external encoded latent displacement, and re-evaluates the
source-fitted bases. The canonical report is
`results/pusht/lewm/external_public_pusht_temporal_20261003/external_action_residual_horizon_transport.md`.
The dynamics-basis contrast remains positive against covariance and random
controls in all three horizon spans and normalization arms. Because the
compact action probe itself has negative cross-fit R² on the latent target,
this is an action-conditioned latent diagnostic rather than proof of
action-independent physical dynamics.

## Separated Research Code Paths

### Proposed GDD-WM workstream

The proposed Decision Preference → Predictive Geometry workstream is kept
separate from the existing theory and empirical packages. Its first executable
surface is the D0 schema/provenance audit:

```bash
python experiments/gdd_wm/build_shared_candidate_dataset.py \
  --input /path/to/candidate_table.json \
  --output results/gdd_wm_candidate_dataset_v0_1
```

The input must provide the shared candidate fields and explicit provenance
flags. The runner can write canonical JSONL only after candidate-set size,
root-disjoint splits, action support, and leakage checks pass. It never creates
teacher preferences, simulator outcomes, or latent references. A passing D0
artifact therefore has the boundary `schema_audit_only_not_teacher_validation`;
D1 preference stability, D2 projection distillation, and independent planning
remain unrun until a real frozen teacher/data package is supplied.

The current v5 continuation also provides
`experiments/gdd_wm/join_rgb_realized_refs.py`, which can only copy and join
dataset-matched realized RGB references after exact path, key, and value
coverage checks. This repaired the runtime-reference support gap in the
PointMaze bank, but D1 remains fail-closed because the available teacher is an
offline cost oracle with `teacher_score = -real_task_cost` and each root has
only one candidate-set context. It does not authorize D2.

The non-leaking control
`experiments/gdd_wm/audit_nonleaking_teacher_control.py` fits a fixed
train-root-only predicted-latent task readout and evaluates held-out ranking.
Its result is diagnostic only: the current PointMaze bank reaches about
`0.51` outcome pairwise consistency and does not establish D-JEPA preference
stability or planning transfer.

`experiments/gdd_wm/build_train_only_readout_teacher_package.py` is the stricter
two-context infrastructure control. It joins exact root/action pairs from
independent checkpoint stages, preserves RGB successor references, and writes
fresh train-only readout scores instead of copying the offline cost oracle. The
current control passes the package eligibility audit but remains a negative
teacher-quality result (flip rate `0.3081`, held-out execution consistency
`0.5800`); it is not D-JEPA evidence and cannot open D2.

`experiments/gdd_wm/summarize_train_only_teacher_feature_panel.py` compares the
predeclared latent/action feature families with root bootstrap. The action-only
family is retained as a confounding baseline because it outperforms the latent
families on the current PointMaze outcome roots; no family is selected for
deployment from this panel.

The same builder's `--residualize-action` mode performs the stronger
`Delta Risk | action summary` check. The current PointMaze result is negative:
the latent residual does not improve over action-only, so the control cannot
authorize D2 or a decision-geometry claim.

`experiments/gdd_wm/audit_train_only_teacher_capacity.py` is the fixed
capacity follow-up. It keeps `source_delta_action` and the action-only
baseline fixed, fits a train-only standardized linear residual and one fixed
CUDA `[64,32]` SiLU MLP, and evaluates held-out outcome roots. The linear
residual has regret delta `-0.1568` versus action-only (root-bootstrap 95% CI
`[-0.2602,-0.0678]`); the MLP delta is `-0.0631` (CI crosses zero). This is a
capacity diagnostic only and does not open D2. Run it from the project root
as a module so package imports are resolved:

```bash
.venv/bin/python -m experiments.gdd_wm.audit_train_only_teacher_capacity \
  --candidate-dataset <train-only-context.json> \
  --arrays <context-arrays.npz> \
  --output <capacity-audit-dir>
```

For candidate-coordinate and estimand controls, add
`--action-frame goal_relative` and/or `--target-mode root_centered`. The
canonical reruns did not retain the raw-action point estimate, so these flags
are required before interpreting a latent residual as within-root transfer.

The fixed feature-family panel
`experiments/gdd_wm/audit_feature_family_panel.py` compares source, predicted
terminal, predicted delta, action-summary, and combined features with one
train-only ridge protocol. It does not select the best family from held-out
results. The current five-bank audit has no family that is stable across both
calibration and outcome, so concatenation is not promoted as the repair.

The follow-up
`experiments/gdd_wm/audit_relative_action_effect_panel.py` evaluates a
root-centered action-effect estimand over the same five banks. It includes
full interaction, train-only PCA low-rank controls, and pairwise ridge
controls, and writes paired root-bootstrap CIs to
`results/gdd_wm_relative_action_effect_panel_20261014/panel.json`. The full
interaction's lower train error is not matched by stable held-out transfer;
neither low-rank compression nor direct pairwise ranking produces a positive
interval across banks. Treat this as an action-support/identifiability
diagnostic, not as D-JEPA, causal-dynamics, adapter, or planning evidence.

The current PointMaze negative result also exposed a bank-coordinate bug:
`shared_candidate_bank` metadata did not imply identical raw action sequences
across roots. Use the optional generator flag
`--canonical-action-bank` to sample once in a goal-relative coordinate frame
and rotate into each root; the metadata records the frame and the bank hash is
computed after canonicalization. The post-hoc goal-frame readout is positive
on three of five outcome banks, while two history-conditioned banks remain
inconclusive. This is a protocol repair using source state/goal fields, not
evidence for RGB planning or D-JEPA distillation.

The fresh CUDA replication is stored under
`results/decision_benchmark_candidate_bank_20261014/*canonical_goalframe_v1/`.
Both frozen checkpoints pass the corrected shared-bank audit. With
`source+delta` as the fixed baseline, goal-frame action features improve
held-out offline outcome consistency and regret in both checkpoints, with
paired root-bootstrap intervals above zero. This is the first clean positive
repair of the raw-action bank confound, but it remains an action-effect
diagnostic: the teacher/D1 gate, realized RGB successor support, and online
planning evidence are still closed.

The cross-seed/architecture aggregate is written by
`experiments/gdd_wm/aggregate_relative_action_effect_panel.py`. On the four
fresh cases (two MLP variants x seeds `8411,8412`), the case-macro goal-frame
consistency gain is `+0.2784` (two-level case/root CI `[+0.2110,+0.3442]`) and
regret reduction is `+0.0739` (CI `[+0.0415,+0.1110]`). The aggregate keeps
architecture/seed uncertainty separate from root uncertainty and remains
offline diagnostic evidence only.

The panel was subsequently extended to all five available seeds for both MLP
variants (10 case units). The goal-frame gain is positive in every case; the
case-macro consistency gain is `+0.2777` (two-level case/root CI
`[+0.2356,+0.3212]`) and regret reduction is `+0.0751` (CI
`[+0.0455,+0.1073]`). The weakest case has positive regret reduction
`+0.0182`. The five-seed artifact is
`results/gdd_wm_relative_action_effect_panel_20261014/canonical_2arch_5seed_aggregate.json`;
this remains offline diagnostic evidence only.

The same 10 cases were verified on independent outcome roots through the
dataset-matched frozen RGB successor bridge. All three verifier gates passed;
the aggregate successor error improvement over the source-latent baseline is
`+14.21%` (two-level case/root CI `[+8.47%,+19.69%]`) and the predicted
successor is closer on `66.56%` of candidate rows (CI `[59.77%,73.05%]`). The
aggregator is
`experiments/gdd_wm/aggregate_rgb_successor_panel.py`, with output at
`results/decision_benchmark_candidate_bank_20261014/rgb_successor_canonical_2arch_5seed_aggregate.json`.
This is realized RGB successor evidence, not closed-loop online planning.

The feature-family audit also includes a train-only latent-to-source-XY probe
that estimates the goal-relative action frame at inference time. Relative to
`source_plus_delta`, `source_delta_estimated_goalframe_action` improves
outcome consistency by `+0.0985` (two-level case/root CI
`[+0.0534,+0.1449]`) and reduces candidate regret by `+0.0546` (CI
`[+0.0176,+0.0888]`). It is positive on 9/10 cases; standardized seed 8413
remains negative. The aggregate is written by
`experiments/gdd_wm/aggregate_feature_family_panel.py`. This is a partial
representation repair, not universal transfer or closed-loop planning.

The failure-case control
`experiments/gdd_wm/audit_rgb_calibrated_candidate_selection.py` fits a
successor calibration only on train-root realized RGB references before task
readout. On standardized seed 8413, train successor RMSE improved to `0.1628`,
but outcome `source_plus_delta` regret worsened from `0.0872` to `0.1503`.
Generic RGB successor calibration therefore does not repair decision geometry.

The first PointMaze P0/P1 implementation is under
`experiments/decision_benchmark/`. It repairs the train-only affine task
readout, audits action identifiability, generates equal-duration candidate
banks, and evaluates paired decision regret:

```bash
python experiments/decision_benchmark/generate_candidate_bank.py \
  --source-export /path/to/frozen_external_export.json \
  --checkpoint /path/to/adapter.pt \
  --output results/decision_benchmark_candidate_bank \
  --candidate-family paired_order_v2

python experiments/decision_benchmark/audit_action_effects.py \
  --checkpoint /path/to/adapter.pt \
  --fit-export /path/to/fit.json \
  --outcome-export /path/to/outcome.json \
  --external-export /path/to/external_export.json \
  --output results/decision_benchmark_action_audit
```

`paired_order_v2` includes same-net-action/different-order pairs.  D0/P1
remains the offline candidate/effect benchmark.  The D1 RGB successor bridge
is now executable with:

```bash
python experiments/decision_benchmark/verify_rgb_successor.py \
  --candidate-dataset results/decision_benchmark_candidate_bank/.../candidate_dataset.json \
  --arrays results/decision_benchmark_candidate_bank/.../arrays.npz \
  --dataset-root data/point_maze_tail1800_1899 \
  --encoder-checkpoint data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar \
  --output results/decision_benchmark_candidate_bank/rgb_successor_v2_temporal \
  --max-roots 40 --max-candidates 16
```

The verifier executes candidates in the legacy MuJoCo dynamics, reconstructs
dataset-matched RGB from the realized five-step block states, and encodes the
full temporal context used by the frozen JEPA checkpoint.  The 2026-10-14
artifact has 40/40 renderer/source-interface gates passing and a 13.9% held-out
outcome improvement over the source-latent baseline in PCA successor error.
This is realized RGB successor evidence, not yet closed-loop online planning
or a universal-world-model result.

The two proposal tracks are implemented as separate packages:

- `experiments/theory_spectral_transfer/`: theory-only spectral-transfer
  simulation and operator diagnostics. It does not load LeWM checkpoints or
  datasets.
- `experiments/empirical_spectral_audit/`: empirical latent eigenspace audit
  over collected LeWM arrays such as `latents.npz`. It does not collect new
  rollouts or write chunks.

Theory proposal entrypoint:

```bash
python experiments/theory_spectral_transfer/run_analysis.py \
  --output results/theory_spectral_transfer.json \
  --dim 128 \
  --intrinsic-dim 2 \
  --input-exponent 1.0 \
  --encoder-exponent 1.0 \
  --tail-encoder-exponent 3.0 \
  --breakpoint 32 \
  --heat-time 0.01
```

This writes the oracle spectra for
`C_theta^{1/2} G_E C_theta^{1/2}`, predictor-filtered spectra, residual spectra,
heat-filtered spectra, log-log slope fits, stable-rank style descriptors, and
the expected oracle slope `(p+q)/d`. The same report also includes oblique-frame
stress tests, known-generator heat-filter recovery, a SIGReg/local-anisotropy
toy check, and an identifiability-vs-controllability toy case.

Empirical proposal entrypoint:

```bash
python experiments/empirical_spectral_audit/run_audit.py \
  --latents results/pusht/lewm/latents.npz \
  --output results/pusht/lewm/empirical_spectral_audit.json \
  --condition states \
  --band-mode energy \
  --min-cell-size 256 \
  --max-samples 200000 \
  --horizons 1 3 5
```

This consumes `latents.npz` and reports DTL/top-band lift, rank-wise transfer
fingerprints, PSE, PPTE, NCR, random-basis controls, norm-matched random tangent
controls, Gaussian tangent controls, shuffle controls, multi-horizon dynamics,
controllability, observability probes, eigenspace transport, and offline
planner-facing interventions. It also emits proposal-level planner and
exploration outputs: `spectral_planner_metric`, `spectral_exploration`, and
optional `ood_spectral_shift` when `ood_latents` is present. Use this after
Phase 1 has finalized `latents.npz`; do not point it at an output directory
whose chunks are still being mutated unless you are reading a completed
`latents.npz`.

Later-stage empirical bridge summary:

```bash
python experiments/empirical_spectral_audit/run_operator_alignment_bridge.py \
  --audit results/pusht/lewm/empirical_spectral_audit.json \
  --theory results/theory_operator_alignment_sweep_20260828/operator_alignment_theorem.json \
  --output results/pusht/lewm/empirical_operator_alignment_bridge_20260829
```

This post-hoc summary is a coordination artifact, not a manuscript. The theory
paper and the empirical paper are tracked separately; this bridge only
compresses the frozen audit into basis-sensitive lift, label-shuffle rivals,
transport, and planner anisotropy.

Optional empirical fields are detected when present:
`dynamics_h*`, `predicted_dynamics_h*`, `prediction_residual_h*`, `goals`,
`success`, `episode_id`, `timestep`, `physical_*`, and `counterfactual_*`.
Use `--condition episode` or `--condition phase` when those fields are present.
`ood_latents` enables the OOD spectral-shift report.
Use `--max-samples N` for large compressed `latents.npz` files; the loader
subsamples rows deterministically without writing a separate sample NPZ.

Both separated packages include `alignment.py` manifests that map proposal
claims to concrete APIs and JSON report fields; `tests/test_proposal_alignment.py`
guards against accidental omissions. The theory package produces
`operator_alignment_oral_object_card.md`, `operator_alignment_paper_outline.md`,
and `operator_alignment_paper_spine.md`; the empirical package now produces
`empirical_operator_alignment_paper_outline.md` and
`empirical_operator_alignment_paper_spine.md`.

The empirical package also contains a learned action-conditioned intervention
benchmark at
`experiments/empirical_spectral_audit/run_alignment_training_synthetic.py`.
Its canonical CPU result is
`results/alignment_composition_20261003/alignment_training_synthetic.json`.
The benchmark compares pooled versus action-conditional alignment using held-
out one-step MSE, repeated-action rollout, and ordered two-action composition;
it is synthetic mechanism evidence only and does not establish nonlinear LeWM
or PushT planning gains.

A second benchmark at
`experiments/empirical_spectral_audit/run_pixel_world_model_synthetic.py`
jointly trains a small pixel encoder and latent predictor with next-embedding
prediction plus SIGReg. It adds random-sequence rollout, action composition,
stable-rank, and validation-checkpoint audits. Its canonical result is
`results/pixel_world_alignment_20261003/pixel_world_model_synthetic.json`;
the result remains synthetic pixel-world evidence rather than a PushT or
end-to-end LeWM claim.

For a mechanism-level gradient-route ablation of this learned pixel benchmark,
run:

```bash
python experiments/empirical_spectral_audit/run_pixel_world_model_synthetic.py \
  --output results/pixel_gradient_route_ablation_20261003/pixel_gradient_route_ablation.json \
  --epochs 120 --seeds 0 1 2 3 4 --weights 0.01 0.1 \
  --gradient-routes predictor encoder joint
python experiments/empirical_spectral_audit/summarize_gradient_route_ablation.py \
  --input results/pixel_gradient_route_ablation_20261003/pixel_gradient_route_ablation.json \
  --output results/pixel_gradient_route_ablation_20261003/gradient_route_paired_summary.json
```

This explicitly separates predictor-only, encoder-only, and historical joint
alignment paths. It records route-specific gradient norms, one-step and
random-sequence rollout error, action composition, commutator diagnostics,
stable rank, covariance entropy, and covariance condition number. The paired
summary uses seed-matched bootstrap intervals and an exact sign-flip reference;
with five seeds these remain descriptive uncertainty summaries. The artifact
is synthetic mechanism evidence and does not establish nonlinear LeWM, PushT,
causal control, or online planning gains.

The follow-up conditioning audit extends the pixel benchmark with detached
`magnitude`, `direction`, and `local` groupings for conditional operator
alignment. The matched 10-seed sweep is recorded in
`results/empirical_spectral_audit/pixel_conditioning_formal10seed_20261006/comparison.md`,
with route-specific controls in
`results/empirical_spectral_audit/pixel_direction_route_formal10seed_20261006/route_comparison.md`.
It finds a non-monotone trade-off: larger alignment weights can reduce the
conditional commutator while worsening rollout and action composition. The
current primary candidate for a real LeWM pilot is direction-conditioned joint
alignment at weight `0.001`, but this remains synthetic evidence and is not a
planning or task-success claim.

For a held-out model-selection audit of the covariance--predictor diagnostic,
use `experiments/empirical_spectral_audit/run_covariance_predictor_diagnostic_benchmark.py`.
The primary family holds the predictor spectrum, eigenvalue pairing, and
rotation plane fixed while varying the rotation angle. Alignment is estimated
from the training covariance only; transfer regret is evaluated on independent
test states. The matched-pair audit uses two fixed-spectrum candidates with
nearly equal one-step error but different alignment and multi-step transfer.
The canonical CPU artifact is
`results/covariance_predictor_diagnostic_benchmark_20261008/diagnostic.json`.
This supports a conditional model-selection result for the declared
fixed-pairing family. A scalar commutator is not a universal selector across
planes or eigengaps; cross-plane stress tests must be reported separately.

The learned-checkpoint extension is implemented in
`experiments/empirical_spectral_audit/run_learned_checkpoint_diagnostic_benchmark.py`.
It applies the same train/validation/test protocol to frozen JEPA-WM and
DINO-WM vector exports from PushT and PointMaze. Each candidate is an
orthogonal conjugate of a rank-16 ridge operator fit in the train split, so
the candidate family preserves the eigenvalue multiset to numerical precision;
these are checkpoint-derived linear probes, not independently retrained neural
checkpoints. The canonical 17- and 65-candidate artifacts are
`results/covariance_predictor_learned_checkpoint_benchmark_20261008/` and
`results/covariance_predictor_learned_checkpoint_benchmark_65_20261008/`.
The validation-constrained alignment tie-break reduces pooled downstream
regret relative to a spectrum-only random tie-break, but the stratified report
shows that this effect is concentrated in JEPA-WM/PointMaze; PushT and
DINO-WM/PointMaze are usually candidate-set degenerate or neutral. Thus the
learned-checkpoint result is a conditional selection utility, not evidence for
a universal alignment objective, nonlinear training improvement, physical
causality, or online planning.

For the coordinate-gauge and non-normality boundary, use
`experiments/empirical_spectral_audit/run_gauge_nonnormal_audit.py`. It verifies
the covariance-basis identity for arbitrary predictors, demonstrates that the
ordinary Euclidean commutator changes under a general invertible latent
reparameterization, and verifies invariance of the metric-aware version when
the latent metric transforms covariantly. It also constructs a spectrum-fixed
upper-triangular Schur family with changing non-normality. The canonical
artifact is `results/gauge_nonnormal_audit_20261008/audit.json`.

For the complete theory-boundary audit, use
`experiments/theory_spectral_transfer/run_intrinsic_transfer_theory_audit.py`.
It adds the full-GL covariance-only no-go, the metric-whitened
orthogonal-similarity check, a Schur/resolvent Cauchy-circle bound at fixed
eigenvalues, and a two-step nonlinear local-Jacobian remainder check. The
canonical artifacts are
`results/transfer_certificate_20261008/intrinsic_transfer_theory_audit.json`
and the corresponding `.md` report. This is mathematical and synthetic
certificate evidence, not learned-model, causal, or planning evidence.

For a frozen PushT latent interpretability audit, use
`experiments/empirical_spectral_audit/run_frozen_subspace_probes.py`. It fits
the covariance basis on a training split only and compares held-out ridge
readouts from top/middle/tail covariance bands against dimension-matched random
subspace controls for state, action, dynamics, predicted-dynamics, and
residual targets. The canonical result is
`results/pusht/lewm/frozen_subspace_probes_20261003/`; this is descriptive
representation evidence, not a causal intervention or planning evaluation.

To explicitly extract dynamics-sensitive directions rather than equating
variance with dynamics, run
`experiments/empirical_spectral_audit/run_dynamics_subspace_probe.py`. It
compares a latent--dynamics cross-covariance basis and an action-residualized
conditional basis against covariance and random-subspace controls using
held-out R². The canonical result is
`results/pusht/lewm/dynamics_subspace_probe_20261003/`.

The extracted bases can be transferred to the external frozen temporal audit
with `experiments/empirical_spectral_audit/run_external_dynamics_basis_intervention.py`.
This tests functional sensitivity of the same latent coordinates under
multi-step rollout; it is not a physical intervention or a planning-success
evaluation. The canonical report is stored under
`results/pusht/lewm/external_public_pusht_temporal_20261003/`.

For the residual-side decomposition, use
`experiments/empirical_spectral_audit/run_residual_subspace_probe.py`.
It fits a latent--prediction-residual cross-covariance basis and a
residual-covariance basis on the training split, then reports held-out
residual/dynamics readouts, covariance/dynamics overlap, and residual energy
profiles. The split-robust result is under
`results/pusht/lewm/residual_subspace_robustness_20261003/`. This diagnoses
where prediction error is represented; it does not show that training
intentionally relocates error or improves control.

For the stricter rank/target and cross-horizon readout audits, use
`experiments/empirical_spectral_audit/run_external_rank_target_specificity_audit.py`
and
`experiments/empirical_spectral_audit/run_external_cross_horizon_readout.py`.
The first tests target specificity over fixed ranks and the second fits the
readout at one horizon and evaluates it at another, with leave-one-episode-out
fits, source-fitted frozen bases, paired episode bootstrap, and covariance,
residual, and random controls. The canonical cross-horizon result is
`results/pusht/lewm/external_cross_horizon_readout_20261004/`.
These are frozen latent predictive diagnostics: positive transfer does not
establish physical causality, online planning, task success, or retraining
improvement.

For a perturbation-magnitude control, use
`experiments/empirical_spectral_audit/run_external_energy_matched_dynamics_intervention.py`.
It rescales covariance and random top-8 controls at every autoregressive step
to match the squared Euclidean latent perturbation energy of dynamics top-8
removal. This separates subspace identity from intervention magnitude, while
remaining a frozen rollout diagnostic rather than causal-control evidence. The
three-split summary is included in
`results/pusht/lewm/dynamics_subspace_robustness_20261003/dynamics_subspace_robustness.md`.

For the cross-source error-transport check, use
`experiments/empirical_spectral_audit/run_external_residual_transport_audit.py`.
It projects unmodified external h=1/2/4/8 rollout errors into the residual,
dynamics, covariance, conditional-dynamics, residual-covariance, and random
bases, followed by episode-cluster paired inference. The split-robust summary
is `external_residual_transport_robustness.md` under the external PushT
results directory. This distinguishes source-domain residual predictability
from residual-energy geometry that actually transfers.

For the functional follow-up, use
`experiments/empirical_spectral_audit/run_external_residual_basis_intervention.py`.
It removes residual cross-covariance and residual-covariance top-8 coordinates
from the autoregressive latent rollout, with dynamics-referenced
energy-matched controls. The three-split summary is
`external_residual_basis_intervention_robustness.md`; it tests rollout
sensitivity and must not be read as a physical causal intervention.

For a direct physical-motion diagnostic, use
`experiments/empirical_spectral_audit/run_external_physical_dynamics_probe.py`.
It probes five-dimensional physical displacement from the last context state
to h=1/4/8 using transferred latent increments and transition pairs, with an
observed-action-only baseline and dimension-matched random bands. The paired
reports under `results/pusht/lewm/external_public_pusht_temporal_20261003/`
separate source-domain latent-dynamics predictability from external physical
motion decoding; neither is causal-control or planning evidence.
Use `summarize_external_physical_dynamics_robustness.py` to compare the same
probe under public, validation-zscore, and raw/100 action normalization.
Use `run_external_physical_action_alignment_probe.py` for within-episode shift
and global-derangement controls that preserve action marginals while breaking
action--transition alignment.

For a stricter latent-dynamics transport check, use
`experiments/empirical_spectral_audit/run_external_latent_dynamics_transfer.py`.
It transfers the internally fitted covariance, dynamics, and
action-residualized dynamics bases to the public external PushT trajectories
without refitting them.  Leave-one-episode-out probes compare action-only,
current-state-only, and rollout-increment features for h=1/4/8; the same
true-trajectory probe is then applied to the frozen autoregressive increment.
The canonical public-normalization report is
`results/pusht/lewm/external_public_pusht_temporal_20261003/external_latent_dynamics_transfer_w8.md`.
The cross-normalization claim matrix is
`external_latent_dynamics_transfer_cross_normalization.md` in the same
directory.  The result supports stronger transport for the dynamics basis in
the public normalization arm, is only partial under validation z-score, and
does not survive raw/100 normalization.  It is therefore a frozen latent
transport diagnostic, not physical-dynamics, causal-control, planning, or
retraining evidence.

This estimator is distinct from the stricter cross-horizon frozen-readout
audit: it uses transferred latent-dynamics probes and remains
normalization-sensitive. The cross-horizon audit fits a compact readout at one
horizon and evaluates it at another; its canonical normalization/density
matrix is `results/pusht/lewm/external_cross_horizon_readout_20261004/`, and
the independent source-split replication is
`results/pusht/lewm/external_cross_horizon_source_split_robustness_20261004/`.
The latter retains positive paired lower bounds across both source splits,
three normalization arms, and two/eight-window densities. These remain frozen
encoded-latent predictive diagnostics, not causal or physical dynamics.

To stress-test the observed-action baseline itself, use
`experiments/empirical_spectral_audit/run_external_cross_horizon_action_control_robustness.py`.
It compares the legacy compact six-dimensional action summary with the full
sequence of per-block two-dimensional action means, left-aligned and zero-
padded to a common 16-dimensional h=8 representation. The canonical six-cell
matrix is under
`results/pusht/lewm/external_cross_horizon_action_control_robustness_20261004/`;
the two-source-split 12-cell replication is under
`results/pusht/lewm/external_cross_horizon_action_control_source_split_robustness_20261004/`.
The aggregate report is
`action_control_robustness_20261004.json`. All cells use CUDA, 21 external
episodes, leave-one-episode-out fitting, and 2,000 episode-cluster bootstrap
replicates. This is a frozen predictive diagnostic beyond an observed action
history, not a causal action-effect, physical-counterfactual, planning, or
task-success result.

For the temporal follow-up that tests whether alignment survives autoregressive
rollout, use
`experiments/empirical_spectral_audit/run_external_temporal_alignment_adapter.py`:

```bash
python experiments/empirical_spectral_audit/run_external_temporal_alignment_adapter.py \
  --val-root /tmp/lewm_pusht_public_val_20261003/pusht_noise/val \
  --output results/pusht/lewm/external_temporal_alignment_adapter_20261003.json \
  --epochs 20 --windows-per-episode 2 --seeds 0 1 2 \
  --split-seeds 3072 3073 3074 --weights 0.01 0.1 1.0
```

The frozen encoder and predictor are unchanged. A residual adapter is trained
only on one-step windows from training episodes and evaluated on held-out
episodes at h=1/2/4/8. Inference uses split-level paired means; adapter
initialization seeds are nested within each episode split. The canonical report
is `results/pusht/lewm/external_temporal_alignment_adapter_20261003.md`.
This remains a frozen-checkpoint latent rollout diagnostic, not end-to-end
retraining, online control, or planning success.

For the theory-side interaction and intervention-strength sweep:

```bash
python experiments/theory_spectral_transfer/run_mechanism_sweep.py \
  --output results/theory_mechanism_sweep_20260801
```

It jointly scans encoder/predictor exponents, heat-filter time/rate settings,
and oblique-frame stress. These are analytical mechanism checks, not empirical
LeWM measurements.

To test whether the completed PushT offline readouts depend on the conditioning
choice, run a shared-sample condition sweep after `latents.npz` has finalized:

```bash
mkdir -p results/pusht/lewm/empirical_condition_sweep_20260801
for condition in states actions latents global; do
  python experiments/empirical_spectral_audit/run_audit.py \
    --latents results/pusht/lewm/latents.npz \
    --output "results/pusht/lewm/empirical_condition_sweep_20260801/${condition}.json" \
    --condition "$condition" \
    --band-mode energy \
    --clusters 32 \
    --min-cell-size 256 \
    --max-samples 100000 \
    --seed 3072 \
    --horizons 1 3 5
done
python experiments/empirical_spectral_audit/summarize_condition_sweep.py \
  --input results/pusht/lewm/empirical_condition_sweep_20260801 \
  --k 32
```

The summary keeps schema warnings explicit. With the current compact PushT
artifact, missing `dynamics_h*`, physical, and counterfactual fields remain
hard evidence gaps rather than negative results.

To assemble the theory and empirical outputs into one evidence package:

```bash
python experiments/summarize_spectral_transfer_evidence.py \
  --theory-mechanism results/theory_mechanism_sweep_20260801/theory_mechanism_sweep.json \
  --theory-dim results/theory_dim_sweep_20260801/cross_dim_summary.json \
  --theory-robustness results/theory_robustness_sweep_20260801/robustness_summary.json \
  --empirical-condition results/pusht/lewm/empirical_condition_sweep_20260801/condition_summary.json \
  --output results/spectral_transfer_evidence_20260801
```

This writes `evidence_summary.json`, `evidence_summary.md`,
`evidence_summary.pdf`, and `evidence_summary.png`, while retaining explicit
boundaries for theory-only, offline empirical, missing-schema, and
dataset-blocked claims.

## Phase 1: Latent Eigenspace Audit

```bash
python experiments/latent_eigenspace_audit.py \
  --env pusht \
  --policy pusht/lewm \
  --output results/pusht/lewm \
  --num-workers 0 \
  --progress-every 10 \
  --chunk-batches 20 \
  --rich-fields
```

Outputs:
- `progress.jsonl`: append-only batch progress.
- `chunks/chunk_*.npz`: recoverable intermediate latent chunks when `--keep-chunks` is set.
- `latents.npz`: sampled `latents`, `dynamics`, `planning`, `nuisance`, optional `predicted_dynamics`, `prediction_residual`, `actions`, `states` for downstream audits. Default sample cap is `--save-max-samples 200000`; pass `--save-max-samples 0` to save all rows.
- With `--rich-fields`, `latents.npz` also saves available `dynamics_h*`, `predicted_dynamics_h*`, `prediction_residual_h*`, `goals`, `success`, `episode_id`, `timestep`, `physical_*`, and `counterfactual_*` arrays. Use `--rich-horizons 1 3 5` to control which multi-step horizons are exported. These enable goal/phase-conditioned audits, observability probes, impossible-transition controls, and true multi-horizon spectral transfer.
- `latents_manifest.json`: total rows, saved rows, and sampling settings.
- `lift_metrics.json`: state-conditioned lift curves, spectrum descriptors, global controls, predicted-vs-real tangent alignment, residual spectra, global band profiles, and action-conditioned summaries.

By default, chunk files are deleted after the sampled `latents.npz` is written. Add `--keep-chunks` only when you need to inspect or reuse the intermediate chunks.

For a first PushT pilot, cap collection before the full dataset:

```bash
python experiments/latent_eigenspace_audit.py \
  --env pusht \
  --policy pusht/lewm \
  --output results/pusht/lewm \
  --num-workers 0 \
  --max-batches 130 \
  --progress-every 10 \
  --chunk-batches 20
```

If collection was interrupted after clean chunks were written, finalize from
those chunks without rescanning the dataset:

```bash
python experiments/latent_eigenspace_audit.py \
  --env pusht \
  --output results/pusht/lewm \
  --finalize-chunks-only
```

Primary readouts:
- H1: `weighted_mean.dynamics_lift` should exceed `shuffled_dynamics_lift` and `gaussian_dynamics_lift`.
- H2: `weighted_mean.predicted_dynamics_lift`, `prediction_tangent_alignment.mean_cosine`, and `prediction_tangent_alignment.band_profile_l1` test whether predictor displacement follows the same spectral directions as real displacement.
- H3: `weighted_mean.prediction_residual_lift` and `global_band_profiles.prediction_residual` localize where encoder transfer and predictor transfer disagree.
- H4: compare `weighted_mean.planning_lift` with `weighted_mean.nuisance_lift`; planning/reachability should dominate nuisance.
- Action geometry: `action_conditioned.weighted_mean.*_lift` checks whether these effects survive action binning instead of only appearing globally.

The same entrypoint covers the follow-up environments:

```bash
python experiments/latent_eigenspace_audit.py --env tworoom --policy tworoom/lewm --output results/tworoom/lewm
python experiments/latent_eigenspace_audit.py --env cube --policy cube/lewm --output results/cube/lewm
python experiments/latent_eigenspace_audit.py --env reacher --policy reacher/lewm --output results/reacher/lewm
```

TwoRooms is for reachability-vs-latent-distance mismatch. Cube/Reacher are for
object pose, joint state, and physical tangent localization.

## Phase 2: Planning Alignment Audit

```bash
python experiments/planning_alignment_audit.py \
  --env pusht \
  --policy pusht/lewm \
  --output results/pusht/lewm \
  --num-workers 0 \
  --progress-every 10 \
  --chunk-batches 50 \
  --spectral-basis results/pusht/lewm/latents.npz
```

Outputs:
- `planning_progress.jsonl`: append-only batch progress.
- `planning_chunks/chunk_*.npz`: recoverable intermediate planning-cost chunks when `--keep-chunks` is set.
- `planning_alignment_raw.npz`
- `planning_alignment.json`: includes vanilla latent-cost alignment plus optional `spectral_planner_costs` when `--spectral-basis` is provided.

By default, planning chunk files are deleted after `planning_alignment_raw.npz` is written. Add `--keep-chunks` only when you need to inspect or reuse them.

This is an offline diagnostic. Online success/score should be measured with
`eval.py`, either using the original latent Euclidean cost or the online
spectral planner cost below.

For a first planning pilot, use the Phase 1 spectral basis and cap the scan:

```bash
python experiments/planning_alignment_audit.py \
  --env pusht \
  --policy pusht/lewm \
  --output results/pusht/lewm \
  --num-workers 0 \
  --max-batches 500 \
  --progress-every 10 \
  --chunk-batches 50 \
  --spectral-basis results/pusht/lewm/latents.npz
```

## One-Command Full Run

Run the full PushT diagnostic suite from a fresh timestamped output directory:

```bash
bash experiments/run_full_spectral_transfer.sh
```

This runs the theory spectral-transfer report, the richer paper-style theory
sweep, the cross-dimension theory sweep, the seed/noise robustness sweep,
the joint theory mechanism sweep, Phase 1 latent eigenspace audit, the
separated empirical spectral audit, the empirical condition-robustness sweep,
Phase 2
planning alignment with the
Phase 1 spectral basis, Phase 3 adapter intervention, the summary table, and
the theory-to-empirical evidence summary.
Progress is written to `stage1.log`, `stage2.log`, `progress.jsonl`,
`planning_progress.jsonl`, `run_status.json`, and `run_events.jsonl`.
By default, Phase 1 writes a sampled `latents.npz` with 200000 rows and Phase 1/2 chunks are temporary. Chunks are removed after each phase succeeds; pass `--save-max-samples 0` to save full Phase 1 arrays or `--keep-chunks` to preserve chunk intermediates.
The default output is `results/pusht/lewm_full_<timestamp>`, so existing chunks
in `results/pusht/lewm` are not overwritten. With `--resume`, the default output
is `results/pusht/lewm`, and Phase 1/2 append after existing chunk files when they are present. Resume
infers the next dataset batch as `num_existing_chunks * --chunk-batches`; if an
older run used a different chunk size, pass `--resume-start-batch` directly to
the underlying Phase 1/2 script.

One command to continue under the already produced PushT data:

```bash
bash experiments/run_full_spectral_transfer.sh \
  --resume \
  --output results/pusht/lewm \
  --empirical-horizons "1 3 5"
```

If another Phase 1/2 writer is already targeting the same output directory, the
script exits before starting a duplicate writer. Use `--skip-completed` after a
clean interruption when you want to reuse already written final artifacts such as
`latents.npz`, `planning_alignment.json`, or `empirical_spectral_audit.json`.

Useful variants:

```bash
# Print the exact commands without running them.
bash experiments/run_full_spectral_transfer.sh --dry-run

# Strong-proposal pilot: rich factor logging, memory-bounded audit, online
# spectral planner eval, and a short parameter-level directional intervention.
bash experiments/run_full_spectral_transfer.sh \
  --output results/pusht/lewm_strong_pilot \
  --rich-fields \
  --empirical-max-samples 200000 \
  --run-eval \
  --eval-spectral \
  --eval-num 50 \
  --run-directional-train \
  --directional-max-epochs 10

# Full run with an explicit output directory and device.
bash experiments/run_full_spectral_transfer.sh \
  --policy pusht/lewm \
  --output results/pusht/lewm_full \
  --device cuda

# Continue from chunks left by an interrupted results/pusht/lewm run.
bash experiments/run_full_spectral_transfer.sh --resume

# Continue existing data and skip stages whose final JSON/NPZ already exists.
bash experiments/run_full_spectral_transfer.sh \
  --resume \
  --output results/pusht/lewm \
  --skip-completed

# Keep a compact status snapshot updated while a run is active.
python experiments/report_run_status.py --output results/pusht/lewm --watch --interval 30

# Continue a specific interrupted full run directory.
bash experiments/run_full_spectral_transfer.sh \
  --resume \
  --output results/pusht/lewm_full_YYYYMMDD_HHMMSS

# Pilot run through the same full pipeline.
bash experiments/run_full_spectral_transfer.sh \
  --output results/pusht/lewm_pilot \
  --stage1-max-batches 130 \
  --stage2-max-batches 500
```

## Robust Background Runs

For long full-dataset runs, launch each phase detached and keep a PID/log file:

```bash
mkdir -p results/pusht/lewm
setsid bash -c 'echo $$ > results/pusht/lewm/stage1.pid; exec .venv/bin/python -u experiments/latent_eigenspace_audit.py --env pusht --policy pusht/lewm --output results/pusht/lewm --num-workers 0 --progress-every 10 --chunk-batches 20 >> results/pusht/lewm/stage1.log 2>&1' >/dev/null 2>&1 < /dev/null &

setsid bash -c 'echo $$ > results/pusht/lewm/stage2.pid; exec .venv/bin/python -u experiments/planning_alignment_audit.py --env pusht --policy pusht/lewm --output results/pusht/lewm --num-workers 0 --progress-every 10 --chunk-batches 50 --spectral-basis results/pusht/lewm/latents.npz >> results/pusht/lewm/stage2.log 2>&1' >/dev/null 2>&1 < /dev/null &
```

Start Phase 2 after Phase 1 has written `latents.npz`. Monitor with `tail -f results/pusht/lewm/stage1.log` or `tail -f results/pusht/lewm/stage2.log`.

## Phase 3: Feature-Space Adapter Intervention

This is the cheapest intervention diagnostic: build `A=(I+lambda Z_target^T Z_target / n)^-1`
from one target tangent, apply it to all tangent families, and check whether target
energy is selectively suppressed more than non-target dynamics/planning energy.

```bash
python experiments/intervention_adapter_audit.py \
  --latents results/pusht/lewm/latents.npz \
  --output results/pusht/lewm/intervention_adapter.json \
  --target nuisance \
  --non-targets dynamics planning predicted_dynamics prediction_residual
```

Primary readout: for a successful selective intervention, target retention should
be lower than non-target retention, while band-profile changes should match the
pre-intervention target overlap profile.

## Phase 4: Online Spectral Planner Evaluation

The online planner intervention wraps the LeWM cost model so CEM/Adam still
optimizes action candidates in the environment loop, but terminal latent cost is
evaluated with an eigenspace-weighted metric estimated from `latents.npz`.

```bash
python eval.py --config-name=pusht.yaml \
  policy=pusht/lewm \
  output.dir=results/pusht/lewm \
  output.json_filename=eval_metrics.json

python eval.py --config-name=pusht.yaml \
  policy=pusht/lewm \
  output.dir=results/pusht/lewm_spectral \
  eval.spectral_planner.enabled=true \
  eval.spectral_planner.latents=results/pusht/lewm/latents.npz \
  eval.spectral_planner.max_samples=200000 \
  eval.spectral_planner.top_k=32 \
  output.json_filename=eval_metrics.json
```

Compare `eval_metrics.json` between Euclidean and spectral-planner runs. This
is the planner-in-the-loop causal test that complements the offline
`planning_alignment.json`.

## Phase 5: Directional Training

Original LeWM:

```bash
python train.py data=pusht output_model_name=lewm_original
```

Directional variants:

```bash
python experiments/train_lewm_directional.py data=pusht output_model_name=lewm_dyn \
  loss.directional.dyn_weight=0.01 loss.directional.reach_weight=0.0 loss.directional.nuisance_weight=0.0

python experiments/train_lewm_directional.py data=pusht output_model_name=lewm_reach \
  loss.directional.dyn_weight=0.0 loss.directional.reach_weight=0.01 loss.directional.nuisance_weight=0.0

python experiments/train_lewm_directional.py data=pusht output_model_name=lewm_dyn_reach \
  loss.directional.dyn_weight=0.01 loss.directional.reach_weight=0.01 loss.directional.nuisance_weight=0.0

python experiments/train_lewm_directional.py data=pusht output_model_name=lewm_dyn_reach_nuis \
  loss.directional.dyn_weight=0.01 loss.directional.reach_weight=0.01 loss.directional.nuisance_weight=0.01

python experiments/train_lewm_directional.py data=pusht output_model_name=lewm_teacher_anchor \
  loss.directional.dyn_weight=0.01 \
  loss.directional.controllability_weight=0.01 \
  loss.directional.teacher_policy=pusht/lewm \
  loss.directional.teacher_anchor_weight=0.01
```

Run each with seeds `3072`, `3073`, and `3074`. Sweep
`{0.001, 0.01, 0.05}` for enabled directional weights, filtering out runs that
substantially worsen one-step prediction loss.

## Phase 6: Evidence Matrix

For each checkpoint and seed, collect:

| Claim | Required artifact | Main fields |
| --- | --- | --- |
| Dynamics localize in latent top eigenspaces | `lift_metrics.json` | `weighted_mean.dynamics_lift`, controls |
| Predictor follows real spectral tangent | `lift_metrics.json` | `predicted_dynamics_lift`, `prediction_tangent_alignment` |
| Residual spectrum localizes prediction failures | `lift_metrics.json` | `prediction_residual_lift`, `global_band_profiles.prediction_residual` |
| Planning directions dominate nuisance | `lift_metrics.json` | `planning_lift` vs `nuisance_lift` |
| Targeted adapter suppresses selected nuisance | `intervention_adapter.json` | `retention`, `non_target_over_target_retention` |
| Latent planning cost tracks task progress | `planning_alignment.json` | Spearman/Pearson cost-success and cost-distance |
| Spectral planner has online causal effect | `eval_metrics.json` | Euclidean vs spectral success/score |
| Directional regularizers improve planning geometry | `summary.md` plus online eval | Lift/alignment changes without worse eval score |

Keep original LeWM checkpoints as the main baseline, then compare directional variants against prediction loss and online `eval.py` success. A useful negative result is also publishable: if prediction loss improves while reachability lift or cost-success alignment worsens, that directly supports the hypothesis that predictive latent geometry is not necessarily plannable latent geometry.

## Phase 7: Summary

```bash
python experiments/summarize_directional_results.py \
  results/pusht/lewm results/pusht/lewm_dyn results/pusht/lewm_dyn_reach \
  --output results/pusht/summary.md
```

## Phase 8: Nonlinear Planner-Facing Metric Audit

The CPU-only nonlinear proxy isolates the planner ranking metric from the
learned predictor. All model conditions share reachable target sequences and
candidate action sets; metric estimates use training transitions only.

```bash
python experiments/empirical_spectral_audit/run_nonlinear_planner_metric_ablation.py \
  --output results/nonlinear_planner_metric_ablation_20261003/report.json \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --horizons 2 4 8 --tasks 64 --candidates 48

python experiments/empirical_spectral_audit/run_nonlinear_planner_candidate_sensitivity.py \
  --output results/nonlinear_planner_metric_ablation_20261003/candidate_sensitivity.json \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --horizons 4 8 --candidate-counts 16 48 96 --tasks 64
```

The compared metrics are Euclidean, covariance precision, action-reachability
weighting, an equal-spectrum random-basis control, and a privileged oracle.
These artifacts are synthetic candidate-set diagnostics, not online LeWM or
PushT control evidence; the absence of nuisance labels is intentional and the
random spectral condition must not be described as nuisance suppression.

The follow-up `experiments/empirical_spectral_audit/run_nonlinear_controllability_metric_ablation.py`
constructs a finite-horizon controllability-Gramian precision metric from a
training-only linear transition surrogate. Its canonical report is
`results/nonlinear_planner_metric_ablation_20261003/controllability_metric.md`.
The paired `run_nonlinear_controllability_ridge_sensitivity.py` audit fixes the
models and candidate sets while varying Gramian ridge fractions; its report is
`results/nonlinear_planner_metric_ablation_20261003/controllability_ridge_sensitivity.md`.
These tests are designed to falsify, rather than assume, the claim that
dynamics-informed geometry improves candidate ranking.

The local follow-up
`experiments/empirical_spectral_audit/run_nonlinear_local_controllability_ablation.py`
uses per-state secant Jacobians and action effects to build a separate Gramian
for each planning task. Its canonical artifact is
`results/nonlinear_planner_metric_ablation_20261003/local_controllability.md`.
The current result is negative: local Gramian precision is less reliable than
Euclidean ranking in this nonlinear proxy, with the effect persisting across
ID/OOD shifts and planning horizons. This is evidence against treating a
low-rank controllability inverse as a generally useful planner metric.

The goal-conditioned follow-up
`experiments/empirical_spectral_audit/run_nonlinear_goal_supervised_metric_ablation.py`
learns a positive diagonal metric from independent success/failure candidate
labels, separately for each planning horizon. The canonical artifact is
`results/nonlinear_planner_metric_ablation_20261003/goal_supervised_metric.md`.
Even when the learned weights recover the controlled task weighting, the
held-out planner metric does not outperform Euclidean distance in this proxy;
this separates task-direction identification from robustness to predicted
endpoint error.

The training-side follow-up
`experiments/empirical_spectral_audit/run_nonlinear_multihorizon_training_ablation.py`
compares matched one-step and autoregressive h=1/2/4 training objectives. Its
canonical report is
`results/nonlinear_planner_metric_ablation_20261003/multihorizon_training.md`.
The current evidence is mixed: multi-horizon training improves ID long-horizon
rollout/planning and OOD h=8, but can hurt OOD h=2/4. This is a horizon/domain
trade-off, not a blanket planning improvement claim.

The weight follow-up
`experiments/empirical_spectral_audit/run_nonlinear_multihorizon_weight_sweep.py`
compares one-step, uniform, short-horizon-favoring, and long-horizon-favoring
losses. Its canonical report is
`results/nonlinear_planner_metric_ablation_20261003/multihorizon_weight_sweep.md`.
No single profile dominates: short-favoring improves ID h=4/8 and OOD h=8 but
still loses on OOD h=2/4, while long-favoring makes that short/mid-horizon OOD
trade-off worse.

For the final cross-experiment audit, run
`experiments/empirical_spectral_audit/summarize_deep_claim_matrix.py`. It joins
the ten-seed alignment/equivariance factorial, checkpoint-matched nonlinear
alignment phase, multi-horizon loss profiles, external latent-dynamics
transport, and the frozen dynamics/residual summary. It does not pool rows or
confidence intervals across benchmarks. Its strict joint-success rule requires
every listed geometry and functional metric to improve for every paired seed;
the resulting Markdown is intended for the supplementary claim-boundary
section rather than as a new performance headline.

## Phase 9: Closed-loop physical closure

The frozen-checkpoint physical intervention and counterfactual scripts are
separate from offline latent transport:

```bash
.venv/bin/python experiments/run_pusht_physical_intervention.py --output results/pusht/lewm/physical_intervention_20261004 --pairs 128 --horizon 8 --bootstrap 2000

.venv/bin/python experiments/run_pusht_counterfactual.py --output results/pusht/lewm/counterfactual_corrected_20261004 --pairs 16 --calibration 64 --horizon 8 --batch-size 8

.venv/bin/python experiments/run_pusht_counterfactual_horizon_sweep.py --output results/pusht/lewm/counterfactual_horizon_sweep_20261004 --pairs 16 --calibration 64 --horizons 1 2 4 8 --batch-size 8
```

`normalized_action_sequence` independently normalizes the live `+a`, `-a`,
and zero arms; zero must not be represented by an all-zero z-score when the
public action mean is nonzero.  The horizon sweep fits a separate physical
readout per horizon and uses identical held-out states across horizons.  These
artifacts establish simulator action response and bounded frozen-checkpoint
latent sensitivity, but do not establish physical counterfactual accuracy,
causal controllability, online planning gain, or task success.

The physical closure benchmark
`experiments/run_pusht_learned_physical_dynamics.py` adds a separate learned
simulator-state transition arm and a frozen-LeWM visual initialization arm.
The contact-covered residual-target run is
`results/pusht/lewm/learned_physical_dynamics_residual_20261004/`; the
absolute-state and contact-coverage ablations are retained in
`learned_physical_dynamics_20261004/` and
`learned_physical_dynamics_contact_20261004/`. The canonical compact run
uses 256 sequential transitions, 128 visual training states, 16 held-out
counterfactual cases, horizons 1/2/4/8, and four matched planning episodes.
It shows that residual parameterization improves the privileged state arm at
one step, while frozen-latent physical initialization remains poor and all
tested MPC arms have zero task success. This is a simulator-level closure
diagnostic, not retraining evidence for the original LeWM predictor or a
universal world-model claim.

The predictor-side follow-up tests whether the remaining failure is caused by
autoregressive exposure bias. It freezes the visual encoder/projector, trains
only the action-conditioned predictor modules with teacher-forced latent
targets at every future step, and evaluates autoregressively on a disjoint
contact split:

```bash
.venv/bin/python experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_teacher_forced_20261004_seed8421 --train-cases 48 --held-out 16 --epochs 8 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --batch-size 32 --batch-size-encode 8 --seed 8421
.venv/bin/python experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_teacher_forced_20261004_seed9271 --train-cases 48 --held-out 16 --epochs 8 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --batch-size 32 --batch-size-encode 8 --seed 9271
.venv/bin/python experiments/summarize_pusht_contact_predictor_repairs.py --input results/pusht/lewm/contact_predictor_teacher_forced_20261004_seed8421/contact_predictor_teacher_forced_repair.json --input results/pusht/lewm/contact_predictor_teacher_forced_20261004_seed9271/contact_predictor_teacher_forced_repair.json --output results/pusht/lewm/contact_predictor_teacher_forced_summary_20261004
```

The paired calibrated $+a$ error improves at point-estimate level for both
seeds at horizons 2/4/8, with the clearest interval-separated gains at
horizon 4. Applying the original readout to repaired latents becomes worse,
so readout recalibration is a necessary control. This is a targeted repair
diagnostic for one checkpoint; it does not establish that the released frozen
model already has the repaired behavior, learned physical causality, online
planning gain, or task success. Targeted regression coverage is in
`tests/test_pusht_contact_predictor_teacher_forced_repair.py`.

The deep follow-up keeps the same two seeds and adds all three intervention
arms, an observed-latent interface upper bound, and calibration-size cases
`8 16 32 48`:

```bash
.venv/bin/python experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_deep_20261005_seed8421 --train-cases 48 --held-out 16 --epochs 8 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --calibration-cases 8 16 32 48 --batch-size 32 --batch-size-encode 8 --seed 8421
.venv/bin/python experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_deep_20261005_seed9271 --train-cases 48 --held-out 16 --epochs 8 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --calibration-cases 8 16 32 48 --batch-size 32 --batch-size-encode 8 --seed 9271
.venv/bin/python experiments/summarize_pusht_contact_predictor_deep.py --input results/pusht/lewm/contact_predictor_deep_20261005_seed8421/contact_predictor_teacher_forced_repair.json --input results/pusht/lewm/contact_predictor_deep_20261005_seed9271/contact_predictor_teacher_forced_repair.json --output results/pusht/lewm/contact_predictor_deep_summary_20261005
```

The explicit one-factor sensitivity artifacts are summarized by
`summarize_pusht_contact_predictor_ablation.py`. The deep result supports a
partial predictor/readout repair at h=4, but the observed-latent ceiling and
single-seed ablations keep the claim bounded: this is not evidence for
original-checkpoint causality, universal counterfactual accuracy, online
planning gain, or task success.

The 32-epoch training-strength audit repeats the enlarged 96/24 protocol at
three independent seeds and keeps the seed-level bootstrap intervals separate:

```bash
.venv/bin/python3 experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_deep32_20261005_seed8421 --train-cases 96 --held-out 24 --epochs 32 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --calibration-cases 16 32 64 96 --batch-size 64 --batch-size-encode 16 --seed 8421 --device cpu
.venv/bin/python3 experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_deep32_20261005_seed9271 --train-cases 96 --held-out 24 --epochs 32 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --calibration-cases 16 32 64 96 --batch-size 64 --batch-size-encode 16 --seed 9271 --device cpu
.venv/bin/python3 experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_deep32_20261005_seed13579 --train-cases 96 --held-out 24 --epochs 32 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --calibration-cases 16 32 64 96 --batch-size 64 --batch-size-encode 16 --seed 13579 --device cpu
.venv/bin/python3 experiments/summarize_pusht_contact_predictor_deep.py --input results/pusht/lewm/contact_predictor_deep32_20261005_seed8421/contact_predictor_teacher_forced_repair.json --input results/pusht/lewm/contact_predictor_deep32_20261005_seed9271/contact_predictor_teacher_forced_repair.json --input results/pusht/lewm/contact_predictor_deep32_20261005_seed13579/contact_predictor_teacher_forced_repair.json --output results/pusht/lewm/contact_predictor_deep32_summary_20261005
```

This audit preserves the bounded interpretation: all three $h=4$ intervals
exclude zero, but two of three $h=8$ intervals include zero even after latent
MSE improves. Longer training therefore does not establish physical causality,
accurate counterfactuals, online planning, or task success.

The decisive coverage follow-up doubles the contact-rich training/test split
to 192/48 while keeping the optimizer, action protocol, context, calibration,
and autoregressive evaluation fixed:

```bash
.venv/bin/python3 experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_moredata_20261005_seed8421 --train-cases 192 --held-out 48 --epochs 32 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --calibration-cases 16 32 64 128 192 --batch-size 64 --batch-size-encode 16 --seed 8421 --device cpu
.venv/bin/python3 experiments/run_pusht_contact_predictor_teacher_forced_repair.py --output results/pusht/lewm/contact_predictor_moredata_20261005_seed9271 --train-cases 192 --held-out 48 --epochs 32 --horizons 1 2 4 8 --train-horizons 1 2 4 8 --calibration-cases 16 32 64 128 192 --batch-size 64 --batch-size-encode 16 --seed 9271 --device cpu
```

Both independent runs close the h=8 paired-gain interval under this targeted
repair protocol. Optional `--free-run-weight` and `--pose-loss-weight` expose
the two rejected alternatives: free-running loss improves latent fit without
closing the physical interval, and an auxiliary pose head only helps short
horizons. Neither option changes the claim boundary for the released frozen
checkpoint.

The closed-loop bridge is implemented separately in
`experiments/run_pusht_repaired_predictor_mpc.py`.  Unlike the constant-arm
repair above, it trains on five contact-mode action-sequence arms per case
(constant, reverse, piecewise noisy, and random), then compares the original
and repaired predictors on identical live MPC candidate proposals:

```bash
.venv/bin/python experiments/run_pusht_repaired_predictor_mpc.py --output results/pusht/lewm/repaired_predictor_mpc_compositional_20261005 --train-cases 96 --held-out 24 --repair-epochs 32 --episodes 4 --prediction-horizon 8 --action-block 2 --candidates 32 --state-source visual --cost task
```

The matched candidate-ranking audit is
`experiments/run_pusht_repaired_predictor_candidate_rank.py`.  It replays each
candidate from the same held-out simulator state, so it separates predictor
ranking from candidate-set coverage:

```bash
.venv/bin/python experiments/run_pusht_repaired_predictor_candidate_rank.py --output results/pusht/lewm/repaired_predictor_candidate_rank_20261005 --train-cases 96 --held-out 24 --episodes 16 --repair-epochs 32 --horizon 8 --candidates 64 --action-block 2
```

In the completed audit, compositional repair reduced held-out autoregressive
$h=8$ latent MSE from $0.860$ to $0.381$ and increased matched candidate
ranking Spearman correlation from $0.107$ to $0.690$; selected regret fell
from $188.1$ to $32.9$ task-cost units.  The oracle best candidate was still
about $169.6$, far from the environment success region, and increasing the
candidate budget to 512 reduced it only to $148.9$.  The live bridge therefore
shows a repaired ranking interface but zero task success under this candidate
protocol.  This is positive evidence for predictor-side ranking repair and
negative evidence for the current action-set coverage; it is not a universal
online-planning claim.  The privileged-state/block-cost matrix is retained as
a diagnostic control, not as a vision result.  Regression coverage is in
`tests/test_pusht_repaired_predictor_mpc.py`.

The follow-up diagnosis separates long-horizon physical-model error from the
planning objective. `run_pusht_multihorizon_contact_closure.py` fits direct
1/2/4/8-step simulator-state maps on contact-balanced states; its canonical
artifact is `results/pusht/lewm/multihorizon_contact_closure_2048x24_20261005/`.
The direct maps do not beat persistence at long horizons, so repeated one-step
composition is not the sole explanation for the negative result.

`run_pusht_two_stage_oracle_planning.py` is the matched privileged-state
planning diagnosis. The coupled arm ranks joint agent/block endpoint cost; the
two-stage arm ranks block pose first and then returns the agent around the
block. Canonical seed-8421 and seed-9173 artifacts are
`results/pusht/lewm/two_stage_oracle_planning_8x96_20261005/` and
`results/pusht/lewm/two_stage_oracle_planning_8x96_seed9173_20261005/`.
Across both seeds the two-stage arm improves block and agent geometry, but
full task success is not yet stable. This is a partial privileged planning
repair, not frozen-LeWM online-planning evidence.

The structural repair is implemented as receding block-pose planning in
`run_pusht_receding_block_pose_planner.py`: the planner executes only a short
prefix, replans from the observed simulator state, stops the push phase at the
conjunctive block-pose criterion, and invokes a bounded detach/return phase only
after closure. A failed baseline push now triggers two independent `512 x 8`
push rescues from the same initial state. A failed ordinary return triggers a
deep `1024 x 16 x 24` terminal rescue, with an explicit intermediate safety
tube (`XY <= 21 px`, angle `<= pi/9 + 0.02`) and strict final PushT success.
On the same 16 stable-contact cases, task and block-pose success increase from
`0/16` to `16/16`; mean final block XY decreases from `141.74` to `11.25` px,
while mean final agent XY is `12.00` px for the repair versus `7.00` px for
the coupled arm. The paired repair-minus-baseline success difference is
`1.0000` (paired bootstrap 95% CI `[1.0000, 1.0000]`). The matched artifacts
are under
`results/pusht/receding_block_pose_push_deep_rescue_matched_8x128_seed8421_20261006/`
and `results/pusht/receding_block_pose_push_deep_rescue_matched_8x128_seed9173_20261006/`;
paired statistics are in
`results/pusht/planning_repair_paired_statistics_deep_rescue_20261006/`.
This is a high-budget privileged simulator-state planning-interface result,
not a visual LeWM, learned-physical-adapter, or general PushT performance
claim.

The matched rescue protocol is reproducible with:

```bash
.venv/bin/python experiments/run_pusht_receding_block_pose_planner.py --output results/pusht/receding_block_pose_push_deep_rescue_matched_8x128_seed8421_20261006 --episodes 8 --seed 8421 --population 128 --cem-iterations 2 --push-rescue-population 512 --push-rescue-iterations 8 --push-rescue-restarts 2 --max-steps 80 --closure-return-population 256 --closure-return-iterations 2 --closure-return-horizon 8 --closure-return-rescue-population 512 --closure-return-rescue-iterations 8 --closure-return-rescue-restarts 2 --closure-return-deep-population 1024 --closure-return-deep-iterations 16 --closure-return-deep-horizon 24 --closure-return-deep-restarts 2
.venv/bin/python experiments/run_pusht_receding_block_pose_planner.py --output results/pusht/receding_block_pose_push_deep_rescue_matched_8x128_seed9173_20261006 --episodes 8 --seed 9173 --population 128 --cem-iterations 2 --push-rescue-population 512 --push-rescue-iterations 8 --push-rescue-restarts 2 --max-steps 80 --closure-return-population 256 --closure-return-iterations 2 --closure-return-horizon 8 --closure-return-rescue-population 512 --closure-return-rescue-iterations 8 --closure-return-rescue-restarts 2 --closure-return-deep-population 1024 --closure-return-deep-iterations 16 --closure-return-deep-horizon 24 --closure-return-deep-restarts 2
.venv/bin/python experiments/summarize_pusht_planning_repair.py --seed8421 results/pusht/receding_block_pose_push_deep_rescue_matched_8x128_seed8421_20261006/receding_block_pose_planning.json --seed9173 results/pusht/receding_block_pose_push_deep_rescue_matched_8x128_seed9173_20261006/receding_block_pose_planning.json --output results/pusht/planning_repair_paired_statistics_deep_rescue_20261006
```

The learned-adapter follow-up is implemented in
`run_pusht_object_centric_learned_planner.py`.  It compares full state/action,
object-centric contact geometry, and action-history inputs under the same
candidate bank.  The default `joint_block_pose` objective reproduces the
agent-drift failure; `--planning-objective contact_gated` acquires predicted
contact before optimizing block pose; and `--planning-objective staged_contact`
adds bounded geometric acquisition and delayed return as a diagnostic
control:

```bash
.venv/bin/python experiments/run_pusht_object_centric_learned_planner.py --output results/pusht/lewm/object_centric_learned_planner_contact_gated_768_8_20261020 --train-transitions 768 --episodes 8 --trees 96 --candidates 96 --horizon 6 --action-block 2 --max-steps 60 --planning-objective contact_gated --seed 20261020
.venv/bin/python experiments/run_pusht_object_centric_learned_planner.py --output results/pusht/lewm/object_centric_learned_planner_contact_gated_768_8_seed20261021_20261020 --train-transitions 768 --episodes 8 --trees 96 --candidates 96 --horizon 6 --action-block 2 --max-steps 60 --planning-objective contact_gated --seed 20261021
```

The two-seed artifact shows that contact gating reduces object-centric agent
XY error by about 1201 pixels relative to the joint endpoint objective, while
the paired block-XY interval includes zero and task success remains zero.
This isolates objective coupling from the surviving learned contact-push and
long-horizon rollout failures; it is not frozen-LeWM visual planning evidence.
The full runs and staged control are recorded in the corresponding
`object_centric_learned_planner_*` directories, with regression coverage in
`tests/test_pusht_object_centric_learned_planner.py`.

The final action-coverage control uses
`--training-protocol contact_intervention`, `--planning-objective
staged_contact_orient`, horizon 1, and one action per replan.  It adds a
`block_only` adapter that predicts block pose effects while retaining the
observed agent state:

```bash
.venv/bin/python experiments/run_pusht_object_centric_learned_planner.py --output results/pusht/lewm/object_centric_learned_planner_contact_intervention_768_8_20261020 --train-transitions 768 --episodes 8 --trees 96 --candidates 96 --horizon 1 --action-block 1 --max-steps 120 --training-protocol contact_intervention --planning-objective staged_contact_orient --seed 20261020
.venv/bin/python experiments/run_pusht_object_centric_learned_planner.py --output results/pusht/lewm/object_centric_learned_planner_contact_intervention_768_8_seed20261021_20261020 --train-transitions 768 --episodes 8 --trees 96 --candidates 96 --horizon 1 --action-block 1 --max-steps 120 --training-protocol contact_intervention --planning-objective staged_contact_orient --seed 20261021
```

This arm improves local block translation but remains at zero task success on
16 paired cases, so it isolates rather than hides the unresolved orientation
and contact-mode closure.

As a stage-gate control, `staged_contact_recovery` widens the learned planner's
contact gate from the strict diagnostic boundary (45 px) to the settled
contact manifold (65 px), preventing already-valid contact states from being
sent back to acquisition. The paired runs are stored under
`results/pusht/lewm/object_centric_learned_planner_recovery_gate_768_8_20261006/`
and its seed-20261021 counterpart. The block-only arm reaches mean block XY
errors of `182.70` and `175.22` px across the two seeds, but task success
remains `0/16`; therefore the stage-gate correction is necessary but not a
sufficient learned orientation/contact closure.

The next objective repair is `staged_contact_local_effect`. It retains routed
acquisition and learned translation, then probes a fixed 24-direction
contact-force fan with one current-state simulator step whenever block XY is
below 45 px but orientation is still open. Probe ranking uses a hard 45-px
translation tube, so orientation cannot be purchased by sacrificing a solved
translation. This is a privileged simulator-state local-effect diagnostic,
not frozen-LeWM visual planning. The two-seed 32-case summary is stored at
`results/pusht/lewm/learned_local_effect_repair_summary_20261006/`:

```bash
.venv/bin/python experiments/run_pusht_object_centric_learned_planner.py --output results/pusht/lewm/object_centric_learned_planner_local_effect_768_16_seed8421_20261006 --train-transitions 768 --episodes 16 --trees 96 --candidates 64 --horizon 6 --action-block 1 --max-steps 60 --planning-objective staged_contact_local_effect --local-effect-directions 24 --bootstrap 4000 --seed 8421
.venv/bin/python experiments/run_pusht_object_centric_learned_planner.py --output results/pusht/lewm/object_centric_learned_planner_local_effect_768_16_seed20261021_20261006 --train-transitions 768 --episodes 16 --trees 96 --candidates 64 --horizon 6 --action-block 1 --max-steps 60 --planning-objective staged_contact_local_effect --local-effect-directions 24 --bootstrap 4000 --seed 20261021
.venv/bin/python experiments/summarize_pusht_local_effect_repair.py --local results/pusht/lewm/object_centric_learned_planner_local_effect_768_16_seed8421_20261006/object_centric_learned_planner.json results/pusht/lewm/object_centric_learned_planner_local_effect_768_16_seed20261021_20261006/object_centric_learned_planner.json --hybrid results/pusht/lewm/object_centric_learned_planner_hybrid_local_effect_768_16_seed8421_20261006/object_centric_learned_planner.json results/pusht/lewm/object_centric_learned_planner_hybrid_local_effect_768_16_seed20261021_20261006/object_centric_learned_planner.json --output results/pusht/lewm/learned_local_effect_repair_summary_20261006
```

The block-only and specialist arms reach `5/32` and `6/32` task successes,
respectively, but their paired success-rate intervals against the same
state/action arm include zero. Mean angle error remains about one radian. The
`block_agent_hybrid` rollout, which additionally predicts agent XY inside the
candidate horizon, is a negative ablation (`3/32`, 142.26-px block XY), so
stale agent geometry is not the sole cause. The validated solution to the
original joint endpoint failure remains the high-budget privileged receding
block-pose controller; the learned local-effect result is only a partial
mitigation and must not be reported as general PushT success.

The follow-up torque-effect identifiability audit is implemented in
`audit_pusht_torque_effect_identifiability.py`. It samples genuine
approach/contact states, evaluates a complete 97-action fan (32 directions at
three magnitudes plus zero), and stratifies both contact distance and current
block-goal distance. It records normal/tangential action components, block
translation, angle change, and whether any action improves angle while staying
inside both a monotone tube (`min(45 px, current XY + 4 px)`) and a broad 45-px
tube. Across two independent seeds, 72 states per seed were audited with all
action fans complete and finite. Stable-contact feasible-action rates were
0.42--0.83 across the three distance strata; near-contact rates were
0.25--0.83. This establishes that local torque actions exist in the simulator
and that the learned-adapter failure is an action-effect ranking/calibration
problem, not evidence that the contact task is locally uncontrollable. The
machine-readable audits are under
`results/pusht/lewm/torque_effect_identifiability_20261022/` and
`results/pusht/lewm/torque_effect_identifiability_20261023/`.

The corresponding learned repair is the
`block_only_torque_readout` arm under
`--planning-objective staged_contact_torque_readout`. It predicts one-step
block `dx`, `dy`, and wrapped `dtheta` from balanced contact interventions,
uses a translation-first phase, then applies a hard translation tube and a
zero-action safety fallback when no predicted torque action is feasible. The
matched two-seed, 32-case summary is under
`results/pusht/lewm/torque_effect_readout_summary_20261006/`. It reduces mean
block XY from 118.64 to 51.32 px relative to the state/action arm, with paired
difference `-67.32` px (95% CI `[-108.70,-28.46]`), but task success changes
from `3/32` to `0/32` and angle remains `1.22` rad. Thus the new readout
successfully extracts local translational effect but does not yet solve
learned orientation closure. The claim-safe conclusion is: the joint endpoint
objective is solved at the validated privileged receding-planner layer, while
the learned adapter still lacks calibrated torque closure. The readout is a
diagnostic repair, not a visual LeWM or general PushT result.

The proposal--verification implementation then underwent a state-consistency
audit. The preliminary `14/32` result was discarded because probe rollouts
were not followed by restoration of the actual current simulator state before
the selected action was committed. The corrected `--planning-objective
staged_contact_verified_torque` arm reaches `6/32` strict successes (2/16 and
4/16), with mean block XY `71.78` px, angle error `0.778` rad, and agent XY
`156.58` px. The corrected artifacts are under
`results/pusht/lewm/object_centric_learned_planner_verified_torque_restored_768_16_seed8421_20261006/`
and
`results/pusht/lewm/object_centric_learned_planner_verified_torque_restored_768_16_seed20261021_20261006/`.

The terminal repair is `--planning-objective
staged_contact_verified_torque_stabilized`. It adds translation hysteresis,
state-consistent all-fan terminal verification, an inertia-aware
least-violation fallback, and a separate closure-preserving return phase. It
obtains `3/32` strict successes (0/16 and 3/16), while reducing mean block XY
to `57.71` px. The safety estimand is positive but distinct from task success:
lock-time acquisition is `0/32`, and all five terminal-return episodes
preserve block closure. The result remains a privileged simulator-state
learned-planner audit; it is not frozen-LeWM visual planning, general PushT
performance, or learned physical counterfactual evidence. The summary is in
`results/pusht/lewm/terminal_stabilization_summary_20261006/`.

We also tested the specific stage-mismatch hypothesis with
`--planning-objective staged_contact_hierarchical`: while block-XY is at least
45 px from the goal, the arm uses the learned block-pose adapter; only inside
the near-goal tube does it use verified local torque proposals. On the same
32 paired cases this negative ablation reaches `4/32`, versus `6/32` for the
corrected top-eight verification arm, and increases mean block XY from
`71.78` to `105.08` px (paired difference `+33.30` px, 95% CI
`[+5.37,+61.77]`). Preventing the torque readout from handling long-range
translation is therefore a necessary interface principle, but is not
sufficient for learned orientation closure. This arm is retained as a
negative ablation rather than promoted as the repair. Its machine-readable
summary is under
`results/pusht/lewm/hierarchical_translation_ablation_20261006/`.

## Six-goal acceptance audit (2026-10-08)

The acceptance matrix for the six proposed upgrades is generated by
`experiments/build_six_goal_evidence_matrix.py` and stored in
`results/six_goal_evidence_20261008/`. It keeps alignment-based selection,
learned pose closure, frozen-latent action-effect prediction, visual online
planning, cross-environment transfer, and the joint transfer diagnostic as
separate estimands. The current statuses are `partial`, `partial_negative`,
`partial`, `blocked`, `partial_proxy`, and `partial_diagnostic`, respectively;
privileged simulator-state results are not used to fill missing visual or
causal evidence.

Two fresh frozen-checkpoint counterfactual sweeps are stored under
`results/pusht/lewm/counterfactual_horizon_sweep_20261011_20261008/` and
`results/pusht/lewm/counterfactual_horizon_sweep_20261012_20261008/`. They use
disjoint calibration and held-out paired `+a/-a/zero` interventions at
horizons 1, 2, 4, and 8. Latent separation is nonzero and increases with
horizon, but decoded physical pose error remains large; this is bounded latent
action-associated evidence, not learned physical causality.

The fresh runtime-only visual planning control is
`results/pusht/lewm/visual_only_online_planning_20261008_seed20261013/`. The
policy receives pixel observations and a pixel goal at runtime; simulator
state is recorded only for evaluation. Across four episodes, vanilla visual
MPC had zero strict successes, so this artifact is a negative boundary for the
visual-planning claim rather than a task-success result.

## Enriched covariance--predictor bridge audit (2026-10-11)

`empirical_spectral_audit/run_enriched_action_task_bridge.py` now supports a
paired observed--zero-action effect metric, a deterministic local
spectrum-preserving candidate family, and augmented state--action geometry for
the row-wise interface `x_next = x A_x + u B_u`. The paired effect is fitted
only on training episodes; model selection uses episode-disjoint validation;
held-out four-step transfer is never used to fit a metric or choose a
candidate. The augmented operator is evaluated in column convention as
`[[A_x.T, B_u.T], [0, 0]]`, so Schur and transient diagnostics do not silently
drop the action channel.

The final two-environment artifact is
`results/enriched_cross_environment_action_task_audit_rankaudit257_20261011/`.
The corresponding per-environment reports are under
`results/enriched_action_task_bridge_pusht_multi_rankaudit257_20261011/`
and `results/enriched_action_task_bridge_pointmaze_rankaudit257_20261011/`.
The local family yields genuine matched candidate pools (PushT minimum/mean
253/255.8; PointMaze 4/192.4 across five splits). Augmented effect-transient
regret is 0.03199 for PushT and 0.00662 for PointMaze, but it does not
uniformly beat validation MSE or Euclidean controls. PointMaze's zero-action
export contains an unpooled source-token serialization in all 100 control
rows; the audit canonicalizes the observed source vector and records this
shape mismatch rather than treating it as an action effect. The matched-pool
rank audit is also negative: observed-zero effect alignment has mean Spearman
rho `-0.125` on PushT and `-0.328` on PointMaze; augmented effect-transient
rho is near zero (`0.004` and `0.033`).

This is a measurement/interface repair and a stronger negative audit, not a
universal model-selection certificate: action sensitivity, visual-only
observability, and cross-environment generality remain conditional.
# Cross-architecture enriched learned-model export

The cross-architecture export contract is implemented in
`empirical_spectral_audit/enriched_record_schema.py`. It requires the same
episode-level fields for each architecture: source latent/state/proprio,
raw and normalized action blocks, and predicted/target latent plus raw target
state for every declared horizon. `canonicalize_enriched_export.py` performs
only a lossless migration of older enriched JEPA artifacts; vector-only
historical DINO exports are explicitly classified as
`incomplete_vector_only_export`. `split_enriched_export.py` creates
deterministic episode-disjoint fit/outcome exports, and
`build_learned_model_selection_records.py` converts those pairs to the strict
`real_checkpoint_selection.v1` format.

The resulting audit is under `results/cross_arch_enriched_20261011/`. The
canonical pool now contains real JEPA-WM PushT/PointMaze and repo-local
compatibility-rerun DINO-WM PushT/PointMaze exports. Historical DINO vector
files remain separately classified as incomplete. The two architectures are
therefore present in both environments, but the strict matcher remains gated:
their learned predictor spectra and prediction-quality scales are not matched,
and only one seed is available per architecture.
# Signed-torque contact audit (2026-10-12)

`run_pusht_signed_torque_audit.py` isolates the remaining PushT terminal
failure into a root-disjoint signed-effect task.  It compares a geometric
centroid sign, a learned signed-angle regressor/classifier, and a multi-output
angle/XY effect model on the same candidate pulse bank.  The runtime audit
uses only the current simulator observation and a declared contact chart; it
does not query candidate successors for selection.

The full body-frame run is stored in
`results/negative_cause_diagnosis_20261012/signed_torque_bodyframe_full_seed9181/`.
It reaches held-out model direction accuracy `0.722` versus geometric
direction accuracy `0.528`, but strict terminal-start success remains `0/32`
with final XY `36.07 px` and angle error `0.972 rad`.  The result is therefore
a mechanism-level repair of signed direction ranking, not stable posture
closure.  The matched support-boundary protocol additionally shows that a
fixed terminal support can lose angular authority or incur large translational
drift; this motivates contact relocation and a staged terminal controller.

The old endpoint-ranker and RGB adapter artifacts remain unchanged.  These
results must not be promoted to visual planning, causal LeWM dynamics, or a
universal world-model claim.

# Lifted contact-effect adapter (2026-10-12)

The literature audit motivated an explicit lift of the contact/action chart:
body-frame geometry, velocity and action-history coordinates, Fourier phase
features, bilinear state/action terms, and distance-local RBFs.  The new
implementation is `run_pusht_lifted_contact_adapter.py`, with regression tests
in `tests/test_pusht_lifted_contact_adapter.py`.

The first root-disjoint smoke artifact is
`results/negative_cause_diagnosis_20261012/lifted_contact_adapter_smoke_seed9181/`.
On 8 held-out roots, lifted direction accuracy is `0.875` (95% bootstrap CI
`[0.625, 1.000]`) and learned ranking regret is approximately `0.00074`, but
pure learned block-pose closure is still `0/8`; therefore the lift improves
local separability without solving contact execution.  The horizon-matched
variant executes the same four-step pulse horizon used for labels and reduces
the mean angle error relative to the mismatched one-step deployment, but still
has `0/8` closure.  The dual-progress guard reduces translation drift but
increases ranking regret and also remains `0/8`.

This is a useful negative mechanism result: explicit dimensional expansion can
separate signed local effects while the remaining contact-mode/closed-loop
interface still prevents posture closure.  The optional `--successor-verify-topk`
arm is a separate privileged proposal-plus-verifier protocol; it must not be
merged with the pure learned result or described as frozen visual planning.

The complete two-seed replication is stored in
`results/negative_cause_diagnosis_20261012/lifted_contact_adapter_full_seed9181/`
and
`results/negative_cause_diagnosis_20261012/lifted_contact_adapter_full_seed9173/`.
The same protocol gives block-pose closure `4/32` and `3/32` respectively
(`7/64 = 10.94%` pooled), with direction accuracy `0.854` and `0.911`, and
learned ranking regret `0.0247` and `0.0277` versus geometric regret `0.0622`
and `0.0756`.  This is replicated improvement in the local ranking gate, not
stable learned posture closure; the original negative strict-success and
frozen-visual-planning claims remain unchanged.

## Lift--separate--project continuation

The next experiment implements the proposed “expand information into a richer
space, separate it there, and project back” principle explicitly. The
temporal protocol uses root-disjoint executed sequences with shared candidate
banks: body-frame contact coordinates, velocity/action phase features,
bilinear state-action terms, distance-local RBFs, and a genuinely observed
previous-action delay coordinate are lifted before fitting the effect model.
The projection is tested separately with direction selection and with a
multi-output pose metric using explicit XY and angle weights.

The runner is run_pusht_lifted_contact_adapter.py, and the pooled audit is
results/negative_cause_diagnosis_20261012/lifted_contact_pooled_audit_20261012/.
Across two seeds and 16 held-out cases per projection family, temporal lifting
raises local direction accuracy to approximately 0.896 for the direction
projection and 0.936 for the pose projection. It does not close the control
loop: all three full projection groups obtain only 1/16 block-pose success,
with contact-loss rates 0.831--0.858.

The result isolates the remaining failure more sharply. Expansion improves
local action-effect separability, while projection back to a multi-step
contact controller remains non-robust because the learned controller leaves
the contact manifold and cannot infer a reliable successor mode. The optional
state-only contact-recovery fallback is reported as a hybrid ablation and is
not part of the pure learned claim. These runs therefore do not establish
causal latent dynamics, frozen visual planning, stable learned posture
closure, or a universal world model.

### Support-normal recovery repair

The pose-regressor temporal protocol exposed an implementation mismatch in its
contact-loss fallback: deployment pointed toward the block centroid, whereas
the training/action bank used the support-boundary inward normal. The fallback
now calls the same declared support-normal interface through
`contact_recovery_action`; this invariant is covered by
`tests/test_pusht_lifted_contact_adapter.py`. On the same two seeds, with no
successor verification, the repaired runs are stored in
`results/negative_cause_diagnosis_20261012/lifted_pose_regressor_supportnormal_seed9173/`
and
`results/negative_cause_diagnosis_20261012/lifted_pose_regressor_supportnormal_seed9181/`.
Final XY decreases from `108.84/90.83 px` to `48.19/30.23 px`, and strict
block-pose closure changes from `2/32, 4/32` to `2/32, 8/32`. Final angle error
remains `1.64/1.35 rad`; this is a validated interface repair, not stable
learned pose closure. The pre-repair runs remain the paired baseline and are
not overwritten.

## Literature-driven mode/precondition and executable-ranking audit (2026-10-12)

The complete evidence map is
`results/negative_cause_diagnosis_20261012/literature_evidence_map_20261012.md`.
It distinguishes contact-mode planning, learned preconditions, local visual
foresight, uncertainty-aware MPC, dynamics-supervised visual models, and
physics-informed world models. The conclusion is that prior successful systems
add an explicit mode/precondition/dynamics interface; none establishes stable
PushT pose closure for a frozen RGB LeWM latent with no tactile signal and no
runtime simulator successor.

The runner
`experiments/run_pusht_learned_sequence_verifier.py` now includes:

- root-disjoint train/calibration/test state splits;
- an executable-task pairwise target requiring contact retention, effective
  motion, and XY non-regression before angle progress;
- staged translation/terminal-orientation hybrid control;
- a fixed calibration gate that falls back to mode-receding when any exposed
  consequence regresses.

Across seeds 9173 and 9181, executable ranking improves contact retention but
does not stabilize angle closure; the hybrid is admitted by calibration for
one seed and rejected for the other. The artifacts are
`results/negative_cause_diagnosis_20261012/taskrank_hybrid_gate_seed9173/` and
`results/negative_cause_diagnosis_20261012/taskrank_hybrid_gate_seed9181/`.
These are mechanism-level repairs and remain separate from the older negative
angle-only, KNN, local-calibration, and predictor-repair results.

## Protocol-corrected lifted temporal audit (2026-10-12)

The first temporal audit exposed two protocol artifacts that can create false
negative evidence: some exports contained only one causal window per episode,
and memory lengths were compared on different current transitions.  The
corrected PointMaze export uses 100 held-out episodes with four within-episode
windows each (`400` records), while the corrected PushT export uses all 21
validation episodes with four windows each (`84` records).  All feature and
metric fitting remains episode-disjoint from outcome evaluation.

The memory audit now reports both the historical unpaired values and a paired
comparison on the same current transitions supported by the largest declared
history.  The latter is the only value admitted to the memory gate.  The
implementation and tests are:

- `experiments/empirical_spectral_audit/run_real_enriched_hybrid_audit.py`
- `experiments/empirical_spectral_audit/run_lifted_signal_information_audit.py`
- `experiments/empirical_spectral_audit/summarize_protocol_corrected_temporal_audit.py`
- `tests/test_real_enriched_hybrid_audit.py`
- `tests/test_lifted_signal_information_audit.py`

The three-split PointMaze summary is
`results/protocol_corrected_temporal_export_20261012/jepa_pointmaze_windows4_100_multiseed_summary.json`.
The ten-split PushT composite summary is
`results/protocol_corrected_temporal_export_20261012/jepa_pusht_windows4_composite_multiseed_summary.json`.
On PushT, delay and state+action time-frequency lifting win in `10/10` seeded
episode splits, with mean matched RMSE reductions of about `15.5%` and `24.9%`;
paired memory-4 wins `8/10`.  Interaction lifting wins only `3/10` after
fit-only rank matching and has mean matched reduction `-7.4%`, so its negative
result is not repaired by the corrected protocol.  A follow-up that adds the
full past action history to the finite-memory model is stored under
`results/projected_hybrid_physics_audit_20261012_action_history/` and reduces
paired memory-4 to `1/10`, identifying low-sample/high-dimensional action
history as an overfitting mechanism rather than a successful fix.  These are
all layers of the same unified covariance--predictor--memory--input--output
diagnostic; they do not establish physical causality, visual online planning,
or a universal world model.

## Mode-aware shared-candidate PushT adapter audit (2026-10-13)

`run_pusht_mode_aware_shared_candidate.py` implements the next mechanism-level
test after the literature reassessment. It separates offline successor-mode
labels from within-mode transition effects, uses the same candidate bank for a
global one-step control and a mode-aware selector, and records a short
lookahead, mode risk, contact-loss rate, and independent closed-loop pose
success. Runtime selection does not query simulator successors. This remains a
privileged simulator-state prerequisite for the RGB/frozen-LeWM interface.

Example smoke command:

```bash
.venv/bin/python experiments/run_pusht_mode_aware_shared_candidate.py \
  --output results/negative_cause_diagnosis_20261013/mode_short_pulse1_fan_smoke_seed20261014 \
  --train-roots 8 --held-out-roots 8 --sequence-steps 8 --pulse-steps 1 \
  --closure-steps 40 --trees 48 --lookahead 1 \
  --inward-magnitude 0.12 --tangent-magnitudes 0.12 0.24 0.40 \
  --seed 20261014
```

The evidence map and claim boundaries are recorded in
`results/negative_cause_diagnosis_20261013/literature_mechanism_to_code_20261013.md`.

### Follow-up: fit-only memory selection and successor-mode experts (2026-10-13)

The adaptive memory audit now contains a nested, fit-only selector.  It splits
the outer fit episodes again, selects among raw history, local event, spectral
phase, and historical first-harmonic features, then refits on the complete
outer-fit set before reading the held-out episodes.  This prevents the
historical Fourier control from being replaced using held-out outcomes.  The
append-only evaluation is in
`results/adaptive_time_frequency_memory_selector_20261013/summary.json`:
the selector improves the current-state/action baseline on all 10 outer
splits, but does not reliably exceed the fixed first-harmonic control.  The
conditional transfer result is therefore retained; a universal memory claim
is not admitted.

The mode-aware adapter was also changed from one regressor per current mode to
successor-mode experts with a fixed-support fallback hierarchy.  This tests
whether mixing stalled/sliding/separation effects caused torque ranking to
fail.  The smoke artifact is
`results/negative_cause_diagnosis_20261013/mode_successor_mixture_smoke_seed20261015/`.
It reduces contact-loss rate relative to the paired global selector in that
smoke and slightly reduces angle error, but pose success remains `1/8`; the
result is a mechanism-level partial repair, not stable pose closure.  The same
artifact also shows zero `separation` successor support, so the separation
expert cannot be identified under this collection protocol.

With the explicit RGB-independent recovery branch enabled, the same mixture
artifact family reaches zero measured contact-loss in the smoke, but spends
about half of its 36-step budget in recovery and still closes pose in only
`1/8` cases.  Increasing the recovery pulse to `0.20` does not improve this
(`1/8`, mean angle about `1.05` rad).  Thus contact retention and pose closure
are now separated empirically: the remaining bottleneck is recovery-to-torque
handoff/angle ranking, not merely loss of contact.  These runs remain
privileged mechanism diagnostics and are not frozen visual-planning evidence.

Adding an explicit outward probe to the shared bank fixes the zero-support
problem: the probe smoke reaches `0.914` successor-mode accuracy, reduces
mode-aware ranking regret to `0.0146` versus `0.0251` for the global selector,
and reduces contact-loss from `3/8` to `0/8`.  It still yields `0/8` pose
closures.  A wider tangent fan (`0.12, 0.24, 0.40`) worsens contact retention
and does not improve closure, so it is retained as a negative action-coverage
control rather than promoted.

Trace-level auditing found that the probe was being committed during the
approach phase, so the selector could trade reacquisition for an outward
translation.  The implementation now applies a fixed contact/approach safety
lease and keeps the probe for identification only.  A separate terminal
geometric-sign handoff was tested but did not improve closure (`1/8`) and is
therefore disabled by default; it remains available only with
`--terminal-handoff` as a negative ablation.  The remaining failure is
terminal torque-effect identifiability under contact, not an unmeasured
separation class.

### Follow-up: observed sign calibration and task-directed recovery (2026-10-13)

The runtime interface now supports a single short contact probe that estimates
the observed `moment -> angle` sign from the committed action and the next
observed angle increment.  An unresolved probe is fail-closed and is never
repeated indefinitely.  Calibration state is tracked by contact mode, because
the trace audit found that one episode-level sign can change after a contact to
approach transition.  The trace additionally records block/goal coordinates,
desired angle sign, action moment sign, observed angle increment, calibration
status, and wrong-sign events.  The relevant flags are
`--observed-sign-calibration` and `--observed-sign-feedback`.

The shared candidate bank also has an explicit, opt-in task-directed tangent
and translation extension (`--goal-translation-magnitudes`).  The recovery
branch can alternatively target the contact point opposite the desired block
translation (`--goal-directed-recovery`); both are privileged-state
mechanism ablations and are kept separate from the legacy bank.

The first paired smoke artifacts are
`results/negative_cause_diagnosis_20261013/mode_gated_no_recovery_smoke_seed20261015/`,
`results/negative_cause_diagnosis_20261013/goal_translation_sign_calibration_v2_smoke_seed20261015/`,
and
`results/negative_cause_diagnosis_20261013/goal_recovery_sign_calibration_smoke_seed20261015/`.
Mode-gating and sign calibration reduce mode-aware final XY from about `69.5`
to `57.0` pixels in the matched four-root smoke without recovery, while pose
success remains `0/4`; task-directed candidate/recovery variants do not yet
produce stable closure.  These results identify the remaining bottleneck as
mode-to-task-space reacquisition and action-effect ranking, not a solved pose
adapter.  All earlier negative artifacts remain unchanged.

### Follow-up: polygon-contact protocol and geometry-conditioned adapter (2026-10-13)

The next audit found a protocol confound in the previous mode definition.  A
center-radius root (`||pusher-block||=42`) is not a contact root for the
non-convex T: the actual polygon-boundary gap ranged from about 1 to 42 px.
The implementation therefore adds `--root-protocol boundary_contact`, which
samples a polygon edge using the environment's fixed T geometry, computes the
outward normal from polygon winding, rejects invalid roots, and verifies a
12--18 px boundary gap.  Runtime mode, recovery, and sign-feedback gates use
the boundary gap (`contact <=24`, `approach <=70`) rather than center distance.
The legacy protocol and its artifacts are preserved.

The adapter now also contains a fixed physical contact chart based on the
nearest polygon edge, edge-position bin, body-frame lever arm, edge tangent,
normal, and action moment.  It is contact-only; approach/free states fall back
to the historical tree model.  A separate opt-in local sign table and
observed same-region/action-bin effect cache are available through
`--contact-geometry-sign-feedback`, `--recent-contact-sign-fallback`, and
`--online-effect-correction`.  All are runtime observation interfaces and do
not query simulator successors.

The 12-root artifact
`results/negative_cause_diagnosis_20261013/contact_boundary_chart_12root_seed20261016/`
shows the physical chart improving held-out one-step regret from `0.0702` to
`0.0645` against the paired global selector, but not improving closed-loop
success (`2/12` for both).  This is a conditional ranking improvement, not
pose closure.

After validating the root and gate fixes, the 8-root artifact
`results/negative_cause_diagnosis_20261013/boundary_contact_validated_recovery_seed20261016/`
reports boundary-aware contact loss `0/8`, final XY `26.6` for the global
selector and `35.8` for the chart selector, but strict pose success only
`2/8` and `1/8`, respectively.  Short execution pulses (`pulse_steps=1`)
reduce recovery budget and ranking regret, yet do not produce stable closure;
the physical-sign and online-effect smoke artifacts remain negative/partial.
Consequently, these changes repair root/mode measurement and improve contact
retention, but the learned adapter is still not admitted as a stable pose
controller or as frozen visual-planning evidence.

### Follow-up: action-frame consistency and contact-force audit (2026-10-13)

The next trace audit found two additional protocol mismatches.  First,
`boundary_contact` roots were generated from the nearest polygon edge, but the
candidate bank and recovery branch still used the legacy center/vertex normal.
Second, `contact_aware_pulse_actions` applied its old 38-pixel center-distance
pulse gate even though runtime mode was now defined by polygon-boundary gap.
The implementation now provides `boundary_inward_normal`, uses the same
boundary frame in candidate generation, safety gating, recovery, training
features, prediction, and lookahead, and uses the standard pulse bank under
the boundary protocol.  `--legacy-center-action-frame` preserves a matched
same-root ablation; the old terminal protocol remains unchanged by default.

The corrected boundary frame passed the geometry invariants (`10 passed`) and
reduced one-step contact-chart regret in the first matched smoke, but did not
close pose.  The consistent full-feature smoke
`results/negative_cause_diagnosis_20261013/boundary_edge_consistent_full_smoke_seed20261017/`
has mode accuracy `0.848`, boundary contact loss `0/4`, and pose success
`0/4`; mode-aware final angle is about `1.31 rad`.  Its matched legacy-frame
control also has pose success `0/4`, so the action-frame mismatch was real but
not the sole cause.

The trace further showed that geometric `r x action` sign is not a stable
contact-force sign: within one edge/position region, both commanded moment
signs can yield the same observed angle sign under sticking/sliding.  Cross-
region sign fallback is therefore fail-closed for the boundary protocol.
The following explicit ablations were implemented and retained:

- `--strict-contact-sign-gate`: lower ranking regret in a tiny smoke but pose
  success remained `0/2`.
- `--direct-observed-sign-reversal`: worsened the tiny smoke and is not a
  solution.
- `--strict-direction-gate`: improved tiny ranking regret to about `0.075`,
  but pose success remained `0/2`.
- `--contact-chart-alpha 0.1`: did not produce closure in the tiny smoke.
- fixed Coulomb-cone/friction-saturation features: mode accuracy and pose
  remained negative in the tiny smoke.

These results narrow the remaining cause to contact-force/action-effect
non-identifiability and delayed hybrid dynamics, rather than polygon contact
loss or a missing scalar score weight.  None of these privileged-state
ablations is evidence for frozen RGB planning or stable learned pose closure.

### Follow-up: event-memory wiring, action-cache aliasing, and direct candidate ranking (2026-10-13)

The temporal intervention was first audited at the call boundary.  The
training rows and imagined two-step rollout already carried the previous
observed angle increment, but several closed-loop `select_action` and trace
calls still used the default zero value.  Those calls now pass the actual
previous pulse increment, and the protocol records
`temporal_event_features=true` and
`previous_observed_angle_delta=true`.  The implementation and regression
tests pass (`21 passed`).

The audit also found an online effect-cache alias: entries were keyed only by
contact region, moment sign, and action norm.  Equal-norm actions with
different normal/tangent decompositions can have different sticking/sliding
effects, so the key now includes quantized local normal and tangent action
components and edge fraction.  A regression test prevents this alias from
returning.

The event-memory tiny smoke
`results/negative_cause_diagnosis_20261013/boundary_edge_temporal_tiny_seed20261017/`
remains negative (`0/2` pose success, mean angle about `1.62` rad, boundary
contact loss `0/2`).  The action-key correction
`results/negative_cause_diagnosis_20261013/boundary_edge_temporal_actionkey_tiny_seed20261017/`
also remains negative; therefore both changes repair real interface bugs but
are not sufficient explanations for pose failure.

To test whether the remaining error is introduced specifically by
``predict effect -> score pose`` composition, a training-only direct
goal-conditioned candidate ranker was added.  It learns one-step pose cost on
the same shared candidate bank and is evaluated on root-disjoint held-out
roots without successor access.  The matched artifact
`results/negative_cause_diagnosis_20261013/boundary_edge_goal_ranker_matched_seed20261017/`
shows:

| selector | pose success | final XY | final angle | boundary contact loss |
|---|---:|---:|---:|---:|
| mode-aware effect planner | 1/6 | 25.06 px | 0.941 rad | 0/6 |
| direct goal-conditioned ranker | 1/6 | 16.12 px | 0.996 rad | 0/6 |

The direct ranker improves translation but not strict pose closure and incurs
more recovery steps.  Thus replacing the second-stage score with direct
candidate ranking is a partial trade-off, not a repair.  The current evidence
supports a narrower diagnosis: contact retention is controllable, while the
within-contact angle effect is not identified robustly across edge/position
regions and delayed hybrid transitions.  The learned adapter remains
unadmitted as a stable pose controller.

The larger fixed-protocol artifact
`results/negative_cause_diagnosis_20261013/boundary_edge_goal_ranker_large_seed20261017/`
uses 16 train roots, 8 root-disjoint held-out roots, and 8 temporal collection
steps.  It raises successor-mode accuracy to `0.962` and gives the
mode-aware effect planner `1/8` pose successes, mean angle `0.878` rad, and
boundary contact loss `0/8`.  The direct ranker gives `2/8` pose successes and
better XY (`19.71` vs `30.33` px), but worse angle (`0.948` rad) and higher
contact-loss (`5/8`).  Thus more data improves mode identification but does
not make the angle controller stable; direct ranking trades translation for
contact/angle reliability.  This is evidence against a mere small-sample or
second-stage-scoring explanation.

## Short-term history, event memory, and contact-mode literature boundary (2026-10-13)

The literature reassessment is recorded in the repository-level document
[MEMORY_EVENT_CONTACT_LITERATURE_20261013.md](../../MEMORY_EVENT_CONTACT_LITERATURE_20261013.md). The relevant experimental
question is not whether another temporal feature lowers an offline error, but
whether history recovers a decision-relevant state for action-to-task transfer.
The audit therefore separates representation, observability, successor
consistency, and closed-loop control gates.

The current protocol-corrected PushT audit reports causal-delay lifting winning
`10/10` splits with about `15.5%` matched RMSE reduction, state+action
time-frequency lifting winning `10/10` with about `24.9%`, paired memory-4
winning `8/10`, and interaction lifting winning only `3/10` with about `-7.4%`
mean reduction. Adding the full past action history reduces paired memory-4 to
`1/10`, consistent with low-sample/high-dimensional overfitting. The nested
fit-only selector improves the current-state/action baseline on all ten outer
splits but does not reliably exceed the fixed historical first-harmonic Fourier
control. These results establish conditional short-term historical signal,
not a universal memory solution.

The literature points to a different mechanism stack for contact-rich tasks:

```text
event-gated memory -> successor-mode prediction -> mode-conditioned effect
-> short pulse -> RGB re-observation -> contact reacquisition
```

PlaNet/RSSM-style latent state and overshooting address partial observability;
Visual Foresight-style local dynamics and receding horizon limit compounding
error; ContactNets, contact-aware MPC, and D-LGP explicitly represent contact
geometry and discrete mode transitions. Fourier remains a fixed strong
baseline because it can provide useful low-dimensional regularization for
periodic or phase-aligned structure. An event-aware feature that beats the
current-state baseline must not be promoted merely because it does not beat
Fourier.

Any new memory implementation must use fit/calibration/test separation and
report event F1, successor-mode accuracy, action inverse risk, ranking regret,
real/predicted successor distance, multi-horizon task error, contact retention,
block XY, angle error, and strict terminal success. It must use shared
candidates, root- or episode-disjoint test data, RGB-only runtime input, and no
privileged successor query. If mode accuracy improves without angle closure,
the result is `mechanism partial`, not a pose-closure solution. If a memory
improves representation error but not action ranking or task-space transfer,
label it `representation-only`.

### Horizon-aware frozen policy and selector stability (2026-10-13)

The completed real batch provides a preliminary held-out check of the frozen
horizon-aware policy. The current qualitative summary is:

- `h=1`: the formal controller remains fixed Fourier; one split has a stable
  selector preference for gated residual, but this does not modify the frozen
  cross-horizon policy;
- `h=2`: all selector decisions fail the stability gate and fall back;
- `h=4`: all five splits fall back to the pre-frozen Fourier+phase policy.

This separates two estimands that must not be conflated: selector stability
within a split, and the performance of a policy fixed before the independent
test. The selector remains a `selection-instability diagnostic`; it is not an
online controller and must not be updated using held-out planning outcomes.
The complete numeric table, stability frequency, fallback frequency, and
paired held-out regret remain an append-only summary artifact; until that
summary is frozen, this result is `partial` evidence rather than a validated
planning gain.

### Context-safe GDD-WM capacity audit

`gdd_wm/audit_train_only_teacher_capacity.py` must be run from the project
root as a module and requires CUDA; it refuses CPU fallback. When repeated
candidate contexts are supplied, the audit keeps each `context x root` ranking
cell separate and bootstraps unique outcome roots. Always report both
`--action-frame` and `--target-mode`; `root_centered` removes root-level cost
from the estimand, while `goal_relative` is required for the canonical
confirmatory protocol. Corrected non-PushT examples are under
`results/gdd_wm_cross_environment_20261016/*capacity_root_centered_context_safe/`.

The optional `--goal-conditioning latent_relative` arm adds only frozen RGB
goal-latent relations (`goal-source`, `predicted-goal`); it does not read
privileged evaluator state and must be reported separately from the canonical
`goal_relative`/`none` protocol. Its current cross-environment result is
diagnostic and not a planning controller.

The methodological interpretation is a horizon-dependent stability transition:
short horizons may contain local event gains, intermediate horizons can have
unstable memory-family selection, and longer horizons may require a declared
phase-aware policy. The current frozen policy is therefore `h=1 -> fixed
Fourier`, `h=4 -> Fourier+phase`, with fail-closed fallback when selector
stability is insufficient.
