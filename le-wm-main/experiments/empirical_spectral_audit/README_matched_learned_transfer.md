# Matched learned transfer-risk benchmark

## Oral experiment coverage audit

The complete requested experiment inventory is audited by
`oral_experiment_coverage_audit.py`.  It produces a field-by-field matrix for
prediction MSE, train-only latent variance, covariance alignment,
principal-angle/projector fields, commutator, Schur/resolvent/transient
geometry, ATM action-inverse risk, real/predicted transfer, reachability,
elite regret, and held-out planning risk.  It also records the architecture,
seed, snapshot, horizon, environment, matched-stratum, leave-one-environment-
out, and paired-statistics gates.

Example:

```bash
python experiments/empirical_spectral_audit/oral_experiment_coverage_audit.py \
  --record-root results/pointmaze_learned_adapter_panel_balanced32_direction_rawaction_fixed_20261013/records \
  --record-root results/alignment_universal_selection_metric_variants_stablefloor_20261011/task_metric_records \
  --results-root results \
  --output results/oral_experiment_coverage_audit_20261013
```

The audit distinguishes three states: a field can be present in a record, a
matched conditional estimand can be identifiable, and an independent planning
consequence can be positive.  These are not interchangeable.  In particular,
the current combined PointMaze/PushT panel has two environments but no
replicated strict matched stratum, so its learned conditional selection claim
remains gated.

`run_matched_learned_transfer_benchmark.py` is the fail-closed audit for the
conditional claim:

> after matching prediction MSE, latent dimension, predictor spectrum,
> training stage, and action distribution, task/action geometry predicts an
> independent held-out transfer or planning risk.

The primary diagnostics are fixed in advance:

- `task_metric_alignment`;
- `task_metric_commutator`;
- `task_metric_schur_offdiagonal_energy`; and
- `task_metric_transient_amplification`.

Matching is performed within `(environment, training_stage,
latent_dimension, action_distribution)` strata. Candidates form a matched
stratum only when every pair in the selected maximum clique satisfies both the
declared symmetric-relative prediction-MSE tolerance and the declared complete
sorted complex-eigenvalue tolerance. This pairwise rule avoids using the
globally best-MSE checkpoint as an invalid anchor: a valid pair can be
mutually matched even when both are farther from that anchor. The selected
clique must contain at least two architectures and two independent seeds.
Metrics are never chosen using held-out outcomes.

Run the current real-checkpoint audit with:

```bash
python experiments/empirical_spectral_audit/run_matched_learned_transfer_benchmark.py \
  --record-root results/alignment_universal_selection_audit_20261011/task_metric_records \
  --output results/matched_learned_transfer_benchmark_20261011
```

The current run is intentionally gated: it has three architectures and five
seeds but only one environment, and the older records do not contain a finite
task-metric Schur field in every export. A real pairwise-compatible
cross-architecture, cross-seed stratum is still useful evidence, but it cannot
support a cross-environment or universal model-selection claim.

The status gate also requires an identifiable matched stratum in each of at
least two environments. Merely adding an unmatched second environment to the
candidate pool cannot promote the result.
That is evidence about protocol identifiability, not a positive or negative
universal claim. New JEPA/DINO enriched exports must populate the same
record-level factors and independent held-out outcome before they can enter
this audit.

## Control-layer connection

`build_learned_model_selection_records.py` now exports a
`learned_transfer_operator.v1` block for each record. It contains the
diagnostic-split column operators `A`, `B`, `D` and the finite-horizon maps
`G_h = D A^(h-1) B`. It also records which bound components are identified.
Missing planner-reachable, action-balanced, or independent readout data are
explicitly marked as unavailable; they are never replaced by zero residuals.

The operational controller is
`experiments/theory_spectral_transfer/risk_aware_control.py`. It combines the
declared external metric, `G_h`, finite-horizon risk, a local metric-tube
condition, and two independent gates (action identifiability and
real/predicted transition consistency). A failed gate returns the declared
fallback action. The synthetic contract audit is
`results/covariant_transfer_risk_control_audit_20261013/`.

The refreshed 90-record panel is under
`results/learned_transfer_operator_panel_20261013/`. It identifies one
PushT matched stratum (stage 75, two architectures and two seeds), but no
second environment; the status remains `gated_single_environment`, with
`NA` cross-environment rank correlation. This is a real implementation
connection, not evidence for a universal certificate.

## Positive-effect gate and PointMaze adapter panel

The benchmark reports `primary_delta_risk_mse_minus_full` with a paired
stratum-level sign-flip p-value, paired bootstrap interval, and `positive_gate`.
A positive point estimate is not promoted unless the one-sided test, interval,
and at-least-two-environment replication gate all pass.

`train_pointmaze_learned_adapter_panel.py` trains a real two-architecture,
five-seed action-conditioned adapter panel over the available JEPA/DINO
PointMaze exports. PCA and adapter fitting use training rows, action-risk uses
a calibration split, and the fixed `{1,2,4}` horizon candidate consequence is
scored on held-out outcome rows. The runner now fails closed when frozen
exports have unequal row support; use `--max-rows` for a deterministic
support-matched audit or explicitly opt into `--allow-unmatched-support` for
exploration. It also derives the task direction from train-only source-to-
target displacement, rather than from differences between unrelated episode
roots.

The corrected support-matched panel is under
`results/pointmaze_learned_adapter_panel_balanced32_direction_rawaction_fixed_20261013/`.
With 32 rows per export, identical train/calibration/outcome sizes, and five
adapter seeds, JEPA-MLP has cluster mean `Delta R=0.1131` (bootstrap 95% CI
`[0.0937, 0.1268]`, sign-flip `p=0.0317`), while DINO-MLP is negative
(`-0.1965`, CI `[-0.2323, -0.1607]`). The DINO result improves after using
the shared raw action chart, but remains negative. JEPA-linear is positive but does not
pass the predeclared sign-flip gate (`p=0.0615`). This is a
PointMaze-only, representation-conditioned result; it is not a
cross-environment or universal transfer claim.

The representation-condition diagnostic is
`results/pointmaze_representation_condition_diagnostic_20261013.json`. It
records the historical action-field mismatch and the remaining DINO
action-to-latent transfer error without using outcome rows for selection.

### Repair panel and fail-closed gate

The same runner now includes controlled train-only repairs:

- `standardized_mlp`: standardizes inputs and absolute targets;
- `task_aligned_mlp`: adds a train-only task-readout loss;
- `task_effect_mlp`: predicts only the train-fitted task effect;
- `task_effect_lowrank_mlp`: uses a train-only four-dimensional action PCA;
- `calibration_task_rmse_penalty_beta1`: subtracts calibration task RMSE from
  candidate progress and keeps h=1 as a fallback.

The raw repair panels are under
`results/pointmaze_standardized_mlp_panel_rawaction_fixed_20261013/`,
`results/pointmaze_task_aligned_mlp_panel_rawaction_fixed_20261013/`,
`results/pointmaze_task_effect_mlp_panel_rawaction_fixed_20261013/`, and
`results/pointmaze_task_effect_lowrank_mlp_panel_rawaction_fixed_20261013/`.
The gated panel is under
`results/pointmaze_task_effect_mlp_gated_panel_rawaction_fixed_20261013/`.

These interventions separate the causes instead of pooling them: normalization
and task alignment improve DINO's negative point estimate but do not produce a
stable positive gate; the calibration-only gate removes the negative DINO point
estimate (`Delta R≈+0.0045`, bootstrap CI `[0.00014,0.00884]`) but its
sign-flip test is not significant (`p≈0.122`). JEPA remains positive under the
same gate (`Delta R≈+0.0980`, five seed clusters). The correct conclusion is
therefore “fail-closed transfer-risk control is partially repaired”, not
“DINO has been solved” or “the adapter is universal”.

Two exclusion checks are also complete. Increasing the train-only PCA rank
from 8 to 16 worsens the DINO raw task-effect result (about `-0.144`), and
using each export's native normalized action chart reproduces the raw-chart
DINO result. Thus the remaining failure is not explained by PCA truncation or
coordinate conversion. The formal controller keeps the shared raw chart and
uses calibration uncertainty to reject unreliable learned candidates.

## Cross-environment and independent-planning gates

Run the cross-environment wrapper with:

```bash
python experiments/empirical_spectral_audit/run_cross_environment_transfer_diagnostic.py \
  --record-root results/pointmaze_learned_adapter_panel_stages_rank8_clean_20261013/records \
  --output results/cross_environment_transfer_diagnostic_20261013
```

It reports per-environment matched strata, the worst identifiable
environment, and leave-one-environment-out audits. It fails closed if an
included environment lacks a diverse matched clique or independent external
action/planning evidence.

`independent_planning_gain_gate.py` consumes a pre-frozen paired JSON list
with `baseline_loss`, `diagnostic_loss`, and protocol flags. Positive gain is
released only after the paired bootstrap/sign-flip test, minimum pair and
environment counts, and all no-privileged/no-test-selection flags pass. A
positive point estimate from an artifact that did not satisfy those flags is
reported as gated, not as an independent planning result.

`build_independent_pointmaze_planning_gain.py` builds such a paired artifact
from one explicitly supplied checkpoint record without searching outcome
episodes. The corrected JEPA-MLP checkpoint artifact has `Delta=0.1065` on
10 held-out outcome rows; this is an independent PointMaze consequence, not a
cross-environment planning claim. The DINO diagnostic remains negative under
the same support-matched protocol, which is retained as the explicit
 representation failure boundary.

### Horizon-conditioned permission gate and vector-artifact audit

The follow-up permission panel is under
`results/pointmaze_task_effect_mlp_permission_horizonscaled_rawaction_fixed_20261013/`.
It uses a frozen calibration-only gate with horizon-specific task-effect
scales: h=1 must have normalized task RMSE at most `0.75` and calibration
action-inverse risk at most `0.05`; additional horizons must have normalized
task RMSE at most `1.25`. A rejected representation is forced to h=1 and is
labelled `rejected_fail_closed` rather than counted as learned planning.

On the same 32-row support-matched panel, all 25 DINO adapter checkpoints are
rejected for both `h1_task_effect_not_identifiable` and
`action_effect_not_identifiable`. All 25 JEPA checkpoints are accepted. JEPA
retains conservative `Delta R=0.0980` (five seed clusters, sign-flip
`p=0.0306`, bootstrap 95% CI `[0.0649,0.1328]`); DINO is exactly the fixed
h=1 fallback (`Delta R=0`). This is a successful safety/transfer-permission
repair, not evidence that DINO learned action transfer was recovered.

The old 200-episode DINO vector-only artifact is audited by
`audit_dino_vector_artifact_join.py`, with output in
`results/pointmaze_dino_vector_artifact_join_audit_20261013.json`. Its episode
prefix matches the 32-row enriched export and predicted vectors are comparable,
but it has no sample IDs, actions, source states, or target states. The formal
action-conditioned join is therefore `rejected_fail_closed`; it remains useful
only as a vector-only temporal diagnostic and cannot be used to enlarge the
adapter support or produce planning claims.
