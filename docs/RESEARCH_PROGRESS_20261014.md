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

The 2026-10-15 continuation is recorded in
`docs/EXECUTION_REPORT_20261015.md`. It adds support-conditioned local-effect
identification and a fail-closed residual/conditioning gate. The continuation
improves contact retention in a support-root intervention but does not open a
strict pose-closure or visual-planning claim; the remaining bottleneck is
within-contact action-to-angle identifiability.

## PushT contact-supported orientation repair (latest continuation)

The RGB-only visual world-model runner now applies orientation/action-effect
losses only on rows with offline contact support (`contact_mode > 0`). This
prevents the abundant free-space no-motion rows from dominating the rare
within-contact orientation signal. Mode 3 remains included because it denotes
separation after a contact-supported transition. The change is in
`le-wm-main/experiments/run_causal_visual_pusht_world_model.py`, with a
regression test in
`le-wm-main/tests/test_causal_visual_world_model_serialization.py`.

The matched seed/data-seed pilot was rerun with identical 32/16 root splits,
fan size 8, and CPU only after the fresh GPU check failed with NVML
`Driver/library version mismatch`:

| Variant | Paired motion top-1 | Contact motion top-1 | Contact orientation top-1 | Angle-delta corr. | Strict planning |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior torque baseline | 0.5625 | 0.8750 | 0.0000 | 0.1610 | 0/2 |
| contact mask + soft mode/all losses | 0.1875 | 0.3750 | 0.7500 | -0.2634 | 0/2 |
| contact mask + no mode routing | 0.1875 | 0.2500 | 0.0000 | -0.4691 | 0/2 |
| contact-masked angle-effect only | **0.6875** | **0.8750** | 0.0000 | -0.1377 | 0/2 |
| contact roots 75% | 0.1250 | 0.1667 | 0.1667 | -0.5816 | 0/2 |
| terminal roots 25% | 0.1875 | 0.2000 | 0.0000 | -0.2556 | 0/2 |

Artifacts are under
`le-wm-main/results/causal_visual_pusht_*_cpu_pilot_20261014/`. The useful
partial repair is specific: contact-masked wrapped-angle supervision improves
translation/action ranking over the prior baseline, but it does not recover
orientation ranking or strict pose closure. Adding soft mode routing,
sin/cos orientation losses, more contact roots, or terminal roots does not
solve that remaining bottleneck and can damage the translation geometry.

The mode audit on the matched split found contact rows are almost entirely
`mode=2` sliding and free rows are `mode=0`; therefore the failure is not a
simple mode-label coding error. It is now localized to RGB pose/action-effect
identifiability and shared-gradient interference, not missing contact labels.

As an interface control, the same learned checkpoint was evaluated with the
RGB geometry servo. It achieved `0/16` strict success, although some episodes
reached small angle error while remaining outside the XY tube. This separates
usable RGB geometry from reliable closed-loop contact transfer; the artifact
is `le-wm-main/results/visual_geometry_servo_from_contactmasked_angleeffect_cpu_20261014/`.

The privileged mode-aware short-pulse benchmark was started with boundary
contact, observed-sign calibration/feedback, paired probes, and fail-closed
support gates, but was interrupted after a long CPU candidate-scoring loop
before producing an artifact. It is not evidence and is not counted in the
table above. Current conclusions remain bounded: no learned latent causal
dynamics, stable pose adapter, frozen visual online-planning, universal
world-model, or universal commutator claim is opened by this continuation.

## Unified ATM -> HTSC -> GDD-WM baseline rerun

The unified paper spine is recorded in
`docs/UNIFIED_DECISION_TRANSFER_PAPER_PLAN_20261014.md`. The fresh P0/P1
post-hoc rerun is under
`le-wm-main/results/unified_decision_transfer_p0p1_20261014/`, with its
machine-readable gate summary in `unified_manifest.json`.

The rerun reproduces the PointMaze task-contrast gap: 960 within-root pairs,
inversion rate `0.4077976818`, mean contrast error `0.1081297232`, and an
independent calibration bound `0.4282840914`. Because realized RGB successor
and support calibration are absent, HTSC remains
`abstain_support_gate_unavailable`.

The ATM post-hoc export audit reproduces inverse-risk matrices but has no
candidate/root/task-cost identifiers, so finite-horizon ranking is explicitly
`unavailable_missing_candidate_root_identifiers`. The refreshed learned-panel
audit has one identifiable stratum in one environment and remains
`gated_single_environment`; the independent cross-environment quality-matched
artifact remains `gated_positive_delta_not_established`. These are the
current Diagnose-stage boundaries. GDD-WM remains proposed and is not trained
from these outcomes.

### GDD-WM D0/D1 audit

The D0/D1 implementation and artifact are in
`le-wm-main/experiments/gdd_wm/` and
`le-wm-main/results/gdd_wm_d0_d1_audit_20261014_v4/`. D0 passes schema and
root-split validation for 640 rows and 40 candidate sets. The v4 artifact
reported 128 outcome rows without predicted/realized latent references; that
support gap has now been repaired in the isolated v5 continuation below. v4
is retained as the historical pre-join audit.

The new D1 audit isolates two causes of the previous apparent positive result:
each root has only one candidate-set context, so preference flip/stability is
not estimable; and `teacher_score` is exactly `-real_task_cost`, making the
observed execution consistency of `1.0` label leakage. D1 therefore rejects
all eligible pairs. The corrected same-root aggregation contains 4,800 pairs
(4,764 non-ties), but this does not change the fail-closed decision. D2 remains
blocked. This is a protocol repair and a negative boundary result, not a
GDD-WM failure after valid teacher training.

#### v5 provenance-safe realized-reference continuation

The join implementation is
`le-wm-main/experiments/gdd_wm/join_rgb_realized_refs.py`. It copies the
candidate bank into a new artifact and joins only the realized RGB projection
references from the dataset-matched `rgb_successor_v1` report. It requires
exact candidate-dataset path agreement, unique and complete
`(instance_id, candidate_id, split)` coverage, matching NPZ projection keys
and values, and refuses to overwrite any existing reference or protected
teacher/cost/split/action field.

The v5 artifact is
`le-wm-main/results/gdd_wm_d0_d1_audit_20261014_v5/`. The join covers 640/640
records and 640/640 projection keys; D0 passes with 40 sets of 16 candidates,
and the source candidate bank remains unchanged. D1 now has zero missing
runtime references, but still rejects fail-closed: preference stability is
unavailable because each root has only one candidate-set context, and
execution consistency is invalid because the teacher is
`offline_environment_cost_v1` with the exact leakage
`teacher_score = -real_task_cost`. Thus the realized-reference repair removes
one protocol defect but does not create D1 evidence or make D2 ready.

#### Non-leaking predicted-latent teacher control

To separate label leakage from representation failure, the control
`le-wm-main/experiments/gdd_wm/audit_nonleaking_teacher_control.py` fits a
fixed standardized ridge readout from predicted latent to `real_task_cost`
using train roots only. It evaluates candidate ranking on untouched
calibration and outcome roots; it is explicitly not a D-JEPA teacher and is
not used to authorize D2.

The artifact is
`le-wm-main/results/gdd_wm_nonleaking_teacher_control_20261014/nonleaking_teacher_control.json`.
Held-out pairwise consistency is `0.4301` on calibration and `0.5096` on
outcome. Outcome mean candidate regret is `0.1510` over eight roots, with
zero roots achieving zero regret. Removing the exact cost-label leakage does
not recover reliable transfer; the predicted latent/action-conditioned
geometry is itself insufficient for held-out task ranking.

#### Predeclared feature-family panel

The next control fixes five feature families before evaluation:
`predicted_terminal`, `source_plus_terminal`, `source_plus_delta`,
`source_delta_action`, and `action_summary`. Each uses the same standardized
train-root-only ridge readout and the same candidate bank protocol. The
runner is
`le-wm-main/experiments/gdd_wm/audit_transfer_sufficient_feature_families.py`,
with the cross-bank wrapper
`le-wm-main/experiments/gdd_wm/audit_feature_family_panel.py`.

The five-bank artifact is
`le-wm-main/results/gdd_wm_transfer_sufficient_feature_family_panel_20261014/panel.json`.
No feature family wins consistently across both calibration and outcome
across all five banks. For the repaired task-effect bank, `action_summary`
reaches outcome consistency `0.6195` and regret `0.1060`, but calibration
consistency is only `0.4626`; `source_delta_action` reaches calibration
consistency `0.6125` but outcome falls to `0.4995`. Across checkpoints the
winner changes. This rules out a fixed concatenation as the current repair and
supports the narrower diagnosis that source-conditioned action-effect geometry
is missing, rather than merely more latent history.

#### Relative-effect, low-rank, and pairwise ranking controls

The follow-up runner
`le-wm-main/experiments/gdd_wm/audit_relative_action_effect_geometry.py` fixes
the candidate-level estimand to a train-root-centered effect. It compares the
source/delta/action carrier with the full interaction and train-only PCA
controls of ranks 2, 4, 8, and 16. The panel wrapper reports paired root
bootstrap intervals in
`le-wm-main/results/gdd_wm_relative_action_effect_panel_20261014/panel.json`.

The result is a negative diagnosis: the full 188-dimensional interaction
improves train fit but its outcome effect intervals cross zero across the five
banks. Low-rank controls reduce capacity but do not produce a positive
held-out consistency and regret interval across all banks. Within-root
pairwise ridge controls, which directly optimize candidate ordering rather
than absolute task cost, also have bank-dependent point estimates and CIs
crossing zero. Thus more interaction capacity, PCA compression, and an
objective swap do not repair transfer. With only eight outcome roots per bank,
the intervals are wide; this remains diagnostic evidence, not a selector or
planning claim. The next admissible repair is independently collected
cross-root action-support data with the same paired estimand. D-JEPA teacher
validation and D2 remain blocked.

#### Root-dependent action-bank mismatch and goal-frame repair

The cross-root audit
`le-wm-main/experiments/gdd_wm/audit_cross_root_action_effect_consistency.py`
found a concrete protocol defect in the current PointMaze bank. Although the
metadata says `shared_candidate_bank=true`, the raw action sequence for the
same candidate id differs across roots (`shared_action_bank=false`). Raw
candidate-pair ranking flips occur for about `0.52/0.69/0.89` of pairs on
calibration/outcome/train in the final MLP bank, and `0.88/0.98/1.00` in the
history-4 bank. Thus the readout was asked to transfer a root-dependent raw
action coordinate system as if it were a shared candidate bank.

The repair adds goal-frame action canonicalization to the relative-effect
audit: each action is represented in the root's
`[goal-direction, perpendicular]` frame before fitting. On the existing
post-hoc records, outcome consistency gain versus the raw-action carrier is
`+0.316` `[+0.211,+0.424]` for `final_mlp_s005`, `+0.313`
`[+0.210,+0.413]` for `final_standardized_s005`, and `+0.371`
`[+0.226,+0.515]` for `task_effect_readout_s005`; corresponding regret
reductions are `+0.139`, `+0.124`, and `+0.135`, with CIs entirely above zero.
The two history-conditioned banks remain inconclusive (`CI` crosses zero).

The candidate generator now also supports `--canonical-action-bank`, samples
the bank once in goal-relative coordinates, rotates it into each simulator
root, and records the coordinate frame explicitly. The positive post-hoc
canonicalization result is a protocol/diagnostic repair using privileged
source state and goal; it is not RGB online planning, D-JEPA evidence, causal
dynamics, or a universal world model. A fresh learned canonical-bank export
still requires a working GPU-first learned-inference environment.

#### Fresh CUDA canonical-bank replication

Host-permission GPU preflight subsequently passed with a real CUDA allocation
on an RTX 5060 Ti. Two fresh frozen-checkpoint banks were regenerated with
`--canonical-action-bank`; each contains 640 records over 40 roots with a
24/8/8 train/calibration/outcome split. The generator now constrains random
goal-frame actions so rotated controls remain in `[-1,1]`; both artifacts have
one bank hash and maximum raw control magnitude `0.99865`.

The corrected cross-root audit reports `shared_action_bank=true` for both
fresh banks. Same-pair real-cost order still flips on `0.675` of outcome
pairs, so canonicalization repairs the candidate protocol but does not remove
state-conditioned dynamics. With `source+delta` as the fixed baseline,
goal-frame action features yield outcome consistency gains of `+0.381`
`[+0.240,+0.513]` and `+0.350` `[+0.223,+0.466]` for the MLP and standardized
checkpoints; regret reductions are `+0.122` `[+0.051,+0.188]` and `+0.149`
`[+0.082,+0.204]`. These are fresh GPU-generated, held-out offline
action-effect diagnostics, not D-JEPA preference validation, RGB online
planning, or causal-dynamics proof. Realized RGB references and D1 remain
blocked.

#### Cross-seed and cross-architecture canonical replication

The same frozen protocol was then rerun for seed `8412` on both the final and
standardized MLP checkpoints, yielding four case units (`architecture x seed`)
with identical 40-root, 16-candidate banks. The aggregation runner is
`le-wm-main/experiments/gdd_wm/aggregate_relative_action_effect_panel.py` and
the aggregate artifact is
`le-wm-main/results/gdd_wm_relative_action_effect_panel_20261014/canonical_2arch_2seed_aggregate.json`.

Against the predeclared `source+delta+raw-action` carrier, goal-frame action
features are positive in all four case units. The case-macro consistency gain
is `+0.2784`, with two-level case/root bootstrap 95% CI
`[+0.2110,+0.3442]`; the corresponding regret reduction is `+0.0739`, CI
`[+0.0415,+0.1110]`. The report keeps case-level and root-level uncertainty
separate rather than treating 32 root-by-checkpoint cells as independent model
replications.

This is the strongest current evidence that the original negative result had
a real action-coordinate protocol component, and that goal-frame
canonicalization gives a reproducible offline action-effect gain across two
MLP variants and two seeds. It still does not establish D-JEPA preference
stability, learned causal dynamics, RGB online planning, or universal transfer:
the teacher remains an offline cost oracle, the outcome is simulator-labeled,
and no closed-loop planning consequence has been measured.

#### Independent RGB successor consequence

The same 10 canonical cases were passed through the frozen RGB successor
verifier on the independent outcome roots (`8 roots x 16 candidates` per
case). All cases passed renderer consistency, realized-RGB-latent availability,
and PCA successor comparison. The verifier outputs are under
`le-wm-main/results/decision_benchmark_candidate_bank_20261014/rgb_successor_canonical_*_outcome8/`;
the cross-case aggregate is
`le-wm-main/results/decision_benchmark_candidate_bank_20261014/rgb_successor_canonical_2arch_5seed_aggregate.json`.

Relative to the no-action/source-latent baseline, the action-conditioned
predictor improves realized RGB successor error by `+14.21%`, with two-level
case/root bootstrap 95% CI `[+8.47%,+19.69%]`; `66.56%` of candidate rows are
closer to the realized RGB successor than the baseline, CI
`[59.77%,73.05%]`. This is an independent successor-transfer consequence of
the repaired protocol, not closed-loop online planning: actions are still
executed in the declared simulator and RGB is reconstructed by the
dataset-matched renderer.

#### Latent-estimated action canonicalization

The raw-action feature family was then tested against a targeted repair. A
train-root-only ridge probe predicts source XY from the frozen source latent;
candidate actions are transformed into the estimated
`[goal-direction, perpendicular]` frame before fitting the task readout. This
adds `source_delta_estimated_goalframe_action` without using held-out state at
inference time. The runner and aggregate are
`le-wm-main/experiments/gdd_wm/audit_transfer_sufficient_feature_families.py`
and
`le-wm-main/results/decision_benchmark_candidate_bank_20261014/canonical_2arch_5seed_feature_family_aggregate.json`.

Relative to `source_plus_delta`, the estimated-frame family improves held-out
consistency by `+0.0985`, two-level case/root CI `[+0.0534,+0.1449]`, and
reduces candidate regret by `+0.0546`, CI `[+0.0176,+0.0888]`. The regret
effect is positive on 9/10 case units; standardized seed `8413` remains
negative (`-0.0249`). Thus latent-estimated canonicalization is a promising
partial repair, but not a universal or architecture-invariant transfer
certificate. In particular, standardized seed `8413` still improves realized
RGB successor error by about `14.39%` while its estimated-frame candidate
regret worsens by `0.0249`; prediction accuracy alone therefore does not close
the action-identifiability-to-planning gap.

#### Successor-calibration negative control

To test whether the remaining failure was merely a systematic RGB successor
error, standardized seed `8413` received a train-only affine/ridge calibration
from predicted terminal projection to realized RGB terminal projection. The
calibration reduced train successor RMSE to `0.1628`, but outcome candidate
regret for `source_plus_delta` increased from `0.0872` to `0.1503`; the
raw-action and estimated-frame variants also remained worse than their
uncalibrated decision readouts. Thus generic successor calibration is not a
valid repair: it can improve prediction-space fidelity while destroying
task-readout geometry. This is a direct negative control for the
prediction-to-decision gap.

#### Five-seed replication

The canonical export was then extended without changing any protocol choice to
all five available seeds (`8411`--`8415`) for both MLP variants. The resulting
10-case panel and aggregate are
`le-wm-main/results/gdd_wm_relative_action_effect_panel_20261014/canonical_2arch_5seed.json`
and
`le-wm-main/results/gdd_wm_relative_action_effect_panel_20261014/canonical_2arch_5seed_aggregate.json`.
Every case contains 40 roots, with 24/8/8 train/calibration/outcome roots.

The goal-frame variant is positive in all 10 architecture-by-seed cases. The
case-macro consistency gain is `+0.2777`, with two-level case/root bootstrap
95% CI `[+0.2356,+0.3212]`; regret reduction is `+0.0751`, CI
`[+0.0455,+0.1073]`. The weakest case remains positive (`+0.0182` regret
reduction), so the result is not driven by a single seed. This closes the
current seed-stability check for the repaired offline action-effect diagnostic,
but not the independent planning consequence required for a planning claim.

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

### Indexed canonical-action repair for the GDD-WM ranking failure (2026-10-14)

The remaining standardized-seed-8413 failure was isolated more sharply. A
train-root-only latent-to-source-XY probe had position RMSE `0.4156` on its
fit roots and `0.6682` over all roots. A quadratic latent chart drove the
training error near zero but worsened held-out candidate regret, so increasing
chart capacity was rejected as overfitting rather than retained as a repair.

The actionable repair uses the declared shared candidate bank as the action
geometry. Each candidate's raw simulator action is transformed to the
goal-relative frame once on train roots and indexed by `candidate_id`.
Calibration and outcome roots use only this frozen train-derived lookup; their
source state is not used to construct features. The post-hoc bank check passes
on all 640 rows with maximum canonical-summary deviation below `7e-16`.

The fresh two-architecture/five-seed panel is at
`le-wm-main/results/decision_benchmark_candidate_bank_20261014/indexed_canonical_action_2arch_5seed_aggregate.json`.
Against the fixed `source+delta` baseline, the indexed canonical-action
readout improves outcome execution consistency by `+0.2845` with a two-level
case/root bootstrap 95% CI `[+0.2425,+0.3275]`, and reduces candidate regret by
`+0.1090` with CI `[+0.0820,+0.1372]`. All 10 architecture-by-seed cases are
positive; standardized seed `8413` is repaired (`+0.0708` regret reduction).

This closes the specific raw-action/latent-frame reconstruction failure in the
offline benchmark. It remains an offline task-readout consequence with
simulator-labeled costs, not D-JEPA preference validation, causal physical
dynamics, closed-loop RGB online planning, or universal-world-model evidence.

### Independent RGB-compatible task consequence (2026-10-14)

The indexed action repair was then evaluated with the frozen RGB successor
reports on the same root-disjoint train/calibration/outcome protocol. The
aggregate artifact is
`le-wm-main/results/decision_benchmark_candidate_bank_20261014/rgb_indexed_canonical_task_2arch_5seed_aggregate.json`.
It contains two MLP architectures and five seeds per architecture, with eight
outcome roots per case. The RGB feature readout is fit on train roots only;
the canonical action bank is also fit on train roots only and accessed on
calibration/outcome through `candidate_id`.

Against the RGB source-plus-predicted-delta baseline, the indexed canonical
action family improves outcome execution consistency by `+0.2845`, with a
two-level case/root bootstrap 95% CI `[+0.2425,+0.3275]`, and reduces outcome
candidate regret by `+0.1090`, CI `[+0.0820,+0.1372]`. All 10 cases are
positive. The previously failing standardized seed `8413` remains positive
(`+0.0708` regret reduction). The standardized-8413 train report used the
content-identical canonical-named artifact because its indexed-named alias was
not emitted; this is recorded as an artifact naming alias, not a new data
selection.

This is an independent RGB-compatible successor/task-readout consequence,
not frozen RGB online planning: RGB is dataset-matched rendered input,
actions are executed in the simulator for labels, and no closed-loop planner
or test planning result was used for feature fitting or selection. The new
synthetic tests enforce train-only bank provenance, root-disjoint transfer,
and fail-closed RGB gates.

### RGB-only visual world-model pilot and runner repairs (2026-10-14)

The first CPU end-to-end smoke of
`experiments/run_causal_visual_pusht_world_model.py` exposed an artifact bug:
the planning trace contained NumPy arrays and could not be serialized. The
runner now recursively converts NumPy arrays/scalars before writing its JSON
artifact. A separate CPU smoke confirmed the repaired pipeline completes
dataset construction, training, held-out counterfactual metrics, and one
RGB-only planning episode.

The smoke still has a clear mechanism failure: paired orientation top-1 is
`0.0`, angle-delta prediction correlation is `0.0543`, and strict success is
`0/1`. This is retained as evidence that the current visual representation
does not yet identify within-contact orientation effects.

A matched contact-frame/torque pilot was then run on the identical root seed
with temporal context, contact-frame dynamics, signed torque features, explicit
angle heads, and visual contact gating. It improves action-motion top-1 to
`0.5625` and angle-delta correlation to `0.1610`, but orientation top-1 stays
`0.0` and planning strict success stays `0/2`; this is a partial translation
mechanism gain, not pose closure.

Finally, enabling all orientation/block-effect losses and soft
mode-conditioned routing on the same data is a negative control: action-motion
top-1 falls to `0.0625`, angle-delta correlation becomes `-0.1717`, and mode
top-1 is only `0.1420`. The result shows that adding orientation losses without
adequate contact-mode support destabilizes the action geometry; it does not
justify increasing those weights further. Artifacts:

- `results/causal_visual_pusht_rgb_only_smoke_20261014/`;
- `results/causal_visual_pusht_contactframe_torque_cpu_pilot_20261014/`;
- `results/causal_visual_pusht_contactframe_torque_angleweighted_cpu_pilot_20261014/`.

The runner fix is covered by nested-NumPy serialization tests. The visual
online-planning claim remains gated because all pilot strict-success rates are
zero and the current runs use CPU only after the CUDA preflight reported both
NVML/driver communication failure and `torch.cuda.is_available() == False`.

### PushT goal-pose label audit and protocol repair (2026-10-14)

The next GPU pilot isolated a separate protocol defect in the visual runner.
PushT's `info["goal"]` is rendered from `info["goal_pose"]`, whereas
`info["goal_state"]` is the simulator reset/evaluation state and carries the
randomly sampled source block angle. The runner had been using
`goal_state[2:6]` as the target-image supervision, so the goal-angle head was
asked to predict a label that was not identifiable from the goal image.

The repair adds `goal_pose_vector()` and carries the rendered target pose
through every source-root constructor, counterfactual dataset, split manifest,
and planning evaluator. It also separates `--explicit-goal-angle-readout`
from the current-frame `--explicit-angle-readout`, so a goal-only calibration
cannot silently replace the current dynamics state with an untrained head.
The targeted regression suite passes (`5 passed`), and the source compiles in
the project environment.

The matched CUDA pilot after the label repair reduced goal-pose readout RMSE
from the prior `43--58` range to `9.334`, and goal-angle readout RMSE to
`0.245`. This confirms a real label/protocol bug. However, the same run still
had angle-delta correlation `-0.571`, contact action-motion top-1 `0.000`,
paired orientation top-1 `0.000`, and strict visual planning `0/2`.
Therefore the repair removes one confound but does not establish learned
physical causality, stable pose closure, or RGB online planning. The remaining
bottleneck is within-contact current-pose/action-effect identification.

The matched CUDA follow-ups were also completed with the corrected labels:
`causal_visual_pusht_gpu_goalposefix_currentgoalreadout_20261014` and
`causal_visual_pusht_gpu_goalposefix_angleeffectonly_20261014`. The former
reached goal-angle RMSE `0.143` and terminal angle error `0.592` on two
episodes, but action-motion top-1 was `0.125`; the latter reached goal-angle
RMSE `0.059`, contact action-motion top-1 `0.500`, but contact orientation
top-1 remained `0.0` and strict success remained `0/2`. These are matched
mechanism results, not a positive planning claim.

### Contact-pose support and runtime-interface continuation (2026-10-14)

The next GPU-first continuation tested the remaining within-contact diagnosis
without changing the corrected goal-pose protocol. A rich temporal pilot that
routed block silhouette differences only to the angle-effect head reached
contact action-motion/orientation top-1 `0.25/0.25`, but angle-delta
correlation was `-0.589` and strict planning was `0/2`.

An RGB-only identifiability control separated representation information from
joint-training interference: a small angle head trained directly on the fixed
RGB block-moment chart achieved held-out contact-angle RMSE `0.184 rad` and
`81.25%` within `15` degrees. This means the contact pose signal is present in
the rendered input, but is not reliably transferred by the shared model under
the small root split. The runner therefore adds predeclared
`--contact-angle-readout-weight`, optional angle-head warmup/freeze, and
`--invariant-angle-readout` (block-only central moments). All checkpoint
loading paths preserve the new architecture fields.

The matched contact-angle-weighted pilot still failed (`contact angle RMSE
2.87`, orientation top-1 `0`, strict success `0/2`). Increasing support to
`256/128` roots improved contact action-motion top-1 to `0.3125` and angle
delta RMSE to `0.130`, but orientation top-1 was `0.125` and strict success
was `0/4`. The formal `1600/240` GPU run improved contact action-motion
top-1 to `0.375`, while contact orientation top-1 remained `0.0417` and
strict planning remained `0/8`.

The same formal checkpoint was evaluated with a frozen RGB geometry state
readout and visual contact gate. It obtained `1/8` on the original evaluation
seed but `0/8` on an independent held-out evaluation seed; therefore this is
an interface diagnostic, not stable planning evidence. Replacing scalar
utility with task-space ranking produced the same `1/8`, and a separately
trained phase-utility checkpoint remained at `0/8`.

Current boundary: the repaired label protocol and larger contact support
improve offline translation/effect metrics, but no stable within-contact
orientation transfer or frozen RGB online-planning claim is open. The next
scientifically admissible intervention must change the action-effect support or
contact observation protocol, not merely add more Fourier/history features or
utility weights. Artifacts are under
`results/causal_visual_pusht_gpu_goalposefix_*_20261014/`.

### RGB contact-interface and angle-readout continuation (2026-10-15)

The next action-effect intervention added an RGB-only pusher-to-block interface
chart: relative centroid, nearest block-pixel direction/gap, object scale, and
validity. It was first supplied as context and then with explicit
lever/action projections and a cross-product interaction. The learned model
never receives simulator contact labels or successor state at runtime.

The CUDA artifacts are:

- `results/causal_visual_pusht_gpu_contactinterface_pilot_20261015/`;
- `results/causal_visual_pusht_gpu_contactinterface_full_20261015/`;
- `results/causal_visual_pusht_gpu_contactinterface_interaction_pilot_20261015/`.

The `1600/240` interface-context run reduced paired effect error to `6.54`,
but contact action-motion/orientation top-1 were only `0.275/0.0583`,
angle-delta correlation was `0.0402`, and RGB planning was `0/8`. The
explicit lever/action interaction pilot was worse (`0.1875/0.03125` contact
action-motion/orientation top-1). Therefore contact geometry is not sufficient
unless its action-effect mapping is identifiable; adding an uncalibrated
interaction feature is not promoted as a repair.

The code audit also found that the existing `current-only-readout` option did
not apply to the explicit angle head. A backward-compatible
`--angle-readout-current-only` architecture flag now excludes previous-frame
difference features from that head. Its isolated CUDA pilot remains negative:
contact action-motion/orientation top-1 `0.09375/0.0` and angle-delta
correlation `-0.266`. Old checkpoints remain loadable (`21` compatibility
tests pass), so this is an implementation repair plus negative boundary
evidence, not a positive planning result.

The current root cause is consequently sharper: RGB contains pose information,
but the deployment candidate action-effect map is not stable across contact
interfaces. The next admissible positive route is an observed-effect,
short-pulse runtime protocol with explicit support and fail-closed release;
further static feature or loss stacking is not justified by these runs.

### Frozen RGB pose chart, goal-domain audit, and short-pulse continuation (2026-10-15)

The GPU-first analytic audit tested whether the negative learned pose result
was caused by a missing visual pose interface. On 2,048 root-disjoint RGB
roots, a calibration-only quadratic harmonic ridge over block mask moments
reached 83.0% within 15 degrees on held-out current-scene roots and 85.1% on
held-out contact roots (directed MAE 0.157 rad). This establishes a usable
frozen RGB pose measurement, but not learned causal dynamics or pose closure.

Applying that current-scene calibration directly to goal renders failed
(`4.9%` within 15 degrees, directed MAE `1.536` rad). A separate goal-image
calibration repaired this PushT protocol because the goal angle is fixed in
this benchmark (`100%` within 15 degrees on the held-out goal roots). The
calibration artifacts and audit are under
`results/rgb_analytic_pose_separate_goal_full_20261015/`.

Replacing the learned angle readout with the two frozen RGB readouts did not
produce closed-loop success: geometry servo remained `0/8`. A code audit then
found and fixed an actual interface bug: the nearest-boundary chart stacked
`[y,x]` while all runtime charts consumed `[x,y]`. The corrected support-aware
servo still remained `0/8` and often pushed the block away from the goal. This
shows that visual pose and visible support are not sufficient to identify the
within-contact torque effect.

The next short-pulse audit used 256 contact roots, an 8-action fan, and
RGB-successor re-observation. The observed angle effect was measurable
(`88.1%` within 15 degrees), but direct re-observed task ranking was only
`23.0%` top-1 (random fan baseline `12.5%`); a held-out global observed-effect
adapter was worse at `18.75%`. The current positive result is therefore a
verification signal, not a planning result. The admissible methodology is
short-pulse effect verification with support-aware fail-closed release, rather
than a global action-effect predictor.

### Mode-aware verification gate continuation (2026-10-15)

The privileged PushT controller was rerun with the complete predeclared
boundary-contact protocol: 64 train roots, 32 root-disjoint held-out roots,
eight shared candidates, 128 trees, one-step lookahead, paired effect probes,
observed-sign feedback, contact-geometry sign feedback, uncertainty gate, and
terminal handoff. CUDA preflight passed before the run (`RTX 5060 Ti`, real
`cuda:0` allocation). The final artifact is
`le-wm-main/results/negative_cause_diagnosis_20261015/mode_aware_verification_gate_64x32_seed20261015_final/`.

Candidate predictions and training effect-gate statistics were batched by
action bank/root without changing the ExtraTrees estimator, seed, labels,
splits, scoring gates, or controller rules. The held-out result is still a
mechanism boundary: successor-mode accuracy is `0.9698`; mode-aware pose
success is `3/32 = 0.09375` (bootstrap CI `[0, 0.21875]`); mode-aware ranking
regret is `0.09645`, versus `0.09510` for the global one-step baseline and
`0.09521` for the contact-chart variant.

Adaptive online effect correction reaches `4/32 = 0.125` pose success and
reduces predicted/observed effect error to `0.08597`, but contact loss is
`0.71875`. Uncertainty gating lowers contact loss to `0.59375`, but pose
success remains `3/32`, root fallback is `0.8125`, and effect error is
`0.14932`; the safe fallback has pose success `2/32` and contact loss `0.5`.
The matched 8-root CUDA pilot has the same direction (`mode accuracy 0.9576`,
mode-aware pose success `0/8`, adaptive online `1/8` with contact loss
`0.875`) and is retained separately rather than pooled.

High mode accuracy with low pose success rules out successor-mode
observability as the dominant cause. The remaining bottleneck is joint
within-contact action-effect identifiability and contact preservation. The
observed-effect correction is a local verification signal, not a stable pose
closure repair; the admissible protocol remains short-pulse RGB re-observation
with support-aware fail-closed release.

### Distance-aware online support control (2026-10-15)

The 32-root trace showed that a local observed-effect entry could be reused
after the pusher had materially separated from the polygon boundary. The
online effect key and neighborhood descriptor were therefore extended with
the measured pusher-to-boundary distance; cache reuse now requires both the
same local action chart and a nearby contact distance. This is a support
restriction only and does not use successor labels.

The CUDA 8-root pilot
`results/negative_cause_diagnosis_20261015/mode_aware_distance_support_64x8_seed20261015_pilot/`
did not improve the downstream gate: mode-aware pose success stayed `0/8`,
adaptive online stayed `1/8` with contact loss `0.875`, and uncertainty-gated
pose success stayed `0/8`. Ranking values were unchanged. The stale-distance
alias was a real implementation risk, but removing it is not sufficient;
the remaining negative result is consistent with nonstationary within-contact
action effects and inadequate candidate support.

## Cross-seed orientation permission and fail-closed control (2026-10-18)

The three independent edge-boundary orientation audits were aggregated without
reading pose or planning outcomes. The CUDA artifact is
`le-wm-main/results/negative_cause_diagnosis_20261015/orientation_gate_cross_seed_20261018/`.
The authorized-stratum union contains 10 edge/fraction strata, but the
three-seed intersection contains only 4 (`0:0__bin1`, `0:2__bin3`,
`1:1__bin1`, `1:3__bin2`), or 40% of the union. Released evaluation coverage
is 24.5%--31.9% per seed. Released sign consistency is 0.972 and successor
mode agreement is 1.0. This supports a local calibrated permission, not a
universal orientation certificate.

That four-stratum intersection was wired into the shared-candidate closed-loop
runner as a fail-closed gate. The CUDA smoke artifact is
`le-wm-main/results/negative_cause_diagnosis_20261015/orientation_gated_control_smoke_20261018/`.
All baselines use the same 8 held-out roots and candidate bank. The gate
blocked 89.1% of contact steps and every root triggered fallback; boundary
contact loss was 0, but pose success remained `0/8` and final angle error was
1.671 rad. Mode-aware and contact-chart comparisons also remained at `0/8`.
The gate exposes insufficient support, but abstention cannot itself recover
within-contact action-effect transfer.

The diagnosis is now narrower: successor-mode observability and local
opposite-moment response can be high on released strata, yet continuous
mode-conditioned action-to-angle transfer is not identified on new contact
interfaces. The next admissible repair must improve local action excitation or
use explicit online paired effect identification.

## Paired local-effect probe continuation (2026-10-18)

The next repair was executed as a separate CUDA smoke artifact at
`le-wm-main/results/negative_cause_diagnosis_20261015/paired_effect_probe_smoke_20261018/`.
Within a contact chart, the controller alternated opposite-moment short pulses
and used only re-observed transitions to update local effect ranking. The
adaptive-online branch reduced predicted/observed effect error to `0.123` and
wrong-sign events to `1.0`; pose success was `2/8` and contact loss remained
`0.50`. This is an encouraging mechanism signal, not stable pose closure. The
active repair direction is retained as paired local excitation plus fail-closed
release, pending a larger root-disjoint multi-seed run.

## Contact-retention metric repair and three-seed replication (2026-10-24)

The paired-effect panel exposed a protocol bug in the contact-retention readout. Boundary-contact roots intentionally place the pusher on a polygon edge, so pusher-to-block-centre distance can exceed 80 px while contact is physically maintained. The previous `contact_loss_rate` therefore counted legal boundary contacts as losses. The runner now uses physical polygon boundary gap (`>70 px`) for boundary-contact roots and reports the old centre-distance predicate separately as `center_distance_loss_rate`. Existing artifacts were not overwritten.

The corrected CUDA replication uses the frozen paired-probe protocol, three independent seeds, 24 train roots and 16 root-disjoint held-out roots per seed. The aggregate artifact is `le-wm-main/results/negative_cause_diagnosis_20261015/contact_metric_repair_panel_20261024/`. Across 48 held-out roots, protocol-correct contact loss is `0.000` for every variant with bootstrap CI `[0, 0]`; the centre-distance diagnostic remains `0.438--0.667`, confirming the original negative contact result was a measurement error rather than physical contact failure.

The repair does not close pose transfer. Adaptive paired probing reaches pose success `0.0833` (CI `[0.0208, 0.1667]`), compared with `0.0417` for the global baseline (CI `[0, 0.1042]`), while final-angle difference is `+0.0321` rad with CI `[-0.0254, 0.0900]`. It produces a stable local mechanism improvement: predicted/observed effect error changes by `-0.0696` (CI `[-0.0884, -0.0541]`) and wrong-sign events by `-3.9583` (CI `[-5.5625, -2.4583]`). The corrected conclusion is narrower and stronger: paired probing identifies local effect and preserves boundary contact, but it still does not yield stable pose closure or online planning.

## Terminal orientation handoff control and execution-order repair (2026-10-24)

The first terminal-handoff artifact (`terminal_handoff_metric_repair_seed20261020`) was invalid as a control: the adaptive online reranker ran after terminal selection and overwrote the terminal action. The runner now gives terminal handoff priority and adds an explicit `--terminal-max-moment` ablation. The corrected artifact is `le-wm-main/results/negative_cause_diagnosis_20261015/terminal_max_moment_metric_repair_seed20261020_v2/`.

The valid maximum-moment handoff preserves physical boundary contact (`0/16`) and reduces local effect error to `0.0433`, but pose success remains `2/16`, final angle is `1.748` rad, and final-angle improvement is not established. It also increases identification-probe rate to `0.694`, indicating that the stronger terminal pulse often fails the support gate and triggers fail-closed probing. Thus the execution bug is fixed, but the mechanism hypothesis that a stronger terminal pulse alone repairs sticking is rejected; no three-seed promotion is justified.

## Contact-frame action-effect interface: three-seed CUDA replication (2026-10-15)

To test the remaining hypothesis without changing the held-out selector or
terminal threshold, the current boundary-contact controller was rerun with
the frozen contact-frame chart, paired moment-sign probes, observed-sign
calibration, uncertainty gate, online local-effect reuse, and terminal
handoff. The three artifacts are:

- `le-wm-main/results/negative_cause_diagnosis_20261015/contact_frame_boundary_panel_seed20261020/`;
- `le-wm-main/results/negative_cause_diagnosis_20261015/contact_frame_boundary_panel_seed20261021/`;
- `le-wm-main/results/negative_cause_diagnosis_20261015/contact_frame_boundary_panel_seed20261022/`.

The CUDA aggregation is
`le-wm-main/results/negative_cause_diagnosis_20261015/contact_frame_boundary_panel_20261015/`.
It uses 24 train roots and 16 root-disjoint held-out roots per seed, the same
shared candidate bank, and 10,000 CUDA bootstrap draws. Across 48 held-out
roots, adaptive online local-effect reuse gives:

- predicted/observed effect error `0.07597` versus `0.12094` for global
  one-step; paired delta `-0.04497`, 95% CI `[-0.05848,-0.03224]`;
- wrong-sign events `4.125` versus `7.1875`; paired delta `-3.0625`, 95% CI
  `[-5.4167,-0.9375]`;
- physical polygon-boundary contact loss `0/48` for both global and adaptive
  arms;
- pose success `4/48` for adaptive versus `2/48` for global; the paired
  success delta is `+0.0417`, 95% CI `[0,0.1042]`;
- final-angle delta `+0.0292` rad with 95% CI `[-0.0576,0.1158]`, so no
  stable orientation improvement is established.

This is a reproducible mechanism-level positive, not a pose-closure result:
paired excitation makes local action-effect estimates and torque-sign errors
better while preserving actual boundary contact, but the finite-horizon
action sequence still fails to transfer that local information into reliable
angle closure. The active diagnosis is now narrower: the bottleneck is not
contact measurement, contact retention, or successor-mode classification; it
is nonstationary within-contact action-to-angle transfer and insufficient
multi-step support. The six strong claims, frozen RGB online planning, and
universal commutator certificate remain closed.

## Horizon-conditioned short-pulse visual MPC replication (2026-10-15)

The next intervention tested whether the one-step-to-multi-step gap could be
reduced by using a shorter declared prediction horizon rather than adding
more static features. The frozen protocol used `prediction_horizon=4`,
`action_block=1`, 64 staged candidates, stable-contact bootstrap, visual
state source, block-pose cost, 96 train cases, 24 held-out cases, and 8
held-out episodes per seed. Base and compositional-repaired predictors used
identical candidate proposals and episode indices. All three runs used host
CUDA on the RTX 5060 Ti.

The source artifacts are:

- `le-wm-main/results/pusht/lewm/h4_shortpulse_visual_mpc_20261015_seed20261101/`;
- `le-wm-main/results/pusht/lewm/h4_shortpulse_visual_mpc_20261015_seed20261102/`;
- `le-wm-main/results/pusht/lewm/h4_shortpulse_visual_mpc_20261015_seed20261103/`.

The CUDA paired aggregation is
`le-wm-main/results/pusht/lewm/h4_shortpulse_visual_mpc_panel_20261015/`.
Across 24 paired episodes, both base and repaired visual MPC have task
success `0/24` and contact reached `0.7917`. Repaired-minus-base deltas are
`+16.26 px` block XY distance (95% CI `[-16.86,50.63]`) and `+383.68` final
distance (95% CI `[100.84,738.25]`); the repaired arm is therefore not a
downstream improvement. The intervention rejects the hypothesis that
shortening the planning horizon alone repairs visual planning.

The remaining actionable bottleneck is action-support/interface transfer:
the predictor can reduce latent or selected-endpoint error under contact
training, but its visual endpoint readout is not aligned with the actual
candidate effect distribution. The next method must change the learned
action-effect interface or add support-aware observed-effect verification;
further horizon-only sweeps are not promoted.

## Paired RGB finite-difference continuation (2026-10-15)

The short-pulse audit was extended to a matched same-root protocol. For every
candidate action, the environment is reset to the same offline root twice: one
run executes a zero-action pulse and the other executes the candidate pulse.
The RGB effect is then measured as

`RGB(action successor) - RGB(zero-action successor)`

instead of `RGB(action successor) - RGB(source)`. This removes passive motion
and contact-release drift from the effect label while keeping runtime inputs
RGB-only. Simulator state is used only for offline reset and reference scoring.

The fixed mechanism protocol used an 8-action fan, magnitude `0.35`, four
repeated physical steps, 64 contact roots per seed, and three independent CUDA
seeds (`20261035/36/37`). The aggregate is
`le-wm-main/results/negative_cause_diagnosis_20261015/paired_reobserve_grid_r4_m035_panel_20261015/`.
All three seeds improved paired effect MAE relative to the source-difference
control: the mean delta was `-0.08749 rad`, with the seed-bootstrap 95% CI
`[-0.17248,-0.03662]`. RGB successor top-1 was `0.3438`, `0.2813`, and
`0.3281` (mean `0.3177`) versus the 8-way random baseline `0.125`.

This is the first stable positive result for the new interface, but its claim
is deliberately narrow: it establishes local RGB action-effect observability,
not a learned global action-effect predictor or closed-loop planning.

The first held-out adapter used a 105-dimensional quadratic expansion and
collapsed on unseen roots (`test_top1=0.15625` in the three repeat-4 runs).
Replacing it with a 25-dimensional linear ridge that explicitly conditions on
the observed zero-successor reduced same-seed effect MAE from `1.106` to
`0.582 rad`, but held-out top-1 remained at chance (`0.125`). This identifies
two distinct failure modes: small-sample overfitting and, after that repair,
nonstationary cross-root action-effect transfer. The runtime controller must
therefore use observed short-pulse verification with support-aware fail-closed
release, or reject the action interface; it must not promote the adapter as a
global predictor.

The paired audit and its aggregate are mechanism artifacts. Frozen RGB online
planning, stable pose closure, reliable physical causality, a universal world
model, and a universal commutator certificate remain closed.

## Active paired-probe support repair (2026-10-15)

The next implementation targeted a concrete runtime cause rather than adding
another memory family.  A one-sample, zero-angle online cache entry was
previously reusable, so the controller could repeat a stale action while the
pusher drifted across the contact interface.  The repair adds informative
support counts, a matched opposite-moment requirement, fail-closed alternating
probes, mirror-aware local-chart matching, and an optional maximum declared
moment with a short identification pulse.

The CUDA panel at
`results/negative_cause_diagnosis_20261015/active_paired_probe_panel_20261015/`
contains three independent seeds and 48 root-disjoint held-out roots.  The
adaptive paired arm reduces predicted/observed effect error from `0.1492` to
`0.1014`, paired delta `-0.04775` with 95% CI `[-0.07267,-0.02157]`, while
physical boundary contact loss remains `0/48` in both arms.  However, pose
success is unchanged at `4/48`, and final angle worsens by `+0.0883` rad with
95% CI `[+0.0576,+0.1212]`.

This resolves the stale-cache and insufficient-excitation implementation
causes and establishes a reproducible local effect-transfer improvement.  It
also rules out the stronger claim that local paired identification alone
repairs multi-step orientation transfer.  The active bottleneck is now the
task-level composition/readout of several local effects, not local action
effect observability.

## Cube gripper protocol repair and v14 CUDA/EGL audit — 2026-10-18

The Cube macro protocol was repaired after tracing an apparent carry failure to
two interface/protocol issues. The OGBench Cube action convention is explicit
in the implementation now: `action[4]=+1` closes/grasp the gripper and
`action[4]=-1` opens/releases it, despite the potentially misleading
`proprio/gripper_opening` field name. The RGB-only sequence now performs a
fixed open prelude, high-height RGB XY approach, descent, close, lift, and
only then effect probing. Empty calibration replay payloads also no longer
silently replace the requested test roots.

The corrected v14 protocol was executed on the host RTX 5060 Ti with
PyTorch `2.12.0+cu130`, CUDA allocation, and MuJoCo EGL rendering all passing.
It used the same eight paired held-out roots as the v6 chart and fit on CUDA.
The result remains negative: contact rate `0.875`, success `0/8`, and mean
final block-goal distance `0.17216 m`. Per-step traces keep the block at
`z≈0.01996 m` during lift/transport, so contact is not carry. The effect map
was therefore not promoted to a successful transport controller.

This narrows the next diagnosis to the physical RGB-to-pinch alignment and
lift/carry interface, rather than gripper sign or calibration replay. The
result is a valid negative boundary artifact; it does not support stable Cube
pose closure, frozen-LeWM online planning, or a universal world-model claim.

## Evo-Depth-inspired multi-view spatial transfer audit — 2026-10-18

The paper [Evo-Depth](https://arxiv.org/abs/2605.14950) suggests three ideas
relevant to the Cube failure: complementary RGB views for implicit spatial
information, lightweight feature modulation rather than replacing the base
representation, and progressive alignment of the spatial branch with action
learning. OGBench Cube exposes a fixed `side_pixels` camera, so we tested a
minimal RGB-only transfer of the first and third ideas without using simulator
state at runtime.

The implementation adds a side-view spatial cue and a calibration-only joint
front/side action Jacobian. The front view supplies the planar alignment cue;
the side view supplies a complementary vertical relation; the controller has a
fixed probe budget and falls back rather than extrapolating unsupported
geometry. A signed-gap bug in the first implementation was caught by the
orientation-invariance test and corrected before the final run.

On the same eight held-out roots used by v6, the corrected v16 depth-gated
single-view-compatible arm remained `0/8` success with contact `0.875`; the
joint-view spatial arm v20 remained `0/8` success with contact `0.625` and
mean final distance `0.18976 m`. At close, side-view gaps remained roughly
`0.13–0.20` and block height stayed at the table. The joint Jacobian is
therefore not a successful carry solution and must not be reported as one.

The actionable conclusion is narrower and more useful: multi-view geometry is
observable, but an online centroid fusion does not learn the camera-extrinsic,
pinch-point, and within-contact action interface. The next valid intervention
is a true staged spatial adapter: first fit view-consistent geometry, then fit
action-effect alignment on held-out calibration roots, and only release when
side-view lift verification passes. This is a transfer-interface problem, not
evidence that adding a second image alone repairs frozen-LeWM planning.

## Staged multi-view spatial/action-effect training — 2026-10-18

The first actual training intervention is now implemented in
`experiments/train_cube_multiview_spatial_adapter.py`. It uses root-disjoint
RGB pairs: for each root and candidate XY pulse, a zero pulse and candidate
pulse are replayed from the same reset, and the target is the front/side RGB
object successor difference after subtracting the zero-pulse successor.
Simulator state, contact, goal distance, and planning outcomes are excluded
from the loss.

The staged model trains a geometry encoder/reconstructor first, freezes that
encoder while fitting the action-effect head, and then jointly fine-tunes both
at a lower learning rate. The CUDA run used 24 train roots, 8 held-out roots,
two Cube tasks, four candidate pulses per root, and 4-step pulses. On the
held-out roots it obtained effect RMSE `0.00258`, effect MAE `0.00184`, and
per-coordinate sign accuracy `83.59%`. The final training artifact is
`results/cube_multiview_spatial_adapter_20261018_v1/summary.json` with its
checkpoint alongside it.

This is a valid learned spatial/action-effect representation result, not yet a
planning result. The next gate is to load the frozen checkpoint into a shared
candidate-bank controller and test root-disjoint action-ranking and carry
consequence against the fixed RGB chart and no-dynamics controls. Until that
gate is passed, the six universal claims remain closed.

## Held-out action-ranking gate and loss reweighting audit — 2026-10-18

The saved root-disjoint panel was extended with a calibration-only zero-target
baseline and an independent four-candidate action-ranking evaluator. The
baseline mean is fit on train roots only; the ranking evaluator uses no test
planning outcome. In the first learned adapter run (24 train roots, 8 held-out
roots), effect RMSE improved from `0.03748` for the zero baseline to `0.00295`,
and MAE from `0.01491` to `0.00203`. However, held-out candidate top-1
agreement was unchanged at `0.50`, while pairwise agreement fell from `0.542`
to `0.375`. Thus prediction error reduction did not carry to action ranking.

The diagnosed cause was loss imbalance: the side-view effect channel has much
larger scale than the front XY channels used by the candidate ranking. A
train-root-only per-channel standardization and checkpointed inverse transform
were implemented, with tests and a reusable `.npz` panel artifact. Reusing
the exact same split, the reweighted v3 adapter reached effect RMSE `0.00280`
and MAE `0.00171`, but ranking still degraded relative to the baseline:
top-1 `0.375` versus `0.500`, pairwise `0.458` versus `0.542`.

This is an important negative result, not a tuning success: channel
standardization fixes the optimization imbalance but does not solve
cross-root action-effect identifiability. The learned adapter is therefore
not promoted to a planner. The formal next gate is observed-effect
verification or a task-conditioned ranking loss with an independently
declared action utility; closed-loop carry remains unproven and requires a
separate side-view lift gate.

## Verified-local action transfer audit — 2026-10-19

The proposed `act -> observe -> update -> rank` protocol was implemented in
`experiments/evaluate_cube_spatial_adapter_transfer.py`. A new data export
also stores the calibration-only RGB goal displacement, so the ranking score
is task-conditioned rather than an effect-norm heuristic. On each held-out
root, fixed `+x` and `+y` pulses are treated as probes; their observed RGB
effects fit a local action map, while `-x` and `-y` remain excluded from the
fit and are ranked afterward. A rank/condition gate falls back to the fixed
chart when local support is insufficient.

The first report exposed an implementation error: the candidate order is
`[+x, -x, +y, -y]`, but the evaluator initially treated the first two rows as
orthogonal probes. This forced rank-one fail-closed behavior on all roots. The
corrected evaluator passed the local support gate on `8/8` roots. It achieved
top-1 agreement `0.50` and pairwise agreement `0.375` on the unprobed
`[-x,-y]` candidates, versus `0.375/0.375` for the global adapter and
`0.375/0.375` for the fixed action chart in the same unprobed comparison.
The local-vs-global top-1 difference is `+0.125`, but the local method remains
below the train-only constant baseline (`0.875` top-1 on this small panel).

This is a mechanism improvement, not a solved planning result: local
verification removes the global-adapter failure mode but does not yet beat a
constant predictor under the current four-action, pre-contact pulse panel.
The next data collection must move the probes into a verified contact phase
(close/lift gate before XY probing) and expand the candidate bank; current
closed-loop carry and pose closure remain unproven.

## Eight-direction held-candidate audit and contact-support diagnosis — 2026-10-19

The evaluator now reads the declared `candidate_actions` array from the data
artifact instead of assuming four rows per root. The 8-direction protocol uses
fixed probes at `+x`, `+y`, and `-x` (positions `[0, 2, 4]`) and ranks the five
remaining diagonal/negative-direction candidates. Held candidate outcomes are
never used for local fitting. This closes the previous evaluator/data-contract
bug and is covered by a unit test.

On the GPU smoke panel (8 train roots, 4 held-out roots, 8 actions, 4-step
pulse, no approach/close), the adapter reduced held-out effect RMSE from
`0.03672` to `0.00386`, but sign accuracy stayed unchanged (`0.46875` versus
`0.46809`). On the five genuinely unprobed candidates, verified-local reached
pairwise agreement `0.625` versus `0.325` for the global adapter, but top-1 was
only `0.25` and the zero baseline remained `0.75`. The per-root audit showed
two roots with zero or near-zero probe effect and two roots where the probe map
was numerically stable but its held-out action direction was not transferable.

A separate RGB approach+close smoke (2 train roots, 2 held-out roots) improved
the transfer picture: verified-local pairwise agreement was `0.817` versus
`0.254` for the global adapter and `0.724` for the zero baseline, with top-1
`0.50` versus `0.0` and `0.5`. This is exploratory and underpowered, so it is
not promoted to a formal positive claim. It does establish the actionable
failure diagnosis: the negative result is caused primarily by missing or
unstable contact/action-effect support, not by predictor RMSE alone. The next
formal intervention must therefore gate on RGB contact support and report
coverage, probe residual, and held-candidate ranking jointly. No planner or
carry claim is opened by either smoke result.

## Pulse-horizon and support-gate follow-up — 2026-10-19

Changing the RGB contact-support panel from a 4-step to a 12-step XY pulse
did not repair transfer. The 12-step panel increased observed effect scale
(held-out effect RMSE `0.07062` versus zero baseline `0.11187`), but verified
local held-candidate pairwise agreement fell to `0.603`, below the global
adapter's `0.802`. This indicates that a longer pulse crosses changing
contact modes; it is not a safe substitute for a verified short pulse.

The evaluator now uses a predeclared minimum mean probe-effect norm of
`5e-4` and reports support coverage separately from ranking. On the 8-root/
4-held-out contact panel, the gate rejected 2/4 roots (`support_coverage =
0.50`). On the two eligible roots, local ranking reached pairwise agreement
`0.80`; unconditional ranking remained only `0.614`. This is the desired
fail-closed behavior: the method identifies where local verification is
usable, but it does not convert partial support into a global planning claim.

## Calibrated RGB approach mapping intervention — 2026-10-19

To test whether the remaining failure was only a camera/action-coordinate
problem, the collector now optionally fits a 2-D RGB effector-action Jacobian
on independent calibration roots (`seed + 100000`) and uses it for the
approach phase. The mapping is recorded in the artifact and is never fitted on
training or held-out outcome roots.

The mapped approach+close smoke produced a full-rank mapping but a calibration
fit RMSE of `0.00787`; its held-out effect RMSE was `0.02724` versus `0.03552`
for the zero baseline. However, held-candidate local pairwise agreement was
only `0.55`, with top-1 `0/2`, and was identical to the fixed/global controls
on this small panel. Thus camera-coordinate calibration alone does not repair
transfer. The remaining failure is root-specific contact/grasp state and
within-mode action-effect nonlinearity. The mapping is retained as an
isolated calibration option, but is not promoted as a positive planning result.

## Explicit RGB contact-proxy routing — 2026-10-19

The source RGB chart reveals a stronger repairability separator than the
camera Jacobian: front-view effector/object distance. The evaluator now
requires this deployable proxy to be at most `0.05` before using the local
probe map; otherwise it abstains before ranking and records
`rgb_contact_proxy_failed`.

On the 8-root/4-held-out contact panel, the gate rejected exactly the two
roots whose front distances were `0.239` and `0.260`; the two accepted roots
had distances `0.038` and `0.030`. Accepted-root local pairwise agreement was
`0.80`, while support coverage was `0.50`. This converts the previous opaque
negative result into an actionable routing rule: contact-unobservable roots
must be approached/re-probed or rejected, and cannot be passed to the action
ranker. It is a selective-transfer result, not yet a closed-loop carry result.
