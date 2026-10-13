# Empirical Spectral Audit

This package implements the empirical LeWM latent audit proposal separately
from the theory simulation code.

It consumes an existing `latents.npz` with arrays such as `latents`, `dynamics`,
`planning`, `nuisance`, `predicted_dynamics`, `prediction_residual`, `states`,
and `actions`.

Optional richer fields are detected automatically: `dynamics_h*`,
`predicted_dynamics_h*`, `prediction_residual_h*`, `goals`, `success`,
`episode_id`, `timestep`, `physical_*`, and `counterfactual_*`.
`episode_id` enables `--condition episode`; `timestep` enables
`--condition phase`.

Run:

```bash
python experiments/empirical_spectral_audit/run_audit.py \
  --latents results/pusht/lewm/latents.npz \
  --output results/pusht/lewm/empirical_spectral_audit.json \
  --condition states \
  --horizons 1 3 5
```

The report includes DTL/top-band lift, rank-wise transfer fingerprints, PSE,
PPTE, NCR, random-basis controls, norm-matched random tangent controls, and
Gaussian tangent controls. It also includes shuffle controls, multi-horizon
dynamics, controllability, observability probes, eigenspace transport, and
offline band/factor interventions when the required arrays are available. When
Phase 1 exports rich fields, the audit consumes `dynamics_h*`,
`predicted_dynamics_h*`, `prediction_residual_h*`, `physical_*`, and
`counterfactual_*` arrays directly. It also implements the final proposal
directions with `spectral_planner_metric`, `spectral_exploration`, and optional
`ood_spectral_shift` when `ood_latents` is included in the input file.

Later-stage empirical bridge summary:

```bash
python experiments/empirical_spectral_audit/run_operator_alignment_bridge.py \
  --audit results/pusht/lewm/empirical_spectral_audit.json \
  --theory results/theory_operator_alignment_sweep_20260828/operator_alignment_theorem.json \
  --output results/pusht/lewm/empirical_operator_alignment_bridge_20260829
```

This is a post-hoc synthesis of the frozen empirical audit. It keeps the
empirical layer separate: the audit remains the measurement surface, and the
bridge only summarizes basis-sensitive lift, label-shuffle rivals, transport,
and planner anisotropy.

Empirical paper spine:

```bash
python experiments/empirical_spectral_audit/run_operator_alignment_bridge.py \
  --audit results/pusht/lewm/empirical_spectral_audit.json \
  --output results/pusht/lewm/empirical_operator_alignment_bridge_20260829
```

This also writes `empirical_operator_alignment_paper_outline.md`, the planning
outline for the empirical paper, and `empirical_operator_alignment_paper_spine.md`,
the manuscript-form spine. They are separate from the synthetic theory paper
and do not depend on the theorem as a premise.

The bridge also writes `empirical_operator_alignment_paper_spine.md`, the
manuscript-form version of the empirical paper structure.

`alignment.py` contains `EMPIRICAL_PROPOSAL_ALIGNMENT`, a machine-checkable
mapping from the empirical proposal topics to concrete APIs and report fields.

## Alignment-regularized LeWM prototype

The research extension is implemented as an opt-in term in
`experiments/train_lewm_directional.py`. It fits a ridge local linear operator
from the current latent batch to the predicted next embeddings, detaches the
current covariance estimate, and penalizes the normalized commutator between
that covariance and operator. The default configuration remains unchanged:
`alignment_mode: none` and `alignment_weight: 0.0`.

The two supported modes are:

- `global`: one covariance--predictor operator over all transitions;
- `conditional`: action-magnitude quantile bins, averaged over groups meeting
  `alignment_min_samples`.

The corresponding configuration fields are `alignment_weight`,
`alignment_mode`, `alignment_bins`, `alignment_min_samples`, and
`alignment_ridge`. The regularizer is an intervention, not an assumption that
zero commutator is universally optimal. In particular, global SIGReg-induced
isotropy can make the statistic uninformative; conditional geometry and
rollout/planning outcomes must be evaluated jointly.

Run the deterministic mechanism smoke without model training:

```bash
python experiments/empirical_spectral_audit/run_alignment_regularizer_sweep.py \
  --output results/alignment_regularizer_sweep_20261002/alignment_regularizer_sweep.json \
  --samples 65536 --dim 8 --seed 0
```

This produces a JSON report and a Markdown summary with fixed-spectrum angle
sweeps and an isotropic-covariance control. It must not be cited as evidence
that the learned LeWM improves until a matched training run with prediction,
multi-step rollout, and planning metrics is complete.

### Learned action-conditioned composition benchmark

`run_alignment_training_synthetic.py` is the matched learned benchmark for the
intervention question. It uses two action groups with heterogeneous covariance
and transition geometry, keeps generated splits and initialization conventions
fixed across modes, and compares `none`, `global`, and `conditional` alignment
across three seeds and five nonzero weights. In addition to one-step prediction
and repeated-action rollout, it reports `action_composition_error` for the
ordered pairs `0->0`, `0->1`, `1->0`, and `1->1`. The composition metric
compares learned operator products with the data-generating products; it is not
a claim that a nonlinear LeWM predictor is globally linear.

The canonical CPU artifact is
`results/alignment_composition_20261003/alignment_training_synthetic.json`,
with the reader-facing summary in the same directory. The experiment supports
only the bounded conclusion that conditional alignment can preserve local
geometry and prediction in this controlled benchmark, while strong global
alignment can impose an inappropriate pooled geometry. A lower commutator is
not sufficient evidence of better prediction, controllability, or planning.

For a trainable pixel encoder rather than a fixed linear predictor, run
`run_pixel_world_model_synthetic.py`. Its canonical three-seed artifact is
`results/pixel_world_alignment_20261003/pixel_world_model_synthetic.json`.
The benchmark uses next-embedding prediction plus SIGReg, validation-selected
checkpoints, random-sequence latent rollouts, all ordered two-action pairs,
and stable-rank diagnostics. It remains synthetic pixel-world evidence and
must not be reported as end-to-end LeWM or PushT evidence.

### Functional alignment follow-up

`run_alignment_functional_synthetic.py` keeps the same learned linear
alignment intervention but evaluates random action-sequence rollouts and a
candidate-set planning proxy. It uses validation-MSE checkpoint selection,
three seeds, the same global/conditional modes and five nonzero weights, and
an independent true target trajectory for the planning goal. The canonical
33-record CPU artifact is
`results/alignment_functional_synthetic_20261003/alignment_functional_synthetic.json`.

The intended result is a functional boundary, not a monotone optimization
claim: conditional alignment can preserve function at small and intermediate
weights, whereas strong global alignment can reduce a geometry diagnostic while
damaging composition, long-horizon rollout, and candidate-set regret. The
planning metric is not online task success, and the benchmark is not evidence
for nonlinear LeWM, PushT, causal control, or GPU performance.

### Nonlinear local-alignment follow-up

`run_nonlinear_local_alignment_synthetic.py` extends the functional intervention
to a nonlinear state-dependent transition. Its conditional loss uses
finite-difference predictor Jacobians and nearest same-action state
neighbourhood covariances; the comparison includes global alignment, local
alignment, predictor equivariance, and their joint condition. The canonical
10-seed artifact is
`results/nonlinear_local_alignment_10seed_20261003/nonlinear_local_alignment.json`.
The companion matched one-step report is under
`results/matched_nonlinear_local_inference_20261003/`. These are CPU synthetic
functional evidence only: they test whether local geometry changes long-horizon
behavior under a controlled nonlinear proxy, not LeWM, PushT, online planning,
or causal-control gains.

The planning-horizon follow-up is stored in
`results/nonlinear_planning_horizon_10seed_20261003/`. It evaluates horizons
2/4/8 using a trajectory-cost candidate-set proxy and reports same-seed
bootstrap/sign-flip/Holm inference. The earlier final-endpoint horizon metric
was retained only as a diagnostic because stable dynamics can make it saturate
at long horizons.

The control extension is stored separately in
`results/nonlinear_local_alignment_controls_10seed_20261003/` with matched
inference in
`results/matched_nonlinear_local_controls_inference_20261003/`. It adds an
anti-alignment arm and a random-Jacobian arm whose auxiliary gradient norm is
matched to conditional alignment. The random arm is not one-step-error matched
in all seeds, so it is a directionality/control diagnostic rather than causal
proof.

### Checkpoint-matched nonlinear geometry follow-up

`run_checkpoint_matched_geometry_synthetic.py` closes the checkpoint-selection
confound at the nonlinear functional boundary. For each seed, the baseline
checkpoint is selected by validation one-step MSE; the global and conditional
intervention checkpoints are then selected by nearest validation one-step MSE.
Only after this selection are test error, ID/OOD rollout, composition, local
geometry, and trajectory-cost planning evaluated. The formal 10-seed artifact
is `results/checkpoint_matched_geometry_10seed_20261003/`. Conditional alignment
improves the two rollout readouts and the local commutator in this matched
comparison, but worsens one-step error, composition error, and planning regret.
This is a CPU-only synthetic checkpoint-matched proxy, not causal evidence or
nonlinear LeWM/PushT performance.

### Nonlinear alignment-strength phase sweep

`run_nonlinear_alignment_phase_synthetic.py` sweeps alignment weights
`1e-4, 1e-3, 1e-2, 1e-1, 1` for global and conditional local alignment in the
nonlinear benchmark. Every seed and weight shares generated splits,
validation-MSE checkpoint selection, rollout generators, and OOD
trajectory-cost planning candidates. The formal artifact is
`results/nonlinear_alignment_phase_10seed_20261003/` (110 records, 10 seeds).
Conditional alignment progressively lowers local commutator and ID/OOD rollout
error, while composition error and trajectory-planning regret increase at
larger weights; global alignment has negligible functional effect. This is
evidence for a strength-dependent trade-off, not evidence that lower
commutator is universally better or that planning improves.

The checkpoint-matched companion is
`results/nonlinear_alignment_phase_matched_10seed_20261003/`. For weights up
to `0.1`, nearest validation-MSE gaps remain below `1.493%` and the rollout
trade-off persists after matching. Conditional weight `1` has a maximum gap of
`49.0%`; it is retained as a stress test but must not be described as a strict
matched-prediction comparison.
