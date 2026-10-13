# Latent State Spectral Transfer

## Finite-horizon theorem package

The formal theorem, proof, estimability table, applicability proposition, and
failure boundaries are documented at
`/home/gbk/code/LeWorldModel/docs/FINITE_HORIZON_TRANSFER_RISK_THEOREM.md`.
Run the executable theory audit with:

```bash
python experiments/theory_spectral_transfer/run_finite_horizon_transfer_risk_decomposition.py \
  --output results/finite_horizon_transfer_theorem_audit_20261013
```

The artifact contains a tightness witness, a matched scalar-diagnostic
adversarial family, a nonlinear Jacobian-only counterexample, a non-normal
transient comparison, and a local learned-nonlinear applicability check. These
are theory/synthetic artifacts, not learned-model planning evidence.

`nonlinear_transfer_certificate.py` returns `certified_local` only when the
local remainder budget, action support, independent planner-reachable
transition split, independent readout calibration, and local-tube conditions
all pass.

This package implements the theoretical latent-state spectral-transfer proposal
separately from empirical rollout audits.  The core paper is not limited to
LeWorldModel: LeWorldModel PushT and JEPA measurements are learned-representation
anchors for the finite-dimensional theory, not replacements for the theorem.

It models:
- source/world-factor spectrum `C_theta`;
- encoder transfer metric `G_E`;
- observed latent spectrum `C_theta^{1/2} G_E C_theta^{1/2}`;
- predictor response and residual spectrum;
- multi-horizon predictor composition and residual growth;
- heat-filter style intervention;
- oblique-frame finite-window slope stress;
- same-spectrum / different-basis operator-alignment noncommutativity;
- known-generator heat-filter recovery;
- SIGReg local-anisotropy and identifiability/controllability toy cases;
- residual-band penalty primitives.

Later-stage branches kept separate:
- operator-alignment / noncommutativity sweep;
- operator-alignment empirical bridge on frozen PushT latents;
- controllability-alignment sweep;
- joint mechanism sweep for encoder/predictor interaction and finite-window bias.

Run:

```bash
python experiments/theory_spectral_transfer/run_analysis.py \
  --output results/theory_spectral_transfer.json
```

The output is a JSON report with spectra, compact spectrum descriptors, and
log-log slope fits. It also includes a multi-horizon section that composes the
one-step predictor transfer across horizons such as 1, 2, 4, and 8 and reports
residual-trace growth across those horizons.

For a paper-style summary sweep, run:

```bash
python experiments/theory_spectral_transfer/run_rich_sweep.py \
  --output results/theory_rich_sweep_20260801
```

This writes `theory_rich_sweep.json`, `paper_summary.md`, `paper_summary.pdf`,
and `paper_summary.png` under the output directory.

For a cross-dimension summary over precomputed sweeps, run:

```bash
python experiments/theory_spectral_transfer/summarize_dimension_sweep.py \
  --input results/theory_dim_sweep_20260801 \
  --output results/theory_dim_sweep_20260801
```

For a seed/noise robustness sweep, run:

```bash
python experiments/theory_spectral_transfer/run_robustness_sweep.py \
  --output results/theory_robustness_sweep_20260801
```

For a separate controllability-alignment sweep with fixed spectrum and
rotated control subspaces, run:

```bash
python experiments/theory_spectral_transfer/run_controllability_sweep.py \
  --output results/theory_controllability_sweep_20260828
```

For the operator-alignment / noncommutativity sweep, run:

```bash
python experiments/theory_spectral_transfer/run_operator_alignment_sweep.py \
  --output results/theory_operator_alignment_sweep_20260828
```

For the stronger falsifier sweep, run:

```bash
python experiments/theory_spectral_transfer/run_operator_alignment_null_sweep.py \
  --output results/theory_operator_alignment_null_sweep_20260829
```

For the continuous anisotropy boundary sweep, run:

```bash
python experiments/theory_spectral_transfer/run_operator_alignment_anisotropy_sweep.py \
  --output results/theory_operator_alignment_anisotropy_sweep_20260829
```

For the transfer-proxy robustness sweep, run:

```bash
python experiments/theory_spectral_transfer/run_operator_alignment_metric_robustness.py \
  --output results/theory_operator_alignment_metric_robustness_20260829
```

For the spectrum-vs-alignment decoupling table, run:

```bash
python experiments/theory_spectral_transfer/run_operator_alignment_decoupling.py \
  --output results/theory_operator_alignment_decoupling_20260829
```

For the theory paper's frozen-latent bridge artifact, run:

```bash
python experiments/theory_spectral_transfer/run_operator_alignment_bridge.py \
  --output results/theory_empirical_bridge_20260829
```

For the compact frozen PushT latent anchor figure used by the theory paper, run:

```bash
python experiments/theory_spectral_transfer/make_frozen_pusht_anchor_figure.py \
  --output-dir results/theory_operator_alignment_iclr_oral_20260829/figures
```

For official JEPA resource acquisition, run:

```bash
python experiments/theory_spectral_transfer/download_jepa_resources.py \
  --root external \
  --include-ijepa-image
```

This writes `external/jepa_acquisition_manifest.json`, downloads the official
JEPA-WM PushT checkpoint from `facebook/jepa-wms`, clones the official
`facebookresearch/jepa-wms` repository, and optionally downloads the official
I-JEPA ViT-H/16 image checkpoint from `facebook/ijepa_vith16_1k`.

For the I-JEPA learned-representation feature anchor, run:

```bash
python experiments/theory_spectral_transfer/run_ijepa_feature_anchor.py \
  --dataset cifar10 \
  --output-dir results/theory_operator_alignment_iclr_oral_20260829/jepa_anchor_cifar10_hflip \
  --max-images 512 \
  --batch-size 4
```

The paper-facing JEPA anchor should use standard natural-image benchmarks.  The
current script supports `cifar10`, `cifar100`, `stl10`, `svhn`,
`tiny-imagenet`, and `food101`; these six datasets have local or cached
validation paths in this checkout.  `cifar100` falls back to the public
Hugging Face mirror when the official Toronto tarball is unavailable or
corrupted.  The older
`sklearn-digits` mode remains only as a smoke test and should not be used as
the main learned-representation evidence.

ImageNet-1K is the field-standard large-scale anchor for I-JEPA, but the
`ILSVRC/imagenet-1k` Hugging Face dataset is gated.  On this machine,
`git clone https://huggingface.co/datasets/ILSVRC/imagenet-1k` fails because the
stored Hugging Face credential is invalid, and the SSH clone path fails before
authentication.  Do not count ImageNet-1K as downloaded until `hf auth login`
has been refreshed and the dataset license gate is accepted.

For the combined I-JEPA figure used by the manuscript, run:

```bash
python experiments/theory_spectral_transfer/make_ijepa_anchor_figure.py \
  --output-dir results/theory_operator_alignment_iclr_oral_20260829/figures
```

This now builds the six-benchmark dataset sweep figure for the frozen I-JEPA
anchor.

For a held-out operator-pair stress test in the same frozen I-JEPA feature
space, run:

```bash
PYTHONPATH=. python -m experiments.theory_spectral_transfer.run_jepa_operator_anchor \
  --model external/ijepa_vith16_1k_hf \
  --dataset sklearn-digits \
  --max-images 120 \
  --operator-rank 16 \
  --output-dir results/theory_operator_alignment_iclr_oral_20260829/jepa_operator_anchor
```

The script fits a ridge linear map from source to transformed-view features in
source PCA coordinates, evaluates prediction on held-out pairs, and compares
its covariance commutator with orthogonal conjugates that preserve the fitted
operator spectrum.  This is a diagnostic of estimator identifiability and a
learned-representation anchor; it is not a JEPA training-quality, planning, or
action-conditioned-dynamics result.  Small-sample runs are intentionally
treated as stress tests rather than main-text evidence.

For the manuscript's rank/sample stress test, summarize the reproduced runs with:

```bash
PYTHONPATH=. python -m experiments.theory_spectral_transfer.summarize_jepa_operator_rank_sweep \
  results/theory_operator_alignment_iclr_oral_20260829/jepa_operator_anchor_cifar10_rank8/jepa_operator_anchor.json \
  results/theory_operator_alignment_iclr_oral_20260829/jepa_operator_anchor_cifar10_rank16/jepa_operator_anchor.json \
  results/theory_operator_alignment_iclr_oral_20260829/jepa_operator_anchor_digits120_rank8/jepa_operator_anchor.json \
  results/theory_operator_alignment_iclr_oral_20260829/jepa_operator_anchor_digits120/jepa_operator_anchor.json \
  results/theory_operator_alignment_iclr_oral_20260829/jepa_operator_anchor_digits120_rank32/jepa_operator_anchor.json \
  --output-dir results/theory_operator_alignment_iclr_oral_20260829/jepa_operator_rank_sweep
```

The current sweep covers CIFAR-10 (ranks 8 and 16) and scikit-learn-digits
(ranks 8, 16, and 32). It is deliberately reported as a negative
estimator-identification boundary: held-out view prediction can remain high
while the covariance--operator commutator stays below the spectrum-preserving
null.

Before admitting an action-conditioned temporal claim, audit the independent
JEPA-WM promotion gate:

```bash
python experiments/theory_spectral_transfer/audit_jepa_wm_temporal_gate.py \
  --checkpoint data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar \
  --dataset-root data/jepa_wm_osf/extracted/pusht_noise \
  --output-dir results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate
```

The audit records checkpoint provenance, trajectory-file presence, required
manifest fields, and the held-out multi-step metrics. A directory created by a
failed or gated download is not treated as a closed trajectory protocol. The
current PushT trajectory bundle is mirrored from the public DINO-WM OSF release
and its SHA-256 is recorded in the experiment log; the temporal metrics remain
available as a claim-bounded one-frame-context diagnostic, while the intended
multi-frame predictor interface remains a separate gate.

With the public bundle available, run the claim-bounded CPU diagnostic:

```bash
source .venv/bin/activate
PYTHONPATH=external/jepa-wms:/tmp/dinov2 \
python experiments/theory_spectral_transfer/run_jepa_wm_temporal_eval.py \
  --dataset-root data/jepa_wm_osf/extracted/pusht_noise \
  --checkpoint data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar \
  --output results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/jepa_wm_temporal_metrics.json
```

The released AdaLN wrapper is evaluated with a two-frame context after the
historical action-token alignment correction. Longer-context and cross-checkpoint
generalization remain outside the promoted claim.

An independent cross-family replay is available from the public DINO-WM
PointMaze release. The public archive omits a serialized split manifest, but its
released loader specifies an episode-level 0.9 split with `torch.randperm` seed
42. The evaluator reconstructs those 1,800/200 IDs and emits an auditable
manifest alongside the metrics:

```bash
source .venv/bin/activate
PYTHONPATH=/tmp/dino_wm:/tmp/dinov2 \
python experiments/theory_spectral_transfer/run_dino_wm_pointmaze_temporal_eval.py \
  --dataset-root data/dino_wm_osf/extracted/point_maze \
  --checkpoint data/dino_wm_osf/extracted/outputs/point_maze/checkpoints/model_latest.pth \
  --output results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/dino_wm_pointmaze_temporal_metrics_200.json \
  --split-mode code_reconstructed --split-seed 42 --split-ratio 0.9 --save-vectors
```

A second DINO-WM checkpoint is evaluated on the same release's explicit PushT
validation directory (18,685 train / 21 validation episodes):

```bash
source .venv/bin/activate
PYTHONPATH=/tmp/dino_wm:/tmp/dinov2 \
python experiments/theory_spectral_transfer/run_dino_wm_pusht_temporal_eval.py \
  --dataset-root data/jepa_wm_osf/extracted/pusht_noise \
  --checkpoint data/dino_wm_osf/extracted/outputs/pusht/checkpoints/model_latest.pth \
  --output results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/dino_wm_pusht_temporal_metrics.json
```

For a matched-environment JEPA-WM cross-check on PointMaze:

```bash
source .venv/bin/activate
PYTHONPATH=external/jepa-wms:/tmp/dinov2 \
python experiments/theory_spectral_transfer/run_jepa_wm_pointmaze_temporal_eval.py \
  --dataset-root data/dino_wm_osf/extracted/point_maze \
  --checkpoint data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar \
  --output results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/jepa_wm_pointmaze_temporal_metrics_200.json \
  --episodes 200 --batch-size 8 --save-vectors
```

Held-out operator controls (paired, pair-shuffled, and spectrum-preserving
rank/regularization ablations) are reproduced with:

```bash
source .venv/bin/activate
python experiments/theory_spectral_transfer/analyze_temporal_operator_controls.py \
  --input results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/jepa_wm_pointmaze_temporal_metrics_200.json \
  --label JEPA-WM\ PointMaze \
  --input results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/dino_wm_pointmaze_temporal_metrics_200.json \
  --label DINO-WM\ PointMaze \
  --output results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/pointmaze_temporal_operator_controls.json
```

Action intervention replays use the same evaluators with
`--action-mode zero`, `--action-mode reverse`, or the stricter DINO-only
`--action-mode future_zero` / `--action-mode future_reverse` variants. The
future-only variants preserve the historical action prefix and modify only
future prediction blocks. The `zero` and `reverse` modes remain the legacy
whole-sequence stress tests. The evaluators also support
`--action-mode episode_shuffle`. For the latter, pass
`--action-permutation-seed <integer>` to use a reproducible random
cross-episode derangement with no fixed points; omit the seed only for the
legacy deterministic one-step cyclic shift. The resulting PointMaze and
PushT controls are stored under the temporal-gate artifact directory. The
multi-seed derangement audit is summarized with:

```bash
python experiments/theory_spectral_transfer/analyze_derangement_sensitivity.py \
  --observed <observed.json> --intervention <seed0.json> \
  --intervention <seed1.json> --intervention <seed2.json> \
  --label <model> --interventions-per-label 3 \
  --output <derangement_sensitivity.json>
```

For a stricter temporal-order control that preserves each episode's exact
future-action block multiset, use `--action-mode within_episode_shuffle` with
the same seed flag. DINO-WM evaluators preserve the historical context action
prefix and shuffle only future blocks; the JEPA-WM evaluator has no explicit
historical action suffix and therefore shuffles all supplied future blocks.
The paired cross-episode/within-episode comparison is generated with
`compare_action_interventions.py`.

The hardened replay matrix is resumable from:
`run_hardened_action_ablations.sh`. It adds Wall split-0/1 checks, three
cross-episode derangement seeds, and future-only DINO controls on PushT,
PointMaze, and Wall. `summarize_hardened_ablation.py` produces the paired
statistics, exact small-sample sign-flip tests, and Holm-adjusted p-values.
The bundle's `notes.txt`, `write_hardened_manifest.py`, and generated SHA-256
manifest are the reproducibility handoff surface.

For a cross-model held-out geometry audit over the already generated temporal
vectors, run:

```bash
source .venv/bin/activate
python experiments/theory_spectral_transfer/run_cross_model_temporal_geometry_audit.py \
  --output results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/cross_model_temporal_geometry_audit.json \
  --split-seeds 0 1 2 --top-k 8
```

The formal artifact compares JEPA-WM and DINO-WM on PushT and PointMaze at
horizons 1/2/4. Operators are fitted on internal split-train vectors and
evaluated on split-holdout residuals. The current 36-cell audit does not
support a universal lower-commutator-is-better law: cross-model associations
are descriptive and must not be promoted to online planning or causal-control
claims.

To test dependence on the reconstructed PointMaze episode membership, rerun
the evaluators with `--split-seed 0`, `--split-seed 1`, and the original
`--split-seed 42`, then summarize the replay metrics with:

```bash
python experiments/theory_spectral_transfer/summarize_split_robustness.py \
  --input <seed42.json> --label <model> \
  --input <seed0.json> --label <model> \
  --input <seed1.json> --label <model> \
  --output <split_robustness.json>
```

The corresponding cross-fitted operator controls are summarized with
`summarize_split_operator_controls.py`. These checks strengthen robustness to
the reconstructed split but do not turn it into a serialized provenance
manifest.

An evaluator-defined tail holdout can be run with
`--split-mode tail --holdout-start 1800 --holdout-end 2000`; report it
separately from the code-reconstructed split because it is a robustness stress
test, not recovery of the checkpoint's original membership.

The third-environment Wall comparison uses the public Wall state/action archive
and the released Wall checkpoints. The downloaded zip contains images for all
1,920 trajectories, while the working tree may contain only an extracted
subset; therefore the evaluator records an explicit image-backed episode
manifest rather than silently skipping unavailable visual episodes. The
current seed-42 manifest contains 192 episodes and is shared by both model
families:

```bash
source .venv/bin/activate
python experiments/theory_spectral_transfer/run_jepa_wm_wall_temporal_eval.py \
  --dataset-root data/dino_wm_osf/extracted/wall_single \
  --checkpoint data/jepa_wm_checkpoints/jepa_wm_wall.pth.tar \
  --output results/theory_operator_alignment_iclr_oral_20260829/wall_temporal_gate/jepa_wm_wall_observed_metrics.json \
  --episodes 192 --split-seed 42 --save-vectors

python experiments/theory_spectral_transfer/run_dino_wm_wall_temporal_eval.py \
  --dataset-root data/dino_wm_osf/extracted/wall_single \
  --checkpoint data/dino_wm_osf/extracted/outputs/wall_single/checkpoints/model_latest.pth \
  --output results/theory_operator_alignment_iclr_oral_20260829/wall_temporal_gate/dino_wm_wall_observed_metrics.json \
  --episodes 192 --split-seed 42 --save-vectors
```

Repeat either command with `--action-mode zero` or `--action-mode reverse` for
the direct action ablations. For the stricter temporal-order ablation, use
`--action-mode within_episode_shuffle --action-permutation-seed 0` (and seeds 1
and 2). Then run `analyze_action_intervention_statistics.py`,
`analyze_derangement_sensitivity.py`, and
`analyze_temporal_operator_controls.py` on the emitted JSON files. The bundled
`summarize_wall_baseline_ablation.py` writes the reader-facing matrix and two
figures, while keeping the latent-error scale warning and checkpoint-local
evidence boundary explicit.

The sweep writes `operator_alignment_theorem.md`, `operator_alignment_law_card.md`,
`operator_alignment_oral_object_card.md`, `operator_alignment_paper_outline.md`,
`operator_alignment_paper_spine.md`, `operator_alignment_paper_draft.md`, and
`operator_alignment_research_plan.md`.

The current theory-paper package keeps the main-text reference set compact:
five core citations in the Related Work section, with adjacent background
material deferred out of the main story.

For the joint encoder/predictor interaction, heat-filter sensitivity, and
oblique-frame mechanism sweep, run:

```bash
python experiments/theory_spectral_transfer/run_mechanism_sweep.py \
  --output results/theory_mechanism_sweep_20260801
```

This writes `theory_mechanism_sweep.json`, `mechanism_summary.md`,
`mechanism_summary.pdf`, and `mechanism_summary.png`. It is analytical
toy-model evidence rather than a trained-checkpoint measurement.

`alignment.py` contains `THEORY_PROPOSAL_ALIGNMENT`, a machine-checkable mapping
from the theory proposal topics to concrete APIs and report fields.
