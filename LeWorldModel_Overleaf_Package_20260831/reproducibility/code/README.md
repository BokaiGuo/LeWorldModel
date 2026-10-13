# Latent State Spectral Transfer

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
- latent/predictor equivariance intervention benchmark;
- operator-alignment empirical bridge on frozen PushT latents;
- controllability-alignment sweep;
- joint mechanism sweep for encoder/predictor interaction and finite-window bias.

For the controlled latent/predictor equivariance ablation, run from the
`equivariance_interventions/` directory:

```bash
python run_predictor_equivariance_synthetic.py \
  --output ../../results/predictor_equivariance_synthetic_20261003/predictor_equivariance_synthetic.json \
  --epochs 120 --seeds 0 1 2 --weights 1.0 10.0
```

This is a CPU-only synthetic vector-world mechanism benchmark. It separates
latent and predictor equivariance interventions and does not claim LeWM,
PushT, planning, causal controllability, or GPU performance.

For the functional OOD and candidate-set planning-proxy follow-up, run:

```bash
python run_equivariance_functional_synthetic.py \
  --output ../../results/equivariance_functional_synthetic_20261003/equivariance_functional_synthetic.json \
  --epochs 120 --seeds 0 1 2 --weights 1 2 3 5 10
```

This produces 48 per-seed records covering action-magnitude OOD, shifted
initial states, combined OOD, h=8 rollout, and a controlled candidate-set
latent-planning proxy. The proxy is not online task success. Results are
synthetic-only and CPU-only. The project PyTorch environment cannot initialize CUDA/NVML
(`torch.cuda.is_available()` is `False` and the visible device count is zero),
so no CUDA training or GPU-efficiency claim is attached.

For the functional alignment follow-up, run:

```bash
python run_alignment_functional_synthetic.py \
  --output ../../results/alignment_functional_synthetic_20261003/alignment_functional_synthetic.json \
  --epochs 300 --seeds 0 1 2 --weights 0.01 0.1 1 10 100
```

This produces 33 per-seed records covering random action-sequence rollout,
ordered composition, and a candidate-set planning proxy with an independent true
target trajectory. The table reports means over three seeds; per-seed records and
standard deviations remain in JSON. It is synthetic-only and CPU-only. The proxy
is not online task success or causal control, and no CUDA or GPU-efficiency claim
is attached because the project PyTorch environment cannot initialize CUDA/NVML.

For the exact action-sequence composition audit, run:

```bash
python run_action_sequence_composition_synthetic.py \
  --output ../../results/action_sequence_composition_20261003/action_sequence_composition_synthetic.json \
  --epochs 300 --seeds 0 1 2 --weights 0.01 0.1 1 10 100
```

This enumerates all binary action sequences at horizons 2, 4, and 8 (4, 16,
and 256 products), compares learned and data-generating products using the
same ordered convention, and joins the result to the matched ID/OOD rollout
records. The association is descriptive evidence from a controlled linear
synthetic benchmark; it is not a causal, nonlinear-LeWM, PushT, online-planning,
or GPU-performance result.

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
  --split-mode code_reconstructed --split-seed 42 --split-ratio 0.9
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
  --output results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/jepa_wm_pointmaze_temporal_metrics_21.json \
  --episodes 21 --batch-size 4
```

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

### Contact-predictor repair diagnostics

The extended PushT repair runner is
`contact_predictor_repair/run_pusht_contact_predictor_teacher_forced_repair.py`.
It freezes the visual encoder/projector, repairs only the action-conditioned
predictor modules, evaluates autoregressively, and records the three matched
`+a/-a/zero` arms, calibration-size sweep, and observed-latent interface upper
bound. The two seed-level results are under
`results/contact_predictor_deep_20261005_seed8421/` and
`results/contact_predictor_deep_20261005_seed9271/`.

From the LeWorldModel worktree, the formal protocol is:

```bash
.venv/bin/python experiments/run_pusht_contact_predictor_teacher_forced_repair.py \
  --output results/pusht/lewm/contact_predictor_deep_20261005_seed8421 \
  --train-cases 48 --held-out 16 --epochs 8 \
  --horizons 1 2 4 8 --train-horizons 1 2 4 8 \
  --calibration-cases 8 16 32 48 --seed 8421
```

The matching `9271` run changes only the output directory and seed. Use
`summarize_pusht_contact_predictor_deep.py` for a seed-separated summary and
`summarize_pusht_contact_predictor_ablation.py` for the explicitly labeled
epoch, anchor-weight, and training-case ablations. The repair is a targeted
diagnostic for one published checkpoint; it does not establish original
frozen-checkpoint causality, universal physical counterfactual accuracy,
online planning gain, or task success.
