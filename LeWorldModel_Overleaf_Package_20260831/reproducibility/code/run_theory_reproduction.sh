#!/usr/bin/env bash
set -euo pipefail

# Run from the root of the original LeWorldModel checkout.
# The scripts below regenerate the compact analytical results used in the paper.
# Learned-model temporal evaluators require the external checkpoints and datasets
# documented in ../provenance/source_manifest_zh.md.

python experiments/theory_spectral_transfer/run_operator_alignment_sweep.py \
  --output results/theory_operator_alignment_sweep_20260828

python experiments/theory_spectral_transfer/run_operator_alignment_null_sweep.py \
  --output results/theory_operator_alignment_null_sweep_20260829

python experiments/theory_spectral_transfer/run_operator_alignment_decoupling.py \
  --output results/theory_operator_alignment_decoupling_20260829

python experiments/theory_spectral_transfer/run_operator_alignment_anisotropy_sweep.py \
  --output results/theory_operator_alignment_anisotropy_sweep_20260829

python experiments/theory_spectral_transfer/run_operator_alignment_metric_robustness.py \
  --output results/theory_operator_alignment_metric_robustness_20260829

echo "Analytical reproduction complete. See reproducibility/data_dictionary_中文.md for field meanings."
