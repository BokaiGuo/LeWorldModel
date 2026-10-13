#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
OUT="$ROOT/results/theory_operator_alignment_iclr_oral_20260829/hardened_ablation_20260901"
WALL_DATA="$ROOT/data/dino_wm_osf/extracted/wall_single"
WALL_JEPA="$ROOT/data/jepa_wm_checkpoints/jepa_wm_wall.pth.tar"
WALL_DINO="$ROOT/data/dino_wm_osf/extracted/outputs/wall_single/checkpoints/model_latest.pth"
POINT_DATA="$ROOT/data/dino_wm_osf/extracted/point_maze"
POINT_DINO="$ROOT/data/dino_wm_osf/extracted/outputs/point_maze/checkpoints/model_latest.pth"
PUSHT_DATA="$ROOT/data/jepa_wm_osf/extracted/pusht_noise"
PUSHT_DINO="$ROOT/data/dino_wm_osf/extracted/outputs/pusht/checkpoints/model_latest.pth"

mkdir -p "$OUT/metrics" "$OUT/logs"

run_if_missing() {
  local output="$1"
  local log="$2"
  shift 2
  if [[ -s "$output" ]]; then
    printf 'skip completed: %s\n' "$output"
    return
  fi
  printf 'running: %s\n' "$output"
  "$@" >"$log" 2>&1
}

for split_seed in 0 1; do
  run_if_missing "$OUT/metrics/jepa_wall_observed_split${split_seed}.json" "$OUT/logs/jepa_wall_observed_split${split_seed}.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_jepa_wm_wall_temporal_eval.py \
    --dataset-root "$WALL_DATA" --checkpoint "$WALL_JEPA" \
    --output "$OUT/metrics/jepa_wall_observed_split${split_seed}.json" \
    --episodes 192 --split-seed "$split_seed" --save-vectors
  run_if_missing "$OUT/metrics/dino_wall_observed_split${split_seed}.json" "$OUT/logs/dino_wall_observed_split${split_seed}.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_dino_wm_wall_temporal_eval.py \
    --dataset-root "$WALL_DATA" --checkpoint "$WALL_DINO" \
    --output "$OUT/metrics/dino_wall_observed_split${split_seed}.json" \
    --episodes 192 --split-seed "$split_seed" --save-vectors
done

for action_seed in 0 1 2; do
  run_if_missing "$OUT/metrics/jepa_wall_episode_shuffle_split42_seed${action_seed}.json" "$OUT/logs/jepa_wall_episode_shuffle_split42_seed${action_seed}.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_jepa_wm_wall_temporal_eval.py \
    --dataset-root "$WALL_DATA" --checkpoint "$WALL_JEPA" \
    --output "$OUT/metrics/jepa_wall_episode_shuffle_split42_seed${action_seed}.json" \
    --episodes 192 --split-seed 42 --action-mode episode_shuffle \
    --action-permutation-seed "$action_seed" --save-vectors
  run_if_missing "$OUT/metrics/dino_wall_episode_shuffle_split42_seed${action_seed}.json" "$OUT/logs/dino_wall_episode_shuffle_split42_seed${action_seed}.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_dino_wm_wall_temporal_eval.py \
    --dataset-root "$WALL_DATA" --checkpoint "$WALL_DINO" \
    --output "$OUT/metrics/dino_wall_episode_shuffle_split42_seed${action_seed}.json" \
    --episodes 192 --split-seed 42 --action-mode episode_shuffle \
    --action-permutation-seed "$action_seed" --save-vectors
done

for split_seed in 0 1; do
  run_if_missing "$OUT/metrics/jepa_wall_episode_shuffle_split${split_seed}_seed0.json" "$OUT/logs/jepa_wall_episode_shuffle_split${split_seed}_seed0.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_jepa_wm_wall_temporal_eval.py \
    --dataset-root "$WALL_DATA" --checkpoint "$WALL_JEPA" \
    --output "$OUT/metrics/jepa_wall_episode_shuffle_split${split_seed}_seed0.json" \
    --episodes 192 --split-seed "$split_seed" --action-mode episode_shuffle \
    --action-permutation-seed 0 --save-vectors
  run_if_missing "$OUT/metrics/dino_wall_episode_shuffle_split${split_seed}_seed0.json" "$OUT/logs/dino_wall_episode_shuffle_split${split_seed}_seed0.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_dino_wm_wall_temporal_eval.py \
    --dataset-root "$WALL_DATA" --checkpoint "$WALL_DINO" \
    --output "$OUT/metrics/dino_wall_episode_shuffle_split${split_seed}_seed0.json" \
    --episodes 192 --split-seed "$split_seed" --action-mode episode_shuffle \
    --action-permutation-seed 0 --save-vectors
done

for mode in future_zero future_reverse; do
  run_if_missing "$OUT/metrics/dino_pusht_${mode}.json" "$OUT/logs/dino_pusht_${mode}.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_dino_wm_pusht_temporal_eval.py \
    --dataset-root "$PUSHT_DATA" --checkpoint "$PUSHT_DINO" \
    --output "$OUT/metrics/dino_pusht_${mode}.json" --action-mode "$mode" --save-vectors
  run_if_missing "$OUT/metrics/dino_pointmaze_${mode}.json" "$OUT/logs/dino_pointmaze_${mode}.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_dino_wm_pointmaze_temporal_eval.py \
    --dataset-root "$POINT_DATA" --checkpoint "$POINT_DINO" \
    --output "$OUT/metrics/dino_pointmaze_${mode}.json" \
    --split-mode code_reconstructed --split-seed 42 --split-ratio 0.9 \
    --action-mode "$mode" --save-vectors
  run_if_missing "$OUT/metrics/dino_wall_${mode}.json" "$OUT/logs/dino_wall_${mode}.log" \
    "$PYTHON" experiments/theory_spectral_transfer/run_dino_wm_wall_temporal_eval.py \
    --dataset-root "$WALL_DATA" --checkpoint "$WALL_DINO" \
    --output "$OUT/metrics/dino_wall_${mode}.json" \
    --episodes 192 --split-seed 42 --action-mode "$mode" --save-vectors
done

for split_seed in 0 1; do
  for mode in future_zero future_reverse; do
    run_if_missing "$OUT/metrics/dino_wall_${mode}_split${split_seed}.json" "$OUT/logs/dino_wall_${mode}_split${split_seed}.log" \
      "$PYTHON" experiments/theory_spectral_transfer/run_dino_wm_wall_temporal_eval.py \
      --dataset-root "$WALL_DATA" --checkpoint "$WALL_DINO" \
      --output "$OUT/metrics/dino_wall_${mode}_split${split_seed}.json" \
      --episodes 192 --split-seed "$split_seed" --action-mode "$mode" --save-vectors
  done
done

printf 'all hardened replay runs completed: %s\n' "$OUT"
