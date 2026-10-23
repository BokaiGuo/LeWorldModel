#!/usr/bin/env bash
# Fresh host-GPU-only V8-r2 control: action-conditioned versus shared-future
# planning.  The original V8 root interval became capacity-infeasible after
# later immutable contexts were registered; r2 uses a new root range.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"; WRAPPER="$ROOT/experiments/run_with_torch_cudnn.sh"
BASE="$ROOT/data/point_maze_v44_opportunity_stability_s5_extension_20261022"
DATASET="$ROOT/data/point_maze_v45_shared_future_v8r2_extension_20261022"
PREFLIGHT="$ROOT/results/pointmaze_shared_future_v8r2_static_preflight_20261022.json"
PANEL="$ROOT/results/pointmaze_shared_future_v8r2_panel_20261022_cuda"
OUTPUT="$ROOT/results/pointmaze_shared_future_v8r2_control_20261022_cuda"
ENCODER="$ROOT/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="$ROOT/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="$ROOT/external/local_dinov2"; ATTEMPTS="${GPU_PREFLIGHT_ATTEMPTS:-3}"
if [[ -e "$PREFLIGHT" || -e "$PANEL" || -e "$OUTPUT" ]]; then echo "refusing to reuse immutable V8-r2 preflight, panel, or outcome path" >&2; exit 2; fi
for required in "$PYTHON" "$WRAPPER" "$BASE" "$ENCODER" "$PCA" "$DINOV2"; do [[ -e "$required" ]] || { echo "missing required V8 input: $required" >&2; exit 3; }; done
for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
  if bash "$WRAPPER" "$PYTHON" -m experiments.gpu_preflight --device cuda --recurrent-smoke; then break; fi
  if (( attempt == ATTEMPTS )); then echo "CUDA preflight failed after ${ATTEMPTS} attempts; refusing CPU fallback" >&2; exit 9; fi
done
if [[ ! -e "$DATASET" ]]; then "$PYTHON" -m experiments.decision_benchmark.extend_pointmaze_state_action_pool --input-root "$BASE" --output "$DATASET" --episode-count 640 --seed 202611500; fi
mapfile -t CONTEXTS < <(find results -type f -name '*context*.json' | sort | while IFS= read -r path; do if jq -e '(.metadata | type == "object") and (.metadata.candidate_family | type == "string") and (.records | type == "array") and all(.records[]; has("start_episode_id") and has("candidate_action_hash") and has("candidate_template_fingerprint"))' "$path" >/dev/null; then readlink -f "$path"; fi; done)
(( ${#CONTEXTS[@]} > 0 )) || { echo "no valid prior contexts" >&2; exit 7; }
"$PYTHON" -m experiments.gdd_wm.preflight_pointmaze_shared_future_v8 --dataset-root "$DATASET" --contexts "${CONTEXTS[@]}" --root-seed 202611501 --held-action-seed 202611502 --train-action-seed 202611503 --output "$PREFLIGHT"
bash "$WRAPPER" "$PYTHON" -m experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel --dataset-root "$DATASET" --contexts "${CONTEXTS[@]}" --output "$PANEL" --encoder-checkpoint "$ENCODER" --pca-checkpoint "$PCA" --dinov2-source "$DINOV2" --candidate-generalization-regime shared_candidate_in_support --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 --root-seed 202611501 --held-action-seed 202611502 --train-action-seed 202611503 --train-root-count 128 --calibration-root-count 192 --outcome-root-count 64 --root-id-min 17720 --root-id-max-exclusive 18360 --device cuda --frozen-preflight-ledger "$PREFLIGHT"
bash "$WRAPPER" "$PYTHON" -m experiments.gdd_wm.run_pointmaze_planning_shaped_control --source-dataset "$PANEL/temporal_ood_v6_source_dataset.json" --source-arrays "$PANEL/temporal_ood_v6_source_arrays.npz" --context "$PANEL/temporal_ood_v6_context.json" --calibration-labels "$PANEL/temporal_ood_v6_calibration_outcomes.json" --output "$OUTPUT" --device cuda --seed 202611504 --width 32 --steps 400 --ensemble-size 4 --action-representation full_ordered_20x2 --include-shared-future-control --require-v8-preflight
"$PYTHON" -m experiments.gdd_wm.evaluate_pointmaze_shared_future_v8_htsc --report "$OUTPUT/report.json" --preflight "$PREFLIGHT" --output "$OUTPUT/htsc_report.json" --alpha 0.1 --risk-quantile 0.9
