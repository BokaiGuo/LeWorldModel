#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pusht"
POLICY="pusht/lewm"
OUTPUT=""
PYTHON_BIN="${PYTHON:-}"
DEVICE=""
NUM_WORKERS=0
PROGRESS_EVERY=10
STAGE1_CHUNK_BATCHES=20
STAGE2_CHUNK_BATCHES=50
STAGE1_MAX_BATCHES=0
STAGE2_MAX_BATCHES=0
SPECTRAL_MAX_SAMPLES=200000
SAVE_MAX_SAMPLES=200000
ADAPTER_TARGET="nuisance"
ADAPTER_MAX_SAMPLES=100000
SUMMARY_OUTPUT=""
KEEP_CHUNKS=0
RESUME=0
DRY_RUN=0
RUN_THEORY=1
RUN_THEORY_RICH=1
RUN_THEORY_DIM_SWEEP=1
RUN_THEORY_ROBUSTNESS=1
RUN_THEORY_MECHANISM=1
RUN_EMPIRICAL=1
RUN_EMPIRICAL_CONDITION=1
RUN_ADAPTER=1
RUN_EVAL=0
RUN_DIRECTIONAL_TRAIN=0
RUN_SUMMARY=1
RUN_EVIDENCE_SUMMARY=1
SKIP_COMPLETED=0
RICH_FIELDS=0
RICH_HORIZONS="1 3 5"
EMPIRICAL_CONDITION="states"
EMPIRICAL_BAND_MODE="energy"
EMPIRICAL_HORIZONS="1 3 5"
EMPIRICAL_KS="4 8 16 32 64"
EMPIRICAL_CLUSTERS=32
EMPIRICAL_MIN_CELL_SIZE=256
EMPIRICAL_MAX_SAMPLES=200000
EMPIRICAL_CONDITION_SWEEP="states actions latents global"
EMPIRICAL_CONDITION_HORIZONS="1 3 5"
EMPIRICAL_CONDITION_MAX_SAMPLES=100000
EVAL_NUM=50
EVAL_SPECTRAL=0
EVAL_SPECTRAL_TOP_K=0
DIRECTIONAL_OUTPUT_MODEL="lewm_directional"
DIRECTIONAL_DYN_WEIGHT=0.01
DIRECTIONAL_CONTROLLABILITY_WEIGHT=0.0
DIRECTIONAL_REACH_WEIGHT=0.01
DIRECTIONAL_NUISANCE_WEIGHT=0.0
DIRECTIONAL_SPECTRAL_PRED_WEIGHT=0.0
DIRECTIONAL_TEACHER_ANCHOR_WEIGHT=0.0
DIRECTIONAL_TEACHER_POLICY=""
DIRECTIONAL_MAX_EPOCHS=100

usage() {
  cat <<'EOF'
Usage:
  bash experiments/run_full_spectral_transfer.sh [options]

Runs the full spectral-transfer diagnostic suite in order:
  0. theory_spectral_transfer/run_analysis.py
  0b. theory_spectral_transfer/run_rich_sweep.py
  0c. theory_spectral_transfer/summarize_dimension_sweep.py
  0d. theory_spectral_transfer/run_robustness_sweep.py
  0e. theory_spectral_transfer/run_mechanism_sweep.py
  1. latent_eigenspace_audit.py
  1b. empirical_spectral_audit/run_audit.py over Phase 1 latents.npz
  1c. empirical_spectral_audit condition-robustness sweep
  2. planning_alignment_audit.py with --spectral-basis from Phase 1
  3. intervention_adapter_audit.py
  4. summarize_directional_results.py
  4b. summarize_spectral_transfer_evidence.py

Defaults are full PushT, fresh timestamped output, and no max-batch cap.
To continue under existing data, use:
  bash experiments/run_full_spectral_transfer.sh --resume --output results/pusht/lewm

Options:
  --env NAME                    Environment preset: pusht, cube, reacher, tworoom. Default: pusht
  --policy NAME_OR_PATH          Policy name/checkpoint. Default: pusht/lewm
  --output DIR                   Run output directory. Default: results/<env>/lewm_full_<timestamp>
  --summary-output PATH          Summary markdown path. Default: <output>/summary.md
  --python PATH                  Python executable. Default: .venv/bin/python if present, else python
  --device DEVICE                Optional device passed to audit scripts, e.g. cuda or cpu
  --num-workers N                DataLoader workers. Default: 0
  --progress-every N             Progress log interval. Default: 10
  --stage1-chunk-batches N       Phase 1 chunk size in batches. Default: 20
  --stage2-chunk-batches N       Phase 2 chunk size in batches. Default: 50
  --stage1-max-batches N         Phase 1 cap; 0 means full dataset. Default: 0
  --stage2-max-batches N         Phase 2 cap; 0 means full dataset. Default: 0
  --spectral-max-samples N       Samples for spectral planner basis. Default: 200000
  --save-max-samples N           Rows saved to Phase 1 latents.npz; 0 saves all. Default: 200000
  --adapter-target NAME          Target tangent for adapter intervention. Default: nuisance
  --adapter-max-samples N        Samples for adapter audit. Default: 100000
  --rich-fields                  Save optional goals/success/episode/phase/physical/counterfactual arrays in Phase 1
  --rich-horizons "LIST"         Space-separated multi-step horizons exported with --rich-fields. Default: "1 3 5"
  --empirical-condition NAME     Empirical condition: states/actions/latents/goals/phase/global. Default: states
  --empirical-band-mode NAME     Empirical bands: rank or energy. Default: energy
  --empirical-horizons "LIST"    Space-separated horizons for empirical audit. Default: "1"
  --empirical-ks "LIST"          Space-separated top-k list. Default: "4 8 16 32 64"
  --empirical-clusters N         Empirical condition clusters. Default: 32
  --empirical-min-cell-size N    Empirical minimum cell size. Default: 256
  --empirical-max-samples N      Deterministic row cap for empirical audit. Default: 200000
  --empirical-condition-sweep "LIST"
                                 Conditions for empirical robustness. Default: "states actions latents global"
  --empirical-condition-horizons "LIST"
                                 Horizons for empirical robustness. Default: "1 3 5"
  --empirical-condition-max-samples N
                                 Row cap per condition audit. Default: 100000
  --run-eval                     Run online eval.py after diagnostics
  --eval-num N                   Online eval episodes. Default: 50
  --eval-spectral                Run eval with online spectral planner cost
  --eval-spectral-top-k N        Restrict spectral planner to top-k ranks; 0 uses all. Default: 0
  --run-directional-train        Run train_lewm_directional.py as a parameter-level intervention stage
  --directional-output-model N   Output model name for directional training. Default: lewm_directional
  --directional-dyn-weight X     Directional dynamics weight. Default: 0.01
  --directional-controllability-weight X Action-controllability alignment weight. Default: 0.0
  --directional-reach-weight X   Directional reach weight. Default: 0.01
  --directional-nuisance-weight X Directional nuisance weight. Default: 0.0
  --directional-spectral-pred-weight X Directional spectral prediction weight. Default: 0.0
  --directional-teacher-anchor-weight X Teacher anchoring weight. Default: 0.0
  --directional-teacher-policy POLICY   Teacher policy/checkpoint for parameter-level anchored fine-tuning
  --directional-max-epochs N     Trainer max epochs for directional training. Default: 100
  --resume                       Continue after existing chunks in the output directory
  --skip-completed               In resume mode, skip stages whose final artifact already exists
  --keep-chunks                  Retain recoverable chunks after each phase succeeds
  --no-keep-chunks               Remove chunks after each phase succeeds (default)
  --no-theory                    Skip theory_spectral_transfer stage
  --no-theory-rich               Skip the richer paper-style theory sweep stage
  --no-theory-dim-sweep          Skip the cross-dimension theory sweep stage
  --no-theory-robustness         Skip the seed/noise robustness theory sweep stage
  --no-theory-mechanism          Skip the joint theory mechanism sweep stage
  --no-empirical                 Skip empirical audit and condition-robustness stages
  --no-empirical-condition       Skip the empirical condition-robustness stage
  --no-adapter                   Skip feature-space adapter stage
  --no-eval                      Skip online eval stage
  --no-directional-train         Skip directional training stage
  --no-summary                   Skip markdown summary stage
  --no-evidence-summary          Skip theory-to-empirical evidence summary stage
  --dry-run                      Print commands without executing
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --summary-output) SUMMARY_OUTPUT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
    --stage1-chunk-batches) STAGE1_CHUNK_BATCHES="$2"; shift 2 ;;
    --stage2-chunk-batches) STAGE2_CHUNK_BATCHES="$2"; shift 2 ;;
    --stage1-max-batches) STAGE1_MAX_BATCHES="$2"; shift 2 ;;
    --stage2-max-batches) STAGE2_MAX_BATCHES="$2"; shift 2 ;;
    --spectral-max-samples) SPECTRAL_MAX_SAMPLES="$2"; shift 2 ;;
    --save-max-samples) SAVE_MAX_SAMPLES="$2"; shift 2 ;;
    --adapter-target) ADAPTER_TARGET="$2"; shift 2 ;;
    --adapter-max-samples) ADAPTER_MAX_SAMPLES="$2"; shift 2 ;;
    --rich-fields) RICH_FIELDS=1; shift ;;
    --rich-horizons) RICH_HORIZONS="$2"; shift 2 ;;
    --empirical-condition) EMPIRICAL_CONDITION="$2"; shift 2 ;;
    --empirical-band-mode) EMPIRICAL_BAND_MODE="$2"; shift 2 ;;
    --empirical-horizons) EMPIRICAL_HORIZONS="$2"; shift 2 ;;
    --empirical-ks) EMPIRICAL_KS="$2"; shift 2 ;;
    --empirical-clusters) EMPIRICAL_CLUSTERS="$2"; shift 2 ;;
    --empirical-min-cell-size) EMPIRICAL_MIN_CELL_SIZE="$2"; shift 2 ;;
    --empirical-max-samples) EMPIRICAL_MAX_SAMPLES="$2"; shift 2 ;;
    --empirical-condition-sweep) EMPIRICAL_CONDITION_SWEEP="$2"; shift 2 ;;
    --empirical-condition-horizons) EMPIRICAL_CONDITION_HORIZONS="$2"; shift 2 ;;
    --empirical-condition-max-samples) EMPIRICAL_CONDITION_MAX_SAMPLES="$2"; shift 2 ;;
    --run-eval) RUN_EVAL=1; shift ;;
    --eval-num) EVAL_NUM="$2"; shift 2 ;;
    --eval-spectral) EVAL_SPECTRAL=1; RUN_EVAL=1; shift ;;
    --eval-spectral-top-k) EVAL_SPECTRAL_TOP_K="$2"; shift 2 ;;
    --run-directional-train) RUN_DIRECTIONAL_TRAIN=1; shift ;;
    --directional-output-model) DIRECTIONAL_OUTPUT_MODEL="$2"; shift 2 ;;
    --directional-dyn-weight) DIRECTIONAL_DYN_WEIGHT="$2"; shift 2 ;;
    --directional-controllability-weight) DIRECTIONAL_CONTROLLABILITY_WEIGHT="$2"; shift 2 ;;
    --directional-reach-weight) DIRECTIONAL_REACH_WEIGHT="$2"; shift 2 ;;
    --directional-nuisance-weight) DIRECTIONAL_NUISANCE_WEIGHT="$2"; shift 2 ;;
    --directional-spectral-pred-weight) DIRECTIONAL_SPECTRAL_PRED_WEIGHT="$2"; shift 2 ;;
    --directional-teacher-anchor-weight) DIRECTIONAL_TEACHER_ANCHOR_WEIGHT="$2"; shift 2 ;;
    --directional-teacher-policy) DIRECTIONAL_TEACHER_POLICY="$2"; shift 2 ;;
    --directional-max-epochs) DIRECTIONAL_MAX_EPOCHS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --skip-completed) SKIP_COMPLETED=1; shift ;;
    --keep-chunks) KEEP_CHUNKS=1; shift ;;
    --no-keep-chunks) KEEP_CHUNKS=0; shift ;;
    --no-theory) RUN_THEORY=0; shift ;;
    --no-theory-rich) RUN_THEORY_RICH=0; shift ;;
    --no-theory-dim-sweep) RUN_THEORY_DIM_SWEEP=0; shift ;;
    --no-theory-robustness) RUN_THEORY_ROBUSTNESS=0; shift ;;
    --no-theory-mechanism) RUN_THEORY_MECHANISM=0; shift ;;
    --no-empirical) RUN_EMPIRICAL=0; RUN_EMPIRICAL_CONDITION=0; shift ;;
    --no-empirical-condition) RUN_EMPIRICAL_CONDITION=0; shift ;;
    --no-adapter) RUN_ADAPTER=0; shift ;;
    --no-eval) RUN_EVAL=0; shift ;;
    --no-directional-train) RUN_DIRECTIONAL_TRAIN=0; shift ;;
    --no-summary) RUN_SUMMARY=0; shift ;;
    --no-evidence-summary) RUN_EVIDENCE_SUMMARY=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -z "$OUTPUT" ]]; then
  if [[ "$RESUME" -eq 1 ]]; then
    OUTPUT="results/${ENV_NAME}/lewm"
  else
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    OUTPUT="results/${ENV_NAME}/lewm_full_${RUN_ID}"
  fi
fi

if [[ -z "$SUMMARY_OUTPUT" ]]; then
  SUMMARY_OUTPUT="${OUTPUT}/summary.md"
fi

mkdir -p "$OUTPUT"

COMMON_ARGS=(--env "$ENV_NAME" --policy "$POLICY" --output "$OUTPUT" --num-workers "$NUM_WORKERS" --progress-every "$PROGRESS_EVERY")
DEVICE_ARGS=()
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi
KEEP_ARGS=()
if [[ "$KEEP_CHUNKS" -eq 1 ]]; then
  KEEP_ARGS=(--keep-chunks)
fi
read -r -a RICH_HORIZON_VALUES <<< "$RICH_HORIZONS"
RICH_ARGS=()
if [[ "$RICH_FIELDS" -eq 1 ]]; then
  RICH_ARGS=(--rich-fields --rich-horizons "${RICH_HORIZON_VALUES[@]}")
fi
RESUME_ARGS=()
if [[ "$RESUME" -eq 1 ]]; then
  RESUME_ARGS=(--resume)
fi
STAGE1_MAX_ARGS=()
if [[ "$STAGE1_MAX_BATCHES" -gt 0 ]]; then
  STAGE1_MAX_ARGS=(--max-batches "$STAGE1_MAX_BATCHES")
fi
STAGE2_MAX_ARGS=()
if [[ "$STAGE2_MAX_BATCHES" -gt 0 ]]; then
  STAGE2_MAX_ARGS=(--max-batches "$STAGE2_MAX_BATCHES")
fi

print_command() {
  printf '[cmd]'
  printf ' %q' "$@"
  printf '\n'
}

run_logged() {
  local log_path="$1"
  shift
  print_command "$@"
  printf '[log] %s\n' "$log_path"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ "$RESUME" -eq 1 ]]; then
      "$@" 2>&1 | tee -a "$log_path"
    else
      "$@" 2>&1 | tee "$log_path"
    fi
  fi
}

run_logged_if_needed() {
  local artifact_path="$1"
  local log_path="$2"
  shift 2
  if [[ "$SKIP_COMPLETED" -eq 1 && -s "$artifact_path" ]]; then
    printf '[skip] artifact exists: %s\n' "$artifact_path"
    return
  fi
  run_logged "$log_path" "$@"
}

guard_no_active_writer() {
  if [[ "$RESUME" -eq 0 || "$DRY_RUN" -eq 1 ]]; then
    return
  fi
  local active
  active="$(ps -ef | grep -F 'experiments/latent_eigenspace_audit.py' | grep -F -- "$OUTPUT" | grep -v grep || true)"
  if [[ -n "$active" ]]; then
    printf '[error] active latent_eigenspace_audit writer already targets output=%s\n' "$OUTPUT" >&2
    printf '%s\n' "$active" >&2
    printf '[hint] wait for that run to finish, or use a different --output directory.\n' >&2
    exit 3
  fi
  active="$(ps -ef | grep -F 'experiments/planning_alignment_audit.py' | grep -F -- "$OUTPUT" | grep -v grep || true)"
  if [[ -n "$active" ]]; then
    printf '[error] active planning_alignment_audit writer already targets output=%s\n' "$OUTPUT" >&2
    printf '%s\n' "$active" >&2
    printf '[hint] wait for that run to finish, or use a different --output directory.\n' >&2
    exit 3
  fi
}

read -r -a EMPIRICAL_HORIZON_VALUES <<< "$EMPIRICAL_HORIZONS"
read -r -a EMPIRICAL_K_VALUES <<< "$EMPIRICAL_KS"
read -r -a EMPIRICAL_CONDITION_VALUES <<< "$EMPIRICAL_CONDITION_SWEEP"
read -r -a EMPIRICAL_CONDITION_HORIZON_VALUES <<< "$EMPIRICAL_CONDITION_HORIZONS"

printf '[full-run] env=%s policy=%s output=%s resume=%s\n' "$ENV_NAME" "$POLICY" "$OUTPUT" "$RESUME"

guard_no_active_writer

if [[ "$RUN_THEORY" -eq 1 ]]; then
  run_logged_if_needed "$OUTPUT/theory_spectral_transfer.json" "$OUTPUT/theory.log" \
    "$PYTHON_BIN" experiments/theory_spectral_transfer/run_analysis.py \
    --output "$OUTPUT/theory_spectral_transfer.json"
fi

if [[ "$RUN_THEORY_RICH" -eq 1 ]]; then
  run_logged_if_needed "$OUTPUT/theory_rich_sweep/paper_summary.pdf" "$OUTPUT/theory_rich_sweep.log" \
    "$PYTHON_BIN" experiments/theory_spectral_transfer/run_rich_sweep.py \
    --output "$OUTPUT/theory_rich_sweep"
fi

if [[ "$RUN_THEORY_DIM_SWEEP" -eq 1 ]]; then
  for dim in 64 128 256; do
    run_logged_if_needed "$OUTPUT/theory_dim_sweep/dim${dim}/theory_rich_sweep.json" "$OUTPUT/theory_dim_sweep/dim${dim}.log" \
      "$PYTHON_BIN" experiments/theory_spectral_transfer/run_rich_sweep.py \
      --output "$OUTPUT/theory_dim_sweep/dim${dim}" \
      --dim "$dim"
  done
  run_logged_if_needed "$OUTPUT/theory_dim_sweep/cross_dim_summary.pdf" "$OUTPUT/theory_dim_sweep_summary.log" \
    "$PYTHON_BIN" experiments/theory_spectral_transfer/summarize_dimension_sweep.py \
    --input "$OUTPUT/theory_dim_sweep" \
    --output "$OUTPUT/theory_dim_sweep"
fi

if [[ "$RUN_THEORY_ROBUSTNESS" -eq 1 ]]; then
  run_logged_if_needed "$OUTPUT/theory_robustness_sweep/robustness_summary.pdf" "$OUTPUT/theory_robustness_sweep.log" \
    "$PYTHON_BIN" experiments/theory_spectral_transfer/run_robustness_sweep.py \
    --output "$OUTPUT/theory_robustness_sweep"
fi

if [[ "$RUN_THEORY_MECHANISM" -eq 1 ]]; then
  run_logged_if_needed "$OUTPUT/theory_mechanism_sweep/mechanism_summary.pdf" "$OUTPUT/theory_mechanism_sweep.log" \
    "$PYTHON_BIN" experiments/theory_spectral_transfer/run_mechanism_sweep.py \
    --output "$OUTPUT/theory_mechanism_sweep"
fi

run_logged_if_needed "$OUTPUT/latents.npz" "$OUTPUT/stage1.log" \
  "$PYTHON_BIN" experiments/latent_eigenspace_audit.py \
  "${COMMON_ARGS[@]}" \
  "${DEVICE_ARGS[@]}" \
  "${KEEP_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${STAGE1_MAX_ARGS[@]}" \
  --chunk-batches "$STAGE1_CHUNK_BATCHES" \
  --save-max-samples "$SAVE_MAX_SAMPLES" \
  "${RICH_ARGS[@]}"

if [[ "$RUN_EMPIRICAL" -eq 1 ]]; then
  run_logged_if_needed "$OUTPUT/empirical_spectral_audit.json" "$OUTPUT/empirical.log" \
    "$PYTHON_BIN" experiments/empirical_spectral_audit/run_audit.py \
    --latents "$OUTPUT/latents.npz" \
    --output "$OUTPUT/empirical_spectral_audit.json" \
    --condition "$EMPIRICAL_CONDITION" \
    --band-mode "$EMPIRICAL_BAND_MODE" \
    --clusters "$EMPIRICAL_CLUSTERS" \
    --min-cell-size "$EMPIRICAL_MIN_CELL_SIZE" \
    --max-samples "$EMPIRICAL_MAX_SAMPLES" \
    --horizons "${EMPIRICAL_HORIZON_VALUES[@]}" \
    --ks "${EMPIRICAL_K_VALUES[@]}"
fi

if [[ "$RUN_EMPIRICAL_CONDITION" -eq 1 ]]; then
  CONDITION_OUTPUT="$OUTPUT/empirical_condition_sweep"
  mkdir -p "$CONDITION_OUTPUT"
  for condition in "${EMPIRICAL_CONDITION_VALUES[@]}"; do
    run_logged_if_needed "$CONDITION_OUTPUT/${condition}.json" "$OUTPUT/empirical_condition_${condition}.log" \
      "$PYTHON_BIN" experiments/empirical_spectral_audit/run_audit.py \
      --latents "$OUTPUT/latents.npz" \
      --output "$CONDITION_OUTPUT/${condition}.json" \
      --condition "$condition" \
      --band-mode "$EMPIRICAL_BAND_MODE" \
      --clusters "$EMPIRICAL_CLUSTERS" \
      --min-cell-size "$EMPIRICAL_MIN_CELL_SIZE" \
      --max-samples "$EMPIRICAL_CONDITION_MAX_SAMPLES" \
      --seed 3072 \
      --horizons "${EMPIRICAL_CONDITION_HORIZON_VALUES[@]}" \
      --ks "${EMPIRICAL_K_VALUES[@]}"
  done
  run_logged_if_needed "$CONDITION_OUTPUT/condition_summary.pdf" "$OUTPUT/empirical_condition_summary.log" \
    "$PYTHON_BIN" experiments/empirical_spectral_audit/summarize_condition_sweep.py \
    --input "$CONDITION_OUTPUT" \
    --reference "$OUTPUT/empirical_spectral_audit.json" \
    --k "${EMPIRICAL_K_VALUES[3]:-32}"
fi

run_logged_if_needed "$OUTPUT/planning_alignment.json" "$OUTPUT/stage2.log" \
  "$PYTHON_BIN" experiments/planning_alignment_audit.py \
  "${COMMON_ARGS[@]}" \
  "${DEVICE_ARGS[@]}" \
  "${KEEP_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${STAGE2_MAX_ARGS[@]}" \
  --chunk-batches "$STAGE2_CHUNK_BATCHES" \
  --spectral-basis "$OUTPUT/latents.npz" \
  --spectral-max-samples "$SPECTRAL_MAX_SAMPLES"

if [[ "$RUN_ADAPTER" -eq 1 ]]; then
  run_logged_if_needed "$OUTPUT/intervention_adapter.json" "$OUTPUT/adapter.log" \
    "$PYTHON_BIN" experiments/intervention_adapter_audit.py \
    --latents "$OUTPUT/latents.npz" \
    --output "$OUTPUT/intervention_adapter.json" \
    --target "$ADAPTER_TARGET" \
    --non-targets dynamics planning predicted_dynamics prediction_residual \
    --max-samples "$ADAPTER_MAX_SAMPLES"
fi

if [[ "$RUN_EVAL" -eq 1 ]]; then
  EVAL_EXTRA=()
  if [[ "$EVAL_SPECTRAL" -eq 1 ]]; then
    EVAL_EXTRA=(
      "eval.spectral_planner.enabled=true"
      "eval.spectral_planner.latents=$OUTPUT/latents.npz"
      "eval.spectral_planner.max_samples=$SPECTRAL_MAX_SAMPLES"
      "eval.spectral_planner.top_k=$EVAL_SPECTRAL_TOP_K"
    )
  fi
  run_logged_if_needed "$OUTPUT/eval_metrics.json" "$OUTPUT/eval.log" \
    "$PYTHON_BIN" eval.py \
    "--config-name=${ENV_NAME}.yaml" \
    "policy=$POLICY" \
    "eval.num_eval=$EVAL_NUM" \
    "output.dir=$OUTPUT" \
    "output.filename=$(basename "$OUTPUT")_results.txt" \
    "output.json_filename=eval_metrics.json" \
    "${EVAL_EXTRA[@]}"
fi

if [[ "$RUN_DIRECTIONAL_TRAIN" -eq 1 ]]; then
  DIRECTIONAL_TEACHER_ARGS=()
  if [[ -n "$DIRECTIONAL_TEACHER_POLICY" ]]; then
    DIRECTIONAL_TEACHER_ARGS=(
      "loss.directional.teacher_policy=$DIRECTIONAL_TEACHER_POLICY"
      "loss.directional.teacher_anchor_weight=$DIRECTIONAL_TEACHER_ANCHOR_WEIGHT"
    )
  fi
  if [[ "$SKIP_COMPLETED" -eq 1 && -s "$OUTPUT/directional_train_complete.txt" ]]; then
    printf '[skip] artifact exists: %s\n' "$OUTPUT/directional_train_complete.txt"
  else
    run_logged "$OUTPUT/directional_train.log" \
      "$PYTHON_BIN" experiments/train_lewm_directional.py \
      "data=$ENV_NAME" \
      "output_model_name=$DIRECTIONAL_OUTPUT_MODEL" \
      "trainer.max_epochs=$DIRECTIONAL_MAX_EPOCHS" \
      "loss.directional.dyn_weight=$DIRECTIONAL_DYN_WEIGHT" \
      "loss.directional.controllability_weight=$DIRECTIONAL_CONTROLLABILITY_WEIGHT" \
      "loss.directional.reach_weight=$DIRECTIONAL_REACH_WEIGHT" \
      "loss.directional.nuisance_weight=$DIRECTIONAL_NUISANCE_WEIGHT" \
      "loss.directional.spectral_pred_weight=$DIRECTIONAL_SPECTRAL_PRED_WEIGHT" \
      "${DIRECTIONAL_TEACHER_ARGS[@]}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      date -Is > "$OUTPUT/directional_train_complete.txt"
    fi
  fi
fi

if [[ "$RUN_SUMMARY" -eq 1 ]]; then
  run_logged "$OUTPUT/summary.log" \
    "$PYTHON_BIN" experiments/summarize_directional_results.py \
    "$OUTPUT" \
    --output "$SUMMARY_OUTPUT"
fi

if [[ "$RUN_EVIDENCE_SUMMARY" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 || (
    -s "$OUTPUT/theory_mechanism_sweep/theory_mechanism_sweep.json" &&
    -s "$OUTPUT/theory_dim_sweep/cross_dim_summary.json" &&
    -s "$OUTPUT/theory_robustness_sweep/robustness_summary.json" &&
    -s "$OUTPUT/empirical_condition_sweep/condition_summary.json"
  ) ]]; then
    run_logged_if_needed "$OUTPUT/evidence_summary.pdf" "$OUTPUT/evidence_summary.log" \
      "$PYTHON_BIN" experiments/summarize_spectral_transfer_evidence.py \
      --theory-mechanism "$OUTPUT/theory_mechanism_sweep/theory_mechanism_sweep.json" \
      --theory-dim "$OUTPUT/theory_dim_sweep/cross_dim_summary.json" \
      --theory-robustness "$OUTPUT/theory_robustness_sweep/robustness_summary.json" \
      --empirical-condition "$OUTPUT/empirical_condition_sweep/condition_summary.json" \
      --output "$OUTPUT"
  else
    printf '[skip] evidence summary requires theory sweep and empirical condition artifacts under %s\n' "$OUTPUT"
  fi
fi

printf '[done] output=%s\n' "$OUTPUT"
printf '[done] summary=%s\n' "$SUMMARY_OUTPUT"
