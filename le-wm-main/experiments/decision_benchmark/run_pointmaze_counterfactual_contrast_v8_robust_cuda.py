"""Run the fresh V8 robust-bank counterfactual-repair confirmation on CUDA.

V7 showed a genuine learned history-action interaction but no transferable
benefit: the action bank was winner-balanced only on one design-root set and
returned to a dominant action prior on held roots.  V8 is a new experiment,
not a replay.  It uses a new, provenance-bound PointMaze extension, larger
train/calibration/outcome splits, and a fixed bank selected by worst-fold
declared-goal headroom across two disjoint candidate-design folds.

The process is deliberately a direct Python CUDA entry point.  It disables
cuDNN because the host's cuDNN RNN library is incompatible with this PyTorch
build and uses manually-unrolled ``GRUCell`` modules preflighted by
``require_cuda_recurrent_cell``.  No CPU model fallback is permitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.decision_benchmark.extend_pointmaze_state_action_pool import build_extension
from experiments.decision_benchmark.run_pointmaze_counterfactual_contrast_v7_cuda import CONTEXTS as V7_CONTEXTS
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = ROOT / "data" / "point_maze_v23_planning_shaped_advantage_extension_20261022"
DATASET = ROOT / "data" / "point_maze_v27_robust_counterfactual_extension_20261022"
OUTPUT = ROOT / "results" / "pointmaze_counterfactual_contrast_v8_robust_bank_20261022_cuda"
ROOT_RANGE = (10_808, 11_320)
SPLIT_COUNTS = {"train": 128, "calibration": 192, "outcome": 64}
EXTENSION_SEED = 202610479
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202610480, 202610481, 202610482
ROBUST_BANK_CONTRACT = {
    "minimum_winner_entropy_normalized": 0.50,
    "minimum_distinct_winner_count": 3,
    "minimum_best_second_gap_ge_0p003_fraction": 0.50,
    "minimum_candidate_cost_spread_q10": 0.05,
}

# All eligible historical V6 contexts are explicitly declared rather than
# discovered at runtime.  V7 contexts also cover the pre-V6 panels whose
# schema used a different context filename.
_EXTRA_CONTEXT_RELATIVE = (
    "results/pointmaze_counterfactual_contrast_v7_20261022_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_planwam_certified_repairability_fresh_v5_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_planwam_counterfactual_fresh_v1_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_planwam_counterfactual_fresh_v2_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_planwam_goalbalanced_counterfactual_fresh_v3_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_planwam_repairability_fresh_v4_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_temporal_ood_v6_9_protocol_repair_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v7_goal_balanced_task_contrast_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v7_goal_balanced_task_contrast_replication_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v8_visual_belief_state_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v9_fixed_direct_confirmation_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v10_expanded_calibration_confirmation_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v11_fixed_belief_confirmation_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v12_decoupled_belief_confirmation_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v13_decoupled_belief_large_confirmation_20261021_cuda/temporal_ood_v6_context.json",
    "results/pointmaze_v14_rgb_geometry_state_confirmation_20261021_cuda/temporal_ood_v6_context.json",
)
CONTEXTS = tuple(dict.fromkeys([*V7_CONTEXTS, *(ROOT / path for path in _EXTRA_CONTEXT_RELATIVE)]))


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _ensure_fresh_extension() -> dict[str, object]:
    """Create once or validate the independent roots before any panel build."""
    expected = {
        "source_root": str(BASE_DATASET.resolve()),
        "seed": EXTENSION_SEED,
        "new_episode_count": ROOT_RANGE[1] - ROOT_RANGE[0],
        "base_root_count": ROOT_RANGE[0],
        "extended_root_count": ROOT_RANGE[1],
        "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
    }
    manifest = DATASET / "extension_manifest.json"
    if DATASET.exists():
        if not manifest.is_file():
            raise RuntimeError("V8 extension directory exists without an immutable manifest")
        payload = _read_json(manifest)
        mismatched = {name: (payload.get(name), value) for name, value in expected.items() if payload.get(name) != value}
        if mismatched:
            raise RuntimeError(f"V8 extension contract mismatch: {mismatched}")
        return {"resumed_existing_extension": True, "manifest": str(manifest.resolve())}
    return build_extension(
        BASE_DATASET, DATASET,
        episode_count=ROOT_RANGE[1] - ROOT_RANGE[0], seed=EXTENSION_SEED,
    )


def _validate_robust_bank_contract(build_report: dict[str, object]) -> dict[str, object]:
    """Reject a new panel before fitting when its bank lacks decision headroom."""
    design = build_report.get("interaction_pair_design")
    if not isinstance(design, dict) or design.get("candidate_design") != "robust_goal_balanced_interaction_pairs":
        raise ValueError("V8 build lacks the declared robust candidate-bank design")
    folds = design.get("design_fold_headroom")
    if not isinstance(folds, list) or len(folds) != 2 or not all(isinstance(item, dict) for item in folds):
        raise ValueError("V8 build lacks exactly two robust design-fold reports")
    violations: list[dict[str, object]] = []
    for fold_index, metrics in enumerate(folds):
        checks = {
            "winner_entropy_normalized": float(metrics.get("winner_entropy_normalized", float("nan"))) >= ROBUST_BANK_CONTRACT["minimum_winner_entropy_normalized"],
            "distinct_winner_count": int(metrics.get("distinct_winner_count", -1)) >= ROBUST_BANK_CONTRACT["minimum_distinct_winner_count"],
            "best_second_gap_ge_0p003_fraction": float(metrics.get("best_second_gap_ge_0p003_fraction", float("nan"))) >= ROBUST_BANK_CONTRACT["minimum_best_second_gap_ge_0p003_fraction"],
            "candidate_cost_spread_q10": float(metrics.get("candidate_cost_spread_q10", float("nan"))) >= ROBUST_BANK_CONTRACT["minimum_candidate_cost_spread_q10"],
        }
        if not all(checks.values()):
            violations.append({"fold": fold_index, "checks": checks, "metrics": metrics})
    if violations:
        raise RuntimeError(f"V8 robust candidate bank fails pre-fit headroom contract: {violations}")
    return {
        "contract": ROBUST_BANK_CONTRACT,
        "folds": folds,
        "status": "passed_before_model_fitting",
    }


def main() -> int:
    source = OUTPUT / "temporal_ood_v6_source_dataset.json"
    arrays = OUTPUT / "temporal_ood_v6_source_arrays.npz"
    context = OUTPUT / "temporal_ood_v6_context.json"
    calibration = OUTPUT / "temporal_ood_v6_calibration_outcomes.json"
    manifest = OUTPUT / "temporal_ood_v6_manifest.json"
    frozen = OUTPUT / "temporal_ood_v6_frozen_selection.json"
    outcome = OUTPUT / "score" / "temporal_ood_v6_outcome_labels.json"
    expected_build = (source, arrays, context, calibration, manifest)
    if outcome.exists():
        raise FileExistsError(f"refusing to reuse completed immutable V8 output path: {OUTPUT}")
    inputs = {
        "base_dataset": BASE_DATASET,
        "encoder": ROOT / "data" / "jepa_wm_checkpoints" / "jepa_wm_pointmaze.pth.tar",
        "pca": ROOT / "results" / "pointmaze_same_backbone_full_support_20261013" / "checkpoints" / "jepa_full_mlp" / "seed_8411" / "stage_005.pt",
        "dinov2": ROOT / "external" / "local_dinov2",
    }
    missing = [str(path) for path in [*inputs.values(), *CONTEXTS] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V8 requires existing bound inputs: {missing}")

    # GPU admission precedes data generation so that this learned experiment
    # cannot silently proceed after a failed CUDA contract.  cuDNN remains
    # disabled only for its incompatible fused-RNN path; Conv/attention stay
    # on CUDA and the recurrent cell is explicitly preflighted below.
    torch.backends.cudnn.enabled = False
    runtime = require_cuda_recurrent_cell("cuda")
    extension = _ensure_fresh_extension()

    if OUTPUT.exists():
        if not all(path.is_file() for path in expected_build):
            raise RuntimeError("V8 partial output is not a build-complete resumable state")
        model_path = interaction._frozen_model_path(frozen)
        score_dir = OUTPUT / "score"
        # A persisted frozen selection is a sealed, score-resumable boundary.
        # Any earlier calibration/model write without it, or any partial score
        # directory without labels, is ambiguous and must receive a fresh path.
        if not frozen.exists() and model_path.exists():
            raise RuntimeError("V8 found an unbound calibration checkpoint without frozen selection")
        if score_dir.exists() and not frozen.exists():
            raise RuntimeError("V8 found a partial score directory without frozen selection")
        build: dict[str, object] = {"resumed_existing_sealed_build": True}
    else:
        build = builder.build_panel(
            DATASET, list(CONTEXTS), OUTPUT,
            encoder_checkpoint=inputs["encoder"], pca_checkpoint=inputs["pca"], dinov2_source=inputs["dinov2"],
            candidate_generalization_regime="shared_candidate_in_support",
            candidate_design="robust_goal_balanced_interaction_pairs", goal_qpos=[0.8, 0.8], device="cuda",
            root_seed=ROOT_SEED, held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
            split_counts=SPLIT_COUNTS, root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
        )
    build_report = _read_json(OUTPUT / "build_v6_matched_report.json")
    robust_bank = _validate_robust_bank_contract(build_report)
    if frozen.exists():
        calibration_result = {"resumed_frozen_selection": str(frozen.resolve())}
    else:
        calibration_result = interaction.calibrate(
            source, arrays, context, calibration, frozen, epochs=160, device="cuda",
            observability_suite=True, history_gate=True, manifest=manifest,
        )
    score = interaction.score(
        source, arrays, context, frozen, OUTPUT / "score", device="cuda",
        observability_suite=True, history_gate=True, manifest=manifest,
    )
    runtime_path = OUTPUT / "v8_cuda_runtime.json"
    runtime_path.write_text(json.dumps({
        "schema": "pointmaze_counterfactual_contrast_v8_runtime.v1",
        "cuda_runtime": runtime, "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "recurrent_implementation": "manually unrolled torch.nn.GRUCell",
        "extension": extension, "build_report": build, "robust_bank_preflight": robust_bank,
        "calibration_result": calibration_result,
        "frozen_selection": str(frozen.resolve()),
        "score_report": str((OUTPUT / "score" / "temporal_ood_v6_score_report.json").resolve()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "active_policy": score["active_policy"],
        "authorization": calibration_result.get("authorization"),
        "selected_contrast_weight": score["selected_contrast_weight"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
