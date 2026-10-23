"""Freeze a position-matched / velocity-aliased C1 PointMaze panel.

V13 showed that the old random-pulse family did not leave history beyond a
position, goal, and global action-effect chart.  V14 changes the controlled
mechanism rather than relaxing that gate: every root is part of a pair that
shares source position but has opposed incoming velocity.  Complete pairs are
kept within train, calibration, outcome, or one of two independent design
folds.  The static admission requires the true candidate winner to reverse
within those velocity pairs and rejects the same position-only proxy as V13.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm.preflight_pointmaze_planning_shaped_advantage_v7 import _sha256
from experiments.run_live_pointmaze_planning import DatasetMatchedRenderer, fit_visual_calibration, green_center, load_episode


SCHEMA = panel.PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA
POOL_SCHEMA = "pointmaze_velocity_matched_state_pool.v1"
MIN_PAIRWISE_Q10 = 0.003
MIN_PAIRWISE_FRACTION = 0.80
MIN_FOLD_REVERSAL = 0.75
MAX_PROXY_ACCURACY = 0.625


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _pair_groups(manifest: dict[str, Any], *, root_count: int, root_id_min: int, root_id_max_exclusive: int) -> list[tuple[int, int]]:
    if manifest.get("schema") != POOL_SCHEMA:
        raise ValueError("V14 needs the immutable velocity-matched state-pool manifest")
    if manifest.get("new_root_id_range") != [int(root_id_min), int(root_id_max_exclusive) - 1]:
        raise ValueError("V14 root range differs from the velocity-matched pool manifest")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) * 2 != int(root_id_max_exclusive) - int(root_id_min):
        raise ValueError("velocity-matched manifest does not cover the V14 root range in complete pairs")
    groups = [(int(root_id_min) + 2 * index, int(root_id_min) + 2 * index + 1) for index in range(len(pairs))]
    if any(left < 0 or right >= int(root_count) for left, right in groups):
        raise ValueError("velocity-matched group lies outside states.pth")
    return groups


def _grouped_split(groups: list[tuple[int, int]], *, seed: int, split_counts: dict[str, int]) -> tuple[dict[int, str], list[int]]:
    if any(int(count) % 2 for count in split_counts.values()):
        raise ValueError("V14 requires even root counts to keep velocity pairs intact")
    design_group_count = panel.INTERACTION_DESIGN_ROOTS // 2
    required = sum(int(count) // 2 for count in split_counts.values()) + design_group_count
    if len(groups) != required:
        raise ValueError(f"V14 exact pool contract needs {required} groups, found {len(groups)}")
    ordered = [groups[index] for index in np.random.default_rng(int(seed)).permutation(len(groups)).tolist()]
    split: dict[int, str] = {}
    offset = 0
    for name in ("train", "calibration", "outcome"):
        width = int(split_counts[name]) // 2
        for group in ordered[offset : offset + width]:
            split[group[0]] = name
            split[group[1]] = name
        offset += width
    design_roots = [root for group in ordered[offset:] for root in group]
    if len(design_roots) != panel.INTERACTION_DESIGN_ROOTS:
        raise RuntimeError("V14 did not reserve exactly two paired design folds")
    return split, design_roots


def _validate_pair_states(states: np.ndarray, groups: list[tuple[int, int]], *, tolerance: float, minimum_speed: float) -> bool:
    for positive, negative in groups:
        left, right = states[positive, panel.SOURCE_FRAME], states[negative, panel.SOURCE_FRAME]
        if (
            float(np.linalg.norm(left[:2] - right[:2])) > float(tolerance)
            or float(np.linalg.norm(left[2:])) < float(minimum_speed)
            or float(np.linalg.norm(right[2:])) < float(minimum_speed)
            or float(np.dot(left[2:], right[2:])) >= 0.0
        ):
            return False
    return True


def _renderer_history_visible(root: Path, states: np.ndarray, *, chart_roots: list[int], episodes: set[int]) -> bool:
    """Verify the exact causal renderer contract before a model is loaded."""
    calibration = fit_visual_calibration(root, chart_roots, max_frames=1600)
    images, _ = load_episode(root, chart_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(images, calibration)
    frames = panel.causal_history8(panel.SOURCE_FRAME, np.zeros((panel.SOURCE_FRAME, 2), dtype=np.float64))[0]
    try:
        for episode in sorted(episodes):
            for frame in frames:
                green_center(renderer.render(states[episode, frame]))
                green_center(renderer.render(states[episode, frame - panel.FRAME_STRIDE]))
    except ValueError:
        return False
    return True


def preflight(
    *, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int,
    train_action_seed: int, root_id_min: int, root_id_max_exclusive: int,
    split_counts: dict[str, int], schema: str = SCHEMA,
    candidate_builder: Any = panel.velocity_matched_acceleration_candidate_banks,
    candidate_design: str = "velocity_matched_acceleration_pairs",
    require_goal_centered_radial_pool: bool = False,
    require_renderer_history_visibility: bool = False,
) -> dict[str, Any]:
    """Run only the frozen C1 feasibility gate; no encoder or labels are opened."""
    root = Path(dataset_root)
    states_path, manifest_path = root / "states.pth", root / "velocity_matched_manifest.json"
    if not states_path.is_file() or not manifest_path.is_file() or not (root / "obses").is_dir():
        raise FileNotFoundError("V14 needs states.pth, obses/, and velocity_matched_manifest.json")
    manifest = _read_object(manifest_path)
    if require_goal_centered_radial_pool and (
        manifest.get("target_construction") != "goal_centered_radial_ring"
        or manifest.get("target_goal") != [0.8, 0.8]
        or manifest.get("target_radius") is None
    ):
        raise ValueError("radial velocity-alias preflight requires its declared goal-centred pool")
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    if states.ndim != 3 or states.shape[1:] != (100, 4):
        raise ValueError("V14 PointMaze states must be [roots,100,4]")
    counts = panel._validated_split_counts(split_counts)
    groups = _pair_groups(manifest, root_count=len(states), root_id_min=root_id_min, root_id_max_exclusive=root_id_max_exclusive)
    split, design_roots = _grouped_split(groups, seed=int(root_seed), split_counts=counts)
    resolved_contexts = [Path(path).resolve() for path in contexts]
    prior_roots, prior_hashes, _templates, context_sha256 = panel._all_prior_identities(resolved_contexts)
    chart_roots = panel._renderer_chart_roots_v6(root, set(split))
    history_visible = (
        _renderer_history_visible(root, states, chart_roots=chart_roots, episodes=set(split) | set(design_roots))
        if require_renderer_history_visibility else True
    )
    goal = panel.validate_goal_qpos((0.8, 0.8))
    held, train, interaction = candidate_builder(
        states[np.asarray(design_roots, dtype=np.int64), panel.SOURCE_FRAME], goal,
        proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
    )
    held_hashes = {panel.sequence_hash(action) for action in held}
    train_hashes = {panel.sequence_hash(action) for action in train}
    contracts = interaction["selected_pair_contract"]
    candidate = {
        "candidate_design": str(candidate_design),
        "goal_qpos": goal.tolist(), "held_shape": list(held.shape), "train_shape": list(train.shape),
        "shared_candidate_in_support": bool(held_hashes <= train_hashes),
        "held_action_hash_count": len(held_hashes), "train_action_hash_count": len(train_hashes),
        "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
        "scorer_candidate_bank_protocol": "held_exact_shared_bank", "scorer_training_action_hash_count": len(held_hashes),
        "pairwise_contrast_contract": "velocity_matched_state_alias_pairs_v1",
        "pairwise_absolute_cost_gap_q10": float(interaction["selected_pooled_absolute_gap_q10"]),
        "pairwise_gap_ge_0p003_fraction": float(interaction["selected_gap_ge_margin_fraction"]),
        "pairwise_sign_mixed_pair_count": int(interaction["selected_sign_mixed_pair_count"]),
        "minimum_fold_velocity_reversal_requirement": MIN_FOLD_REVERSAL,
        "minimum_fold_velocity_reversal": min(float(item["minimum_fold_velocity_reversal_fraction"]) for item in contracts),
        "maximum_crossfold_position_only_sign_accuracy_requirement": MAX_PROXY_ACCURACY,
        "maximum_crossfold_position_only_sign_accuracy": max(float(item["maximum_crossfold_position_only_sign_accuracy"]) for item in contracts),
        "paired_velocity_group_contract": "source-position-matched_opposed-velocity_complete-group-splits_v1",
        "pool_manifest_sha256": _sha256(manifest_path),
        "selected_pair_contract": contracts, "selection_order": interaction["selection_order"],
    }
    split_roots = {name: sorted(int(episode) for episode, assigned in split.items() if assigned == name) for name in counts}
    checks = {
        "contexts_present": bool(resolved_contexts),
        "velocity_pool_schema": manifest.get("schema") == POOL_SCHEMA,
        "velocity_pairs_mechanically_valid": _validate_pair_states(
            states, groups, tolerance=float(manifest["position_tolerance"]), minimum_speed=float(manifest["minimum_source_speed"]),
        ),
        "split_disjoint_from_prior_roots": not bool(set(split) & prior_roots),
        "complete_velocity_pairs_stay_within_split_or_design": all(split.get(left) == split.get(right) for left, right in groups if left in split),
        "design_disjoint_from_evaluation": not bool(set(design_roots) & set(split)),
        "design_disjoint_from_prior_roots": not bool(set(design_roots) & prior_roots),
        "design_disjoint_from_renderer_chart": not bool(set(design_roots) & set(chart_roots)),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not bool((held_hashes | train_hashes) & prior_hashes),
        "pairwise_absolute_cost_gap_q10": candidate["pairwise_absolute_cost_gap_q10"] >= MIN_PAIRWISE_Q10,
        "pairwise_decision_sensitive_fraction": candidate["pairwise_gap_ge_0p003_fraction"] >= MIN_PAIRWISE_FRACTION,
        "pairwise_state_conditional_signs": candidate["pairwise_sign_mixed_pair_count"] == panel.HELD_CANDIDATES // 2,
        "opposed_velocity_winner_reversal": candidate["minimum_fold_velocity_reversal"] >= MIN_FOLD_REVERSAL,
        "position_only_proxy_rejected": candidate["maximum_crossfold_position_only_sign_accuracy"] <= MAX_PROXY_ACCURACY,
    }
    if require_renderer_history_visibility:
        checks["causal_renderer_history_visible"] = bool(history_visible)
    return {
        "schema": str(schema), "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks, "dataset": str(root.resolve()), "dataset_states_sha256": _sha256(states_path),
        "construction": {
            "root_seed": int(root_seed), "held_action_seed": int(held_action_seed), "train_action_seed": int(train_action_seed),
            "root_id_min": int(root_id_min), "root_id_max_exclusive": int(root_id_max_exclusive),
            "velocity_matched_pool_seed": int(manifest["seed"]),
        },
        "prior_context_sha256": context_sha256, "root_counts": {name: len(roots) for name, roots in split_roots.items()},
        "split_roots": split_roots, "interaction_design_roots": design_roots, "candidate_bank": candidate,
        "claim_boundary": "Static V14 velocity-alias feasibility only; no visual encoder/model fit, calibration label, outcome label, repairability, or planning claim.",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--held-action-seed", type=int, required=True)
    parser.add_argument("--train-action-seed", type=int, required=True)
    parser.add_argument("--root-id-min", type=int, required=True)
    parser.add_argument("--root-id-max-exclusive", type=int, required=True)
    parser.add_argument("--train-root-count", type=int, required=True)
    parser.add_argument("--calibration-root-count", type=int, required=True)
    parser.add_argument("--outcome-root-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V14 static preflight {args.output}")
    result = preflight(
        dataset_root=args.dataset_root, contexts=args.contexts, root_seed=args.root_seed,
        held_action_seed=args.held_action_seed, train_action_seed=args.train_action_seed,
        root_id_min=args.root_id_min, root_id_max_exclusive=args.root_id_max_exclusive,
        split_counts={"train": args.train_root_count, "calibration": args.calibration_root_count, "outcome": args.outcome_root_count},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
