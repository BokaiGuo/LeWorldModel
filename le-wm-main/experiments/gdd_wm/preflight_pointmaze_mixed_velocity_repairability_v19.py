"""Static C2 admission for a mixed visible-motion PointMaze panel.

V19 fixes a mechanistic router before model fitting: nonzero causal visual
motion receives the history repair; zero-motion roots retain action-only.
All candidate-bank selection is performed on disjoint high-motion design
roots.  Neither calibration nor outcome labels are accessed here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v14_velocity_alias as v14


SCHEMA = panel.MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA
POOL_SCHEMA = "pointmaze_mixed_velocity_repairability_pool.v1"
HIGH_RANGE = (26_376, 26_824)
DIRECT_RANGE = (26_824, 27_208)
MOTION_THRESHOLD = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _groups(bounds: tuple[int, int]) -> list[tuple[int, int]]:
    return [(root, root + 1) for root in range(bounds[0], bounds[1], 2)]


def _assign(seed: int, split_counts: dict[str, int]) -> tuple[dict[int, str], list[int], dict[str, dict[str, int]]]:
    if split_counts != {"train": 256, "calibration": 256, "outcome": 256}:
        raise ValueError("V19 fixes an exactly balanced 256/256/256 root protocol")
    high, direct = _groups(HIGH_RANGE), _groups(DIRECT_RANGE)
    if len(high) != 224 or len(direct) != 192:
        raise RuntimeError("V19 mixed-pool group counts are malformed")
    rng = np.random.default_rng(int(seed))
    high = [high[index] for index in rng.permutation(len(high)).tolist()]
    direct = [direct[index] for index in rng.permutation(len(direct)).tolist()]
    split: dict[int, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for index, name in enumerate(("train", "calibration", "outcome")):
        high_chunk, direct_chunk = high[index * 64 : (index + 1) * 64], direct[index * 64 : (index + 1) * 64]
        for group in [*high_chunk, *direct_chunk]:
            split[group[0]] = name
            split[group[1]] = name
        counts[name] = {"high_motion": len(high_chunk) * 2, "direct_zero_motion": len(direct_chunk) * 2}
    design = [root for group in high[192:] for root in group]
    if len(split) != 768 or len(design) != 64 or set(split) & set(design):
        raise RuntimeError("V19 split construction failed")
    return split, design, counts


def _regime_checks(states: np.ndarray) -> dict[str, bool]:
    high_ok = all(
        float(np.linalg.norm(states[left, panel.SOURCE_FRAME, :2] - states[right, panel.SOURCE_FRAME, :2])) <= 0.025
        and float(np.linalg.norm(states[left, panel.SOURCE_FRAME, 2:])) >= 0.08
        and float(np.linalg.norm(states[right, panel.SOURCE_FRAME, 2:])) >= 0.08
        and float(np.dot(states[left, panel.SOURCE_FRAME, 2:], states[right, panel.SOURCE_FRAME, 2:])) < 0.0
        for left, right in _groups(HIGH_RANGE)
    )
    direct_ok = all(
        float(np.linalg.norm(states[left, panel.SOURCE_FRAME, :2] - states[right, panel.SOURCE_FRAME, :2])) <= 0.025
        and float(np.linalg.norm(states[left, panel.SOURCE_FRAME, 2:])) <= 0.08
        and float(np.linalg.norm(states[right, panel.SOURCE_FRAME, 2:])) <= 0.08
        for left, right in _groups(DIRECT_RANGE)
    )
    return {"high_velocity_alias_groups_valid": high_ok, "direct_zero_motion_groups_valid": direct_ok}


def preflight(
    *, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int,
    train_action_seed: int, root_id_min: int, root_id_max_exclusive: int,
    split_counts: dict[str, int],
) -> dict[str, Any]:
    root = Path(dataset_root)
    manifest_path = root / "mixed_velocity_repairability_manifest.json"
    if not manifest_path.is_file() or not (root / "states.pth").is_file() or not (root / "obses").is_dir():
        raise FileNotFoundError("V19 requires the sealed mixed pool, states.pth, and renderer observations")
    if (int(root_id_min), int(root_id_max_exclusive)) != (HIGH_RANGE[0], DIRECT_RANGE[1]):
        raise ValueError("V19 root range must cover the complete frozen mixed pool")
    manifest = _read(manifest_path)
    if manifest.get("schema") != POOL_SCHEMA or manifest.get("route_contract") != "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only":
        raise ValueError("V19 mixed-pool manifest or route contract is invalid")
    states = np.asarray(torch.load(root / "states.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    if states.ndim != 3 or states.shape[1:] != (100, 4):
        raise ValueError("V19 states must be [roots,100,4]")
    if manifest.get("states_sha256") != _sha256(root / "states.pth"):
        raise ValueError("V19 mixed-pool states hash mismatch")
    split, design, regime_counts = _assign(int(root_seed), dict(split_counts))
    resolved = [Path(path).resolve() for path in contexts]
    prior_roots, prior_hashes, _templates, context_sha256 = panel._all_prior_identities(resolved)
    chart_roots = panel._renderer_chart_roots_v6(root, set(split))
    visible = v14._renderer_history_visible(root, states, chart_roots=chart_roots, episodes=set(split) | set(design))
    goal = panel.validate_goal_qpos((0.8, 0.8))
    held, train, interaction = panel.velocity_matched_braking_candidate_banks(
        states[np.asarray(design, dtype=np.int64), panel.SOURCE_FRAME], goal,
        proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
    )
    held_hashes = {panel.sequence_hash(action) for action in held}
    train_hashes = {panel.sequence_hash(action) for action in train}
    contracts = interaction["selected_pair_contract"]
    candidate = {
        "candidate_design": "velocity_matched_braking_pairs", "goal_qpos": goal.tolist(),
        "held_shape": list(held.shape), "train_shape": list(train.shape),
        "shared_candidate_in_support": bool(held_hashes <= train_hashes),
        "held_action_hash_count": len(held_hashes), "train_action_hash_count": len(train_hashes),
        "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
        "scorer_candidate_bank_protocol": "held_exact_shared_bank", "scorer_training_action_hash_count": len(held_hashes),
        "pairwise_contrast_contract": "mixed_velocity_alias_direct_route_v1",
        "pairwise_absolute_cost_gap_q10": float(interaction["selected_pooled_absolute_gap_q10"]),
        "pairwise_gap_ge_0p003_fraction": float(interaction["selected_gap_ge_margin_fraction"]),
        "pairwise_sign_mixed_pair_count": int(interaction["selected_sign_mixed_pair_count"]),
        "minimum_fold_velocity_reversal_requirement": 0.75,
        "minimum_fold_velocity_reversal": min(float(item["minimum_fold_velocity_reversal_fraction"]) for item in contracts),
        "maximum_crossfold_position_only_sign_accuracy_requirement": 0.625,
        "maximum_crossfold_position_only_sign_accuracy": max(float(item["maximum_crossfold_position_only_sign_accuracy"]) for item in contracts),
        "selected_pair_contract": contracts, "selection_order": interaction["selection_order"],
        "mechanism_route": "visual_history_xy_displacement_norm_ge_0.02: recurrent_history; otherwise: action_only",
    }
    split_roots = {name: sorted(root_id for root_id, assigned in split.items() if assigned == name) for name in split_counts}
    checks = {
        "contexts_present": bool(resolved), "mixed_pool_schema": manifest.get("schema") == POOL_SCHEMA,
        **_regime_checks(states), "root_groups_complete": len(split) == 768 and len(design) == 64,
        "split_disjoint_from_prior_roots": not bool(set(split) & prior_roots),
        "design_disjoint_from_prior_roots": not bool(set(design) & prior_roots),
        "design_disjoint_from_evaluation": not bool(set(design) & set(split)),
        "design_disjoint_from_renderer_chart": not bool(set(design) & set(chart_roots)),
        "causal_renderer_history_visible": bool(visible),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not bool((held_hashes | train_hashes) & prior_hashes),
        "pairwise_absolute_cost_gap_q10": candidate["pairwise_absolute_cost_gap_q10"] >= 0.003,
        "pairwise_decision_sensitive_fraction": candidate["pairwise_gap_ge_0p003_fraction"] >= 0.80,
        "opposed_velocity_winner_reversal": candidate["minimum_fold_velocity_reversal"] >= 0.75,
        "position_only_proxy_rejected": candidate["maximum_crossfold_position_only_sign_accuracy"] <= 0.625,
        "fixed_visual_motion_router_threshold": MOTION_THRESHOLD == 0.02,
    }
    return {
        "schema": SCHEMA, "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks, "dataset": str(root.resolve()), "dataset_states_sha256": _sha256(root / "states.pth"),
        "construction": {"root_seed": int(root_seed), "held_action_seed": int(held_action_seed), "train_action_seed": int(train_action_seed), "root_id_min": int(root_id_min), "root_id_max_exclusive": int(root_id_max_exclusive)},
        "prior_context_sha256": context_sha256, "root_counts": {name: len(values) for name, values in split_roots.items()},
        "split_roots": split_roots, "interaction_design_roots": design, "regime_root_counts": regime_counts,
        "candidate_bank": candidate,
        "claim_boundary": "Static C2 route feasibility only; no learned fit, calibration outcome, held outcome, or repair-effect claim.",
    }
