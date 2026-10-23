"""Host-only, fresh-root admission audit for a position-aliased C1 bank.

V12 balanced true pair winners but left a strong action-only task-geometry
baseline. V13 adds a pre-fit design gate: a cross-fold position/goal plus
mean-action-effect proxy must not reliably identify the winner. This is only
a static feasibility test; it fits no encoder and opens no outcome labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.decision_benchmark import extend_pointmaze_state_action_pool as extension_tool
from experiments.gdd_wm.audit_pointmaze_candidate_design_frontier import pairwise_contrast_metrics
from experiments.gdd_wm.preflight_pointmaze_planning_shaped_advantage_v7 import _collision_pairs, _sha256


SCHEMA = "pointmaze_pairwise_contrast_v13_position_alias_static_preflight.v1"
ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = ROOT / "data" / "point_maze_v55_pairwise_contrast_v12_balance_extension_20261022"
DATASET = ROOT / "data" / "point_maze_v56_pairwise_contrast_v13_position_alias_extension_20261023"
OUTPUT = ROOT / "results" / "pointmaze_pairwise_contrast_v13_position_alias_static_preflight_20261023.json"
ROOT_RANGE = (21_384, 22_216)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
EXTENSION_SEED = 202611231
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611232, 202611233, 202611234
MIN_Q10, MIN_COVERAGE, MIN_FOLD_BALANCE, MAX_PROXY_ACCURACY = 0.003, 0.80, 0.40, 0.625


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prior_contexts() -> tuple[Path, ...]:
    contexts: list[Path] = []
    for path in sorted((ROOT / "results").rglob("*context*.json")):
        try:
            payload = _read_object(path)
            metadata, records = payload.get("metadata"), payload.get("records")
            if (
                isinstance(metadata, dict) and isinstance(metadata.get("candidate_family"), str)
                and isinstance(records, list)
                and all(isinstance(row, dict) and {"start_episode_id", "candidate_action_hash", "candidate_template_fingerprint"} <= set(row) for row in records)
            ):
                contexts.append(path.resolve())
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            continue
    if not contexts:
        raise RuntimeError("V13 needs at least one compatible historical context for identity exclusion")
    return tuple(contexts)


def _ensure_extension() -> dict[str, Any]:
    expected = {
        "source_root": str(BASE_DATASET.resolve()), "seed": EXTENSION_SEED,
        "new_episode_count": ROOT_RANGE[1] - ROOT_RANGE[0], "base_root_count": ROOT_RANGE[0],
        "extended_root_count": ROOT_RANGE[1], "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
    }
    manifest = DATASET / "extension_manifest.json"
    if DATASET.exists():
        if not manifest.is_file():
            raise RuntimeError("V13 extension directory exists without immutable manifest")
        payload = _read_object(manifest)
        mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatch:
            raise RuntimeError(f"V13 extension contract mismatch: {mismatch}")
        return {"resumed_existing_extension": True, "manifest": str(manifest.resolve())}
    return extension_tool.build_extension(
        BASE_DATASET, DATASET, episode_count=ROOT_RANGE[1] - ROOT_RANGE[0], seed=EXTENSION_SEED,
    )


def preflight(*, contexts: tuple[Path, ...]) -> dict[str, Any]:
    prior_roots, prior_hashes, _templates, context_hashes = panel._all_prior_identities(list(contexts))
    states = np.asarray(torch.load(DATASET / "states.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    split = panel._split_roots_v6(
        seed=ROOT_SEED, forbidden=prior_roots, root_count=len(states), split_counts=SPLIT_COUNTS,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
    )
    chart_roots = panel._renderer_chart_roots_v6(DATASET, set(split))
    design_roots = panel._interaction_design_roots(
        seed=ROOT_SEED, forbidden=prior_roots | set(chart_roots), evaluation_roots=set(split), root_count=len(states),
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1],
    )
    goal = panel.validate_goal_qpos((0.8, 0.8))
    try:
        held, train, report = panel.pairwise_contrast_position_aliased_interaction_candidate_banks(
            states[np.asarray(design_roots, dtype=np.int64), panel.SOURCE_FRAME], goal,
            proposal_seed=HELD_ACTION_SEED, supplemental_seed=TRAIN_ACTION_SEED,
        )
    except ValueError as error:
        message = str(error)
        if not message.startswith("position-aliased pair design has insufficient feasible pairs"):
            raise
        return {
            "schema": SCHEMA, "status": "rejected_static_preflight",
            "checks": {"position_alias_candidate_bank_feasible": False},
            "dataset": str(DATASET.resolve()), "dataset_states_sha256": _sha256(DATASET / "states.pth"),
            "construction": {"extension_seed": EXTENSION_SEED, "root_seed": ROOT_SEED, "held_action_seed": HELD_ACTION_SEED, "train_action_seed": TRAIN_ACTION_SEED, "root_id_min": ROOT_RANGE[0], "root_id_max_exclusive": ROOT_RANGE[1]},
            "prior_context_sha256": context_hashes, "root_counts": {name: sum(value == name for value in split.values()) for name in SPLIT_COUNTS},
            "interaction_design_roots": design_roots,
            "rejection_reason": message,
            "claim_boundary": "Static V13 position-alias rejection only; no visual encoder/model fit, calibration label, outcome label, or planning claim.",
        }
    metrics = pairwise_contrast_metrics(states[np.asarray(design_roots, dtype=np.int64), panel.SOURCE_FRAME], held, goal)
    held_hashes = {panel.sequence_hash(action) for action in held}
    train_hashes = {panel.sequence_hash(action) for action in train}
    contracts = report["selected_pair_contract"]
    checks = {
        "contexts_present": bool(contexts), "split_disjoint_from_prior_roots": not bool(set(split) & prior_roots),
        "design_disjoint_from_evaluation": not bool(set(design_roots) & set(split)),
        "design_disjoint_from_prior_roots": not bool(set(design_roots) & prior_roots),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not bool((held_hashes | train_hashes) & prior_hashes),
        "minimum_summary_collision_pairs": _collision_pairs(held) >= len(held) // 2,
        "pairwise_absolute_cost_gap_q10": float(metrics["pooled_absolute_gap_q10"]) >= MIN_Q10,
        "pairwise_decision_sensitive_fraction": float(metrics["pooled_gap_ge_0p003_fraction"]) >= MIN_COVERAGE,
        "pairwise_state_conditional_signs": int(metrics["sign_mixed_pair_count"]) == len(held) // 2,
        "pairwise_two_fold_sign_balance": min(float(item["minimum_fold_sign_balance"]) for item in contracts) >= MIN_FOLD_BALANCE,
        "position_only_proxy_rejected": max(float(item["maximum_crossfold_position_only_sign_accuracy"]) for item in contracts) <= MAX_PROXY_ACCURACY,
    }
    return {
        "schema": SCHEMA, "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks, "dataset": str(DATASET.resolve()), "dataset_states_sha256": _sha256(DATASET / "states.pth"),
        "construction": {"extension_seed": EXTENSION_SEED, "root_seed": ROOT_SEED, "held_action_seed": HELD_ACTION_SEED, "train_action_seed": TRAIN_ACTION_SEED, "root_id_min": ROOT_RANGE[0], "root_id_max_exclusive": ROOT_RANGE[1]},
        "prior_context_sha256": context_hashes, "root_counts": {name: sum(value == name for value in split.values()) for name in SPLIT_COUNTS},
        "interaction_design_roots": design_roots,
        "candidate_bank": {
            "candidate_design": "pairwise_contrast_position_aliased_interaction_pairs", "goal_qpos": goal.tolist(),
            "held_shape": list(held.shape), "train_shape": list(train.shape),
            "shared_candidate_in_support": bool(held_hashes <= train_hashes), "held_action_hash_count": len(held_hashes),
            "train_action_hash_count": len(train_hashes), "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
            "scorer_candidate_bank_protocol": "held_exact_shared_bank", "scorer_training_action_hash_count": len(held_hashes),
            "canonical_summary_collision_pair_count": _collision_pairs(held), "pairwise_contrast_contract": "state_conditional_summary_collision_pairs_v1",
            "pairwise_absolute_cost_gap_q10": float(metrics["pooled_absolute_gap_q10"]), "pairwise_gap_ge_0p003_fraction": float(metrics["pooled_gap_ge_0p003_fraction"]),
            "pairwise_sign_mixed_pair_count": int(metrics["sign_mixed_pair_count"]), "position_alias_contract": report,
        },
        "claim_boundary": "Static V13 position-alias feasibility only; no visual encoder/model fit, calibration label, outcome label, or planning claim.",
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to reopen immutable V13 static audit: {OUTPUT}")
    extension = _ensure_extension()
    contexts = _prior_contexts()
    result = preflight(contexts=contexts)
    result["extension"] = extension
    result["prior_context_count"] = len(contexts)
    result["prior_context_sha256"] = {str(path): _sha(path) for path in contexts}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": result["status"], "checks": result["checks"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
