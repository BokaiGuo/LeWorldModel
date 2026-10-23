"""Freeze the V9 opportunity-gated router panel before GPU model fitting.

V9 is a fresh test of the V5 mechanism finding.  It does *not* inspect a
future scorer disagreement rate: that quantity only exists after the frozen
scorers have been fit.  Instead this ledger fixes the independent root pool,
the supported action bank, capacity requirements, and the fail-closed rule
that will reject the panel if its cross-fit or calibration opportunity strata
are too small.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm.preflight_pointmaze_planning_shaped_advantage_v7 import _collision_pairs, _sha256
from experiments.gdd_wm.run_pointmaze_planning_shaped_repairability import (
    ADVANTAGE_ROUTER_KINDS,
    V7_POST_EXECUTION_ALIASING_CONTRACT,
    V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE,
    V9_MIN_ROUTER_ROOT_SURPLUS,
    V9_MIN_TRAIN_OPPORTUNITY_ROOTS,
    V9_ROUTER_FEATURE_CONTRACT,
    V9_ROUTER_FEATURE_DIMENSION,
)


SCHEMA = "pointmaze_planning_shaped_opportunity_router_v9_static_preflight.v1"
# The original V9 range was subsequently found to overlap an already-opened
# counterfactual panel.  Keep that pool immutable and move this independent
# opportunity-router test to a genuinely fresh continuation range.
EXTENSION_SEED = 202610930
ROOT_RANGE = (14264, 14776)
# V5 observed scorer disagreement on roughly 22% of roots.  Keeping 128
# training roots would therefore expect only about 28 routing opportunities,
# below the frozen 31-root capacity gate.  V9 is unrun, so reserve the same
# total 384 evaluation roots but allocate 160/160/64 before fitting anything.
SPLIT_COUNTS = {"train": 160, "calibration": 160, "outcome": 64}
MIN_TOP2_GAP_Q10 = 0.001
MIN_TOP2_GAP_FRACTION = 0.50
MIN_COST_SPREAD_Q10 = 0.05
OPPORTUNITY_CANDIDATE_DESIGNS = frozenset({
    "goal_balanced_interaction_pairs",
    "tail_robust_goal_balanced_interaction_pairs",
    "expanded_tail_robust_goal_balanced_interaction_pairs",
    "ultra_tail_robust_goal_balanced_interaction_pairs",
    "directional_tail_robust_goal_balanced_interaction_pairs",
})


def _interaction_headroom(interaction: dict[str, Any]) -> dict[str, float]:
    """Normalize legacy and tail-robust design reports for the fixed gates.

    The original goal-balanced constructor exports an aggregate
    ``design_oracle_headroom`` map.  The tail-robust constructors instead
    export the corresponding aggregate as ``design_all_root_headroom`` after
    recording per-fold diagnostics.  Both quantify the same declared-goal,
    pre-outcome candidate-bank properties, so the gate deliberately consumes
    their named values rather than silently weakening a threshold.
    """
    if not isinstance(interaction, dict):
        raise ValueError("interaction candidate constructor must return a report object")
    headroom = interaction.get("design_oracle_headroom", interaction.get("design_all_root_headroom"))
    if not isinstance(headroom, dict):
        raise ValueError("interaction candidate report lacks aggregate declared-goal headroom")
    gap = headroom.get("best_second_cost_gap")
    spread = headroom.get("candidate_cost_spread")
    return {
        "winner_entropy_normalized": float(
            interaction.get("design_winner_entropy_normalized", headroom.get("winner_entropy_normalized", float("nan")))
        ),
        "best_second_gap_q10": float(
            gap.get("q10", float("nan")) if isinstance(gap, dict) else headroom.get("best_second_gap_q10", float("nan"))
        ),
        "best_second_gap_ge_0p003_fraction": float(
            gap.get("ge_0p003_fraction", float("nan")) if isinstance(gap, dict) else headroom.get("best_second_gap_ge_0p003_fraction", float("nan"))
        ),
        "candidate_cost_spread_q10": float(
            spread.get("q10", float("nan")) if isinstance(spread, dict) else headroom.get("candidate_cost_spread_q10", float("nan"))
        ),
    }


def _extension_contract(
    root: Path, *, extension_seed: int = EXTENSION_SEED,
    root_range: tuple[int, int] = ROOT_RANGE,
) -> dict[str, Any]:
    """Validate an immutable extension against the declared root contract.

    V9's defaults intentionally remain frozen.  Later prospective panels may
    reuse the *mechanism* only by explicitly declaring a disjoint extension
    seed and range at preflight time; silently inheriting V9's range would
    either reject the panel or, worse, make its provenance ambiguous.
    """
    if int(root_range[0]) < 0 or int(root_range[1]) <= int(root_range[0]):
        raise ValueError("extension root range must be a non-empty non-negative half-open interval")
    path = root / "extension_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("V9 dataset requires extension_manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema": value.get("schema") == "pointmaze_independent_state_action_extension.v1",
        "seed": value.get("seed") == int(extension_seed),
        "new_episode_count": value.get("new_episode_count") == root_range[1] - root_range[0],
        "range": value.get("new_root_id_range") == [root_range[0], root_range[1] - 1],
        "base_root_count": value.get("base_root_count") == root_range[0],
        "extended_root_count": value.get("extended_root_count") == root_range[1],
    }
    if not all(checks.values()):
        raise ValueError(f"V9 extension manifest violates frozen contract: {checks}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "checks": checks}


def preflight(*, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int,
              train_action_seed: int, router_kind: str = "advantage_ridge",
              extension_seed: int = EXTENSION_SEED, root_id_min: int = ROOT_RANGE[0],
              root_id_max_exclusive: int = ROOT_RANGE[1],
              candidate_design: str = "goal_balanced_interaction_pairs",
              split_counts: dict[str, int] | None = None) -> dict[str, Any]:
    if router_kind not in ADVANTAGE_ROUTER_KINDS:
        raise ValueError("V9 preflight requires a declared advantage router")
    if candidate_design not in OPPORTUNITY_CANDIDATE_DESIGNS:
        raise ValueError(f"V9 preflight does not support candidate design {candidate_design!r}")
    root_range = (int(root_id_min), int(root_id_max_exclusive))
    if root_range[0] < 0 or root_range[1] <= root_range[0]:
        raise ValueError("V9 root range must be a non-empty non-negative half-open interval")
    root = Path(dataset_root)
    declared_split_counts = panel._validated_split_counts(SPLIT_COUNTS if split_counts is None else split_counts)
    states_path = root / "states.pth"
    if not states_path.is_file() or not (root / "obses").is_dir():
        raise FileNotFoundError("V9 dataset requires states.pth and obses")
    extension = _extension_contract(root, extension_seed=int(extension_seed), root_range=root_range)
    resolved_contexts = [Path(item).resolve() for item in contexts]
    old_roots, old_hashes, _templates, context_sha256 = panel._all_prior_identities(resolved_contexts)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    if states.shape != (root_range[1], 100, 4) or not np.isfinite(states).all():
        raise ValueError(f"V9 state pool must be finite [{root_range[1]},100,4]")
    split = panel._split_roots_v6(
        seed=int(root_seed), forbidden=old_roots, root_count=len(states), split_counts=declared_split_counts,
        root_id_min=root_range[0], root_id_max_exclusive=root_range[1],
    )
    visual = panel._renderer_chart_roots_v6(root, set(split))
    design = panel._interaction_design_roots(
        seed=int(root_seed), forbidden=old_roots | set(visual), evaluation_roots=set(split), root_count=len(states),
        root_id_min=root_range[0], root_id_max_exclusive=root_range[1],
    )
    goal = panel.validate_goal_qpos((0.8, 0.8))
    design_states = states[np.asarray(design, dtype=np.int64), panel.SOURCE_FRAME]
    constructors = {
        "goal_balanced_interaction_pairs": panel.goal_balanced_interaction_candidate_banks,
        "tail_robust_goal_balanced_interaction_pairs": panel.tail_robust_goal_balanced_interaction_candidate_banks,
        "expanded_tail_robust_goal_balanced_interaction_pairs": panel.expanded_tail_robust_goal_balanced_interaction_candidate_banks,
        "ultra_tail_robust_goal_balanced_interaction_pairs": panel.ultra_tail_robust_goal_balanced_interaction_candidate_banks,
        "directional_tail_robust_goal_balanced_interaction_pairs": panel.directional_tail_robust_goal_balanced_interaction_candidate_banks,
    }
    held, train, interaction = constructors[candidate_design](
        design_states, goal, proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
    )
    held_hashes = {panel.sequence_hash(item) for item in held}
    train_hashes = {panel.sequence_hash(item) for item in train}
    headroom = _interaction_headroom(interaction)
    calibration_roles = {"selector": 0, "certification": 0}
    for root_id, role in split.items():
        if role == "calibration":
            fold = int(hashlib.sha256(str(root_id).encode("utf-8")).hexdigest()[:8], 16) % 4
            calibration_roles["selector" if fold % 2 == 0 else "certification"] += 1
    checks = {
        "contexts_present": bool(resolved_contexts),
        "split_disjoint_from_prior_roots": not (set(split) & old_roots),
        "design_disjoint_from_evaluation": not (set(design) & set(split)),
        "design_disjoint_from_prior_roots": not (set(design) & old_roots),
        "held_actions_supported_by_train": held_hashes <= train_hashes,
        "actions_fresh_against_prior_contexts": not ((held_hashes | train_hashes) & old_hashes),
        "goal_balanced_design_entropy": headroom["winner_entropy_normalized"] >= 0.50,
        "top2_gap_lower_tail": headroom["best_second_gap_q10"] >= MIN_TOP2_GAP_Q10,
        "decision_sensitive_fraction": headroom["best_second_gap_ge_0p003_fraction"] >= MIN_TOP2_GAP_FRACTION,
        "candidate_cost_spread": headroom["candidate_cost_spread_q10"] >= MIN_COST_SPREAD_Q10,
        "minimum_summary_collision_pairs": _collision_pairs(held) >= 8,
        "router_train_root_surplus": declared_split_counts["train"] >= V9_ROUTER_FEATURE_DIMENSION + V9_MIN_ROUTER_ROOT_SURPLUS,
        "calibration_roles_populated": min(calibration_roles.values()) >= V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE,
    }
    return {
        "schema": SCHEMA, "status": "passed_static_preflight" if all(checks.values()) else "rejected_static_preflight",
        "checks": checks, "dataset": str(root.resolve()), "dataset_states_sha256": _sha256(states_path), "extension": extension,
        "construction": {"root_seed": int(root_seed), "held_action_seed": int(held_action_seed), "train_action_seed": int(train_action_seed), "extension_seed": int(extension_seed), "root_id_min": root_range[0], "root_id_max_exclusive": root_range[1]},
        "prior_context_count": len(resolved_contexts), "prior_context_sha256": context_sha256,
        "root_counts": {name: sum(value == name for value in split.values()) for name in declared_split_counts},
        "split_roots": {name: sorted(root_id for root_id, value in split.items() if value == name) for name in declared_split_counts},
        "visual_chart_root_count": len(visual), "interaction_design_root_count": len(design), "interaction_design_roots": design,
        "post_execution_aliasing_contract": dict(V7_POST_EXECUTION_ALIASING_CONTRACT),
        "candidate_bank": {
            "candidate_design": candidate_design, "goal_qpos": goal.tolist(),
            "held_shape": list(held.shape), "train_shape": list(train.shape),
            "held_action_hash_count": len(held_hashes), "train_action_hash_count": len(train_hashes),
            "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
            "shared_candidate_in_support": bool(held_hashes <= train_hashes),
            "scorer_candidate_bank_protocol": "held_exact_shared_bank", "scorer_training_action_hash_count": len(held_hashes),
            "canonical_summary_collision_pair_count": _collision_pairs(held),
            "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
            "design_best_second_cost_gap_q10": headroom["best_second_gap_q10"],
            "design_best_second_cost_gap_ge_0p003_fraction": headroom["best_second_gap_ge_0p003_fraction"],
            "design_candidate_cost_spread_q10": headroom["candidate_cost_spread_q10"],
            "router_feature_contract": V9_ROUTER_FEATURE_CONTRACT,
            "router_feature_dimension": V9_ROUTER_FEATURE_DIMENSION, "router_train_root_count": declared_split_counts["train"],
            "minimum_router_train_root_surplus": V9_MIN_ROUTER_ROOT_SURPLUS, "router_kind": router_kind,
            "router_action_representation": "full_ordered_20x2",
            "router_regularization_selection": "train_root_only_four_fold_blocked_advantage_mse",
            "router_training_scope": "scorer_disagreement_only",
            "opportunity_gate": "cost_planning_candidate_disagreement",
            "minimum_router_opportunity_root_count": V9_MIN_TRAIN_OPPORTUNITY_ROOTS,
            "minimum_calibration_opportunity_roots_per_role": V9_MIN_CALIBRATION_OPPORTUNITY_ROOTS_PER_ROLE,
        },
        "claim_boundary": "Static V9 identity/support/capacity ledger only; actual scorer disagreement and efficacy are deferred to the fresh CUDA execution and cannot be inferred from this preflight.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--contexts", required=True, nargs="+", type=Path)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--held-action-seed", required=True, type=int)
    parser.add_argument("--train-action-seed", required=True, type=int)
    parser.add_argument("--router-kind", choices=tuple(sorted(ADVANTAGE_ROUTER_KINDS)), default="advantage_ridge")
    parser.add_argument("--extension-seed", type=int, default=EXTENSION_SEED)
    parser.add_argument("--root-id-min", type=int, default=ROOT_RANGE[0])
    parser.add_argument("--root-id-max-exclusive", type=int, default=ROOT_RANGE[1])
    parser.add_argument("--candidate-design", choices=tuple(sorted(OPPORTUNITY_CANDIDATE_DESIGNS)), default="goal_balanced_interaction_pairs")
    parser.add_argument("--train-root-count", type=int, default=SPLIT_COUNTS["train"])
    parser.add_argument("--calibration-root-count", type=int, default=SPLIT_COUNTS["calibration"])
    parser.add_argument("--outcome-root-count", type=int, default=SPLIT_COUNTS["outcome"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-identical-output", action="store_true")
    args = vars(parser.parse_args())
    output, reuse = args.pop("output"), args.pop("reuse_identical_output")
    args["split_counts"] = {
        "train": args.pop("train_root_count"), "calibration": args.pop("calibration_root_count"),
        "outcome": args.pop("outcome_root_count"),
    }
    result = preflight(**args)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if not reuse or output.read_text(encoding="utf-8") != serialized:
            raise ValueError("refusing to overwrite or alter an immutable V9 preflight")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"status": result["status"], "root_counts": result["root_counts"]}, indent=2))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
