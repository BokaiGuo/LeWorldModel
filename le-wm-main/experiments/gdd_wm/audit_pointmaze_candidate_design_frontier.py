"""Compare frozen PointMaze candidate-bank constructions before model fitting.

The repairability routes require a nontrivial oracle decision margin.  A
candidate bank whose lower-tail best-versus-second-best gap is nearly zero
cannot distinguish a learned-ranking error from an intrinsically tied choice.
This static audit evaluates only declared design roots, the declared goal, and
the deterministic simulator used for candidate-bank construction.  It never
loads RGB, fits a learned model, reads calibration/outcome labels, or selects a
deployment policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm.preflight_pointmaze_endpoint_repairability_router_v10 import (
    MIN_COST_SPREAD_Q10,
    MIN_TOP2_GAP_FRACTION,
    MIN_TOP2_GAP_Q10,
)


SCHEMA = "pointmaze_candidate_design_frontier_audit.v1"
SUPPORTED_DESIGNS = (
    "expanded_tail_robust_goal_balanced_interaction_pairs",
    "ultra_tail_robust_goal_balanced_interaction_pairs",
    "directional_tail_robust_goal_balanced_interaction_pairs",
)


def _constructor(name: str) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    constructors: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]] = {
        "expanded_tail_robust_goal_balanced_interaction_pairs": (
            panel.expanded_tail_robust_goal_balanced_interaction_candidate_banks
        ),
        "ultra_tail_robust_goal_balanced_interaction_pairs": (
            panel.ultra_tail_robust_goal_balanced_interaction_candidate_banks
        ),
        "directional_tail_robust_goal_balanced_interaction_pairs": (
            panel.directional_tail_robust_goal_balanced_interaction_candidate_banks
        ),
    }
    if name not in constructors:
        raise ValueError(f"unsupported candidate design {name!r}")
    return constructors[name]


def acceptance_checks(report: dict[str, Any]) -> dict[str, bool]:
    """Evaluate the exact V10 lower-tail headroom contract."""

    metrics = report.get("design_all_root_headroom")
    if not isinstance(metrics, dict):
        raise ValueError("candidate-design report lacks aggregate headroom")
    required = {
        "winner_entropy_normalized",
        "best_second_gap_q10",
        "best_second_gap_ge_0p003_fraction",
        "candidate_cost_spread_q10",
    }
    if not required <= set(metrics):
        raise ValueError("candidate-design report has incomplete aggregate headroom")
    return {
        "winner_entropy": float(metrics["winner_entropy_normalized"]) >= 0.50,
        "top2_gap_lower_tail": float(metrics["best_second_gap_q10"]) >= MIN_TOP2_GAP_Q10,
        "decision_sensitive_fraction": (
            float(metrics["best_second_gap_ge_0p003_fraction"]) >= MIN_TOP2_GAP_FRACTION
        ),
        "candidate_cost_spread": float(metrics["candidate_cost_spread_q10"]) >= MIN_COST_SPREAD_Q10,
    }


def pairwise_contrast_metrics(
    source_states: np.ndarray, held_actions: np.ndarray, goal_qpos: np.ndarray,
    *, simulator: Callable[[np.ndarray, np.ndarray], np.ndarray] = panel._simulate_sequence,
) -> dict[str, Any]:
    """Measure matched-pair task contrasts without using a full-bank argmin.

    The interaction constructions emit adjacent actions as summary-colliding
    order permutations.  Their pairwise cost difference is the correct P1
    counterfactual estimand: unlike a full-bank top-2 gap, it is not made
    vacuous merely because *another* pair happens to be near-optimal for a
    particular root.
    """

    states = np.asarray(source_states, dtype=np.float64)
    actions = np.asarray(held_actions, dtype=np.float64)
    goal = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("source_states must be finite [roots, 4]")
    if actions.ndim != 3 or actions.shape[1:] != (panel.TOTAL_STEPS, 2) or len(actions) % 2:
        raise ValueError("held_actions must contain an even number of [20, 2] actions")
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("goal_qpos must be finite [2]")
    costs = np.empty((len(states), len(actions)), dtype=np.float64)
    for root_index, state in enumerate(states):
        for action_index, action in enumerate(actions):
            terminal = np.asarray(simulator(state, action), dtype=np.float64)
            costs[root_index, action_index] = float(np.linalg.norm(terminal[:2] - goal))
    signed = costs[:, 0::2] - costs[:, 1::2]
    absolute = np.abs(signed)
    per_pair = []
    for index in range(signed.shape[1]):
        value = signed[:, index]
        per_pair.append({
            "pair_index": int(index),
            "absolute_gap_q10": float(np.quantile(np.abs(value), 0.10)),
            "absolute_gap_mean": float(np.mean(np.abs(value))),
            "gap_ge_0p003_fraction": float(np.mean(np.abs(value) >= 0.003)),
            "left_better_fraction": float(np.mean(value < 0.0)),
            "right_better_fraction": float(np.mean(value > 0.0)),
        })
    return {
        "pair_count": int(signed.shape[1]),
        "root_count": int(signed.shape[0]),
        "pooled_absolute_gap_q10": float(np.quantile(absolute.reshape(-1), 0.10)),
        "pooled_gap_ge_0p003_fraction": float(np.mean(absolute >= 0.003)),
        "sign_mixed_pair_count": int(sum(
            item["left_better_fraction"] > 0.0 and item["right_better_fraction"] > 0.0
            for item in per_pair
        )),
        "per_pair": per_pair,
    }


def audit(
    *, dataset_root: Path, contexts: list[Path], root_seed: int,
    held_action_seed: int, train_action_seed: int, root_id_min: int,
    root_id_max_exclusive: int, split_counts: dict[str, int], designs: tuple[str, ...],
) -> dict[str, Any]:
    """Audit candidate-bank headroom without touching fit or outcome labels."""

    root = Path(dataset_root)
    states_path = root / "states.pth"
    if not states_path.is_file():
        raise FileNotFoundError(f"missing PointMaze states: {states_path}")
    if not designs or any(name not in SUPPORTED_DESIGNS for name in designs):
        raise ValueError(f"designs must be a non-empty subset of {SUPPORTED_DESIGNS}")
    resolved_contexts = [Path(path).resolve() for path in contexts]
    prior_roots, prior_hashes, _prior_templates, context_sha256 = panel._all_prior_identities(resolved_contexts)
    states = np.asarray(torch.load(states_path, map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    validated_counts = panel._validated_split_counts(split_counts)
    split = panel._split_roots_v6(
        seed=int(root_seed), forbidden=prior_roots, root_count=len(states),
        split_counts=validated_counts, root_id_min=int(root_id_min),
        root_id_max_exclusive=int(root_id_max_exclusive),
    )
    chart_roots = panel._renderer_chart_roots_v6(root, set(split))
    design_roots = panel._interaction_design_roots(
        seed=int(root_seed), forbidden=prior_roots | set(chart_roots), evaluation_roots=set(split),
        root_count=len(states), root_id_min=int(root_id_min), root_id_max_exclusive=int(root_id_max_exclusive),
    )
    source_states = states[np.asarray(design_roots, dtype=np.int64), panel.SOURCE_FRAME]
    goal = panel.validate_goal_qpos((0.8, 0.8))
    entries: list[dict[str, Any]] = []
    for name in designs:
        held, train, report = _constructor(name)(
            source_states, goal, proposal_seed=int(held_action_seed),
            supplemental_seed=int(train_action_seed),
        )
        held_hashes = {panel.sequence_hash(action) for action in held}
        train_hashes = {panel.sequence_hash(action) for action in train}
        checks = acceptance_checks(report)
        pairwise = pairwise_contrast_metrics(source_states, held, goal)
        metrics = dict(report["design_all_root_headroom"])
        entries.append({
            "candidate_design": name,
            "checks": checks,
            "passes_headroom_gate": all(checks.values()),
            "design_all_root_headroom": metrics,
            "pairwise_contrast": pairwise,
            "held_action_hash_count": len(held_hashes),
            "held_train_action_hash_overlap_count": len(held_hashes & train_hashes),
            "actions_fresh_against_prior_contexts": not bool((held_hashes | train_hashes) & prior_hashes),
            "selection_order": report.get("selection_order"),
            "proposal_coverage": report.get("proposal_coverage", report.get("proposal_count")),
        })
    return {
        "schema": SCHEMA,
        "status": "completed_static_candidate_design_frontier",
        "claim_boundary": (
            "Static pre-fit candidate-bank evidence only. It does not fit a world model, "
            "open calibration/outcome labels, establish repair efficacy, or authorize deployment."
        ),
        "inputs": {
            "dataset": str(root.resolve()),
            "contexts": [str(path) for path in resolved_contexts],
            "prior_context_sha256": context_sha256,
            "construction": {
                "root_seed": int(root_seed), "held_action_seed": int(held_action_seed),
                "train_action_seed": int(train_action_seed), "root_id_min": int(root_id_min),
                "root_id_max_exclusive": int(root_id_max_exclusive),
            },
            "split_counts": validated_counts,
            "goal_qpos": goal.tolist(),
        },
        "design_root_count": len(design_roots),
        "design_roots": design_roots,
        "entries": entries,
    }


def main() -> int:
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
    parser.add_argument("--designs", choices=SUPPORTED_DESIGNS, nargs="+", default=list(SUPPORTED_DESIGNS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite static candidate-design audit {args.output}")
    result = audit(
        dataset_root=args.dataset_root, contexts=args.contexts, root_seed=args.root_seed,
        held_action_seed=args.held_action_seed, train_action_seed=args.train_action_seed,
        root_id_min=args.root_id_min, root_id_max_exclusive=args.root_id_max_exclusive,
        split_counts={"train": args.train_root_count, "calibration": args.calibration_root_count, "outcome": args.outcome_root_count},
        designs=tuple(args.designs),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "designs": [{"name": row["candidate_design"], "passes": row["passes_headroom_gate"]} for row in result["entries"]],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
