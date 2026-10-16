"""Calibration-only selection of a conservative frozen-model cost tie-break."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.gdd_wm.audit_train_only_teacher_capacity import (
    _bootstrap,
    _macro_root_regrets,
    _proposal_relative_action_summaries,
)
from experiments.gdd_wm.build_train_only_readout_teacher_package import _action_summary


def _readout(features: np.ndarray, artifact: Any, prefix: str) -> np.ndarray:
    standardized = (
        features - np.asarray(artifact[f"{prefix}_mean"], dtype=np.float64)
    ) / np.asarray(artifact[f"{prefix}_scale"], dtype=np.float64)
    design = np.concatenate(
        [np.ones((len(standardized), 1), dtype=np.float64), standardized], axis=1
    )
    return design @ np.asarray(artifact[f"{prefix}_coefficient"], dtype=np.float64)


def _no_dynamics_scores(
    source: np.ndarray, action: np.ndarray, artifact: Any
) -> np.ndarray:
    score = _readout(action, artifact, "action")
    features = np.concatenate([source, action], axis=1)
    hidden = (
        features
        - np.asarray(artifact["mlp_no_dynamics_feature_mean"], dtype=np.float64)
    ) / np.asarray(artifact["mlp_no_dynamics_feature_scale"], dtype=np.float64)
    for layer in (0, 2):
        hidden = (
            hidden
            @ np.asarray(
                artifact[f"mlp_no_dynamics_layer_{layer}_weight"], dtype=np.float64
            ).T
            + np.asarray(
                artifact[f"mlp_no_dynamics_layer_{layer}_bias"], dtype=np.float64
            )
        )
        hidden = hidden / (1.0 + np.exp(-hidden))
    normalized = (
        hidden
        @ np.asarray(artifact["mlp_no_dynamics_layer_4_weight"], dtype=np.float64).T
        + np.asarray(artifact["mlp_no_dynamics_layer_4_bias"], dtype=np.float64)
    ).reshape(-1)
    residual = (
        normalized
        * float(np.asarray(artifact["mlp_no_dynamics_residual_scale"]).reshape(-1)[0])
        + float(np.asarray(artifact["mlp_no_dynamics_residual_mean"]).reshape(-1)[0])
    )
    return score + residual


def _shuffle_within_cells(values: np.ndarray, groups: list[str]) -> np.ndarray:
    shuffled = np.asarray(values, dtype=np.float64).copy()
    group_array = np.asarray(groups, dtype=object)
    for group in np.unique(group_array):
        indexes = np.flatnonzero(group_array == group)
        shuffled[indexes] = np.roll(shuffled[indexes], 1)
    return shuffled


def _elite_selection_scores(
    no_dynamics: np.ndarray,
    model_cost: np.ndarray,
    groups: list[str],
    k: int,
) -> np.ndarray:
    if k < 1:
        raise ValueError("elite k must be positive")
    no_dynamics = np.asarray(no_dynamics, dtype=np.float64)
    model_cost = np.asarray(model_cost, dtype=np.float64)
    if no_dynamics.shape != model_cost.shape or len(groups) != len(no_dynamics):
        raise ValueError("elite score arrays/groups must align")
    scores = np.full(len(no_dynamics), np.inf, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    for group in np.unique(group_array):
        indexes = np.flatnonzero(group_array == group)
        elite = indexes[
            np.argsort(no_dynamics[indexes], kind="stable")[: min(k, len(indexes))]
        ]
        selected = elite[int(np.argmin(model_cost[elite]))]
        scores[selected] = 0.0
    return scores


def run(
    candidate_dataset: Path,
    readout_artifact: Path,
    output: Path,
    *,
    k_grid: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    draws: int = 20_000,
    seed: int = 20261022,
) -> dict[str, Any]:
    payload = json.loads(candidate_dataset.read_text(encoding="utf-8"))
    rows = payload.get("records", [])
    if not rows:
        raise ValueError("candidate dataset contains no records")
    context = str(payload.get("metadata", {}).get("candidate_bank_hash", "context"))
    groups = [f"{context}::{row['instance_id']}" for row in rows]
    roots = [str(row["instance_id"]) for row in rows]
    splits = np.asarray([str(row["split"]) for row in rows], dtype=object)
    costs = np.asarray([float(row["real_task_cost"]) for row in rows])
    source = np.asarray([row["source_latent_model"] for row in rows], dtype=np.float64)
    action = _proposal_relative_action_summaries(
        np.asarray([_action_summary(row["action_sequence"]) for row in rows]), groups
    )
    model_cost = np.asarray([-float(row["teacher_score"]) for row in rows])
    shuffled_cost = _shuffle_within_cells(model_cost, groups)
    with np.load(readout_artifact, allow_pickle=False) as artifact:
        no_dynamics = _no_dynamics_scores(source, action, artifact)
    calibration_mask = splits == "calibration"
    outcome_mask = splits == "outcome"
    if not calibration_mask.any() or not outcome_mask.any():
        raise ValueError("candidate dataset requires calibration and outcome roots")

    baseline_scores = _elite_selection_scores(no_dynamics, model_cost, groups, 1)
    baseline_calibration = _macro_root_regrets(
        baseline_scores, costs, groups, roots, calibration_mask, 1.0e-3
    )
    calibration: dict[str, Any] = {}
    selected_k = 1
    for index, k in enumerate(k_grid):
        true_scores = _elite_selection_scores(no_dynamics, model_cost, groups, k)
        shuffled_scores = _elite_selection_scores(
            no_dynamics, shuffled_cost, groups, k
        )
        true_regret = _macro_root_regrets(
            true_scores, costs, groups, roots, calibration_mask, 1.0e-3
        )
        shuffled_regret = _macro_root_regrets(
            shuffled_scores, costs, groups, roots, calibration_mask, 1.0e-3
        )
        common = sorted(baseline_calibration)
        vs_baseline = _bootstrap(
            np.asarray(
                [true_regret[root] - baseline_calibration[root] for root in common]
            ),
            np.random.default_rng(seed + 10 * index),
            draws,
        )
        vs_shuffled = _bootstrap(
            np.asarray(
                [true_regret[root] - shuffled_regret[root] for root in common]
            ),
            np.random.default_rng(seed + 10 * index + 1),
            draws,
        )
        passed = bool(vs_baseline["ci95"][1] < 0.0 and vs_shuffled["ci95"][1] < 0.0)
        calibration[str(k)] = {
            "true_mean_regret": float(np.mean(list(true_regret.values()))),
            "shuffled_mean_regret": float(np.mean(list(shuffled_regret.values()))),
            "true_minus_no_dynamics": vs_baseline,
            "true_minus_shuffled": vs_shuffled,
            "gate_passed": passed,
        }
        if selected_k == 1 and passed:
            selected_k = int(k)

    def outcome_regrets(base_cost: np.ndarray, k: int) -> dict[str, float]:
        scores = _elite_selection_scores(no_dynamics, base_cost, groups, k)
        return _macro_root_regrets(
            scores, costs, groups, roots, outcome_mask, 1.0e-3
        )

    baseline_outcome = outcome_regrets(model_cost, 1)
    true_outcome = outcome_regrets(model_cost, selected_k)
    shuffled_outcome = outcome_regrets(shuffled_cost, selected_k)
    outcome_roots = sorted(baseline_outcome)
    report = {
        "schema": "reacher_model_cost_elite_selection.v1",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "readout_artifact": str(readout_artifact.resolve()),
            "selection_split": "calibration",
            "evaluation_split": "outcome",
            "outcome_used_for_selection": False,
            "selection_rule": (
                "smallest k whose calibration root-bootstrap CI95 is below zero "
                "versus both no-dynamics and within-cell shuffled model cost"
            ),
            "k_grid": list(k_grid),
            "draws": draws,
            "seed": seed,
        },
        "selection": {
            "selected_k": selected_k,
            "fallback": selected_k == 1,
            "calibration": calibration,
        },
        "outcome": {
            "no_dynamics_mean_regret": float(np.mean(list(baseline_outcome.values()))),
            "true_elite_mean_regret": float(np.mean(list(true_outcome.values()))),
            "shuffled_elite_mean_regret": float(np.mean(list(shuffled_outcome.values()))),
            "true_minus_no_dynamics": _bootstrap(
                np.asarray(
                    [true_outcome[root] - baseline_outcome[root] for root in outcome_roots]
                ),
                np.random.default_rng(seed + 1001),
                draws,
            ),
            "true_minus_shuffled": _bootstrap(
                np.asarray(
                    [true_outcome[root] - shuffled_outcome[root] for root in outcome_roots]
                ),
                np.random.default_rng(seed + 1002),
                draws,
            ),
        },
        "claim_boundary": (
            "offline calibration-selected elite tie-break diagnostic; online and shuffled "
            "consequences are required before any predicted-dynamics claim"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--readout-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    run(
        args.candidate_dataset,
        args.readout_artifact,
        args.output,
        draws=args.draws,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
