"""Paired three-arm audit of frozen-model cost fusion on fresh Reacher roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "live_reacher_sequence_candidate_planning.v1":
        raise ValueError(f"{path}: unexpected schema")
    return payload


def _bootstrap(values: np.ndarray, *, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired bootstrap requires at least two roots")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indexes].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def _arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = payload["frozen_lewm"]
    return {
        "success": np.asarray(rows["episode_successes"], dtype=np.float64),
        "final_qpos_distance": np.asarray(rows["final_qpos_distance"], dtype=np.float64),
        "min_qpos_distance": np.asarray(rows["min_qpos_distance"], dtype=np.float64),
    }


def _paired(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    result = {
        key: _bootstrap(right[key] - left[key], draws=draws, seed=seed + offset)
        for offset, key in enumerate(
            ("success", "final_qpos_distance", "min_qpos_distance")
        )
    }
    result["success_discordance"] = {
        "right_only_wins": int(
            np.sum((right["success"] == 1) & (left["success"] == 0))
        ),
        "left_only_wins": int(
            np.sum((right["success"] == 0) & (left["success"] == 1))
        ),
        "ties": int(np.sum(right["success"] == left["success"])),
    }
    return result


def compare(
    no_dynamics_path: Path,
    fusion_path: Path,
    shuffled_path: Path,
    *,
    draws: int = 20_000,
    seed: int = 20261021,
) -> dict[str, Any]:
    reports = {
        "no_dynamics": _load(no_dynamics_path),
        "fusion": _load(fusion_path),
        "shuffled": _load(shuffled_path),
    }
    reference = reports["no_dynamics"]
    reference_protocol = reference["protocol"]
    reference_seeds = reference["frozen_lewm"]["seeds"]
    required_equal = (
        "checkpoint",
        "horizon",
        "receding_horizon",
        "action_block",
        "candidate_bank",
        "candidate_count",
        "final_action_selection",
        "target_delta",
        "capacity_readout_artifact",
    )
    for name, report in reports.items():
        if report["frozen_lewm"]["seeds"] != reference_seeds:
            raise ValueError(f"{name}: root seeds do not match")
        mismatches = {
            key: [reference_protocol.get(key), report["protocol"].get(key)]
            for key in required_equal
            if reference_protocol.get(key) != report["protocol"].get(key)
        }
        if mismatches:
            raise ValueError(f"{name}: unmatched protocol: {mismatches}")
        if report["protocol"].get("cost_metric") != "capacity_safe_readout":
            raise ValueError(f"{name}: expected capacity_safe_readout")

    blend_no = reference_protocol.get("capacity_base_cost_blend_weight")
    blend_fusion = reports["fusion"]["protocol"].get(
        "capacity_base_cost_blend_weight"
    )
    blend_shuffled = reports["shuffled"]["protocol"].get(
        "capacity_base_cost_blend_weight"
    )
    elite_no = reference_protocol.get("capacity_base_cost_elite_k")
    elite_fusion = reports["fusion"]["protocol"].get("capacity_base_cost_elite_k")
    elite_shuffled = reports["shuffled"]["protocol"].get(
        "capacity_base_cost_elite_k"
    )
    if blend_no not in (None, 0.0) or elite_no not in (None, 1):
        raise ValueError("no-dynamics arm must disable base-cost interventions")
    if blend_fusion is not None and float(blend_fusion) > 0.0:
        intervention_type = "convex_blend"
        intervention_value = float(blend_fusion)
        if float(blend_shuffled) != intervention_value:
            raise ValueError("fusion and shuffled arms must use the same blend weight")
        if elite_fusion not in (None, 1) or elite_shuffled not in (None, 1):
            raise ValueError("blend comparison cannot also enable elite tie-break")
    elif elite_fusion is not None and int(elite_fusion) > 1:
        intervention_type = "source_action_elite_tiebreak"
        intervention_value = int(elite_fusion)
        if int(elite_shuffled) != intervention_value:
            raise ValueError("fusion and shuffled arms must use the same elite k")
        if blend_shuffled not in (None, 0.0):
            raise ValueError("elite comparison cannot also enable base-cost blend")
    else:
        raise ValueError("fusion arm must enable one base-cost intervention")
    if reports["fusion"]["protocol"].get("capacity_shuffle_base_cost_control"):
        raise ValueError("fusion arm must preserve candidate/model-cost pairing")
    if not reports["shuffled"]["protocol"].get(
        "capacity_shuffle_base_cost_control"
    ):
        raise ValueError("shuffled arm must break candidate/model-cost pairing")

    arrays = {name: _arrays(report) for name, report in reports.items()}
    n = len(reference_seeds)
    if any(any(len(values) != n for values in arm.values()) for arm in arrays.values()):
        raise ValueError("three-arm metric arrays have inconsistent lengths")
    point_estimates = {
        name: {
            "successes": int(np.sum(values["success"])),
            "mean_final_qpos_distance": float(np.mean(values["final_qpos_distance"])),
            "mean_min_qpos_distance": float(np.mean(values["min_qpos_distance"])),
        }
        for name, values in arrays.items()
    }
    return {
        "schema": "reacher_model_cost_fusion_paired_comparison.v1",
        "protocol": {
            "no_dynamics_report": str(no_dynamics_path.resolve()),
            "fusion_report": str(fusion_path.resolve()),
            "shuffled_report": str(shuffled_path.resolve()),
            "seeds": reference_seeds,
            "episode_paired": True,
            "intervention_type": intervention_type,
            "intervention_value": intervention_value,
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "test_outcome_used_for_parameter_selection": False,
            "parameter_status": "development_frozen_then_fresh_root_evaluation",
        },
        "point_estimates": point_estimates,
        "paired": {
            "fusion_minus_no_dynamics": _paired(
                arrays["no_dynamics"], arrays["fusion"], draws=draws, seed=seed
            ),
            "shuffled_minus_no_dynamics": _paired(
                arrays["no_dynamics"], arrays["shuffled"], draws=draws, seed=seed + 10
            ),
            "fusion_minus_shuffled": _paired(
                arrays["shuffled"], arrays["fusion"], draws=draws, seed=seed + 20
            ),
        },
        "runtime_diagnostics": {
            name: report["frozen_lewm"].get("runtime_diagnostics", {})
            for name, report in reports.items()
        },
        "claim_boundary": (
            "fresh Reacher control testing whether frozen predicted-to-goal cost adds "
            "candidate-paired value beyond source-action repair; an intervention gain "
            "is not accepted unless it also beats the shuffled-cost placebo"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-dynamics", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    report = compare(
        args.no_dynamics,
        args.fusion,
        args.shuffled,
        draws=args.draws,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
