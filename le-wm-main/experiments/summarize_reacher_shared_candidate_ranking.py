"""Aggregate predeclared Reacher shared-candidate ranking conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "reacher_shared_candidate_ranking.v1":
        raise ValueError(f"{path}: unexpected schema")
    if not payload.get("protocol", {}).get("root_disjoint", False):
        raise ValueError(f"{path}: root-disjoint protocol is required")
    return payload


def _ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = np.empty(4000, dtype=np.float64)
    for index in range(len(samples)):
        samples[index] = generator.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _metrics(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    inversion = np.asarray([row["action_ranking_inversion"] for row in rows], dtype=np.float64)
    task_error = np.asarray([row["multistep_task_error"] for row in rows], dtype=np.float64)
    angle = np.asarray([row["angle_closure"] for row in rows], dtype=np.float64)
    regret = np.asarray([row["planning_regret"] for row in rows], dtype=np.float64)
    return {
        "n": int(len(rows)),
        "top1_accuracy": float(np.mean([row["top1_hit"] for row in rows])),
        "top4_accuracy": float(np.mean([row["top4_hit"] for row in rows])),
        "action_ranking_inversion": {"estimate": float(inversion.mean()), "ci95": _ci(inversion, seed + 1)},
        "multistep_task_error": {"estimate": float(task_error.mean()), "ci95": _ci(task_error, seed + 2)},
        "angle_closure": {"estimate": float(angle.mean()), "ci95": _ci(angle, seed + 3)},
        "planning_regret": {"estimate": float(regret.mean()), "ci95": _ci(regret, seed + 4)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--condition-target-delta",
        type=float,
        nargs=2,
        action="append",
        required=True,
        metavar=("DELTA_Q1", "DELTA_Q2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261390)
    args = parser.parse_args()
    if len(args.input) < 2 or len(args.condition_target_delta) != len(args.input):
        raise ValueError("provide one explicit target geometry for every input and at least two conditions")
    payloads = [_read(path) for path in args.input]
    rows = []
    conditions = []
    for path, payload, target_delta in zip(args.input, payloads, args.condition_target_delta):
        recorded_target = payload["protocol"].get("target_delta")
        if recorded_target is not None and not np.allclose(recorded_target, target_delta):
            raise ValueError(f"{path}: explicit target geometry disagrees with artifact protocol")
        conditions.append({"source": str(path), "target_delta": target_delta, "metrics": _metrics(payload["roots"], args.seed)})
        rows.extend(payload["roots"])
    aggregate = {
        "schema": "reacher_shared_candidate_ranking.aggregate.v1",
        "protocol": {
            "condition_count": len(conditions),
            "root_disjoint_within_condition": True,
            "candidate_bank_shared": True,
            "checkpoint_frozen": True,
            "runtime_state_access": False,
            "test_planning_used_for_selection": False,
            "device": "cuda",
        },
        "conditions": conditions,
        "pooled": _metrics(rows, args.seed + 100),
        "contact_retention": {"status": "not_applicable", "reason": "Reacher is non-contact"},
        "claim_boundary": (
            "Pooled target-geometry action-ranking diagnostic. Target conditions are "
            "reported separately; pooling does not establish universal transfer."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": aggregate["pooled"], "conditions": conditions}, indent=2))


if __name__ == "__main__":
    main()
