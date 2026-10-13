"""Aggregate pure learned PushT closure probes at the seed-cluster level.

This separates the no-successor-verification direct adapter from the stronger
privileged verifier-assisted arm.  A seed is the uncertainty cluster; episode
rows are averaged within seed.  The output is deliberately a gate report, not
a success claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            raise FloatingPointError(f"non-finite value: {value}")


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(sample, 0.025)),
        "ci_high": float(np.quantile(sample, 0.975)),
        "n_seed_clusters": int(values.size),
    }


def _contains_verification(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (key == "simulator_verification_used" and bool(item))
            or _contains_verification(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_verification(item) for item in value)
    return False


def audit(paths: list[Path], model: str = "block_only_torque_readout") -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two seed artifacts are required")
    by_seed: dict[int, list[dict[str, Any]]] = {}
    protocols = []
    for path in paths:
        payload = json.loads(path.read_text())
        _finite(payload)
        protocol = payload["protocol"]
        protocols.append(protocol)
        rows = [row for row in payload["rows"] if row.get("model") == model]
        if not rows:
            raise ValueError(f"{path} has no rows for model {model!r}")
        if protocol.get("direct_goal_conditioned_feasibility") is not True:
            raise ValueError("pure closure audit requires direct goal-conditioned feasibility")
        if protocol.get("planning_objective") != "staged_contact_direct_torque_readout":
            raise ValueError("unexpected planning objective")
        if any(_contains_verification(row) for row in rows):
            raise ValueError("successor verification detected in pure learned rows")
        by_seed[int(protocol["seed"])] = rows
    metrics: dict[str, dict[str, float | int]] = {}
    for key in ("strict_task_success", "block_xy_distance", "angle_error", "agent_xy_distance"):
        seed_means = np.asarray([np.mean([float(row[key]) for row in rows]) for rows in by_seed.values()])
        metrics[key] = _bootstrap(seed_means, 20261008 + len(metrics))
    successes = int(sum(sum(bool(row["strict_task_success"]) for row in rows) for rows in by_seed.values()))
    episodes = int(sum(len(rows) for rows in by_seed.values()))
    report = {
        "protocol": {
            "model": model,
            "seed_count": len(by_seed),
            "episode_count": episodes,
            "independent_unit": "seed cluster; episodes averaged within seed",
            "successor_verification": False,
            "training_protocols": sorted({str(protocol.get("training_protocol")) for protocol in protocols}),
            "planning_objectives": sorted({str(protocol.get("planning_objective")) for protocol in protocols}),
        },
        "metrics": metrics,
        "success_count": successes,
        "success_denominator": episodes,
        "seed_rows": [
            {
                "seed": seed,
                "episodes": len(rows),
                "strict_successes": int(sum(bool(row["strict_task_success"]) for row in rows)),
                "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])),
                "mean_angle_error": float(np.mean([row["angle_error"] for row in rows])),
                "mean_agent_xy_distance": float(np.mean([row["agent_xy_distance"] for row in rows])),
                "verification_used_any": bool(any(_contains_verification(row) for row in rows)),
            }
            for seed, rows in sorted(by_seed.items())
        ],
        "sanity_checks": {
            "all_finite": True,
            "all_seeds_no_successor_verification": True,
            "same_model": True,
        },
        "claim_boundary": [
            "This is a pure learned privileged simulator-state adapter probe, not visual LeWM planning.",
            "The direct adapter does not use simulator successor verification at selection time.",
            "Binary closure success is reported as observed; no success is converted into a universal physical-causality claim.",
        ],
    }
    _finite(report)
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pure learned PushT closure audit",
        "",
        "The selected direct adapter uses no simulator successor verification at action selection. Seed clusters, not episodes, are the uncertainty units.",
        "",
        f"- Strict success: {report['success_count']}/{report['success_denominator']}.",
        "",
        "| Metric | Mean | 95% seed-cluster bootstrap CI |",
        "| --- | ---: | ---: |",
    ]
    for name, value in report["metrics"].items():
        lines.append(f"| {name} | {value['mean']:.6g} | [{value['ci_low']:.6g}, {value['ci_high']:.6g}] |")
    lines.extend(["", "| Seed | Episodes | Strict successes | Mean block XY | Mean angle | Mean agent XY |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in report["seed_rows"]:
        lines.append(f"| {row['seed']} | {row['episodes']} | {row['strict_successes']} | {row['mean_block_xy_distance']:.6g} | {row['mean_angle_error']:.6g} | {row['mean_agent_xy_distance']:.6g} |")
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.artifact)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pure_learned_closure_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "pure_learned_closure_audit.md").write_text(markdown(report))
    print(json.dumps({"success_count": report["success_count"], "success_denominator": report["success_denominator"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
