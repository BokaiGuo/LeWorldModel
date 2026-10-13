"""Paired summary for learned-adapter contact routing and gate ablations.

The input runs use the same seeds, held-out cases, candidate bank, and
one-step learned-adapter planner.  Only the contact gate is changed.  This
summary keeps route reacquisition, block geometry, orientation, and strict
task success as separate estimands; it is not a visual-LeWM result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", nargs=2, required=True)
    parser.add_argument("--nogate", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261006)
    return parser.parse_args()


def load(path: str) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    assert report["protocol"]["same_candidate_bank_seed_schedule"]
    assert report["sanity_checks"]["all_finite"]
    return report


def rows(report: dict[str, Any], model: str) -> list[dict[str, Any]]:
    result = [row for row in report["rows"] if row["model"] == model]
    if len(result) != int(report["protocol"]["episodes"]):
        raise ValueError(f"unexpected row count for {model}: {len(result)}")
    return result


def value(row: dict[str, Any], metric: str) -> float:
    if metric == "success":
        return float(bool(row["success"]))
    if metric == "route_reacquisitions":
        return float(len(row.get("route_info", [])))
    return float(row[metric])


def paired_bootstrap(delta: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "estimate": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(len(delta)),
    }


def main() -> None:
    args = parse_args()
    gate = [load(path) for path in args.gate]
    nogate = [load(path) for path in args.nogate]
    models = ("block_only", "block_only_specialist")
    metrics = ("success", "block_xy_distance", "angle_error", "agent_xy_distance", "route_reacquisitions")
    comparisons: dict[str, Any] = {}
    for model_index, model in enumerate(models):
        comparisons[model] = {}
        for metric_index, metric in enumerate(metrics):
            gate_values = np.asarray(
                [value(row, metric) for report in gate for row in rows(report, model)],
                dtype=np.float64,
            )
            nogate_values = np.asarray(
                [value(row, metric) for report in nogate for row in rows(report, model)],
                dtype=np.float64,
            )
            if len(gate_values) != len(nogate_values):
                raise ValueError("gate and no-gate runs are not matched")
            comparisons[model][metric] = {
                "gate_mean": float(gate_values.mean()),
                "nogate_mean": float(nogate_values.mean()),
                "gate_minus_nogate": paired_bootstrap(
                    gate_values - nogate_values,
                    args.bootstrap_draws,
                    args.seed + 100 * model_index + metric_index,
                ),
            }
    report = {
        "protocol": {
            "gate_runs": args.gate,
            "nogate_runs": args.nogate,
            "models": list(models),
            "n_per_condition": 16,
            "paired_unit": "episode index within each seed run",
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
        },
        "comparisons": comparisons,
        "claim_boundary": (
            "Privileged simulator-state learned-adapter ablation. The route and gate arms do not use frozen LeWM pixels or future simulator states; "
            "the result diagnoses contact acquisition and model-predicted block updates, not general online planning or causal physical control."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "learned_contact_repair_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# Learned-adapter contact repair ablation",
        "",
        "Gate minus no-gate paired bootstrap summaries over 16 matched cases.",
        "",
        "| model | metric | gate | no gate | gate - no gate (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in models:
        for metric in metrics:
            item = comparisons[model][metric]
            delta = item["gate_minus_nogate"]
            lines.append(
                "| %s | %s | %.4f | %.4f | %.4f [%.4f, %.4f] |"
                % (
                    model,
                    metric,
                    item["gate_mean"],
                    item["nogate_mean"],
                    delta["estimate"],
                    delta["ci95_low"],
                    delta["ci95_high"],
                )
            )
    lines += ["", report["claim_boundary"], ""]
    (output / "learned_contact_repair_summary.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
