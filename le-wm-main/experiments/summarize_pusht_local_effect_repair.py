"""Summarize the matched learned local-effect and hybrid ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", nargs="+", required=True)
    parser.add_argument("--hybrid", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261006)
    return parser.parse_args()


def load(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        report = json.loads(Path(path).read_text())
        seed = int(report["protocol"]["seed"])
        for row in report["rows"]:
            rows.append({"seed": seed, **row})
    return rows


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(values)),
    }


def paired(
    rows: list[dict[str, Any]],
    target: str,
    baseline: str,
    metric: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    left = {(row["seed"], row["episode"]): row for row in rows if row["model"] == target}
    right = {(row["seed"], row["episode"]): row for row in rows if row["model"] == baseline}
    keys = sorted(set(left) & set(right))
    values = np.asarray([
        float(left[key][metric]) - float(right[key][metric]) for key in keys
    ])
    return bootstrap(values, draws, seed)


def summarize_group(
    rows: list[dict[str, Any]],
    models: list[str],
    baseline: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, model in enumerate(models):
        subset = [row for row in rows if row["model"] == model]
        result[model] = {
            metric: bootstrap(
                np.asarray([float(row[metric]) if metric != "success" else float(row["success"]) for row in subset]),
                draws,
                seed + index * 100 + offset,
            )
            for offset, metric in enumerate(("success", "block_xy_distance", "angle_error", "agent_xy_distance"))
        }
        if model != baseline:
            result[model]["paired_minus_baseline"] = {
                metric: paired(rows, model, baseline, metric, draws, seed + 1000 + index * 100 + offset)
                for offset, metric in enumerate(("success", "block_xy_distance", "angle_error", "agent_xy_distance"))
            }
    return result


def phase_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        local_counts = [sum(item.get("phase") == "local_effect_orientation" for item in row["trace"]) for row in subset]
        result[model] = {
            "episodes": len(subset),
            "local_effect_decisions": int(sum(local_counts)),
            "episodes_using_local_effect": int(sum(count > 0 for count in local_counts)),
            "task_success": int(sum(bool(row["success"]) for row in subset)),
            "block_pose_success": int(sum(
                row["block_xy_distance"] < 20.0 and row["angle_error"] < np.pi / 9.0
                for row in subset
            )),
        }
    return result


def main() -> None:
    args = parse_args()
    local = load(args.local)
    hybrid = load(args.hybrid)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    local_models = ["state_action", "object_centric", "history", "block_only", "block_only_specialist"]
    hybrid_models = ["block_agent_hybrid"]
    report = {
        "claim_boundary": (
            "Privileged simulator-state learned-adapter diagnostic. Local-effect probes use current-state "
            "one-step simulator interventions; neither group is frozen-LeWM visual planning or general PushT success."
        ),
        "protocol": {
            "local_reports": args.local,
            "hybrid_reports": args.hybrid,
            "n_per_model": len(local) // len(local_models),
            "seeds": sorted({row["seed"] for row in local}),
            "bootstrap_draws": args.draws,
            "paired_unit": "seed and episode index within the local-effect reports",
        },
        "local_effect": summarize_group(local, local_models, "state_action", args.draws, args.seed),
        "hybrid": summarize_group(hybrid, hybrid_models, "state_action", args.draws, args.seed + 5000),
        "phase_audit": phase_audit(local),
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(float(row[metric]))
                for row in local + hybrid
                for metric in ("block_xy_distance", "angle_error", "agent_xy_distance")
            )),
            "local_models_present": sorted({row["model"] for row in local}) == sorted(local_models),
            "hybrid_model_present": "block_agent_hybrid" in {row["model"] for row in hybrid},
            "same_local_case_count_per_model": len({sum(row["model"] == model for row in local) for model in local_models}) == 1,
        },
    }
    (output / "local_effect_repair_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Learned local-effect and hybrid repair summary",
        "",
        "| group | model | success | block XY | angle | agent XY |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group, values in (("local-effect", report["local_effect"]), ("hybrid", report["hybrid"])):
        for model, item in values.items():
            lines.append(
                "| %s | %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |"
                % (
                    group, model,
                    item["success"]["estimate"], item["success"]["ci95_low"], item["success"]["ci95_high"],
                    item["block_xy_distance"]["estimate"], item["block_xy_distance"]["ci95_low"], item["block_xy_distance"]["ci95_high"],
                    item["angle_error"]["estimate"], item["angle_error"]["ci95_low"], item["angle_error"]["ci95_high"],
                    item["agent_xy_distance"]["estimate"], item["agent_xy_distance"]["ci95_low"], item["agent_xy_distance"]["ci95_high"],
                )
            )
    lines += ["", report["claim_boundary"], ""]
    (output / "local_effect_repair_summary.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
