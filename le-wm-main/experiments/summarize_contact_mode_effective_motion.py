"""Paired summary for the contact-mode/effective-motion ranking audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "selected_feasible",
    "top3_hit",
    "top8_hit",
    "top3_hit_given_oracle",
    "top8_hit_given_oracle",
    "selected_useful_motion",
    "selected_preserves_contact",
    "selected_rank_by_true_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261011)
    return parser.parse_args()


def bootstrap_difference(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(values.size),
    }


def load_rows(path: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    report = json.loads((Path(path) / "contact_mode_effective_motion_audit.json").read_text())
    by_selector = {item["selector"]: item["rows"] for item in report["reports"]}
    required = {"existing_feasibility", "contact_mode_effective_motion"}
    if set(by_selector) & required != required:
        raise ValueError(f"{path} does not contain both required selectors")
    if len(by_selector["existing_feasibility"]) != len(by_selector["contact_mode_effective_motion"]):
        raise ValueError(f"selector row counts differ in {path}")
    return report["protocol"], by_selector


def summarize_one(path: str, draws: int, seed: int) -> dict[str, Any]:
    protocol, rows = load_rows(path)
    old = rows["existing_feasibility"]
    new = rows["contact_mode_effective_motion"]
    paired = {}
    for index, metric in enumerate(METRICS):
        difference = np.asarray([n[metric] - o[metric] for o, n in zip(old, new)], dtype=np.float64)
        paired[metric] = bootstrap_difference(difference, draws, seed + index)
    return {"input": path, "protocol": protocol, "paired_new_minus_existing": paired}


def pooled_summary(summaries: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    pooled = {}
    for index, metric in enumerate(METRICS):
        values = []
        for summary in summaries:
            protocol, rows = load_rows(summary["input"])
            del protocol
            values.extend(
                n[metric] - o[metric]
                for o, n in zip(rows["existing_feasibility"], rows["contact_mode_effective_motion"])
            )
        pooled[metric] = bootstrap_difference(np.asarray(values), draws, seed + 100 + index)
    return pooled


def main() -> None:
    args = parse_args()
    summaries = [summarize_one(path, args.bootstrap, args.seed + index * 1000) for index, path in enumerate(args.inputs)]
    report = {
        "protocol": {
            "inputs": args.inputs,
            "comparison": "contact_mode_effective_motion minus existing_feasibility",
            "paired_within_each_held_out_group": True,
            "metric_definitions": {
                "selected_feasible": "top-1 actual feasibility",
                "top3_hit": "actual feasible candidate in predicted top-3",
                "top8_hit": "actual feasible candidate in predicted top-8",
                "top3_hit_given_oracle": "top-3 hit conditional on at least one feasible candidate",
                "top8_hit_given_oracle": "top-8 hit conditional on at least one feasible candidate",
                "selected_useful_motion": "selected block displacement >= 0.5 pixels",
                "selected_preserves_contact": "selected successor remains in contact mode",
                "selected_rank_by_true_score": "rank of selected candidate under true successor score; lower is better",
            },
        },
        "per_seed": summaries,
        "pooled_paired_new_minus_existing": pooled_summary(summaries, args.bootstrap, args.seed),
        "claim_boundary": (
            "This is a paired held-out ranking comparison in privileged simulator state. "
            "It does not establish pure learned closed-loop task success, frozen-latent causality, "
            "visual online planning, or universal world modeling."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact_mode_effective_motion_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Contact-mode/effective-motion paired summary",
        "",
        "| metric | pooled new minus existing | 95% CI | n |",
        "|---|---:|---:|---:|",
    ]
    for metric, value in report["pooled_paired_new_minus_existing"].items():
        lines.append(
            "| %s | %.4f | [%.4f, %.4f] | %d |"
            % (metric, value["estimate"], value["ci95_low"], value["ci95_high"], value["n"])
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "contact_mode_effective_motion_summary.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
