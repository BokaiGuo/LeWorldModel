"""Summarize matched live PushT planning runs across action normalizations.

The input runs must share the same policy set, episode seed, and planning
protocol.  The script keeps success, physical distance, return, and clipping
as separate estimands and reports paired policy-minus-zero bootstrap intervals.
It deliberately labels the result as a CPU pilot when the number of episodes
is small; aggregation does not promote a pilot to a powered task claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("success", "final_distance", "block_xy_distance", "return", "clip_fraction")


def _stable_offset(*parts: str) -> int:
    """Return a process-independent seed offset for a metric cell."""
    return sum((index + 1) * ord(char) for index, char in enumerate("|".join(parts))) % 100000


def _load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _bootstrap(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired bootstrap requires at least two episodes")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_episodes": int(values.size),
    }


def _episode_metric(row: dict[str, Any], metric: str) -> float:
    if metric == "success":
        return float(bool(row["success"]))
    if metric == "clip_fraction":
        return float(row["clip_fraction"] / max(int(row["steps"]), 1))
    return float(row[metric])


def summarize(run_paths: list[Path], *, seed: int = 991, draws: int = 2000) -> dict[str, Any]:
    if len(run_paths) < 2:
        raise ValueError("provide at least two normalization runs")
    reports = [_load(path) for path in run_paths]
    protocols = [report["protocol"] for report in reports]
    names = [str(protocol["action_normalization"]) for protocol in protocols]
    if len(set(names)) != len(names):
        raise ValueError("normalization runs must have distinct action_normalization values")

    reference = protocols[0]
    comparable_keys = ("episodes", "seed", "horizon", "receding_horizon", "action_block", "samples", "cem_steps", "topk", "policies")
    for protocol in protocols[1:]:
        mismatches = [key for key in comparable_keys if protocol.get(key) != reference.get(key)]
        if mismatches:
            raise ValueError(f"runs are not protocol-matched: {mismatches}")

    by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, report in zip(names, reports):
        by_run[name] = {item["policy"]: item["episodes"] for item in report["reports"]}
    policies = list(by_run[names[0]])
    if any(set(by_run[name]) != set(policies) for name in names):
        raise ValueError("normalization runs must contain the same policies")

    n = len(by_run[names[0]][policies[0]])
    episode_ids = [int(row["episode"]) for row in by_run[names[0]][policies[0]]]
    for name in names:
        for policy in policies:
            rows = by_run[name][policy]
            ids = [int(row["episode"]) for row in rows]
            if ids != episode_ids:
                raise ValueError(f"episode pairing mismatch for {name}/{policy}")

    result: dict[str, Any] = {
        "protocol": {
            "normalizations": names,
            "matched_seed": int(reference["seed"]),
            "episodes": n,
            "paired_unit": "episode within fixed seed",
            "bootstrap_draws": draws,
            "status": "partial_cpu_pilot" if n < 16 else "final_protocol_matched",
            "source_runs": [str(path) for path in run_paths],
        },
        "summaries": {},
        "paired_vs_zero": {},
        "sanity_checks": {
            "same_comparable_protocol": True,
            "same_episode_ids": True,
            "finite_all": True,
        },
    }
    for name in names:
        result["summaries"][name] = {}
        result["paired_vs_zero"][name] = {}
        zero_rows = by_run[name].get("zero")
        for policy in policies:
            result["summaries"][name][policy] = {}
            for metric in METRICS:
                values = np.asarray([_episode_metric(row, metric) for row in by_run[name][policy]], dtype=np.float64)
                result["summaries"][name][policy][metric] = _bootstrap(values, seed + _stable_offset(name, policy, metric), draws)
                result["sanity_checks"]["finite_all"] &= bool(np.isfinite(values).all())
            if policy != "zero" and zero_rows is not None:
                result["paired_vs_zero"][name][policy] = {}
                for metric in METRICS:
                    values = np.asarray([
                        _episode_metric(row, metric) - _episode_metric(zrow, metric)
                        for row, zrow in zip(by_run[name][policy], zero_rows)
                    ], dtype=np.float64)
                    result["paired_vs_zero"][name][policy][metric] = _bootstrap(values, seed + _stable_offset(name, policy, "delta", metric), draws)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    report = summarize([Path(path) for path in args.runs], seed=args.seed, draws=args.draws)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    md = [
        "# Matched online PushT planning normalization matrix",
        "",
        "All rows use the same seeds and planning protocol; entries are episode-level paired bootstrap summaries.",
        "",
        "| normalization | policy | success | final distance | block XY distance | return | clip fraction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for normalization, policies in report["summaries"].items():
        for policy, metrics in policies.items():
            md.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.4f |" % (
                normalization,
                policy,
                metrics["success"]["estimate"],
                metrics["final_distance"]["estimate"],
                metrics["block_xy_distance"]["estimate"],
                metrics["return"]["estimate"],
                metrics["clip_fraction"]["estimate"],
            ))
    md += ["", "Status: ``%s``." % report["protocol"]["status"], ""]
    output.with_suffix(".md").write_text("\n".join(md))
    print(json.dumps({"output": str(output), "protocol": report["protocol"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
