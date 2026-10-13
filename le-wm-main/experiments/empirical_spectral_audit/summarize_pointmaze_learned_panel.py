"""Summarize the repaired PointMaze learned-adapter panel.

The unit of the descriptive table is a checkpoint record, while uncertainty
is reported at the adapter-seed level after averaging over training stages.
This prevents the five snapshots of one seed from being presented as five
independent replications.  The script never selects a checkpoint from the
held-out outcome: it only recomputes the predeclared fixed-horizon-vs-
prediction-only consequence and reports the result.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from diagnose_independent_planning_failure import diagnose


def _base_arch(record: dict[str, Any]) -> tuple[str, str]:
    group = str(record.get("group", ""))
    match = re.search(r"/learned/([^/]+)/seed_", group)
    if match is None:
        raise ValueError(f"cannot parse base/architecture from {group!r}")
    name = match.group(1)
    if "_" not in name:
        return name, "unknown"
    base, architecture = name.split("_", 1)
    return base, architecture


def _seed(record: dict[str, Any]) -> str:
    match = re.search(r"seed_([^/]+)", str(record.get("group", "")))
    return match.group(1) if match else "unknown"


def _bootstrap(values: np.ndarray, seed: int, draws: int = 10000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
    }


def _sign_flip(values: np.ndarray, seed: int, draws: int = 20000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "p_one_sided": None, "positive_gate": False}
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    null = (rng.choice([-1.0, 1.0], size=(draws, values.size)) * values).mean(axis=1)
    p_value = float((1.0 + np.sum(null >= observed - 1e-15)) / (draws + 1.0))
    return {
        "n": int(values.size),
        "mean": observed,
        "p_one_sided": p_value,
        "positive_gate": bool(observed > 0.0 and p_value < 0.05),
    }


def summarize(record_root: Path) -> dict[str, Any]:
    paths = sorted(record_root.rglob("checkpoint_record.json"))
    if not paths:
        raise ValueError(f"no checkpoint_record.json under {record_root}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        outer = json.loads(path.read_text())
        record = outer.get("record", outer)
        report = diagnose(path)
        base, architecture = _base_arch(record)
        stage = int(record["model_factors"]["training_stage"])
        seed = _seed(record)
        gain = float(report["fixed_horizon"]["1"]["mean_gain_vs_diagnostic"])
        conservative = record.get("heldout", {}).get("planning_details_conservative")
        conservative_risk = record.get("heldout", {}).get("planning_risk_conservative")
        conservative_gain = None
        if conservative is not None and conservative_risk is not None:
            conservative_gain = float(conservative["fixed_h1_loss"] - conservative_risk)
        rows.append({
            "path": str(path.resolve()),
            "base": base,
            "architecture": architecture,
            "seed": seed,
            "stage": stage,
            "planning_gain_fixed_h1_minus_diagnostic": gain,
            "planning_gain_fixed_h1_minus_conservative": conservative_gain,
            "prediction_mse": float(record["diagnostics"]["prediction_mse"]),
            "action_inverse_risk": float(record["model_factors"]["action_inverse_risk"]),
            "diagnostic_risk": float(report["diagnostic_loss_mean"]),
            "train_count": int(report["split_sizes"]["train"]),
            "calibration_count": int(report["split_sizes"]["calibration"]),
            "outcome_count": int(report["split_sizes"]["outcome"]),
        })

    by_stage: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stage[(row["base"], row["architecture"], row["stage"])].append(row)

    stage_summary = []
    for (base, architecture, stage), group in sorted(by_stage.items()):
        gains = np.asarray([row["planning_gain_fixed_h1_minus_diagnostic"] for row in group])
        stage_summary.append({
            "base": base,
            "architecture": architecture,
            "stage": stage,
            "seed_count": len({row["seed"] for row in group}),
            "gain": _bootstrap(gains, 20261013 + stage),
            "sign_flip": _sign_flip(gains, 20262013 + stage),
            "prediction_mse_mean": float(np.mean([row["prediction_mse"] for row in group])),
            "action_inverse_risk_mean": float(np.mean([row["action_inverse_risk"] for row in group])),
        })

    # Cluster by adapter seed after averaging its stages.  Stages are snapshots
    # of the same optimization trajectory and must not be counted as fresh
    # replications in the architecture comparison.
    by_seed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_seed[(row["base"], row["architecture"], row["seed"])].append(
            row["planning_gain_fixed_h1_minus_diagnostic"]
        )
    cluster_summary = []
    for (base, architecture, seed), values in sorted(by_seed.items()):
        cluster_summary.append({
            "base": base,
            "architecture": architecture,
            "seed": seed,
            "mean_gain_over_stages": float(np.mean(values)),
            "stage_count": len(values),
        })

    by_family: dict[tuple[str, str], list[float]] = defaultdict(list)
    conservative_by_seed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in cluster_summary:
        by_family[(row["base"], row["architecture"])].append(row["mean_gain_over_stages"])
    for row in rows:
        if row["planning_gain_fixed_h1_minus_conservative"] is not None:
            conservative_by_seed[(row["base"], row["architecture"], row["seed"])].append(
                row["planning_gain_fixed_h1_minus_conservative"]
            )
    conservative_family_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (base, architecture, seed_key), values in conservative_by_seed.items():
        conservative_family_values[(base, architecture)].append(float(np.mean(values)))
    family_summary = []
    for (base, architecture), values in sorted(by_family.items()):
        values_array = np.asarray(values)
        family_summary.append({
            "base": base,
            "architecture": architecture,
            "seed_count": len(values),
            "cluster_bootstrap_gain": _bootstrap(values_array, 20263013 + len(family_summary)),
            "cluster_sign_flip": _sign_flip(values_array, 20264013 + len(family_summary)),
            "conservative_cluster_bootstrap_gain": _bootstrap(
                np.asarray(conservative_family_values.get((base, architecture), [])),
                20265013 + len(family_summary),
            ),
            "conservative_cluster_sign_flip": _sign_flip(
                np.asarray(conservative_family_values.get((base, architecture), [])),
                20266013 + len(family_summary),
            ),
        })

    return {
        "schema": "pointmaze_learned_panel_summary.v1",
        "protocol": {
            "record_root": str(record_root.resolve()),
            "heldout_outcome_used_for_selection": False,
            "gain_definition": "fixed h=1 outcome loss minus prediction-only diagnostic outcome loss",
            "uncertainty_unit": "adapter seed after averaging optimization stages",
            "stage_as_replication": False,
            "outcome_role": "independent consequence scoring only",
        },
        "record_count": len(rows),
        "rows": rows,
        "stage_summary": stage_summary,
        "seed_clusters": cluster_summary,
        "family_summary": family_summary,
        "claim_boundary": (
            "This is a PointMaze architecture-conditioned panel summary. It can "
            "diagnose a reproducible representation condition, but it does not "
            "establish cross-environment or universal transfer."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PointMaze learned-adapter panel summary",
        "",
        "Positive gain means the prediction-only diagnostic has lower held-out consequence loss than fixed h=1.",
        "",
        "| Base | Adapter | Seeds | Raw gain mean [95% CI] | Conservative gain mean [95% CI] | Raw gate | Conservative gate |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for item in report["family_summary"]:
        gain = item["cluster_bootstrap_gain"]
        test = item["cluster_sign_flip"]
        text = "NA" if gain["mean"] is None else f"{gain['mean']:.6g} [{gain['ci95_low']:.6g}, {gain['ci95_high']:.6g}]"
        conservative = item["conservative_cluster_bootstrap_gain"]
        conservative_text = "NA" if conservative["mean"] is None else f"{conservative['mean']:.6g} [{conservative['ci95_low']:.6g}, {conservative['ci95_high']:.6g}]"
        lines.append(f"| {item['base']} | {item['architecture']} | {item['seed_count']} | {text} | {conservative_text} | {test['positive_gate']} | {item['conservative_cluster_sign_flip']['positive_gate']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The uncertainty unit is adapter seed; optimization stages are snapshots, not independent replications.",
        "The conservative column is a calibration-only fail-closed gate; it may fall back to h=1 and is not pooled with the raw diagnostic result.",
        "A negative DINO family result remains a representation-conditioned failure and is not pooled away by the positive JEPA family.",
        "",
        "## Claim boundary",
        "",
        report["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.record_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"record_count": report["record_count"], "family_summary": report["family_summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
