"""Audit pre-registered task/action metric variants on frozen checkpoints.

The runner consumes records produced by ``audit_causal_visual_checkpoint``.
Every metric is fit or declared on the diagnostic split; held-out planning is
used only for the final audit.  No metric weight is selected from planning.
This is an attribution experiment for the previous task-metric negative
result, not a license to report the best variant as a universal certificate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = (
    "default_task_space",
    "agent_weight_low",
    "agent_weight_high",
    "block_xy_only",
    "orientation_only",
    "block_xy_plus_orientation",
    "action_effect",
    "stable_block_xy_focus",
    "stable_orientation_focus",
    "stable_block_xy_plus_orientation",
    "stable_action_effect",
)
DIAGNOSTICS = (
    "task_metric_alignment",
    "task_metric_commutator",
    "task_metric_schur_offdiagonal_energy",
    "task_metric_departure_from_normality",
    "task_metric_transient_amplification",
    "task_metric_spectral_radius",
)


def _bootstrap(values: list[float], seed: int, draws: int = 4000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_groups": 0}
    rng = np.random.default_rng(seed)
    sample = array[rng.integers(0, array.size, size=(draws, array.size))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n_groups": int(array.size),
    }


def _paired_summary(values: list[float], seed: int) -> dict[str, Any]:
    """Bootstrap a paired per-group difference; negative means better."""
    return _bootstrap(values, seed)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def _spearman(metric: np.ndarray, loss: np.ndarray) -> float | None:
    if len(metric) < 3 or np.ptp(metric) == 0.0 or np.ptp(loss) == 0.0:
        return None
    return float(np.corrcoef(_rank(metric), _rank(loss))[0, 1])


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("record_schema") != "real_checkpoint_selection.v1":
        raise ValueError(f"{path} has incompatible record schema")
    record = payload["record"]
    variants = record.get("diagnostics", {}).get("task_metric_variants")
    if not isinstance(variants, dict) or any(name not in variants for name in VARIANTS):
        raise ValueError(f"{path} lacks complete pre-registered task metric variants")
    for name in VARIANTS:
        for field in DIAGNOSTICS:
            value = variants[name].get(field)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"{path} has invalid {name}/{field}")
    return record


def audit(records: list[dict[str, Any]], *, bootstrap_seed: int = 20261011) -> dict[str, Any]:
    if not records:
        raise ValueError("no records supplied")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[str(record["group"])].append(record)
    regret: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, list[float]] = defaultdict(list)
    correlations: dict[str, list[float]] = defaultdict(list)
    paired_random: dict[str, list[float]] = defaultdict(list)
    paired_mse: dict[str, list[float]] = defaultdict(list)
    condition_numbers: dict[str, list[float]] = defaultdict(list)
    group_reports: list[dict[str, Any]] = []
    for group, candidates in sorted(by_group.items()):
        if len(candidates) < 2:
            continue
        losses = np.asarray([float(row["heldout"]["planning_loss"]) for row in candidates])
        oracle_index = int(np.argmin(losses))
        oracle_loss = float(losses[oracle_index])
        diagnostics = {
            f"{name}/{field}": np.asarray([
                float(row["diagnostics"]["task_metric_variants"][name][field])
                for row in candidates
            ]) for name in VARIANTS for field in DIAGNOSTICS
        }
        diagnostics.update({
            "legacy/alignment": np.asarray([
                float(row["diagnostics"]["alignment"]) for row in candidates
            ]),
            "legacy/transient_amplification": np.asarray([
                float(row["diagnostics"]["transient_amplification"]) for row in candidates
            ]),
            "prediction_mse": np.asarray([
                float(row["diagnostics"]["prediction_mse"]) for row in candidates
            ]),
            "variance": np.asarray([
                -float(row["diagnostics"]["latent_variance"]) for row in candidates
            ]),
            "spectrum_radius": np.asarray([
                abs(float(row["diagnostics"]["spectral_radius"]) - 1.0) for row in candidates
            ]),
        })
        for candidate in candidates:
            metadata = candidate["diagnostics"].get("task_metric_variant_metadata", {})
            for name in VARIANTS:
                matrix = np.asarray(metadata[name]["matrix"], dtype=np.float64)
                condition_numbers[name].append(float(np.linalg.cond(matrix)))
        # Equal-rank composite is declared before outcome inspection.  It is
        # reported only as a diagnostic, never promoted by this script.
        composite_values = {}
        for name in VARIANTS:
            stack = np.stack([_rank(diagnostics[f"{name}/{field}"]) for field in DIAGNOSTICS])
            composite_values[f"{name}/geometry_composite"] = stack.mean(axis=0)
        diagnostics.update(composite_values)
        selectors = {name: int(np.argmin(values)) for name, values in diagnostics.items()}
        selectors["random_expected"] = -1
        group_row: dict[str, Any] = {
            "group": group,
            "candidate_count": len(candidates),
            "oracle_checkpoint": candidates[oracle_index]["checkpoint"],
            "oracle_loss": oracle_loss,
            "selectors": {},
            "within_group_spearman": {},
        }
        for name, values in diagnostics.items():
            selected = selectors[name]
            selected_loss = float(losses[selected])
            regret[name].append(selected_loss - oracle_loss)
            wins[name].append(float(selected == oracle_index))
            correlation = _spearman(values, losses)
            if correlation is not None:
                correlations[name].append(correlation)
            group_row["selectors"][name] = {
                "checkpoint": candidates[selected]["checkpoint"],
                "planning_loss": selected_loss,
                "regret": selected_loss - oracle_loss,
                "is_oracle": bool(selected == oracle_index),
            }
            group_row["within_group_spearman"][name] = correlation
        random_regret = float(losses.mean() - oracle_loss)
        random_win = float(np.mean(losses == oracle_loss))
        regret["random_expected"].append(random_regret)
        wins["random_expected"].append(random_win)
        group_row["selectors"]["random_expected"] = {
            "planning_loss": float(losses.mean()),
            "regret": random_regret,
            "is_oracle": None,
        }
        for name, selected in selectors.items():
            if name == "random_expected":
                continue
            selected_loss = float(losses[int(selected)])
            paired_random[name].append(selected_loss - float(losses.mean()))
            mse_selected = int(np.argmin(diagnostics["prediction_mse"]))
            paired_mse[name].append(selected_loss - float(losses[mse_selected]))
        group_reports.append(group_row)

    summary = {}
    for index, name in enumerate(sorted(regret)):
        summary[name] = {
            "regret": _bootstrap(regret[name], bootstrap_seed + index),
            "oracle_hit_rate": float(np.mean(wins[name])) if wins[name] else None,
            "spearman_mean": float(np.mean(correlations[name])) if correlations[name] else None,
            "spearman_n_groups": len(correlations[name]),
        }
    paired_comparisons = {
        "vs_random_expected": {
            name: _paired_summary(values, bootstrap_seed + 1000 + index)
            for index, (name, values) in enumerate(sorted(paired_random.items()))
        },
        "vs_prediction_mse": {
            name: _paired_summary(values, bootstrap_seed + 2000 + index)
            for index, (name, values) in enumerate(sorted(paired_mse.items()))
        },
    }
    condition_summary = {
        name: {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
            "n": len(values),
        }
        for name, values in sorted(condition_numbers.items())
    }
    return {
        "protocol": {
            "metric_variants": list(VARIANTS),
            "metric_fit_split": "diagnostic split only",
            "selection_inputs": "task-metric alignment only; no held-out planning access",
            "outcome": "held-out planning loss",
            "grouping": "same architecture, seed, and environment across snapshots",
            "negative_result_policy": "all variants and failed selectors are retained; no post-hoc weight tuning",
        },
        "summary": summary,
        "paired_comparisons": paired_comparisons,
        "metric_condition_number_summary": condition_summary,
        "groups": group_reports,
        "claim_boundary": (
            "This is a pre-registered component attribution on frozen PushT checkpoints. "
            "Metric selection uses diagnostic data only; held-out planning is an outcome, not a tuning input. "
            "A variant can be promoted only if it beats declared baselines across independent "
            "architectures, seeds, and environments; selecting the lowest-regret variant here "
            "would be post-hoc and is not allowed."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pre-registered task/action metric variant audit",
        "",
        "All metric variants were computed from the diagnostic split; planning loss was held out.",
        "",
        "| Variant | Regret mean | 95% CI | Oracle hit rate | Mean within-group Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["summary"].items():
        ci = row["regret"]
        lines.append(
            f"| {name} | {ci['mean']:.6g} | [{ci['ci95_low']:.6g}, {ci['ci95_high']:.6g}] | "
            f"{row['oracle_hit_rate']:.4g} | {row['spearman_mean'] if row['spearman_mean'] is not None else 'NA'} |"
        )
    lines += ["", "## Claim boundary", "", report["claim_boundary"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", type=Path)
    parser.add_argument("--record-root", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = list(args.record or [])
    if args.record_root is not None:
        paths.extend(sorted(args.record_root.glob("*/*/*/checkpoint_record.json")))
    if not paths:
        parser.error("provide --record or --record-root")
    report = audit([_load(path) for path in paths])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "task_metric_variant_selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "task_metric_variant_selection.md").write_text(markdown(report))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
