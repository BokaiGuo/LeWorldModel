"""Fail-closed, root-disjoint aggregation of two immutable P3 CUDA panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_mean(values: np.ndarray, *, seed: int, draws: int = 20_000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def summarize(primary: dict[str, Any], independent: dict[str, Any]) -> dict[str, Any]:
    """Aggregate paired spectral-minus-immediate regrets without tuning a protocol."""

    failures: list[str] = []
    expected_schema = "sra_p3_fixed_tree_horizon_expansion.v1"
    for label, report in (("primary", primary), ("independent", independent)):
        if report.get("schema") != expected_schema or report.get("status") != "mechanism_ready":
            failures.append(f"{label} report is not a valid P3 receipt")
        if not all(report.get("gates", {}).values()):
            failures.append(f"{label} learned-model gate failed")
    first, second = primary.get("inputs", {}), independent.get("inputs", {})
    for key in ("horizons", "model_batch_size", "checkpoint_sha256", "physical_position_backend"):
        if first.get(key) != second.get(key):
            failures.append(f"protocol input mismatch: {key}")
    first_roots, second_roots = first.get("roots", []), second.get("roots", [])
    if not first_roots or not second_roots or set(first_roots) & set(second_roots):
        failures.append("panels are not root-disjoint")

    horizons = first.get("horizons", [])
    aggregated: dict[str, Any] = {}
    for horizon in horizons:
        key = str(horizon)
        primary_values = np.asarray([row["horizons"][key]["spectral_regret"] - row["horizons"][key]["immediate_regret"] for row in primary.get("rows", [])], dtype=np.float64)
        independent_values = np.asarray([row["horizons"][key]["spectral_regret"] - row["horizons"][key]["immediate_regret"] for row in independent.get("rows", [])], dtype=np.float64)
        values = np.concatenate([primary_values, independent_values])
        if len(values) != len(first_roots) + len(second_roots) or not np.isfinite(values).all():
            failures.append(f"invalid paired regrets at H={horizon}")
            continue
        low, high = _bootstrap_mean(values, seed=2026102400 + int(horizon))
        aggregated[key] = {
            "metric": "spectral_regret_minus_immediate_regret; lower is better",
            "primary_panel_mean": float(primary_values.mean()),
            "independent_panel_mean": float(independent_values.mean()),
            "pooled_root_mean": float(values.mean()),
            "bootstrap_95pct_ci": [low, high],
            "root_count": int(len(values)),
        }
    return {
        "schema": "sra_p3_root_disjoint_confirmation.v1",
        "status": "pass" if not failures else "invalid",
        "scope": "fixed-tree PointMaze diagnostic only; no RGB closed-loop or cross-checkpoint claim",
        "gates": {
            "both_reports_valid": not any("valid P3 receipt" in item or "gate failed" in item for item in failures),
            "same_checkpoint_and_physical_contract": not any(item.startswith("protocol input mismatch") for item in failures),
            "root_disjoint_panels": "panels are not root-disjoint" not in failures,
            "finite_complete_paired_regrets": not any(item.startswith("invalid paired regrets") for item in failures),
        },
        "horizons": aggregated,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    result = summarize(_load(args.primary), _load(args.independent))
    result.update({
        "primary_report": str(args.primary), "primary_report_sha256": _sha256(args.primary),
        "independent_report": str(args.independent), "independent_report_sha256": _sha256(args.independent),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise RuntimeError("P3 root-disjoint confirmation failed")


if __name__ == "__main__":
    main()
