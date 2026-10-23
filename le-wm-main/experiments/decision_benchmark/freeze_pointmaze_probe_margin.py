"""Freeze probe margins from an independent pre-calibration design panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pointmaze.probe_margin_design_freeze.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(report: dict[str, Any], method: str, quantiles: list[float]) -> dict[str, Any]:
    values: list[float] = []
    roots: set[int] = set()
    for row in report.get("rows", []):
        if row.get("method") != method:
            continue
        root = int(row["episode_id"])
        decisions = [
            event for event in row.get("active_probe_trace", [])
            if isinstance(event, dict)
            and bool(event.get("probe_decision"))
            and event.get("probe_conditioned_visual_margin") is not None
        ]
        if len(decisions) != 1:
            raise ValueError("each design root requires exactly one visual-margin decision")
        value = float(decisions[0]["probe_conditioned_visual_margin"])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("design margins must be finite and non-negative")
        roots.add(root)
        values.append(value)
    if len(values) < 8 or len(roots) != len(values):
        raise ValueError("design panel requires at least eight unique local-method roots")
    if not quantiles or any(not np.isfinite(q) or q <= 0.0 or q >= 1.0 for q in quantiles):
        raise ValueError("quantiles must lie strictly between zero and one")
    ordered = np.asarray(values, dtype=np.float64)
    thresholds = []
    for quantile in sorted(set(float(q) for q in quantiles)):
        thresholds.append({
            "label": f"q{int(round(100 * quantile)):02d}",
            "quantile": quantile,
            "margin": float(np.quantile(ordered, quantile, method="linear")),
        })
    return {
        "schema": SCHEMA,
        "local_method": method,
        "design_root_count": len(values),
        "design_roots": sorted(roots),
        "margin_min": float(np.min(ordered)),
        "margin_max": float(np.max(ordered)),
        "thresholds": thresholds,
        "claim_boundary": (
            "Margins are frozen from a root-disjoint gate-design panel before "
            "calibration selection and outcome evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-method", required=True)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite immutable probe-margin freeze")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    frozen = freeze(report, str(args.local_method), list(args.quantiles))
    frozen["source_report"] = str(args.input.resolve())
    frozen["source_report_sha256"] = _sha256(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "thresholds": frozen["thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
