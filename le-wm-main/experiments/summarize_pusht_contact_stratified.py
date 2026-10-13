"""Paired summary for pooled versus contact-mode-specialist adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261007)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads((args.input / "contact_stratified_paired_intervention.json").read_text())
    indexed = {
        (item["model"], item["stratum"], int(item["horizon"])): item
        for item in report["reports"]
    }
    strata = report["protocol"]["strata"]
    horizons = report["protocol"]["horizons"]
    comparisons = []
    for stratum_index, stratum in enumerate(strata):
        for horizon in horizons:
            pooled = indexed[("contact_geometry", stratum, int(horizon))]
            specialist = indexed[("specialist_geometry", stratum, int(horizon))]
            pooled_rows = pooled["rows"]
            specialist_rows = specialist["rows"]
            if len(pooled_rows) != len(specialist_rows):
                raise ValueError(f"unmatched rows for {stratum} h={horizon}")
            pooled_error = np.asarray([np.mean(row["block_xy_error"]) for row in pooled_rows], dtype=np.float64)
            specialist_error = np.asarray([np.mean(row["block_xy_error"]) for row in specialist_rows], dtype=np.float64)
            pooled_sep = np.asarray([row["separation_abs_error"] for row in pooled_rows], dtype=np.float64)
            specialist_sep = np.asarray([row["separation_abs_error"] for row in specialist_rows], dtype=np.float64)
            comparisons.append({
                "stratum": stratum,
                "horizon": int(horizon),
                "pooled_block_xy_error": bootstrap(pooled_error, args.draws, args.seed + 10 * stratum_index + horizon),
                "specialist_block_xy_error": bootstrap(specialist_error, args.draws, args.seed + 100 + 10 * stratum_index + horizon),
                "specialist_minus_pooled_block_xy_error": bootstrap(specialist_error - pooled_error, args.draws, args.seed + 200 + 10 * stratum_index + horizon),
                "pooled_separation_abs_error": bootstrap(pooled_sep, args.draws, args.seed + 300 + 10 * stratum_index + horizon),
                "specialist_separation_abs_error": bootstrap(specialist_sep, args.draws, args.seed + 400 + 10 * stratum_index + horizon),
                "specialist_minus_pooled_separation_abs_error": bootstrap(specialist_sep - pooled_sep, args.draws, args.seed + 500 + 10 * stratum_index + horizon),
                "true_separation": pooled["metrics"]["true_plus_minus_separation"],
            })
    result = {
        "protocol": {
            "source": str(args.input),
            "paired_unit": "held-out state; mean over +a/-a/zero interventions",
            "draws": args.draws,
            "seed": args.seed,
        },
        "comparisons": comparisons,
        "claim_boundary": (
            "Privileged simulator-state physical-adapter diagnosis. Specialist gains are mode-conditional and do not "
            "establish a universal contact dynamics model, frozen-LeWM causality, or online planning."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "contact_stratified_paired_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = [
        "# Contact-stratified paired summary",
        "",
        "Differences are specialist minus pooled contact-geometry adapter; intervals are paired bootstrap 95% CIs over held-out states.",
        "",
        "| stratum | horizon | pooled XY error | specialist XY error | specialist minus pooled | pooled separation error | specialist separation error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        def fmt(metric: str) -> str:
            value = item[metric]
            return "%.3f [%.3f, %.3f]" % (value["estimate"], value["ci95_low"], value["ci95_high"])
        lines.append(
            "| %s | %d | %s | %s | %s | %s | %s |" % (
                item["stratum"], item["horizon"], fmt("pooled_block_xy_error"),
                fmt("specialist_block_xy_error"), fmt("specialist_minus_pooled_block_xy_error"),
                fmt("pooled_separation_abs_error"), fmt("specialist_separation_abs_error"),
            )
        )
    lines += ["", result["claim_boundary"], ""]
    (args.output / "contact_stratified_paired_summary.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
