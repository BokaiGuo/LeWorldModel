"""Summarize frozen temporal-adapter ablations across action normalizations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def summarize(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two normalization reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    if not all(_finite(report) for report in reports):
        raise ValueError("an adapter report contains non-finite values")
    normalizations = [report["protocol"]["normalization_mode"] for report in reports]
    if len(set(normalizations)) != len(normalizations):
        raise ValueError("adapter inputs must use distinct normalization modes")
    reference = reports[0]
    protocol_identity = (
        reference["protocol"]["split_seeds"],
        reference["protocol"]["seeds"],
        reference["protocol"]["weights"],
        reference["source"]["episodes"],
        reference["source"]["windows"],
    )
    for report in reports[1:]:
        current = (
            report["protocol"]["split_seeds"],
            report["protocol"]["seeds"],
            report["protocol"]["weights"],
            report["source"]["episodes"],
            report["source"]["windows"],
        )
        if current != protocol_identity:
            raise ValueError("adapter reports do not share the same split/seed protocol")

    rows: list[dict[str, Any]] = []
    for normalization, report in zip(normalizations, reports):
        for condition, summary in report["summary"].items():
            mode, weight = condition.split(":", 1)
            for horizon, metrics in summary["horizons"].items():
                rows.append(
                    {
                        "normalization": normalization,
                        "mode": mode,
                        "weight": float(weight),
                        "horizon": int(horizon),
                        "mse_mean": float(metrics["mse_mean"]),
                        "delta_vs_none_mean": float(metrics["delta_vs_none_mean"]),
                        "delta_vs_none_ci95": [
                            float(metrics["delta_vs_none_ci95"][0]),
                            float(metrics["delta_vs_none_ci95"][1]),
                        ],
                        "exact_sign_flip_p": metrics["exact_sign_flip_p"],
                        "n_split_seeds": int(metrics["n_split_seeds"]),
                    }
                )

    def select(normalization: str, mode: str, weight: float, horizon: int) -> dict[str, Any]:
        matches = [
            row
            for row in rows
            if row["normalization"] == normalization
            and row["mode"] == mode
            and row["weight"] == weight
            and row["horizon"] == horizon
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one row for {normalization}/{mode}/{weight}/h{horizon}")
        return matches[0]

    h8_global = {
        str(weight): {
            normalization: select(normalization, "global", weight, 8)
            for normalization in normalizations
        }
        for weight in reference["protocol"]["weights"]
    }
    h8_conditional = {
        str(weight): {
            normalization: select(normalization, "conditional", weight, 8)
            for normalization in normalizations
        }
        for weight in reference["protocol"]["weights"]
    }
    report = {
        "artifact": "external_frozen_lewm_temporal_alignment_adapter_normalization_robustness",
        "status": "cross_normalization_summary",
        "inputs": [str(path) for path in paths],
        "normalizations": normalizations,
        "protocol": {
            "split_seeds": protocol_identity[0],
            "adapter_seeds": protocol_identity[1],
            "weights": protocol_identity[2],
            "episodes": protocol_identity[3],
            "windows": protocol_identity[4],
            "independent_unit": "episode split seed; adapter initialization seeds nested within split",
        },
        "rows": rows,
        "h8": {"global": h8_global, "conditional": h8_conditional},
        "claim_boundary": [
            "The summary compares frozen residual-adapter interventions under matched episode-disjoint protocols; it does not pool windows across normalization arms.",
            "A positive or negative h=8 delta is a frozen latent-rollout diagnostic, not evidence for or against online control, physical causality, or end-to-end retraining gains.",
        ],
    }
    if not _finite(report):
        raise ValueError("adapter summary contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Frozen external temporal adapter: normalization robustness",
        "",
        f"Matched episode-disjoint protocol: {protocol_identity[3]} episodes, {protocol_identity[4]} windows, split seeds {protocol_identity[0]}, adapter seeds {protocol_identity[1]}.",
        "",
        "| normalization | mode | weight | h=1 delta | h=4 delta | h=8 delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for normalization in normalizations:
        for mode in ("global", "conditional"):
            for weight in reference["protocol"]["weights"]:
                cells = []
                for horizon in (1, 4, 8):
                    item = select(normalization, mode, float(weight), horizon)
                    cells.append(
                        f"{item['delta_vs_none_mean']:.6g} [{item['delta_vs_none_ci95'][0]:.6g}, {item['delta_vs_none_ci95'][1]:.6g}]"
                    )
                lines.append(
                    f"| {normalization} | {mode} | {float(weight):.6g} | "
                    + " | ".join(cells)
                    + " |"
                )
    lines.extend(
        [
            "",
            "Intervals are split-level paired summaries; negative values favor the adapter over the matched no-alignment adapter. The h=8 interval is not uniformly negative for conditional alignment, so the evidence is a weight- and condition-sensitive trade-off rather than a universal improvement claim.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.inputs, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
