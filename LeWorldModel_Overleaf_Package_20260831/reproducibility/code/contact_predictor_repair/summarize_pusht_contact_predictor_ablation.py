"""Summarize one-factor PushT predictor-repair ablations.

Inputs are labeled as ``label=path`` so the output remains auditable and does
not infer a best setting from a hidden selection procedure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="label=repair-json")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_input(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"expected label=path, got {value!r}")
    return label, Path(path)


def summarize(inputs: list[str], horizon: int) -> dict[str, Any]:
    rows = []
    for value in inputs:
        label, path = parse_input(value)
        report = json.loads(path.read_text())
        matches = [item for item in report["horizons"] if int(item["horizon"]) == horizon]
        if len(matches) != 1:
            raise ValueError(f"{path} does not contain exactly one h={horizon} row")
        item = matches[0]
        metrics = item["metrics"]
        rows.append(
            {
                "label": label,
                "path": str(path),
                "seed": int(report["protocol"]["seed"]),
                "train_cases": int(report["protocol"]["train_cases"]),
                "epochs": int(report["protocol"].get("epochs", report["training"].get("epochs", -1))),
                "anchor_weight": report["protocol"].get("anchor_weight"),
                "final_loss": report["training"]["final_loss"],
                "base_error_plus": metrics["base_error_plus"],
                "repaired_calibrated_error_plus": metrics["repaired_calibrated_error_plus"],
                "gain_plus": metrics["paired_calibrated_gain_plus"],
                "upper_bound_error_plus": metrics["observed_latent_upper_bound_error_plus"],
                "plus_gain": item["by_arm"]["plus"]["paired_calibrated_gain"],
                "minus_gain": item["by_arm"]["minus"]["paired_calibrated_gain"],
                "zero_gain": item["by_arm"]["zero"]["paired_calibrated_gain"],
            }
        )
    return {
        "protocol": {
            "horizon": horizon,
            "aggregation": "single-seed diagnostic rows; no pooled interval",
            "selection": "labels supplied explicitly; no hidden best-of sweep",
        },
        "rows": rows,
        "claim_boundary": (
            "Training-strength and sample-size ablations for a targeted repair of one "
            "published PushT checkpoint; not evidence for original frozen-checkpoint "
            "causality, universal counterfactual accuracy, online planning, or task success."
        ),
    }


def main() -> None:
    args = parse_args()
    report = summarize(args.input, args.horizon)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact_predictor_ablation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        f"# Contact-predictor ablation summary (h={args.horizon})",
        "",
        "| setting | train cases | epochs | anchor | base + error | repaired calibrated + error | paired +a gain | 95% CI | observed-latent upper bound |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        gain = row["gain_plus"]
        anchor = "n/a" if row["anchor_weight"] is None else f"{row['anchor_weight']:.3g}"
        lines.append(
            "| %s | %d | %s | %s | %.3f | %.3f | %.3f | [%.3f, %.3f] | %.3f |"
            % (
                row["label"],
                row["train_cases"],
                row["epochs"] if row["epochs"] >= 0 else "n/a",
                anchor,
                row["base_error_plus"]["estimate"],
                row["repaired_calibrated_error_plus"]["estimate"],
                gain["estimate"],
                gain["ci95_low"],
                gain["ci95_high"],
                row["upper_bound_error_plus"]["estimate"],
            )
        )
    lines += [
        "",
        "The three-arm gains (+a, -a, zero) are stored in the JSON artifact.",
        "Claim boundary: targeted repair ablation, not a causal or planning result for the unmodified checkpoint.",
        "",
    ]
    (output / "contact_predictor_ablation_summary.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
