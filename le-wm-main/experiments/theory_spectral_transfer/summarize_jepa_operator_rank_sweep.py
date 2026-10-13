"""Summarize held-out frozen I-JEPA operator fits across ranks and datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.inputs:
        payload = json.loads(path.read_text())
        cfg, holdout = payload["config"], payload["holdout"]
        pair, null, transfer = payload["operator_pair"], payload["spectrum_preserving_null"], payload["transfer_readout"]
        rows.append(
            {
                "dataset": cfg["dataset"],
                "view_transform": cfg["view_transform"],
                "images": cfg["images"],
                "train_images": cfg["train_images"],
                "test_images": cfg["test_images"],
                "operator_rank": cfg["operator_rank"],
                "top_k": cfg["top_k"],
                "holdout_r2": holdout["r2"],
                "commutator": pair["commutator_norm"],
                "null_mean": null["mean"],
                "null_p05": null["p05"],
                "null_p95": null["p95"],
                "commutator_minus_null_mean": pair["commutator_norm"] - null["mean"],
                "predicted_lift": transfer["heldout_predicted_displacement_top_k_lift"],
                "pair_shuffled_lift": transfer["heldout_pair_shuffled_top_k_lift"],
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["operator_rank"]))
    fieldnames = list(rows[0])
    with (args.output_dir / "jepa_operator_rank_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "jepa_operator_rank_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Frozen I-JEPA Operator-Rank Sweep",
        "",
        "The held-out low-rank fits are predictive, but the covariance--operator commutator is not consistently larger than the spectrum-preserving random-basis null. This separates view predictability from identified transfer geometry.",
        "",
        "| dataset | rank | held-out $R^2$ | commutator | null mean | comm.-null | predicted lift | pair-shuffled lift |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['operator_rank']} | {row['holdout_r2']:.5f} | "
            f"{row['commutator']:.5f} | {row['null_mean']:.5f} | {row['commutator_minus_null_mean']:.5f} | "
            f"{row['predicted_lift']:.5f} | {row['pair_shuffled_lift']:.5f} |"
        )
    lines.extend(
        [
            "",
            "All rows use the frozen I-JEPA ViT-H/16 checkpoint and deterministic horizontal-flip pairs. The fit is an estimator stress test, not a JEPA training-quality or action-conditioned world-model evaluation.",
            "",
        ]
    )
    (args.output_dir / "jepa_operator_rank_sweep.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
