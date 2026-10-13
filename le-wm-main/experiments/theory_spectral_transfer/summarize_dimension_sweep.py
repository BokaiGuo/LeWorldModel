"""Summarize precomputed theory sweeps across latent dimensions.

Expected input layout:
    <input>/dim64/theory_rich_sweep.json
    <input>/dim128/theory_rich_sweep.json
    ...

The script emits a cross-dimension JSON/Markdown summary and a publication
style PDF/PNG figure.  If a completed PushT empirical audit is available, its
rank-wise lift curves are included as trend-level context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/theory_dim_sweep_20260801",
        help="directory containing dim*/theory_rich_sweep.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory; defaults to --input",
    )
    parser.add_argument(
        "--empirical",
        default="results/pusht/lewm/empirical_spectral_audit.json",
        help="optional empirical audit JSON for trend-level context",
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_records(input_dir: Path, dims: list[int]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for dim in dims:
        path = input_dir / f"dim{dim}" / "theory_rich_sweep.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing theory sweep for dim={dim}: {path}")
        payload = json.loads(path.read_text())
        cases = {case["name"]: case for case in payload["cases"]}
        predictors = sorted(
            (case for case in payload["cases"] if case["name"].startswith("predictor_")),
            key=lambda case: case["predictor_exponent"],
        )
        tail3 = next(case for case in payload["cases"] if case["name"].startswith("broken_tail_3"))
        records[dim] = {
            "analysis_windows": payload["analysis_windows"],
            "predictor_exponents": [case["predictor_exponent"] for case in predictors],
            "predictor_residual_ratio": [
                case["residual_growth_ratio_h8_over_h1"] for case in predictors
            ],
            "predictor_decay_ratio": [
                case["predicted_decay_ratio_h8_over_h1"] for case in predictors
            ],
            "baseline_residual_ratio": cases["baseline"]["residual_growth_ratio_h8_over_h1"],
            "p05_residual_ratio": cases["predictor_0p5"]["residual_growth_ratio_h8_over_h1"],
            "p20_residual_ratio": cases["predictor_2p0"]["residual_growth_ratio_h8_over_h1"],
            "tail3_gap": tail3["latent_head_tail_gap"],
            "tail3_bias": tail3["oblique_bias_abs_mean"],
            "heat_r2": payload["support"]["heat_filter_recovery"]["median_r2"],
        }
    return records


def _load_empirical(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    weighted_mean = payload.get("weighted_mean", {}).get("lifts", {})
    controllability = payload.get("controllability", {})
    return {
        "weighted_mean_lifts": {
            key: weighted_mean[key]
            for key in ["dynamics", "predicted_dynamics", "prediction_residual"]
            if key in weighted_mean
        },
        "controllability": {
            key: controllability[key]
            for key in ["controllable_lift", "unexplained_lift"]
            if key in controllability
        },
    }


def _write_summary(
    output: Path,
    dims: list[int],
    records: dict[int, dict[str, Any]],
    empirical: dict[str, Any] | None,
) -> None:
    lines = [
        "# Cross-Dimension Theory Sweep Summary",
        "",
        "## Theory Sweep",
        "",
        "| dim | head window | tail window | baseline residual h8/h1 | predictor_0.5 residual h8/h1 | predictor_2.0 residual h8/h1 | predictor sweep monotone | tail-3 gap | tail-3 oblique bias | heat R2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dim in dims:
        record = records[dim]
        head = record["analysis_windows"]["head"]
        tail = record["analysis_windows"]["tail"]
        monotone = all(
            a >= b
            for a, b in zip(
                record["predictor_residual_ratio"],
                record["predictor_residual_ratio"][1:],
            )
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(dim),
                    f"{head[0]}:{head[1]}",
                    f"{tail[0]}:{tail[1]}",
                    _fmt(record["baseline_residual_ratio"]),
                    _fmt(record["p05_residual_ratio"]),
                    _fmt(record["p20_residual_ratio"]),
                    "yes" if monotone else "no",
                    _fmt(record["tail3_gap"]),
                    _fmt(record["tail3_bias"]),
                    _fmt(record["heat_r2"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Theory Readout",
            "",
            "- Predictor-exponent ordering is checked independently at each dimension.",
            "- The broken-tail gap is measured over dimension-scaled head and tail windows.",
            "- Empirical PushT values below are trend-level context, not a numerical identity with the theory model.",
        ]
    )
    if empirical:
        lines.extend(
            [
                "",
                "## PushT Empirical Context",
                "",
                "| metric | k=4 | k=8 | k=16 | k=32 | k=64 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for metric, values in empirical["weighted_mean_lifts"].items():
            lines.append(
                "| "
                + " | ".join(
                    [metric]
                    + [_fmt(values[str(k)]) for k in [4, 8, 16, 32, 64]]
                )
                + " |"
            )
    (output / "cross_dim_summary.md").write_text("\n".join(lines) + "\n")


def _write_figure(
    output: Path,
    dims: list[int],
    records: dict[int, dict[str, Any]],
) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.9)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8), constrained_layout=True)
    palette = sns.color_palette("colorblind", n_colors=len(dims))

    ax = axes[0, 0]
    for color, dim in zip(palette, dims):
        record = records[dim]
        ax.plot(
            record["predictor_exponents"],
            record["predictor_residual_ratio"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"dim {dim}",
        )
    ax.set_title("Residual growth vs predictor exponent", fontsize=9, pad=4)
    ax.set_xlabel("predictor exponent")
    ax.set_ylabel("residual h8/h1")
    ax.set_xticks(records[dims[0]]["predictor_exponents"])
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(-0.14, 1.05, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[0, 1]
    for color, dim in zip(palette, dims):
        record = records[dim]
        ax.plot(
            record["predictor_exponents"],
            record["predictor_decay_ratio"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"dim {dim}",
        )
    ax.set_title("Predicted decay vs predictor exponent", fontsize=9, pad=4)
    ax.set_xlabel("predictor exponent")
    ax.set_ylabel("predicted h8/h1")
    ax.set_xticks(records[dims[0]]["predictor_exponents"])
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(-0.14, 1.05, "B", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[1, 0]
    selected = [
        ("baseline", "baseline_residual_ratio", "#444444"),
        ("predictor_0p5", "p05_residual_ratio", palette[0]),
        ("predictor_2p0", "p20_residual_ratio", palette[-1]),
    ]
    for label, key, color in selected:
        ax.plot(
            dims,
            [records[dim][key] for dim in dims],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=label,
        )
    ax.set_title("Scale effect on residual growth", fontsize=9, pad=4)
    ax.set_xlabel("dim")
    ax.set_ylabel("residual h8/h1")
    ax.set_xticks(dims)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(-0.14, 1.05, "C", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[1, 1]
    tail_bias = [records[dim]["tail3_bias"] for dim in dims]
    ax.plot(dims, tail_bias, marker="s", linewidth=1.8, markersize=4, color=palette[1 if len(palette) > 1 else 0], label="tail exponent 3")
    ax.set_title("Broken-tail oblique bias", fontsize=9, pad=4)
    ax.set_xlabel("dim")
    ax.set_ylabel("oblique bias")
    ax.set_xticks(dims)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(-0.14, 1.05, "D", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    fig.savefig(output / "cross_dim_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "cross_dim_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output = Path(args.output) if args.output else input_dir
    output.mkdir(parents=True, exist_ok=True)
    dims = sorted(set(int(dim) for dim in args.dims))
    records = _load_records(input_dir, dims)
    empirical = _load_empirical(Path(args.empirical))
    payload = {
        "dims": dims,
        "records": records,
        "empirical_context": empirical,
    }
    (output / "cross_dim_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_summary(output, dims, records, empirical)
    _write_figure(output, dims, records)
    print(output / "cross_dim_summary.json")
    print(output / "cross_dim_summary.md")
    print(output / "cross_dim_summary.pdf")
    print(output / "cross_dim_summary.png")


if __name__ == "__main__":
    main()
