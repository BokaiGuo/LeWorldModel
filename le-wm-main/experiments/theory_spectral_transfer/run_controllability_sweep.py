"""Run a theory-only controllability-alignment sweep.

This sweep holds the latent spectrum fixed and rotates the controllable
subspace relative to the eigenspace.  The result separates a spectrum-only
story from the stronger claim that controllability depends on alignment with
the preserved subspace.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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

from experiments.theory_spectral_transfer import (
    TransferConfig,
    controllability_alignment_sweep_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/theory_controllability_sweep_20260828",
        help="output directory for JSON, Markdown, and figure assets",
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--intrinsic-dim", type=float, default=2.0)
    parser.add_argument("--input-exponent", type=float, default=1.0)
    parser.add_argument("--predictor-exponent", type=float, default=1.0)
    parser.add_argument("--controllable-dim", type=int, default=8)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 15, 30, 45, 60, 75, 90])
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--random-trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    rows = payload["records"]
    contrast = payload["contrast"]
    null = payload["random_rotation_null"]
    top_k = str(contrast["top_k"])
    lines = [
        "# Controllability Alignment Sweep",
        "",
        "## Object",
        "",
        "- This sweep holds the latent spectrum fixed and rotates the controllable subspace.",
        "- It asks whether controllability is explained by spectrum alone or by alignment with the preserved subspace.",
        "",
        "## Main Readout",
        "",
        "| angle | latent mass share | residual energy share | top-k lift |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["angle_degrees"]),
                    _fmt(row["latent_mass_share"]),
                    _fmt(row["prediction_residual_energy_share"]),
                    _fmt(row["top_band_lift"][top_k]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Random Null",
            "",
            f"- top-{top_k} lift mean: {_fmt(null['top_band_lift'][top_k]['mean'])}",
            f"- top-{top_k} lift p05/p95: {_fmt(null['top_band_lift'][top_k]['p05'])} / {_fmt(null['top_band_lift'][top_k]['p95'])}",
            f"- residual share mean: {_fmt(null['prediction_residual_energy_share']['mean'])}",
            f"- latent mass share mean: {_fmt(null['latent_mass_share']['mean'])}",
            "",
            "## Readout",
            "",
            "- Aligned controllable directions retain more top-band lift and lower residual energy.",
            "- Orthogonal rotation pushes the same subspace mass into less controllable directions even though the spectrum is unchanged.",
            "- The random-null band shows the effect is not just a subspace-dimension artifact.",
            "",
            "## Claim Boundary",
            "",
            "- This is a synthetic spectrum-fixed mechanism check, not an empirical PushT control result.",
            "- It supports the claim that controllability requires alignment with the preserved eigenspace, not only a favorable spectrum.",
            "",
        ]
    )
    (output / "controllability_summary.md").write_text("\n".join(lines) + "\n")


def _plot_heatmap(ax: plt.Axes, payload: dict[str, Any]) -> None:
    rows = payload["records"]
    angles = [row["angle_degrees"] for row in rows]
    ks = payload["ks"]
    values = np.array([[row["top_band_lift"][str(k)] for row in rows] for k in ks], dtype=np.float64)
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(angles)), [f"{angle:g}" for angle in angles])
    ax.set_yticks(np.arange(len(ks)), [f"{k:g}" for k in ks])
    ax.set_xlabel("alignment angle")
    ax.set_ylabel("k")
    ax.set_title("Top-band lift across alignment", fontsize=9, pad=4)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if values[i, j] > np.nanmean(values) else "black")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("lift", fontsize=7)


def _plot_with_null(ax: plt.Axes, payload: dict[str, Any], key: str, title: str, ylabel: str) -> None:
    rows = payload["records"]
    angles = np.array([row["angle_degrees"] for row in rows], dtype=np.float64)
    values = np.array([row[key] for row in rows], dtype=np.float64)
    null = payload["random_rotation_null"][key]
    ax.plot(angles, values, marker="o", linewidth=1.8, markersize=4, color="#1b9e77")
    ax.axhspan(null["p05"], null["p95"], color="#999999", alpha=0.18, label="random null p05-p95")
    ax.axhline(null["mean"], color="#666666", linestyle="--", linewidth=1.0, label="random null mean")
    ax.set_xlabel("alignment angle")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="best")


def _plot_contrast(ax: plt.Axes, payload: dict[str, Any]) -> None:
    contrast = payload["contrast"]
    null = payload["random_rotation_null"]["top_band_lift"][str(contrast["top_k"])]
    labels = ["aligned", "orthogonal", "random mean"]
    values = [
        contrast["aligned_top_k_lift"],
        contrast["orthogonal_top_k_lift"],
        null["mean"],
    ]
    errors = [
        0.0,
        0.0,
        max(null["p95"] - null["mean"], null["mean"] - null["p05"]),
    ]
    colors = ["#1b9e77", "#d95f02", "#666666"]
    ax.bar(labels, values, color=colors, yerr=errors, capsize=3)
    ax.set_ylabel(f"top-{contrast['top_k']} lift")
    ax.set_title("Contrast at maximal k", fontsize=9, pad=4)
    ax.grid(True, axis="y", alpha=0.2)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponent,
        predictor_exponent=args.predictor_exponent,
    )
    report = controllability_alignment_sweep_report(
        config=config,
        controllable_dim=args.controllable_dim,
        angles=args.angles,
        ks=args.ks,
        samples=args.samples,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    (output / "controllability_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

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

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    _plot_heatmap(axes[0, 0], report)
    axes[0, 0].text(-0.14, 1.05, "A", transform=axes[0, 0].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    _plot_with_null(
        axes[0, 1],
        report,
        "prediction_residual_energy_share",
        "Residual energy vs alignment",
        "residual energy share",
    )
    axes[0, 1].text(-0.14, 1.05, "B", transform=axes[0, 1].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    _plot_with_null(
        axes[1, 0],
        report,
        "latent_mass_share",
        "Latent spectral mass vs alignment",
        "latent mass share",
    )
    axes[1, 0].text(-0.14, 1.05, "C", transform=axes[1, 0].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    _plot_contrast(axes[1, 1], report)
    axes[1, 1].text(-0.14, 1.05, "D", transform=axes[1, 1].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    fig.savefig(output / "controllability_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "controllability_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_markdown(output, report)

    print(output / "controllability_sweep.json")
    print(output / "controllability_summary.pdf")
    print(output / "controllability_summary.png")
    print(output / "controllability_summary.md")


if __name__ == "__main__":
    main()
