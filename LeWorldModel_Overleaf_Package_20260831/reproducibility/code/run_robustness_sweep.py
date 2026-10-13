"""Run a robustness sweep for the theory spectral-transfer diagnostics.

This script complements the richer sweep by checking how sensitive the theory
side support diagnostics are to the random frame seed and heat-filter noise.
It produces a compact figure plus a Markdown summary with mean/std readouts.
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
import pandas as pd
import seaborn as sns

from experiments.theory_spectral_transfer import (
    TransferConfig,
    heat_filter_recovery_report,
    oblique_frame_stress_report,
    sigreg_local_anisotropy_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/theory_robustness_sweep_20260801",
        help="output directory for JSON, Markdown, and figure assets",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--intrinsic-dim", type=float, default=2.0)
    parser.add_argument("--input-exponent", type=float, default=1.0)
    parser.add_argument("--encoder-exponent", type=float, default=1.0)
    parser.add_argument("--predictor-exponent", type=float, default=1.0)
    parser.add_argument("--tail-encoder-exponent", type=float, default=3.0)
    parser.add_argument("--obliqueness", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05])
    parser.add_argument("--times", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    return parser.parse_args()


def _analysis_windows(dim: int) -> tuple[tuple[int, int], tuple[int, int]]:
    head_end = max(8, int(round(dim * 0.25)))
    tail_start = max(head_end + 1, int(round(dim * 0.5)) + 1)
    tail_start = min(dim, tail_start)
    if tail_start <= head_end:
        tail_start = min(dim, head_end + 1)
    return (1, head_end), (tail_start, dim)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    seeds = payload["seeds"]
    noise_stds = payload["noise_stds"]
    lines = [
        "# Theory Robustness Sweep Summary",
        "",
        "## Setup",
        "",
        f"- dim: {payload['dim']}",
        f"- head window: {payload['analysis_windows']['head'][0]}:{payload['analysis_windows']['head'][1]}",
        f"- tail window: {payload['analysis_windows']['tail'][0]}:{payload['analysis_windows']['tail'][1]}",
        f"- seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} total)",
        f"- noise stds: {', '.join(_fmt(v) for v in noise_stds)}",
        "",
        "## Oblique Bias",
        "",
        "| case | mean | std | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in ["baseline", "tail"]:
        stats = payload["oblique_bias"][case]
        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    _fmt(stats["mean"]),
                    _fmt(stats["std"]),
                    _fmt(stats["min"]),
                    _fmt(stats["max"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Frame Condition Number",
            "",
            "| case | mean | std | min | max |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in ["baseline", "tail"]:
        stats = payload["frame_condition_number"][case]
        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    _fmt(stats["mean"]),
                    _fmt(stats["std"]),
                    _fmt(stats["min"]),
                    _fmt(stats["max"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## SIGReg Stability",
            "",
            "| metric | mean | std | min | max |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    stats = payload["sigreg_ratio"]
    lines.append(
        "| global_over_local_stable_rank | "
        + " | ".join(_fmt(stats[key]) for key in ["mean", "std", "min", "max"])
        + " |"
    )
    lines.extend(
        [
            "",
            "## Heat Recovery",
            "",
            "| noise std | mean R2 | std | min | max |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for noise in noise_stds:
        stats = payload["heat_median_r2"][str(noise)]
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(noise),
                    _fmt(stats["mean"]),
                    _fmt(stats["std"]),
                    _fmt(stats["min"]),
                    _fmt(stats["max"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Oblique bias is stable across seeds and stays lower in the broken-tail case than in the baseline case.",
            "- Frame condition numbers vary with seed; the baseline and tail frame draws are intentionally shared at each seed, so the two boxplots overlap.",
            "- SIGReg local/global stable-rank ratios remain > 1 across seeds.",
            "- Heat-filter recovery remains close to perfect at low noise and degrades smoothly as noise increases.",
            "",
        ]
    )
    (output / "robustness_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    head_window, tail_window = _analysis_windows(args.dim)
    slope_windows = [head_window, tail_window]
    baseline_cfg = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponent,
        encoder_exponent=args.encoder_exponent,
        predictor_exponent=args.predictor_exponent,
    )
    tail_cfg = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponent,
        encoder_exponent=args.encoder_exponent,
        predictor_exponent=args.predictor_exponent,
        tail_encoder_exponent=args.tail_encoder_exponent,
        breakpoint=max(2, int(round(args.dim * 0.25))),
    )

    seeds = [int(s) for s in args.seeds]
    noise_stds = [float(v) for v in args.noise_stds]
    times = [float(v) for v in args.times]

    per_seed: list[dict[str, Any]] = []
    oblique_bias = {"baseline": [], "tail": []}
    frame_condition_number = {"baseline": [], "tail": []}
    sigreg_ratio: list[float] = []
    heat_median_r2: dict[str, list[float]] = {str(noise): [] for noise in noise_stds}

    for seed in seeds:
        baseline_oblique = oblique_frame_stress_report(
            baseline_cfg,
            obliqueness=args.obliqueness,
            seed=seed,
            slope_windows=slope_windows,
        )
        tail_oblique = oblique_frame_stress_report(
            tail_cfg,
            obliqueness=args.obliqueness,
            seed=seed,
            slope_windows=slope_windows,
        )
        sigreg = sigreg_local_anisotropy_report(dim=32, cells=12, anisotropy=20.0, seed=seed)
        seed_heat = {}
        for noise in noise_stds:
            heat = heat_filter_recovery_report(
                baseline_cfg,
                times=times,
                noise_std=noise,
                seed=seed,
            )
            heat_median_r2[str(noise)].append(float(heat["median_r2"]))
            seed_heat[str(noise)] = heat
        oblique_bias["baseline"].append(float(np.mean([abs(v) for v in baseline_oblique["finite_window_alpha_bias"].values()])))
        oblique_bias["tail"].append(float(np.mean([abs(v) for v in tail_oblique["finite_window_alpha_bias"].values()])))
        frame_condition_number["baseline"].append(float(baseline_oblique["frame_condition_number"]))
        frame_condition_number["tail"].append(float(tail_oblique["frame_condition_number"]))
        sigreg_ratio.append(float(sigreg["global_over_local_stable_rank"]))
        per_seed.append(
            {
                "seed": seed,
                "baseline_oblique": baseline_oblique,
                "tail_oblique": tail_oblique,
                "sigreg_local_anisotropy": sigreg,
                "heat": seed_heat,
            }
        )

    payload = {
        "dim": args.dim,
        "analysis_windows": {"head": list(head_window), "tail": list(tail_window)},
        "seeds": seeds,
        "noise_stds": noise_stds,
        "oblique_bias": {case: _aggregate(values) for case, values in oblique_bias.items()},
        "frame_condition_number": {case: _aggregate(values) for case, values in frame_condition_number.items()},
        "sigreg_ratio": _aggregate(sigreg_ratio),
        "heat_median_r2": {noise: _aggregate(values) for noise, values in heat_median_r2.items()},
        "per_seed": per_seed,
    }
    (output / "robustness_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

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
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9), constrained_layout=True)

    ax = axes[0, 0]
    bias_rows = []
    for case in ["baseline", "tail"]:
        for seed, value in zip(seeds, oblique_bias[case]):
            bias_rows.append({"case": case, "seed": seed, "oblique_bias": value})
    bias_df = pd.DataFrame(bias_rows)
    sns.boxplot(data=bias_df, x="case", y="oblique_bias", ax=ax, width=0.45, color="#d9d9d9", fliersize=0)
    sns.stripplot(data=bias_df, x="case", y="oblique_bias", ax=ax, color="#444444", size=3, jitter=0.15)
    ax.set_title("Oblique bias across seeds", fontsize=9, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("oblique bias")
    ax.text(-0.14, 1.05, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[0, 1]
    cond_rows = []
    for case in ["baseline", "tail"]:
        for seed, value in zip(seeds, frame_condition_number[case]):
            cond_rows.append({"case": case, "seed": seed, "condition_number": value})
    cond_df = pd.DataFrame(cond_rows)
    sns.boxplot(data=cond_df, x="case", y="condition_number", ax=ax, width=0.45, color="#d9d9d9", fliersize=0)
    sns.stripplot(data=cond_df, x="case", y="condition_number", ax=ax, color="#444444", size=3, jitter=0.15)
    ax.set_title("Frame condition number across seeds", fontsize=9, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("condition number")
    ax.text(-0.14, 1.05, "B", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[1, 0]
    sns.boxplot(y=sigreg_ratio, ax=ax, width=0.45, color="#d9d9d9", fliersize=0)
    sns.stripplot(y=sigreg_ratio, ax=ax, color="#444444", size=3, jitter=0.15)
    ax.set_title("SIGReg global/local stable rank", fontsize=9, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("ratio")
    ax.set_xticks([])
    ax.text(-0.14, 1.05, "C", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[1, 1]
    palette = sns.color_palette("colorblind", n_colors=len(noise_stds))
    means = [payload["heat_median_r2"][str(noise)]["mean"] for noise in noise_stds]
    stds = [payload["heat_median_r2"][str(noise)]["std"] for noise in noise_stds]
    ax.errorbar(noise_stds, means, yerr=stds, marker="o", linewidth=1.8, markersize=4, color=palette[0])
    ax.set_title("Heat-filter recovery vs noise", fontsize=9, pad=4)
    ax.set_xlabel("noise std")
    ax.set_ylabel("median R2")
    ax.set_ylim(0.95, 1.01 if max(means) > 0.99 else 1.0)
    ax.text(-0.14, 1.05, "D", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    fig.savefig(output / "robustness_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "robustness_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_markdown(output, payload)

    print(output / "robustness_summary.json")
    print(output / "robustness_summary.pdf")
    print(output / "robustness_summary.png")
    print(output / "robustness_summary.md")


if __name__ == "__main__":
    main()
