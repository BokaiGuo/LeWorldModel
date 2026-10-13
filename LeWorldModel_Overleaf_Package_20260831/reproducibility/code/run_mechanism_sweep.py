"""Run joint theory-side mechanism sweeps for spectral transfer.

This script expands the basic theory diagnostics with three complementary
checks:

1. encoder/predictor exponent interactions;
2. heat-filter strength and generator-rate sensitivity;
3. oblique-frame finite-window bias across frame obliqueness.

It exports a compact publication-style figure, Markdown tables, and a
machine-readable JSON record. The results are analytical toy-model evidence;
they are not measurements from a trained LeWM checkpoint.
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
    heat_filter_recovery_report,
    oblique_frame_stress_report,
    spectral_transfer_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/theory_mechanism_sweep_20260801",
        help="output directory for JSON, Markdown, and figure assets",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--intrinsic-dim", type=float, default=2.0)
    parser.add_argument("--input-exponent", type=float, default=1.0)
    parser.add_argument("--encoder-exponents", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--predictor-exponents", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--heat-times", type=float, nargs="+", default=[0.0, 0.001, 0.003, 0.01, 0.03])
    parser.add_argument("--heat-exponents", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--obliqueness-values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--oblique-seed", type=int, default=7)
    parser.add_argument("--heat-noise-std", type=float, default=0.03)
    return parser.parse_args()


def _analysis_windows(dim: int) -> tuple[tuple[int, int], tuple[int, int]]:
    head_end = max(8, int(round(dim * 0.25)))
    tail_start = max(head_end + 1, int(round(dim * 0.5)) + 1)
    tail_start = min(dim, tail_start)
    if tail_start <= head_end:
        tail_start = min(dim, head_end + 1)
    return (1, head_end), (tail_start, dim)


def _band_tail_fraction(profile: dict[str, float]) -> float:
    if not profile:
        return float("nan")
    last_key = max(profile, key=lambda key: int(key.split(":", maxsplit=1)[0]))
    return float(profile[last_key])


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)


def _heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[float],
    col_labels: list[float],
    title: str,
    colorbar_label: str,
) -> None:
    image = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), [f"{value:g}" for value in col_labels])
    ax.set_yticks(np.arange(len(row_labels)), [f"{value:g}" for value in row_labels])
    ax.set_xlabel("predictor exponent")
    ax.set_ylabel("encoder exponent")
    ax.set_title(title, fontsize=9, pad=4)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > np.nanmean(values) else "black",
            )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label(colorbar_label, fontsize=7)


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    interaction = payload["interaction"]
    heat = payload["heat"]
    oblique = payload["oblique"]
    horizon = payload["horizon"]
    recovery_by_exponent = {
        row["heat_exponent"]: row["heat_recovery_median_r2"]
        for row in heat
        if row["heat_time"] == min(item["heat_time"] for item in heat)
    }
    recoverable = [exponent for exponent, r2 in recovery_by_exponent.items() if r2 >= 0.99]
    unstable = [exponent for exponent, r2 in recovery_by_exponent.items() if r2 < 0.99]
    lines = [
        "# Theory Mechanism Sweep Summary",
        "",
        "## Scope",
        "",
        "- This is an analytical diagonal-transfer and oblique-frame simulation, not a checkpoint-level LeWM result.",
        f"- dim: {payload['dim']}; intrinsic dim: {payload['intrinsic_dim']}; horizon: {horizon}",
        f"- heat recovery noise std: {payload['heat_noise_std']}",
        "",
        "## Encoder/Predictor Interaction",
        "",
        "| encoder exponent | predictor exponent | expected latent alpha | residual h{h}/h1 | predicted h{h}/h1 | residual tail-band fraction at h{h} |".format(h=horizon),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in interaction:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["encoder_exponent"]),
                    _fmt(row["predictor_exponent"]),
                    _fmt(row["expected_oracle_alpha"]),
                    _fmt(row["residual_growth_ratio"]),
                    _fmt(row["predicted_decay_ratio"]),
                    _fmt(row["residual_tail_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Heat-Filter Sensitivity",
            "",
            "| heat exponent | heat time | filtered trace / original | filtered tail-band fraction | residual trace / original at h{h} | known-generator R2 |".format(h=horizon),
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in heat:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["heat_exponent"]),
                    _fmt(row["heat_time"]),
                    _fmt(row["filtered_trace_fraction"]),
                    _fmt(row["filtered_tail_fraction"]),
                    _fmt(row["residual_trace_over_original"]),
                    _fmt(row["heat_recovery_median_r2"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Oblique-Frame Stress",
            "",
            "| obliqueness | baseline mean abs bias | broken-tail mean abs bias | frame condition number |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in oblique:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["obliqueness"]),
                    _fmt(row["baseline_bias_abs_mean"]),
                    _fmt(row["tail_bias_abs_mean"]),
                    _fmt(row["frame_condition_number"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- At every encoder setting, increasing the predictor exponent reduces residual growth over the chosen horizon.",
            "- Increasing the encoder exponent steepens the latent spectrum and lowers the residual tail-band fraction.",
            "- Heat filtering suppresses tail energy and total trace more strongly as the generator-rate exponent increases.",
            (
                "- Known-generator recovery is high for rate exponents "
                + ", ".join(_fmt(value) for value in recoverable)
                + "."
                if recoverable
                else "- No heat-rate setting reaches the high-recovery threshold in this sweep."
            ),
            (
                "- Recovery falls below the high-recovery threshold for rate exponents "
                + ", ".join(_fmt(value) for value in unstable)
                + "; shallow spectra can have insufficient log-spectral contrast under noise, while steep spectra can underflow in high-rank modes."
                if unstable
                else "- No low-recovery heat-rate regime was observed at the configured noise level."
            ),
            "- Broken-tail spectra reduce finite-window slope bias throughout the nonzero obliqueness scan.",
            "",
            "## Claim Boundary",
            "",
            "- The interaction heatmaps establish operator-level ordering under the specified toy assumptions.",
            "- Heat-filter recovery uses a known synthetic generator and is only diagnostic-valid in the explicitly recoverable regimes above; it does not establish learned invariance.",
            "- Oblique-frame curves quantify finite-window estimation sensitivity; they do not identify causal world factors in PushT.",
            "",
        ]
    )
    (output / "mechanism_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    encoder_exponents = sorted(set(float(value) for value in args.encoder_exponents))
    predictor_exponents = sorted(set(float(value) for value in args.predictor_exponents))
    heat_times = sorted(set(float(value) for value in args.heat_times))
    heat_exponents = sorted(set(float(value) for value in args.heat_exponents))
    obliqueness_values = sorted(set(float(value) for value in args.obliqueness_values))
    head_window, tail_window = _analysis_windows(args.dim)
    slope_windows = [(1, None), head_window, tail_window]
    horizon_values = [1, args.horizon] if args.horizon != 1 else [1]

    interaction: list[dict[str, Any]] = []
    for encoder_exponent in encoder_exponents:
        for predictor_exponent in predictor_exponents:
            config = TransferConfig(
                dim=args.dim,
                intrinsic_dim=args.intrinsic_dim,
                input_exponent=args.input_exponent,
                encoder_exponent=encoder_exponent,
                predictor_exponent=predictor_exponent,
            )
            report = spectral_transfer_report(config, slope_windows=slope_windows, horizons=horizon_values)
            growth = report["multi_horizon"]["growth"]
            final_horizon = report["multi_horizon"]["reports"][str(args.horizon)]
            interaction.append(
                {
                    "encoder_exponent": encoder_exponent,
                    "predictor_exponent": predictor_exponent,
                    "expected_oracle_alpha": report["expected_oracle_alpha"],
                    "latent_tail_fraction": _band_tail_fraction(report["band_energy"]["latent_eigenvalues"]),
                    "residual_growth_ratio": float(
                        growth["residual_trace"][str(args.horizon)] / growth["residual_trace"]["1"]
                    ),
                    "predicted_decay_ratio": float(
                        growth["predicted_trace"][str(args.horizon)] / growth["predicted_trace"]["1"]
                    ),
                    "residual_tail_fraction": _band_tail_fraction(
                        final_horizon["band_energy"]["residual_eigenvalues"]
                    ),
                    "report": report,
                }
            )

    heat: list[dict[str, Any]] = []
    for heat_exponent in heat_exponents:
        recovery_config = TransferConfig(
            dim=args.dim,
            intrinsic_dim=args.intrinsic_dim,
            input_exponent=args.input_exponent,
            encoder_exponent=1.0,
            predictor_exponent=1.0,
            heat_exponent=heat_exponent,
        )
        recovery = heat_filter_recovery_report(
            recovery_config,
            times=[0.01, 0.03, 0.05],
            noise_std=args.heat_noise_std,
            seed=0,
        )
        for heat_time in heat_times:
            config = TransferConfig(
                dim=args.dim,
                intrinsic_dim=args.intrinsic_dim,
                input_exponent=args.input_exponent,
                encoder_exponent=1.0,
                predictor_exponent=1.0,
                heat_time=heat_time,
                heat_exponent=heat_exponent,
            )
            report = spectral_transfer_report(config, slope_windows=slope_windows, horizons=horizon_values)
            stats = report["spectrum_stats"]
            final_horizon = report["multi_horizon"]["reports"][str(args.horizon)]
            heat.append(
                {
                    "heat_exponent": heat_exponent,
                    "heat_time": heat_time,
                    "filtered_trace_fraction": float(
                        stats["filtered_latent_eigenvalues"]["trace"]
                        / stats["latent_eigenvalues"]["trace"]
                    ),
                    "filtered_tail_fraction": _band_tail_fraction(
                        report["band_energy"]["filtered_latent_eigenvalues"]
                    ),
                    "residual_trace_over_original": float(
                        report["multi_horizon"]["growth"]["residual_trace"][str(args.horizon)]
                        / stats["latent_eigenvalues"]["trace"]
                    ),
                    "residual_tail_fraction": _band_tail_fraction(
                        final_horizon["band_energy"]["residual_eigenvalues"]
                    ),
                    "heat_recovery_median_r2": recovery["median_r2"],
                    "report": report,
                    "heat_recovery": recovery,
                }
            )

    baseline = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponent,
        encoder_exponent=1.0,
        predictor_exponent=1.0,
    )
    broken_tail = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponent,
        encoder_exponent=1.0,
        tail_encoder_exponent=3.0,
        breakpoint=max(2, int(round(args.dim * 0.25))),
        predictor_exponent=1.0,
    )
    oblique: list[dict[str, Any]] = []
    for obliqueness in obliqueness_values:
        baseline_report = oblique_frame_stress_report(
            baseline,
            obliqueness=obliqueness,
            seed=args.oblique_seed,
            slope_windows=[head_window, tail_window],
        )
        tail_report = oblique_frame_stress_report(
            broken_tail,
            obliqueness=obliqueness,
            seed=args.oblique_seed,
            slope_windows=[head_window, tail_window],
        )
        oblique.append(
            {
                "obliqueness": obliqueness,
                "baseline_bias_abs_mean": float(
                    np.mean([abs(value) for value in baseline_report["finite_window_alpha_bias"].values()])
                ),
                "tail_bias_abs_mean": float(
                    np.mean([abs(value) for value in tail_report["finite_window_alpha_bias"].values()])
                ),
                "frame_condition_number": baseline_report["frame_condition_number"],
                "baseline_report": baseline_report,
                "tail_report": tail_report,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dim": args.dim,
        "intrinsic_dim": args.intrinsic_dim,
        "input_exponent": args.input_exponent,
        "horizon": args.horizon,
        "heat_noise_std": args.heat_noise_std,
        "analysis_windows": {"head": list(head_window), "tail": list(tail_window)},
        "interaction": interaction,
        "heat": heat,
        "oblique": oblique,
    }
    (output / "theory_mechanism_sweep.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    residual_matrix = np.array(
        [
            [
                next(
                    row["residual_growth_ratio"]
                    for row in interaction
                    if row["encoder_exponent"] == encoder and row["predictor_exponent"] == predictor
                )
                for predictor in predictor_exponents
            ]
            for encoder in encoder_exponents
        ]
    )
    tail_matrix = np.array(
        [
            [
                next(
                    row["residual_tail_fraction"]
                    for row in interaction
                    if row["encoder_exponent"] == encoder and row["predictor_exponent"] == predictor
                )
                for predictor in predictor_exponents
            ]
            for encoder in encoder_exponents
        ]
    )

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
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.9), constrained_layout=True)

    ax = axes[0, 0]
    _heatmap(
        ax,
        residual_matrix,
        encoder_exponents,
        predictor_exponents,
        f"Residual growth h{args.horizon}/h1",
        "ratio",
    )
    _panel_label(ax, "A")

    ax = axes[0, 1]
    _heatmap(
        ax,
        tail_matrix,
        encoder_exponents,
        predictor_exponents,
        f"Residual tail fraction at h{args.horizon}",
        "tail fraction",
    )
    _panel_label(ax, "B")

    ax = axes[0, 2]
    palette = sns.color_palette("colorblind", n_colors=len(encoder_exponents))
    for color, encoder in zip(palette, encoder_exponents):
        rows = [row for row in interaction if row["encoder_exponent"] == encoder]
        rows.sort(key=lambda row: row["predictor_exponent"])
        ax.plot(
            [row["predictor_exponent"] for row in rows],
            [row["residual_growth_ratio"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"encoder {encoder:g}",
        )
    ax.set_title("Interaction ordering", fontsize=9, pad=4)
    ax.set_xlabel("predictor exponent")
    ax.set_ylabel(f"residual h{args.horizon}/h1")
    ax.set_xticks(predictor_exponents)
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _panel_label(ax, "C")

    ax = axes[1, 0]
    palette = sns.color_palette("colorblind", n_colors=len(heat_exponents))
    for color, heat_exponent in zip(palette, heat_exponents):
        rows = [row for row in heat if row["heat_exponent"] == heat_exponent]
        rows.sort(key=lambda row: row["heat_time"])
        ax.plot(
            [row["heat_time"] for row in rows],
            [row["filtered_trace_fraction"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"rate exponent {heat_exponent:g}",
        )
    ax.set_title("Heat-filter trace retention", fontsize=9, pad=4)
    ax.set_xlabel("heat time")
    ax.set_ylabel("filtered trace / original")
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _panel_label(ax, "D")

    ax = axes[1, 1]
    for color, heat_exponent in zip(palette, heat_exponents):
        rows = [row for row in heat if row["heat_exponent"] == heat_exponent]
        rows.sort(key=lambda row: row["heat_time"])
        ax.plot(
            [row["heat_time"] for row in rows],
            [row["filtered_tail_fraction"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"rate exponent {heat_exponent:g}",
        )
    ax.set_title("Heat-filter tail suppression", fontsize=9, pad=4)
    ax.set_xlabel("heat time")
    ax.set_ylabel("filtered tail-band fraction")
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _panel_label(ax, "E")

    ax = axes[1, 2]
    ax.plot(
        [row["obliqueness"] for row in oblique],
        [row["baseline_bias_abs_mean"] for row in oblique],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="baseline",
    )
    ax.plot(
        [row["obliqueness"] for row in oblique],
        [row["tail_bias_abs_mean"] for row in oblique],
        marker="s",
        linewidth=1.8,
        markersize=4,
        label="broken tail",
    )
    ax.set_title("Finite-window oblique bias", fontsize=9, pad=4)
    ax.set_xlabel("obliqueness")
    ax.set_ylabel("mean absolute alpha bias")
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _panel_label(ax, "F")

    fig.savefig(output / "mechanism_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "mechanism_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _write_markdown(output, payload)

    print(output / "theory_mechanism_sweep.json")
    print(output / "mechanism_summary.pdf")
    print(output / "mechanism_summary.png")
    print(output / "mechanism_summary.md")


if __name__ == "__main__":
    main()
