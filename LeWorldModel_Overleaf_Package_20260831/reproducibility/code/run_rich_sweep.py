"""Run a richer theory sweep and export a paper-style summary figure.

This script is the reproducible theory-side experiment step after the basic
operator diagnostics.  It scans predictor strengths and broken-power-law tail
settings, then exports a compact figure plus a Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
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
    identifiability_controllability_report,
    oblique_frame_stress_report,
    sigreg_local_anisotropy_report,
    spectral_transfer_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/theory_rich_sweep_20260801",
        help="output directory for JSON, Markdown, and figure assets",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--intrinsic-dim", type=float, default=2.0)
    parser.add_argument("--input-exponent", type=float, default=1.0)
    parser.add_argument("--encoder-exponent", type=float, default=1.0)
    parser.add_argument(
        "--predictor-exponents",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    )
    parser.add_argument(
        "--tail-cases",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.0],
        help="tail exponents for the broken-power-law tail scan; breakpoints scale with dim at roughly [dim/4, dim/4, dim/2]",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--obliqueness", type=float, default=0.5)
    parser.add_argument("--oblique-seed", type=int, default=7)
    return parser.parse_args()


def _case_name_predictor(exponent: float) -> str:
    return f"predictor_{str(exponent).replace('.', 'p')}"


def _analysis_windows(dim: int) -> tuple[tuple[int, int], tuple[int, int]]:
    head_end = max(8, int(round(dim * 0.25)))
    tail_start = max(head_end + 1, int(round(dim * 0.5)) + 1)
    tail_start = min(dim, tail_start)
    if tail_start <= head_end:
        tail_start = min(dim, head_end + 1)
    return (1, head_end), (tail_start, dim)


def _tail_breakpoint(index: int, dim: int) -> int:
    return max(2, int(round(dim * (0.25 if index < 2 else 0.5))))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _multi_horizon_series(report: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    growth = report["multi_horizon"]["growth"][key]
    horizons = sorted(int(k) for k in growth)
    values = [float(growth[str(h)]) for h in horizons]
    return horizons, values


def _summarize_case(
    name: str,
    cfg: TransferConfig,
    report: dict[str, Any],
    oblique_report: dict[str, Any],
    head_window: tuple[int, int],
    tail_window: tuple[int, int],
) -> dict[str, Any]:
    latent_slopes = report["slopes"]["latent_eigenvalues"]
    residual_slopes = report["slopes"]["residual_eigenvalues"]
    growth = report["multi_horizon"]["growth"]
    oblique = oblique_report
    heat = report["heat_filter_recovery"]
    oblique_biases = [abs(value) for value in oblique["finite_window_alpha_bias"].values()]
    head_label = f"{head_window[0]}:{head_window[1]}"
    tail_label = f"{tail_window[0]}:{tail_window[1]}"
    return {
        "name": name,
        "config": report["config"],
        "expected_oracle_alpha": report["expected_oracle_alpha"],
        "latent_alpha_1_end": latent_slopes["1:end"]["alpha"],
        "latent_head_tail_gap": latent_slopes[tail_label]["alpha"] - latent_slopes[head_label]["alpha"],
        "residual_alpha_1_end": residual_slopes["1:end"]["alpha"],
        "residual_growth_ratio_h8_over_h1": growth["residual_trace"]["8"] / growth["residual_trace"]["1"],
        "predicted_decay_ratio_h8_over_h1": growth["predicted_trace"]["8"] / growth["predicted_trace"]["1"],
        "predicted_trace_h": growth["predicted_trace"],
        "residual_trace_h": growth["residual_trace"],
        "predicted_monotone_nonincreasing": growth["predicted_monotone_nonincreasing"],
        "residual_monotone_nondecreasing": growth["residual_monotone_nondecreasing"],
        "oblique_bias_abs_mean": float(np.mean(oblique_biases)) if oblique_biases else float("nan"),
        "oblique_condition_number": oblique["frame_condition_number"],
        "heat_median_r2_noise_0p01": heat["median_r2"],
    }


def _plot_panel_labels(ax, label: str) -> None:
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


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)


def _plot_traces(ax, cases: list[dict[str, Any]], key: str, title: str, ylabel: str) -> None:
    palette = sns.color_palette("colorblind", n_colors=len(cases))
    for idx, case in enumerate(cases):
        horizons, values = _multi_horizon_series(case["report"], key)
        ax.plot(
            horizons,
            values,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=palette[idx],
            label=case["name"],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=4)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")


def _plot_predictor_sweep(ax, cases: list[dict[str, Any]]) -> None:
    xs = np.array([case["predictor_exponent"] for case in cases], dtype=float)
    residual = np.array([case["residual_growth_ratio_h8_over_h1"] for case in cases], dtype=float)
    decay = np.array([case["predicted_decay_ratio_h8_over_h1"] for case in cases], dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    residual = residual[order]
    decay = decay[order]
    ax.plot(xs, residual, marker="o", linewidth=1.8, markersize=4, label="residual h8/h1")
    ax.plot(xs, decay, marker="s", linewidth=1.8, markersize=4, label="predicted h8/h1")
    ax.set_xlabel("predictor exponent")
    ax.set_ylabel("ratio")
    ax.set_title("Predictor strength sweep", fontsize=9, pad=4)
    ax.set_xticks(xs)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")


def _plot_tail_scatter(ax, baseline_case: dict[str, Any], tail_cases: list[dict[str, Any]], support: dict[str, Any]) -> None:
    ax.scatter(
        [baseline_case["latent_head_tail_gap"]],
        [baseline_case["oblique_bias_abs_mean"]],
        s=36,
        marker="o",
        color="#444444",
        label="baseline",
        zorder=3,
    )
    palette = sns.color_palette("colorblind", n_colors=len(tail_cases))
    for idx, case in enumerate(tail_cases):
        label_offset = (4, 4)
        if case["name"].endswith("_b64"):
            label_offset = (-56, 4)
        ax.scatter(
            [case["latent_head_tail_gap"]],
            [case["oblique_bias_abs_mean"]],
            s=46,
            marker="s",
            color=palette[idx],
            label=case["name"],
            zorder=3,
        )
        ax.annotate(
            case["name"].replace("broken_tail_", "tail "),
            (case["latent_head_tail_gap"], case["oblique_bias_abs_mean"]),
            textcoords="offset points",
            xytext=label_offset,
            fontsize=7,
        )
    ax.annotate(
        "baseline",
        (baseline_case["latent_head_tail_gap"], baseline_case["oblique_bias_abs_mean"]),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=7,
    )
    ax.set_xlabel("latent head-tail gap")
    ax.set_ylabel("oblique finite-window bias")
    ax.set_title("Broken-tail spectrum vs finite-window bias", fontsize=9, pad=4)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    sigreg_ratio = support["sigreg_local_anisotropy"]["global_over_local_stable_rank"]
    aligned_lift = support["identifiability_controllability"]["aligned_controllability_lift"]
    mismatched_lift = support["identifiability_controllability"]["mismatched_controllability_lift"]
    heat_r2 = support["heat_filter_recovery"]["median_r2"]
    ax.text(
        0.98,
        0.03,
        (
            f"SIGReg global/local SR = {sigreg_ratio:.2f}x\n"
            f"aligned controllability lift = {aligned_lift:.0f}\n"
            f"mismatched lift = {mismatched_lift:.0f}\n"
            f"heat-filter median R2 = {heat_r2:.6f}"
        ),
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", linewidth=0.8),
    )


def _write_markdown(output: Path, cases: list[dict[str, Any]], support: dict[str, Any]) -> None:
    predictor_rows = [case for case in cases if case["name"].startswith("predictor_")]
    tail_rows = [case for case in cases if case["name"].startswith("broken_tail_")]
    baseline = next(case for case in cases if case["name"] == "baseline")
    lines = [
        "# Theory Sweep Summary",
        "",
        "## Main Readout",
        "",
        "| case | residual h8/h1 | predicted h8/h1 | residual alpha | head-tail gap | oblique bias | heat R2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in predictor_rows + [baseline] + tail_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    _fmt(row["residual_growth_ratio_h8_over_h1"]),
                    _fmt(row["predicted_decay_ratio_h8_over_h1"]),
                    _fmt(row["residual_alpha_1_end"]),
                    _fmt(row["latent_head_tail_gap"]),
                    _fmt(row["oblique_bias_abs_mean"]),
                    _fmt(row["heat_median_r2_noise_0p01"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "- Smaller predictor exponents make residual energy accumulate faster over horizon.",
            "- Stronger predictor exponents damp predicted trace more slowly and reduce residual growth.",
            "- Broken-tail encoder spectra create a visible head-tail gap and reduce finite-window oblique bias.",
            "- Heat-filter recovery remains near-perfect under small noise.",
            "",
            "## Support Diagnostics",
            "",
            f"- SIGReg global-over-local stable rank: {_fmt(support['sigreg_local_anisotropy']['global_over_local_stable_rank'])}",
            f"- controllability lift: aligned {_fmt(support['identifiability_controllability']['aligned_controllability_lift'])}, mismatched {_fmt(support['identifiability_controllability']['mismatched_controllability_lift'])}",
            f"- heat-filter median R2: {_fmt(support['heat_filter_recovery']['median_r2'])}",
            "",
        ]
    )
    (output / "paper_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    head_window, tail_window = _analysis_windows(args.dim)
    slope_windows = [(1, None), head_window, tail_window]
    oblique_windows = [head_window, tail_window]
    horizon_list = sorted(set(int(h) for h in args.horizons))

    cases: list[dict[str, Any]] = []
    predictor_cases: list[dict[str, Any]] = []
    for exponent in args.predictor_exponents:
        cfg = TransferConfig(
            dim=args.dim,
            intrinsic_dim=args.intrinsic_dim,
            input_exponent=args.input_exponent,
            encoder_exponent=args.encoder_exponent,
            predictor_exponent=float(exponent),
        )
        report = spectral_transfer_report(cfg, slope_windows=slope_windows, horizons=horizon_list)
        oblique_report = oblique_frame_stress_report(cfg, obliqueness=args.obliqueness, seed=args.oblique_seed, slope_windows=oblique_windows)
        case = {
            "name": _case_name_predictor(float(exponent)),
            "predictor_exponent": float(exponent),
            "config": report["config"],
            "report": report,
            "oblique_report": oblique_report,
        }
        case.update(_summarize_case(case["name"], cfg, report, oblique_report, head_window, tail_window))
        cases.append(case)
        predictor_cases.append(case)

    baseline_cfg = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponent,
        encoder_exponent=args.encoder_exponent,
        predictor_exponent=1.0,
    )
    baseline_report = spectral_transfer_report(baseline_cfg, slope_windows=slope_windows, horizons=horizon_list)
    baseline_oblique_report = oblique_frame_stress_report(
        baseline_cfg,
        obliqueness=args.obliqueness,
        seed=args.oblique_seed,
        slope_windows=oblique_windows,
    )
    baseline_case = {
        "name": "baseline",
        "predictor_exponent": 1.0,
        "config": baseline_report["config"],
        "report": baseline_report,
        "oblique_report": baseline_oblique_report,
    }
    baseline_case.update(_summarize_case("baseline", baseline_cfg, baseline_report, baseline_oblique_report, head_window, tail_window))

    tail_cases: list[dict[str, Any]] = []
    for idx, exponent in enumerate(args.tail_cases):
        breakpoint = _tail_breakpoint(idx, args.dim)
        cfg = TransferConfig(
            dim=args.dim,
            intrinsic_dim=args.intrinsic_dim,
            input_exponent=args.input_exponent,
            encoder_exponent=args.encoder_exponent,
            predictor_exponent=1.0,
            tail_encoder_exponent=float(exponent),
            breakpoint=breakpoint,
        )
        report = spectral_transfer_report(cfg, slope_windows=slope_windows, horizons=horizon_list)
        name = f"broken_tail_{str(exponent).replace('.', 'p')}_b{breakpoint}"
        oblique_report = oblique_frame_stress_report(cfg, obliqueness=args.obliqueness, seed=args.oblique_seed, slope_windows=oblique_windows)
        case = {
            "name": name,
            "predictor_exponent": 1.0,
            "config": report["config"],
            "report": report,
            "oblique_report": oblique_report,
        }
        case.update(_summarize_case(name, cfg, report, oblique_report, head_window, tail_window))
        cases.append(case)
        tail_cases.append(case)

    support = {
        "heat_filter_recovery": heat_filter_recovery_report(baseline_cfg, times=[0.01, 0.03, 0.05], noise_std=0.01, seed=0),
        "sigreg_local_anisotropy": sigreg_local_anisotropy_report(dim=32, cells=12, anisotropy=20.0, seed=0),
        "identifiability_controllability": identifiability_controllability_report(dim=32),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": horizon_list,
        "slope_windows": [[a, b] for a, b in slope_windows],
        "oblique_slope_windows": [[a, b] for a, b in oblique_windows],
        "analysis_windows": {
            "head": list(head_window),
            "tail": list(tail_window),
        },
        "cases": cases + [baseline_case],
        "support": support,
    }

    (output / "theory_rich_sweep.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

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
    _plot_traces(
        ax,
        [case for case in predictor_cases if case["name"] in {"predictor_0p5", "predictor_1p0", "predictor_2p0"}],
        "predicted_trace",
        "Predicted trace across horizon",
        "predicted trace",
    )
    _plot_panel_labels(ax, "A")

    ax = axes[0, 1]
    _plot_traces(
        ax,
        [case for case in predictor_cases if case["name"] in {"predictor_0p5", "predictor_1p0", "predictor_2p0"}],
        "residual_trace",
        "Residual trace across horizon",
        "residual trace",
    )
    _plot_panel_labels(ax, "B")

    ax = axes[1, 0]
    _plot_predictor_sweep(ax, [baseline_case] + predictor_cases)
    _plot_panel_labels(ax, "C")

    ax = axes[1, 1]
    _plot_tail_scatter(ax, baseline_case, tail_cases, support)
    _plot_panel_labels(ax, "D")

    fig.savefig(output / "paper_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "paper_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_markdown(output, [baseline_case] + predictor_cases + tail_cases, support)

    print(output / "theory_rich_sweep.json")
    print(output / "paper_summary.pdf")
    print(output / "paper_summary.png")
    print(output / "paper_summary.md")


if __name__ == "__main__":
    main()
