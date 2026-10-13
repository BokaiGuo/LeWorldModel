"""Summarize empirical spectral audits across conditioning choices.

Expected input layout:
    <input>/states.json
    <input>/actions.json
    <input>/latents.json
    <input>/global.json

All input reports should use the same latent file, deterministic sample cap,
top-k grid, and seed. The summary compares condition robustness only; it does
not turn a one-step latent artifact into multi-horizon or online-control
evidence.
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import seaborn as sns


DEFAULT_CONDITIONS = ["states", "actions", "latents", "global"]
TANGENTS = ["dynamics", "predicted_dynamics", "prediction_residual", "planning", "nuisance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/pusht/lewm/empirical_condition_sweep_20260801",
        help="directory containing one <condition>.json report per condition",
    )
    parser.add_argument("--output", default=None, help="output directory; defaults to --input")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--reference",
        default="results/pusht/lewm/empirical_spectral_audit.json",
        help="optional completed state-conditioned audit for sample-cap stability comparison",
    )
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _curve_value(curve: dict[str, Any], k: int) -> float:
    value = curve.get(str(k), float("nan")) if isinstance(curve, dict) else float("nan")
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _extract(path: Path, k: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    weighted = payload.get("weighted_mean", {})
    lifts = weighted.get("lifts", {})
    controls = payload.get("label_controls", {})
    time_shuffle = controls.get("time_shuffle", {}).get("weighted_mean", {}).get("lifts", {})
    state_shuffle = controls.get("state_cluster_shuffle", {}).get("weighted_mean", {}).get("lifts", {})
    controllability = payload.get("controllability", {})
    uncertainty = payload.get("uncertainty_residual", {}).get("prediction_residual", {})
    multi_horizon = payload.get("multi_horizon", {})
    config = payload.get("config", {})
    horizon_lifts = {
        name: {
            "lift": _curve_value(info.get("lift", {}), k),
            "predicted_cosine": info.get("predicted_cosine"),
            "norm": info.get("norm"),
        }
        for name, info in multi_horizon.items()
        if isinstance(info, dict) and "lift" in info
    }
    return {
        "path": str(path),
        "condition": payload.get("condition", path.stem),
        "num_samples": config.get("num_samples"),
        "num_cells": config.get("num_cells"),
        "ks": config.get("ks", []),
        "lifts": {name: _curve_value(lifts.get(name, {}), k) for name in TANGENTS},
        "lift_curves": {name: lifts.get(name, {}) for name in TANGENTS if name in lifts},
        "time_shuffle_dynamics": _curve_value(time_shuffle.get("dynamics", {}), k),
        "state_shuffle_dynamics": _curve_value(state_shuffle.get("dynamics", {}), k),
        "action_to_dynamics_r2": controllability.get("action_to_dynamics_r2"),
        "controllable_lift": _curve_value(controllability.get("controllable_lift", {}), k),
        "unexplained_lift": _curve_value(controllability.get("unexplained_lift", {}), k),
        "prediction_mean_cosine": weighted.get("prediction_spectral_error", {}).get("mean_cosine"),
        "residual_uncertainty_lift": _curve_value(uncertainty.get("lift", {}), k),
        "multi_horizon": horizon_lifts,
        "warnings": payload.get("warnings", []),
    }


def _sample_stability(records: list[dict[str, Any]], reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    sample = next((record for record in records if record["condition"] == "states"), None)
    if sample is None or reference["condition"] != "states":
        return None
    ratio = {
        tangent: float(sample["lifts"][tangent] / max(reference["lifts"][tangent], 1e-12))
        for tangent in TANGENTS
    }
    return {
        "sample_path": sample["path"],
        "reference_path": reference["path"],
        "sample_num_samples": sample["num_samples"],
        "reference_num_samples": reference["num_samples"],
        "sample_lifts": sample["lifts"],
        "reference_lifts": reference["lifts"],
        "lift_ratio": ratio,
        "prediction_cosine_ratio": float(
            sample["prediction_mean_cosine"] / max(float(reference["prediction_mean_cosine"]), 1e-12)
        ),
        "action_to_dynamics_r2_ratio": float(
            sample["action_to_dynamics_r2"] / max(float(reference["action_to_dynamics_r2"]), 1e-12)
        ),
    }


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
    ax.grid(True, axis="y", alpha=0.2)
    ax.tick_params(labelsize=8)


def _write_markdown(
    output: Path,
    records: list[dict[str, Any]],
    k: int,
    sample_stability: dict[str, Any] | None,
) -> None:
    warnings = sorted({warning for record in records for warning in record["warnings"]})
    extra_horizons = sorted(
        {name for record in records for name in record["multi_horizon"] if name != "dynamics"}
    )
    lines = [
        "# PushT Empirical Condition Sweep",
        "",
        "## Setup",
        "",
        f"- Top-k readout: k={k}",
        f"- Conditions: {', '.join(record['condition'] for record in records)}",
        f"- Shared deterministic samples per report: {', '.join(str(record['num_samples']) for record in records)}",
        "",
        "## Condition Robustness",
        "",
        "| condition | cells | dynamics lift | predicted lift | residual lift | planning lift | nuisance lift | pred cosine |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["condition"]),
                    _fmt(record["num_cells"]),
                    _fmt(record["lifts"]["dynamics"]),
                    _fmt(record["lifts"]["predicted_dynamics"]),
                    _fmt(record["lifts"]["prediction_residual"]),
                    _fmt(record["lifts"]["planning"]),
                    _fmt(record["lifts"]["nuisance"]),
                    _fmt(record["prediction_mean_cosine"]),
                ]
            )
            + " |"
        )
    if sample_stability:
        lines.extend(
            [
                "",
                "## Sample-Cap Stability",
                "",
                f"- State-conditioned reference: {sample_stability['reference_num_samples']} rows; condition sweep: {sample_stability['sample_num_samples']} rows.",
                "",
                "| tangent | 100k state lift | 200k state lift | 100k / 200k |",
                "| --- | --- | --- | --- |",
            ]
        )
        for tangent in TANGENTS:
            lines.append(
                "| "
                + " | ".join(
                    [
                        tangent,
                        _fmt(sample_stability["sample_lifts"][tangent]),
                        _fmt(sample_stability["reference_lifts"][tangent]),
                        _fmt(sample_stability["lift_ratio"][tangent]),
                    ]
                )
                + " |"
            )
        lines.append(
            f"- Prediction-cosine ratio: {_fmt(sample_stability['prediction_cosine_ratio'])}; action-to-dynamics-R2 ratio: {_fmt(sample_stability['action_to_dynamics_r2_ratio'])}."
        )
    lines.extend(
        [
            "",
            "## Controls and Action Decomposition",
            "",
        "| condition | time-shuffle dynamics | state-shuffle dynamics | controllable lift | unexplained lift | action-to-dynamics R2 | residual uncertainty lift |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    )
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["condition"]),
                    _fmt(record["time_shuffle_dynamics"]),
                    _fmt(record["state_shuffle_dynamics"]),
                    _fmt(record["controllable_lift"]),
                    _fmt(record["unexplained_lift"]),
                    _fmt(record["action_to_dynamics_r2"]),
                    _fmt(record["residual_uncertainty_lift"]),
                ]
            )
            + " |"
        )
    if extra_horizons:
        lines.extend(
            [
                "",
                "## Multi-Horizon Dynamics",
                "",
                "| condition | h1 lift | "
                + " | ".join(f"h{name.split('_h', 1)[1]} lift" for name in extra_horizons)
                + " | "
                + " | ".join(f"h{name.split('_h', 1)[1]} / h1" for name in extra_horizons)
                + " |",
                "| --- | --- | "
                + " | ".join("---" for _ in extra_horizons)
                + " | "
                + " | ".join("---" for _ in extra_horizons)
                + " |",
            ]
        )
        for record in records:
            base = record["multi_horizon"].get("dynamics", {})
            base_lift = base.get("lift", float("nan"))
            row = [str(record["condition"]), _fmt(base_lift)]
            for name in extra_horizons:
                lift = record["multi_horizon"].get(name, {}).get("lift", float("nan"))
                row.append(_fmt(lift))
            for name in extra_horizons:
                lift = record["multi_horizon"].get(name, {}).get("lift", float("nan"))
                row.append(_fmt(lift / max(base_lift, 1e-12)))
            lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Data Limits",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This compares conditioning choices on one frozen PushT latent artifact and a shared deterministic row sample.",
            "- It supports or weakens condition robustness of offline eigenspace readouts only.",
            "- Multi-horizon rows are included only when `dynamics_h*` arrays are present in the source audits.",
            "- Missing physical and counterfactual arrays mean observability and impossible-transition controls remain untested.",
            "- Existing online evaluation and planning results remain blocked until the PushT HDF5 dataset is restored.",
            "",
        ]
    )
    (output / "condition_summary.md").write_text("\n".join(lines))


def _write_figure(
    output: Path,
    records: list[dict[str, Any]],
    k: int,
    sample_stability: dict[str, Any] | None,
) -> None:
    labels = [record["condition"] for record in records]
    x = np.arange(len(labels))
    palette = sns.color_palette("colorblind", n_colors=max(len(records), len(TANGENTS)))
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
    width = 0.15
    offsets = np.linspace(-2 * width, 2 * width, len(TANGENTS))
    for color, offset, tangent in zip(palette, offsets, TANGENTS):
        ax.bar(
            x + offset,
            [record["lifts"][tangent] for record in records],
            width,
            color=color,
            label=tangent.replace("prediction_", "pred. "),
        )
    ax.set_xticks(x, labels)
    ax.set_title(f"Conditioned lift at k={k}", fontsize=9, pad=4)
    ax.set_ylabel("lift")
    _style(ax)
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    _panel_label(ax, "A")

    ax = axes[0, 1]
    for color, record in zip(palette, records):
        curve = record["lift_curves"].get("dynamics", {})
        ks = sorted(int(key) for key in curve)
        ax.plot(
            ks,
            [_curve_value(curve, value) for value in ks],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            color=color,
            label=record["condition"],
        )
    ax.set_title("Dynamics lift curve", fontsize=9, pad=4)
    ax.set_xlabel("top-k")
    ax.set_ylabel("lift")
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _panel_label(ax, "B")

    ax = axes[0, 2]
    width = 0.24
    ax.bar(x - width, [record["lifts"]["dynamics"] for record in records], width, label="observed")
    ax.bar(x, [record["time_shuffle_dynamics"] for record in records], width, label="time shuffle")
    ax.bar(x + width, [record["state_shuffle_dynamics"] for record in records], width, label="state shuffle")
    ax.set_xticks(x, labels)
    ax.set_title(f"Dynamics controls at k={k}", fontsize=9, pad=4)
    ax.set_ylabel("lift")
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _panel_label(ax, "C")

    ax = axes[1, 0]
    width = 0.22
    ax.bar(x - width, [record["controllable_lift"] for record in records], width, label="controllable")
    ax.bar(x, [record["unexplained_lift"] for record in records], width, label="unexplained")
    ax.bar(x + width, [record["residual_uncertainty_lift"] for record in records], width, label="residual")
    ax.set_xticks(x, labels)
    ax.set_title(f"Action and residual split at k={k}", fontsize=9, pad=4)
    ax.set_ylabel("lift")
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _panel_label(ax, "D")

    ax = axes[1, 1]
    values = np.array(
        [
            [record["lifts"][tangent] for tangent in TANGENTS]
            for record in records
        ]
    )
    baseline = values[labels.index("global")] if "global" in labels else values.mean(axis=0)
    ratio = values / np.maximum(baseline, 1e-12)
    vmax = max(1.2, float(np.nanmax(ratio)))
    norm = TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=vmax)
    image = ax.imshow(ratio, cmap="coolwarm", aspect="auto", norm=norm)
    ax.set_xticks(np.arange(len(TANGENTS)), [name.replace("prediction_", "pred. ") for name in TANGENTS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Lift relative to global", fontsize=9, pad=4)
    for row in range(ratio.shape[0]):
        for col in range(ratio.shape[1]):
            ax.text(col, row, f"{ratio[row, col]:.2f}", ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=7)
    _panel_label(ax, "E")

    ax = axes[1, 2]
    if sample_stability:
        ratios = [sample_stability["lift_ratio"][tangent] for tangent in TANGENTS]
        ax.bar(np.arange(len(TANGENTS)), ratios, color=palette[: len(TANGENTS)])
        ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xticks(
            np.arange(len(TANGENTS)),
            [name.replace("prediction_", "pred. ") for name in TANGENTS],
            rotation=25,
            ha="right",
        )
        lo = min(ratios)
        hi = max(ratios)
        pad = max((hi - lo) * 0.35, 0.01)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title("100k / 200k state-audit lift", fontsize=9, pad=4)
        ax.set_ylabel("lift ratio")
        _style(ax)
    else:
        r2_values = [
            float(record["action_to_dynamics_r2"])
            if isinstance(record["action_to_dynamics_r2"], (int, float))
            else float("nan")
            for record in records
        ]
        cosine_values = [
            float(record["prediction_mean_cosine"])
            if isinstance(record["prediction_mean_cosine"], (int, float))
            else float("nan")
            for record in records
        ]
        ax.plot(x, r2_values, marker="o", linewidth=1.8, markersize=4, label="action-to-dynamics R2")
        ax.plot(x, cosine_values, marker="s", linewidth=1.8, markersize=4, label="prediction cosine")
        ax.set_xticks(x, labels)
        ax.set_ylim(bottom=0)
        ax.set_title("Shared global readouts", fontsize=9, pad=4)
        ax.set_ylabel("value")
        _style(ax)
        ax.legend(frameon=False, fontsize=7, loc="best")
    _panel_label(ax, "F")

    fig.savefig(output / "condition_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "condition_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output = Path(args.output) if args.output else input_dir
    output.mkdir(parents=True, exist_ok=True)
    conditions = list(dict.fromkeys(args.conditions))
    records = []
    for condition in conditions:
        path = input_dir / f"{condition}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing condition report: {path}")
        records.append(_extract(path, args.k))
    reference_path = Path(args.reference) if args.reference else None
    reference = _extract(reference_path, args.k) if reference_path and reference_path.is_file() else None
    sample_stability = _sample_stability(records, reference)

    payload = {
        "input": str(input_dir),
        "conditions": conditions,
        "k": args.k,
        "records": records,
        "multi_horizon": {record["condition"]: record["multi_horizon"] for record in records},
        "reference": reference,
        "sample_stability": sample_stability,
    }
    (output / "condition_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(output, records, args.k, sample_stability)
    _write_figure(output, records, args.k, sample_stability)
    print(output / "condition_summary.json")
    print(output / "condition_summary.md")
    print(output / "condition_summary.pdf")
    print(output / "condition_summary.png")


if __name__ == "__main__":
    main()
