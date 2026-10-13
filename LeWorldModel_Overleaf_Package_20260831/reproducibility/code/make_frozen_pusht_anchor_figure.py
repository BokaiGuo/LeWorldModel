"""Generate the bounded frozen-PushT latent anchor figure.

The figure is intentionally compact: it supports learned-representation
relevance for alignment-sensitive readouts, not a learned-dynamics theorem or
an online-planning claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#3B6FB6"
ORANGE = "#D55E00"
GREEN = "#238B7B"
PURPLE = "#6F63B6"
INK = "#202124"
MID_GRAY = "#6B7280"
LIGHT_GRAY = "#D7DADF"
BAND_GRAY = "#E3E6EB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge-json",
        default=(
            "results/pusht/lewm/empirical_operator_alignment_bridge_20260829/"
            "empirical_operator_alignment_bridge.json"
        ),
        help="Stored frozen PushT latent bridge measurements.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/theory_operator_alignment_iclr_oral_20260829/figures",
        help="Directory for manuscript figure outputs.",
    )
    return parser.parse_args()


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8A8F98")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.4,
        width=0.55,
        color="#6F747C",
        labelcolor=INK,
        pad=1.8,
    )
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.42, alpha=0.62)
    ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(
        f"{label} {title}",
        loc="left",
        fontsize=7.0,
        fontweight="semibold",
        color=INK,
        pad=6,
    )


def _short_label(name: str) -> str:
    return {
        "dynamics": "Dynamics",
        "planning": "Planner",
        "predicted_dynamics": "Pred.",
        "prediction_residual": "Residual",
    }[name]


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.bridge_json).read_text())
    bridge = payload["bridge"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    readout_keys = [
        "dynamics",
        "planning",
        "predicted_dynamics",
        "prediction_residual",
    ]
    observed = np.asarray([bridge["primary_lifts"][key] for key in readout_keys])
    random_control = np.asarray(
        [bridge["control_lifts"]["norm_matched_random_dynamics_lift"] for _ in readout_keys]
    )
    action_shuffle = np.asarray(
        [bridge["label_shuffle_lifts"]["action_shuffle"][key] for key in readout_keys]
    )
    state_shuffle = np.asarray(
        [bridge["label_shuffle_lifts"]["state_cluster_shuffle"][key] for key in readout_keys]
    )
    time_shuffle = np.asarray(
        [bridge["label_shuffle_lifts"]["time_shuffle"][key] for key in readout_keys]
    )

    overlap_observed = bridge["transport"]["observed"]["mean"]
    overlap_null = bridge["transport"]["null"]["mean"]
    overlap_null_p05 = bridge["transport"]["null"]["p05"]
    overlap_null_p95 = bridge["transport"]["null"]["p95"]
    top_weight = bridge["planner_weights"]["top_weight_mean"]
    tail_weight = bridge["planner_weights"]["tail_weight_mean"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 6.8,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.05, 1.70),
        gridspec_kw={"width_ratios": [1.50, 1.10, 0.82, 0.78], "wspace": 0.66},
        facecolor="white",
    )
    ax_lift, ax_shuffle, ax_overlap, ax_planner = axes

    x = np.arange(len(readout_keys))
    width = 0.36
    ax_lift.bar(x - width / 2, observed, width=width, color=GREEN, label="Observed")
    ax_lift.bar(
        x + width / 2,
        random_control,
        width=width,
        color=PURPLE,
        alpha=0.82,
        label="Norm-matched random",
    )
    ax_lift.set_xticks(x)
    ax_lift.set_xticklabels([_short_label(key) for key in readout_keys], rotation=20, ha="right")
    ax_lift.set_ylabel(r"Top-$32$ lift")
    ax_lift.set_ylim(0, 5.25)
    ax_lift.set_yticks([0, 2, 4])
    ax_lift.legend(frameon=False, loc="upper right", handlelength=1.3)
    _style_axis(ax_lift)
    _panel_label(ax_lift, "(a)", "Readout lifts")

    shuffle_labels = ["Obs.", "Action", "State", "Time"]
    shuffle_values = [
        bridge["primary_lifts"]["dynamics"],
        bridge["label_shuffle_lifts"]["action_shuffle"]["dynamics"],
        bridge["label_shuffle_lifts"]["state_cluster_shuffle"]["dynamics"],
        bridge["label_shuffle_lifts"]["time_shuffle"]["dynamics"],
    ]
    ax_shuffle.bar(
        np.arange(len(shuffle_values)),
        shuffle_values,
        color=[GREEN, BAND_GRAY, BAND_GRAY, BAND_GRAY],
        edgecolor=[GREEN, "#AEB4BE", "#AEB4BE", "#AEB4BE"],
        linewidth=0.6,
    )
    ax_shuffle.set_xticks(np.arange(len(shuffle_values)))
    ax_shuffle.set_xticklabels(shuffle_labels)
    ax_shuffle.set_ylabel(r"Dynamics lift")
    ax_shuffle.set_ylim(0, 5.25)
    ax_shuffle.set_yticks([0, 2, 4])
    _style_axis(ax_shuffle)
    _panel_label(ax_shuffle, "(b)", "Shuffle controls")

    ax_overlap.bar([0], [overlap_observed], width=0.48, color=GREEN, label="Observed")
    ax_overlap.bar([1], [overlap_null], width=0.48, color=PURPLE, alpha=0.82, label="Null mean")
    ax_overlap.errorbar(
        [1],
        [overlap_null],
        yerr=[[overlap_null - overlap_null_p05], [overlap_null_p95 - overlap_null]],
        color=INK,
        capsize=2.2,
        linewidth=0.7,
        fmt="none",
        zorder=4,
    )
    ax_overlap.set_xticks([0, 1])
    ax_overlap.set_xticklabels(["Observed", "Random\nsubspace"])
    ax_overlap.set_ylabel("Transition overlap")
    ax_overlap.set_ylim(0, 0.040)
    ax_overlap.set_yticks([0.00, 0.02, 0.04])
    _style_axis(ax_overlap)
    _panel_label(ax_overlap, "(c)", "Transition overlap")

    ax_planner.bar([0, 1], [top_weight, tail_weight], width=0.5, color=[GREEN, MID_GRAY])
    ax_planner.set_xticks([0, 1])
    ax_planner.set_xticklabels(["Top", "Tail"])
    ax_planner.set_ylabel("Planner weight")
    ax_planner.set_ylim(0, 1.35)
    ax_planner.set_yticks([0.0, 0.6, 1.2])
    ratio = top_weight / tail_weight
    ax_planner.text(
        0.50,
        0.93,
        f"{ratio:.2f}x",
        transform=ax_planner.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color=INK,
    )
    _style_axis(ax_planner)
    _panel_label(ax_planner, "(d)", "Planner weights")

    for suffix in ("pdf", "png", "svg"):
        path = out_dir / f"frozen_pusht_latent_anchor.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if suffix == "png" else None)
    plt.close(fig)
    print(f"wrote {out_dir / 'frozen_pusht_latent_anchor.pdf'}")


if __name__ == "__main__":
    main()
