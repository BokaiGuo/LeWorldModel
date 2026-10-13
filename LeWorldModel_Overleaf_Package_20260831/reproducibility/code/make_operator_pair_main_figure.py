"""Generate the manuscript Figure 1 for the isospectral operator-pair result.

The figure follows the morphology of a formal theory/representation-learning
paper: three compact axes, a white background, restrained annotations, and no
figure-internal headline or conclusion cards. Every quantitative element is
read from the stored same-spectrum sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.theory_spectral_transfer import TransferConfig, diagonal_transfer_spectra


BLUE = "#3B6FB6"
ORANGE = "#D55E00"
GREEN = "#238B7B"
INK = "#202124"
MID_GRAY = "#6B7280"
LIGHT_GRAY = "#D7DADF"
BAND_GRAY = "#D9DDE3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-json",
        default="results/theory_operator_alignment_sweep_20260828/operator_alignment_sweep.json",
        help="Stored same-spectrum rotation sweep.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/theory_operator_alignment_iclr_oral_20260829/figures",
        help="Directory for manuscript and editable outputs.",
    )
    return parser.parse_args()


def _config_from_payload(payload: dict) -> TransferConfig:
    return TransferConfig(**payload["config"])


def _theorem_curve(angles: np.ndarray, peak: float) -> np.ndarray:
    return float(peak) * np.abs(np.sin(2.0 * np.deg2rad(angles)))


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8A8F98")
        ax.spines[side].set_linewidth(0.65)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.6,
        width=0.6,
        color="#6F747C",
        labelcolor=INK,
        pad=2.0,
    )
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.16,
        1.155,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        -0.02,
        1.155,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        fontweight="semibold",
        color=INK,
        clip_on=False,
    )


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.sweep_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(sweep_path.read_text())
    records = payload["records"]
    angles = np.asarray([row["angle_degrees"] for row in records], dtype=float)
    commutator = np.asarray([row["commutator_norm"] for row in records], dtype=float)
    top32_h1 = np.asarray(
        [row["horizons"]["1"]["predicted_band_lift"]["32"] for row in records],
        dtype=float,
    )
    residual_growth = np.asarray(
        [row["residual_growth_ratio"] for row in records],
        dtype=float,
    )
    random_null = payload["random_null"]["commutator_norm"]
    peak = float(payload["contrast"]["peak_oblique_commutator_norm"])

    spectra = diagonal_transfer_spectra(_config_from_payload(payload))
    predictor_eigenvalues = np.asarray(spectra["predictor_response"], dtype=float)
    ranks = np.arange(1, predictor_eigenvalues.size + 1)
    shown = ranks <= 32

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.24),
        gridspec_kw={"width_ratios": [0.90, 1.08, 1.06], "wspace": 0.43},
        facecolor="white",
    )
    ax_spec, ax_comm, ax_transfer = axes

    ax_spec.plot(
        ranks[shown],
        predictor_eigenvalues[shown],
        color=BLUE,
        linewidth=1.8,
        label=r"$P_{0^\circ}$",
    )
    ax_spec.plot(
        ranks[shown],
        predictor_eigenvalues[shown],
        color=ORANGE,
        linewidth=1.2,
        linestyle=(0, (4, 2.2)),
        label=r"$P_{45^\circ}$",
    )
    ax_spec.set_xlim(1, 32)
    ax_spec.set_ylim(0.14, 1.03)
    ax_spec.set_xticks([1, 8, 16, 24, 32])
    ax_spec.set_yticks([0.2, 0.6, 1.0])
    ax_spec.set_xlabel("Eigenvalue rank")
    ax_spec.set_ylabel("Predictor eigenvalue")
    ax_spec.legend(
        frameon=False,
        loc="upper right",
        handlelength=2.2,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    ax_spec.text(
        0.05,
        0.09,
        r"$\mathrm{spec}(P_{0^\circ})=\mathrm{spec}(P_{45^\circ})$",
        transform=ax_spec.transAxes,
        fontsize=6.4,
        color=INK,
        ha="left",
        va="bottom",
    )
    _style_axis(ax_spec)
    _panel_label(ax_spec, "(a)", "Isospectral predictors")

    fine_angles = np.linspace(0.0, 90.0, 361)
    ax_comm.axhspan(
        random_null["p05"],
        random_null["p95"],
        color=BAND_GRAY,
        alpha=0.75,
        linewidth=0,
        label="Random basis (5–95%)",
        zorder=0,
    )
    ax_comm.axhline(
        random_null["mean"],
        color=MID_GRAY,
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax_comm.plot(
        fine_angles,
        _theorem_curve(fine_angles, peak),
        color=BLUE,
        linewidth=1.55,
        label="Identity",
        zorder=2,
    )
    ax_comm.plot(
        angles,
        commutator,
        linestyle="none",
        marker="o",
        markersize=3.5,
        markerfacecolor=ORANGE,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="Sweep",
        zorder=3,
    )
    ax_comm.axvline(45.0, color="#A0A4AA", linewidth=0.6, linestyle=(0, (2, 2)))
    ax_comm.annotate(
        f"{peak:.3f}",
        xy=(45.0, peak),
        xytext=(56.0, 0.207),
        fontsize=6.0,
        color=ORANGE,
        arrowprops={"arrowstyle": "-", "lw": 0.55, "color": ORANGE},
        ha="left",
        va="center",
    )
    ax_comm.set_xlim(0, 90)
    ax_comm.set_ylim(-0.006, 0.246)
    ax_comm.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax_comm.set_yticks([0.0, 0.1, 0.2])
    ax_comm.set_xlabel(r"Eigenspace rotation $\theta$ (degrees)")
    ax_comm.set_ylabel(r"Normalized commutator $\mathcal{C}(\theta)$")
    handles, labels = ax_comm.get_legend_handles_labels()
    order = [1, 2, 0]
    ax_comm.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=0.9,
        borderaxespad=0.0,
        labelspacing=0.2,
        fontsize=5.5,
    )
    _style_axis(ax_comm)
    _panel_label(ax_comm, "(b)", "Operator-pair geometry")

    ax_transfer.plot(
        angles,
        top32_h1,
        color=BLUE,
        linewidth=1.45,
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=r"Top-band lift ($h=1$, $k=32$)",
    )
    ax_transfer.plot(
        angles,
        residual_growth,
        color=GREEN,
        linewidth=1.45,
        marker="s",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="Residual-growth ratio",
    )
    ax_transfer.axvline(45.0, color="#A0A4AA", linewidth=0.6, linestyle=(0, (2, 2)))
    ax_transfer.set_xlim(0, 90)
    ax_transfer.set_ylim(1.5, 2.32)
    ax_transfer.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax_transfer.set_yticks([1.6, 1.8, 2.0, 2.2])
    ax_transfer.set_xlabel(r"Eigenspace rotation $\theta$ (degrees)")
    ax_transfer.set_ylabel("Dimensionless readout")
    ax_transfer.legend(
        frameon=False,
        loc="upper right",
        handlelength=2.0,
        borderaxespad=0.15,
        labelspacing=0.3,
    )
    ax_transfer.text(
        45.0,
        1.515,
        r"$45^\circ$",
        fontsize=5.8,
        color=MID_GRAY,
        ha="center",
        va="bottom",
    )
    _style_axis(ax_transfer)
    _panel_label(ax_transfer, "(c)", "Transfer readouts")

    fig.subplots_adjust(left=0.080, right=0.992, bottom=0.245, top=0.835)

    png_path = out_dir / "operator_pair_main_figure.png"
    pdf_path = out_dir / "operator_pair_main_figure.pdf"
    svg_path = out_dir / "operator_pair_main_figure.svg"
    figma_svg_path = out_dir / "operator_pair_main_figure_figma.svg"

    fig.savefig(png_path, dpi=320, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.025)

    plt.rcParams["svg.fonttype"] = "path"
    fig.savefig(figma_svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    print(png_path)
    print(pdf_path)
    print(svg_path)
    print(figma_svg_path)


if __name__ == "__main__":
    main()
