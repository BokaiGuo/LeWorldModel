"""Draw the LeWorldModel-grounded introduction schematic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.theory_spectral_transfer import TransferConfig, diagonal_transfer_spectra


BLUE = "#2F6FA5"
BLUE_DARK = "#22557E"
BLUE_PALE = "#E8F1F7"
ORANGE = "#C85B22"
GREEN = "#16856F"
GREEN_PALE = "#E6F3EF"
RED = "#C94843"
GRAY = "#6B7280"
LIGHT = "#D7DEE5"
INK = "#20252B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-json",
        default="results/theory_operator_alignment_sweep_20260828/operator_alignment_sweep.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/theory_operator_alignment_iclr_oral_20260829/figures",
    )
    return parser.parse_args()


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    linewidth: float = 1.0,
    linestyle: str = "-",
    mutation_scale: float = 7.0,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def _capsule(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str,
    edgecolor: str,
    textcolor: str = INK,
    fontsize: float = 6.1,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.75,
            zorder=4,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        fontsize=fontsize,
        color=textcolor,
        ha="center",
        va="center",
        zorder=5,
    )


def _sim_observation(
    ax: plt.Axes,
    image_path: Path,
    extent: tuple[float, float, float, float],
) -> None:
    x0, x1, y0, y1 = extent
    image = mpimg.imread(image_path)
    # Tight crop keeps the simulator state legible at two-column paper scale.
    crop = max(1, int(round(image.shape[0] * 0.15)))
    image = image[crop:-crop, crop:-crop]
    ax.imshow(image, extent=extent, interpolation="lanczos", zorder=1, aspect="auto")
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.002,rounding_size=0.025",
            facecolor="none",
            edgecolor="#B7C0CA",
            linewidth=0.7,
            zorder=2,
        )
    )


def _latent_cloud(
    ax: plt.Axes,
    center: tuple[float, float],
    *,
    scale: float = 1.0,
    predicted: bool = False,
) -> None:
    cx, cy = center
    edge = GREEN if predicted else BLUE
    fill = GREEN_PALE if predicted else BLUE_PALE
    ax.add_patch(
        Ellipse(
            center,
            width=0.105 * scale,
            height=0.155 * scale,
            angle=18,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.75,
            zorder=3,
        )
    )
    offsets = np.array(
        [
            [-0.027, -0.038],
            [-0.012, 0.026],
            [0.020, -0.016],
            [0.031, 0.041],
            [-0.032, 0.049],
            [0.007, 0.061],
        ]
    ) * scale
    ax.scatter(
        cx + offsets[:, 0],
        cy + offsets[:, 1],
        s=4.2 * scale,
        facecolor="white",
        edgecolor=edge,
        linewidth=0.45,
        zorder=4,
    )


def _double_axis(
    ax: plt.Axes,
    center: tuple[float, float],
    angle_degrees: float,
    length: float,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    zorder: int = 5,
) -> None:
    theta = np.deg2rad(angle_degrees)
    delta = np.array([np.cos(theta), np.sin(theta)]) * length
    c = np.asarray(center)
    ax.add_patch(
        FancyArrowPatch(
            tuple(c - delta),
            tuple(c + delta),
            arrowstyle="<->",
            mutation_scale=6.5,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def _line_through(
    ax: plt.Axes,
    center: tuple[float, float],
    angle_degrees: float,
    length: float,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 4,
) -> None:
    theta = np.deg2rad(angle_degrees)
    delta = np.array([np.cos(theta), np.sin(theta)]) * length
    c = np.asarray(center)
    ax.plot(
        [c[0] - delta[0], c[0] + delta[0]],
        [c[1] - delta[1], c[1] + delta[1]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=zorder,
    )


def _operator_lens(
    ax: plt.Axes,
    *,
    aligned_output: np.ndarray,
    oblique_output: np.ndarray,
) -> None:
    cx, cy = 0.79, 0.55
    ax.plot([0.56, 0.56], [0.105, 0.91], color=LIGHT, linewidth=0.65, zorder=0)
    ax.text(
        cx,
        0.955,
        "same eigenvalues, different eigenspaces",
        fontsize=5.6,
        fontweight="semibold",
        color=INK,
        ha="center",
        va="center",
    )
    ax.text(
        cx,
        0.89,
        r"$\operatorname{spec}(P_0)=\operatorname{spec}(P_{45})$",
        fontsize=7.2,
        color=INK,
        ha="center",
        va="center",
    )
    ax.add_patch(
        Ellipse(
            (cx, cy),
            width=0.27,
            height=0.36,
            facecolor=BLUE_PALE,
            edgecolor=BLUE,
            linewidth=0.85,
            zorder=1,
        )
    )
    for width, height, alpha in ((0.205, 0.27, 0.42), (0.135, 0.18, 0.28)):
        ax.add_patch(
            Ellipse(
                (cx, cy),
                width=width,
                height=height,
                facecolor="none",
                edgecolor=BLUE,
                linewidth=0.48,
                alpha=alpha,
                zorder=2,
            )
        )

    _line_through(ax, (cx, cy), 0, 0.145, color=BLUE, linewidth=0.72, alpha=0.9)
    _line_through(ax, (cx, cy), 90, 0.178, color=BLUE, linewidth=0.72, alpha=0.9)
    _line_through(ax, (cx, cy), 45, 0.145, color=ORANGE, linewidth=1.0, linestyle="--")
    _line_through(ax, (cx, cy), 135, 0.13, color=ORANGE, linewidth=1.0, linestyle="--")
    ax.text(
        0.665,
        0.755,
        r"latent covariance $\Sigma_z$",
        fontsize=4.8,
        color=BLUE_DARK,
        ha="left",
    )
    ax.text(
        0.915,
        0.755,
        r"eigenbasis of $P_{45}$",
        fontsize=4.8,
        color=ORANGE,
        ha="right",
    )
    ax.text(cx + 0.153, cy - 0.018, r"$u_1$", fontsize=5.3, color=BLUE, ha="left")
    ax.text(cx + 0.006, cy + 0.185, r"$u_2$", fontsize=5.3, color=BLUE, ha="left")

    aligned = np.asarray(aligned_output, dtype=float)
    aligned = aligned / max(np.linalg.norm(aligned), 1e-12) * 0.12
    oblique = np.asarray(oblique_output, dtype=float)
    oblique = oblique / max(np.linalg.norm(oblique), 1e-12) * 0.12
    _arrow(
        ax,
        (cx, cy),
        (cx + aligned[0], cy + aligned[1]),
        color=BLUE_DARK,
        linewidth=1.35,
        mutation_scale=7.5,
        zorder=7,
    )
    _arrow(
        ax,
        (cx, cy),
        (cx + oblique[0], cy + oblique[1]),
        color=GREEN,
        linewidth=1.65,
        mutation_scale=8.0,
        zorder=8,
    )
    ax.scatter([cx], [cy], s=8, facecolor=INK, edgecolor="white", linewidth=0.4, zorder=9)
    ax.text(
        cx + aligned[0] - 0.003,
        cy - 0.067,
        r"$P_0u_1$",
        fontsize=4.7,
        color=BLUE_DARK,
        ha="center",
    )
    ax.text(
        cx + oblique[0] + 0.002,
        cy + oblique[1] + 0.052,
        r"$P_{45}u_1$",
        fontsize=4.7,
        color=GREEN,
        ha="center",
    )

    ax.text(
        0.69,
        0.255,
        "shared eigenbasis",
        fontsize=5.15,
        fontweight="semibold",
        color=BLUE_DARK,
        ha="center",
        va="center",
    )
    ax.text(
        0.69,
        0.178,
        r"$[\Sigma_z,P_0]=0$",
        fontsize=6.3,
        color=BLUE_DARK,
        ha="center",
        va="center",
    )
    ax.text(
        0.69,
        0.125,
        "no cross-axis mixing",
        fontsize=4.65,
        color=INK,
        ha="center",
        va="center",
    )
    ax.text(
        0.89,
        0.255,
        "rotated eigenbasis",
        fontsize=5.15,
        fontweight="semibold",
        color=ORANGE,
        ha="center",
        va="center",
    )
    ax.text(
        0.89,
        0.178,
        r"$[\Sigma_z,P_{45}]\ne0$",
        fontsize=6.3,
        color=GREEN,
        ha="center",
        va="center",
    )
    ax.text(
        0.89,
        0.125,
        "cross-axis transport",
        fontsize=4.65,
        color=INK,
        ha="center",
        va="center",
    )


def _assert_no_text_collisions(fig: plt.Figure) -> None:
    """Fail generation if any independently placed text boxes overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        item
        for axis in fig.axes
        for item in axis.texts
        if item.get_visible() and item.get_text().strip()
    ]
    boxes = [item.get_window_extent(renderer=renderer) for item in texts]
    collisions: list[str] = []
    for i, first in enumerate(texts):
        for j in range(i + 1, len(texts)):
            if boxes[i].overlaps(boxes[j]):
                collisions.append(f"{first.get_text()!r} overlaps {texts[j].get_text()!r}")
    if collisions:
        raise RuntimeError("Text collision audit failed: " + "; ".join(collisions))


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.sweep_json).read_text())
    config = TransferConfig(**payload["config"])
    spectra = diagonal_transfer_spectra(config)

    predictor = np.asarray(spectra["predictor_response"], dtype=float)
    lambda_i = float(predictor[0])
    lambda_j = float(predictor[-1])
    theta = np.deg2rad(45.0)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    diagonal = np.diag([lambda_i, lambda_j])
    z = np.array([1.0, 0.0], dtype=float)
    aligned_output = diagonal @ z
    oblique_output = rotation @ diagonal @ rotation.T @ z

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.32), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    out_dir = Path(args.output_dir)
    # Deterministic PushT simulator renders identify the LeWM pixel-to-latent
    # setting without implying that this paper reports real-robot experiments.
    sim_t = out_dir / "pusht_intro_ot.png"
    sim_t1 = out_dir / "pusht_intro_ot1.png"
    if not sim_t.exists() or not sim_t1.exists():
        raise FileNotFoundError(
            "Run render_pusht_intro_frames.py with the project virtual environment first"
        )
    _sim_observation(ax, sim_t, (0.025, 0.126, 0.10, 0.40))
    _sim_observation(ax, sim_t1, (0.435, 0.535, 0.10, 0.40))
    ax.text(0.075, 0.055, r"$o_t$", fontsize=6.6, color=INK, ha="center")
    ax.text(0.485, 0.055, r"$o_{t+1}$", fontsize=6.6, color=INK, ha="center")

    _capsule(
        ax,
        (0.039, 0.465),
        0.072,
        0.105,
        "E",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        textcolor=BLUE_DARK,
        fontsize=7.1,
    )
    _capsule(
        ax,
        (0.449, 0.465),
        0.072,
        0.105,
        "E",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        textcolor=BLUE_DARK,
        fontsize=7.1,
    )
    _arrow(ax, (0.075, 0.405), (0.075, 0.458), color=BLUE, linewidth=0.95)
    _arrow(ax, (0.485, 0.405), (0.485, 0.458), color=BLUE, linewidth=0.95)

    _latent_cloud(ax, (0.075, 0.705), scale=0.82)
    _latent_cloud(ax, (0.485, 0.705), scale=0.90)
    _latent_cloud(ax, (0.315, 0.705), scale=0.88, predicted=True)
    _arrow(ax, (0.075, 0.575), (0.075, 0.625), color=BLUE, linewidth=0.95)
    _arrow(ax, (0.485, 0.575), (0.485, 0.625), color=BLUE, linewidth=0.95)

    ax.text(0.075, 0.825, r"$z_t$", fontsize=6.5, color=BLUE_DARK, ha="center")
    ax.text(0.315, 0.825, r"$\hat z_{t+1}$", fontsize=6.5, color=GREEN, ha="center")
    ax.text(0.485, 0.825, r"$z_{t+1}$", fontsize=6.5, color=BLUE_DARK, ha="center")

    _capsule(
        ax,
        (0.158, 0.658),
        0.082,
        0.094,
        r"$P_{a_t}$",
        facecolor=GREEN_PALE,
        edgecolor=GREEN,
        textcolor=GREEN,
        fontsize=6.7,
    )
    _arrow(ax, (0.122, 0.705), (0.151, 0.705), color=GREEN, linewidth=1.1)
    _arrow(ax, (0.246, 0.705), (0.267, 0.705), color=GREEN, linewidth=1.1)

    ax.add_patch(
        FancyArrowPatch(
            (0.371, 0.705),
            (0.429, 0.705),
            arrowstyle="<->",
            mutation_scale=6.5,
            linewidth=0.85,
            color=RED,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )
    ax.text(0.400, 0.768, "MSE", fontsize=5.35, color=RED, ha="center")

    ax.text(
        0.28,
        0.57,
        r"$\mathrm{SIGReg}(z)$",
        fontsize=6.4,
        color=ORANGE,
        ha="center",
    )
    ax.text(
        0.28,
        0.985,
        "latent prediction from pixels",
        fontsize=6.7,
        fontweight="semibold",
        color=INK,
        ha="center",
        va="top",
    )

    _operator_lens(ax, aligned_output=aligned_output, oblique_output=oblique_output)

    _assert_no_text_collisions(fig)

    png_path = out_dir / "operator_pair_intro_figure.png"
    pdf_path = out_dir / "operator_pair_intro_figure.pdf"
    svg_path = out_dir / "operator_pair_intro_figure.svg"
    figma_svg_path = out_dir / "operator_pair_intro_figure_figma.svg"

    fig.savefig(png_path, dpi=320, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    plt.rcParams["svg.fonttype"] = "path"
    fig.savefig(figma_svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    for path in (png_path, pdf_path, svg_path, figma_svg_path):
        print(path)


if __name__ == "__main__":
    main()
