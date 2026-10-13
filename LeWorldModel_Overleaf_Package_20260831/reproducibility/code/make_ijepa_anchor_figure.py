"""Create the paper figure for the frozen I-JEPA feature anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            Path("results/theory_operator_alignment_iclr_oral_20260829/jepa_anchor_cifar10_hflip32/ijepa_feature_anchor.json"),
            Path("results/theory_operator_alignment_iclr_oral_20260829/jepa_anchor_cifar100_hflip/ijepa_feature_anchor.json"),
            Path("results/theory_operator_alignment_iclr_oral_20260829/jepa_anchor_stl10_hflip/ijepa_feature_anchor.json"),
            Path("results/theory_operator_alignment_iclr_oral_20260829/jepa_anchor_svhn_hflip/ijepa_feature_anchor.json"),
            Path("results/theory_operator_alignment_iclr_oral_20260829/jepa_anchor_tinyimagenet_hflip/ijepa_feature_anchor.json"),
            Path("results/theory_operator_alignment_iclr_oral_20260829/jepa_anchor_food101_hflip/ijepa_feature_anchor.json"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/theory_operator_alignment_iclr_oral_20260829/figures"),
    )
    return parser.parse_args()


def load_results(paths: list[Path]) -> list[dict]:
    return [json.loads(path.read_text()) for path in paths]


def pretty_label(name: str) -> str:
    return {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "stl10": "STL10",
        "svhn": "SVHN",
        "tiny-imagenet": "TinyImgNet",
        "food101": "Food-101",
    }.get(name, name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.inputs)

    labels = [pretty_label(result["config"]["dataset"]) for result in results]
    overlap_obs = [result["top_subspace_overlap"]["observed_source_vs_displacement"] for result in results]
    overlap_null = [result["top_subspace_overlap"]["random_subspace_mean"] for result in results]
    overlap_lo = [result["top_subspace_overlap"]["random_subspace_p05"] for result in results]
    overlap_hi = [result["top_subspace_overlap"]["random_subspace_p95"] for result in results]
    lift_obs = [result["top_lift"]["observed_view_pair"] for result in results]
    lift_null = [result["top_lift"]["norm_matched_random_mean"] for result in results]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 6,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 1.95), constrained_layout=True, sharey=True)

    y = np.arange(len(labels))
    height = 0.34
    axes[0].barh(y - height / 2, overlap_obs, height, label="observed", color="#1b4f72")
    axes[0].barh(y + height / 2, overlap_null, height, label="random subspace", color="#b7950b")
    axes[0].errorbar(
        overlap_null,
        y + height / 2,
        xerr=[
            np.array(overlap_null) - np.array(overlap_lo),
            np.array(overlap_hi) - np.array(overlap_null),
        ],
        fmt="none",
        color="black",
        capsize=2,
        linewidth=0.8,
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("top-16 subspace overlap")
    axes[0].set_title("(a) Displacement overlap")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.1)

    axes[1].barh(y - height / 2, lift_obs, height, label="observed", color="#1b4f72")
    axes[1].barh(y + height / 2, lift_null, height, label="norm-matched random", color="#b7950b")
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("top-16 lift over isotropic")
    axes[1].set_title("(b) Top-eigenspace lift")
    axes[1].legend(frameon=False, loc="lower right", handlelength=1.1)

    source_rank = [result["source_stable_rank"] for result in results]
    delta_rank = [result["displacement_stable_rank"] for result in results]
    axes[2].scatter(source_rank, y, marker="o", s=24, label="source")
    axes[2].scatter(delta_rank, y, marker="s", s=24, label="displacement")
    for idx, (src, disp) in enumerate(zip(source_rank, delta_rank)):
        axes[2].plot([src, disp], [idx, idx], color="#808b96", linewidth=0.9, zorder=0)
    axes[2].set_xlabel("stable rank")
    axes[2].set_title("(c) Rank contrast")
    axes[2].legend(frameon=False, handlelength=1.1)

    for ax in axes:
        ax.grid(axis="x", color="#d5d8dc", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.output_dir / f"ijepa_feature_anchor.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
