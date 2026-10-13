"""Summarize Wall comparison baselines and action ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--operator-controls", type=Path, required=True)
    parser.add_argument("--within-sensitivity", type=Path)
    parser.add_argument("--context-controls", type=Path)
    parser.add_argument("--context1-metrics", type=Path)
    parser.add_argument("--context2-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def short_label(label: str) -> str:
    return label.replace("_Wall", "").replace("_", " ")


def main() -> None:
    args = parse_args()
    statistics = json.loads(args.statistics.read_text())
    controls = json.loads(args.operator_controls.read_text())
    within = json.loads(args.within_sensitivity.read_text()) if args.within_sensitivity else None
    context_controls = json.loads(args.context_controls.read_text()) if args.context_controls else None
    context1 = json.loads(args.context1_metrics.read_text()) if args.context1_metrics else None
    context2 = json.loads(args.context2_metrics.read_text()) if args.context2_metrics else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "protocol": {
            "environment": "Wall",
            "action_modes": ["observed", "zero", "reverse"],
            "horizons": [1, 2, 4],
            "operator_primary_setting": controls["protocol"]["primary_setting"],
            "paired_statistics": statistics["protocol"],
        },
        "models": statistics["models"],
        "operator_controls": controls["models"],
        "within_episode_sensitivity": within,
        "context_ablation": context_controls,
        "evidence_boundary": (
            "Fixed-checkpoint Wall replay. Latent errors are representation-scale specific and must not be used "
            "to rank JEPA-WM against DINO-WM directly. Action sensitivity and held-out controls are interpreted "
            "within checkpoint, not as causal task performance."
        ),
    }
    (args.output_dir / "wall_baseline_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    markdown = [
        "# Wall comparison and ablation matrix",
        "",
        "All runs use the same 192-episode code-reconstructed seed-42 split, frame stride 5, and horizons 1/2/4. "
        "Checkpoints are frozen. `zero` removes normalized action information and `reverse` reverses the supplied "
        "action-block order.",
        "",
        "## Latent prediction error and paired intervention tests",
        "",
        "| Model | Horizon | Observed MSE | Zero MSE | Zero−observed [95% CI] | Reverse MSE | Reverse−observed [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, model in statistics["models"].items():
        for horizon, values in model["horizons"].items():
            zero = values["zero_minus_observed"]
            reverse = values["reverse_minus_observed"]
            markdown.append(
                f"| {short_label(label)} | {horizon} | {values['observed_mse_mean']:.6f} | "
                f"{values['zero_mse_mean']:.6f} | {zero['mean_delta']:+.6f} "
                f"[{zero['bootstrap_ci95'][0]:+.6f},{zero['bootstrap_ci95'][1]:+.6f}] | "
                f"{values['reverse_mse_mean']:.6f} | {reverse['mean_delta']:+.6f} "
                f"[{reverse['bootstrap_ci95'][0]:+.6f},{reverse['bootstrap_ci95'][1]:+.6f}] |"
            )

    markdown.extend([
        "",
        "## Cross-fitted operator baselines",
        "",
        "Primary setting: rank 8, ridge 1e-2. Values are held-out means across five folds and split seeds 0/1/2.",
        "",
        "| Model/mode | Horizon | Paired R² | Identity R² | Pair-shuffled R² | Spectral-control R² | Paired−shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, model in controls["models"].items():
        for horizon, values in model["horizons"].items():
            primary = values["primary"]
            markdown.append(
                f"| {short_label(label)} | {horizon} | {primary['paired_r2']['mean']:.3f} | "
                f"{primary['identity_r2']['mean']:.3f} | {primary['shuffled_r2']['mean']:.3f} | "
                f"{primary['spectral_null_r2']['mean']:.3f} | "
                f"{primary['paired_minus_shuffled_r2']['mean_difference']:+.3f} |"
            )
    if within:
        markdown.extend([
            "",
            "## Within-episode temporal-order ablation",
            "",
            "The intervention keeps episode identity and the exact future-action block multiset, changing only "
            "temporal order. Reported values are seed-level mean delta and range across permutation seeds 0/1/2; "
            "seeds are not pooled as independent episodes.",
            "",
            "| Model | Horizon | Mean ΔMSE | Seed range | All seed Δ > 0 |",
            "|---|---:|---:|---:|:---:|",
        ])
        for label, model in within["models"].items():
            for horizon, values in model["horizons"].items():
                markdown.append(
                    f"| {short_label(label)} | {horizon} | {values['seed_mean_delta']:+.6f} | "
                    f"[{values['seed_min_delta']:+.6f},{values['seed_max_delta']:+.6f}] | "
                    f"{'yes' if values['all_seed_deltas_positive'] else 'no'} |"
                )
    if context_controls and context1 and context2:
        markdown.extend([
            "",
            "## JEPA-WM context-length ablation",
            "",
            "The checkpoint and 192-episode split are fixed; only the number of observed context frames changes.",
            "",
            "| Horizon | Context 1 MSE | Context 2 MSE | Context 1 paired R² | Context 2 paired R² |",
            "|---:|---:|---:|---:|---:|",
        ])
        for horizon in ["1", "2", "4"]:
            c1_r2 = context_controls["models"]["JEPA-WM_Wall_context1"]["horizons"][horizon]["primary"]["paired_r2"]["mean"]
            c2_r2 = context_controls["models"]["JEPA-WM_Wall_context2"]["horizons"][horizon]["primary"]["paired_r2"]["mean"]
            markdown.append(
                f"| {horizon} | {context1['latent_prediction_error_by_horizon'][horizon]['mean']:.6f} | "
                f"{context2['latent_prediction_error_by_horizon'][horizon]['mean']:.6f} | {c1_r2:.3f} | {c2_r2:.3f} |"
            )
    markdown.extend(["", "## Evidence boundary", "", summary["evidence_boundary"]])
    (args.output_dir / "wall_baseline_ablation_summary.md").write_text("\n".join(markdown) + "\n")

    import matplotlib.pyplot as plt
    import numpy as np

    horizons = ["1", "2", "4"]
    fig, axes = plt.subplots(1, len(statistics["models"]), figsize=(10, 3.8), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for axis, (label, model) in zip(axes, statistics["models"].items()):
        x = np.arange(len(horizons))
        width = 0.34
        for offset, mode, color in [(-width / 2, "zero_minus_observed", "#D55E00"), (width / 2, "reverse_minus_observed", "#0072B2")]:
            means = [model["horizons"][h][mode]["mean_delta"] for h in horizons]
            lower = [means[i] - model["horizons"][h][mode]["bootstrap_ci95"][0] for i, h in enumerate(horizons)]
            upper = [model["horizons"][h][mode]["bootstrap_ci95"][1] - means[i] for i, h in enumerate(horizons)]
            axis.bar(x + offset, means, width, label=mode.split("_")[0].title(), color=color, alpha=0.88)
            axis.errorbar(x + offset, means, yerr=[lower, upper], fmt="none", color="black", capsize=3, linewidth=1)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [f"h={h}" for h in horizons])
        axis.set_title(short_label(label))
        axis.set_ylabel("Intervention − observed latent MSE")
        axis.legend(frameon=False)
    fig.savefig(args.output_dir / "Figure_1_wall_action_ablation.png", dpi=220)
    plt.close(fig)

    observed_labels = [label for label in controls["models"] if label.endswith("_observed")]
    fig, axes = plt.subplots(1, len(observed_labels), figsize=(10, 3.8), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    baseline_names = ["paired_r2", "identity_r2", "shuffled_r2", "spectral_null_r2"]
    colors = ["#009E73", "#56B4E9", "#999999", "#CC79A7"]
    display = ["Paired", "Identity", "Pair-shuffled", "Spectral control"]
    for axis, label in zip(axes, observed_labels):
        x = np.arange(len(horizons))
        width = 0.19
        for index, (metric, color, name) in enumerate(zip(baseline_names, colors, display)):
            values = [controls["models"][label]["horizons"][h]["primary"][metric]["mean"] for h in horizons]
            axis.bar(x + (index - 1.5) * width, values, width, color=color, label=name)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [f"h={h}" for h in horizons])
        axis.set_title(short_label(label).replace(" observed", ""))
        axis.set_ylabel("Held-out R²")
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(args.output_dir / "Figure_2_wall_operator_baselines.png", dpi=220)
    plt.close(fig)

    print(json.dumps({
        "markdown": str(args.output_dir / "wall_baseline_ablation_summary.md"),
        "figure_1": str(args.output_dir / "Figure_1_wall_action_ablation.png"),
        "figure_2": str(args.output_dir / "Figure_2_wall_operator_baselines.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
