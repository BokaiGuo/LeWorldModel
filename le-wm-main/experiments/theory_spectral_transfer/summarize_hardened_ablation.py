"""Summarize the hardened action and split-robustness replay matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analyze_action_intervention_statistics import holm_adjust, mse_vectors, paired_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if "operator_fit_vectors" not in payload:
        raise ValueError(f"{path} lacks operator_fit_vectors")
    return payload


def assert_paired(observed: dict, intervention: dict, label: str) -> None:
    observed_protocol = observed.get("protocol", {})
    intervention_protocol = intervention.get("protocol", {})
    if observed_protocol.get("episodes_used") != intervention_protocol.get("episodes_used"):
        raise ValueError(f"{label}: episode manifests differ")
    for horizon in observed["operator_fit_vectors"]["target"]:
        target_a = torch.tensor(observed["operator_fit_vectors"]["target"][horizon])
        target_b = torch.tensor(intervention["operator_fit_vectors"]["target"][horizon])
        if not torch.equal(target_a, target_b):
            raise ValueError(f"{label}, horizon {horizon}: target vectors differ")


def paired_delta(observed: dict, intervention: dict, label: str) -> dict:
    assert_paired(observed, intervention, label)
    result = {}
    for horizon in sorted(observed["operator_fit_vectors"]["target"], key=int):
        observed_mse, _ = mse_vectors(observed, horizon)
        intervention_mse, _ = mse_vectors(intervention, horizon)
        result[horizon] = paired_stats(
            intervention_mse - observed_mse,
            bootstrap_trials=20_000,
            permutation_trials=50_000,
            seed=100 + int(horizon),
        )
        result[horizon]["observed_mse"] = float(observed_mse.mean())
        result[horizon]["intervention_mse"] = float(intervention_mse.mean())
    return result


def main() -> None:
    args = parse_args()
    root = args.root
    base = root.parent / "wall_temporal_gate"
    source = Path(__file__).resolve().parents[2]
    metrics = root / "metrics"
    observed_by_env = {
        "PushT": load(source / "results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/dino_wm_pusht_temporal_metrics.json"),
        "PointMaze": load(source / "results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/dino_wm_pointmaze_temporal_metrics_200.json"),
        "Wall": load(base / "dino_wm_wall_observed_metrics.json"),
    }
    future_effects = {}
    tests = []
    for environment, observed in observed_by_env.items():
        future_effects[environment] = {}
        for mode in ("future_zero", "future_reverse"):
            intervention = load(metrics / f"dino_{environment.lower()}_{mode}.json") if environment != "PointMaze" else load(metrics / f"dino_pointmaze_{mode}.json")
            result = paired_delta(observed, intervention, f"DINO {environment} {mode}")
            future_effects[environment][mode] = result
            for horizon, stats in result.items():
                tests.append((environment, mode, horizon, stats))

    wall_split_robustness = {}
    for model in ("jepa", "dino"):
        observed_42 = load(base / f"{model}_wm_wall_observed_metrics.json")
        rows = []
        for split_seed in (0, 1):
            payload = load(metrics / f"{model}_wall_observed_split{split_seed}.json")
            rows.append({
                "split_seed": split_seed,
                "episodes": len(payload["protocol"]["episodes_used"]),
                "mse_by_horizon": {
                    h: payload["latent_prediction_error_by_horizon"][h]["mean"]
                    for h in sorted(payload["latent_prediction_error_by_horizon"], key=int)
                },
            })
        wall_split_robustness[model.upper()] = {
            "reference_split_seed": 42,
            "reference_mse_by_horizon": {
                h: observed_42["latent_prediction_error_by_horizon"][h]["mean"]
                for h in sorted(observed_42["latent_prediction_error_by_horizon"], key=int)
            },
            "additional_splits": rows,
        }

    wall_future_only_split_robustness = {}
    dino_observed_42 = observed_by_env["Wall"]
    for split_seed in (0, 1):
        observed = load(metrics / f"dino_wall_observed_split{split_seed}.json")
        row = {"split_seed": split_seed, "horizons": {}}
        for mode in ("future_zero", "future_reverse"):
            intervention = load(metrics / f"dino_wall_{mode}_split{split_seed}.json")
            row["horizons"][mode] = paired_delta(observed, intervention, f"DINO Wall split {split_seed} {mode}")
            for horizon, stats in row["horizons"][mode].items():
                tests.append((f"DINO-Wall-split{split_seed}", mode, horizon, stats))
        wall_future_only_split_robustness[split_seed] = row
    wall_future_only_split_robustness[42] = {
        "split_seed": 42,
        "horizons": {
            mode: future_effects["Wall"][mode]
            for mode in ("future_zero", "future_reverse")
        },
    }

    wall_episode_shuffle = {}
    wall_shuffle_split_robustness = {}
    for model in ("jepa", "dino"):
        observed = load(base / f"{model}_wm_wall_observed_metrics.json")
        seeds = []
        for seed in (0, 1, 2):
            intervention = load(root / "metrics" / f"{model}_wall_episode_shuffle_split42_seed{seed}.json")
            result = paired_delta(observed, intervention, f"Wall {model} episode shuffle seed {seed}")
            seeds.append({"seed": seed, "horizons": result})
        wall_episode_shuffle[model.upper()] = seeds
        for row in seeds:
            for horizon, stats in row["horizons"].items():
                tests.append((f"Wall-{model.upper()}", "episode_shuffle", horizon, stats))

        split_rows = []
        for split_seed in (0, 1):
            split_observed = load(metrics / f"{model}_wall_observed_split{split_seed}.json")
            split_intervention = load(metrics / f"{model}_wall_episode_shuffle_split{split_seed}_seed0.json")
            split_rows.append({
                "split_seed": split_seed,
                "horizons": paired_delta(
                    split_observed,
                    split_intervention,
                    f"Wall {model} split {split_seed} episode shuffle",
                ),
            })
        split_rows.append({"split_seed": 42, "horizons": seeds[0]["horizons"]})
        wall_shuffle_split_robustness[model.upper()] = split_rows
        for row in split_rows[:2]:
            for horizon, stats in row["horizons"].items():
                tests.append((f"Wall-{model.upper()}-split{row['split_seed']}", "episode_shuffle", horizon, stats))

    adjusted = holm_adjust([stats["sign_flip_p_two_sided"] for _, _, _, stats in tests])
    for (_, _, _, stats), value in zip(tests, adjusted):
        stats["holm_adjusted_p"] = value
        stats["holm_reject_0_05"] = value < 0.05

    report = {
        "protocol": {
            "fixed_checkpoint": True,
            "all_paired_tests": True,
            "multiple_testing": "Holm correction across future-only DINO cells and Wall episode-shuffle seed cells",
            "small_sample_test": "exact sign-flip enumeration for n <= 21; Monte Carlo sign-flip otherwise",
        },
        "future_only_action_ablation": future_effects,
        "wall_split_robustness": wall_split_robustness,
        "wall_episode_shuffle": wall_episode_shuffle,
        "wall_episode_shuffle_split_robustness": wall_shuffle_split_robustness,
        "wall_future_only_split_robustness": wall_future_only_split_robustness,
        "evidence_boundary": (
            "These are frozen-checkpoint replay diagnostics. Future-only interventions isolate the predicted-action "
            "suffix while preserving historical action context; they do not establish causal task performance, online "
            "planning gains, original training-split provenance, or a universal world-model theorem."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    markdown = [
        "# Hardened comparison and ablation audit",
        "",
        "All comparisons use frozen checkpoints and paired per-episode target vectors. `future_zero` and "
        "`future_reverse` modify only future action blocks and preserve the historical context prefix.",
        "",
        "## DINO future-only action ablation",
        "",
        "| Environment | Mode | Horizon | Observed MSE | Intervened MSE | ΔMSE | 95% CI | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for environment, modes in future_effects.items():
        for mode, horizons in modes.items():
            for horizon, stats in horizons.items():
                markdown.append(
                    f"| {environment} | {mode} | {horizon} | {stats['observed_mse']:.6f} | "
                    f"{stats['intervention_mse']:.6f} | {stats['mean_delta']:+.6f} | "
                    f"[{stats['bootstrap_ci95'][0]:+.6f},{stats['bootstrap_ci95'][1]:+.6f}] | "
                    f"{stats['holm_adjusted_p']:.6g} |"
                )
    markdown.extend([
        "",
        "## Wall split robustness",
        "",
        "| Model | Split | Episodes | h=1 MSE | h=2 MSE | h=4 MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for model, values in wall_split_robustness.items():
        ref = values["reference_mse_by_horizon"]
        markdown.append(f"| {model} | 42 | 192 | {ref['1']:.6f} | {ref['2']:.6f} | {ref['4']:.6f} |")
        for row in values["additional_splits"]:
            mse = row["mse_by_horizon"]
            markdown.append(f"| {model} | {row['split_seed']} | {row['episodes']} | {mse['1']:.6f} | {mse['2']:.6f} | {mse['4']:.6f} |")
    markdown.extend([
        "",
        "## Wall distribution-preserving episode derangement",
        "",
        "| Model | Horizon | Mean ΔMSE across 3 seeds | Seed range | All seeds positive |",
        "|---|---:|---:|---:|:---:|",
    ])
    for model, seeds in wall_episode_shuffle.items():
        for horizon in ("1", "2", "4"):
            values = [row["horizons"][horizon]["mean_delta"] for row in seeds]
            markdown.append(
                f"| {model} | {horizon} | {sum(values)/len(values):+.6f} | "
                f"[{min(values):+.6f},{max(values):+.6f}] | {'yes' if all(v > 0 for v in values) else 'no'} |"
            )
    markdown.extend([
        "",
        "## Wall episode-derangement split robustness",
        "",
        "| Model | Horizon | Mean ΔMSE across splits 0/1/42 | Split range | All splits positive |",
        "|---|---:|---:|---:|:---:|",
    ])
    for model, rows in wall_shuffle_split_robustness.items():
        for horizon in ("1", "2", "4"):
            values = [row["horizons"][horizon]["mean_delta"] for row in rows]
            markdown.append(
                f"| {model} | {horizon} | {sum(values)/len(values):+.6f} | "
                f"[{min(values):+.6f},{max(values):+.6f}] | {'yes' if all(v > 0 for v in values) else 'no'} |"
            )
    markdown.extend([
        "",
        "## DINO future-only action effect across Wall splits",
        "",
        "| Split | Mode | Horizon | ΔMSE | 95% CI | Holm p |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for split_seed in (42, 0, 1):
        for mode in ("future_zero", "future_reverse"):
            for horizon, stats in wall_future_only_split_robustness[split_seed]["horizons"][mode].items():
                markdown.append(
                    f"| {split_seed} | {mode} | {horizon} | {stats['mean_delta']:+.6f} | "
                    f"[{stats['bootstrap_ci95'][0]:+.6f},{stats['bootstrap_ci95'][1]:+.6f}] | "
                    f"{stats.get('holm_adjusted_p', float('nan')):.6g} |"
                )
    markdown.extend(["", "## Evidence boundary", "", report["evidence_boundary"]])
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")

    import matplotlib.pyplot as plt
    import numpy as np

    environments = list(future_effects)
    fig, axes = plt.subplots(1, len(environments), figsize=(12, 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, environment in zip(axes, environments):
        positions = np.arange(3)
        width = 0.34
        for offset, mode, color in [(-width / 2, "future_zero", "#D55E00"), (width / 2, "future_reverse", "#0072B2")]:
            means = [future_effects[environment][mode][h]["mean_delta"] for h in ("1", "2", "4")]
            lower = [means[i] - future_effects[environment][mode][h]["bootstrap_ci95"][0] for i, h in enumerate(("1", "2", "4"))]
            upper = [future_effects[environment][mode][h]["bootstrap_ci95"][1] - means[i] for i, h in enumerate(("1", "2", "4"))]
            axis.bar(positions + offset, means, width, label=mode.replace("future_", "").title(), color=color)
            axis.errorbar(positions + offset, means, yerr=[lower, upper], fmt="none", color="black", capsize=3, linewidth=1)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, ["h=1", "h=2", "h=4"])
        axis.set_title(environment)
        axis.set_ylabel("Future intervention − observed MSE")
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(args.output.parent / "Figure_1_hardened_future_action_ablation.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True, sharey=False)
    split_labels = ["42", "0", "1"]
    positions = np.arange(3)
    for axis, model in zip(axes, ("JEPA", "DINO")):
        rows = wall_split_robustness[model]
        for index, horizon in enumerate(("1", "2", "4")):
            values = [rows["reference_mse_by_horizon"][horizon]] + [row["mse_by_horizon"][horizon] for row in rows["additional_splits"]]
            axis.plot(positions, values, marker="o", label=f"h={horizon}")
        axis.set_xticks(positions, split_labels)
        axis.set_xlabel("Split seed")
        axis.set_ylabel("Latent prediction MSE")
        axis.set_title(model)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(args.output.parent / "Figure_2_hardened_split_robustness.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"json": str(args.output), "markdown": str(args.output.with_suffix('.md'))}, indent=2))


if __name__ == "__main__":
    main()
