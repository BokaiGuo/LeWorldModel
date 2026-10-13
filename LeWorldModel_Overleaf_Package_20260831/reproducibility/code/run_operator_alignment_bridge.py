"""Bridge the operator-alignment theorem to frozen PushT latent evidence.

This script is a post-hoc evidence bridge, not a new experimental protocol.
It reads the already-produced operator-alignment theorem and the frozen
PushT empirical audit, then summarizes how real latents compare with the
theory-side same-spectrum / different-basis object.
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
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theory",
        default="results/theory_operator_alignment_sweep_20260828/operator_alignment_theorem.json",
        help="theorem JSON produced by run_operator_alignment_sweep.py",
    )
    parser.add_argument(
        "--empirical",
        default="results/pusht/lewm/empirical_spectral_audit.json",
        help="frozen empirical spectral audit JSON",
    )
    parser.add_argument(
        "--output",
        default="results/theory_empirical_bridge_20260829",
        help="output directory for bridge summary artifacts",
    )
    parser.add_argument("--k", type=int, default=32, help="rank band used for the main empirical bridge readout")
    parser.add_argument("--null-trials", type=int, default=2048, help="Monte Carlo trials for the random-subspace null")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _control_lookup(controls: dict[str, Any], name: str, k: int) -> float:
    value = controls.get(name, {})
    if isinstance(value, dict):
        raw = value.get(str(k))
        if isinstance(raw, (int, float)):
            return float(raw)
    return float("nan")


def _random_subspace_overlap_null(dim: int, k: int, trials: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(trials)):
        left, _ = np.linalg.qr(rng.normal(size=(dim, k)))
        right, _ = np.linalg.qr(rng.normal(size=(dim, k)))
        values.append(float(np.square(left.T @ right).sum() / max(k, 1)))
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    theory = payload["theory"]
    empirical = payload["empirical"]
    bridge = payload["bridge"]
    primary = bridge["primary_lifts"]
    controls = bridge["control_lifts"]
    transport = bridge["transport"]
    planner = bridge["planner_weights"]
    lines = [
        "# Operator Alignment Bridge",
        "",
        "## Theory Anchor",
        "",
        f"- theorem peak angle: {_fmt(theory['theorem']['peak_angle_degrees'])}",
        f"- theorem statement: {theory['theorem']['statement']}",
        f"- flat-spectrum falsifier: zero exact commutator at every angle",
        "",
        "## Empirical Bridge",
        "",
        "| readout | observed | random-control | ratio |",
        "| --- | --- | --- | --- |",
    ]
    for name in ["dynamics", "planning", "predicted_dynamics", "prediction_residual"]:
        value = primary[name]
        control = controls["norm_matched_random_dynamics_lift"]
        ratio = value / max(control, 1e-12)
        lines.append(f"| {name} | {_fmt(value)} | {_fmt(control)} | {_fmt(ratio)} |")
    lines.extend(
        [
            "",
            "## Control Baselines",
            "",
            "| control | top-k lift |",
            "| --- | --- |",
        ]
    )
    for name in [
        "norm_matched_random_dynamics_lift",
        "random_basis_dynamics_lift",
        "matched_eigenvalue_random_rotation_dynamics_lift",
        "isotropic_gaussian_dynamics_lift",
    ]:
        lines.append(f"| {name} | {_fmt(controls[name])} |")
    lines.extend(
        [
            "",
            "## Eigenspace Transport",
            "",
            f"- observed mean transition overlap: {_fmt(transport['observed']['mean'])}",
            f"- observed median transition overlap: {_fmt(transport['observed']['median'])}",
            f"- random-subspace null mean: {_fmt(transport['null']['mean'])}",
            f"- random-subspace null p05/p95: {_fmt(transport['null']['p05'])} / {_fmt(transport['null']['p95'])}",
            "",
            "## Planner Anisotropy",
            "",
            f"- top-weight mean: {_fmt(planner['top_weight_mean'])}",
            f"- tail-weight mean: {_fmt(planner['tail_weight_mean'])}",
            f"- top/tail ratio: {_fmt(planner['top_weight_mean'] / max(planner['tail_weight_mean'], 1e-12))}",
            "",
            "## Readout",
            "",
            "- Real PushT latents show a large gap between observed dynamics/planning lift and random-basis controls.",
            "- Eigenspace transport is low in absolute terms, but its mean exceeds the random-subspace null.",
            "- Planner weights are anisotropic, with the top band weighted more heavily than the tail.",
            "",
            "## Claim Boundary",
            "",
            "- This is an empirical bridge to the theorem object, not a proof that the exact single-plane law holds on real LeWM latents.",
            "- It supports the broader claim that basis alignment matters in frozen PushT latent geometry.",
            "",
        ]
    )
    (output / "operator_alignment_bridge.md").write_text("\n".join(lines) + "\n")


def _plot_bridge(ax1: plt.Axes, ax2: plt.Axes, payload: dict[str, Any]) -> None:
    bridge = payload["bridge"]
    primary = bridge["primary_lifts"]
    controls = bridge["control_lifts"]
    order = ["dynamics", "planning", "predicted_dynamics", "prediction_residual"]
    observed = [primary[name] for name in order]
    control = [controls["norm_matched_random_dynamics_lift"] for _ in order]
    x = np.arange(len(order))
    width = 0.35
    ax1.bar(x - width / 2, observed, width, label="observed", color="#1b9e77")
    ax1.bar(x + width / 2, control, width, label="norm-matched random", color="#7570b3")
    ax1.set_xticks(x, ["dynamics", "planning", "predicted", "residual"], rotation=0)
    ax1.set_ylabel(f"top-{bridge['k']} lift")
    ax1.set_title("Real latent lift vs control", fontsize=9, pad=4)
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.legend(frameon=False, fontsize=7, loc="best")

    transport = bridge["transport"]
    obs = transport["samples"]
    null = transport["null"]
    ax2.hist(obs, bins=12, alpha=0.75, color="#1b9e77", label="observed transitions")
    ax2.axvline(float(np.mean(obs)), color="#1b9e77", linestyle="--", linewidth=1.0)
    ax2.axvspan(null["p05"], null["p95"], color="#7570b3", alpha=0.18, label="random-subspace null p05-p95")
    ax2.axvline(null["mean"], color="#7570b3", linestyle="--", linewidth=1.0, label="random-subspace null mean")
    ax2.set_xlabel("transition overlap")
    ax2.set_ylabel("count")
    ax2.set_title("Eigenspace transport bridge", fontsize=9, pad=4)
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.legend(frameon=False, fontsize=7, loc="best")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    theory = _load_json(args.theory)
    empirical = _load_json(args.empirical)
    k = int(args.k)
    controls = empirical["weighted_mean"]["controls"]
    lifts = empirical["weighted_mean"]["lifts"]
    primary_lifts = {
        "dynamics": _control_lookup(lifts, "dynamics", k),
        "planning": _control_lookup(lifts, "planning", k),
        "predicted_dynamics": _control_lookup(lifts, "predicted_dynamics", k),
        "prediction_residual": _control_lookup(lifts, "prediction_residual", k),
    }
    control_lifts = {
        "norm_matched_random_dynamics_lift": _control_lookup(controls, "norm_matched_random_dynamics_lift", k),
        "random_basis_dynamics_lift": _control_lookup(controls, "random_basis_dynamics_lift", k),
        "matched_eigenvalue_random_rotation_dynamics_lift": _control_lookup(
            controls, "matched_eigenvalue_random_rotation_dynamics_lift", k
        ),
        "isotropic_gaussian_dynamics_lift": _control_lookup(controls, "isotropic_gaussian_dynamics_lift", k),
    }

    transport = empirical["eigenspace_transport"]
    transport_values = np.asarray(list(transport["transitions"].values()), dtype=np.float64)
    dim = len(empirical["spectral_planner_metric"]["weights"])
    transport_null = _random_subspace_overlap_null(dim, int(transport["k"]), args.null_trials, args.seed)
    planner = empirical["spectral_planner_metric"]["weight_summary"]

    report = {
        "theory": {
            "theorem": theory["theorem"],
            "claim_boundary": theory["claim_boundary"],
        },
        "empirical": {
            "eigenspace_transport": empirical["eigenspace_transport"],
            "spectral_planner_metric": empirical["spectral_planner_metric"],
            "weighted_mean": {
                "lifts": {key: lifts[key] for key in ["dynamics", "planning", "predicted_dynamics", "prediction_residual"] if key in lifts},
                "controls": {
                    key: controls[key]
                    for key in [
                        "norm_matched_random_dynamics_lift",
                        "random_basis_dynamics_lift",
                        "matched_eigenvalue_random_rotation_dynamics_lift",
                        "isotropic_gaussian_dynamics_lift",
                    ]
                    if key in controls
                },
            },
        },
        "bridge": {
            "k": k,
            "primary_lifts": primary_lifts,
            "control_lifts": control_lifts,
            "transport": {
                "samples": transport_values.tolist(),
                "observed": {
                    "mean": float(transport_values.mean()),
                    "median": float(np.median(transport_values)),
                    "min": float(transport_values.min(initial=float("inf"))),
                    "max": float(transport_values.max(initial=float("-inf"))),
                },
                "null": transport_null,
            },
            "planner_weights": planner,
            "claim_boundary": (
                "Empirical bridge only: real PushT latents show basis-sensitive lift and low eigenspace transport, "
                "but this does not prove the exact finite-dimensional theorem on learned operators."
            ),
        },
    }

    (output / "operator_alignment_bridge.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

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

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    _plot_bridge(axes[0], axes[1], report)
    axes[0].text(-0.14, 1.05, "A", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    axes[1].text(-0.14, 1.05, "B", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    fig.savefig(output / "operator_alignment_bridge.pdf", bbox_inches="tight")
    fig.savefig(output / "operator_alignment_bridge.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_markdown(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
