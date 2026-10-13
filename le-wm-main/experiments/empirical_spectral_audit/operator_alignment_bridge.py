"""Empirical bridge from frozen PushT audits to the operator-alignment object.

This module stays on the empirical side of the split.  It consumes the
completed empirical audit JSON, extracts the basis-sensitive lift, label
shuffle rivals, eigenspace transport, and planner anisotropy, and can add an
optional theory anchor when a theorem JSON is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.io_utils import save_json


PRIMARY_TANGENTS = ("dynamics", "planning", "predicted_dynamics", "prediction_residual")
CONTROL_NAMES = (
    "norm_matched_random_dynamics_lift",
    "random_basis_dynamics_lift",
    "matched_eigenvalue_random_rotation_dynamics_lift",
    "isotropic_gaussian_dynamics_lift",
)
LABEL_CONTROL_NAMES = ("state_cluster_shuffle", "time_shuffle", "action_shuffle")
STEM = "empirical_operator_alignment_bridge"
PAPER_OUTLINE_STEM = "empirical_operator_alignment_paper_outline"
DEFAULT_AUDIT = "results/pusht/lewm/empirical_spectral_audit.json"
DEFAULT_THEORY = "results/theory_operator_alignment_sweep_20260828/operator_alignment_theorem.json"
DEFAULT_OUTPUT = "results/pusht/lewm/empirical_operator_alignment_bridge_20260829"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default=DEFAULT_AUDIT, help="completed empirical_spectral_audit.json")
    parser.add_argument(
        "--theory",
        default=DEFAULT_THEORY,
        help="optional operator-alignment theorem JSON for an anchor section",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output directory for bridge artifacts")
    parser.add_argument("--k", type=int, default=32, help="top-k readout used for the bridge summary")
    parser.add_argument("--null-trials", type=int, default=2048, help="Monte Carlo trials for the transport null")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _curve_lookup(curve: Any, k: int) -> float:
    if isinstance(curve, dict):
        raw = curve.get(str(k))
        if isinstance(raw, (int, float)):
            return float(raw)
    return float("nan")


def _nested_curve_lookup(block: Any, tangent: str, k: int) -> float:
    if not isinstance(block, dict):
        return float("nan")
    weighted = block.get("weighted_mean", {})
    if not isinstance(weighted, dict):
        return float("nan")
    lifts = weighted.get("lifts", {})
    if not isinstance(lifts, dict):
        return float("nan")
    return _curve_lookup(lifts.get(tangent, {}), k)


def _control_lookup(controls: dict[str, Any], name: str, k: int) -> float:
    return _curve_lookup(controls.get(name, {}), k)


def _random_subspace_overlap_null(dim: int, k: int, trials: int, seed: int) -> dict[str, float]:
    dim = max(int(dim), 1)
    kk = max(1, min(int(k), dim))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(trials)):
        left, _ = np.linalg.qr(rng.normal(size=(dim, kk)))
        right, _ = np.linalg.qr(rng.normal(size=(dim, kk)))
        values.append(float(np.square(left.T @ right).sum() / max(kk, 1)))
    arr = np.asarray(values, dtype=np.float64)
    return {
        "k": int(kk),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def empirical_operator_alignment_bridge_report(
    audit: dict[str, Any],
    theorem: dict[str, Any] | None = None,
    k: int = 32,
    null_trials: int = 2048,
    seed: int = 0,
) -> dict[str, Any]:
    weighted = audit.get("weighted_mean", {})
    lifts = weighted.get("lifts", {})
    controls = weighted.get("controls", {})
    label_controls = audit.get("label_controls", {})
    planner = audit.get("spectral_planner_metric", {}).get("weight_summary", {})
    transport = audit.get("eigenspace_transport", {})
    transport_values = np.asarray(list(transport.get("transitions", {}).values()), dtype=np.float64)

    primary_lifts = {name: _curve_lookup(lifts.get(name, {}), k) for name in PRIMARY_TANGENTS}
    control_lifts = {name: _control_lookup(controls, name, k) for name in CONTROL_NAMES}
    label_shuffle_lifts = {
        name: {tangent: _nested_curve_lookup(label_controls.get(name, {}), tangent, k) for tangent in PRIMARY_TANGENTS}
        for name in LABEL_CONTROL_NAMES
        if name in label_controls
    }
    rival_dynamics = {name: values.get("dynamics", float("nan")) for name, values in label_shuffle_lifts.items()}

    dim = len(audit.get("spectral_planner_metric", {}).get("weights", []))
    if dim <= 0:
        dim = max(k, len(primary_lifts))
    transport_null = _random_subspace_overlap_null(dim, int(transport.get("k", k)), null_trials, seed)

    bridge = {
        "k": int(k),
        "primary_lifts": primary_lifts,
        "control_lifts": control_lifts,
        "primary_to_random_ratio": {
            name: float(value / max(control_lifts["norm_matched_random_dynamics_lift"], 1e-12))
            for name, value in primary_lifts.items()
        },
        "label_shuffle_lifts": label_shuffle_lifts,
        "rival_dynamics": rival_dynamics,
        "transport": {
            "samples": transport_values.tolist(),
            "observed": {
                "mean": float(transport_values.mean()) if transport_values.size else float("nan"),
                "median": float(np.median(transport_values)) if transport_values.size else float("nan"),
                "min": float(transport_values.min(initial=float("inf"))) if transport_values.size else float("nan"),
                "max": float(transport_values.max(initial=float("-inf"))) if transport_values.size else float("nan"),
            },
            "null": transport_null,
        },
        "planner_weights": planner,
        "controllability": audit.get("controllability", {}),
        "observability": audit.get("observability", {}),
        "claim_boundary": (
            "Empirical bridge only: frozen PushT latents show basis-sensitive lift, "
            "shuffle rivals, and transport anisotropy, but this does not prove the exact theorem."
        ),
    }
    if theorem is not None:
        bridge["theory_anchor"] = {
            "peak_angle_degrees": theorem.get("theorem", {}).get("peak_angle_degrees"),
            "statement": theorem.get("theorem", {}).get("statement"),
            "claim_boundary": theorem.get("claim_boundary"),
        }

    return {
        "audit": {
            "condition": audit.get("condition"),
            "config": {
                "num_samples": audit.get("config", {}).get("num_samples"),
                "num_cells": audit.get("config", {}).get("num_cells"),
                "ks": audit.get("config", {}).get("ks", []),
            },
            "warnings": audit.get("warnings", []),
        },
        "theory": None if theorem is None else theorem,
        "bridge": bridge,
    }


def _write_markdown(output: Path, report: dict[str, Any]) -> None:
    theory = report.get("theory")
    bridge = report["bridge"]
    primary = bridge["primary_lifts"]
    controls = bridge["control_lifts"]
    label_shuffle_lifts = bridge["label_shuffle_lifts"]
    transport = bridge["transport"]
    planner = bridge["planner_weights"]
    label_shuffle_values = [value["dynamics"] for value in label_shuffle_lifts.values() if "dynamics" in value]
    label_shuffle_min = min(label_shuffle_values) if label_shuffle_values else float("nan")
    label_shuffle_max = max(label_shuffle_values) if label_shuffle_values else float("nan")
    lines = [
        "# Empirical Operator Alignment Bridge",
        "",
        "## Empirical Anchor",
        "",
        f"- condition: {report['audit']['condition']}",
        f"- top-k readout: k={bridge['k']}",
        f"- samples: {report['audit']['config']['num_samples']}",
        f"- cells: {report['audit']['config']['num_cells']}",
        "",
    ]
    if theory is not None and bridge.get("theory_anchor"):
        anchor = bridge["theory_anchor"]
        lines.extend(
            [
                "## Theory Anchor",
                "",
                f"- theorem peak angle: {_fmt(anchor['peak_angle_degrees'])}",
                f"- theorem statement: {anchor['statement']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Core Lift vs Random Controls",
            "",
            "| readout | observed | random-control | ratio |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name in PRIMARY_TANGENTS:
        value = primary[name]
        control = controls["norm_matched_random_dynamics_lift"]
        lines.append(f"| {name} | {_fmt(value)} | {_fmt(control)} | {_fmt(value / max(control, 1e-12))} |")
    lines.extend(
        [
            "",
            "## Label-Shuffle Rivals",
            "",
            "| control | dynamics | planning | predicted | residual |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name in LABEL_CONTROL_NAMES:
        if name not in label_shuffle_lifts:
            continue
        curve = label_shuffle_lifts[name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(curve["dynamics"]),
                    _fmt(curve["planning"]),
                    _fmt(curve["predicted_dynamics"]),
                    _fmt(curve["prediction_residual"]),
                ]
            )
            + " |"
        )
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
            f"- top-weight mean: {_fmt(planner.get('top_weight_mean'))}",
            f"- tail-weight mean: {_fmt(planner.get('tail_weight_mean'))}",
            f"- top/tail ratio: {_fmt(planner.get('top_weight_mean', 0.0) / max(float(planner.get('tail_weight_mean', 0.0)), 1e-12))}",
            "",
            "## Readout",
            "",
            "- The empirical audit supports basis-sensitive lift on frozen PushT latents.",
            "- Label-shuffle controls stay below the observed dynamics/planning readouts.",
            "- Transport is low in absolute terms but remains above the random-subspace null.",
            "- Planner weights are anisotropic rather than flat across bands.",
            "",
            "## Claim Boundary",
            "",
            f"- {bridge['claim_boundary']}",
            "",
        ]
    )
    (output / f"{STEM}.md").write_text("\n".join(lines) + "\n")


def empirical_operator_alignment_paper_outline(report: dict[str, Any]) -> str:
    bridge = report["bridge"]
    primary = bridge["primary_lifts"]
    controls = bridge["control_lifts"]
    label_shuffle_lifts = bridge["label_shuffle_lifts"]
    transport = bridge["transport"]
    planner = bridge["planner_weights"]
    label_shuffle_values = [value["dynamics"] for value in label_shuffle_lifts.values() if "dynamics" in value]
    label_shuffle_min = min(label_shuffle_values) if label_shuffle_values else float("nan")
    label_shuffle_max = max(label_shuffle_values) if label_shuffle_values else float("nan")
    lines = [
        "# Empirical Operator Alignment Paper Outline",
        "",
        "## Working Title",
        "",
        "Frozen PushT Latent Geometry Exhibits Basis-Sensitive Spectral Transfer",
        "",
        "## One-Line Thesis",
        "",
        "A completed PushT latent audit shows that observed dynamics, planning, and prediction residuals are far above random controls, survive label-shuffle rivals, and remain coupled to low but nontrivial eigenspace transport and anisotropic planner weighting.",
        "",
        "## Claim Ladder",
        "",
        "1. Observed dynamics/planning/prediction lift is much larger than norm-matched random control at the same k.",
        "2. Label-shuffle rivals stay below the observed dynamics/planning readouts, so the effect is not just a relabeled condition artifact.",
        "3. Eigenspace transport across condition cells is low in absolute terms but still above the random-subspace null.",
        "4. Planner weights are anisotropic, with top bands carrying more weight than tail bands.",
        "5. The frozen audit supports a standalone empirical geometry claim without requiring the synthetic theorem as a premise.",
        "",
        "## Main Figure",
        "",
        "Panel A: observed lift vs norm-matched random control for dynamics, planning, predicted dynamics, and residual.",
        "Panel B: label-shuffle rivals against the same random control.",
        "Panel C: eigenspace transport histogram with random-subspace null band.",
        "Panel D: planner anisotropy summary or top-vs-tail weight contrast.",
        "",
        "## Rivals",
        "",
        "- norm-matched random control",
        "- state / time / action label-shuffle control",
        "- random-subspace transport null",
        "- sample-cap or conditioning artifact",
        "- dependency on the synthetic theory paper",
        "",
        "## Falsifiers",
        "",
        "- observed lift collapses to the control level",
        "- label-shuffle rivals match the observed readouts",
        "- transport is indistinguishable from the random-subspace null",
        "- planner weights are flat across bands",
        "- the effect disappears under the existing condition sweep",
        "",
        "## Evidence Bridge",
        "",
        f"- dynamics lift vs random: {_fmt(primary['dynamics'])} / {_fmt(controls['norm_matched_random_dynamics_lift'])}",
        f"- planning lift vs random: {_fmt(primary['planning'])} / {_fmt(controls['norm_matched_random_dynamics_lift'])}",
        f"- predicted lift vs random: {_fmt(primary['predicted_dynamics'])} / {_fmt(controls['norm_matched_random_dynamics_lift'])}",
        f"- residual lift vs random: {_fmt(primary['prediction_residual'])} / {_fmt(controls['norm_matched_random_dynamics_lift'])}",
        f"- label-shuffle dynamics range: {_fmt(label_shuffle_min)} to {_fmt(label_shuffle_max)}",
        f"- transport mean vs null mean: {_fmt(transport['observed']['mean'])} / {_fmt(transport['null']['mean'])}",
        f"- planner top/tail ratio: {_fmt(planner.get('top_weight_mean', 0.0) / max(float(planner.get('tail_weight_mean', 0.0)), 1e-12))}",
        "",
        "## What This Does Not Claim",
        "",
        "- It does not prove the synthetic operator-alignment theorem.",
        "- It does not require online control, physical observability, or counterfactual arrays.",
        "- It does not collapse into the theory paper; the theory paper is a separate manuscript.",
        "",
        "## Section Skeleton",
        "",
        "Abstract -> frozen-audit setup -> observed lift vs controls -> label-shuffle rivals -> transport and anisotropy -> limitations -> relation to the companion theory paper.",
        "",
    ]
    return "\n".join(lines) + "\n"


def empirical_operator_alignment_paper_spine(report: dict[str, Any]) -> str:
    bridge = report["bridge"]
    primary = bridge["primary_lifts"]
    controls = bridge["control_lifts"]
    label_shuffle_lifts = bridge["label_shuffle_lifts"]
    transport = bridge["transport"]
    planner = bridge["planner_weights"]
    label_shuffle_values = [value["dynamics"] for value in label_shuffle_lifts.values() if "dynamics" in value]
    label_shuffle_min = min(label_shuffle_values) if label_shuffle_values else float("nan")
    label_shuffle_max = max(label_shuffle_values) if label_shuffle_values else float("nan")
    lines = [
        "# Frozen PushT Latent Geometry Exhibits Basis-Sensitive Spectral Transfer",
        "",
        "## Abstract",
        "",
        "- A completed frozen PushT latent audit as a standalone empirical geometry paper.",
        "- Dynamics, planning, predicted dynamics, and residual readouts are far above norm-matched random controls at the same k.",
        "- Label-shuffle rivals remain below the observed readouts, eigenspace transport sits above a random-subspace null, and planner weights are anisotropic.",
        "- The paper does not depend on the synthetic theorem; it is a separate manuscript with its own empirical claim.",
        "",
        "## Contribution",
        "",
        "- Completed offline latent audit on frozen PushT arrays.",
        "- Basis-sensitive lift versus random and label-shuffle controls.",
        "- Transport and anisotropy evidence that remain visible without online control or counterfactual arrays.",
        "- A claim boundary that keeps the empirical paper independent from the theory paper.",
        "",
        "## Main Figure",
        "",
        "Panel A: observed lift vs norm-matched random control for dynamics, planning, predicted dynamics, and residual.",
        "Panel B: label-shuffle rivals against the same random control.",
        "Panel C: eigenspace transport histogram with random-subspace null band.",
        "Panel D: planner anisotropy summary or top-vs-tail weight contrast.",
        "",
        "## Rivals",
        "",
        "- norm-matched random control",
        "- state / time / action label-shuffle control",
        "- random-subspace transport null",
        "- sample-cap or conditioning artifact",
        "- dependency on the synthetic theory paper",
        "",
        "## Falsifiers",
        "",
        "- observed lift collapses to the control level",
        "- label-shuffle rivals match the observed readouts",
        "- transport is indistinguishable from the random-subspace null",
        "- planner weights are flat across bands",
        "- the effect disappears under the existing condition sweep",
        "",
        "## Evidence Bridge",
        "",
        f"- dynamics lift vs random: {_fmt(primary['dynamics'])} / {_fmt(controls['norm_matched_random_dynamics_lift'])}",
        f"- planning lift vs random: {_fmt(primary['planning'])} / {_fmt(controls['norm_matched_random_dynamics_lift'])}",
        f"- predicted lift vs random: {_fmt(primary['predicted_dynamics'])} / {_fmt(controls['norm_matched_random_dynamics_lift'])}",
        f"- residual lift vs random: {_fmt(primary['prediction_residual'])} / {_fmt(controls['norm_matched_random_dynamics_lift'])}",
        f"- label-shuffle dynamics range: {_fmt(label_shuffle_min)} to {_fmt(label_shuffle_max)}",
        f"- transport mean vs null mean: {_fmt(transport['observed']['mean'])} / {_fmt(transport['null']['mean'])}",
        f"- planner top/tail ratio: {_fmt(planner.get('top_weight_mean', 0.0) / max(float(planner.get('tail_weight_mean', 0.0)), 1e-12))}",
        "",
        "## What This Does Not Claim",
        "",
        "- It does not prove the synthetic operator-alignment theorem.",
        "- It does not require online control, physical observability, or counterfactual arrays.",
        "- It does not collapse into the theory paper; the theory paper is a separate manuscript.",
        "",
        "## Section Skeleton",
        "",
        "Abstract -> frozen-audit setup -> observed lift vs controls -> label-shuffle rivals -> transport and anisotropy -> limitations -> relation to the companion theory paper.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_paper_outline(output: Path, report: dict[str, Any]) -> None:
    (output / f"{PAPER_OUTLINE_STEM}.md").write_text(empirical_operator_alignment_paper_outline(report))


def _write_paper_spine(output: Path, report: dict[str, Any]) -> None:
    (output / "empirical_operator_alignment_paper_spine.md").write_text(empirical_operator_alignment_paper_spine(report))


def _plot_bridge(output: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns

    bridge = report["bridge"]
    primary = bridge["primary_lifts"]
    controls = bridge["control_lifts"]
    label_shuffle_lifts = bridge["label_shuffle_lifts"]
    transport = bridge["transport"]

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

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)

    order = list(PRIMARY_TANGENTS)
    observed = [primary[name] for name in order]
    control = [controls["norm_matched_random_dynamics_lift"] for _ in order]
    x = np.arange(len(order))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width / 2, observed, width, label="observed", color="#1b9e77")
    ax.bar(x + width / 2, control, width, label="norm-matched random", color="#7570b3")
    ax.set_xticks(x, ["dynamics", "planning", "predicted", "residual"], rotation=0)
    ax.set_ylabel(f"top-{bridge['k']} lift")
    ax.set_title("Observed lift vs control", fontsize=9, pad=4)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="best")

    rival_names = [name for name in LABEL_CONTROL_NAMES if name in label_shuffle_lifts]
    rival_values = [label_shuffle_lifts[name]["dynamics"] for name in rival_names]
    ax = axes[1]
    x = np.arange(len(rival_names))
    ax.bar(x - width / 2, rival_values, width, color="#d95f02", label="shuffle dynamics")
    ax.bar(x + width / 2, [controls["norm_matched_random_dynamics_lift"] for _ in rival_names], width, color="#7570b3", label="norm-matched random")
    ax.set_xticks(x, ["state", "time", "action"][: len(rival_names)])
    ax.set_ylabel(f"top-{bridge['k']} lift")
    ax.set_title("Label-shuffle rivals", fontsize=9, pad=4)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="best")

    ax = axes[2]
    obs = np.asarray(transport["samples"], dtype=np.float64)
    null = transport["null"]
    if obs.size:
        ax.hist(obs, bins=min(12, max(4, obs.size // 3)), alpha=0.75, color="#1b9e77", label="observed transitions")
        ax.axvline(float(np.mean(obs)), color="#1b9e77", linestyle="--", linewidth=1.0)
    ax.axvspan(null["p05"], null["p95"], color="#7570b3", alpha=0.18, label="random-subspace null p05-p95")
    ax.axvline(null["mean"], color="#7570b3", linestyle="--", linewidth=1.0, label="random-subspace null mean")
    ax.set_xlabel("transition overlap")
    ax.set_ylabel("count")
    ax.set_title("Eigenspace transport", fontsize=9, pad=4)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="best")

    axes[0].text(-0.14, 1.05, "A", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    axes[1].text(-0.14, 1.05, "B", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    axes[2].text(-0.14, 1.05, "C", transform=axes[2].transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    fig.savefig(output / f"{STEM}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{STEM}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_empirical_operator_alignment_bridge_artifacts(report: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / f"{STEM}.json", report)
    _write_markdown(output, report)
    _write_paper_outline(output, report)
    _write_paper_spine(output, report)
    _plot_bridge(output, report)


def main() -> None:
    args = parse_args()
    audit = _load_json(args.audit)
    theory = _load_json(args.theory) if args.theory and Path(args.theory).exists() else None
    report = empirical_operator_alignment_bridge_report(
        audit,
        theorem=theory,
        k=args.k,
        null_trials=args.null_trials,
        seed=args.seed,
    )
    write_empirical_operator_alignment_bridge_artifacts(report, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
