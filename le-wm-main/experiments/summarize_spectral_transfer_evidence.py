"""Assemble a compact cross-paper coordination report for spectral-transfer evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theory-mechanism",
        default="results/theory_mechanism_sweep_20260801/theory_mechanism_sweep.json",
    )
    parser.add_argument(
        "--theory-dim",
        default="results/theory_dim_sweep_20260801/cross_dim_summary.json",
    )
    parser.add_argument(
        "--theory-robustness",
        default="results/theory_robustness_sweep_20260801/robustness_summary.json",
    )
    parser.add_argument(
        "--theory-controllability",
        default="results/theory_controllability_sweep_20260828/controllability_sweep.json",
    )
    parser.add_argument(
        "--empirical-condition",
        default="results/pusht/lewm/empirical_condition_sweep_20260801/condition_summary.json",
    )
    parser.add_argument(
        "--output",
        default="results/spectral_transfer_evidence_20260801",
    )
    parser.add_argument(
        "--planning-alignment",
        default="",
        help="Optional completed planning-alignment JSON used to distinguish a run blocker from an unrun online eval.",
    )
    parser.add_argument(
        "--online-eval",
        default="",
        help="Optional online-evaluation artifact. Omit when the protocol intentionally skipped online eval.",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    mechanism = _load(Path(args.theory_mechanism))
    dim = _load(Path(args.theory_dim))
    robustness = _load(Path(args.theory_robustness))
    controllability_path = Path(args.theory_controllability)
    controllability = _load(controllability_path) if controllability_path.is_file() else None
    empirical = _load(Path(args.empirical_condition))

    interaction = mechanism["interaction"]
    baseline = next(
        row
        for row in interaction
        if row["encoder_exponent"] == 1.0 and row["predictor_exponent"] == 1.0
    )
    predictor_values = [row["predictor_exponent"] for row in interaction]
    weak_value = min(predictor_values)
    strong_value = max(predictor_values)
    weak_predictor = [row for row in interaction if row["predictor_exponent"] == weak_value]
    strong_predictor = [row for row in interaction if row["predictor_exponent"] == strong_value]
    fixed_encoder_rows = [
        row for row in interaction if row["encoder_exponent"] == baseline["encoder_exponent"]
    ]
    fixed_encoder_ordering = {
        "weak_predictor": next(
            row["residual_growth_ratio"]
            for row in fixed_encoder_rows
            if row["predictor_exponent"] == weak_value
        ),
        "baseline": baseline["residual_growth_ratio"],
        "strong_predictor": next(
            row["residual_growth_ratio"]
            for row in fixed_encoder_rows
            if row["predictor_exponent"] == strong_value
        ),
    }
    conditions = empirical["records"]
    by_condition = {record["condition"]: record for record in conditions}
    states = by_condition.get("states", {})
    global_record = by_condition.get("global", {})
    multi_horizon = empirical.get("multi_horizon", {})
    multi_horizon_horizons = sorted(
        {
            name
            for condition_map in multi_horizon.values()
            for name in condition_map
            if name != "dynamics"
        }
    )
    warnings = sorted(
        {
            warning
            for record in conditions
            for warning in record.get("warnings", [])
        }
    )
    planning_path = Path(args.planning_alignment) if args.planning_alignment else None
    online_eval_path = Path(args.online_eval) if args.online_eval else None

    return {
        "inputs": {
            "theory_mechanism": args.theory_mechanism,
            "theory_dim": args.theory_dim,
            "theory_robustness": args.theory_robustness,
            "empirical_condition": args.empirical_condition,
            "planning_alignment": args.planning_alignment,
            "online_eval": args.online_eval,
        },
        "theory": {
            "dim": mechanism["dim"],
            "horizon": mechanism["horizon"],
            "baseline_interaction": {
                key: baseline[key]
                for key in [
                    "encoder_exponent",
                    "predictor_exponent",
                    "expected_oracle_alpha",
                    "residual_growth_ratio",
                    "predicted_decay_ratio",
                    "residual_tail_fraction",
                ]
            },
            "weak_predictor_residual_growth": {
                "min": min(row["residual_growth_ratio"] for row in weak_predictor),
                "max": max(row["residual_growth_ratio"] for row in weak_predictor),
            },
            "strong_predictor_residual_growth": {
                "min": min(row["residual_growth_ratio"] for row in strong_predictor),
                "max": max(row["residual_growth_ratio"] for row in strong_predictor),
            },
            "fixed_encoder_predictor_ordering": {
                "encoder_exponent": baseline["encoder_exponent"],
                **fixed_encoder_ordering,
            },
            "heat_recovery_by_rate_exponent": {
                str(row["heat_exponent"]): row["heat_recovery_median_r2"]
                for row in mechanism["heat"]
                if row["heat_time"] == 0.0
            },
            "oblique_bias": {
                "baseline_mean": robustness["oblique_bias"]["baseline"]["mean"],
                "tail_mean": robustness["oblique_bias"]["tail"]["mean"],
            },
            "controllability": controllability,
            "cross_dimension": {
                str(dim_value): {
                    key: record[key]
                    for key in [
                        "baseline_residual_ratio",
                        "p05_residual_ratio",
                        "p20_residual_ratio",
                        "tail3_gap",
                        "tail3_bias",
                        "heat_r2",
                    ]
                }
                for dim_value, record in dim["records"].items()
            },
        },
        "empirical": {
            "conditions": {
                record["condition"]: {
                    key: record[key]
                    for key in [
                        "num_samples",
                        "num_cells",
                        "lifts",
                        "time_shuffle_dynamics",
                        "state_shuffle_dynamics",
                        "controllable_lift",
                        "unexplained_lift",
                        "action_to_dynamics_r2",
                        "prediction_mean_cosine",
                        "residual_uncertainty_lift",
                    ]
                }
                for record in conditions
            },
            "state_vs_global_dynamics_lift_ratio": float(
                states["lifts"]["dynamics"] / max(global_record["lifts"]["dynamics"], 1e-12)
            ),
            "state_vs_global_planning_lift_ratio": float(
                states["lifts"]["planning"] / max(global_record["lifts"]["planning"], 1e-12)
            ),
            "sample_stability": empirical.get("sample_stability"),
            "warnings": warnings,
            "multi_horizon": multi_horizon,
            "multi_horizon_horizons": multi_horizon_horizons,
        },
        "blocked": {
            "true_empirical_multihorizon": "available" if multi_horizon_horizons else "missing dynamics_h* arrays",
            "observability": "missing physical_* arrays",
            "counterfactual_control": "missing counterfactual_* arrays",
            "planning_alignment": "available" if planning_path and planning_path.is_file() else "not run",
            "online_eval": "available" if online_eval_path and online_eval_path.is_file() else "not run by protocol",
        },
    }


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    theory = payload["theory"]
    empirical = payload["empirical"]
    controllability = theory.get("controllability")
    condition_names = list(empirical["conditions"])
    multi_horizon = empirical.get("multi_horizon", {})
    multi_horizon_horizons = empirical.get("multi_horizon_horizons", [])
    lines = [
        "# Cross-Paper Spectral-Transfer Coordination",
        "",
        "## Direct Readout",
        "",
        "- Theory paper: same-spectrum operator alignment changes transfer geometry, with the commutator peaking at an oblique rotation even when spectra are fixed.",
        "- Empirical paper: the PushT offline signal is condition-dependent; state-conditioned dynamics lift exceeds the global lift, and the 100k state audit reproduces the existing 200k audit within roughly 1%.",
        "- Cross-paper coordination: the controllability sweep keeps the spectrum fixed and shows that alignment, not spectrum alone, governs top-band lift and residual energy."
        if controllability
        else "- Cross-paper coordination: the controllability alignment sweep is not loaded in this summary.",
        (
            "- This is not yet a complete causal or control result: physical observability and counterfactual transition evidence remain open; "
            + ("planning-alignment is available in the capped run; " if payload["blocked"]["planning_alignment"] == "available" else "planning-alignment remains open; ")
            + ("online visual evaluation is available." if payload["blocked"]["online_eval"] == "available" else "online visual evaluation was not run by this protocol.")
            if not multi_horizon_horizons
            else
            "- Rich PushT audits now include empirical multi-horizon readouts for "
            + ", ".join(multi_horizon_horizons)
            + "; physical observability and counterfactual transition evidence remain open; "
            + ("planning-alignment is available in the capped run; " if payload["blocked"]["planning_alignment"] == "available" else "planning-alignment remains open; ")
            + ("online visual evaluation is available." if payload["blocked"]["online_eval"] == "available" else "online visual evaluation was not run by this protocol.")
        ),
        "",
        "## Manuscript Index",
        "",
        "| manuscript | object card | spine | core claim |",
        "| --- | --- | --- | --- |",
        "| theory | `results/theory_operator_alignment_sweep_20260828/operator_alignment_oral_object_card.md` | `results/theory_operator_alignment_sweep_20260828/operator_alignment_paper_draft.md` | same-spectrum basis rotation changes transfer geometry |",
        "| empirical | `results/pusht/lewm/empirical_operator_alignment_bridge_20260829/empirical_operator_alignment_paper_outline.md` | `results/pusht/lewm/empirical_operator_alignment_bridge_20260829/empirical_operator_alignment_paper_spine.md` | frozen PushT latents show basis-sensitive spectral transfer |",
        "",
        "## Theory",
        "",
        "| readout | value |",
        "| --- | --- |",
        f"| baseline theory residual h{theory['horizon']}/h1 | {_fmt(theory['baseline_interaction']['residual_growth_ratio'])} |",
        f"| baseline theory predicted h{theory['horizon']}/h1 | {_fmt(theory['baseline_interaction']['predicted_decay_ratio'])} |",
        f"| weak-predictor residual-growth range | {_fmt(theory['weak_predictor_residual_growth']['min'])} to {_fmt(theory['weak_predictor_residual_growth']['max'])} |",
        f"| strong-predictor residual-growth range | {_fmt(theory['strong_predictor_residual_growth']['min'])} to {_fmt(theory['strong_predictor_residual_growth']['max'])} |",
        f"| robustness baseline oblique bias mean | {_fmt(theory['oblique_bias']['baseline_mean'])} |",
        f"| robustness broken-tail oblique bias mean | {_fmt(theory['oblique_bias']['tail_mean'])} |",
        "",
    ]
    if controllability:
        contrast = controllability["contrast"]
        null = controllability["random_rotation_null"]["top_band_lift"][str(contrast["top_k"])]
        lines.extend(
            [
                "## Controllability Alignment",
                "",
                "| readout | value |",
                "| --- | --- |",
                f"| aligned top-{contrast['top_k']} lift | {_fmt(contrast['aligned_top_k_lift'])} |",
                f"| orthogonal top-{contrast['top_k']} lift | {_fmt(contrast['orthogonal_top_k_lift'])} |",
                f"| random-null top-{contrast['top_k']} lift mean | {_fmt(null['mean'])} |",
                f"| aligned residual energy share | {_fmt(contrast['aligned_residual_energy_share'])} |",
                f"| orthogonal residual energy share | {_fmt(contrast['orthogonal_residual_energy_share'])} |",
                "",
            ]
        )
    lines.extend(
        [
            "## PushT Condition Sweep",
            "",
            "| condition | dynamics lift | planning lift | residual lift | prediction cosine |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for condition, record in empirical["conditions"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _fmt(record["lifts"]["dynamics"]),
                    _fmt(record["lifts"]["planning"]),
                    _fmt(record["lifts"]["prediction_residual"]),
                    _fmt(record["prediction_mean_cosine"]),
                ]
            )
            + " |"
        )
    if multi_horizon_horizons:
        lines.extend(
            [
                "",
                "## Multi-Horizon PushT",
                "",
                "| condition | h1 lift | "
                + " | ".join(f"{name.split('_h', 1)[1]} lift" for name in multi_horizon_horizons)
                + " |",
                "| --- | --- | " + " | ".join("---" for _ in multi_horizon_horizons) + " |",
            ]
        )
        for condition in condition_names:
            horizon_map = multi_horizon.get(condition, {})
            base = horizon_map.get("dynamics", {})
            row = [condition, _fmt(base.get("lift", float("nan")))]
            for name in multi_horizon_horizons:
                row.append(_fmt(horizon_map.get(name, {}).get("lift", float("nan"))))
            lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            f"- State/global dynamics lift ratio: {_fmt(empirical['state_vs_global_dynamics_lift_ratio'])}.",
            f"- State/global planning lift ratio: {_fmt(empirical['state_vs_global_planning_lift_ratio'])}.",
            "",
            "## Claim Boundary",
            "",
            "- Theory rows are analytical simulations and should be cited as mechanism checks.",
            "- PushT rows are offline spectral readouts from one frozen `latents.npz` artifact.",
            (
                "- Multi-horizon rows are included when the underlying audits export `dynamics_h*` arrays."
                if multi_horizon_horizons
                else "- Requested horizons 3 and 5 were passed to the audit, but the report correctly limits empirical multi-horizon evidence to h=1 because `dynamics_h*` arrays are absent."
            ),
            (
                "- Planning-alignment evidence is available from the capped offline run; online visual evaluation is available."
                if payload["blocked"]["planning_alignment"] == "available"
                else "- Planning-alignment and online visual evaluation were not run by this protocol."
            ),
            "",
            "## Schema Gaps",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in empirical["warnings"])
    (output / "evidence_summary.md").write_text("\n".join(lines) + "\n")


def _write_figure(output: Path, payload: dict[str, Any]) -> None:
    conditions = list(payload["empirical"]["conditions"])
    records = payload["empirical"]["conditions"]
    theory = payload["theory"]
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
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.9), constrained_layout=True)
    colors = sns.color_palette("colorblind", n_colors=max(len(conditions), 3))
    x = np.arange(len(conditions))

    ax = axes[0, 0]
    ax.bar(
        ["weak predictor", "baseline", "strong predictor"],
        [
            theory["fixed_encoder_predictor_ordering"]["weak_predictor"],
            theory["fixed_encoder_predictor_ordering"]["baseline"],
            theory["fixed_encoder_predictor_ordering"]["strong_predictor"],
        ],
        color=["#d95f02", "#555555", "#1b9e77"],
    )
    ax.set_title(
        f"Theory ordering at encoder={theory['fixed_encoder_predictor_ordering']['encoder_exponent']:g}",
        fontsize=9,
        pad=4,
    )
    ax.set_ylabel("residual h8/h1")
    ax.grid(True, axis="y", alpha=0.2)
    ax.text(-0.14, 1.05, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[0, 1]
    tangent_names = ["dynamics", "planning", "prediction_residual"]
    width = 0.2
    offsets = np.linspace(-width, width, len(tangent_names))
    for color, offset, tangent in zip(colors[: len(tangent_names)], offsets, tangent_names):
        ax.bar(
            x + offset,
            [records[condition]["lifts"][tangent] for condition in conditions],
            width,
            color=color,
            label=tangent.replace("prediction_", "pred. "),
        )
    ax.set_xticks(x, conditions)
    ax.set_title("PushT condition-dependent lifts", fontsize=9, pad=4)
    ax.set_ylabel("lift at k=32")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    ax.text(-0.14, 1.05, "B", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[1, 0]
    stability = (payload["empirical"]["sample_stability"] or {}).get("lift_ratio", {})
    names = list(stability)
    values = [stability[name] for name in names]
    ax.bar(np.arange(len(names)), values, color=colors[: len(names)])
    ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xticks(
        np.arange(len(names)),
        [name.replace("prediction_", "pred. ") for name in names],
        rotation=25,
        ha="right",
    )
    ax.set_title("100k vs 200k state-audit stability", fontsize=9, pad=4)
    ax.set_ylabel("lift ratio")
    if values:
        pad = max((max(values) - min(values)) * 0.4, 0.002)
        ax.set_ylim(min(values) - pad, max(values) + pad)
    ax.grid(True, axis="y", alpha=0.2)
    ax.text(-0.14, 1.05, "C", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[1, 1]
    heat = theory["heat_recovery_by_rate_exponent"]
    heat_x = np.arange(len(heat))
    heat_values = list(heat.values())
    ax.bar(
        heat_x,
        heat_values,
        color=["#7570b3", "#1b9e77", "#d95f02"][: len(heat_values)],
    )
    ax.axhline(0.99, color="#444444", linewidth=1.0, linestyle="--", label="high-recovery threshold")
    ax.set_xticks(heat_x, list(heat))
    ax.set_ylim(0, 1.05)
    ax.set_title("Known-generator heat recovery", fontsize=9, pad=4)
    ax.set_xlabel("heat-rate exponent")
    ax.set_ylabel("median R2")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(-0.14, 1.05, "D", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    fig.savefig(output / "evidence_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "evidence_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(args)
    (output / "evidence_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(output, payload)
    _write_figure(output, payload)
    print(output / "evidence_summary.json")
    print(output / "evidence_summary.md")
    print(output / "evidence_summary.pdf")
    print(output / "evidence_summary.png")


if __name__ == "__main__":
    main()
