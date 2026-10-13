"""Build a cross-experiment claim matrix without pooling estimands.

This report is intentionally a *join* over completed artifacts.  It never
combines observations from different benchmarks into one confidence interval.
Instead, it computes paired seed-level summaries inside each artifact and
labels an intervention as geometry-only, functional, jointly successful, or
mixed.  The distinction is useful here because lower commutators and lower
equivariance errors are not sufficient evidence of better rollout or planning
proxy performance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable


LOWER_IS_BETTER = {
    "one_step_mse",
    "rollout_h8_mse",
    "composition_error",
    "predictor_equivariance_error",
    "operator_alignment",
    "conditional_local_commutator",
    "planning_regret",
    "rollout_id_h1",
    "rollout_id_h4",
    "rollout_id_h8",
    "planning_ood_h4",
    "planning_ood_h8",
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    if not _finite(value):
        raise FloatingPointError(f"non-finite value in {path}")
    return value


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return float(sum(values) / len(values))


def _bootstrap_ci(values: list[float], seed: int, repeats: int = 10000) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty collection")
    rng = random.Random(seed)
    means = [
        _mean(rng.choice(values) for _ in range(len(values)))
        for _ in range(repeats)
    ]
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def _paired_summary(
    target: dict[int, dict[str, float]],
    baseline: dict[int, dict[str, float]],
    metrics: list[str],
    seed: int,
) -> dict[str, Any]:
    seeds = sorted(set(target).intersection(baseline))
    if len(seeds) < 3:
        raise ValueError(f"need at least three paired seeds, got {len(seeds)}")
    output: dict[str, Any] = {"n_paired_seeds": len(seeds), "metrics": {}}
    for index, metric in enumerate(metrics):
        deltas = [target[s][metric] - baseline[s][metric] for s in seeds]
        output["metrics"][metric] = {
            "mean_delta_target_minus_baseline": _mean(deltas),
            "bootstrap_ci95": _bootstrap_ci(deltas, seed + index),
            "fraction_target_better": sum(value < 0.0 for value in deltas) / len(deltas),
            "paired_deltas": deltas,
            "lower_is_better": metric in LOWER_IS_BETTER,
        }
    return output


def _joint_rows(source: dict[str, Any]) -> dict[tuple[str, float, float], dict[int, dict[str, float]]]:
    rows: dict[tuple[str, float, float], dict[int, dict[str, float]]] = {}
    for row in source["records"]:
        key = (str(row["mode"]), float(row["predictor_weight"]), float(row["alignment_weight"]))
        rows.setdefault(key, {})[int(row["seed"])] = {
            "one_step_mse": float(row["test_one_step_mse"]),
            "rollout_h8_mse": float(row["rollout_mse"]["8"]),
            "composition_error": float(row["composition_error"]),
            "predictor_equivariance_error": float(row["equivariance"]["predictor_equivariance_error"]),
            "operator_alignment": float(row["operator_alignment"]["operator_alignment"]),
        }
    return rows


def _classification(
    summary: dict[str, Any], geometry: list[str], functional: list[str]
) -> dict[str, Any]:
    gm = summary["metrics"]
    geometry_success = all(gm[name]["fraction_target_better"] == 1.0 for name in geometry)
    functional_success = all(gm[name]["fraction_target_better"] == 1.0 for name in functional)
    all_success = geometry_success and functional_success
    if all_success:
        label = "all-metric joint success"
    elif geometry_success and not functional_success:
        label = "geometry-only improvement"
    elif functional_success and not geometry_success:
        label = "functional improvement without geometry dominance"
    else:
        label = "mixed or negative"
    return {
        "label": label,
        "geometry_all_seed_better": geometry_success,
        "functional_all_seed_better": functional_success,
        "all_metrics_all_seed_better": all_success,
    }


def _matched_rows(source: dict[str, Any], mode: str, weight: float) -> dict[int, dict[str, float]]:
    rows = {
        int(row["seed"]): {name: float(value) for name, value in row["deltas"].items()}
        for row in source["records"]
        if str(row["mode"]) == mode and float(row["alignment_weight"]) == float(weight)
    }
    if not rows:
        raise ValueError(f"missing matched rows for {mode}:{weight}")
    return rows


def _multihorizon_rows(source: dict[str, Any], condition: str) -> dict[int, dict[str, float]]:
    by_seed: dict[int, dict[str, float]] = {}
    for row in source["records"]:
        if str(row.get("condition", row.get("profile"))) != condition:
            continue
        seed = int(row["seed"])
        family = str(row.get("metric_family", row.get("family")))
        shift = float(row["state_shift"])
        horizon = int(row["horizon"])
        if family == "rollout_mse" and shift == 0.0 and horizon in (1, 4, 8):
            name = f"rollout_id_h{horizon}"
        elif family == "planning_regret" and shift == 1.5 and horizon in (4, 8):
            name = f"planning_ood_h{horizon}"
        else:
            continue
        by_seed.setdefault(seed, {})[name] = float(row["value"])
    required = {"rollout_id_h1", "rollout_id_h4", "rollout_id_h8", "planning_ood_h4", "planning_ood_h8"}
    if any(set(values) != required for values in by_seed.values()):
        raise ValueError(f"incomplete multi-horizon matrix for {condition}")
    return by_seed


def _external_summary(source: dict[str, Any]) -> dict[str, Any]:
    rows = source["rows"]
    supported = [row for row in rows if row["verdict"] == "supported"]
    inconclusive = [row for row in rows if row["verdict"] == "inconclusive"]
    by_norm: dict[str, dict[str, int]] = {}
    for row in rows:
        block = by_norm.setdefault(str(row["normalization"]), {"supported": 0, "inconclusive": 0, "reversed": 0})
        block[str(row["verdict"])] += 1
    return {
        "n_rows": len(rows),
        "supported_rows": len(supported),
        "inconclusive_rows": len(inconclusive),
        "by_normalization": by_norm,
        "public_all_supported": by_norm.get("public", {}).get("supported", 0) == 6,
        "raw_all_inconclusive": by_norm.get("raw_div100", {}).get("inconclusive", 0) == 6,
    }


def _horizon_transport_summary(source: dict[str, Any]) -> dict[str, Any]:
    focus = {
        key: value
        for key, value in source["summary"].items()
        if key.startswith("dynamics_basis/") and key.endswith("/k8")
    }
    expected = {
        "dynamics_basis/h1_to_h2/k8",
        "dynamics_basis/h2_to_h4/k8",
        "dynamics_basis/h4_to_h8/k8",
    }
    if set(focus) != expected:
        raise ValueError("horizon transport summary lacks the three k=8 displacement pairs")
    return {
        "pairs": {
            key.split("/")[1]: {
                "delta_vs_covariance": float(value["delta_vs_covariance_mean_across_sources"]),
                "delta_vs_random": float(value["delta_vs_random_mean_across_sources"]),
                "positive_vs_covariance_all_sources": bool(value["delta_vs_covariance_positive_all_sources"]),
                "positive_vs_random_all_sources": bool(value["delta_vs_random_positive_all_sources"]),
            }
            for key, value in focus.items()
        },
        "all_pairs_positive_vs_covariance": all(
            bool(value["delta_vs_covariance_positive_all_sources"]) for value in focus.values()
        ),
        "all_pairs_positive_vs_random": all(
            bool(value["delta_vs_random_positive_all_sources"]) for value in focus.values()
        ),
        "verdict": source["claim_matrix"][0]["verdict"],
    }


def _action_residual_transport_summary(source: dict[str, Any]) -> dict[str, Any]:
    focus = {
        key: value
        for key, value in source["summary"].items()
        if "/dynamics_basis/" in key and key.endswith("/k8")
    }
    expected = {
        f"{normalization}/dynamics_basis/h{a}_to_h{b}/k8"
        for normalization in ("public", "val_zscore", "raw_div100")
        for a, b in ((1, 2), (2, 4), (4, 8))
    }
    if set(focus) != expected:
        raise ValueError("action-residual transport summary lacks the complete k=8 matrix")
    return {
        "action_probe": source["action_probe"],
        "all_arms_positive_vs_covariance": all(
            bool(value["delta_vs_covariance_positive_all_sources"]) for value in focus.values()
        ),
        "all_arms_positive_vs_random": all(
            bool(value["delta_vs_random_positive_all_sources"]) for value in focus.values()
        ),
        "dynamics_rows": {
            key: {
                "delta_vs_covariance": float(value["delta_vs_covariance_mean_across_sources"]),
                "delta_vs_random": float(value["delta_vs_random_mean_across_sources"]),
                "positive_vs_covariance_all_sources": bool(value["delta_vs_covariance_positive_all_sources"]),
                "positive_vs_random_all_sources": bool(value["delta_vs_random_positive_all_sources"]),
            }
            for key, value in focus.items()
        },
        "verdict": source["claim_matrix"][0]["verdict"],
    }


def run_report(root: Path, output: Path, *, bootstrap_repeats: int = 10000) -> dict[str, Any]:
    del bootstrap_repeats  # Kept in the protocol for explicit provenance; the fixed helper is deterministic.
    results = root / "results"
    joint = _load(results / "alignment_equivariance_joint_balanced_10seed_20261003")
    matched = _load(results / "nonlinear_alignment_phase_matched_10seed_20261003" / "nonlinear_alignment_phase_matched.json")
    multi = _load(results / "nonlinear_planner_metric_ablation_20261003" / "multihorizon_weight_sweep_curriculum.json")
    composition = _load(results / "nonlinear_planner_metric_ablation_20261003" / "composition_aware_training.json")
    external = _load(results / "pusht/lewm/external_public_pusht_temporal_20261003/external_latent_dynamics_transfer_cross_normalization.json")
    horizon_transport = _load(results / "pusht/lewm/external_public_pusht_temporal_20261003/external_dynamics_horizon_transport.json")
    action_residual_transport = _load(results / "pusht/lewm/external_public_pusht_temporal_20261003/external_action_residual_horizon_transport.json")
    dynamics = _load(results / "pusht/lewm/dynamics_residual_planner_evidence_summary_20261004.json")

    joint_map = _joint_rows(joint)
    none_key = ("none", 0.0, 0.0)
    joint_comparisons: dict[str, Any] = {}
    joint_specs = {
        "predictor_only": ("predictor", 1.0, 0.0),
        "alignment_only": ("alignment", 0.0, 0.01),
        "predictor_plus_alignment": ("both", 1.0, 0.01),
    }
    joint_metrics = [
        "one_step_mse", "rollout_h8_mse", "composition_error",
        "predictor_equivariance_error", "operator_alignment",
    ]
    for name, key in joint_specs.items():
        summary = _paired_summary(joint_map[key], joint_map[none_key], joint_metrics, 20261004)
        summary["classification"] = _classification(
            summary,
            ["predictor_equivariance_error", "operator_alignment"],
            ["one_step_mse", "rollout_h8_mse", "composition_error"],
        )
        joint_comparisons[name] = summary

    matched_comparisons: dict[str, Any] = {}
    matched_metrics = [
        "test_one_step_mse", "id_rollout_h8_mse", "ood_rollout_h8_mse",
        "state_shift_composition_error", "planning_trajectory_regret",
        "conditional_local_commutator",
    ]
    for mode, weight in (("global", 0.1), ("conditional", 0.01), ("conditional", 0.1)):
        summary = _paired_summary(
            _matched_rows(matched, mode, weight),
            {int(row["seed"]): {metric: 0.0 for metric in matched_metrics} for row in matched["records"] if int(row["seed"]) in _matched_rows(matched, mode, weight)},
            matched_metrics,
            20261014,
        )
        # The matched artifact already stores intervention-minus-baseline deltas;
        # reconstructing against zero preserves the reported paired estimand.
        summary["classification"] = _classification(
            summary,
            ["conditional_local_commutator"],
            ["test_one_step_mse", "id_rollout_h8_mse", "ood_rollout_h8_mse", "state_shift_composition_error", "planning_trajectory_regret"],
        )
        matched_comparisons[f"{mode}:{weight:g}"] = summary

    multi_comparisons: dict[str, Any] = {}
    baseline_multi = _multihorizon_rows(multi, "one_step")
    multi_metrics = ["rollout_id_h1", "rollout_id_h4", "rollout_id_h8", "planning_ood_h4", "planning_ood_h8"]
    for condition in ("uniform", "short_favoring", "long_favoring", "curriculum"):
        summary = _paired_summary(
            _multihorizon_rows(multi, condition), baseline_multi, multi_metrics, 20261024
        )
        multi_comparisons[condition] = summary

    report = {
        "artifact": "deep_cross_experiment_claim_matrix",
        "status": "complete",
        "protocol": {
            "join_unit": "independent artifact; no rows pooled across benchmarks",
            "paired_unit": "same seed within each artifact",
            "bootstrap_unit": "paired seed-level differences",
            "bootstrap_repeats": 10000,
            "all_metric_rule": "strict improvement on every listed metric for every paired seed",
            "geometry_metrics": ["predictor_equivariance_error", "operator_alignment", "conditional_local_commutator"],
            "functional_metrics": ["one-step/rollout", "composition", "planning proxy"],
        },
        "source_artifacts": {
            "joint_alignment_equivariance": str(results / "alignment_equivariance_joint_balanced_10seed_20261003"),
            "matched_nonlinear_alignment": str(results / "nonlinear_alignment_phase_matched_10seed_20261003/nonlinear_alignment_phase_matched.json"),
            "multihorizon_training": str(results / "nonlinear_planner_metric_ablation_20261003/multihorizon_weight_sweep_curriculum.json"),
            "composition_aware_training": str(results / "nonlinear_planner_metric_ablation_20261003/composition_aware_training.json"),
            "external_latent_transport": str(results / "pusht/lewm/external_public_pusht_temporal_20261003/external_latent_dynamics_transfer_cross_normalization.json"),
            "external_horizon_transport": str(results / "pusht/lewm/external_public_pusht_temporal_20261003/external_dynamics_horizon_transport.json"),
            "external_action_residual_transport": str(results / "pusht/lewm/external_public_pusht_temporal_20261003/external_action_residual_horizon_transport.json"),
            "dynamics_residual_summary": str(results / "pusht/lewm/dynamics_residual_planner_evidence_summary_20261004.json"),
        },
        "joint_alignment_equivariance": joint_comparisons,
        "matched_nonlinear_alignment": matched_comparisons,
        "multihorizon_training": multi_comparisons,
        "composition_aware_training": composition["paired_inference_vs_one_step"]["composition_aware_vs_multi_horizon"],
        "external_transport": _external_summary(external),
        "external_horizon_transport": _horizon_transport_summary(horizon_transport),
        "external_action_residual_transport": _action_residual_transport_summary(action_residual_transport),
        "frozen_dynamics_residual_claim_matrix": dynamics["claim_matrix"],
        "claim_matrix": [
            {
                "claim": "Reducing a geometry diagnostic is sufficient for functional improvement.",
                "evidence": "factorial joint ablation and checkpoint-matched nonlinear phase sweep",
                "verdict": "not supported; geometry-only and mixed outcomes occur",
            },
            {
                "claim": "Predictor equivariance plus alignment gives all-metric joint success.",
                "evidence": "10-seed joint comparison with one-step, h=8 rollout, composition, predictor equivariance, and operator alignment",
                "verdict": joint_comparisons["predictor_plus_alignment"]["classification"]["label"],
            },
            {
                "claim": "A multi-horizon objective uniformly improves rollout and planning proxy performance.",
                "evidence": "10-seed profile sweep across ID h=1/4/8 and OOD planning h=4/8",
                "verdict": "not supported; long-horizon gains coexist with short/mid-horizon OOD losses",
            },
            {
                "claim": "The frozen dynamics basis transports latent dynamics across sources.",
                "evidence": "21-episode external transfer under three action normalizations",
                "verdict": "conditional: public normalization supported, raw/100 inconclusive",
            },
            {
                "claim": "The source-fitted dynamics basis captures external latent displacement across multiple horizon spans.",
                "evidence": "three independent source bases, three external horizon-pair displacements, covariance and random-subspace controls",
                "verdict": "supported as conditional latent-trajectory transport",
            },
            {
                "claim": "Dynamics-basis enrichment survives cross-fitted action residualization in the encoded latent target.",
                "evidence": "leave-one-episode-out action residuals with three source bases, three spans, and three normalization arms",
                "verdict": "supported as action-conditioned latent diagnostic",
            },
            {
                "claim": "The extracted dynamics subspace adds physical-motion prediction beyond observed actions.",
                "evidence": "frozen dynamics/residual evidence summary",
                "verdict": "not supported",
            },
            {
                "claim": "Dynamics and residual subspaces are target-specific rather than interchangeable latent bands.",
                "evidence": (
                    "paired rank curve: dynamics-band h=4 top-8 incremental R2 "
                    f"{dynamics['evidence']['target_specificity']['dynamics_target_h4_incremental_r2']:.6f}, "
                    "residual-band h=8 residual-basis incremental R2 "
                    f"{dynamics['evidence']['target_specificity']['residual_target_h8_incremental_r2']:.6f}, "
                    "physical-displacement h=8 dynamics-basis incremental R2 "
                    f"{dynamics['evidence']['target_specificity']['physical_target_h8_incremental_r2']:.6f}"
                ),
                "verdict": "supported as a target-specific frozen diagnostic",
            },
            {
                "claim": "Explicitly weighting action-switching prefixes improves mixed-action functionality over uniform multi-horizon training.",
                "evidence": "10-seed composition-aware training contrast against the matched multi-horizon condition",
                "verdict": "not supported; the switch-gap diagnostic improves while switch error and OOD rollout worsen",
            },
        ],
        "claim_boundary": [
            "The matrix is a reader-facing audit of separate estimands, not a meta-analysis.",
            "Synthetic vector/pixel/nonlinear experiments are mechanism proxies and are not nonlinear LeWM or PushT performance results.",
            "External frozen PushT rows are latent predictive/rollout diagnostics, not physical counterfactuals, causal control, online planning, or task success.",
            "All-metric joint success is a strict within-artifact diagnostic and is not a universal guarantee.",
        ],
    }
    if not _finite(report):
        raise FloatingPointError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("README.md").write_text(
        "# Deep cross-experiment claim matrix\n\n"
        "This directory joins completed artifacts without pooling rows or confidence intervals. "
        "The JSON is the canonical machine-readable report; the Markdown is the reader-facing summary.\n"
    )
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Deep cross-experiment claim matrix",
        "",
        "This audit joins independent artifacts but never pools their observations or confidence intervals. Paired summaries are computed only within an artifact and use the same random seed across conditions.",
        "",
        "## Claim matrix",
        "",
        "| claim | verdict | evidence |",
        "| --- | --- | --- |",
    ]
    for item in report["claim_matrix"]:
        lines.append(f"| {item['claim']} | {item['verdict']} | {item['evidence']} |")

    lines.extend([
        "",
        "## Strict joint-success audit",
        "",
        "| artifact / comparison | classification | geometry all seeds better | functional all seeds better | all metrics all seeds better |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for name, value in report["joint_alignment_equivariance"].items():
        cls = value["classification"]
        lines.append(
            f"| joint/{name} | {cls['label']} | {str(cls['geometry_all_seed_better'])} | "
            f"{str(cls['functional_all_seed_better'])} | {str(cls['all_metrics_all_seed_better'])} |"
        )
    for name, value in report["matched_nonlinear_alignment"].items():
        cls = value["classification"]
        lines.append(
            f"| matched/{name} | {cls['label']} | {str(cls['geometry_all_seed_better'])} | "
            f"{str(cls['functional_all_seed_better'])} | {str(cls['all_metrics_all_seed_better'])} |"
        )

    lines.extend([
        "",
        "## Multi-horizon paired deltas",
        "",
        "Negative values favor the profile because every listed metric is an error or regret.",
        "",
        "| profile vs one-step | ID h=1 | ID h=4 | ID h=8 | OOD planning h=4 | OOD planning h=8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, value in report["multihorizon_training"].items():
        metrics = value["metrics"]
        lines.append(
            f"| {name} | {metrics['rollout_id_h1']['mean_delta_target_minus_baseline']:.4g} | "
            f"{metrics['rollout_id_h4']['mean_delta_target_minus_baseline']:.4g} | "
            f"{metrics['rollout_id_h8']['mean_delta_target_minus_baseline']:.4g} | "
            f"{metrics['planning_ood_h4']['mean_delta_target_minus_baseline']:.4g} | "
            f"{metrics['planning_ood_h8']['mean_delta_target_minus_baseline']:.4g} |"
        )

    composition = report["composition_aware_training"]
    lines.extend([
        "",
        "## Composition-aware training contrast",
        "",
        "The direct contrast is composition-aware minus uniform multi-horizon training; negative values favor composition-aware training.",
        "",
        "| metric | mean delta | 95% bootstrap CI | fraction better |",
        "| --- | ---: | --- | ---: |",
    ])
    for metric in ("composition_switch_mse", "composition_constant_mse", "composition_switch_minus_constant", "rollout_id_h8_mse", "rollout_ood_h4_mse", "rollout_ood_h8_mse"):
        value = composition[metric]
        ci = value["bootstrap_ci95"]
        lines.append(f"| {metric} | {value['mean_delta_target_minus_baseline']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['fraction_target_better']:.2f} |")

    external = report["external_transport"]
    lines.extend([
        "",
        "## External normalization audit",
        "",
        "| normalization | supported contrasts | inconclusive contrasts |",
        "| --- | ---: | ---: |",
    ])
    for normalization, counts in external["by_normalization"].items():
        lines.append(f"| {normalization} | {counts['supported']} | {counts['inconclusive']} |")
    horizon = report["external_horizon_transport"]
    lines.extend([
        "",
        "## Cross-horizon latent-displacement transport",
        "",
        "| displacement | dynamics vs covariance | dynamics vs random | all source signs positive |",
        "| --- | ---: | ---: | :---: |",
    ])
    for pair, value in horizon["pairs"].items():
        lines.append(
            f"| {pair} | {value['delta_vs_covariance']:+.6f} | {value['delta_vs_random']:+.6f} | "
            f"{value['positive_vs_covariance_all_sources'] and value['positive_vs_random_all_sources']} |"
        )
    residual_transport = report["external_action_residual_transport"]
    lines.extend([
        "",
        "## Cross-fitted action-residual transport",
        "",
        "The action probe is fitted leave-one-episode-out; the residual target is encoded latent displacement, not physical displacement.",
        "",
        "| normalization | displacement | action R² | dynamics vs covariance | dynamics vs random | all source signs positive |",
        "| --- | --- | ---: | ---: | ---: | :---: |",
    ])
    for key, value in residual_transport["dynamics_rows"].items():
        normalization, _, pair, _ = key.split("/")
        action_r2 = residual_transport["action_probe"][f"{normalization}/{pair}"]["crossfit_action_r2"]
        lines.append(
            f"| {normalization} | {pair} | {action_r2:+.6f} | {value['delta_vs_covariance']:+.6f} | "
            f"{value['delta_vs_random']:+.6f} | {value['positive_vs_covariance_all_sources'] and value['positive_vs_random_all_sources']} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="le-wm-main repository root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_report(args.root, args.output)
    print(f"wrote {args.output} with {len(report['claim_matrix'])} claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
