"""Aggregate the predeclared independent TwoRoom action-effect V4 panels.

The primary question is deliberately stricter than an adapter-versus-frozen
comparison: does the full latent-plus-predicted-dynamics effect adapter beat
the action-only and no-dynamics effect controls on fresh closed-loop roots?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "tworoom_action_effect_v4_panel_audit.v1"
REPORT_SCHEMA = "tworoom_action_effect_repairability.v1"
EXPECTED_PANELS = 3
EXPECTED_ROOTS_PER_PANEL = 160
EXPECTED_OUTCOME_ROOTS_PER_PANEL = 32


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{path}: unexpected report schema")
    protocol = value.get("protocol", {})
    if protocol.get("root_counts") != {"train": 96, "calibration": 32, "outcome": 32}:
        raise ValueError(f"{path}: requires the fixed 160-root 96/32/32 split")
    if protocol.get("runtime_state_access") is not False or protocol.get("training_outcome_roots_read") is not False:
        raise ValueError(f"{path}: privileged deployment or outcome-training access")
    return value


def _paired(values: dict[str, Any], left: str, right: str, key: str) -> tuple[list[int], np.ndarray]:
    left_arm = values["closed_loop"][left]
    right_arm = values["closed_loop"][right]
    def by_root(arm: dict[str, Any]) -> dict[int, float]:
        roots = [int(root) for root in arm["roots"]]
        measurements = np.asarray(arm[key], dtype=np.float64)
        if measurements.shape != (len(roots),) or len(set(roots)) != len(roots):
            raise ValueError(f"{left}/{right}: malformed closed-loop {key}")
        return dict(zip(roots, measurements.tolist()))
    lhs_by_root, rhs_by_root = by_root(left_arm), by_root(right_arm)
    if set(lhs_by_root) != set(rhs_by_root):
        raise ValueError(f"{left}/{right}: closed-loop roots differ")
    roots = sorted(lhs_by_root)
    lhs = np.asarray([lhs_by_root[root] for root in roots], dtype=np.float64)
    rhs = np.asarray([rhs_by_root[root] for root in roots], dtype=np.float64)
    if lhs.shape != (len(roots),) or rhs.shape != (len(roots),):
        raise ValueError(f"{left}/{right}: malformed closed-loop {key}")
    return roots, lhs - rhs


def _bootstrap(values: np.ndarray, *, seed: int, repeats: int = 10_000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "repeats": int(repeats),
    }


def _panel_inference(panel_values: list[np.ndarray], *, seed: int, repeats: int = 20_000) -> dict[str, Any]:
    """Separate within-panel root uncertainty from between-panel replication.

    Panels are the independent replication units. Roots are paired units nested
    within each panel. With only three panels, the panel-level interval and
    sign-flip test are intentionally reported as low-resolution evidence.
    """

    if len(panel_values) != EXPECTED_PANELS or any(values.ndim != 1 or not len(values) for values in panel_values):
        raise ValueError("panel inference requires three non-empty one-dimensional panel effects")
    panel_means = np.asarray([values.mean() for values in panel_values], dtype=np.float64)
    panel_count = len(panel_means)
    mean = float(panel_means.mean())
    if panel_count > 1:
        standard_error = float(panel_means.std(ddof=1) / np.sqrt(panel_count))
        # 95% two-sided Student-t critical value for df=2.
        t_critical = 4.302652729911275 if panel_count == 3 else float("nan")
        t_interval = [mean - t_critical * standard_error, mean + t_critical * standard_error]
    else:
        t_interval = [float("nan"), float("nan")]

    observed = abs(mean)
    sign_flip_means = []
    for mask in range(1 << panel_count):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(panel_count)])
        sign_flip_means.append(float(np.mean(panel_means * signs)))
    sign_flip_p = float(np.mean(np.abs(sign_flip_means) >= observed - 1.0e-12))

    # Resample independent panels first, then roots within each sampled panel.
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for draw in range(repeats):
        sampled_panels = rng.integers(0, panel_count, size=panel_count)
        sampled_means = []
        for panel_index in sampled_panels:
            roots = panel_values[int(panel_index)]
            sampled_means.append(float(roots[rng.integers(0, len(roots), size=len(roots))].mean()))
        draws[draw] = float(np.mean(sampled_means))

    pooled = np.concatenate(panel_values)
    return {
        "independent_panel_count": panel_count,
        "root_count_nested_within_panels": int(len(pooled)),
        "equal_panel_weighted_estimate": mean,
        "panel_means": [float(value) for value in panel_means],
        "panel_positive_count": int(np.count_nonzero(panel_means > 0.0)),
        "panel_nonnegative_count": int(np.count_nonzero(panel_means >= 0.0)),
        "panel_level_t_interval_95": [float(value) for value in t_interval],
        "panel_level_exact_two_sided_sign_flip_p": sign_flip_p,
        "nested_panel_root_bootstrap_95": {
            "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975)),
            "repeats": int(repeats),
        },
        "within_panel_root_bootstrap_95": _bootstrap(pooled, seed=seed + 1),
        "interpretation": (
            "Root bootstrap is conditional on these panels. Panel-level inference has only three independent units; "
            "exact sign-flip p-values are coarse and do not justify broad environment generalization."
        ),
    }


def _ranking_delta(values: dict[str, Any], control: str) -> tuple[list[int], np.ndarray]:
    root_ids = sorted(int(root) for root in values["protocol"]["root_ids"]["outcome"])
    adapter = values["ranking"]["effect_adapter"]["outcome"]["root_regret"]
    baseline = values["ranking"][control]["outcome"]["root_regret"]
    expected = {str(root) for root in root_ids}
    if set(adapter) != expected or set(baseline) != expected:
        raise ValueError(f"effect_adapter/{control}: ranking regret roots do not match the declared outcome panel")
    # Positive means the full adapter reduced true selected-candidate regret.
    return root_ids, np.asarray([
        float(baseline[str(root)]) - float(adapter[str(root)]) for root in root_ids
    ], dtype=np.float64)


def audit(report_paths: list[Path]) -> dict[str, Any]:
    if len(report_paths) != EXPECTED_PANELS:
        raise ValueError(f"requires exactly {EXPECTED_PANELS} fixed V4 panels")
    reports = [_load(path) for path in report_paths]
    all_roots: set[int] = set()
    root_counts: dict[str, set[int]] = {role: set() for role in ("train", "calibration", "outcome")}
    distances: dict[str, list[np.ndarray]] = {"action_only": [], "no_dynamics": []}
    successes: dict[str, list[np.ndarray]] = {"action_only": [], "no_dynamics": []}
    regrets: dict[str, list[np.ndarray]] = {"action_only": [], "no_dynamics": []}
    per_panel: list[dict[str, Any]] = []
    for path, report in zip(report_paths, reports):
        protocol_roots = report["protocol"]["root_ids"]
        panel_roots: set[int] = set()
        for role in root_counts:
            role_roots = {int(root) for root in protocol_roots[role]}
            if len(role_roots) != int(report["protocol"]["root_counts"][role]):
                raise ValueError(f"{path}: duplicate or malformed {role} roots")
            overlap = sorted(role_roots & panel_roots)
            if overlap:
                raise ValueError(f"{path}: root roles overlap within the panel: {overlap[:5]}")
            panel_roots |= role_roots
            root_counts[role] |= role_roots
        overlap = sorted(panel_roots & all_roots)
        if overlap:
            raise ValueError(f"{path}: roots overlap an earlier V4 panel across roles: {overlap[:5]}")
        all_roots |= panel_roots
        root_ids = {int(root) for root in protocol_roots["outcome"]}
        if len(root_ids) != EXPECTED_OUTCOME_ROOTS_PER_PANEL:
            raise ValueError(f"{path}: malformed outcome root panel")
        panel_row: dict[str, Any] = {"report": str(path.resolve()), "outcome_roots": sorted(root_ids)}
        for control in distances:
            roots, distance_delta = _paired(report, control, "effect_adapter", "final_distance")
            _, success_delta = _paired(report, "effect_adapter", control, "success")
            regret_roots, regret_delta = _ranking_delta(report, control)
            if roots != regret_roots:
                raise ValueError(f"{path}: ranking and closed-loop root order differ")
            # Positive distance delta means the adapter is closer to the goal.
            distances[control].append(distance_delta)
            successes[control].append(success_delta)
            regrets[control].append(regret_delta)
            panel_row[control] = {
                "adapter_distance_gain": float(distance_delta.mean()),
                "adapter_success_gain": float(success_delta.mean()),
                "adapter_ranking_regret_gain": float(regret_delta.mean()),
                "root_count": len(roots),
            }
        per_panel.append(panel_row)
    comparisons = {}
    for index, control in enumerate(distances):
        comparisons[control] = {
            "adapter_closed_loop_distance_gain": _panel_inference(distances[control], seed=202610231 + index),
            "adapter_closed_loop_success_gain": _panel_inference(successes[control], seed=202610241 + index),
            "adapter_ranking_regret_gain": _panel_inference(regrets[control], seed=202610251 + index),
        }
    return {
        "schema": SCHEMA,
        "status": "completed_root_disjoint_v4_panel_audit",
        "panel_count": EXPECTED_PANELS,
        "root_counts_across_panels": {role: len(values) for role, values in root_counts.items()},
        "independent_panel_count": EXPECTED_PANELS,
        "per_panel": per_panel,
        "comparisons": comparisons,
        "primary_estimand": (
            "paired final-distance gain = control terminal distance minus effect-adapter terminal distance; "
            "positive values favour the full adapter"
        ),
        "secondary_estimand": (
            "paired ranking-regret gain = control true selected-candidate regret minus effect-adapter regret; "
            "positive values favour the full adapter"
        ),
        "inference_contract": (
            "Roots are paired units nested within each panel; panels are the independent replication units. "
            "Both nested root-bootstrap intervals and low-resolution panel-level inference are reported."
        ),
        "claim_boundary": (
            "Root-disjoint TwoRoom action-effect replication only. It does not establish prospective repairability, "
            "GDD teacher value, calibrated deployment, or cross-environment transfer."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V4 panel audit: {args.output}")
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "outcome_root_count": result["root_counts_across_panels"]["outcome"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
