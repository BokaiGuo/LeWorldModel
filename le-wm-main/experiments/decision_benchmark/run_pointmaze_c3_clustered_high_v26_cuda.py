"""Run V26 with fresh roots and paired-source cluster calibration."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from experiments.decision_benchmark import run_pointmaze_c3_stratified_repair_v24_cuda as core
from experiments.gdd_wm import preflight_pointmaze_c3_clustered_high_v26 as preflight


ROOT = core.ROOT
HIGH_RANGE, DIRECT_RANGE = (35_080, 35_528), (35_528, 35_912)
core.DATASET = ROOT / "data" / "point_maze_v80_c3_clustered_high_20261023"
core.OUTPUT = ROOT / "results" / "pointmaze_c3_clustered_high_v26_20261023_cuda"
core.PREFLIGHT = ROOT / "results" / "pointmaze_c3_clustered_high_v26_static_preflight_20261023.json"
core.ROOT_RANGE = (HIGH_RANGE[0], DIRECT_RANGE[1])
core.ROOT_SEED, core.HELD_ACTION_SEED, core.TRAIN_ACTION_SEED = 202611923, 202611924, 202611925
core.POOL_MODULE = "experiments.decision_benchmark.build_pointmaze_v26_clustered_high_pool"
core.preflight = preflight
_ROOT_COMBINE = core._combine


def _root_id(number: int) -> str:
    return f"pointmaze_v6_episode_{number}"


def _cluster_mapping() -> dict[str, str]:
    result: dict[str, str] = {}
    for root in range(HIGH_RANGE[0], HIGH_RANGE[1], 2):
        result[_root_id(root)] = _root_id(root)
        result[_root_id(root + 1)] = _root_id(root)
    for root in range(DIRECT_RANGE[0], DIRECT_RANGE[1]):
        result[_root_id(root)] = _root_id(root)
    return result


core.CLUSTER_BY_ROOT = _cluster_mapping()


def _combine_clustered(strata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Keep root decisions visible but make the certificate summary clusterwise."""

    combined = _ROOT_COMBINE(strata)
    rootwise = combined["conformal_outcome_coverage"]
    clusters = [row for report in strata.values() for row in report["test"]["cluster_conformal_outcome_coverage"]["clusters"]]
    covered = [bool(row["covered"]) for row in clusters]
    roots = combined["roots"]
    gain_by_cluster: dict[str, list[float]] = {}
    success_by_cluster: dict[str, list[float]] = {}
    for row in roots:
        cluster = core.CLUSTER_BY_ROOT[str(row["root_id"])]
        gain_by_cluster.setdefault(cluster, []).append(float(row["paired_regret_gain"]))
        success_by_cluster.setdefault(cluster, []).append(float(row["paired_success_gain"]))
    gains = np.asarray([np.mean(values) for values in gain_by_cluster.values()], dtype=np.float64)
    successes = np.asarray([np.mean(values) for values in success_by_cluster.values()], dtype=np.float64)
    combined["rootwise_outcome_diagnostic"] = rootwise
    combined["conformal_outcome_coverage"] = {
        "outcome_cluster_count": len(clusters), "covered_cluster_count": int(sum(covered)),
        "empirical_cluster_coverage": float(np.mean(covered)), "clusters": clusters,
        "claim_boundary": "fixed-bank empirical coverage over declared independent source clusters",
    }
    combined["independent_cluster_count"] = len(gains)
    combined["paired_regret_gain_mean"] = float(gains.mean())
    combined["paired_success_gain_mean"] = float(successes.mean())
    combined["paired_regret_gain_bootstrap"] = core.htsc._bootstrap_mean(gains, draws=10_000, seed=202611926)
    combined["paired_success_gain_bootstrap"] = core.htsc._bootstrap_mean(successes, draws=10_000, seed=202611927)
    return combined


core._combine = _combine_clustered


def _require_sealed_outcome_binding() -> None:
    gate_path = core.OUTPUT / "v24_frozen_stratified_gate.json"
    labels = core._read(core.OUTPUT / "v24_outcome_labels.json")
    gate = core._read(gate_path)
    if gate.get("test_outcomes_opened") is not False or labels.get("gate_sha256") != core.v23._sha256(gate_path):
        raise RuntimeError("V26 recovery labels are not bound to the sealed pre-outcome gate")


def _recovery() -> dict[str, Any]:
    _require_sealed_outcome_binding()
    frozen = core._read(core.OUTPUT / "temporal_ood_v6_frozen_selection.json")
    _meta, rows = core.interaction._read_panel(core.OUTPUT / "temporal_ood_v6_context.json", core.interaction.CONTEXT_SCHEMA)
    core.assert_unique_root_observations(rows)
    nonoutcome = [row for row in rows if str(row["split"]) != "outcome"]
    scores = core._costs(nonoutcome, frozen)
    calibration_records = core._records(nonoutcome, core.v23._labels(nonoutcome), scores[2], scores[1])
    outcome = [row for row in rows if str(row["split"]) == "outcome"]
    scores = core._costs(outcome, frozen)
    outcome_records = core._records(outcome, core._read(core.OUTPUT / "v24_outcome_labels.json")["labels"], scores[2], scores[1])
    gate = core._read(core.OUTPUT / "v24_frozen_stratified_gate.json")
    reports = {
        name: core.htsc.evaluate_records(
            [row for row in calibration_records + outcome_records if row["stratum"] == name], alpha=.1, risk_quantile=.9,
            bootstrap_draws=10_000, seed=202611926 if name == "high_motion" else 202611927,
            certificate_kind="regret_budget", regret_budget=core.REGRET_BUDGET,
            cluster_by_root={str(row["root_id"]): core.CLUSTER_BY_ROOT[str(row["root_id"])] for row in calibration_records + outcome_records if row["stratum"] == name},
        ) for name in ("high_motion", "direct_zero_motion")
    }
    for name, value in reports.items():
        if value["calibration"]["contrast"]["bound"] != gate["strata"][name]["contrast"]["bound"] or value["calibration"]["risk_threshold"] != gate["strata"][name]["risk"]["bound"]:
            raise RuntimeError("V26 recovery cannot reproduce the sealed cluster calibration")
    return {"schema": "pointmaze_c3_clustered_high_v26.v1", "recovered_after_post_label_report_assembly_bug": True,
            "frozen_gate": str((core.OUTPUT / "v24_frozen_stratified_gate.json").resolve()),
            "outcome_labels": str((core.OUTPUT / "v24_outcome_labels.json").resolve()),
            "protocol": {"route": preflight.ROUTE_CONTRACT, "high_motion_independent_unit": "opposed_velocity_source_pair", "test_outcome_used_for_selection": False},
            "calibration": {"by_stratum": {name: value["calibration"] for name, value in reports.items()}},
            "test": core._combine(reports), "strata": reports,
            "claim_boundary": "Fresh-root offline fixed-bank cluster-calibrated result only; recovered without refit or outcome regeneration."}


def main() -> int:
    try:
        status = core.main()
    except FileExistsError:
        report, status = _recovery(), 0
    path = core.OUTPUT / "v24_c3_report.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
    report["schema"] = "pointmaze_c3_clustered_high_v26.v1"
    report["successor_protocol"] = "fresh_roots_after_v25_paired_cluster_calibration_invalidation"
    report["high_motion_independent_unit"] = "opposed_velocity_source_pair"
    report.setdefault("protocol", {})["calibration"] = (
        "separate fixed-bank envelopes: one maximum contrast/risk score per opposed-velocity "
        "source pair in high motion, one score per independent direct root"
    )
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
