"""Run the fresh V25 successor after V24's duplicate-root invalidation."""

from __future__ import annotations

import json

from experiments.decision_benchmark import run_pointmaze_c3_stratified_repair_v24_cuda as core
from experiments.gdd_wm import preflight_pointmaze_c3_independent_direct_v25 as preflight


ROOT = core.ROOT
core.DATASET = ROOT / "data" / "point_maze_v78_c3_independent_direct_20261023"
core.OUTPUT = ROOT / "results" / "pointmaze_c3_independent_direct_v25_20261023_cuda"
core.PREFLIGHT = ROOT / "results" / "pointmaze_c3_independent_direct_v25_static_preflight_20261023.json"
core.ROOT_RANGE = (34_248, 35_080)
core.ROOT_SEED, core.HELD_ACTION_SEED, core.TRAIN_ACTION_SEED = 202611904, 202611905, 202611906
core.POOL_MODULE = "experiments.decision_benchmark.build_pointmaze_v25_c3_independent_direct_pool"
core.preflight = preflight


def _require_sealed_outcome_binding() -> None:
    """Verify recovery can only read labels opened under this exact gate."""

    gate_path = core.OUTPUT / "v24_frozen_stratified_gate.json"
    labels_path = core.OUTPUT / "v24_outcome_labels.json"
    gate = core._read(gate_path)
    labels = core._read(labels_path)
    if gate.get("test_outcomes_opened") is not False:
        raise RuntimeError("V25 recovery refuses a gate written after outcomes opened")
    if labels.get("schema") != "pointmaze_c3_stratified_repair_outcome_labels.v1":
        raise RuntimeError("V25 recovery outcome-label schema mismatch")
    if labels.get("gate_sha256") != core.v23._sha256(gate_path):
        raise RuntimeError("V25 recovery labels are not bound to the sealed gate")


def main() -> int:
    try:
        status = core.main()
    except FileExistsError:
        # V25 may have opened its immutable labels before a post-label report
        # assembly failure. Recover only from the frozen model, sealed gate,
        # and existing labels; never refit or regenerate outcomes.
        _require_sealed_outcome_binding()
        frozen = core._read(core.OUTPUT / "temporal_ood_v6_frozen_selection.json")
        _meta, rows = core.interaction._read_panel(core.OUTPUT / "temporal_ood_v6_context.json", core.interaction.CONTEXT_SCHEMA)
        core.assert_unique_root_observations(rows)
        nonoutcome = [row for row in rows if str(row["split"]) != "outcome"]
        calibration_records = core._records(nonoutcome, core.v23._labels(nonoutcome), core._costs(nonoutcome, frozen)[2], core._costs(nonoutcome, frozen)[1])
        outcome = [row for row in rows if str(row["split"]) == "outcome"]
        labels = core._read(core.OUTPUT / "v24_outcome_labels.json")["labels"]
        scores = core._costs(outcome, frozen)
        all_records = calibration_records + core._records(outcome, labels, scores[2], scores[1])
        gate = core._read(core.OUTPUT / "v24_frozen_stratified_gate.json")
        reports = {name: core.htsc.evaluate_records([row for row in all_records if row["stratum"] == name], alpha=.1, risk_quantile=.9, bootstrap_draws=10_000, seed=202611809 if name == "high_motion" else 202611810, certificate_kind="regret_budget", regret_budget=core.REGRET_BUDGET) for name in ("high_motion", "direct_zero_motion")}
        for name, report in reports.items():
            if report["calibration"]["contrast"]["bound"] != gate["strata"][name]["contrast"]["bound"] or report["calibration"]["risk_threshold"] != gate["strata"][name]["risk"]["bound"]:
                raise RuntimeError("V25 recovery cannot reproduce sealed calibration")
        report = {"schema": "pointmaze_c3_independent_direct_v25.v1", "recovered_after_post_label_report_assembly_bug": True, "frozen_gate": str((core.OUTPUT / "v24_frozen_stratified_gate.json").resolve()), "outcome_labels": str((core.OUTPUT / "v24_outcome_labels.json").resolve()), "protocol": {"route": preflight.ROUTE_CONTRACT, "test_outcome_used_for_selection": False}, "calibration": {"by_stratum": {name: value["calibration"] for name, value in reports.items()}}, "test": core._combine(reports), "strata": reports, "claim_boundary": "Fresh-root offline fixed-bank result only; recovered without refit or outcome regeneration."}
        status = 0
    path = core.OUTPUT / "v24_c3_report.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        report["schema"] = "pointmaze_c3_independent_direct_v25.v1"
    report["successor_protocol"] = "fresh_roots_after_v24_duplicate_zero_motion_root_invalidation"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
