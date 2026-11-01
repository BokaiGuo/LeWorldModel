"""Saved-only P2--P5 baseline resource inventory and fail-closed preflight.

This module reads pinned reports only.  It never imports torch, initializes
CUDA, dispatches physics, or promotes a saved calibration result into a fair
comparison or stage admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


NAME = "baseline_pstages_saved_ledger_20261101_parallel"
ROOT = Path(__file__).resolve().parents[2]

STATUSES = {"measured", "declared", "derived", "absent_by_contract", "unknown"}
CATEGORIES = {
    "encoding": ("rgb_encode", "project"),
    "prediction": ("action_encode", "predict", "predictor_samples", "context_tokens"),
    "differentiation": ("backward", "vjp", "jacobian", "svd", "solve"),
    "search": ("candidate_evaluate", "replay", "selection"),
    "physical": ("physical_reset", "physical_step", "render"),
    "timing": ("planning_wall", "end_to_end_wall", "cuda_time"),
    "memory": ("peak_cuda_allocated", "peak_cuda_reserved"),
}
OPERATIONS = tuple(operation for operations in CATEGORIES.values() for operation in operations)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pairs_unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024**2:
        raise ValueError(f"bounded regular saved report required: {path}")
    raw = path.read_bytes()
    actual = _sha256(raw)
    if actual != expected_sha256:
        raise ValueError(f"saved report SHA-256 mismatch: {path}")

    def bad_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    data = json.loads(raw, object_pairs_hook=_pairs_unique, parse_constant=bad_constant)
    if not isinstance(data, dict):
        raise ValueError(f"saved report must be a JSON object: {path}")
    return data


def _entry(
    status: str,
    value: int | float | None = None,
    unit: str | None = None,
    evidence: str = "",
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unknown resource status: {status}")
    if status == "unknown":
        if value is not None:
            raise ValueError("unknown resource cannot have a numeric value")
    else:
        if type(value) not in (int, float) or value < 0:
            raise ValueError("known resource must have a nonnegative numeric value")
    return {"status": status, "value": value, "unit": unit, "evidence": evidence}


def _blank() -> dict[str, dict[str, Any]]:
    return {operation: _entry("unknown", evidence="not covered by the pinned report")
            for operation in OPERATIONS}


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    values = [row.get(key) for row in rows]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"missing or invalid nonnegative integer row field: {key}")
    return sum(values)


def _require_schema(data: dict[str, Any], schema: str) -> None:
    if data.get("schema") != schema:
        raise ValueError(f"unexpected saved schema: {data.get('schema')!r}")


def _extract_zero(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require_schema(data, "codex_p2_h3_zero_control_20261030_v1.report")
    rows = data.get("rows")
    receipt = data.get("physical_call_receipt")
    if (data.get("new_neural_calls") != 0 or not isinstance(rows, list) or len(rows) != 4
            or not isinstance(receipt, dict)):
        raise ValueError("unexpected P2 zero-control saved contract")
    ledger = _blank()
    for operation in ("rgb_encode", "project", "action_encode", "predict",
                      "predictor_samples", "context_tokens", "backward", "vjp",
                      "jacobian", "svd", "solve", "candidate_evaluate", "replay",
                      "selection"):
        ledger[operation] = _entry(
            "absent_by_contract", 0, "call",
            "fixed-zero report records new_neural_calls=0 and no planner search",
        )
    ledger["physical_reset"] = _entry(
        "measured", receipt["total_env_reset_attempts"], "public reset attempt",
        "physical_call_receipt.total_env_reset_attempts",
    )
    ledger["physical_step"] = _entry(
        "measured", receipt["total_public_env_step_attempts"], "public step attempt",
        "physical_call_receipt includes warmup and selected calls",
    )
    ledger["render"] = _entry(
        "derived", sum(1 for row in rows if row.get("terminal_rgb_sha256")), "terminal RGB",
        "one saved terminal RGB digest per completed row",
    )
    return ledger


def _extract_sra(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require_schema(data, "p2_no_composition_sra_20261027_v2.report")
    ledger = _blank()
    direct = {
        "action_encode": ("actual_action_encoder_calls", "call"),
        "predict": ("actual_predict_calls", "call"),
        "context_tokens": ("actual_processed_context_token_samples", "token-sample"),
        "selection": ("actual_selection_evaluations", "evaluation"),
        "jacobian": ("completed_jacobian_calls", "call"),
        "vjp": ("completed_scalar_output_vjp_directions", "direction"),
        "svd": ("completed_svd_calls", "call"),
        "peak_cuda_allocated": ("peak_cuda_allocated_bytes", "byte"),
    }
    for operation, (key, unit) in direct.items():
        value = data.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"missing SRA resource field: {key}")
        ledger[operation] = _entry("measured", value, unit, key)
    ledger["physical_step"] = _entry(
        "absent_by_contract", 0, "public step", "new_physical_steps=0 calibration report",
    )
    return ledger


def _extract_host_top(data: dict[str, Any], schema: str) -> dict[str, dict[str, Any]]:
    _require_schema(data, schema)
    elapsed = data.get("elapsed_seconds")
    peak = data.get("peak")
    if type(elapsed) not in (int, float) or elapsed < 0 or not isinstance(peak, dict):
        raise ValueError("invalid host top-level timing or peak memory")
    ledger = _blank()
    ledger["end_to_end_wall"] = _entry(
        "measured", elapsed, "second", "top-level elapsed_seconds including host runner",
    )
    for operation, key in (
        ("peak_cuda_allocated", "allocated"),
        ("peak_cuda_reserved", "reserved"),
    ):
        value = peak.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid host peak field: {key}")
        ledger[operation] = _entry("measured", value, "byte", f"peak.{key}")
    return ledger


def _extract_grasp(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require_schema(data, "p2_grasp_h1_cuda_runner_20261027_v1.report")
    rows = data.get("rows")
    contract = data.get("contract")
    if (not isinstance(rows, list) or len(rows) != 8 or not isinstance(contract, dict)
            or contract.get("official_equivalence") is not False):
        raise ValueError("unexpected local GRASP proxy contract")
    ledger = _blank()
    action_calls = sum(row["module_work"]["action_encoder"]["completed_calls"] for row in rows)
    predictor_calls = sum(row["module_work"]["predictor"]["completed_calls"] for row in rows)
    context_tokens = sum(row["module_work"]["predictor"]["completed_context_tokens"] for row in rows)
    backward = sum(row["receipt"]["backward_completed"] for row in rows)
    ledger["action_encode"] = _entry("measured", action_calls, "call", "sum rows.module_work")
    ledger["predict"] = _entry("measured", predictor_calls, "call", "sum predictor module calls")
    ledger["predictor_samples"] = _entry(
        "measured", predictor_calls, "sample", "all recorded predictor calls have one sample",
    )
    ledger["context_tokens"] = _entry(
        "measured", context_tokens, "token-sample", "sum completed predictor context tokens",
    )
    ledger["backward"] = _entry("measured", backward, "scalar backward call", "sum receipts")
    ledger["physical_step"] = _entry(
        "absent_by_contract", 0, "public step", "new_physical_steps=0",
    )
    return ledger


def _extract_p3_schedule(
    data: dict[str, Any], *, dynamic: bool,
) -> dict[str, dict[str, Any]]:
    expected = ("p3_action_jacobian_budget_schedule_20261027_v3.report"
                if dynamic else None)
    if expected is not None:
        _require_schema(data, expected)
    elif data.get("status") != "calibration_recorded":
        raise ValueError("unexpected fixed P3 schedule report")
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("expected eight P3 schedule rows")
    ledger = _blank()
    ledger["action_encode"] = _entry("measured", _sum(rows, "action_encoder_calls"), "call",
                                     "sum per-row action encoder calls")
    predict_calls = sum(
        phase["predict_calls"]
        for row in rows
        for phase in row["prediction_receipt"]["phases"]
    )
    ledger["predict"] = _entry(
        "measured", predict_calls, "call", "sum per-row prediction phase predict_calls",
    )
    ledger["predictor_samples"] = _entry(
        "measured", data["known_predict_samples"], "sample", "known_predict_samples",
    )
    ledger["context_tokens"] = _entry(
        "measured", _sum(rows, "processed_action_context_tokens"), "token-sample",
        "sum per-row processed action context tokens",
    )
    ledger["candidate_evaluate"] = _entry(
        "derived", sum(len(row["search"]["events"]) for row in rows), "search event",
        "count of saved search events",
    )
    for operation, key in (
        ("backward", "runtime_action_backward_calls"),
        ("vjp", "runtime_action_vjp_rows"),
    ):
        value = _sum(rows, key)
        status = "absent_by_contract" if value == 0 else "measured"
        ledger[operation] = _entry(status, value, "call" if operation != "vjp" else "row",
                                   f"sum per-row {key}")
    if all(row.get("dense_jacobian_materializations") is None for row in rows):
        ledger["jacobian"] = _entry(
            "unknown", evidence="dense_jacobian_materializations is null in every saved row",
        )
    else:
        value = _sum(rows, "dense_jacobian_materializations")
        ledger["jacobian"] = _entry(
            "absent_by_contract" if value == 0 else "measured",
            value,
            "materialization",
            "sum per-row dense_jacobian_materializations",
        )
    ledger["physical_step"] = _entry(
        "absent_by_contract", _sum(rows, "new_physical_steps"), "public step",
        "saved schedule calibration did not dispatch physics",
    )
    ledger["end_to_end_wall"] = _entry(
        "measured", data["elapsed_seconds_including_setup"], "second",
        "elapsed_seconds_including_setup; not per-arm planning wall",
    )
    if dynamic:
        inherited = data.get("inherited_cost")
        if not isinstance(inherited, dict) or inherited.get("status") != "inherited_unknown_cost":
            raise ValueError("dynamic schedule must retain inherited unknown cost")
    return ledger


def _extract_p3_cem(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require_schema(data, "p3_reacher_feedback_cap_v2_20261101_codex.report.v1")
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("expected fixed and dynamic Reacher CEM rows")
    ledger = _blank()
    for operation, key, unit in (
        ("rgb_encode", "encoder_calls", "call"),
        ("action_encode", "action_encoder_calls", "call"),
        ("predict", "predict_calls", "call"),
        ("predictor_samples", "predict_samples", "sample"),
    ):
        value = sum(row["budget"]["attempted"][key] for row in rows)
        ledger[operation] = _entry("measured", value, unit, f"sum rows.budget.attempted.{key}")
    ledger["physical_step"] = _entry(
        "measured", sum(row["physical_step_attempts"] for row in rows), "public step attempt",
        "sum per-arm physical_step_attempts",
    )
    for operation in ("backward", "vjp", "jacobian", "svd", "solve"):
        ledger[operation] = _entry(
            "absent_by_contract", 0, "call", "CEM saved planning records zero derivative work",
        )
    return ledger


def _extract_p4(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require_schema(data, "p4_feedback_live_saved_audit_20261027_v1.report")
    counts = data.get("API_counts")
    physics = data.get("physics")
    if not isinstance(counts, dict) or not isinstance(physics, dict):
        raise ValueError("P4 report lacks saved API or physics counts")
    ledger = _blank()
    mapping = {
        "rgb_encode": "encoder",
        "project": "projector",
        "action_encode": "action_encoder",
        "predict": "predict",
        "backward": "backward_api",
        "jacobian": "jacobian_api",
    }
    for operation, key in mapping.items():
        value = counts[key]["attempted"]
        status = "absent_by_contract" if value == 0 else "measured"
        ledger[operation] = _entry(status, value, "API call", f"API_counts.{key}.attempted")
    ledger["vjp"] = _entry("measured", counts["grad_api"]["attempted"], "grad API call",
                           "API_counts.grad_api.attempted; not scalar-output directions")
    ledger["physical_reset"] = _entry(
        "measured", physics["branch_resets"] + physics["root_resets"], "public reset",
        "branch_resets + root_resets",
    )
    ledger["physical_step"] = _entry(
        "measured", physics["selected_steps"] + physics["warmup_steps"], "public step",
        "selected_steps + warmup_steps",
    )
    ledger["end_to_end_wall"] = _entry(
        "measured", data["recorded_wall_seconds"], "second", "recorded_wall_seconds",
    )
    peaks = data["recorded_peak_cuda_bytes"]
    ledger["peak_cuda_allocated"] = _entry(
        "measured", peaks[0], "byte", "recorded_peak_cuda_bytes[0]",
    )
    ledger["peak_cuda_reserved"] = _entry(
        "measured", peaks[1], "byte", "recorded_peak_cuda_bytes[1]",
    )
    return ledger


def _extract_p5(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require_schema(data, "p5_architecture_readiness_20261027_v2")
    if data.get("cuda_executed") is not False or data.get("P5_admission") is not False:
        raise ValueError("P5 readiness report unexpectedly claims execution or admission")
    return _blank()


Extractor = Callable[[dict[str, Any]], dict[str, dict[str, Any]]]

ASSETS: dict[str, dict[str, Any]] = {
    "p2_zero_h3": {
        "stage": "P2", "identity": "equivalence_control",
        "cohort": "p2_zero_h3_four_known_roots",
        "path": "results/codex_p2_h3_zero_control_20261030_v1_RUN01/report.json",
        "sha256": "41f8dd24cdbb6eeb59816dda3711488f503620f7d175c60098c8f2395f87fc7d",
        "entrypoint": "experiments.spectral_reachability.codex_p2_h3_zero_control_20261030_v1",
        "evidence_level": "saved_physical_control_known_roots",
        "extractor": _extract_zero,
    },
    "p2_no_composition_sra": {
        "stage": "P2", "identity": "local_algorithm",
        "cohort": "p2_no_composition_sra_four_calibration_roots",
        "path": "results/p2_no_composition_sra_20261027_v2_prepared01/report.json",
        "sha256": "522afa5a68b23a6ff829c5c6d820d858953dccc0f778548e747ba594ff73937a",
        "entrypoint": "experiments.spectral_reachability.p2_no_composition_sra_20261027_v2",
        "evidence_level": "saved_host_cuda_calibration",
        "extractor": _extract_sra,
    },
    "p2_gn_h15": {
        "stage": "P2", "identity": "local_algorithm",
        "cohort": "p2_h15_matched_host_two_known_roots",
        "path": "results/p2_h15_gn_matched_host_20261029_v1_cuda_MAIN01/report.json",
        "sha256": "2d41a16c0ad568c79c9b627fffe6e9cd608b0a38a2c26503542b3b935b7a1ceb",
        "entrypoint": "experiments.spectral_reachability.p2_h15_gn_matched_host_20261029_v1",
        "evidence_level": "saved_host_cuda_top_level",
        "extractor": lambda data: _extract_host_top(
            data, "p2_h15_gn_matched_host_20261029_v1.report"),
    },
    "p2_cem_h15": {
        "stage": "P2", "identity": "local_algorithm",
        "cohort": "p2_h15_matched_host_two_known_roots",
        "path": "results/p2_h15_cem_matched_host_20261029_v1_cuda_MAIN01/report.json",
        "sha256": "3b675cb7bb99591152704a8fe1ef1e579524e4f3c6c839f37d7a9c78934ab799",
        "entrypoint": "experiments.spectral_reachability.p2_h15_cem_matched_host_20261029_v1",
        "evidence_level": "saved_host_cuda_top_level",
        "extractor": lambda data: _extract_host_top(
            data, "p2_h15_cem_matched_host_20261029_v1.report"),
    },
    "p2_grasp_h1_local": {
        "stage": "P2", "identity": "local_proxy",
        "cohort": "p2_grasp_h1_four_known_roots",
        "path": "results/p2_grasp_h1_cuda_runner_20261027_v1_cuda_main_v1/report.json",
        "sha256": "a6d208d2e6dbbcd8650f2a0948b5ce628e464918693593a0fc0474334c0a2133",
        "entrypoint": "experiments.spectral_reachability.p2_grasp_h1_cuda_runner_20261027_v1",
        "evidence_level": "saved_host_cuda_local_proxy_not_official",
        "extractor": _extract_grasp,
    },
    "p3_fixed_schedule": {
        "stage": "P3", "identity": "local_algorithm",
        "cohort": "p3_pusht_fixed_h3_eight_known_rows",
        "path": "results/p3_fixed_horizon_budget_schedule_20261027_v2_cuda_main_v1/report.json",
        "sha256": "1d7f9da67e545af1cfe7a081b78fb5e0bbde918ce368f01914433a7e892046c5",
        "entrypoint": "experiments.spectral_reachability.p3_fixed_horizon_budget_schedule_20261027_v2",
        "evidence_level": "saved_host_cuda_calibration",
        "extractor": lambda data: _extract_p3_schedule(data, dynamic=False),
    },
    "p3_action_jacobian_schedule": {
        "stage": "P3", "identity": "local_algorithm",
        "cohort": "p3_pusht_fixed_h3_eight_known_rows",
        "path": "results/p3_action_jacobian_budget_schedule_20261027_v3_cuda_main_v1/report.json",
        "sha256": "c13dab20228fd590c3f9375e29b6be905a8db083942a547c862d97c90cf53921",
        "entrypoint": "experiments.spectral_reachability.p3_action_jacobian_budget_schedule_20261027_v3",
        "evidence_level": "saved_host_cuda_calibration_inherited_cost_unknown",
        "extractor": lambda data: _extract_p3_schedule(data, dynamic=True),
    },
    "p3_reacher_cem_feedback": {
        "stage": "P3", "identity": "local_algorithm",
        "cohort": "p3_reacher_feedback_fixed_dynamic_one_known_root",
        "path": "results/p3_reacher_feedback_cap_v2_20261101_codex_RUN_MAIN02/report.json",
        "sha256": "37b4fb29c07459bd91c4e478b322de305a7c404e2d3775ea06ee6a311c1bf45f",
        "entrypoint": "experiments.spectral_reachability.p3_reacher_feedback_cap_v2_20261101_codex",
        "evidence_level": "saved_host_cuda_and_physical_known_root",
        "extractor": _extract_p3_cem,
    },
    "p4_feedback_pair": {
        "stage": "P4", "identity": "local_algorithm",
        "cohort": "p4_feedback_pair_single_known_root",
        "path": "results/p4_feedback_live_saved_audit_20261027_v1_main01/report.json",
        "sha256": "6bdd30c1da733417147504050beed1b6a60ddd513abee45a083fd0955d741953",
        "entrypoint": "experiments.spectral_reachability.p4_feedback_live_saved_audit_20261027_v1",
        "evidence_level": "saved_audit_model_physical_rgb_known_root",
        "extractor": _extract_p4,
    },
    "p5_architecture_readiness": {
        "stage": "P5", "identity": "readiness_only",
        "cohort": "p5_architecture_static_readiness",
        "path": "results/p5_architecture_readiness_20261027_v2/report.json",
        "sha256": "4b9d8e612222f11b80ac42123a3af3562add6a7942dffba1c311e2b96fdfcdaf",
        "entrypoint": "experiments.spectral_reachability.p5_architecture_readiness_20261027_v2",
        "evidence_level": "static_readiness_no_neural_execution",
        "extractor": _extract_p5,
    },
}


def _category_coverage(ledger: dict[str, dict[str, Any]]) -> dict[str, str]:
    coverage: dict[str, str] = {}
    for category, operations in CATEGORIES.items():
        statuses = {ledger[operation]["status"] for operation in operations}
        if "unknown" in statuses:
            coverage[category] = "incomplete"
        elif statuses <= {"absent_by_contract"}:
            coverage[category] = "absent_by_contract"
        else:
            coverage[category] = "complete"
    return coverage


def inventory() -> dict[str, Any]:
    rows = []
    for asset_id, spec in ASSETS.items():
        path = ROOT / spec["path"]
        data = _read_json(path, spec["sha256"])
        ledger = spec["extractor"](data)
        if set(ledger) != set(OPERATIONS):
            raise ValueError(f"incomplete unified operation roster: {asset_id}")
        rows.append({
            "asset_id": asset_id,
            "stage": spec["stage"],
            "method_identity": spec["identity"],
            "protocol_cohort": spec["cohort"],
            "report_path": str(path),
            "report_sha256": spec["sha256"],
            "entrypoint": spec["entrypoint"],
            "entrypoint_present": (ROOT / (spec["entrypoint"].replace(".", "/") + ".py")).is_file(),
            "saved_evidence_level": spec["evidence_level"],
            "resources": ledger,
            "category_coverage": _category_coverage(ledger),
            "paper_reported_value_included": False,
            "local_performance_ranking": False,
            "stage_admission": False,
        })
    return {
        "schema": NAME + ".inventory.v1",
        "rows": rows,
        "saved_assets_verified": True,
        "paper_reported_values_included": False,
        "fair_comparison_verified": False,
        "P2_admission": False,
        "P3_admission": False,
        "P4_admission": False,
        "P5_admission": False,
        "warning": "resource operations are separate axes and must not be summed into FLOPs",
    }


def preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    exact = {"schema", "asset_ids", "required_categories", "required_operations", "comparison"}
    if not isinstance(manifest, dict) or set(manifest) != exact:
        raise ValueError("exact preflight manifest fields required")
    if manifest["schema"] != NAME + ".preflight.v1":
        raise ValueError("preflight manifest schema")
    asset_ids = manifest["asset_ids"]
    categories = manifest["required_categories"]
    operations = manifest["required_operations"]
    comparison = manifest["comparison"]
    if (not isinstance(asset_ids, list) or not asset_ids or len(set(asset_ids)) != len(asset_ids)
            or any(asset_id not in ASSETS for asset_id in asset_ids)):
        raise ValueError("unique known local asset_ids required")
    if (not isinstance(categories, list) or len(set(categories)) != len(categories)
            or any(category not in CATEGORIES for category in categories)):
        raise ValueError("unknown resource category")
    if (not isinstance(operations, list) or len(set(operations)) != len(operations)
            or any(operation not in OPERATIONS for operation in operations)):
        raise ValueError("unknown operation")
    if (not isinstance(comparison, dict)
            or set(comparison) != {"requested", "ranking", "protocol_cohort"}
            or type(comparison["requested"]) is not bool
            or type(comparison["ranking"]) is not bool
            or comparison["protocol_cohort"] is not None
            and type(comparison["protocol_cohort"]) is not str):
        raise ValueError("exact comparison request required")
    if comparison["ranking"] and not comparison["requested"]:
        raise ValueError("ranking requires comparison.requested=true")

    rows_by_id = {row["asset_id"]: row for row in inventory()["rows"]}
    selected = [rows_by_id[asset_id] for asset_id in asset_ids]
    cohorts = {row["protocol_cohort"] for row in selected}
    blockers: list[str] = []
    if comparison["requested"]:
        requested_cohort = comparison["protocol_cohort"]
        if len(cohorts) != 1 or requested_cohort not in cohorts:
            raise ValueError("protocol cohort mismatch; cross-protocol comparison rejected")
        if comparison["ranking"]:
            blockers.append("saved resource ledger does not authorize a performance ranking")

    required = set(operations)
    for category in categories:
        required.update(CATEGORIES[category])
    missing: dict[str, list[str]] = {}
    for row in selected:
        unknown = sorted(operation for operation in required
                         if row["resources"][operation]["status"] == "unknown")
        if unknown:
            missing[row["asset_id"]] = unknown
            blockers.append(f"{row['asset_id']} has unknown required operations: {', '.join(unknown)}")

    return {
        "schema": NAME + ".preflight_result.v1",
        "asset_ids": asset_ids,
        "protocol_cohorts": sorted(cohorts),
        "saved_assets_verified": True,
        "resource_coverage_complete": not missing,
        "executable_saved_only": not blockers,
        "missing_operations": missing,
        "blockers": blockers,
        "fair_comparison_verified": False,
        "local_performance_ranking_authorized": False,
        "paper_reported_values_included": False,
        "P2_admission": False,
        "P3_admission": False,
        "P4_admission": False,
        "P5_admission": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    check = sub.add_parser("preflight")
    check.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory()
        code = 0
    else:
        raw = args.manifest.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_pairs_unique)
        result = preflight(manifest)
        code = 0 if result["executable_saved_only"] else 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
