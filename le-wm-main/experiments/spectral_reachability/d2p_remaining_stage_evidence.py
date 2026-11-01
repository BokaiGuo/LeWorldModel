"""Bounded P4/P5 evidence inventory, not an admission authority.

Run from le-wm-main with python -m
experiments.spectral_reachability.d2p_remaining_stage_evidence.
Only explicit documents/reports are read; JSON is emitted to stdout. No model,
GPU, subprocess, directory traversal, weights, artifact writes or source execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat

MAX_BYTES = 2_000_000
DOCUMENTS = {
    "plan": "docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md",
    "closure": "docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md",
    "matrix": "docs/D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md",
    "latest_integration": "docs/D2P_DOWNSTREAM_CUDA_VERIFIED_PROGRESS_20261026.md",
    "latest_program_search": "docs/D2P_PROGRAM_SEARCH_CUDA_INTEGRATION_20261026.md",
    "latest_search_execution": "docs/D2P_SEARCH_PHYSICAL_EXECUTION_20261026.md",
}
# Exact schemas and gate names are from the inspected receipts, not new gates.
REPORTS = {
    "search_physical": (
        "pusht_search_physical4_v1_20261026_cuda/report.json",
        "pusht_search_physical_cuda.v1", "integration_completed",
        ("all_four_retained", "no_execution_errors", "commands_match_sealed_plan",
         "parameters_unchanged", "buffers_unchanged", "frozen_eval", "inputs_unchanged")),
    "program_search_v2": (
        "pusht_program_search4_v2_20261026_cuda/report.json",
        "pusht_program_search_cuda.v1", "integration_completed",
        ("four_roots", "all_full_replays", "actual_planning_samples_320", "actual_budget_caps",
         "parameters_unchanged", "buffers_unchanged", "eval", "inputs_unchanged")),
    "program_search_v1": (
        "pusht_program_search4_v1_20261026_cuda/report.json",
        "pusht_program_search_cuda.v1", "integration_completed", ()),
    "pointmaze_sra_p4": (
        "sra_p4_rgb_closed_loop_r3_full_20261024_cuda/report.json",
        "sra_pointmaze_p4_rgb_closed_loop.v1", "admission_pass",
        ("cuda_only", "frozen_parameter_hash_unchanged", "matched_two_step_candidate_rollout_counts",
         "rgb_only_policy_input", "root_disjoint_calibration_outcome")),
    "pusht_calibration": (
        "pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json",
        "pusht_d2p_p1_reconfiguration_preflight.v5", "pass",
        ("cuda_only", "frozen_parameter_hash_unchanged", "rgb_goal_action_only")),
    "observed_prefix": (
        "pusht_observed_prefix_prediction4_v1_20261026_cuda/report.json",
        "pusht_observed_prefix_prediction.v1", "completed",
        ("four_roots", "predict_samples_64", "parameters_unchanged", "buffers_unchanged", "eval", "inputs_unchanged")),
    "composition_replay": (
        "pusht_composition_replay4_v4_20261026_cuda/report.json",
        None, "completed", ()),
    "budget_guard": (
        "pusht_d2p_budget_guard_probe4x16_v1_20261026_cuda/report.json",
        "pusht_d2p_budget_guard_cuda.v1", "cuda_integration_pass",
        ("all_blocked_before_predict", "all_frozen_eval", "all_predictions_matched", "buffers_unchanged",
         "four_roots", "inputs_unchanged", "parameters_unchanged", "sources_unchanged")),
    "sealed_execution": (
        "pusht_d2p_sealed_program_execution4_v1_20261026_cuda/report.json",
        "pusht_sealed_execution.v1", "completed",
        ("all_64_executed", "parameters_unchanged", "buffers_unchanged", "eval", "inputs_unchanged")),
    "historical_pusht_p1_outcome": (
        "pusht_d2p_p1_outcome_physical_20261024_cuda/report.json",
        "pusht_d2p_p1_physical_calibration.v1", "pass",
        ("cuda_only", "frozen_parameter_hash_unchanged", "root_disjoint_from_physical_calibration",
         "all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero")),
}


def bounded_read(root: Path, relative: str) -> bytes:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes evidence root")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("regular file required")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("report exceeds 2 MB limit; not read")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("report exceeds 2 MB limit")
    return raw


def strict_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    def finite(node):
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError("nonfinite JSON number")
        if isinstance(node, dict):
            for item in node.values():
                finite(item)
        elif isinstance(node, list):
            for item in node:
                finite(item)
    finite(value)
    if not isinstance(value, dict) or not value:
        raise ValueError("nonempty report object required")
    return value


def inspect_report(root: Path, spec: tuple) -> tuple[dict, dict]:
    relative, schema, status, required = spec
    entry = {"path": "results/" + relative, "availability": "unknown", "evidence_status": "blocked"}
    try:
        (root / entry["path"]).lstat()
        entry["availability"] = "present"
        raw = bounded_read(root, entry["path"])
        payload = strict_json(raw)
        gates = payload.get("gates")
        missing = [key for key in required if not isinstance(gates, dict) or key not in gates]
        failed = [key for key, value in gates.items() if value is not True] if isinstance(gates, dict) else []
        issues = []
        if schema is None:
            issues.append("schema contract not independently inspected")
        elif payload.get("schema") != schema:
            issues.append("schema mismatch")
        if payload.get("status") != status:
            issues.append("status does not match expected report completion")
        if not isinstance(gates, dict) or not gates or missing or failed:
            issues.append("missing/failed/nonboolean integrity gates")
        entry.update(sha256=hashlib.sha256(raw).hexdigest(), reported_schema=payload.get("schema"),
                     reported_status=payload.get("status"), missing_gates=missing, failed_gates=failed,
                     reported_gate_count=len(gates) if isinstance(gates, dict) else 0,
                     issues=issues, evidence_status="blocked" if issues else "report_checks_pass",
                     admission=False, seal_verified=False)
        entry["reported_admission_flags"] = {key: payload[key] for key in
            ("P1_admission", "P2_admission", "P3_admission", "P4_admission", "P5_admission") if key in payload}
        return entry, payload
    except FileNotFoundError:
        entry.update(availability="missing", evidence_status="not_run", issues=["explicit report missing"])
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        entry["issues"] = [str(exc)]
    return entry, {}


def root_set(values):
    if not isinstance(values, list) or not values:
        raise ValueError("nonempty explicit root list required")
    if any(type(v) not in (str, int) or (isinstance(v, str) and not v.strip()) for v in values):
        raise ValueError("invalid root identity")
    # Avoid coercing integer indices into content hashes or other namespaces.
    roots = {(type(v).__name__, v) for v in values}
    if len(roots) != len(values):
        raise ValueError("duplicate root identity")
    return roots


def check_sra_roots(report: dict) -> dict:
    """Recompute report-local split isolation and method pairing, not provenance."""
    try:
        inputs = report["inputs"]
        calibration = root_set(inputs["calibration_roots"])
        outcome = root_set(inputs["outcome_roots"])
        proxy = root_set(report["proxy_calibration"]["roots"])
        rows = report["rows"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing outcome rows")
        methods = {}
        for row in rows:
            method = row["method"]
            if not isinstance(method, str) or not method:
                raise ValueError("invalid method identity")
            methods.setdefault(method, []).append(row["root"])
        paired = {method: root_set(roots) for method, roots in methods.items()}
        if len(paired) < 2 or any(roots != outcome for roots in paired.values()):
            raise ValueError("missing/extra/unpaired method roots")
        if calibration & outcome or proxy != calibration:
            raise ValueError("calibration/outcome overlap or proxy split mismatch")
        return {"status": "report_local_disjoint_and_paired", "calibration_roots": len(calibration),
                "outcome_roots": len(outcome), "methods": sorted(methods),
                "source_episode_independence": "unknown", "scope": "same declared archive/index namespace only"}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "blocked", "reason": str(exc)}


def compare_saved_roots(calibration: dict, candidate: dict) -> dict:
    """Known calibration reuse is not independent outcome; never choose roots."""
    try:
        left = root_set([row["root_sha256"] for row in calibration["rows"]])
        right = root_set([row["root_sha256"] for row in candidate["rows"]])
        for _, value in left | right:
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("expected RGB SHA256 identity")
        overlap = len(left & right)
        return {"status": "known_calibration_reuse" if overlap else "rgb_hash_disjoint_only",
                "calibration_roots": len(left), "candidate_roots": len(right), "overlap": overlap,
                "source_episode_independence": "unknown", "independent_outcome_established": False}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "unknown", "reason": str(exc), "independent_outcome_established": False}


def search_accounting(report: dict) -> dict:
    """Recount saved phase receipts only; not independent CUDA trace verification."""
    try:
        rows = report["rows"]
        root_set([row["root_sha256"] for row in rows])
        if len(rows) != 4:
            raise ValueError("four saved calibration roots required")
        totals = {"planning": 0, "evaluation": 0}
        per_root = []
        for row in rows:
            budget = row["budget"]
            if budget["binding"]["root_id"] != row["root_sha256"] or row["full_replay_exact"] is not True:
                raise ValueError("root binding or full replay failure")
            phases = budget["phases"]
            if not isinstance(phases, list) or len(phases) != 2:
                raise ValueError("two distinct phase receipts required")
            local = {}
            for phase in phases:
                name = phase["phase"]
                count = phase["predict_samples"]
                if (name not in totals or name in local or type(count) is not int or count <= 0
                        or phase["completed"] is not True or phase["failed_predict_calls"] != 0):
                    raise ValueError("invalid or failed phase receipt")
                local[name] = count
            actual = sum(local.values())
            if (local["planning"] != row["planning_predict_samples"] or local["planning"] != 80
                    or type(budget["attempted_predict_samples"]) is not int
                    or actual != budget["attempted_predict_samples"]
                    or type(budget["predict_sample_limit"]) is not int
                    or budget["predict_sample_limit"] != 82 or actual > 82):
                raise ValueError("saved sample accounting/cap mismatch")
            per_root.append(actual)
            for name, count in local.items():
                totals[name] += count
        if totals != {"planning": 320, "evaluation": 7}:
            raise ValueError("saved totals differ from inspected 320 planning plus 7 replay samples")
        return {"status": "saved_receipts_consistent", "planning_predict_samples": 320,
                "replay_predict_samples": 7, "actual_predict_samples": 327, "per_root_actual_samples": per_root,
                "physical_execution_evidence": False, "compute_matched_claim": False,
                "scope": "saved receipt arithmetic; no new inference or independent CUDA trace"}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "blocked", "reason": str(exc)}


def audit(project_root: Path) -> dict:
    project_root = Path(project_root)
    documents = {}
    for name, relative in DOCUMENTS.items():
        try:
            raw = bounded_read(project_root, relative)
            anchors = [{"line": number, "text": line} for number, line in enumerate(raw.decode().splitlines(), 1)
                       if line.startswith("| P4") or line.startswith("| P5") or line.startswith("| P4：") or line.startswith("| P5：")]
            documents[name] = {"path": relative, "status": "read", "sha256": hashlib.sha256(raw).hexdigest(), "anchors": anchors}
        except (OSError, ValueError, UnicodeError) as exc:
            documents[name] = {"path": relative, "status": "blocked", "reason": str(exc), "anchors": []}
    reports, payloads = {}, {}
    for name, spec in REPORTS.items():
        reports[name], payloads[name] = inspect_report(project_root / "le-wm-main", spec)
    reports["pointmaze_sra_p4"]["root_audit"] = check_sra_roots(payloads["pointmaze_sra_p4"])
    for name in ("observed_prefix", "composition_replay", "budget_guard", "sealed_execution", "program_search_v2", "search_physical"):
        reports[name]["root_audit"] = compare_saved_roots(payloads["pusht_calibration"], payloads[name])
    # Preserve every saved execution row, including guard stops; no success filtering.
    rows = payloads["search_physical"].get("rows", [])
    reports["search_physical"]["saved_execution_rows"] = [
        {key: row.get(key) for key in ("root_index", "root_sha256", "status", "error",
            "planned_commands", "completed_commands", "remaining_commands_not_executed", "official")}
        for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    reports["search_physical"]["role"] = "single finite-length sealed-plan physical execution on known calibration roots, including observation guard stop; not multi-round observe/recompile P4, independent outcome, task benefit or P1"
    reports["program_search_v2"]["saved_accounting"] = search_accounting(payloads["program_search_v2"])
    reports["program_search_v2"]["role"] = "model-only fixed-bank beam/tree CUDA integration; not physical/P1/P2/P3 or faithful official kinodynamic replication"
    reports["program_search_v1"]["attempt_status"] = "failed attempt documented in latest_program_search; never backfill with v2"
    if reports["program_search_v1"]["availability"] == "missing":
        reports["program_search_v1"]["evidence_status"] = "blocked"
        reports["program_search_v1"]["issues"] = ["completion report missing; v1 code/instrumentation error documented, not a method negative result"]
    requirements = {
        "P4": ["each environment's own P1 and applicable P2/P3 admission", "fixed starts/goals and total budget",
               "RGB observe/recompile multi-round execution to termination, evaluator-only physical state",
               "same checkpoint, matched optimization budget, root-paired physical success/cost and failures",
               "PushT approach/contact/push/closure breakdown and final object task success"],
        "P5": ["P1 and replicated core components admitted and frozen", "predeclared independent frozen model/architecture and tasks",
               "per-model interface/P0 and task prerequisites", "replicated mechanism and physical benefit with training-free constraint",
               "same-model multi-task results separate from cross-model replication; disclose pretrained/extra learning cost"],
    }
    slots = {
        "P4_pointmaze_d2p": "PointMaze Full D2P RGB closed-loop outcome",
        "P4_reacher_d2p": "Reacher Full D2P RGB closed-loop outcome",
        "P4_pusht_d2p": "PushT Full D2P RGB closed-loop outcome and contact breakdown",
        "P5_independent_model": "predeclared independent frozen architecture replication",
        "P5_same_model_tasks": "separate same-checkpoint task matrix",
    }
    return {"schema": "d2p_remaining_stage_evidence.v1", "read_only": True,
            "scope": "explicit 20261026 allowlist snapshot, not repository-wide proof of absence or admission",
            "neural_work": 0, "gpu_use": False, "new_physical_execution": 0,
            "P1_admitted": False, "efficacy_claim": False, "documents": documents, "reports": reports,
            "required_report_slots": {key: {"definition": value, "path": None, "status": "not_run",
                "reason": "no formal report bound in inspected canonical documents; existence elsewhere unknown"}
                for key, value in slots.items()},
            "stages": {stage: {"status": "blocked", "admitted": False, "requirements": items,
                "blockers": ["P1 not passed", "formal stage reports and root/source/protocol-bound validation missing"],
                "definition_sources": [DOCUMENTS["plan"], DOCUMENTS["closure"], DOCUMENTS["matrix"]]}
                for stage, items in requirements.items()},
            "boundary": "SRA PointMaze limited P4 remains separate. Completed replay/calibration, positive flags and unit tests never unlock D2P P4/P5. No source seals or host CUDA traces re-certified."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    result = audit(parser.parse_args(argv).project_root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    # Nonzero means stages remain blocked, not that an experiment was attempted.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
