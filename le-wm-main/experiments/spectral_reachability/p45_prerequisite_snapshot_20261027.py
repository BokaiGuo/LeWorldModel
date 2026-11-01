"""Recompute a bounded evidence snapshot, never stage admission or neural work.

Only explicitly named reports/documents are read. No weights, datasets, model
imports, root selection, subprocesses or GPU calls. --write creates a NEW fixed
results directory; an existing snapshot is never overwritten. Exit 1 means the
scientific stages remain blocked, including when the inventory was successful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat

from experiments.spectral_reachability import d2p_remaining_stage_evidence as prior

NAME = "p45_prerequisite_snapshot_20261027"
MAX_BYTES = 8_000_000
DOCS = dict(prior.DOCUMENTS, **{
    "remaining": "docs/D2P_REMAINING_STAGE_EVIDENCE_20261026.md",
    "grasp_hilewm": "docs/D2P_REPORTED_BASELINES_GRASP_HILEWM_20261026.md",
    "lewm_planwam": "docs/D2P_REPORTED_BASELINES_LEWM_PLANWAM_20261026.md",
    "baseline_policy": "docs/D2P_BASELINE_EVIDENCE_POLICY_20261026.md",
    "baseline_faithfulness": "docs/D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md",
    "cross_model": "docs/D2P_CROSS_MODEL_BASELINE_ASSET_AUDIT_20261026.md",
    "representation_runtime": "docs/D2P_REPRESENTATION_RUNTIME_AUDIT_20261027.md",
})
SPECS = dict(prior.REPORTS)
SPECS["composition_replay"] = (
    "pusht_composition_replay4_v4_20261026_cuda/report.json",
    "pusht_composition_replay_cuda.v2", "completed",
    ("64_programs", "384_predict_samples", "parameters_unchanged", "buffers_unchanged", "eval", "inputs_unchanged"))
SPECS["representation_runtime"] = (
    "pusht_representation_runtime_audit4_v1_20261027_cuda/report.json",
    "pusht_representation_runtime_cuda.v1", "audit_completed",
    ("twelve_real_images", "all64_programs", "actual_predict_samples448", "parameters_unchanged",
     "buffers_unchanged", "frozen_eval", "inputs_unchanged"))
for _key, _directory in (
    ("reacher_default", "reacher_d2p_p0_predict_accounting_calibration2_v1_20261026_cuda"),
    ("reacher_sample_matched", "reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda"),
):
    SPECS[_key] = (_directory + "/report.json", "d2p_reacher_p0_goal_progress.v1", "calibration_only",
                  ("cuda_only", "frozen_parameter_hash_unchanged"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes snapshot root")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
        raise ValueError("regular file within 8 MB required")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("8 MB read limit")
    return raw


def inspect(root, spec):
    relative, schema, status, required = spec
    path = "le-wm-main/results/" + relative
    entry = {"path": path, "admission": False, "seal_recertified": False}
    try:
        raw = read(root, path)
        report = prior.strict_json(raw)
        if (not isinstance(report.get("protocol", {}), dict) or
                not isinstance(report.get("rows", []), list) or
                any(not isinstance(row, dict) for row in report.get("rows", []))):
            raise ValueError("malformed protocol/rows")
        gates = report.get("gates")
        missing = [k for k in required if not isinstance(gates, dict) or k not in gates]
        failed = [k for k, v in gates.items() if v is not True] if isinstance(gates, dict) else ["gates"]
        good = (report.get("schema") == schema and report.get("status") == status
                and isinstance(gates, dict) and bool(gates) and not missing and not failed)
        entry.update(sha256=sha(raw), bytes=len(raw), availability="read",
                     status="report_checks_pass" if good else "blocked",
                     schema=report.get("schema"), reported_status=report.get("status"),
                     gates=gates, missing_gates=missing, failed_gates=failed,
                     protocol=report.get("protocol", "unknown"), inputs=report.get("inputs", "unknown"),
                     host=report.get("host", report.get("cuda_host", "unknown")),
                     reported_source_verification=report.get("source_verification", "unknown"))
        # Explicit pointer-addressed identities, no assumption that seed == RGB hash.
        entry["roots"] = [{"pointer": f"/rows/{i}", **{k: row[k] for k in
            ("root_sha256", "root_id", "root", "seed", "root_index", "split", "goal_rgb_sha256", "history_rgb_sha256") if k in row},
            **({"executed_context": row["executed_context"]} if "executed_context" in row else {})}
            for i, row in enumerate(report.get("rows", []))]
        return entry, report
    except (OSError, ValueError, UnicodeError, RecursionError, TypeError) as exc:
        entry.update(availability="missing" if isinstance(exc, FileNotFoundError) else "unreadable_or_invalid",
                     status="blocked", reason=str(exc))
        return entry, {}


def composition_arithmetic(report):
    """Rehash saved witnesses and chain boundaries; NO transition/model replay."""
    try:
        rows = report["rows"]
        prior.root_set([r["root_sha256"] for r in rows])
        if len(rows) != 4:
            raise ValueError("four roots required")
        programs = 0
        for row in rows:
            ids = prior.root_set([p["candidate_id"] for p in row["programs"]])
            if ids != {("str", f"prefix_{i}") for i in range(16)}:
                raise ValueError("complete fixed prefix IDs required")
            for program in row["programs"]:
                receipt = program["receipt"]
                body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
                if digest(body) != receipt["receipt_sha256"]:
                    raise ValueError("receipt digest mismatch")
                if (receipt["root"]["root_id"] != row["root_sha256"] or
                        receipt["actions_sha256"] != digest(receipt["actions"]) or
                        len(receipt["actions"]) != 10):
                    raise ValueError("root/actions mismatch")
                context_hash = digest(receipt["root"]["context"])
                transitions = receipt["transitions"]
                if len(transitions) != 2:
                    raise ValueError("two transitions required")
                for i, edge in enumerate(transitions):
                    if (type(edge["start_step"]) is not int or type(edge["end_step"]) is not int or
                            edge["start_step"] != 5*i or edge["end_step"] != 5*(i+1) or
                            edge["input_context_sha256"] != context_hash or
                            edge["actions_sha256"] != digest(receipt["actions"][5*i:5*(i+1)])):
                        raise ValueError("transition chain mismatch")
                    context_hash = edge["output_context_sha256"]
                if context_hash != digest(receipt["terminal_context"]):
                    raise ValueError("terminal context mismatch")
                if program["split_full_exact"] is not True or program["saved_forecast_matches"] is not True:
                    raise ValueError("saved CUDA replay flags failed")
                programs += 1
        records = report["predict"]
        if not records or any(r["completed"] is not True or r["failed_predict_calls"] != 0 or
                any(type(r[k]) is not int or r[k] < 0 for k in ("predict_samples", "predict_calls")) for r in records):
            raise ValueError("invalid predict receipts")
        samples = sum(r["predict_samples"] for r in records)
        calls = sum(r["predict_calls"] for r in records)
        if programs != 64 or samples != 384 or calls != 384:
            raise ValueError("64/384/384 accounting mismatch")
        return {"status": "saved_content_recomputed", "programs": programs, "predict_samples": samples,
                "predict_calls": calls, "rehash_receipts": programs, "rehash_chain_edges": 2*programs,
                "independent_transition_replay": False, "mechanism_proven": False}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "blocked", "reason": str(exc)}


def p0_counts(report):
    try:
        prior.root_set([r["seed"] for r in report["rows"]])
        result = []
        for i, row in enumerate(report["rows"]):
            methods = {}
            for method, value in row["methods"].items():
                receipt = value["receipt"]["actual_planning_predict"]
                if (receipt["method"] != method or receipt["completed"] is not True or
                        receipt["failed_predict_calls"] != 0 or any(type(receipt[k]) is not int or receipt[k] < 0
                            for k in ("predict_calls", "predict_samples"))):
                    raise ValueError("invalid method count receipt")
                methods[method] = {k: receipt[k] for k in ("predict_calls", "predict_samples")}
            result.append({"seed": row["seed"], "pointer": f"/rows/{i}/methods", "methods": methods,
                           "d2p_cem_samples_equal": methods["d2p_goal_progress"]["predict_samples"] == methods["cem"]["predict_samples"],
                           "d2p_cem_calls_equal": methods["d2p_goal_progress"]["predict_calls"] == methods["cem"]["predict_calls"]})
        if not result:
            raise ValueError("missing saved roots")
        return {"status": "saved_counts_recomputed", "rows": result, "total_compute_matched": False,
                "official_grasp_reproduction": False, "independent_outcome": False}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "blocked", "reason": str(exc)}


def runtime_counts(report):
    try:
        rows = report["rows"]
        prior.root_set([r["root_sha256"] for r in rows])
        programs = [p for row in rows for p in row["programs"]]
        images = report["encoding"]
        checks = report["checks"]
        if (len(rows) != 4 or len(programs) != 64 or len(images) != 12 or
                len(checks) != 5 or any(v is not True for v in checks.values())):
            raise ValueError("missing runtime/image checks")
        predicts = report["predict"]
        if (not predicts or any(r["completed"] is not True or r["failed_predict_calls"] != 0 or
                type(r["predict_samples"]) is not int for r in predicts) or
                sum(r["predict_samples"] for r in predicts) != 448):
            raise ValueError("runtime 448 samples receipt mismatch")
        for row in rows:
            if prior.root_set([p["candidate_id"] for p in row["programs"]]) != {("str", f"prefix_{i}") for i in range(16)}:
                raise ValueError("runtime candidate coverage")
        if any(p["direct_vs_correct_official"]["tolerance_match"] is not True or
               p["direct_vs_saved"]["tolerance_match"] is not True or
               p["official_SSE_equals_192_MSE"] is not True or
               p["correct_predicted_frames"] != 5 or p["legacy_predicted_frames"] != 6 for p in programs):
            raise ValueError("runtime numerical/frame check")
        return {"status": "saved_runtime_checks_recounted", "programs": len(programs), "images": len(images),
                "predict_samples": sum(r["predict_samples"] for r in predicts),
                "max_direct_official_error": max(p["direct_vs_correct_official"]["max_absolute"] for p in programs),
                "max_projected_embedding_error": max(i["projected"]["max_absolute"] for i in images),
                "context": "root-anchored emb3 + between-state actions2, then new B,C; not incoming3 extra step",
                "independent_neural_reexecution": False, "Q_task_efficacy_repaired": False}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "blocked", "reason": str(exc)}


def document(root, path):
    try:
        raw = read(root, path)
        lines = raw.decode().splitlines()
        anchors = [{"line": i, "text": line} for i, line in enumerate(lines, 1)
                   if any(term in line for term in ("P4", "P5", "GRASP", "Hi-LeWM", "PlanWAM", "LeWM", "Table 10", "90.9", "93.8"))]
        return {"path": path, "sha256": sha(raw), "status": "read", "anchors": anchors}
    except (OSError, ValueError, UnicodeError) as exc:
        return {"path": path, "status": "blocked", "reason": str(exc), "anchors": []}


def baseline_findings(documents):
    # Findings only hold if exact source fragments are present in the live document.
    rules = (
        ("GRASP", "grasp_hilewm", "性能条目为 0", "insufficient_metadata", "no verified external numeric entries in inspected report"),
        ("Hi-LeWM", "grasp_hilewm", "性能条目为 0", "insufficient_metadata", "identity/version/table unresolved, not a zero performance result"),
        ("LeWM", "lewm_planwam", "Table 10及其他规划表", "insufficient_metadata", "planning table values not verified; local weights do not bind original paper protocol"),
        ("PlanWAM", "lewm_planwam", "90.9", "context_only", "NAVSIM/HUGSIM, trained planning modules, multi-camera history, driving metrics; not local Reacher/PushT"),
    )
    findings = []
    for name, key, needle, classification, difference in rules:
        source = documents[key]
        quotes = [a for a in source["anchors"] if needle in a["text"]]
        findings.append({"method": name, "classification": classification if quotes else "source_unavailable",
                         "source": source["path"], "source_sha256": source.get("sha256"), "anchors": quotes,
                         "protocol_model_task_difference": difference, "direct_comparable": False,
                         "local_reproduction": False, "same_protocol_misuse_detected_in_inspected_reports": False if quotes else None,
                         "scope": "these two named baseline reports only; not a repository-wide claim audit"})
        if name == "PlanWAM":
            findings[-1]["reported_values_source_rows"] = [a for a in source["anchors"]
                if a["text"].startswith("| Table ") and "PlanWAM" in a["text"]]
            findings[-1]["primary_source_reverified_this_run"] = False
    return findings


def snapshot(root):
    root = Path(root).resolve()
    documents = {k: document(root, p) for k, p in DOCS.items()}
    artifacts, payloads = {}, {}
    for key, spec in SPECS.items():
        artifacts[key], payloads[key] = inspect(root, spec)
    calibration = payloads["pusht_calibration"]
    for key in ("observed_prefix", "composition_replay", "budget_guard", "sealed_execution", "program_search_v2", "search_physical", "representation_runtime"):
        artifacts[key]["root_relation_to_calibration"] = prior.compare_saved_roots(calibration, payloads[key])
    artifacts["pointmaze_sra_p4"]["root_audit"] = prior.check_sra_roots(payloads["pointmaze_sra_p4"])
    artifacts["composition_replay"]["recomputed"] = composition_arithmetic(payloads["composition_replay"])
    artifacts["representation_runtime"]["recomputed"] = runtime_counts(payloads["representation_runtime"])
    artifacts["program_search_v2"]["recomputed"] = prior.search_accounting(payloads["program_search_v2"])
    for key in ("reacher_default", "reacher_sample_matched"):
        artifacts[key]["recomputed"] = p0_counts(payloads[key])
    artifacts["search_physical"]["execution_rows"] = [{k: row.get(k) for k in
        ("root_sha256", "status", "error", "planned_commands", "completed_commands", "remaining_commands_not_executed", "official")}
        for row in payloads["search_physical"].get("rows", [])]
    # Compare only already allowlisted report bytes. Other declared bindings are
    # retained but NOT followed, especially weights and external filesystem paths.
    current = {str(root / a["path"]): a.get("sha256") for a in artifacts.values()}
    for key, payload in payloads.items():
        artifacts[key]["binding_checks"] = [{"path": path, "declared_sha256": value,
            "current_allowlisted_sha256": current.get(path),
            "status": ("match" if current[path] == value else "mismatch") if current.get(path) else "not_rehashed"}
            for path, value in payload.get("protocol", {}).get("bindings", {}).items()]
        artifacts[key]["model_identity"] = {
            "declared_checkpoint_inputs": {k: payload.get("inputs", {}).get(k, "unknown") for k in
                ("checkpoint", "checkpoint_sha256", "weights_sha256")},
            "declared_weight_config_bindings": [b for b in artifacts[key]["binding_checks"]
                if Path(b["path"]).name in ("weights.pt", "config.json")],
            "weights_rehashed_or_loaded": False,
            "boundary": "reported model identities only; missing identity stays unknown; same family is not same checkpoint/architecture replication"}
    # Explicit requirement rows, with evidence and *which* axis fails. These do
    # not assert that completed integration satisfies a scientific prerequisite.
    definitions = (
        ("P4_rgb_recurrence", ["observed_prefix", "composition_replay", "representation_runtime"], "partial_integration",
         "same four calibration RGB roots; imagined split/full recurrence and fixed suffix predictions; no multi-round physical recompile to termination"),
        ("P4_closed_loop_physics", ["search_physical", "sealed_execution"], "partial_execution",
         "known PushT roots; finite sealed plan, per-root completed/planned commands and observation stop retained below; not long-horizon task success"),
        ("P4_budget_and_baselines", ["budget_guard", "program_search_v2", "reacher_default", "reacher_sample_matched"], "partial_accounting",
         "guard caps/search receipts and two-root P0 sample matching; calls/backward/encoding/time/feedback not matched; restart Adam is not official GRASP"),
        ("P4_pointmaze_branch", ["pointmaze_sra_p4"], "separate_limited_branch",
         "PointMaze-SRA, own checkpoint/archive/index split/four-block bank; cannot transfer to Full D2P or Reacher/PushT"),
        ("P4_formal_d2p", ["historical_pusht_p1_outcome", "pusht_calibration"], "blocked_P1",
         "failed historical physical P1 is not repaired by model preflight on reused calibration roots; independent admitted mechanism and full closed-loop panel missing"),
        ("P5_model_task_replication", ["pointmaze_sra_p4", "reacher_sample_matched", "pusht_calibration"], "blocked_P1_and_assets",
         "different task-specific LeWM checkpoints/branches are not a predeclared independent architecture replication; extra-trained adapters/readouts excluded; per-model P0/task/mechanism evidence missing"),
    )
    prerequisites = [{"id": key, "status": status, "artifact_refs": [
        {"id": ref, "path": artifacts[ref]["path"], "sha256": artifacts[ref].get("sha256"),
         "report_status": artifacts[ref]["status"]} for ref in refs],
        "root_protocol_model_task_gap": gap, "formal_stage_satisfied": False,
        "artifact_read_checks_pass": all(artifacts[ref]["status"] == "report_checks_pass" for ref in refs),
        "definition_refs": [DOCS[k] for k in ("plan", "closure", "matrix")]}
        for key, refs, status, gap in definitions]
    independent = [
        {"work": "rehash saved composition witnesses and recount baseline/search receipts", "depends_on_P1": False,
         "performed_now": True, "evidence_refs": ["composition_replay", "program_search_v2", "reacher_sample_matched"],
         "boundary": "saved content arithmetic, not new CUDA/physical evidence or source seal recertification"},
        {"work": "audit paper/protocol/asset identity and official GRASP algorithm mapping", "depends_on_P1": False,
         "performed_now": "partial", "document_refs": [DOCS[k] for k in ("grasp_hilewm", "lewm_planwam", "cross_model", "baseline_faithfulness")],
         "boundary": "missing primary tables/license/checkpoint revision remain unknown; no fetching/loading/training authorized here"},
        {"work": "audit RGB recurrence, guard stops and checkpoint/root binding before future closed-loop panels", "depends_on_P1": False,
         "performed_now": True, "evidence_refs": ["observed_prefix", "composition_replay", "search_physical", "budget_guard"],
         "boundary": "offline audit only; any new neural diagnostic requires host CUDA and main-thread scheduling; no formal P4/P5 efficacy run"},
    ]
    dependencies = {p: document(root, p) for p in (
        "le-wm-main/experiments/spectral_reachability/" + NAME + ".py",
        "le-wm-main/experiments/spectral_reachability/d2p_remaining_stage_evidence.py")}
    # Detect concurrent edits during this bounded read pass. A generated snapshot
    # is not an atomic repository transaction or a certification of current seals.
    stability = []
    for item in [*documents.values(), *artifacts.values(), *dependencies.values()]:
        if "sha256" in item:
            try:
                same = sha(read(root, item["path"])) == item["sha256"]
            except (OSError, ValueError):
                same = False
            stability.append({"path": item["path"], "unchanged_on_second_read": same})
    return {"schema": NAME + ".v1", "source_dependencies": dependencies, "documents": documents,
        "artifacts": artifacts, "prerequisites": prerequisites, "independent_of_P1_work": independent,
        "reported_baseline_audit": baseline_findings(documents),
        "P1_admitted": False, "stages": {k: {"status": "blocked", "admitted": False} for k in ("P4_full_D2P", "P5")},
        "scope": "fixed allowlist, not repository-wide absence proof; no outcome roots selected or new outcomes opened",
        "new_neural_calls": 0, "new_physical_execution": 0, "gpu_used": False,
        "read_stability": stability, "all_read_bytes_stable": bool(stability) and all(s["unchanged_on_second_read"] for s in stability),
        "Q_metric_boundary": "agent-motion capability with static object remains possible; interface/cost calibration only, no task efficacy repair",
        "snapshot_seals_recertified": False, "efficacy_claim": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = snapshot(args.project_root)
    raw = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.write:
        destination = args.project_root / "le-wm-main/results" / NAME
        # Reject symlink escape; mkdir(exist_ok=False) protects shared artifacts.
        if not destination.resolve().is_relative_to(args.project_root.resolve()):
            raise ValueError("output escapes project root")
        destination.mkdir(exist_ok=False)
        with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(raw)
        print(str(destination / "manifest.json"))
    else:
        print(raw, end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
