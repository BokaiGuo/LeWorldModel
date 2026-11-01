"""Pinned saved-metadata requirements, never a fresh certificate issuer.

Does not follow catalog/report paths, decode NPZ, inspect reserved roots,
collect evidence, or approve origin assertions.  Reuses the existing authority
request protocol instead of creating a competing response/certificate schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NAME = "agent_p4_certificate_requirements_20261028_v1"
ROOT = Path(__file__).resolve().parents[2]
CAP = 16 * 1024**2
PINS = {
    "audit_report": ("results/agent_p4_fresh_certificate_audit_20261028_v2_report_MAIN01.json", "15e3ece87e36eba469a0cc2188efbb7edaa94fa00f1562f9e6adc2052e4e4ae0"),
    "audit_source": ("experiments/spectral_reachability/agent_p4_fresh_certificate_audit_20261028_v2.py", "3242a7d1d32c6667563588744bc153c6a5ec3f21d2274762ed07385239549092"),
    "authority_source": ("experiments/spectral_reachability/agent_freshroot_authority_20261028_v1.py", "f7a30fcad18483e1848170acdd1fe4bd96e7d08b38804b18f1cf2cf1087870e7"),
    "root_manifest": ("results/chat_freshroots_20261027_root_manifest.json", "3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838"),
    "consumer_manifest": ("results/chat_freshroots_20261027_consumer_manifest_v2.json", "2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7"),
    "coverage": ("results/agent_freshroots_resume_20261027_coverage.json", "00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb"),
    "inventory": ("results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json", "e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591"),
    "closure": ("results/moving_authority_identity_link_closure_20261027_v1_MAIN01.json", "7c7e1792c5a17986aaf3a38539a280ec46d5910eb4d9351babbe9bf2a70b3576"),
    "exclusion": ("results/moving_root_exclusion_manifest_20261027_v1_inventory8_main01.json", "a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060"),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=unique)
    canonical(result)
    return result


def file_bytes(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(), "canonical regular file")
    need(path.stat().st_size <= CAP, "metadata byte cap")
    raw = path.read_bytes()
    need(len(raw) <= CAP, "metadata grew beyond cap")
    return raw


def load_inputs():
    documents = {}
    for key, (relative, expected) in PINS.items():
        raw = file_bytes(ROOT / relative)
        need(digest(raw) == expected, "pinned metadata drift " + key)
        if not key.endswith("_source"):
            documents[key] = parse(raw)
    return documents


def reference(key, pointer):
    relative, expected = PINS[key]
    return {"path": str(ROOT / relative), "sha256": expected, "pointer": pointer}


def own_closure():
    paths = [Path(__file__).resolve(), ROOT / "tests" / ("test_" + NAME + ".py"), ROOT / "innerdocs" / (NAME + ".md")]
    return {str(path): digest(file_bytes(path)) for path in paths}


def requirement(identifier, key, pointer, present, value, fields, responsible):
    return {"id": identifier, "status": "missing_or_unapproved", "evidence": reference(key, pointer),
            "field_present": present, "observed": value, "missing_fields": fields,
            "responsible": responsible, "can_be_satisfied_by_new_execution": False}


def build():
    data = load_inputs()
    audit, cov, excl = [data[key] for key in ("audit_report", "coverage", "exclusion")]
    need(audit["schema"] == "agent_p4_fresh_certificate_audit_20261028_v2.report" and audit["status"] == "blocked_no_legal_fresh_certificate", "blocked audit schema/status")
    need(audit["certificate_created"] is False and audit["reserved_roots_opened"] is False, "audit safety flags")
    # This verifies the metadata subset actually read, not the referenced NPZ or
    # sourcepack.  Those remain report-level citations, never direct reads.
    for key in ("root_manifest", "consumer_manifest", "coverage", "inventory", "closure", "exclusion"):
        relative, expected = PINS[key]
        need(audit["bindings"].get(str(ROOT / relative)) == expected, "audit upstream binding " + key)
    need(cov["coverage_complete"] is False and cov["original_execution_bounds_certified"] == 0, "history status changed; new version required")
    need(excl["approval"]["approved"] is False and excl["source_run_preconditions_met"] is False, "approval status changed")
    unresolved = excl["unresolved_sources"]
    need(len(unresolved) == audit["history"]["unresolved_count"] == 975, "complete unresolved denominator")
    from experiments.spectral_reachability import agent_freshroot_authority_20261028_v1 as authority
    need(Path(authority.__file__).resolve() == ROOT / PINS["authority_source"][0], "authority import origin")
    request = authority.requests(cov)
    history_requirements = []
    dispatches = cov["recovery"]["dispatches"]
    for source in request["sources"]:
        for run in source["dispatches"]:
            index = next(i for i, item in enumerate(dispatches) if item["call_id"] == run["call_id"])
            history_requirements.append({
                "source_id": source["source_id"], "call_id": run["call_id"], "report": source["report"],
                "inherited_request_ids": [item["request_id"] for item in source["inherited_refs"]],
                "evidence": reference("coverage", "/recovery/dispatches/" + str(index)),
                "status": "original_reset_bound_not_certified",
                "missing_fields": ["assertion.attempts", "assertion.attempts_complete", "assertion.restarts_complete",
                                   "assertion.reset_evidence", "origin_citation"],
                "route_options": [
                    {"route": "exhaustive_ledger", "required": ["call_id", "attempts[index,reset_kind,disposition,identity_sha256]"]},
                    {"route": "loaded_closure_and_exhaustive_envelope", "required": ["call_id", "upper_bound", "derivation", "original_load_provenance"],
                     "loaded_roles": sorted(authority.load_roles(run))}],
                "attempts": None, "origin_review_required": True,
            })
    need(len(history_requirements) == 10, "full known dispatch denominator")
    needs = []
    for key in ("root_manifest", "consumer_manifest"):
        manifest = data[key]
        need(manifest["freshness_certified"] is False and manifest["source_run_authorized"] is False, "manifest unexpectedly authorized")
        need(manifest.get("history_certificate") is None, "certificate state changed")
        needs.append(requirement(key + ".history_certificate", key, "/history_certificate", "history_certificate" in manifest,
                                 manifest.get("history_certificate"), ["history_certificate"], "original run owners and MAIN"))
    needs.extend([
        requirement("history_scope", "coverage", "/scope", True, cov["scope"],
                    ["historical_scope.machines", "historical_scope.worktrees", "historical_scope.time_interval",
                     "historical_scope.unsaved_failures", "historical_scope.origin_citation", "cutoff_delta_disposition"], "original run owners and MAIN"),
        requirement("coverage_approval", "exclusion", "/approval", True, excl["approval"],
                    ["external_file", "schema", "authority", "approved", "coverage_scope", "cutoff_utc", "manifest_content_sha256"], "MAIN only; no self approval"),
        requirement("exclusion_unresolved", "exclusion", "/unresolved_sources", True, {"count": len(unresolved)},
                    ["original identity/reset evidence", "approved identity bridges", "complete family/scope coverage"], "original source owners and MAIN"),
        requirement("future_pre_witness_identity", "audit_report", "/freshness/certified_fresh_roots", True, [],
                    ["actual state7", "wrapped pose", "snapshot", "RGB3", "raw history", "task/goal identity",
                     "historical exclusion comparison", "within-batch dedup", "access/disclosure history", "pre-witness ordering"], "independent root pipeline after legal authorization"),
    ])
    # Future actual identity requires a later authorized collection; this request
    # does not demand opening the reserved roots now or fabricate their identity.
    needs[-1]["future_stage"] = "requires_separate_MAIN_authorization_before_collection"
    inventory = data["inventory"]
    identity_rows = inventory["history_file_requests"]
    need(len(identity_rows) == inventory["summary"]["captured_files"] == 618, "full identity metadata inventory")
    families = [{"evidence": reference("inventory", "/authority_requests/" + str(i)),
                 "family": item["family"], "inventory_paths": item["inventory_paths"],
                 "status": item["status"], "missing_fields": [item["remaining"]]}
                for i, item in enumerate(inventory["authority_requests"])]
    return {"schema": NAME + ".request", "status": "blocked_upstream_certificate_requirements",
            "implementation_closure": own_closure(), "bindings": {key: reference(key, "") for key in PINS},
            "authority_request": request, "authority_request_sha256": digest(canonical(request)),
            "authority_interface": {"module": "experiments.spectral_reachability.agent_freshroot_authority_20261028_v1",
                                    "request_schema": request["schema"], "response_schema": authority.NAME + ".response",
                                    "validation_is_approval": False},
            "historical_dispatch_requirements": history_requirements, "requirements": needs,
            "family_requirements": families,
            "identity_inventory": {"evidence": reference("inventory", "/summary"), "summary": inventory["summary"],
                                   "closure_summary_evidence": reference("closure", "/summary"), "closure_summary": data["closure"]["summary"],
                                   "records": [{"evidence": reference("inventory", "/history_file_requests/" + str(i)),
                                                "path": item["path"], "sha256": item["sha256"], "blockers": item["blockers"],
                                                "status": item["status"]} for i, item in enumerate(identity_rows)]},
            "unresolved_sources": [{"evidence": reference("exclusion", "/unresolved_sources/" + str(i)), "record": item} for i, item in enumerate(unresolved)],
            "denominators": {"historical_sources": 8, "inherited_refs": 16, "known_dispatches": 10, "unresolved_occurrences": len(unresolved)},
            "construction_disposition": audit["certificate_disposition"],
            "reported_only_not_opened": audit["certificate_candidates"],
            "other_audit_bindings_not_read": {path: expected for path, expected in audit["bindings"].items()
                                             if path not in {str(ROOT / rel) for rel, _ in PINS.values()}},
            "snapshot_limitations": ["Oct27 coverage is not all subsequent history; MAIN must declare cutoff and later delta", "975 occurrences are not 975 independent runs", "missing evidence is not proof of nonexistence", "construction certificates do not certify fresh same-task roots"],
            "certified_fresh_roots": [], "history_certificate": None, "source_run_authorized": False,
            "reserved_root_access_authorized": False, "certificate_created": False, "approval_created": False,
            "P1_admission": False, "scientific_admission": False, "reserved_roots_opened": False,
            "new_neural_calls": 0, "new_physics_calls": 0}


def consume(path, expected):
    path = Path(path)
    need(path.name.startswith(NAME + "_") and path.suffix == ".json", "only own request filenames accepted")
    raw = file_bytes(path)
    need(digest(raw) == expected, "request SHA")
    need(canonical(parse(raw)) == canonical(build()), "request reconstruction and input closure")
    return {"schema": NAME + ".consume", "status": "blocked", "request_sha256": expected,
            "source_run_authorized": False, "reserved_root_access_authorized": False,
            "certificate_created": False, "approval_created": False,
            "next_action": "obtain_original_evidence_and_MAIN_scope_review_not_root_execution"}


def write(path, value):
    path = Path(path)
    need(path.is_absolute() and path.parent.resolve() == path.parent and path.name.startswith(NAME + "_") and path.suffix == ".json", "new own request path")
    raw = canonical(value)
    need(len(raw) <= CAP, "request cap")
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "short request write")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("request", "authority-request"):
        option = sub.add_parser(command)
        option.add_argument("--output", type=Path)
    option = sub.add_parser("consume")
    option.add_argument("--request", type=Path, required=True)
    option.add_argument("--sha", required=True)
    args = parser.parse_args()
    if args.command == "consume":
        result = consume(args.request, args.sha)
    else:
        value = build()
        result = value["authority_request"] if args.command == "authority-request" else value
        if args.output is not None:
            write(args.output, result)
            result = {"output": str(args.output), "sha256": digest(canonical(result)), "source_run_authorized": False}
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
