"""Strict saved-data consumer for the new bank-bound frontier, no physics."""
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p2_bank_bind_before_compare_20261030_v1 as bound
from experiments.spectral_reachability import codex_p2_host_physical_bridge_20261030_v1 as bridge

frontier = bound.frontier
need = bound.need


def consume(*, directory, expected_report_sha256, ready, compiled, protocol, prepared, source_bindings):
    """Authenticate externally retained report/source hashes and revalidate all slots."""
    path = Path(directory)
    need(path.is_absolute() and path.is_dir() and not path.is_symlink(), "attempt directory")
    bridge.check_sources(source_bindings)
    report_path = path / "report.json"
    need(frontier.file_sha(report_path) == expected_report_sha256, "report hash")
    report = frontier.read(report_path)
    need(report["schema"] == "codex_p2_bank_bound_frontier_v1" and
         report["status"] == "exploratory_complete" and report["fixture"] is False and
         report["physical_authorized"] is False and report["P1_admission"] is False and
         report["P2_admission"] is False and report["equal_total_compute"] is False,
         "complete unadmitted native report")
    expected_binding = {"root_sha256": ready["root_sha256"],
                        "protocol_sha256": frontier.digest(protocol),
                        "prepared_sha256": frontier.digest(prepared)}
    need(report["binding"] == expected_binding, "prepared/root binding")
    need(frontier.file_sha(path / "bank.json") == report["bank_file_sha256"] and
         frontier.file_sha(path / "bank_binding.json") == report["bank_binding_sha256"],
         "bank or binding hash")
    bank_file = frontier.read(path / "bank.json")
    bank = frontier.candidate_bank(bank_file["base"], ready["root_sha256"])
    need(bank_file["expanded"] == bank, "bank recipe")
    provenances = {a: frontier.provenance(protocol, prepared, ready, bank, a) for a in bound.ARMS}
    binding = frontier.read(path / "bank_binding.json")
    need(binding == {"schema": "codex_p2_runtime_bank_binding_v1",
         "root_sha256": ready["root_sha256"], "bank_file_sha256": report["bank_file_sha256"],
         "bank_digest": frontier.digest(bank), "provenances": provenances,
         "P1_admission": False, "P2_admission": False} and
         report["provenances"] == provenances, "runtime provenance binding")
    journal = path / "events.jsonl"
    data = journal.read_bytes()
    need(data.endswith(b"\n") and frontier.file_sha(journal) == report["events_sha256"],
         "complete report-bound journal")
    events = [json.loads(line) for line in data.splitlines()]
    markers = [(i, e) for i, e in enumerate(events) if e.get("kind") == "bank_bound_before_compare"]
    started = [(i, e) for i, e in enumerate(events) if e.get("kind") == "arm_started"]
    verified = [(i, e) for i, e in enumerate(events) if e.get("kind") == "arm_verified"]
    need(len(markers) == 1 and len(started) == len(verified) == 2 and
         [e["arm"] for _, e in started] == list(bound.ARMS) and
         [e["arm"] for _, e in verified] == list(bound.ARMS) and
         markers[0][0] < started[0][0] < verified[0][0] < started[1][0] < verified[1][0] and
         all(e["binding_sha256"] == report["bank_binding_sha256"]
             for _, e in markers + started + verified), "bank must precede comparison")
    entries = [e["entry"] for e in events if e.get("kind") == "entry_terminal"]
    frontier.validate_actual(entries)
    ops = [e for e in events if e.get("kind") in ("jacobian", "suffix_jacobian",
           "bounded_ridge_svd", "mode_svd", "suffix_solve")]
    frontier.validate_operations(ops, report["results"])
    pseudo = {"status": report["status"], "fixture": False, "P1_admission": False,
              "P2_admission": False, "physical_authorized": False, "equal_total_compute": False,
              "results": report["results"]}
    plans = bridge.validated_plans(pseudo, bank, ready, compiled, provenances)
    bridge.check_sources(source_bindings)
    need(frontier.file_sha(report_path) == expected_report_sha256, "report drift")
    return {"plans": plans, "report_sha256": expected_report_sha256,
            "journal_sha256": frontier.file_sha(journal), "physical_authorized": False,
            "equal_total_compute": False, "P1_admission": False, "P2_admission": False}
