"""CPU-only replay of two P3 pairs from the anchored, partial host batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import codex_p3_resolved_batch_audit_20261030_v1 as batch
from . import codex_p3_resolved_host_20261030_v1 as host
from . import p3_online_pair_contract_20261102_codex_exclusive as pair

ROOT = host.ROOT
NAME = "p3_online_pair_sealed_audit_20261102_codex_exclusive"
BATCH = ROOT / "results/codex_p3_resolved_delivery_20261030_v1_RUN_CODEX01"
BATCH_SHA = "f94e83cbfde9927da4710e982f9129a9b90f6df23ec03dbcf3328c5176bdbdf8"
SLOTS = tuple((p, a) for p in (20, 30) for a in ("signal", "reverse"))


def need(value, message):
    if not value:
        raise ValueError(message)


def verify_prepare_isolated(path, expected_sha):
    """Run the original verifier in a clean interpreter to avoid import drift."""
    script = (
        "import sys; from pathlib import Path; "
        "from experiments.spectral_reachability import "
        "codex_p3_resolved_host_20261030_v1 as h; "
        "h.verify_prepare(Path(sys.argv[1]), sys.argv[2]); print('verified')"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(path), expected_sha],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=False,
    )
    need(result.returncode == 0 and result.stdout.strip() == "verified",
         "isolated original prepare verifier failed: " + result.stderr[-1500:])


def build():
    # This verifies the externally pinned batch report and its hash-chained
    # journal, as well as the published per-slot report and receipt hashes.
    batch_audit = batch.build(BATCH, BATCH_SHA)
    need(batch_audit["status"] == "partial_or_failed", "batch status changed")
    batch_report = json.loads((BATCH / "report.json").read_bytes())
    by_slot = {(r["point"], r["arm"]): r for r in batch_report["rows"]}
    need(len(by_slot) == len(batch_report["rows"]), "duplicate batch slot")
    verified = {}
    plan_sha = None
    tasks = banks = plan = None
    evidence = {}
    for point, arm in SLOTS:
        row = by_slot[point, arm]
        need(row["delivered_and_saved"] is True, "missing target slot")
        run = ROOT / "results" / (host.NAME + f"_RUN_WALL_P{point}_{arm}_01")
        receipt = ROOT / "results" / (host.NAME + f"_CONSUME_WALL_P{point}_{arm}_01.json")
        need(row["run"] == str(run) and row["receipt"] == str(receipt)
             and not run.is_symlink() and not receipt.is_symlink(), "batch path binding")
        need(host.prior.sha(run / "report.json") == row["report_sha256"]
             and host.prior.sha(receipt) == row["receipt_sha256"], "report or receipt hash")
        report = json.loads((run / "report.json").read_bytes())
        need(report["status"] == "panel_completed_unverified" and report["error"] is None
             and report["scientific_pass"] is False and report["physical_release"] is False
             and (report["point"], report["arm"]) == (point, arm), "host report")
        current_sha = host.prior.sha(run / "prepare.json")
        need(current_sha == report["prepare_sha256"], "prepare report binding")
        if plan_sha is None:
            plan_sha = current_sha
            verify_prepare_isolated(run / "prepare.json", plan_sha)
            plan = json.loads((run / "prepare.json").read_bytes())
            _, tasks, _ = host.prior.verify(host.PREVIOUS, host.PREVIOUS_SHA)
            banks = host.bridge.banks(host.prior.dependency().old.BALL)
            need(plan["predecessor_sha256"] == host.PREVIOUS_SHA
                 and plan["task_sha256"] == [host.prior.digest(t) for t in tasks]
                 and plan["bank_sha256"] == [host.prior.digest(b) for b in banks],
                 "task/bank reconstruction")
        else:
            need(current_sha == plan_sha, "different prepare or task/bank identity")
        need(report["model_before"] == report["model_after"] == plan["expected_model_hash"],
             "saved model state identity")
        host.identity.verify(report["loaded_python_sources"])
        events = host.journal.read(run / "events.jsonl", report["journal_tail"], host.CAP)
        need(len(events) == report["journal_events"], "journal count")
        checked = pair.consumer.verify_events(events, tasks, banks, point, arm)
        receipt_value = json.loads(receipt.read_bytes())
        need(checked == receipt_value, "saved consumer receipt differs from reconstruction")
        verified[point, arm] = checked
        evidence[f"{point}_{arm}"] = {
            "report": str(run / "report.json"),
            "report_sha256": row["report_sha256"],
            "prepare_sha256": current_sha,
            "receipt_sha256": row["receipt_sha256"],
            "journal_tail": report["journal_tail"],
            "journal_event_count": len(events),
            "runtime_source_files_verified": len(report["loaded_python_sources"]),
            "saved_model_before_after_match": True,
        }
    comparisons = [pair.compare_verified_slots(verified[p, "signal"], verified[p, "reverse"], p)
                   for p in (20, 30)]
    checkpoint = [r for r in plan["sources"]
                  if r["path"] in (str(Path(plan["model_path"]) / "weights.pt"),
                                   str(Path(plan["model_path"]) / "config.json"))]
    need(len(checkpoint) == 2, "checkpoint source records")
    return {
        "schema": NAME + ".v1",
        "status": "paired_saved_semantics_checked_not_scientifically_admitted",
        "batch_report_sha256": BATCH_SHA,
        "batch_status": batch_audit["status"],
        "batch_20_uniform": "failed_or_incomplete; not included in pair comparison",
        "slots": evidence,
        "identity": {
            "common_prepare_sha256": plan_sha,
            "task_sha256": plan["task_sha256"],
            "bank_sha256": plan["bank_sha256"],
            "task_bank_reconstructed_by_host_verify_prepare": True,
            "prepare_verification_process": "original host verifier in clean subprocess; current process additionally reconstructs tasks and banks",
            "checkpoint_files_verified_against_prepare": checkpoint,
            "expected_tensor_state_sha256": plan["expected_model_hash"],
            "tensor_state_evidence": "host report before/after assertions; not independently recomputed",
            "source_inventory_verified_against_current_files": True,
            "dynamic_source_closure": "unknown",
            "independent_model_array_recomputation": False,
            "historical_report_authenticity": "hash bound to pinned local batch report, not external signature",
        },
        "comparisons": comparisons,
        "physical_release": False,
        "scientific_pass": False,
        "P3_admission": False,
        "C3_admission": False,
        "equal_total_compute": False,
        "new_model_execution": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and output.suffix == ".json",
         "exclusive output path")
    print("Reconstructing sealed paired host slots on CPU; no model execution.")
    result = build()
    digest = host.write_new(output, result)
    print(json.dumps({"output": str(output), "sha256": digest, "status": result["status"]}))


if __name__ == "__main__":
    main()
