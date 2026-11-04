"""Join newly fixed winners to already saved physical records, without replay."""
import argparse
import json
from pathlib import Path

from . import p3_preoutcome_winner_physical_20261102_codexexclusive as new
from . import p3_preoutcome_winner_physical_saved_20261102_codexexclusive as saved
from . import p3_venv_winner_cpu_physical_20261102_codexexclusive as old

NEW_PREP = new.ROOT / "results/p3_preoutcome_winner_physical_20261102_codexexclusive_PREP_02.json"
NEW_SHA = "a633a07eb7beae33227ec7f86dd176ccf703a4e45620573ef207e5971d8f45e9"
OLD_PREP = new.ROOT / "results/p3_venv_winner_cpu_physical_20261102_codexexclusive_PREP_01.json"
OLD_SHA = "7d5cb06276b9c1fb2d86344a38f3baf4f67b08a6740a60bc321e3e033033df08"
OLD_RUN = new.ROOT / "results/p3_venv_winner_cpu_physical_20261102_codexexclusive_RUN_01"
OLD_REPORT_SHA = "672b020c82b289481f528c23722a41741b7fed72e7f7407e039492d8e5f4335f"
OLD_JOURNAL_SHA = "a656d5048e0b7f4e157681b8062aa9b74fd6f006e0dfac1700112ac068bc78e7"
NAME = "p3_preoutcome_physical_reuse_audit_20261102_codexexclusive"


def build():
    plan, _ = new.verify(NEW_PREP, NEW_SHA)
    old_plan, source = old.verify(OLD_PREP, OLD_SHA)
    new.require(new.moving.sha(OLD_RUN / "report.json") == OLD_REPORT_SHA
                and new.moving.sha(OLD_RUN / "journal.jsonl") == OLD_JOURNAL_SHA, "old saved bytes drift")
    report = new.moving.read(OLD_RUN / "report.json")
    events = new.moving.journal(OLD_RUN / "journal.jsonl", report["journal_chain_tail"])
    old_plan["model_accounting"] = {"scope": "old plan physical verification only"}
    events[0]["prepare"] = old_plan
    checked = saved.validate(old_plan, source["references"], report, events, OLD_RUN)
    new.require(checked["status"] == "saved_physical_records_verified", "old physical records incomplete")
    mapped = []
    for item in plan["plans"]:
        matches = []
        for index, (prior, row) in enumerate(zip(old_plan["plans"], report["rows"], strict=True)):
            if (item["root_sha256"] == prior["root_sha256"]
                    and item["raw_float32_sha256"] == prior["raw_float32_sha256"]
                    and item["commands"] == prior["commands"]):
                new.require(row["initial_exact"] and row["status"] == "horizon_completed"
                            and row["own_returned"] == 10, "old physical branch incomplete")
                matches.append({"old_branch": index, "old_arm": prior["arm"],
                                "old_candidate_index": prior["candidate_index"],
                                "final_metrics": row["final_metrics"]})
        new.require(matches, "new winner not represented by old physical record")
        mapped.append({"arm": item["arm"], "root_slot": item["root_slot"],
                       "candidate_index": item["candidate_index"], "root_sha256": item["root_sha256"],
                       "raw_float32_sha256": item["raw_float32_sha256"], "old_matches": matches})
    return {"status": "all_new_winners_reuse_existing_physical_inputs",
            "new_prepare_sha256": NEW_SHA, "old_prepare_sha256": OLD_SHA,
            "old_report_sha256": OLD_REPORT_SHA, "old_journal_sha256": OLD_JOURNAL_SHA,
            "model_outcomes_open_before_physical_contract": True, "mapped_slots": mapped,
            "new_physical_execution": False, "new_physical_observations": 0,
            "scope": "saved root/action identity and old physical records, not a new experiment",
            "old_physics_independently_replayed": False, "equal_total_compute": False,
            "independent_outcome": False, "P3_admission": False, "C3_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    path = Path(a.output)
    new.require(path.is_absolute() and path.parent == new.ROOT / "results"
                and path.name.startswith(NAME + "_") and not path.exists(), "new audit output")
    result = build()
    print(json.dumps({"status": result["status"], "path": str(path),
                      "sha256": new.producer.host.write_new(path, result)}))


if __name__ == "__main__":
    main()
