"""Saved model-output consumer and fixed physical-handoff verifier; no execution."""
import argparse
import json
import math
from pathlib import Path

from . import p2_multi_joint_development_20261102_codexexclusive as multi


def consume(folder, report_sha):
    folder = Path(folder)
    multi.need(folder.is_absolute() and folder.parent == multi.RESULTS and not folder.is_symlink()
               and folder.name.startswith(multi.NAME + "_RUN_"), "run directory")
    report = multi.read_checked(folder / "report.json", report_sha)
    binding = json.loads((folder / "binding.json").read_bytes())
    plan = multi.verify_plan(binding["plan"], binding["plan_sha256"])
    multi.need(report["schema"] == multi.NAME + ".report_v1" and report["plan_sha256"] == binding["plan_sha256"]
               and report["status"] == "complete_model_only_physical_blocked" and report["error"] is None
               and report["frozen_parameters_unchanged"] is True and report["device"]["uuid"] == multi.host.UUID
               and all(report[k] is False for k in ("physical_execution", "historical_sra_authenticated",
                                                      "equal_total_compute", "independent_confirmation", "P2_admission"))
               and type(report["wall_seconds"]) in (float, int) and math.isfinite(report["wall_seconds"])
               and report["wall_seconds"] >= 0, "report")
    cases, bank = multi.input_cases()
    multi.need([r["base"] for r in report["rows"]] == list(multi.CASES), "root census")
    summaries = []
    names = {"binding.json", "report.json"}
    for row, case in zip(report["rows"], cases, strict=True):
        base = case["base"]
        jname, rname, hname = (f"root_{base}_journal.jsonl", f"root_{base}_result.json",
                                f"root_{base}_handoff.json")
        names.update((jname, rname, hname))
        multi.need(row["status"] == "complete" and row["root"] == case["root"]
                   and multi.host.base.file_sha(folder / jname) == row["journal_sha256"], "root receipt")
        result = multi.read_checked(folder / rname, row["result_sha256"])
        handoff = multi.read_checked(folder / hname, row["handoff_sha256"])
        multi.need(result["schema"] == multi.NAME + ".root_v1"
                   and result["control_method"] == "joint_gauss_newton_not_historical_sra"
                   and all(result[k] is False for k in ("equal_total_compute", "independent_confirmation", "P2_admission"))
                   and result["counts"]["attempted"] == result["counts"]["returned"] == multi.matched.CAPS
                   and result["counts"]["equal_total_compute"] is False, "root result")
        audit = multi.consumer.audit_complete(folder / jname, result, bank)
        multi.need(audit == row["audit"] and handoff == multi.physical_handoff(case, result), "audit/handoff")
        summaries.append({"base": base, "root": case["root"], "task_id": case["task_id"],
                          "result_sha256": row["result_sha256"], "handoff_sha256": row["handoff_sha256"],
                          "audit": audit, "physical_ready": False})
    multi.need(set(x.name for x in folder.iterdir()) == names and
               all(x.is_file() and not x.is_symlink() for x in folder.iterdir()), "output inventory")
    multi.need(multi.host.base.file_sha(folder / "report.json") == report_sha, "report drift")
    return {"schema": multi.NAME + ".consume_v1", "status": "verified_model_only_physical_blocked",
            "report_sha256": report_sha, "plan_sha256": binding["plan_sha256"], "roots": summaries,
            "physical_execution": False, "historical_sra_authenticated": False, "equal_total_compute": False,
            "independent_confirmation": False, "P2_admission": False,
            "limitations": plan["source_limits"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = multi.path_new(a.output, "_CONSUME_")
    value = consume(a.run, a.report_sha256)
    print(json.dumps({"receipt": str(out), "sha256": multi.host.base.write_new(out, value)}))


if __name__ == "__main__":
    main()
