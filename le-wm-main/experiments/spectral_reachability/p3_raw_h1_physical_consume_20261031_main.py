"""Verify completed six-slot batches and report partial batches without imputation."""
import hashlib
import json
from pathlib import Path

from . import p3_raw_h1_physical_20261031_main as initial
from . import p3_raw_h1_physical_continue_20261031_main as continuation
from . import codex_p3_online_budget_physical_saved_20261030_v1 as saved

host, ROOT, require = initial.host, initial.ROOT, initial.require
LATER = ROOT / "results/p3_raw_h1_physical_continue_20261031_main_RUN_01"
LATER_SHA = "626553f6852704c076ebfc4bf56f243166d7dcc903a4ab22cdd41d5aa7790b29"
OUT = ROOT / "results/p3_raw_h1_physical_consume_20261031_main_RUN_01.json"


def run():
    require(host.sha(continuation.PREVIOUS / "batch_report.json") == continuation.PREVIOUS_SHA
            and host.sha(LATER / "batch_report.json") == LATER_SHA, "batch authorities")
    first = json.loads((continuation.PREVIOUS / "batch_report.json").read_bytes())
    later = json.loads((LATER / "batch_report.json").read_bytes())
    records = first["batches"] + later["batches"]
    require([r["batch"] for r in records] == list(range(12)), "complete attempt roster")
    plans, bindings, contract, data = initial.prepare()
    complete, partial = [], []
    for record in records:
        batch = record["batch"]
        directory = (continuation.PREVIOUS if batch < 2 else LATER) / f"batch_{batch:02}"
        require(host.sha(directory / "report.json") == record["report_sha256"], "report hash")
        report = json.loads((directory / "report.json").read_bytes())
        p = json.loads((directory / "prepare.json").read_bytes())
        require(p["plans"] == plans[6*batch:6*batch+6] and p["contract"] == contract
                and all(p["bindings"].get(k) == v and host.sha(k) == v for k, v in bindings.items()),
                "source and plans")
        if report["status"] != "recorded_not_admitted":
            partial.append({"batch": batch, "report_sha256": record["report_sha256"],
                            "rows": [{"arm": r["arm"], "status": r["status"],
                                      "returned": r["own_returned"], "error": r["first_error"]}
                                     for r in report["rows"]]})
            continue
        host.NAME = (initial.NAME if batch < 2 else continuation.NAME) + f"_B{batch:02}"
        digest = hashlib.sha256((directory / "prepare.json").read_bytes()).hexdigest()
        host.verify = lambda path, expected: (p, data) if host.sha(path) == expected == digest else (
            (_ for _ in ()).throw(ValueError("prepare hash")))
        receipt = saved.consume(directory, record["report_sha256"], directory / "prepare.json", digest)
        require(receipt["status"] == "saved_semantics_checked_not_scientifically_admitted",
                "saved consumer")
        complete.extend({"batch": batch, "plan": plan, "status": row["status"],
                         "returned": row["own_returned"], "success": row["native_success"],
                         "final_metrics": row["final_metrics"] if row["full_horizon"] else None}
                        for plan, row in zip(p["plans"], report["rows"], strict=True))
    result = {"schema": "p3_raw_h1_physical_consume_20261031_main",
              "complete_batch_rows": complete, "partial_batches": partial,
              "complete_batch_count": len(complete) // 6, "fixed_branches": 72,
              "partial_batch_rows_not_independently_consumed": True, "P1_admission": False,
              "P3_admission": False, "independent_confirmation": False,
              "equal_budget_certified": False}
    digest = host.bind.host.write_new(OUT, result)
    print(json.dumps({"sha256": digest, "complete_rows": len(complete),
                      "partial_batches": len(partial)}))


if __name__ == "__main__":
    run()
