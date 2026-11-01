"""Summarize the sealed joint physical run without changing any gate."""
import json
from pathlib import Path

from . import p1_joint_physical_20261030 as physical
from . import codex_p1_h10_paired_table_20261030_main as metrics

ROOT = physical.ROOT
DIRECTORY = ROOT / "results/p1_joint_physical_20261030_RUN01"
REPORT_SHA = "3ff76885ec7eaf68411355ac278b658cabd22b10b4940bdbf73e7d12c48ff404"
JOIN_SHA = "c951fbe737cf8e38b3e13341aeaed80968a8f866c6eef78fdf29f67badd6e908"
OUT = ROOT / "results/p1_joint_physical_analysis_20261030_RUN01.json"


def main():
    r = physical.runner
    r.require(r.sha(DIRECTORY / "report.json") == REPORT_SHA, "report hash")
    report = r.native.read(DIRECTORY / "report.json", cap=64 * 1024**2)
    binding = report["execution_binding"]
    prepared, _ = physical.verify(binding["prepared"], binding["prepare_sha256"], JOIN_SHA)
    receipt = physical.consumer.check(DIRECTORY, report, prepared)
    r.require(receipt["status"] == "saved_checks_passed_not_admitted" and len(report["rows"]) == 16,
              "saved receipt")
    pairs = []
    for i in range(8):
        a, b = report["rows"][2*i:2*i+2]
        r.require(a["role"] == "incumbent" and b["role"] == "model_selected"
                  and all(a[k] == b[k] for k in ("root_slot", "condition", "arm", "task_id", "goal")),
                  "paired identity")
        entries = []
        for row in (a, b):
            cp = {item["step"]: item["state"] for item in row["checkpoints"]}
            root = metrics.physical_metrics(cp.get(0), row["goal"])
            prefix = metrics.physical_metrics(cp.get(5), row["goal"])
            end = metrics.physical_metrics(cp.get(15), row["goal"])
            entries.append({"role": row["role"], "status": row["status"], "accepted_steps": row["own_returned"],
                            "root": root, "prefix": prefix, "prefix_minus_root": metrics.paired_difference(prefix, root),
                            "endpoint": end, "missing_checkpoints": row["missing_checkpoints"]})
        pairs.append({"root_slot": a["root_slot"], "condition": a["condition"], "arm": a["arm"],
                      "task_id": a["task_id"], "incumbent": entries[0], "selected": entries[1],
                      "selected_minus_incumbent_endpoint": metrics.paired_difference(
                          entries[1]["endpoint"], entries[0]["endpoint"])})
    result = {"schema": "p1_joint_physical_analysis_20261030", "report_sha256": REPORT_SHA,
              "analysis_code_sha256": r.sha(Path(__file__)), "metric_code_sha256": r.sha(Path(metrics.__file__)),
              "saved_check": receipt, "pairs": pairs, "independent_roots": False,
              "equal_budget_certified": False, "capability_probe": False, "P1_admission": False}
    r.require(not OUT.exists(), "exclusive output")
    print(json.dumps({"sha256": replay_write(OUT, result), "pairs": len(pairs)}))


def replay_write(path, value):
    return physical.replay.old.one.saved.write(path, value)


if __name__ == "__main__":
    main()
