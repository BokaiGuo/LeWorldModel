"""Analyze all paired physical outcomes after full saved-state verification."""
import argparse
import json
from pathlib import Path

from . import p1_uphill_physical_20261030 as physical
from . import codex_p1_h10_paired_table_20261030_main as metric

ROOT = physical.ROOT
DIRECTORY = ROOT / "results/p1_uphill_physical_20261030_RUN01"
JOIN_SHA = "2f8c75e502794248c399385e01db5cc991fb69b1eab7f34e4676becd1fe04e81"


def analyze(report, receipt):
    need = physical.need
    need(receipt["status"] == "saved_checks_passed_not_admitted"
         and len(report["rows"]) == 16, "complete verified report")
    pairs = []
    for i in range(8):
        original, selected = report["rows"][2*i:2*i+2]
        need(original["role"] == "incumbent" and selected["role"] == "model_selected"
             and all(original[k] == selected[k] for k in
                     ("root_slot", "condition", "arm", "task_id", "goal")), "paired identity")
        outputs = []
        for row in (original, selected):
            cp = {x["step"]: x["state"] for x in row["checkpoints"]}
            need(len(cp) == len(row["checkpoints"]), "duplicate checkpoint")
            root = metric.physical_metrics(cp.get(0), row["goal"])
            prefix = metric.physical_metrics(cp.get(5), row["goal"])
            endpoint = metric.physical_metrics(cp.get(15), row["goal"])
            need((endpoint is not None) == (row["own_returned"] == 15), "endpoint censorship")
            outputs.append({"role": row["role"], "status": row["status"],
                            "stop_reason": row.get("stop_reason"), "returned_steps": row["own_returned"],
                            "missing_checkpoints": row["missing_checkpoints"], "root": root,
                            "prefix": prefix, "prefix_minus_root": metric.paired_difference(prefix, root),
                            "endpoint": endpoint, "raw15_sha256": row["plan"]["raw15_sha256"]})
        pairs.append({"root_slot": original["root_slot"], "condition": original["condition"],
                      "arm": original["arm"], "task_id": original["task_id"],
                      "original_joint_winner": outputs[0], "latent_rerank": outputs[1],
                      "same_actions": outputs[0]["raw15_sha256"] == outputs[1]["raw15_sha256"],
                      "rerank_minus_original_endpoint": metric.paired_difference(outputs[1]["endpoint"],
                                                                                   outputs[0]["endpoint"])})
    return {"schema": "p1_uphill_physical_analysis_20261030", "pairs": pairs, "saved_check": receipt,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitation": "Known roots, repeated arms, open action chains. Both prefix and suffix can differ."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    need = physical.need
    need(out.is_absolute() and out.parent == ROOT / "results"
         and out.name.startswith("p1_uphill_physical_analysis_") and not out.exists()
         and not out.is_symlink(), "exclusive output")
    path = DIRECTORY / "report.json"
    need(physical.sha(path) == args.report_sha, "report SHA")
    report = physical.runner.native.read(path, cap=64*1024**2)
    binding = report["execution_binding"]
    prepared, _ = physical.verify(binding["prepared"], binding["prepare_sha256"], JOIN_SHA)
    need(binding == physical.runner.native.read(DIRECTORY / "execution_binding.json")
         and binding["join_sha256"] == JOIN_SHA, "execution binding")
    receipt = physical.consumer.check(DIRECTORY, report, prepared)
    result = analyze(report, receipt)
    need(physical.sha(path) == args.report_sha, "report drift")
    result.update(report_sha256=args.report_sha, join_sha256=JOIN_SHA,
                  analysis_sha256=physical.sha(Path(__file__)),
                  metric_sha256=physical.sha(Path(metric.__file__)))
    digest = physical.replay.engine.old.one.saved.write(out, result)
    print(json.dumps({"sha256": digest, "pairs": len(result["pairs"])}))


if __name__ == "__main__":
    main()
