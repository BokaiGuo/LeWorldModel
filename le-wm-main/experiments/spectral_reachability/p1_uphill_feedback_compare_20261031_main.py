"""Compare feedback vs open only after verifying identical first five transitions."""
import argparse
import json
from pathlib import Path

from . import codex_p1_h10_paired_table_20261030_main as metric
from . import codex_p1_same_first5_feedback_compare_20261030_main as traces
from . import p1_uphill_midpoint_feedback_physical_20261031_main as feedback
from . import p1_uphill_multistart_rgb_20261030_main as open_run

ROOT = feedback.ROOT
NAME = "p1_uphill_feedback_compare_20261031_main"
FB_DIR = ROOT / "results/p1_uphill_midpoint_feedback_physical_20261031_main_run_RUN01"
FB_REPORT = "e070b0c94ae96dad201a76de36a2de75d943a6526706fca0d449f2e336911727"
FB_SAVED = ROOT / "results/p1_uphill_midpoint_feedback_physical_20261031_main_consume_RUN01.json"
FB_SAVED_SHA = "b1e2792472db637e826e120fe170638bdb7d865a3fd22dab2949fc8517ef1c9e"
need = feedback.need


def compare(anchors, opened, refreshed, open_report, fb_report, get_transitions):
    need(all(len(x["rows"]) == 24 for x in (opened, refreshed))
         and len(anchors["anchors"]) == len(open_report["shards"]) == len(fb_report["shards"]) == 24,
         "roster")
    rows = []
    for i, (anchor, a, b, oa, fb) in enumerate(zip(
            anchors["anchors"], opened["rows"], refreshed["rows"], open_report["shards"],
            fb_report["shards"], strict=True)):
        need(anchor["index"] == a["index"] == b["index"] == oa["index"] == fb["index"] == i
             and all(a[k] == b[k] == anchor[k] for k in ("root_slot", "condition", "arm", "role")),
             "identity")
        old_trans, new_trans = get_transitions(i, oa, fb)
        need(len(old_trans) >= 5 and len(new_trans) >= 5, "early before midpoint")
        for j, (x, y) in enumerate(zip(old_trans[:5], new_trans[:5], strict=True), 1):
            need(x["step"] == y["step"] == j and all(
                x[k] == y[k] for k in ("actual_bits", "state", "snapshot", "terminated", "truncated")),
                "first five differ")
        old_endpoint = metric.physical_metrics(a["completed_endpoint"], anchor["goal"])
        new_endpoint = metric.physical_metrics(b["completed_endpoint"], anchor["goal"])
        need((old_endpoint is not None) == (a["status"] == "probe_completed")
             and (new_endpoint is not None) == (b["status"] == "probe_completed"), "endpoint censoring")
        rows.append({"index": i, "logical_pair": anchor["logical_pair"], "root_slot": anchor["root_slot"],
                     "condition": anchor["condition"], "arm": anchor["arm"], "role": anchor["role"],
                     "first5_exact": True, "open_status": a["status"], "feedback_status": b["status"],
                     "open_returned_suffix": a["returned_suffix"], "feedback_returned_suffix": b["returned_suffix"],
                     "open_last_success": None if a["last_accepted_state_metrics"] is None
                     else a["last_accepted_state_metrics"]["native_success"],
                     "feedback_last_success": None if b["last_accepted_state_metrics"] is None
                     else b["last_accepted_state_metrics"]["native_success"],
                     "open_endpoint": old_endpoint, "feedback_endpoint": new_endpoint,
                     "feedback_minus_open": metric.paired_difference(new_endpoint, old_endpoint)})
    return {"schema": NAME, "rows": rows, "P1_admission": False, "independent_confirmation": False,
            "equal_total_budget_certified": False,
            "limitations": "Known roots and repeated arms; feedback has extra observation/optimization; "
                           "early stopped horizon endpoints remain null."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results"
         and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(), "output")
    inputs = ((open_run.paired.ANCHORS, open_run.paired.ANCHORS_SHA), (open_run.paired.CONSUME, open_run.paired.CONSUME_SHA),
              (FB_SAVED, FB_SAVED_SHA), (open_run.RUN / "run_report.json", open_run.REPORT_SHA),
              (FB_DIR / "run_report.json", FB_REPORT))
    anchors, opened, refreshed, op_report, fb_report = [open_run.paired.read(*x) for x in inputs]

    def transitions(i, a, b):
        return (traces.transitions(open_run.RUN / f"row_{i:02d}", a["report_sha256"]),
                traces.transitions(FB_DIR / f"row_{i:02d}", b["report_sha256"]))

    result = compare(anchors, opened, refreshed, op_report, fb_report, transitions)
    result["sources"] = {str(path): digest for path, digest in inputs}
    result["analysis_sha256"] = feedback.sha(Path(__file__).resolve())
    need(all(feedback.sha(path) == digest for path, digest in result["sources"].items()), "source drift")
    print(json.dumps({"sha256": feedback.original.physical.design.run._write_new(out, result),
                      "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
