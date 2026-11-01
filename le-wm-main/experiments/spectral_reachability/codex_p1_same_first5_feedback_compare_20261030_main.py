"""Compare feedback vs open after verifying identical first five physical transitions."""
import argparse
import json
from pathlib import Path

from . import codex_p1_open10_matched_feedback_20261030_main as replay
from . import p1_backtrack_physical_consume_20261030 as journal

ROOT = replay.ROOT
NAME = "codex_p1_same_first5_feedback_compare_20261030_main"
OPEN_DIR = ROOT/"results/codex_p1_open10_matched_feedback_20261030_main_RUN01"
OPEN_REPORT = "13ffc1dcb17ec0b4560cec7f44af9b6dc2376663cffab675e7bb142118db5c21"
OPEN_SAVED = ROOT/"results/codex_p1_open10_matched_feedback_20261030_main_saved_MAIN01.json"
OPEN_SHA = "ad907ca79c5391786d34ba69df3276fbce9f80ddff33351069381eac2353c7e9"
FB_SAVED = ROOT/"results/codex_p1_feedback_10x2_20261030_main_saved_MAIN01.json"
FB_SHA = "554c7d6a965916382a8fb1604f0b408882a63411bcd5768f590db6cdd3d172b5"
FIELDS = ("object_position_error", "object_angle_error_wrapped",
          "pusher_position_error", "native_joint_xy_error")
IDS = ("index", "logical_pair", "root_slot", "condition", "arm", "role")
need = replay.need


def read(path, sha):
    return replay.physical.gap.read(path, sha)[0]


def transitions(folder, report_sha):
    report = read(folder/"report.json", report_sha)
    path = folder/"journal.jsonl"
    need(replay.physical.old.sha(path) == report["journal_sha256"], "journal file hash")
    events = journal.events(path, report["journal_chain_tail"])
    return [e for e in events if e["kind"] == "returned_transition" and e["phase"] == "probe"]


def compare(open_saved, fb_saved, open_report, fb_report, get_transitions):
    need(all(len(x["rows"]) == 24 for x in (open_saved, fb_saved))
         and len(open_report["shards"]) == len(fb_report["shards"]) == 24, "roster")
    rows = []
    for i, (a, b) in enumerate(zip(open_saved["rows"], fb_saved["rows"], strict=True)):
        need(all(a[k] == b[k] for k in IDS) and a["index"] == i, "identity")
        sa, sb = open_report["shards"][i], fb_report["shards"][i]
        need(sa["index"] == sb["index"] == i, "shard identity")
        ta, tb = get_transitions(i, sa, sb)
        need(len(ta) >= 5 and len(tb) >= 5, "early stop before common prefix")
        for j in range(5):
            need(ta[j]["step"] == tb[j]["step"] == j+1 and all(
                ta[j][key] == tb[j][key] for key in
                ("actual_bits", "state", "snapshot", "terminated", "truncated")), "first five differ")
        x, y = a["endpoint_metrics"], b["endpoint_metrics"]
        difference = None if x is None or y is None else {k: y[k]-x[k] for k in FIELDS}
        rows.append({"identity": {k: a[k] for k in IDS}, "common_steps": 5,
                     "open_status": a["status"], "feedback_status": b["status"],
                     "open_steps": a["returned_suffix"], "feedback_steps": b["accepted_suffix_steps"],
                     "open_endpoint": x, "feedback_endpoint": y, "feedback_minus_open": difference})
    comparable = [r for r in rows if r["feedback_minus_open"] is not None]
    return {"schema": NAME, "status": "known_root_matched_replay", "rows": rows,
            "summary": {"rows": len(rows), "common_first5_verified": len(rows), "comparable": len(comparable),
                        "feedback_lower_counts": {k: sum(r["feedback_minus_open"][k] < 0 for r in comparable)
                                                  for k in FIELDS},
                        "open_native_success": sum(bool(r["open_endpoint"]["native_success"]) for r in comparable),
                        "feedback_native_success": sum(bool(r["feedback_endpoint"]["native_success"]) for r in comparable)},
            "P1_admission": False, "equal_total_budget_certified": False, "independent_confirmation": False,
            "limitations": ["known roots, repeated trajectories, post-outcome development",
                            "feedback performs additional 10-backward optimization and one encoding",
                            "same first five does not isolate observation from additional optimization",
                            "same initial stage used saved plans; open replay adds no model inference"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT/"results" and out.name.startswith(NAME+"_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    op, fb = read(OPEN_SAVED, OPEN_SHA), read(FB_SAVED, FB_SHA)
    orp, frp = read(OPEN_DIR/"run_report.json", OPEN_REPORT), read(replay.FB_DIR/"run_report.json", replay.FB_REPORT_SHA)
    def load(i, a, b):
        return (transitions(OPEN_DIR/f"row_{i:02d}", a["report_sha256"]),
                transitions(replay.FB_DIR/f"row_{i:02d}"/"physical", b["report_sha256"]))
    result = compare(op, fb, orp, frp, load)
    result["sources"] = {str(OPEN_SAVED): OPEN_SHA, str(FB_SAVED): FB_SHA,
                         str(OPEN_DIR/"run_report.json"): OPEN_REPORT,
                         str(replay.FB_DIR/"run_report.json"): replay.FB_REPORT_SHA}
    result["code_sha256"] = replay.physical.old.sha(Path(__file__).resolve())
    print(json.dumps({"sha256": replay.physical.old.h10._write_new(out, result), "summary": result["summary"]}))


if __name__ == "__main__":
    main()
