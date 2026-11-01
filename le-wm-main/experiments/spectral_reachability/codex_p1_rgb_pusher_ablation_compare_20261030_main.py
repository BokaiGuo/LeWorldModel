"""Paired comparison of fixed RGB policy variants; censored endpoints remain null."""
import argparse
import json
from pathlib import Path

from . import codex_p1_same_first5_feedback_compare_20261030_main as pair
from . import codex_p1_horizon1_prediction_gap_20261030_main as images
from . import codex_p1_rgb_pusher_ablation_20261030_main as ablation

NAME = "codex_p1_rgb_pusher_ablation_compare_20261030_main"
ROOT = ablation.launch.ROOT
OLD_DIR = ROOT / "results/codex_p1_rgb_feedback_launcher_20261030_v1_run_CODEX01"
OLD_REPORT_SHA = "a1c55ba949850607f979452dc36852b96cdf293cd9b994c6c2f92f24c981e536"
OLD_SAVED = ROOT / "results/codex_p1_rgb_feedback_saved_20261030_v1_CODEX01.json"
OLD_SAVED_SHA = "a5c96489369da4b12ce85fc4acbaee36b032161de718b6bfb8d66ec880cb8113"
FIELDS = pair.FIELDS
need = pair.need


def compare(old, new, old_report, new_report, load):
    need(all(len(x["rows"]) == 24 for x in (old, new))
         and len(old_report["shards"]) == len(new_report["shards"]) == 24, "fixed roster")
    rows = []
    for i, (a, b) in enumerate(zip(old["rows"], new["rows"], strict=True)):
        need(a["index"] == b["index"] == i and all(a[k] == b[k] for k in pair.IDS), "row identity")
        s1, s2 = old_report["shards"][i], new_report["shards"][i]
        need(s1["index"] == s2["index"] == i, "shard identity")
        ta, tb, same_pixels = load(i, s1, s2, a["condition"])
        matched = len(ta) >= 5 and len(tb) >= 5
        if matched:
            for j in range(5):
                need(ta[j]["step"] == tb[j]["step"] == j + 1
                     and all(ta[j][k] == tb[j][k] for k in
                             ("actual_bits", "state", "snapshot", "terminated", "truncated")),
                     "first five mismatch")
            need(same_pixels, "midpoint RGB mismatch")
        else:
            need(not same_pixels, "short trajectory has no authenticated midpoint")
        x, y = a["endpoint_metrics"], b["endpoint_metrics"]
        delta = None if x is None or y is None else {k: y[k] - x[k] for k in FIELDS}
        rows.append({"identity": {k: a[k] for k in pair.IDS},
                     "common_first5_and_rgb": matched,
                     "old_status": a["status"], "new_status": b["status"],
                     "old_decision": a["rgb_decision"], "new_decision": b["rgb_decision"],
                     "old_steps": a["accepted_suffix_steps"], "new_steps": b["accepted_suffix_steps"],
                     "old_endpoint": x, "new_endpoint": y, "new_minus_old": delta})
    comparable = [r for r in rows if r["new_minus_old"] is not None]
    return {"schema": NAME, "status": "descriptive_known_root", "rows": rows,
            "summary": {"rows": len(rows), "common_midpoint": sum(r["common_first5_and_rgb"] for r in rows),
                        "comparable": len(comparable),
                        "new_corrections": sum(r["new_decision"] == "rgb_pusher_correction" for r in rows),
                        "new_rejections": sum(r["new_status"] == "rgb_rejected" for r in rows),
                        "old_rejections": sum(r["old_status"] == "rgb_rejected" for r in rows),
                        "new_lower_counts": {k: sum(r["new_minus_old"][k] < 0 for r in comparable)
                                             for k in FIELDS},
                        "old_success": sum(bool(r["old_endpoint"]["native_success"]) for r in comparable),
                        "new_success": sum(bool(r["new_endpoint"]["native_success"]) for r in comparable)},
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": ["fixed repeated known-root trajectories, post-outcome development",
                            "censored endpoint excluded from endpoint difference, retained in row",
                            "no total resource fairness or P1 capability claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("directory", "report-sha", "saved", "saved-sha", "output"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    directory, output = Path(args.directory), Path(args.output)
    need(directory.is_absolute() and directory.parent == ROOT / "results"
         and output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
         "exclusive inputs/output")
    old, new = pair.read(OLD_SAVED, OLD_SAVED_SHA), pair.read(Path(args.saved), args.saved_sha)
    ro, rn = pair.read(OLD_DIR / "run_report.json", OLD_REPORT_SHA), pair.read(directory / "run_report.json", args.report_sha)

    def load(i, a, b, condition):
        first, second = OLD_DIR / f"row_{i:02d}" / "physical", directory / f"row_{i:02d}" / "physical"
        t1, t2 = pair.transitions(first, a["report_sha256"]), pair.transitions(second, b["report_sha256"])
        if len(t1) < 5 or len(t2) < 5:
            return t1, t2, False
        r1, r2 = pair.read(first / "report.json", a["report_sha256"]), pair.read(second / "report.json", b["report_sha256"])
        return t1, t2, images.frame(first, r1, condition, 5)[1] == images.frame(second, r2, condition, 5)[1]

    result = compare(old, new, ro, rn, load)
    result["sources"] = {str(OLD_SAVED): OLD_SAVED_SHA, str(OLD_DIR / "run_report.json"): OLD_REPORT_SHA,
                         args.saved: args.saved_sha, str(directory / "run_report.json"): args.report_sha}
    result["code_sha256"] = ablation.launch.sha(Path(__file__).resolve())
    print(json.dumps({"sha256": ablation.launch.h10._write_new(output, result), "summary": result["summary"]}))


if __name__ == "__main__":
    main()
