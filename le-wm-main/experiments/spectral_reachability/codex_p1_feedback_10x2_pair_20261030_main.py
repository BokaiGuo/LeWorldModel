"""Descriptive identity-matched 2x10 feedback versus open 20 comparison."""
import argparse
import json
from pathlib import Path

from . import codex_p1_complete_paired_view_20261030_main as util

ROOT = util.base.ROOT
FEEDBACK = ROOT / "results/codex_p1_feedback_10x2_20261030_main_saved_MAIN01.json"
PAIR = ROOT / "results/codex_p1_feedback_open_paired_20261030_v1_CODEX01.json"
SHAS = ("554c7d6a965916382a8fb1604f0b408882a63411bcd5768f590db6cdd3d172b5",
        "4f5fa2887cfa6eac1f5e99e0fbfb2a64f8077a9c29f337a5cee8ad0be9da98ea")
NAME = "codex_p1_feedback_10x2_pair_20261030_main"
FIELDS = ("object_position_error", "object_angle_error_wrapped",
          "pusher_position_error", "native_joint_xy_error")
IDENTITY = ("index", "logical_pair", "root_slot", "condition", "arm", "role")


def build(feedback, pair):
    if len(feedback["rows"]) != 24 or len(pair["rows"]) != 24:
        raise ValueError("roster incomplete")
    rows = []
    for i, (f, p) in enumerate(zip(feedback["rows"], pair["rows"], strict=True)):
        if f["index"] != i or any(f[k] != p["identity"][k] for k in IDENTITY):
            raise ValueError("identity mismatch")
        a, b = f["endpoint_metrics"], p["open_20"]["endpoint"]
        if f["status"] != "probe_completed" or p["open_20"]["status"] != "probe_completed":
            raise ValueError("this fixed comparison requires both full endpoints")
        if a is None or b is None or len(f["stage_resources"]) != 2:
            raise ValueError("missing full endpoint or stage")
        if any(x["planning"]["charged_upper_bounds"]["backward_calls"] != 10
               for x in f["stage_resources"]):
            raise ValueError("feedback budget drift")
        rows.append({"identity": p["identity"], "feedback": a, "open": b,
                     "feedback_minus_open": {k: a[k] - b[k] for k in FIELDS},
                     "feedback_status": f["status"], "open_status": p["open_20"]["status"],
                     "feedback_stage_resources": f["stage_resources"],
                     "open_stage_resources": p["open_20"]["stage_resources"]})
    counts = {k: sum(r["feedback_minus_open"][k] < 0 for r in rows) for k in FIELDS}
    return {"schema": NAME, "status": "descriptive_known_root_comparison", "rows": rows,
            "summary": {"rows": len(rows), "feedback_lower_error_count": counts,
                        "feedback_native_success": sum(r["feedback"]["native_success"] for r in rows),
                        "open_native_success": sum(r["open"]["native_success"] for r in rows)},
            "sources": {str(path): sha for path, sha in zip((FEEDBACK, PAIR), SHAS, strict=True)},
            "equal_total_budget_certified": False, "independent_confirmation": False, "P1_admission": False,
            "limitations": ["two known roots, repeated trajectories, post-outcome exploratory design",
                            "first stage uses 10 versus 20 updates; comparison does not isolate feedback",
                            "same backward count does not imply equal encoding, forward, wall, or end-to-end cost",
                            "source checks do not establish complete dependency or training-data provenance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != ROOT / "results" or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    sources = [util.gap.read(p, s)[0] for p, s in zip((FEEDBACK, PAIR), SHAS, strict=True)]
    value = build(*sources)
    value["code_sha256"] = util.base.sha(Path(__file__).resolve())
    print(json.dumps({"sha256": util.base.write(out, value), "summary": value["summary"]}))


if __name__ == "__main__":
    main()
