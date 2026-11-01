"""Pair zero-penalty and trust-penalty best-iterate physical executions."""
import json
from pathlib import Path

from . import codex_p1_adam_best_iterate_compare_20261030_main as previous
from . import codex_p1_unreg_best_physical_20261030_main as physical

compare = previous.compare
ROOT = compare.base.ROOT
NAME = "codex_p1_unreg_best_compare_20261030_main"
REPORT_SHA = "83099e4b6334086c0550dfbb6e5fd16a102a544c2f1ffeb54cd78bfebc12df4a"
SAVED_SHA = "a2139af7ba866b9de3b6ade2ebbba8e8af380f6947bcd798aa02a13c0ace097f"
OLD_UNREG = compare.plan.PLAN if hasattr(compare.plan, "PLAN") else ROOT / "results/codex_p1_adam_feedback_plan_20261030_main_RUN01.json"
OLD_UNREG_SHA = "3fe7a35c3b42600a29d74423cdd80cf75a85499a54b31e276dd03261b4ae50c5"


def build():
    prior = previous.build()
    compare.DIR = ROOT / f"results/{physical.NAME}_RUN01"
    compare.REPORT_SHA = REPORT_SHA
    compare.SAVED = ROOT / f"results/{physical.NAME}_SAVED_RUN01.json"
    compare.SAVED_SHA = SAVED_SHA
    current = compare.build()
    read = compare.plan.gap.prior.gap.read
    unreg, _ = read(physical.PLAN, physical.SHA)
    old, _ = read(OLD_UNREG, OLD_UNREG_SHA)
    trust, _ = read(previous.physical.PLAN, previous.physical.SHA)
    if unreg["penalty_weight"] != 0 or trust["penalty_weight"] != 0.1:
        raise ValueError("penalty")
    if any(len(x) != 24 for x in (prior["rows"], current["rows"], unreg["rows"], old["rows"], trust["rows"])):
        raise ValueError("roster")
    rows = []
    for i, (a, b, u, o, t) in enumerate(zip(prior["rows"], current["rows"], unreg["rows"],
                                             old["rows"], trust["rows"], strict=True)):
        if any(x["index"] != i for x in (a, b, u, o, t)):
            raise ValueError("identity")
        if a["root_slot"] != b["root_slot"] or a["condition"] != b["condition"] or a["role"] != b["role"]:
            raise ValueError("task identity")
        if u["first_packet_hex"] != t["first_packet_hex"] or u["midpoint_pixel_sha256"] != t["midpoint_pixel_sha256"]:
            raise ValueError("input identity")
        if abs(u["optimized_score"]-o["optimized_score"]) > 1e-6 or abs(u["baseline_score"]-o["baseline_score"]) > 1e-6:
            raise ValueError("historical unregularized path mismatch")
        x, y = a["new_endpoint"], b["feedback_endpoint"]
        delta = None if x is None or y is None else {k: y[k]-x[k] for k in compare.FIELDS}
        rows.append({"index": i, "root_slot": a["root_slot"], "condition": a["condition"], "role": a["role"],
                     "trust_status": a["new_status"], "unreg_status": b["feedback_status"],
                     "trust_success": a["new_last_success"], "unreg_success": b["feedback_last_success"],
                     "trust_steps": a["new_steps"], "unreg_steps": b["feedback_steps"],
                     "same_suffix": u["second_packet_hex"] == t["second_packet_hex"],
                     "unreg_selected_step": u["selected_step"], "trust_endpoint": x, "unreg_endpoint": y,
                     "unreg_minus_trust": delta})
    usable = [r for r in rows if r["unreg_minus_trust"] is not None]
    return {"schema": NAME, "status": "complete", "rows": rows,
            "summary": {"fixed": 24, "same_suffix": sum(r["same_suffix"] for r in rows),
                        "trust_success": sum(r["trust_success"] for r in rows),
                        "unreg_success": sum(r["unreg_success"] for r in rows), "comparable": len(usable),
                        "unreg_lower": {k: sum(r["unreg_minus_trust"][k] < -1e-8 for r in usable)
                                        for k in compare.FIELDS},
                        "mean_delta": {k: sum(r["unreg_minus_trust"][k] for r in usable)/len(usable)
                                       for k in compare.FIELDS}},
            "plan_sha256": physical.SHA, "trust_plan_sha256": previous.physical.SHA,
            "physical_report_sha256": REPORT_SHA, "saved_sha256": SAVED_SHA,
            "code_sha256": compare.base.sha(Path(__file__).resolve()),
            "P1_admission": False, "independent_confirmation": False,
            "limitations": "known roots and repeated trajectories; offline midpoint; censored endpoints excluded"}


if __name__ == "__main__":
    result = build()
    out = ROOT / f"results/{NAME}_RUN01.json"
    print(json.dumps({"sha256": compare.base.write(out, result), "summary": result["summary"]}), flush=True)
