"""Pair the guarded best-iterate physical execution with trust and open plans."""
import json
from pathlib import Path

from . import codex_p1_adam_feedback_compare_20261030_main as compare
from . import codex_p1_adam_trust_compare_20261030_main as trust
from . import codex_p1_adam_best_iterate_physical_20261030_main as physical

ROOT = compare.base.ROOT
NAME = "codex_p1_adam_best_iterate_compare_20261030_main"
REPORT_SHA = "16e9f9214c6ee89878fc311297229f2c9ef9ccb74db7d0f7c28a24e819628384"
SAVED_SHA = "1935e79cc01eb33419e40f8d07b5f3ea4122360805a8ca4e56e1de5a7daf39e9"
TRUST_PLAN = ROOT / "results/codex_p1_adam_trust_plan_20261030_main_RUN01.json"
TRUST_PLAN_SHA = "e42a4e8cd961c2cacc0f41b25d5a09a13b0c4b748a222383bd1f7a58a419bfe4"
FIELDS = compare.FIELDS


def build():
    reference = trust.build()
    compare.DIR = ROOT / f"results/{physical.NAME}_RUN01"
    compare.REPORT_SHA = REPORT_SHA
    compare.SAVED = ROOT / f"results/{physical.NAME}_SAVED_RUN01.json"
    compare.SAVED_SHA = SAVED_SHA
    observed = compare.build()
    read = compare.plan.gap.prior.gap.read
    new_plan, _ = read(physical.PLAN, physical.SHA)
    old_plan, _ = read(TRUST_PLAN, TRUST_PLAN_SHA)
    if any(len(x) != 24 for x in (reference["rows"], observed["rows"], new_plan["rows"], old_plan["rows"])):
        raise ValueError("fixed roster")
    rows = []
    for i, (old, new, np, op) in enumerate(zip(reference["rows"], observed["rows"],
                                                new_plan["rows"], old_plan["rows"], strict=True)):
        if old["index"] != new["index"] or new["index"] != np["index"] or np["index"] != op["index"] or i != np["index"]:
            raise ValueError("identity")
        if old["midpoint_sha256"] != new["midpoint_sha256"] or np["midpoint_pixel_sha256"] != old["midpoint_sha256"]:
            raise ValueError("midpoint mismatch")
        if np["first_packet_hex"] != op["first_packet_hex"]:
            raise ValueError("first packet mismatch")
        if abs(np["optimized_score"]-op["optimized_score"]) > 1e-6 or abs(np["baseline_score"]-op["baseline_score"]) > 1e-6:
            raise ValueError("optimizer path mismatch")
        if np["selected_step"] == 20 and np["second_packet_hex"] != op["second_packet_hex"]:
            raise ValueError("last iterate action mismatch")
        x, y = old["feedback_endpoint"], new["feedback_endpoint"]
        delta = None if x is None or y is None else {k: y[k]-x[k] for k in FIELDS}
        rows.append({"index": i, "root_slot": old["root_slot"], "condition": old["condition"], "role": old["role"],
                     "logical_pair": old["logical_pair"], "selected_step": np["selected_step"],
                     "same_suffix_as_trust": np["second_packet_hex"] == op["second_packet_hex"],
                     "objective_delta_vs_trust": np["selected_score"] - min(op["baseline_score"], op["optimized_score"]),
                     "trust_status": old["feedback_status"], "new_status": new["feedback_status"],
                     "trust_steps": old["feedback_steps"], "new_steps": new["feedback_steps"],
                     "trust_last_success": old["feedback_last_success"], "new_last_success": new["feedback_last_success"],
                     "trust_endpoint": x, "new_endpoint": y, "new_minus_trust": delta,
                     "new_minus_open": new["feedback_minus_open"]})
    comparable = [row for row in rows if row["new_minus_trust"] is not None]
    return {"schema": NAME, "rows": rows, "status": "complete",
            "summary": {"rows": len(rows), "same_suffix": sum(r["same_suffix_as_trust"] for r in rows),
                        "trust_success_records": sum(r["trust_last_success"] for r in rows),
                        "new_success_records": sum(r["new_last_success"] for r in rows),
                        "comparable": len(comparable),
                        "new_lower": {k: sum(r["new_minus_trust"][k] < -1e-8 for r in comparable) for k in FIELDS},
                        "new_mean_minus_trust": {k: sum(r["new_minus_trust"][k] for r in comparable)/len(comparable)
                                                 for k in FIELDS}},
            "code_sha256": compare.base.sha(Path(__file__).resolve()), "plan_sha256": physical.SHA,
            "trust_plan_sha256": TRUST_PLAN_SHA, "physical_report_sha256": REPORT_SHA,
            "physical_saved_sha256": SAVED_SHA, "P1_admission": False, "independent_confirmation": False,
            "limitations": "known roots and duplicates; offline midpoint; censored endpoints excluded; no fairness claim"}


if __name__ == "__main__":
    result = build()
    out = ROOT / f"results/{NAME}_RUN01.json"
    print(json.dumps({"sha256": compare.base.write(out, result), "summary": result["summary"]}), flush=True)
