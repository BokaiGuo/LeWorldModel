"""Recompute second-boundary risk selection and bind to sealed physical execution."""
import argparse
from pathlib import Path

from . import p1_dev305_second_boundary_risk_20261102 as risk
from . import p1_dev305_contact_second_boundary_saved_20261102 as physical

base, need, ROOT = risk.base, risk.need, risk.ROOT
NAME = "p1_dev305_second_boundary_risk_saved_20261102"
PREP_SHA = "46a4b88d3c652ad7ef8a470fb981a18f206c2275563ac8872dc367ba2b793f48"
REPORT_SHA = "066fa912b95e80e50433868e0d2b1d569d2361637a012c6f4963864338ca0ac5"


def analyze():
    spec = base.read(ROOT / "results" / f"{risk.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{risk.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    plan = base.read(ROOT / "results" / f"{risk.source.NAME}_PLAN_DEV01/report.json", risk.PLAN_SHA)
    executed = physical.analyze()
    need(spec == risk.protocol() and report["contract_sha256"] == PREP_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"]
         and report["counts"] == {"encoder": {"calls": 34, "samples": 34},
                                  "projector": {"calls": 34, "samples": 34}}, "sources")
    rows = []
    for role, predicted, plan_row in zip(spec["roles"], report["rows"], plan["rows"], strict=True):
        need(role["arm"] == predicted["arm"] == plan_row["arm"], "role")
        screen = risk.risk.visibility.inspect(risk.run.coherent.base.frame(role["images"][-1]))
        boxes = {name: screen[name]["bbox"] for name in ("object", "pusher")}
        need(boxes == predicted["source_boxes"], "boxes")
        projection = risk.risk.predict_screen(predicted["jacobian"], plan_row["predicted_goal_delta"],
                                              predicted["current_goal_delta"], boxes)
        need(projection == predicted["projections"], "projection recompute")
        eligible = [i for i, item in enumerate(projection) if item["predicted_visible"]]
        selected = min(eligible, key=lambda i: (plan_row["scores"][i], i)) if eligible else None
        need(selected == predicted["selected_index"], "selection recompute")
        for i, label in enumerate(("toward_02", "away_02", "zero", "toward_05")):
            outcome = next(row for row in executed["rows"] if row["arm"] == role["arm"] + "__" + label)
            rows.append({"arm": role["arm"], "candidate": label, "selected": i == selected,
                         "predicted_visible": projection[i]["predicted_visible"],
                         "actual_steps": outcome["steps"], "actual_status": outcome["status"],
                         "actual_final_visible": outcome["final_visible"], "actual_reward": outcome["reward"],
                         "actual_success": outcome["success"]})
    return {"schema": NAME + ".v1", "rows": rows,
            "scope": "post-outcome on a known root; executed branches stopped by guard, early outcomes censored; no causal new rollout",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
