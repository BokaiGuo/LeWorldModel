"""Recompute risk projections and compare with sealed real terminal RGB frames."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_contact_risk_projection_20261102 as source
from . import p1_dev305_contact_model_select_saved_20261102 as real

base, need, ROOT = source.base, source.need, source.ROOT
PREP_SHA = "8d33e6bf437ab98eac8850d888f7697f0ac216e2c996378a808bc329d9823d25"
REPORT_SHA = "77a6d82e127188d95c0c13e911bdf6282b19bb1fd93bbf9da14f7ecede6d0382"


def analyze():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    plan = base.read(ROOT / "results" / f"{source.source.NAME}_PLAN_DEV01/report.json", source.PLAN_SHA)
    real_spec = base.read(ROOT / "results" / f"{real.source.NAME}_PREP_DEV01.json", real.PREP_SHA)
    physical = base.read(ROOT / "results" / f"{real.source.NAME}_RUN_DEV01/report.json", real.REPORT_SHA)
    need(spec == source.protocol() and report["contract_sha256"] == PREP_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"]
         and not physical["errors"] and len(physical["rows"]) == len(real_spec["roles"]) == 12, "sources")
    directory = ROOT / "results" / f"{real.source.NAME}_RUN_DEV01"
    rows = []
    for role, predicted, plan_row in zip(spec["roles"], report["rows"], plan["rows"], strict=True):
        need(role["arm"] == predicted["arm"] == plan_row["arm"], "role")
        current = source.run.coherent.base.frame(role["images"][-1])
        screen = source.visibility.inspect(current)
        boxes = {name: screen[name]["bbox"] for name in ("object", "pusher")}
        need(boxes == predicted["source_boxes"], "boxes")
        check = source.predict_screen(predicted["jacobian"], plan_row["predicted_goal_delta"],
                                      predicted["current_goal_delta"], boxes)
        need(check == predicted["projections"], "projection recompute")
        scores = plan_row["scores"]
        safe = [i for i, item in enumerate(check) if item["predicted_visible"]]
        chosen = min(safe, key=lambda i: (scores[i], i)) if safe else None
        need(chosen == predicted["selected_index"], "selection")
        for i, label in enumerate(("toward_02", "away_02", "zero", "toward_05")):
            key = role["arm"] + "__" + label
            prior_role = next(x for x in real_spec["roles"] if x["arm"] == key)
            item = next(x for x in physical["rows"] if x["arm"] == key)
            need(item["status"] == "fixed_horizon_completed"
                 and np.array_equal(np.asarray(prior_role["tail"], dtype="<f4"),
                                    np.asarray(role["actions"][i], dtype="<f4")), "physical binding")
            frame = item["observations"][-1]
            ref = {"path": str((directory / frame["file"]).resolve()), "file_sha256": frame["file_sha256"],
                   "pixel_sha256": frame["pixel_sha256"]}
            observed = source.visibility.inspect(source.run.coherent.base.frame(ref))
            rows.append({"arm": role["arm"], "candidate": label, "selected": i == chosen,
                         "predicted_visible": check[i]["predicted_visible"], "actual_visible": observed["eligible"],
                         "estimated_boxes": check[i]["estimated_boxes"],
                         "actual_boxes": {name: observed[name]["bbox"] for name in ("object", "pusher")},
                         "physical_reward": item["transitions"][-1]["reward"]})
    return {"schema": "p1_dev305_contact_risk_projection_saved_20261102.v1", "rows": rows,
            "scope": "post-outcome one known root; predicted visibility is an uncalibrated heuristic",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
