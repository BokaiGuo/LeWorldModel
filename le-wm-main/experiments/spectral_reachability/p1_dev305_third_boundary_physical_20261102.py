"""Execute risk-selected and zero candidates after the next real feedback boundary."""
import argparse
import importlib.util

import numpy as np

from . import p1_dev305_contact_third_boundary_20261102 as planner
from . import p1_dev305_third_boundary_risk_20261102 as risk
from . import p1_dev305_contact_second_boundary_physical_20261102 as template

base, need, ROOT = planner.base, planner.need, planner.ROOT
NAME = "p1_dev305_third_boundary_physical_20261102"
AUTHORITY = "DEV305_THIRD_BOUNDARY_PHYSICAL"
RISK_PREP_SHA = "5723ef0c98e7fb60fa4e1f70b10522b1dfa0e12f07d5f898e77a179bb353b7a9"
RISK_REPORT_SHA = "cca8adf245bfa4d43d865f18e6a7856d50fddb791c7d1467f6a0356b8e8dc9e3"


def protocol():
    spec = base.read(ROOT / "results" / f"{planner.NAME}_PREP_DEV01.json", risk.PREP_SHA)
    plan = base.read(ROOT / "results" / f"{planner.NAME}_PLAN_DEV01/report.json", risk.PLAN_SHA)
    risk_spec = base.read(ROOT / "results" / f"{risk.NAME}_PREP_DEV01.json", RISK_PREP_SHA)
    projected = base.read(ROOT / "results" / f"{risk.NAME}_RUN_DEV01/report.json", RISK_REPORT_SHA)
    prior_spec = base.read(ROOT / "results" / f"{template.NAME}_PREP_DEV01.json",
                           planner.risk_saved.physical.CONTRACT_SHA)
    prior_report = base.read(ROOT / "results" / f"{template.NAME}_RUN_DEV01/report.json",
                             planner.risk_saved.physical.REPORT_SHA)
    need(spec == planner.protocol() and risk_spec == risk.protocol()
         and plan["contract_sha256"] == risk.PREP_SHA and projected["contract_sha256"] == RISK_PREP_SHA
         and plan["model_before"] == plan["model_after"] == spec["model_hash"]
         and projected["model_before"] == projected["model_after"] == spec["model_hash"], "source")
    roles = []
    for source, pred, result in zip(spec["roles"], plan["rows"], projected["rows"], strict=True):
        need(source["arm"] == pred["arm"] == result["arm"], "arm")
        pixels = planner.run.coherent.base.frame(source["images"][-1])
        screen = risk.engine.risk.visibility.inspect(pixels)
        boxes = {name: screen[name]["bbox"] for name in ("object", "pusher")}
        need(boxes == result["source_boxes"], "source boxes")
        projections = risk.engine.risk.predict_screen(result["jacobian"], pred["predicted_goal_delta"],
                                                      result["current_goal_delta"], boxes)
        need(projections == result["projections"], "projection")
        eligible = [i for i, item in enumerate(projections) if item["predicted_visible"]]
        chosen = min(eligible, key=lambda i: (pred["scores"][i], i)) if eligible else None
        need(chosen is not None and chosen == result["selected_index"], "risk-selected candidate")
        key = source["arm"] + "__" + source["previous_candidate"]
        replay = next(row for row in prior_spec["roles"] if row["arm"] == key)
        actual = next(row for row in prior_report["rows"] if row["arm"] == key)
        history = replay["history"] + [row["action"] for row in actual["transitions"]]
        states = replay["states"] + [row["state"] for row in actual["transitions"]]
        need(len(history) == len(states) == 25
             and np.array_equal(np.asarray(history[-10:], dtype="<f4"),
                                np.asarray(source["actual_history"], dtype="<f4")), "history binding")
        for label, index in (("risk_selected", chosen), ("zero_control", 2)):
            actions = np.asarray(source["actions"][index], dtype="<f4")
            need(actions.shape == (5, 2) and np.isfinite(actions).all()
                 and (np.abs(actions) <= 1).all(), "bounded action")
            roles.append({"arm": source["arm"] + "__" + label, "selected": label == "risk_selected",
                          "candidate": spec["bank_order"][index], "history": history, "states": states,
                          "checkpoint": actual["observations"][-1], "tail": actions.tolist()})
    need(len(roles) == 6, "coverage")
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "physical_engine_sha256": base.sha(template.__file__), "risk_contract_sha256": RISK_PREP_SHA,
            "risk_report_sha256": RISK_REPORT_SHA, "seed": prior_spec["seed"], "roles": roles,
            "scope": "known-root development; risk-selected versus zero, some are identical; fixed physical follow-up",
            "p1_admission": False, "independent_confirmation": False}


module_spec = importlib.util.spec_from_file_location(
    "experiments.spectral_reachability.p1_dev305_third_boundary_physical_engine", template.__file__)
engine = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(engine)
engine.protocol = protocol
engine.AUTHORITY = AUTHORITY

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    path = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(path, protocol()))
    else:
        print(engine.execute(path, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
