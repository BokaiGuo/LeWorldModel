"""Execute RGB-feedback plans after the selected bank's first packet."""
import argparse
import hashlib

import numpy as np

from . import p1_dev305_selected_bank_first_feedback_20261102 as planner
from . import p1_dev305_triprefix_midfeedback_physical_20261102 as engine

base, run, ROOT, need = planner.base, planner.run, planner.ROOT, planner.need
NAME = "p1_dev305_selected_bank_first_feedback_physical_20261102"
AUTHORITY = "DEV305_SELECTED_BANK_FIRST_FEEDBACK_PHYSICAL"
PREP_SHA = "2e5463199b1924c89bf0347c9e9a60f936c80c51d6cff217d4516f3a2916cfb2"
PLAN_SHA = "592654f46dd11eeed1d5d489c3343437fbd39d51823c5b7bc8f2816d41695f38"


def protocol():
    spec = base.read(ROOT / "results" / f"{planner.NAME}_PREP_DEV01.json", PREP_SHA)
    plan = base.read(ROOT / "results" / f"{planner.NAME}_PLAN_DEV01/report.json", PLAN_SHA)
    need(spec == planner.protocol() and plan["contract_sha256"] == PREP_SHA
         and plan["model_before"] == plan["model_after"] == spec["model_hash"], "plan binding")
    replay = planner.mid.replay
    replay_spec = base.read(ROOT / "results" / f"{replay.NAME}_PREP_DEV01.json", planner.mid.REPLAY_CONTRACT_SHA)
    replay_result = base.read(ROOT / "results" / f"{replay.NAME}_RUN_DEV01/report.json", planner.mid.REPLAY_REPORT_SHA)
    compass = replay.compare.compass
    old = base.read(compass.ROOT / f"{compass.source.NAME}_RUN_DEV01/report.json", compass.REPORT_SHA)
    initial = base.read(ROOT / "results" / f"{run.NAME}_EXEC_DEV01/report.json",
                        compass.source.prior.previous.EXEC_SHA)
    shared = next(x for x in initial["rows"] if x["arm"] == "original")
    physical_bank = base.read(ROOT / "results" / f"{planner.bank.model.physical.NAME}_RUN_DEV01/report.json",
                              "0472a27f45dd5c5075c67a06b95edba274a2d8617fdc21e30e2b99df27237ef2")
    roles = []
    for role, prediction in zip(spec["roles"], plan["rows"], strict=True):
        action = np.asarray(prediction["selected"]["actions"], dtype="<f4")
        need(prediction["arm"] == role["arm"] and action.shape == (2, 5, 2)
             and np.isfinite(action).all() and (np.abs(action) <= 1).all()
             and hashlib.sha256(action.tobytes()).hexdigest() == prediction["selected_actions_sha256"], "actions")
        branch = next(x for x in old["rows"] if x["arm"] == role["arm"] + "__m10_d4")
        checkpoint = next(x["midpoint"] for x in replay_result["rows"]
                          if x["arm"] == role["arm"] and x["name"] == "m10_d4")
        first_ten = branch["transitions"][:10]
        history = replay_spec["shared_actions"] + [t["action"] for t in first_ten]
        states = [t["state"] for t in shared["transitions"]] + [t["state"] for t in first_ten]
        need(len(history) == len(states) == 15 and np.array_equal(
            np.asarray(history[-10:], dtype="<f4"), np.asarray(role["actual_history"], dtype="<f4")),
            "full actual history")
        i, j = role["old_selected_pair"]
        old_tail = next(x for x in physical_bank["rows"]
                        if x["prefix"] == role["arm"] and x["first"] == i and x["second"] == j)
        need(old_tail["status"] == "fixed_horizon_completed"
             and old_tail["transitions"][4]["state"] == states[-1], "same checkpoint")
        roles.append({"arm": role["arm"], "history": history, "states": states, "checkpoint": checkpoint,
                      "tail": action[0].tolist(), "plan_sha256": prediction["selected_actions_sha256"],
                      "old_tail_final_state_evaluation_only": old_tail["transitions"][-1]["state"],
                      "old_tail_final_reward_evaluation_only": old_tail["transitions"][-1]["reward"]})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "physical_engine_sha256": base.sha(engine.__file__), "plan_sha256": PLAN_SHA,
            "seed": run.SEED, "roles": roles, "classification": "known_root_post_outcome_development",
            "p1_admission": False, "independent_confirmation": False}


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
        engine.protocol = protocol
        engine.AUTHORITY = AUTHORITY
        print(engine.execute(path, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
