"""Execute every fixed candidate and model-selected labels from the sealed model plan."""
import argparse
import hashlib

import numpy as np

from . import p1_dev305_contact_model_select_20261102 as planner
from . import p1_dev305_selected_bank_first_feedback_physical_20261102 as prior
from . import p1_dev305_triprefix_midfeedback_physical_20261102 as engine

base, need, ROOT = planner.base, planner.need, planner.ROOT
NAME = "p1_dev305_contact_model_select_physical_20261102"
AUTHORITY = "DEV305_CONTACT_MODEL_SELECT_PHYSICAL"
PREP_SHA = "3e3ba3414842473e37b76734d9568d33ce00d9ce5107db4779438c533bef7056"
PLAN_SHA = "bd74bb56394622f7e8366e0739dc17aacb1b9dee12bf12514e5640b7959ab7ca"
OLD_PHYS_SHA = "b813af00ba8fe5472c9b9a4066d4d7dd9317897b788e512cfcb435db21cb4d7c"


def protocol():
    spec = base.read(ROOT / "results" / f"{planner.NAME}_PREP_DEV01.json", PREP_SHA)
    plan = base.read(ROOT / "results" / f"{planner.NAME}_PLAN_DEV01/report.json", PLAN_SHA)
    physical = base.read(ROOT / "results" / f"{prior.NAME}_PREP_DEV01.json", OLD_PHYS_SHA)
    need(spec == planner.protocol() and physical == prior.protocol() and plan["contract_sha256"] == PREP_SHA
         and plan["model_before"] == plan["model_after"] == spec["model_hash"], "sources")
    roles = []
    for rgb, prediction, replay in zip(spec["roles"], plan["rows"], physical["roles"], strict=True):
        need(rgb["arm"] == prediction["arm"] == replay["arm"], "role")
        delta = np.asarray(prediction["predicted_goal_delta"], dtype=np.float32)
        scores = np.asarray(prediction["scores"])
        need(delta.shape == (4, spec["latent_dim"]) and np.isfinite(delta).all()
             and np.allclose((delta ** 2).mean(axis=1), scores, atol=2e-6, rtol=0)
             and int(np.argmin(scores)) == prediction["selected_index"], "score binding")
        for i, (label, action) in enumerate(zip(spec["bank_order"], rgb["actions"], strict=True)):
            a = np.asarray(action, dtype="<f4")
            need(a.shape == (5, 2) and np.isfinite(a).all() and (np.abs(a) <= 1).all(), "action")
            roles.append({"arm": rgb["arm"] + "__" + label, "history": replay["history"],
                          "states": replay["states"], "checkpoint": replay["checkpoint"], "tail": a.tolist(),
                          "selected": i == prediction["selected_index"],
                          "action_sha256": hashlib.sha256(a.tobytes()).hexdigest()})
    need(len(roles) == 12 and sum(role["selected"] for role in roles) == 3, "coverage")
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "engine_sha256": base.sha(engine.__file__), "plan_sha256": PLAN_SHA,
            "seed": physical["seed"], "roles": roles,
            "scope": "post-outcome known-root physical replay; model scores do not certify visibility",
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
