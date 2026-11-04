"""Verify recorded cp5 feedback plans and physical prefix/tail execution."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from . import p1_dev304_diverse_feedback_physical_20261102 as physical
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

f, saved, base, run = physical.feedback, physical.saved, physical.base, physical.run
RESULT_SHA = "096b4ccd4223269a4fea6fc9a252e5a8c63ce74a605e93a872aee82b9490d5f0"
RUN = physical.ROOT / f"results/{physical.NAME}_DEV01"
PLAN = physical.ROOT / f"results/{f.NAME}_PLAN_DEV01"
need = physical.require


def analyze():
    spec = base.read(physical.ROOT / f"results/{f.NAME}_PREP_DEV01.json", physical.CONTRACT_SHA)
    need(spec == f.protocol(), "frozen feedback inputs")
    plan = base.read(PLAN / "report.json", physical.PLAN_SHA)
    result = base.read(RUN / "report.json", RESULT_SHA)
    need(plan["contract_sha256"] == physical.CONTRACT_SHA
         and plan["model_before"] == plan["model_after"] == spec["model_hash"]
         and result["status"] == "recorded" and not result["errors"]
         and result["plan_sha256"] == physical.PLAN_SHA
         and base.sha(RUN / "journal.jsonl") == result["journal_sha256"], "report bindings")
    for label in ("encoder", "projector"):
        need(plan["ledger"]["encode:" + label] == {"calls": 7, "samples": 7}, "encode ledger")
    goal = base.read(saved.ROOT / f"{saved.NAME}_CAPTURE_DEV01/report.json",
                     saved.CAPTURE_SHA)["goal_state_evaluation_only"]
    prior = base.read(saved.ROOT / f"{saved.NAME}_EXEC_DEV01/report.json", saved.EXEC_SHA)
    need([r["arm"] for r in spec["roles"]] == [r["arm"] for r in plan["rows"]]
         == [r["arm"] for r in result["rows"]] == [r["arm"] for r in prior["rows"]], "arm roster")
    rows, frames = [], 0
    for role, planned, row, old in zip(spec["roles"], plan["rows"], result["rows"], prior["rows"], strict=True):
        arm = role["arm"]
        need(base.read(PLAN / (arm + ".json")) == planned
             and base.read(RUN / (arm + ".row.json")) == row, "row shards")
        for label in ("action_encoder", "predictor"):
            need(plan["ledger"][arm + ":" + label] == {"calls": 10, "samples": 3200}, "model ledger")
        prefix = np.asarray(role["actual_history"], dtype="<f4")[5:]
        tail = np.asarray(planned["selected"]["actions"], dtype="<f4")
        need(prefix.shape == (5, 2) and tail.shape == (2, 5, 2)
             and hashlib.sha256(prefix.tobytes()).hexdigest() == role["prefix_sha256"] == planned["prefix_sha256"]
             and hashlib.sha256(tail.tobytes()).hexdigest() == planned["selected_actions_sha256"], "action hashes")
        need(row["status"] == "fixed_horizon_completed" and len(row["transitions"]) == 10, "no censoring")
        for i, transition in enumerate(row["transitions"]):
            expected = prefix[i] if i < 5 else tail[0, i-5]
            need(transition["phase"] == ("prefix" if i < 5 else "tail")
                 and transition["step"] == (i % 5) + 1 and np.array_equal(transition["action"], expected)
                 and not transition["terminated"] and not transition["truncated"], "action trace")
            if i < 5:
                need(transition["state"] == old["transitions"][i]["state"], "prefix replay")
        obs = row["observations"]
        need([(o["phase"], o["step"]) for o in obs]
             == [("warmup", x) for x in (0, 5, 10)] + [("prefix", 5)] + [("tail", x) for x in range(1, 6)],
             "frame roster")
        for frame in obs:
            saved.verify_frame(RUN, frame)
            frames += 1
        for a, b in zip(obs[:4], [*old["observations"][:3], old["observations"][7]], strict=True):
            need(all(a[k] == b[k] for k in ("pixel_sha256", "state", "snapshot")), "replayed snapshot")
        need(all(o["state"] == t["state"] for o, t in zip(obs[4:], row["transitions"][5:], strict=True)),
             "tail observation")
        rows.append({"arm": arm, "after_prefix": measures(row["transitions"][4]["state"], goal),
                     "after_tail": measures(row["transitions"][-1]["state"], goal),
                     "final_reward": row["transitions"][-1]["reward"],
                     "native_success": False, "status": row["status"]})
    return {"schema": "p1_dev304_diverse_feedback_saved.v1", "rows": rows,
            "frames_verified": frames, "classification": "post_outcome_development_only",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
