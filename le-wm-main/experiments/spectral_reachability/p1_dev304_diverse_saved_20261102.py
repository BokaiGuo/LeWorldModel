"""Verify the post-outcome terminal-objective development comparison."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_dev304_diverse_cem_20261102 as run
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

ROOT = run.ROOT / "results"
NAME = run.NAME
SPEC_SHA = "2e602c54832f6d9d641ba1e00c6cd574719ae1055602209568fa4df25ea7cfb8"
CAPTURE_SHA = "aab1e65ed1786863dbf409f255723ccb9e58e3c1b898ea2befb5667cc2ce5dc0"
PLAN_SHA = "02a9d9ad58be92a8407e49e688de2831bfd2ab7ad438711ee9e610cb7b0ec2fe"
EXEC_SHA = "a7a81b50454a2fe0db833f5c0e88806fcae652d2c7119012db905e814c899815"
need = run.require


def verify_frame(directory, ref):
    path = directory / ref["file"]
    need(path.parent == directory and not path.is_symlink()
         and run.base.sha(path) == ref["file_sha256"], "frame file binding")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    need(pixels.shape == (224, 224, 3)
         and hashlib.sha256(pixels.tobytes()).hexdigest() == ref["pixel_sha256"], "frame pixels")


def analyze():
    spec_path = ROOT / f"{NAME}_PREP_DEV01.json"
    capture_dir = ROOT / f"{NAME}_CAPTURE_DEV01"
    plan_dir = ROOT / f"{NAME}_PLAN_DEV01"
    execution_dir = ROOT / f"{NAME}_EXEC_DEV01"
    spec = run.base.read(spec_path, SPEC_SHA)
    need(spec == run.specification(), "specification/source binding")
    capture = run.base.read(capture_dir / "report.json", CAPTURE_SHA)
    plan = run.base.read(plan_dir / "report.json", PLAN_SHA)
    execution = run.base.read(execution_dir / "report.json", EXEC_SHA)
    need(capture["contract_sha256"] == SPEC_SHA
         and run.base.sha(capture_dir / "journal.jsonl") == capture["journal_sha256"], "capture journal")
    blind = run.base.read(capture_dir / "blind.json", capture["blind_sha256"])
    need(set(blind) == {"schema", "seed", "actions", "images", "goal"}
         and blind["seed"] == run.SEED
         and np.array_equal(blind["actions"], np.zeros((10, 2))), "blind input isolation")
    need(len(blind["images"]) == len(capture["checkpoints"]) == 3, "capture roster")
    for checkpoint, image, step in zip(capture["checkpoints"], blind["images"], (0, 5, 10), strict=True):
        need(checkpoint["phase"] == "warmup" and checkpoint["step"] == step
             and all(checkpoint[k] == image[k] for k in ("file", "file_sha256", "pixel_sha256")),
             "capture checkpoint binding")
        verify_frame(capture_dir, image)
    verify_frame(capture_dir, blind["goal"])
    expected_plan = run.plan_contract(spec_path, SPEC_SHA, capture_dir, CAPTURE_SHA)
    need(run.base.read(plan_dir / "plan_contract.json", plan["plan_contract_sha256"]) == expected_plan
         and plan["capture_sha256"] == CAPTURE_SHA
         and plan["model_before"] == plan["model_after"] == spec["model_hash"], "plan binding")
    for component in ("encoder", "projector"):
        need(plan["ledger"]["encode:" + component] == {"calls": 4, "samples": 4}, "encoding ledger")
    for arm in ("original", "coherent"):
        for component in ("action_encoder", "predictor"):
            need(plan["ledger"][arm + ":" + component] == {"calls": 10, "samples": 3200}, "model ledger")
    need(execution["status"] == "recorded" and not execution["errors"]
         and execution["capture_sha256"] == CAPTURE_SHA and execution["plan_sha256"] == PLAN_SHA
         and run.base.sha(execution_dir / "journal.jsonl") == execution["journal_sha256"], "execution binding")
    need([r["arm"] for r in plan["rows"]] == [r["arm"] for r in execution["rows"]]
         == ["original", "coherent"], "arm roster")
    results = []
    for planned, row in zip(plan["rows"], execution["rows"], strict=True):
        arm = row["arm"]
        need(run.base.read(plan_dir / (arm + ".json")) == planned
             and run.base.read(execution_dir / (arm + ".row.json")) == row, "row shards")
        actions = np.asarray(planned["selected"]["actions"], dtype="<f4")
        need(actions.shape == (2, 5, 2) and np.isfinite(actions).all()
             and np.all(np.abs(actions) <= 1)
             and hashlib.sha256(actions.tobytes()).hexdigest() == planned["selected_actions_sha256"],
             "selected action binding")
        direction = np.asarray(expected_plan["contact_direction_raw_xy"], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        ranking = []
        for iteration, event in enumerate(planned["iterations"]):
            bank = np.asarray(event["bank"], dtype=np.float32)
            scores = np.asarray(event["all_scores"], dtype=float)
            need(bank.shape == (64, 2, 5, 2) and scores.shape == (64, 5)
                 and np.isfinite(bank).all() and np.isfinite(scores).all()
                 and np.max(np.abs(bank)) <= 1 and event["iteration"] == iteration, "bank")
            feasible = np.einsum("sij,j->s", bank[:, 0, :3], direction) >= .15
            need(feasible.tolist() == event["feasible"], "feasibility")
            robust = run.coherent.base.first.engine.base.common.robust_scores(scores)
            groups = 2 if arm == "coherent" else 1
            size = 64 // groups
            expected_elites = []
            for group in range(groups):
                indices = np.arange(group * size, (group + 1) * size)
                order = indices[feasible[indices]]
                order = order[np.argsort(robust[order], kind="stable")]
                need(len(order) > 0, "empty population")
                count = 4 if groups == 2 else 8
                expected_elites.append(np.pad(order, (0, max(0, count-len(order))), mode="edge")[:count].tolist())
            need(event["elite_indices_by_population"] == expected_elites, "elite ranking")
            ranking.extend((float(robust[i]), iteration, int(i)) for i in np.flatnonzero(feasible))
        winner = min(ranking)
        need(winner == (planned["selected"]["robust_score"], planned["selected"]["iteration"],
                        planned["selected"]["candidate"])
             and np.array_equal(actions, np.asarray(planned["iterations"][winner[1]]["bank"][winner[2]],
                                                    dtype=np.float32)), "global winner")
        need(row["status"] == "fixed_horizon_completed" and len(row["transitions"]) == 5, "censoring")
        for step, (transition, action) in enumerate(zip(row["transitions"], actions[0], strict=True), 1):
            need(transition["step"] == step and np.array_equal(transition["action"], action)
                 and not transition["terminated"] and not transition["truncated"], "transition binding")
        observations = row["observations"]
        need([(o["phase"], o["step"]) for o in observations]
             == [("warmup", x) for x in (0, 5, 10)] + [("execute", x) for x in range(1, 6)],
             "observation roster")
        for obs in observations:
            verify_frame(execution_dir, obs)
        for obs, original in zip(observations[:3], capture["checkpoints"], strict=True):
            need(all(obs[k] == original[k] for k in ("pixel_sha256", "state", "snapshot")), "warmup binding")
        for obs, transition in zip(observations[3:], row["transitions"], strict=True):
            need(obs["state"] == transition["state"], "observation state binding")
        goal = capture["goal_state_evaluation_only"]
        results.append({"arm": arm, "selected_actions_sha256": planned["selected_actions_sha256"],
                        "before": measures(observations[2]["state"], goal),
                        "after": measures(row["transitions"][-1]["state"], goal),
                        "reward": row["transitions"][-1]["reward"],
                        "success": False, "status": row["status"]})
    a, b = plan["rows"]
    differences = [float(np.max(np.abs(np.asarray(x["all_scores"]) - np.asarray(y["all_scores"]))))
                   for x, y in zip(a["iterations"], b["iterations"], strict=True)]
    return {"schema": NAME + ".saved.v1", "rows": results, "frames_verified": 20,
            "score_max_absolute_difference_by_iteration": differences,
            "feasible_candidates_by_iteration": {r["arm"]: [sum(x["feasible"]) for x in r["iterations"]]
                                                   for r in plan["rows"]},
            "same_selected_actions": a["selected_actions_sha256"] == b["selected_actions_sha256"],
            "classification": "post_outcome_development_history_and_training_membership_unknown",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(run.base.write_new(parser.parse_args().output, analyze()))
