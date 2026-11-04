"""Execute sealed coherent-seed cp10 plans on the fixed development panel."""
import argparse
import hashlib
import importlib
import inspect
import os
from pathlib import Path

import numpy as np

from . import p1_cp10_coherent_replan_20261102 as model
from . import p1_second_feedback_saved_20261102 as previous

ROOT = model.ROOT
NAME = "p1_cp10_coherent_physical_20261102"
MODEL_SHA = "714d5f9ab5ca5228f33c7d48448ae297cbab72699deaf46b44e2b03e10484f39"
MODEL_PREP_SHA = "ac9f5d9facf8cb9c693adba6d9710869c69ec7f05fc495c540e29a69608762e8"
AUTHORITY = "DEV303_CP10_COHERENT_SIMULATION"
require = model.require
base = previous.physical.previous


def protocol():
    verified = previous.analyze()
    old = base.read(previous.CONTRACT, previous.CONTRACT_SHA)
    path = ROOT / "results/p1_cp10_coherent_replan_20261102_RUN_DEV01/report.json"
    report = base.read(path, MODEL_SHA)
    require(verified["frames_verified"] == 112 and report["schema"] == model.NAME + ".report.v1"
            and report["protocol_sha256"] == MODEL_PREP_SHA
            and report["model_before"] == report["model_after"], "source verification")
    require([r["arm"] for r in report["rows"]] == [r["arm"] for r in old["roles"]], "roster")
    roles = []
    for o, r in zip(old["roles"], report["rows"], strict=True):
        actions = np.asarray(r["selected"]["actions"], dtype="<f4")
        require(actions.shape == (2, 5, 2) and np.isfinite(actions).all()
                and (np.abs(actions) <= 1).all()
                and hashlib.sha256(actions.tobytes()).hexdigest() == r["selected_actions_sha256"]
                and r["cp10_pixel_sha256"] == o["cp10_checkpoint"]["pixel_sha256"], "model actions and cp10")
        roles.append({"arm": o["arm"], "prefix": o["prefix"], "old_first_packet": o["previous_first_packet"],
                      "new_first_packet": actions[0].tolist(), "warmup_checkpoints": o["warmup_checkpoints"],
                      "prefix_checkpoints": o["prefix_checkpoints"], "cp10_checkpoint": o["cp10_checkpoint"]})
    return {"schema": NAME + ".protocol.v1", "classification": "known_development_root_only",
            "seed": old["seed"], "goal_state": old["goal_state"], "goal_rgb_sha256": old["goal_rgb_sha256"],
            "roles": roles, "model_report_sha256": MODEL_SHA, "prior_report_sha256": previous.REPORT_SHA,
            "source_sha256": {str(Path(__file__).resolve()): base.sha(__file__)},
            "independent_confirmation": False, "P1_admission": False}


def execute(contract_path, digest, output, authority):
    require(authority == AUTHORITY and base.sha(contract_path) == digest
            and base.read(contract_path) == protocol(), "contract")
    spec = base.read(contract_path)
    store = base.base.Store(output)
    store.emit("run_started", prepare_sha256=digest)
    rows, errors = [], []
    for role in spec["roles"]:
        row = {"arm": role["arm"], "status": "running", "transitions": [], "observations": []}
        env = None
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(base.sha(inspect.getsourcefile(cls)) == base.base.native.ENV_SHA, "environment")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            obs, info = env.reset(seed=spec["seed"])
            require(np.array_equal(obs["state"], role["warmup_checkpoints"][0]["state"])
                    and np.array_equal(info["goal_state"], spec["goal_state"])
                    and hashlib.sha256(np.asarray(info["goal"]).tobytes()).hexdigest()
                    == spec["goal_rgb_sha256"], "reset and goal")
            row["observations"].append(base.base.capture(
                store, env, role["arm"], "warmup", 0, role["warmup_checkpoints"][0]))
            stages = (("warmup", np.zeros((10, 2), np.float32)),
                      ("prefix", np.asarray(role["prefix"], np.float32)),
                      ("old_tail", np.asarray(role["old_first_packet"], np.float32)),
                      ("coherent_tail", np.asarray(role["new_first_packet"], np.float32)))
            for phase, actions in stages:
                for step, action in enumerate(actions, 1):
                    obs, reward, terminated, truncated, info = env.step(action.copy())
                    require(np.array_equal(env.latest_action, action), "action")
                    row["transitions"].append({"phase": phase, "step": step, "action": action.tolist(),
                                               "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                               "terminated": bool(terminated), "truncated": bool(truncated)})
                    if phase == "warmup" and step in (5, 10):
                        row["observations"].append(base.base.capture(
                            store, env, role["arm"], phase, step, role["warmup_checkpoints"][[5, 10].index(step)+1]))
                    elif phase == "prefix":
                        row["observations"].append(base.base.capture(
                            store, env, role["arm"], phase, step, role["prefix_checkpoints"][step-1]))
                    elif phase == "old_tail" and step == 5:
                        row["observations"].append(base.base.capture(
                            store, env, role["arm"], "tail", 5, role["cp10_checkpoint"]))
                    elif phase == "coherent_tail":
                        row["observations"].append(base.base.capture(store, env, role["arm"], phase, step))
                    if terminated or truncated:
                        require(phase == "coherent_tail", "censored before cp10")
                        row["status"] = "native_early_stop"
                        break
                if row["status"] == "native_early_stop":
                    break
            else:
                row["status"] = "fixed_horizon_completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            errors.append({"arm": role["arm"], **row["error"]})
        finally:
            if env is not None:
                env.close()
        store.json(role["arm"]+".row.json", row)
        store.emit("branch_closed", arm=role["arm"], status=row["status"])
        rows.append(row)
    store.emit("run_finished", errors=errors)
    store.json("report.json", {"schema": NAME + ".report.v1", "status": "recorded" if not errors else "failed",
                               "prepare_sha256": digest, "rows": rows, "errors": errors,
                               "journal_sha256": base.sha(store.path/"journal.jsonl"),
                               "independent_confirmation": False, "P1_admission": False})
    return base.sha(store.path/"report.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--tag", required=True)
    p.add_argument("--sha256")
    p.add_argument("--authority")
    a = p.parse_args()
    require(a.tag.isascii() and a.tag.isalnum(), "tag")
    contract = ROOT/"results"/f"{NAME}_PREP_{a.tag}.json"
    output = ROOT/"results"/f"{NAME}_RUN_{a.tag}"
    if a.command == "prepare":
        require(not a.sha256 and not a.authority and not os.path.lexists(output), "prepare")
        print(base.write_new(contract, protocol()))
    else:
        require(a.sha256 and a.authority, "authorization")
        print(execute(contract, a.sha256, output, a.authority))
