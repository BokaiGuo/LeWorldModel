"""Replay fixed development trajectories through cp10, execute replanned packet."""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
from pathlib import Path

import numpy as np

from . import p1_residual_cp5_timing_physical_20261102 as previous
from . import p1_residual_cp5_timing_saved_20261102 as saved
from . import p1_second_feedback_replan_20261102 as replan

ROOT = previous.ROOT
NAME = "p1_second_feedback_physical_20261102"
MODEL_SHA = "c7180ba2818c761c7a6cbd1e9b0fe702ce0badad5009be03230c3d26e88181c8"
AUTHORITY = "DEV303_SECOND_FEEDBACK_DEVELOPMENT_SIMULATION"
require = previous.require


def build(model_sha):
    require(model_sha == MODEL_SHA, "fixed model report")
    verified = saved.analyze()
    old_contract = previous.read(saved.CONTRACT, saved.CONTRACT_SHA)
    old_report = previous.read(saved.RUN / "report.json", saved.REPORT_SHA)
    model_path = ROOT / "results/p1_second_feedback_replan_20261102_RUN_DEV01/report.json"
    model = previous.read(model_path, model_sha)
    require(model["schema"] == replan.NAME + ".report.v1"
            and model["status"] == "model_only_complete"
            and model["protocol_sha256"] == "d667aa06e82cdebb1e72abbedcd0d5ebe7110866a622621a5774170936ca74c0"
            and model["model_before"] == model["model_after"]
            and verified["frame_references_verified"] == 144, "model and trajectory binding")
    require([r["arm"] for r in model["rows"]] == [r["arm"] for r in old_contract["roles"]]
            == [r["arm"] for r in old_report["rows"]], "arm roster")
    roles = []
    for original, prior, planned in zip(old_contract["roles"], old_report["rows"], model["rows"], strict=True):
        actions = np.asarray(planned["selected"]["actions"], dtype="<f4").reshape(10, 2)
        require(np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
                and previous.base.hashlib.sha256(actions.tobytes()).hexdigest()
                == planned["selected_actions_sha256"], "new plan action SHA")
        checkpoint = next(o for o in prior["observations"] if o["phase"] == "tail" and o["step"] == 5)
        require(checkpoint["pixel_sha256"] == planned["cp10_pixel_sha256"], "actual feedback binding")
        roles.append({"arm": original["arm"], "prefix": original["prefix"],
                      "previous_first_packet": original["tail"][:5],
                      "new_first_packet": actions[:5].tolist(),
                      "prefix_checkpoints": original["prefix_checkpoints"],
                      "warmup_checkpoints": original["warmup_checkpoints"],
                      "cp10_checkpoint": checkpoint})
    return {"schema": NAME + ".contract.v1", "classification": "known_development_root_only",
            "old_contract_sha256": saved.CONTRACT_SHA, "old_report_sha256": saved.REPORT_SHA,
            "model_report_sha256": model_sha, "goal_state": old_contract["goal_state"],
            "goal_rgb_sha256": old_contract["goal_rgb_sha256"], "seed": previous.SEED,
            "roles": roles, "source_sha256": {str(Path(__file__).resolve()): previous.sha(__file__),
                                             str(Path(replan.__file__).resolve()): previous.sha(replan.__file__)},
            "control": "compare paired old continuation at identical verified cp10; development only",
            "P1_admission": False, "independent_confirmation": False}


def execute(contract_path, contract_sha, model_sha, output, authority):
    require(authority == AUTHORITY and previous.sha(contract_path) == contract_sha
            and previous.read(contract_path) == build(model_sha), "contract closure")
    contract = previous.read(contract_path)
    store = previous.base.Store(output)
    store.emit("run_started", prepare_sha256=contract_sha)
    rows, errors = [], []
    for role in contract["roles"]:
        row = {"arm": role["arm"], "status": "running", "transitions": [], "observations": []}
        env = None
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(previous.sha(inspect.getsourcefile(cls)) == previous.base.native.ENV_SHA, "env SHA")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            obs, info = env.reset(seed=contract["seed"])
            require(np.array_equal(np.asarray(obs["state"]), np.asarray(role["warmup_checkpoints"][0]["state"]))
                    and np.array_equal(np.asarray(info["goal_state"]), np.asarray(contract["goal_state"]))
                    and previous.base.hashlib.sha256(np.asarray(info["goal"]).tobytes()).hexdigest()
                    == contract["goal_rgb_sha256"], "reset binding")
            row["observations"].append(previous.base.capture(
                store, env, role["arm"], "warmup", 0, role["warmup_checkpoints"][0]))
            stages = (("warmup", np.zeros((10, 2), dtype="<f4")),
                      ("prefix", np.asarray(role["prefix"], dtype="<f4")),
                      ("old_tail", np.asarray(role["previous_first_packet"], dtype="<f4")),
                      ("new_tail", np.asarray(role["new_first_packet"], dtype="<f4")))
            for phase, actions in stages:
                for step, action in enumerate(actions, 1):
                    obs, reward, terminated, truncated, info = env.step(action.copy())
                    require(np.array_equal(env.latest_action, action), "action binding")
                    row["transitions"].append({"phase": phase, "step": step, "action": action.tolist(),
                                               "state": np.asarray(obs["state"]).tolist(),
                                               "reward": float(reward), "terminated": bool(terminated),
                                               "truncated": bool(truncated)})
                    if phase == "warmup" and step in (5, 10):
                        row["observations"].append(previous.base.capture(
                            store, env, role["arm"], phase, step, role["warmup_checkpoints"][[5, 10].index(step)+1]))
                    elif phase == "prefix":
                        row["observations"].append(previous.base.capture(
                            store, env, role["arm"], phase, step, role["prefix_checkpoints"][step-1]))
                    elif phase == "old_tail" and step == 5:
                        # Capture compares full physical snapshot and RGB to the original cp10.
                        row["observations"].append(previous.base.capture(
                            store, env, role["arm"], "tail", 5, role["cp10_checkpoint"]))
                    elif phase == "new_tail":
                        row["observations"].append(previous.base.capture(store, env, role["arm"], phase, step))
                    if terminated or truncated:
                        require(phase == "new_tail", "early stop before feedback checkpoint")
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
        store.json(role["arm"] + ".row.json", row)
        store.emit("branch_closed", arm=role["arm"], status=row["status"])
        rows.append(row)
    store.emit("run_finished", errors=errors)
    report = {"schema": NAME + ".report.v1", "status": "recorded" if not errors else "partial_or_failed",
              "prepare_sha256": contract_sha, "model_report_sha256": model_sha,
              "rows": rows, "errors": errors, "journal_sha256": previous.sha(store.path / "journal.jsonl"),
              "P1_admission": False, "independent_confirmation": False}
    store.json("report.json", report)
    return previous.sha(store.path / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "invalid tag")
    contract = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.sha256 and not args.authority and not os.path.lexists(output), "prepare only")
        print(previous.write_new(contract, build(args.model_sha)))
    else:
        require(args.sha256 and args.authority, "missing execution binding")
        print(execute(contract, args.sha256, args.model_sha, output, args.authority))


if __name__ == "__main__":
    main()
