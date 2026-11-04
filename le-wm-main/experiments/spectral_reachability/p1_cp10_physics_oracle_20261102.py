"""Development-only PushT candidate oracle at two previously observed cp10 states.

Simulator state and goal are deliberately used by this *offline oracle*; its
output must never enter an RGB-only planner or count as independent evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import os
import time
from pathlib import Path

import numpy as np

from . import p1_second_feedback_saved_20261102 as saved

ROOT = saved.ROOT
NAME = "p1_cp10_physics_oracle_20261102"
AUTHORITY = "DEV303_CP10_PRIVILEGED_CANDIDATE_ORACLE"
ARMS = ("candidate_04_terminal_latent", "candidate_19_terminal_latent")
SEED = 20261102
COUNT = 64
require = saved.need


def bank(old, new):
    """Fixed 64-candidate packet bank; identical random candidates across arms."""
    old, new = np.asarray(old, dtype="<f4"), np.asarray(new, dtype="<f4")
    require(old.shape == new.shape == (5, 2) and np.isfinite(old).all() and np.isfinite(new).all()
            and (np.abs(old) <= 1).all() and (np.abs(new) <= 1).all(), "bounded old/new packets")
    rng = np.random.default_rng(SEED)
    # 31 constant-direction and 30 independent-action candidates plus zero.
    constant = rng.uniform(-1, 1, size=(31, 1, 2)).astype("<f4")
    independent = rng.uniform(-1, 1, size=(30, 5, 2)).astype("<f4")
    actions = np.concatenate((old[None], new[None], np.zeros((1, 5, 2), dtype="<f4"),
                              np.repeat(constant, 5, axis=1), independent))
    require(actions.shape == (COUNT, 5, 2), "candidate count")
    return actions


def protocol():
    verification = saved.analyze()
    original_contract = saved.physical.previous.read(saved.old.CONTRACT, saved.old.CONTRACT_SHA)
    new_contract = saved.physical.previous.read(saved.CONTRACT, saved.CONTRACT_SHA)
    require(verification["frames_verified"] == 112, "verified physical panel")
    old_by = {r["arm"]: r for r in original_contract["roles"]}
    new_by = {r["arm"]: r for r in new_contract["roles"]}
    rows = []
    for arm in ARMS:
        old, new = old_by[arm], new_by[arm]
        actions = bank(old["tail"][5:], new["new_first_packet"])
        rows.append({"arm": arm, "prefix": new["prefix"],
                     "previous_first_packet": new["previous_first_packet"],
                     "cp10_snapshot": new["cp10_checkpoint"]["snapshot"],
                     "candidate_actions": actions.tolist(),
                     "candidate_sha256": hashlib.sha256(actions.tobytes()).hexdigest()})
    return {"schema": NAME + ".protocol.v1", "classification": "privileged_development_oracle",
            "seed": new_contract["seed"], "goal_state": new_contract["goal_state"], "arms": rows,
            "goal_rgb_sha256": new_contract["goal_rgb_sha256"], "candidate_count": COUNT,
            "candidate_seed": SEED, "execution_horizon": 5,
            "oracle_score": "object position error + 40 * wrapped angle error; lower is better",
            "source_report_sha256": saved.REPORT_SHA,
            "environment_source_sha256": saved.physical.previous.base.native.ENV_SHA,
            "script_sha256": saved.physical.previous.sha(__file__),
            "uses_privileged_state": True, "planner_uses_oracle_outputs": False,
            "independent_confirmation": False, "P1_admission": False}


def objective(state, goal):
    state, goal = np.asarray(state), np.asarray(goal)
    require(state.shape == goal.shape == (7,) and np.isfinite(state).all() and np.isfinite(goal).all(),
            "finite state")
    position = float(np.linalg.norm(state[2:4] - goal[2:4]))
    angle = float(abs((state[4] - goal[4] + np.pi) % (2*np.pi) - np.pi))
    return {"position": position, "angle": angle, "score": position + 40 * angle}


def execute(path, digest, output, authority):
    prev = saved.physical.previous
    require(authority == AUTHORITY and prev.sha(path) == digest and prev.read(path) == protocol(),
            "sealed contract and authority")
    require(not os.path.lexists(output), "exclusive output")
    output = Path(output)
    output.mkdir()
    contract = prev.read(path)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    require(prev.sha(inspect.getsourcefile(cls)) == contract["environment_source_sha256"], "environment SHA")
    rows, started = [], time.monotonic()
    steps = 0
    try:
        for role in contract["arms"]:
            candidates = np.asarray(role["candidate_actions"], dtype="<f4")
            require(hashlib.sha256(candidates.tobytes()).hexdigest() == role["candidate_sha256"], "bank SHA")
            outputs = []
            for index, candidate in enumerate(candidates):
                env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                try:
                    _obs, info = env.reset(seed=contract["seed"])
                    require(np.array_equal(info["goal_state"], contract["goal_state"])
                            and hashlib.sha256(np.asarray(info["goal"]).tobytes()).hexdigest()
                            == contract["goal_rgb_sha256"], "goal")
                    replay = np.concatenate((np.zeros((10, 2), dtype="<f4"),
                                             np.asarray(role["prefix"], dtype="<f4"),
                                             np.asarray(role["previous_first_packet"], dtype="<f4")))
                    for action in replay:
                        _, _, terminated, truncated, _ = env.step(action.copy())
                        steps += 1
                        require(not terminated and not truncated and np.array_equal(env.latest_action, action),
                                "replay")
                    require(prev.base.native.snapshot(env) == role["cp10_snapshot"], "exact cp10 replay")
                    before = objective(env._get_obs(), contract["goal_state"])
                    for action_step, action in enumerate(candidate, 1):
                        obs, reward, terminated, truncated, _ = env.step(action.copy())
                        steps += 1
                        require(np.array_equal(env.latest_action, action) and not truncated, "oracle action")
                        if terminated:
                            break
                    result = objective(obs["state"], contract["goal_state"])
                    outputs.append({"index": index, "initial": before, "terminal": result,
                                    "reward": float(reward), "success": bool(terminated),
                                    "steps": action_step})
                finally:
                    env.close()
            selected = min(outputs, key=lambda x: (x["terminal"]["score"], x["index"]))
            row = {"arm": role["arm"], "candidate_sha256": role["candidate_sha256"],
                   "rows": outputs, "oracle_selected_index": selected["index"]}
            rows.append(row)
            prev.write_new(output / (role["arm"] + ".json"), row)
        require(prev.read(path) == protocol(), "postrun source closure")
        return prev.write_new(output / "report.json", {
            "schema": NAME + ".report.v1", "classification": "privileged_development_oracle",
            "contract_sha256": digest, "rows": rows, "physics_steps": steps,
            "wall_seconds": time.monotonic()-started, "independent_confirmation": False, "P1_admission": False})
    except Exception as exc:
        prev.write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                 "completed_arms": len(rows), "physics_steps": steps})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "tag")
    contract = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.sha256 and not args.authority and not os.path.lexists(output), "prepare only")
        print(saved.physical.previous.write_new(contract, protocol()))
    else:
        require(args.sha256 and args.authority, "missing authorization")
        print(execute(contract, args.sha256, output, args.authority))


if __name__ == "__main__":
    main()
