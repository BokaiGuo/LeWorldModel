"""Prespecified cross-wall development tasks, physical witnesses, and RGB CEM baselines."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_CROSSWALL_MECHANISM_DEV_PROTOCOL_20261107_CODEX.md"
PUBLIC = ROOT / "results/pc_tworoom_crosswall_mechanism_public_20261107_codex.npz"
TRUTH = ROOT / "results/pc_tworoom_crosswall_mechanism_truth_20261107_codex.npz"
REFERENCE = ROOT / "results/pc_tworoom_crosswall_mechanism_reference_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_crosswall_mechanism_cem_20261107_codex.json"


def task(index):
    if type(index) is not int or not 0 <= index < 40:
        raise ValueError("invalid task")
    rng = np.random.default_rng(1888800 + index)
    left, right = rng.uniform(75, 91), rng.uniform(133, 149)
    start_y, goal_y = rng.uniform(70, 100, size=2)
    return (np.asarray([left if index % 2 == 0 else right, start_y], np.float32),
            np.asarray([right if index % 2 == 0 else left, goal_y], np.float32))


def waypoint_action(position, goal, start_left, speed, stage):
    position, goal = np.asarray(position), np.asarray(goal)
    if position.shape != (2,) or goal.shape != (2,) or speed <= 0 or stage not in (0, 1, 2):
        raise ValueError("invalid witness state")
    points = (np.array([96 if start_left else 128, 49.]),
              np.array([128 if start_left else 96, 49.]), goal)
    while stage < 2 and np.max(np.abs(position - points[stage])) < 1e-4:
        stage += 1
    return np.clip((points[stage] - position) / speed, -1, 1).astype(np.float32), stage


def prepare():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if any(p.exists() for p in (PUBLIC, TRUTH, REFERENCE)):
        raise FileExistsError("task preparation outputs exist")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    images, goals, starts, targets, witnesses, rows = [], [], [], [], [], []
    try:
        for i in range(40):
            start, target = task(i)
            env.reset(seed=1888800+i, options={"state": start, "target_state": target})
            actor = env.unwrapped
            if actor.wall_axis != 1 or not np.array_equal(actor.agent_position.cpu().numpy(), start):
                raise ValueError("unexpected task contract")
            current = actor.render().copy()
            goal = actor._target_img.cpu().numpy().transpose(1, 2, 0).copy()
            speed = float(actor.variation_space["agent"]["speed"].value.item())
            trace, actions, stage = [], [], 0
            for _ in range(25):
                action, stage = waypoint_action(actor.agent_position.cpu().numpy(), target, i % 2 == 0, speed, stage)
                _, _, term, trunc, info = actor.step(action.copy())
                actions.append(action)
                trace.append({"distance": float(info["distance_to_target"]), "terminated": bool(term),
                              "truncated": bool(trunc), "stage": stage})
            rows.append({"seed": 1888800+i, "witness_success": any(t["terminated"] for t in trace),
                         "initial_success": bool(np.linalg.norm(start-target) < 16), "witness_trace": trace})
            images.append(current)
            goals.append(goal)
            starts.append(start)
            targets.append(target)
            witnesses.append(actions)
    finally:
        env.close()
    with PUBLIC.open("xb") as stream:
        np.savez_compressed(stream, seeds=np.arange(1888800, 1888840), current=np.asarray(images), goal=np.asarray(goals))
    with TRUTH.open("xb") as stream:
        np.savez_compressed(stream, initial=np.asarray(starts), target=np.asarray(targets),
                            witness=np.asarray(witnesses))
    with REFERENCE.open("x") as stream:
        json.dump({"rows": rows, "public_sha256": file_hash(PUBLIC), "truth_sha256": file_hash(TRUTH),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "scope": "prespecified cross-wall development; privileged witnesses are evaluation only"},
                  stream, indent=2, allow_nan=False)
    print(f"sealed 40 tasks, witnessed success={sum(r['witness_success'] for r in rows)}", flush=True)


def evaluate():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if RESULT.exists():
        raise FileExistsError(RESULT)
    reference = json.loads(REFERENCE.read_text())
    for key, path in (("public_sha256", PUBLIC), ("truth_sha256", TRUTH),
                      ("protocol_sha256", PROTOCOL), ("runner_sha256", __file__)):
        if reference[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    model, scaler, transform, normalizer = source.common.model_setup()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, PUBLIC,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, HDF5 action normalizer",
                         clock="five physical steps per packet")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Run CEM16 and CEM300 RGB feedback on all prespecified cross-wall tasks.", flush=True)
    try:
        with np.load(PUBLIC, allow_pickle=False) as public, np.load(TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            for i, seed in enumerate(public["seeds"]):
                goal = transform(public["goal"][i])[None, None, None].cuda()
                arms = {}
                for budget in (16, 300):
                    env.reset(seed=int(seed), options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    np.testing.assert_array_equal(actor._target_img.cpu().numpy().transpose(1, 2, 0), public["goal"][i])
                    wrapper = RGBLeWM(model, audit)
                    trace, decisions, tail = [], [], None
                    for iteration in range(5):
                        remaining = 25 - len(trace)
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        result = source.cem_temporal(wrapper, current, scaler, tail, remaining,
                                                     18811000 + i * 10 + iteration, budget)
                        actions, dispatched = result["actions"], []
                        for action in actions[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            dispatched.append(action.tolist())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        decisions.append({"remaining": remaining, "chosen": result["chosen"],
                                          "chosen_cost": result["chosen_cost"], "queries": result["queries"],
                                          "packet_steps": result["packet_steps"], "actions": actions.tolist(),
                                          "dispatched": dispatched})
                        tail = actions[len(dispatched):].copy()
                        if term or trunc:
                            break
                    arms[str(budget)] = {"trace": trace, "decisions": decisions,
                                         "success": any(x["terminated"] for x in trace)}
                rows.append({"seed": int(seed), "initial_success": reference["rows"][i]["initial_success"],
                             "witness_success": reference["rows"][i]["witness_success"], "arms": arms})
                print(f"task={i} witness={rows[-1]['witness_success']} 16={arms['16']['success']} "
                      f"300={arms['300']['success']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "track": RGB, "reference_sha256": file_hash(REFERENCE),
                   "public_sha256": file_hash(PUBLIC), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer,
                   "scope": "cross-wall mechanism development, not official or independent"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["evaluate"]:
        evaluate()
    else:
        raise SystemExit("expected prepare|evaluate")
