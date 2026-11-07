"""Generate unfiltered witnessed TwoRoom development tasks, then evaluate RGB CEM300."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_GENERATED_WITNESS_DEV_PROTOCOL_20261107_CODEX.md"
PUBLIC = ROOT / "results/pc_tworoom_generated_witness_public_20261107_codex.npz"
TRUTH = ROOT / "results/pc_tworoom_generated_witness_truth_20261107_codex.npz"
REFERENCE = ROOT / "results/pc_tworoom_generated_witness_reference_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_generated_witness_cem300_20261107_codex.json"


def witness_actions(index):
    if type(index) is not int or not 0 <= index < 40:
        raise ValueError("invalid task index")
    blocks = np.random.default_rng(1779900 + index).uniform(-1, 1, (5, 2)).astype(np.float32)
    return np.repeat(blocks, 5, axis=0)


def prepare():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if any(path.exists() for path in (PUBLIC, TRUTH, REFERENCE)):
        raise FileExistsError("preparation outputs exist")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    images, goals, states, targets, actions, rows = [], [], [], [], [], []
    try:
        for i in range(40):
            seed = 1778800 + i
            env.reset(seed=seed)
            actor = env.unwrapped
            initial = actor.agent_position.detach().cpu().numpy().copy()
            image = actor.render().copy()
            witness = witness_actions(i)
            for action in witness:
                actor.step(action.copy())
            target = actor.agent_position.detach().cpu().numpy().copy()
            actor._set_goal_state(target)
            goal = actor._target_img.detach().cpu().numpy().transpose(1, 2, 0).copy()
            env.reset(seed=seed)
            actor = env.unwrapped
            if not np.array_equal(actor.agent_position.cpu().numpy(), initial) or not np.array_equal(actor.render(), image):
                raise ValueError("reset reproducibility failed")
            actor._set_goal_state(target)
            if not np.array_equal(actor._target_img.cpu().numpy().transpose(1, 2, 0), goal):
                raise ValueError("goal render mismatch")
            trace = []
            for action in witness:
                _, _, term, trunc, info = actor.step(action.copy())
                trace.append({"distance": float(info["distance_to_target"]),
                              "terminated": bool(term), "truncated": bool(trunc)})
            if not np.array_equal(actor.agent_position.cpu().numpy(), target) or not trace[-1]["terminated"]:
                raise ValueError("witness failed physical replay")
            rows.append({"seed": seed, "initial_success": bool(np.linalg.norm(initial-target) < 16),
                         "witness_trace": trace})
            images.append(image)
            goals.append(goal)
            states.append(initial)
            targets.append(target)
            actions.append(witness)
    finally:
        env.close()
    states, targets = np.asarray(states), np.asarray(targets)
    with h5py.File(source.common.DATA, "r") as h5:
        offsets, lengths = np.asarray(h5["ep_offset"]), np.asarray(h5["ep_len"])
        eligible = offsets[lengths > 25]
        pos = np.asarray(h5["pos_agent"])
        initial, final = pos[eligible], pos[eligible + 25]
        for i, row in enumerate(rows):
            row["hdf5_equal_initial_goal_position_pairs"] = int(
                np.sum(np.all(initial == states[i], axis=1) & np.all(final == targets[i], axis=1)))
    with PUBLIC.open("xb") as stream:
        np.savez_compressed(stream, seeds=np.arange(1778800, 1778840), current=np.asarray(images),
                            goal=np.asarray(goals))
    with TRUTH.open("xb") as stream:
        np.savez_compressed(stream, initial=states, target=targets, witness=np.asarray(actions))
    with REFERENCE.open("x") as stream:
        json.dump({"rows": rows, "public_sha256": file_hash(PUBLIC), "truth_sha256": file_hash(TRUTH),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "scope": "unfiltered witnessed synthetic development; no independence certification"},
                  stream, indent=2, allow_nan=False)
    print(f"sealed {len(rows)} witnessed generated tasks", flush=True)


def evaluate():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if RESULT.exists():
        raise FileExistsError(RESULT)
    reference = json.loads(REFERENCE.read_text())
    for name, path in (("public_sha256", PUBLIC), ("truth_sha256", TRUTH),
                       ("protocol_sha256", PROTOCOL), ("runner_sha256", __file__)):
        if reference[name] != file_hash(path):
            raise ValueError(f"reference mismatch: {name}")
    model, scaler, transform, normalizer = source.common.model_setup()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, PUBLIC,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; HDF5 action scaler",
                         clock="five environment steps per packet")
    boundary = RGBLeWM(model, audit)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Evaluate RGB CEM300 on all generated witnessed tasks; no outcome filtering.", flush=True)
    try:
        with np.load(PUBLIC, allow_pickle=False) as public, np.load(TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            for i, seed in enumerate(public["seeds"]):
                env.reset(seed=int(seed))
                actor = env.unwrapped
                if not np.array_equal(actor.agent_position.cpu().numpy(), truth["initial"][i]):
                    raise ValueError("reset state mismatch")
                actor._set_goal_state(truth["target"][i])
                if not np.array_equal(actor.render(), public["current"][i]):
                    raise ValueError("current image mismatch")
                goal = transform(public["goal"][i])[None, None, None].cuda()
                tail, trace, decisions = None, [], []
                for iteration in range(5):
                    remaining = 25 - len(trace)
                    current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                               "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    result = source.cem_temporal(boundary, current, scaler, tail, remaining,
                                                 17711000 + 10*i + iteration, 300)
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
                rows.append({"seed": int(seed), "initial_success": reference["rows"][i]["initial_success"],
                             "trace": trace, "decisions": decisions,
                             "success": any(x["terminated"] for x in trace)})
                print(f"task={i} initial={rows[-1]['initial_success']} success={rows[-1]['success']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with RESULT.open("x") as stream:
        json.dump({"track": RGB, "rows": rows, "public_sha256": file_hash(PUBLIC),
                   "reference_sha256": file_hash(REFERENCE), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer,
                   "scope": "witness-conditioned generated development tasks, not official or independent"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["evaluate"]:
        evaluate()
    else:
        raise SystemExit("expected prepare|evaluate")
