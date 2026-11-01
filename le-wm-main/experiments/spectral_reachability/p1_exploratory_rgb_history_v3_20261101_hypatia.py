"""Two fixed exploratory roots, actual RGB history; never a freshness certificate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as source

ROOT = source.PROJECT
NAME = "p1_exploratory_rgb_history_v3_20261101_hypatia"
REQUEST = ROOT / "results/p1_exploration_seed_preseal_20261101_hypatia_request.json"
RUN = ROOT / "results/p1_exploration_seed_preseal_20261101_hypatia_run01"
RECEIPT = ROOT / "results/p1_exploration_run01_receipt_20261101_hypatia_LOCAL02.json"
PROTOCOL = ROOT.parent / "docs/D2P_P1_EXPLORATORY_RGB_HISTORY_V3_20261101_HYPATIA.md"
ENV = ROOT / ".venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py"
PINNED = {
    REQUEST: "bcc5d39dcb85b0f7fba1c66f718ec00acf1909871568023ee809338a10814b53",
    RUN / "sourcepack.json": "5312e3eec05c91698d3689ce4dcbe08732ca3c76800e7cbb5fe7d5016c1465ab",
    RUN / "journal.jsonl": "eadbb66295647e3ff97c17428f0f521ce27082d2aeed9f6d65c7af8a39955800",
    RUN / "source_0.npz": "1163eb2b5efa1897c5ed106351e5f8e65ada27041a3eacbb3c9d5f1232d92b19",
    RUN / "source_1.npz": "356fe60e6499ee5a1957a3c01189826607ea5cf72afd630186ca71569cb7c0f1",
    RECEIPT: "c30698c00d66c30f5d822ab14dd4bf333d679cb0cfcd5047cf2280123f78c66e",
    ENV: "c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2",
    ROOT / "results/p1_exploratory_rgb_history_v2_20261101_hypatia_RUN01/receipt.json": "d4a14f4f000ca345d9cdd233b14e1ac3ba6e0d4e9f366c43334b90f04ce9b832",
    ROOT / "results/p1_exploratory_rgb_history_v2_20261101_hypatia_RUN01/journal.jsonl": "387c51c59f710c9df51aac43fc417c545bd9cdf970aa47bdad3949dc2db0121c",
    ROOT / ".venv/lib/python3.12/site-packages/stable_worldmodel/envs/__init__.py": "118a262677a647c6d6ef8889443b096abe082db4e365fc0a5b1249715e7e3c1a",
}
PREPARED = ROOT / f"results/{NAME}_PREP01.json"
OUTPUT = ROOT / f"results/{NAME}_RUN01"


def sha(path):
    return source.file_sha(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        require(stream.write(data) == len(data), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def prepare():
    for path, digest in PINNED.items():
        require(sha(path) == digest, "pinned input drift: " + str(path))
    receipt = json.loads(RECEIPT.read_bytes())
    require(receipt["freshness"] == "unknown" and receipt["P1_admission"] is False
            and len(receipt["selected_sources"]) == 2, "source receipt")
    require(all(not r["known_history_hits"] and not r["batch_hits"] and not r["historical_journal_hits"]
                for r in receipt["reset_comparisons"]), "known collision in saved source")
    files = dict(PINNED)
    for path in (Path(__file__).resolve(), Path(source.__file__).resolve(), PROTOCOL):
        files[path] = sha(path)
    return {"schema": NAME + ".prepare.v1", "files": {str(k): v for k, v in files.items()},
            "slots": [0, 1], "actions_float32": [[0.1, 0.0]] * 5 + [[0.0, 0.1]] * 5,
            "capture_after_steps": [0, 5, 10], "max_resets": 4, "max_public_steps": 20,
            "max_explicit_renders": 6, "retry": False, "selection_by_outcome": False,
            "scope": "target-on evaluator only; forbidden as moving-fresh planner observation",
            "with_target": True, "native_gate_unchanged": True,
            "historical_coverage": "unknown", "training_disjoint": "unknown", "freshness": "unknown"}


def equal_observation(obs, info, frame, packet):
    return (np.array_equal(obs["state"], packet["state"])
            and np.array_equal(info["goal_state"], packet["goal"])
            and np.array_equal(info["goal"], packet["goal_rgb"])
            and np.array_equal(frame, packet["rgb_history"][0]))


def frame_check(frame):
    value = np.asarray(frame)
    require(value.shape == (224, 224, 3) and value.dtype == np.uint8, "native RGB shape")
    return value


def angles(goal, state, native_result):
    raw = abs(float(goal[4]) - float(state[4]))
    native = min(raw, 2 * math.pi - raw)
    wrapped = abs(math.atan2(math.sin(float(state[4]) - float(goal[4])),
                             math.cos(float(state[4]) - float(goal[4]))))
    position = float(np.linalg.norm(np.asarray(goal[:4]) - np.asarray(state[:4])))
    return {"native_eval_success": bool(native_result[0]), "native_eval_state_distance": float(native_result[1]),
            "native_angle_expression": native, "strict_wrapped_angle": wrapped,
            "wrapped_diagnostic_success": bool(position < 20 and wrapped < math.pi / 9),
            "strict_value_is_diagnostic_only": True}


class Journal:
    def __init__(self, path):
        self.stream = path.open("xb")
        self.previous = None
        self.index = 0

    def emit(self, kind, **fields):
        body = {"index": self.index, "kind": kind, "previous_sha256": self.previous, **fields}
        data = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.previous = hashlib.sha256(data).hexdigest()
        self.stream.write((json.dumps({**body, "sha256": self.previous}, sort_keys=True, allow_nan=False) + "\n").encode())
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.index += 1


def save_frame(directory, slot, step, frame):
    path = directory / f"slot{slot}_rgb{step:02}.npy"
    with path.open("xb") as stream:
        np.save(stream, frame, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": path.name, "file_sha256": sha(path),
            "pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest()}


def execute(prepared, prepare_sha):
    require(sha(PREPARED) == prepare_sha, "prepare SHA mismatch")
    for path, digest in prepared["files"].items():
        require(sha(path) == digest, "prepared input drift: " + path)
    require(prepared == prepare(), "prepared contract drift")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    request = json.loads(REQUEST.read_bytes())
    p = request["plan"]
    OUTPUT.mkdir()
    journal = Journal(OUTPUT / "journal.jsonl")
    journal.emit("run_started", prepare_sha256=prepare_sha, contract=prepared)
    results = []
    for slot_id in prepared["slots"]:
        env = None
        status, error = "not_started", None
        steps = 0
        try:
            packet = source.load_raw_source(RUN / f"source_{slot_id}.npz", p)
            meta = packet["metadata"]
            slot = p["slots"][slot_id]
            require(slot["candidate_id"] == slot_id and meta["candidate_id"] == slot_id, "slot mapping")
            env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=True)
            require(env.unwrapped.with_target is True, "target rendering binding")
            seed = meta["selected_attempt"]["random_reset_seed"]
            journal.emit("reset_started", slot=slot_id, phase="original_seed", seed=seed)
            _, info = env.reset(seed=seed)
            journal.emit("reset_completed", slot=slot_id, phase="original_seed",
                         goal_state=source.vector(info["goal_state"]).tolist())
            goal = source.vector(info["goal_state"])
            requested = source.sample_state(goal, list(env.unwrapped.block.shapes), slot,
                                            meta["source"]["proposal_index"],
                                            meta["selected_attempt"]["attempt_index"], p)
            journal.emit("reset_started", slot=slot_id, phase="placement", seed=0,
                         requested_state=requested.tolist(), goal_state=goal.tolist())
            obs, info = env.reset(seed=0, options={"state": requested.copy(), "goal_state": goal.copy()})
            journal.emit("reset_completed", slot=slot_id, phase="placement",
                         actual_state=source.vector(obs["state"]).tolist(),
                         target_visible_flag=bool(env.unwrapped.with_target
                                                  and env.unwrapped.variation_space["rendering"]["render_goal"].value),
                         renderer_goal_pose=np.asarray(env.unwrapped.goal_pose).tolist(),
                         angle_diagnostics=angles(goal, obs["state"], env.unwrapped.eval_state(goal, obs["state"])))
            journal.emit("render_started", slot=slot_id, step=0)
            frame = frame_check(env.render())
            journal.emit("render_completed", slot=slot_id, step=0, **save_frame(OUTPUT, slot_id, 0, frame))
            if not equal_observation(obs, info, frame, packet):
                raise ValueError("initial state/goal/RGB differs from sealed source")
            journal.emit("initial_observation_bound", slot=slot_id, source_sha256=PINNED[RUN / f"source_{slot_id}.npz"])
            actions = np.asarray(prepared["actions_float32"], dtype=np.float32)
            status = "running"
            for i, action in enumerate(actions, 1):
                journal.emit("step_started", slot=slot_id, step=i, raw_float32_bits=action.tobytes().hex())
                obs, _, terminated, truncated, _ = env.step(action.copy())
                actual = np.asarray(env.unwrapped.latest_action)
                require(actual.dtype == np.float32 and actual.shape == (2,)
                        and actual.tobytes() == action.tobytes(), "native action mismatch")
                steps = i
                journal.emit("step_completed", slot=slot_id, step=i, state=source.vector(obs["state"]).tolist(),
                             raw_float32_bits=actual.tobytes().hex(),
                             terminated=bool(terminated), truncated=bool(truncated),
                             angle_diagnostics=angles(goal, obs["state"], env.unwrapped.eval_state(goal, obs["state"])))
                if i in (5, 10):
                    journal.emit("render_started", slot=slot_id, step=i)
                    frame = frame_check(env.render())
                    journal.emit("render_completed", slot=slot_id, step=i, **save_frame(OUTPUT, slot_id, i, frame))
                if terminated or truncated:
                    status = "stopped_early" if i < 10 else "completed_with_stop"
                    break
            else:
                status = "completed"
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    journal.emit("cleanup_failed", slot=slot_id, error=f"{type(exc).__name__}: {exc}")
                    status = "failed"
        row = {"slot": slot_id, "status": status, "completed_steps": steps, "error": error}
        results.append(row)
        journal.emit("slot_finished", **row)
    journal.emit("run_finished", slots=results)
    journal.stream.close()
    report = {"schema": NAME + ".receipt.v1", "prepare_sha256": prepare_sha, "slots": results,
              "journal_sha256": sha(OUTPUT / "journal.jsonl"), "history_complete": False,
              "training_disjoint": "unknown", "freshness": "unknown", "P1_admission": False,
              "planner_observation_authorized": False, "target_render_mode": True}
    write(OUTPUT / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"path": str(PREPARED), "sha256": write(PREPARED, prepare())}))
    else:
        require(args.authorize_physics and args.prepare_sha256, "explicit physics authorization and prepare SHA")
        result = execute(json.loads(PREPARED.read_bytes()), args.prepare_sha256)
        print(json.dumps({"slots": result["slots"], "freshness": "unknown", "P1_admission": False}))


if __name__ == "__main__":
    main()
