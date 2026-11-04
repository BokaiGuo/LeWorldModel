"""Replay two fixed multistart trajectories and audit saved plan tails."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import LEGACY, RUNTIME, ShadowRGBBridge, _scene

ROOT = prior.ROOT
MULTI = ROOT / "results/pc_shadow_multistart_dev_20261103_CODEX/multistart"
OLD = ROOT / "results/pc_shadow_spectral_paired_dev_20261103_CODEX"
OUTPUT = ROOT / "results/pc_shadow_multistart_replay_timing_audit_20261103_CODEX.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_MULTISTART_REPLAY_TIMING_AUDIT_PROTOCOL_20261103_CODEX.md"
WORKER = Path(__file__).with_name("pc_shadow_spectral_legacy_position_worker_20261103_codex.py")
ROSTER = ((0, "r8343_a0_p2"), (8, "r8345_a0_p2"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_audit(directory):
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        actions = data["actions"].copy()
        final_score = float(data["final_rgb_score"])
    records = []
    for i in range(5):
        with np.load(directory / f"baseline_command{13+i}.npz", allow_pickle=False) as data:
            plan = data["plan_raw"].copy()
            scores = data["final_scores"].copy()
            winner = int(data["winner"])
        np.testing.assert_allclose(plan[0], actions[12+i], rtol=0, atol=1e-7)
        available = 4-i
        delta = np.linalg.norm(plan[1:1+available] - actions[13+i:17], axis=1)
        records.append({"decision": i, "model_horizon": 5, "executed_before_replan": 1,
                        "available_following_actions": available, "different_following_actions": int((delta > 1e-6).sum()),
                        "max_l2_difference": float(delta.max()) if available else None,
                        "planned_terminal_score": float(scores[winner]),
                        "realized_score_at_matching_clock": final_score if i == 0 else None})
    return records


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    events = [json.loads(line) for line in (MULTI / "events.jsonl").read_text().splitlines()]
    completed = {e["root"]: e for e in events if e["event"] == "branch_completed"}
    if len(completed) != 2 or events[-1]["event"] != "completed":
        raise ValueError("incomplete multistart run")
    _, _, names = prior.files()
    starts, actions, rows = [], [], []
    with np.load(prior.SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        for idx, name in ROSTER:
            if names[idx] != name:
                raise ValueError("root mismatch")
            directory = MULTI / (name + "_baseline")
            path = directory / "trajectory.npz"
            if sha(path) != completed[name]["trajectory_sha256"]:
                raise ValueError("trajectory hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                sequence, frames = data["actions"].copy(), data["frames"].copy()
            with ShadowRGBBridge(source["starts"][idx]) as bridge:
                np.testing.assert_array_equal(bridge.initial_rgb, frames[0])
                for tick, action in enumerate(sequence, 1):
                    np.testing.assert_array_equal(bridge.step(action[None]), frames[tick])
            starts.append(source["starts"][idx].copy())
            actions.append(sequence)
            rows.append({"root": name, "rgb_frames_equal": 18, "plan_audit": plan_audit(directory),
                         "rgb_score": completed[name]["final_rgb_score"]})
    starts, actions = np.repeat(np.asarray(starts), 2, axis=0), np.repeat(np.asarray(actions), 2, axis=0)
    with tempfile.TemporaryDirectory(prefix="pc_multistart_eval_") as directory:
        folder = Path(directory)
        request, xml, response = folder / "request.npz", folder / "scene.xml", folder / "response.npz"
        np.savez_compressed(request, starts=starts, actions=actions)
        xml.write_bytes(_scene())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                            "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request), str(xml), str(response)],
                       env=env, check=True, timeout=60)
        with np.load(response, allow_pickle=False) as replay:
            if str(replay["request_sha256"].item()) != sha(request):
                raise ValueError("replay input mismatch")
            states = replay["states"].copy()
    np.testing.assert_array_equal(states[0], states[1])
    np.testing.assert_array_equal(states[2], states[3])
    endpoints = prior.h.physical.replay_position_endpoints(starts, actions)
    goal = np.asarray(prior.h.GOAL[:2], dtype=float)
    for index, row in enumerate(rows):
        state = states[2*index]
        row["physical_states_eval_only"] = state.tolist()
        row["final_distance"] = float(np.linalg.norm(state[-1, :2] - goal))
        row["endpoint_21_vs_22_position_max_abs"] = float(np.max(np.abs(state[-1, :2] - endpoints[2*index, :2])))
    old_audits = []
    for _, name in ROSTER:
        for arm in ("baseline", "spectral"):
            directory = OLD / f"{name}_{arm}"
            # Older arm names are embedded in snapshot filenames.
            with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
                saved_actions, final_score = data["actions"].copy(), float(data["final_rgb_score"])
            records = []
            for i in range(5):
                with np.load(directory / f"{arm}_command{13+i}.npz", allow_pickle=False) as data:
                    plan, score, winner = data["plan_raw"].copy(), data["final_scores"].copy(), int(data["winner"])
                np.testing.assert_allclose(plan[0], saved_actions[12+i], rtol=0, atol=1e-7)
                available = 4-i
                delta = np.linalg.norm(plan[1:1+available] - saved_actions[13+i:17], axis=1)
                records.append({"decision": i, "available_following_actions": available,
                                "different_following_actions": int((delta > 1e-6).sum()),
                                "max_l2_difference": float(delta.max()) if available else None,
                                "planned_terminal_score": float(score[winner]),
                                "realized_score_at_matching_clock": final_score if i == 0 else None})
            old_audits.append({"root": name, "arm": arm, "records": records})
    result = {"schema": "pc_shadow_multistart_replay_timing.v1", "protocol_sha256": sha(PROTOCOL),
              "multistart_journal_sha256": sha(MULTI / "events.jsonl"),
              "prior_journal_sha256": sha(OLD / "events.jsonl"),
              "rows": rows, "prior_plan_audits": old_audits, "independent_confirmation": False}
    with OUTPUT.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
