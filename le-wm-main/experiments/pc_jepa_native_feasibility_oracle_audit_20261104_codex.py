"""Recompute oracle accounting and independently replay its best saved trajectory."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from experiments import pc_jepa_native_feedback_dev_20261104_codex as prior

ROOT, SOURCE = prior.ROOT, prior.OUT
OUT = ROOT / "results/pc_jepa_native_feasibility_oracle_20261104_codex"


def main():
    destination = OUT / "audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    result = json.loads((OUT / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("run incomplete")
    for path, expected in result["bindings"].items():
        if prior.sha(Path(path)) != expected:
            raise ValueError(f"binding changed: {path}")
    for name, key in (("actions.npy", "actions_sha256"), ("best_trajectory.npz", "best_trajectory_sha256")):
        if prior.sha(OUT / name) != result[key]:
            raise ValueError(f"artifact changed: {name}")
    actions = np.load(OUT / "actions.npy", allow_pickle=False)
    target = np.load(SOURCE / "goal_state_evaluation_only.npy", allow_pickle=False)
    rows = result["rows"]
    if actions.shape != (194, 15, 2) or len(rows) != 194 or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
        raise ValueError("invalid action bank")
    costs = []
    for index, row in enumerate(rows):
        state = np.asarray(row["terminal_state"], np.float64)
        joint = float(np.linalg.norm(state[:4] - target[:4]))
        angle = float(abs((state[4] - target[4] + np.pi) % (2*np.pi) - np.pi))
        cost = (joint / 20.)**2 + (angle / (np.pi / 9.))**2
        if row["index"] != index or abs(cost - row["cost"]) > 1e-10:
            raise ValueError("incorrect row cost")
        if row["metrics"]["success"] != bool(joint < 20 and angle < np.pi/9):
            raise ValueError("incorrect success")
        costs.append(cost)
    if result["best_index"] != int(np.argmin(costs)):
        raise ValueError("incorrect best")
    anchors = {}
    for arm in ("response", "cem"):
        anchors[arm] = np.asarray([json.loads((SOURCE / f"{arm}_physical/command_{i:02d}.json").read_text())["action"]
                                   for i in range(11, 26)], np.float64)
    np.testing.assert_array_equal(actions[0], anchors["response"])
    np.testing.assert_array_equal(actions[1], anchors["cem"])
    rng, cursor = np.random.default_rng(20261104), 2
    for arm in ("zero", "response", "cem"):
        mean = np.zeros((15, 2)) if arm == "zero" else anchors[arm].copy()
        sigma = np.full((15, 2), .5)
        for generation in range(4):
            record = result["generations"][{"zero": 0, "response": 4, "cem": 8}[arm] + generation]
            np.testing.assert_array_equal(record["mean_before"], mean)
            np.testing.assert_array_equal(record["sigma_before"], sigma)
            sample = np.clip(rng.normal(mean, sigma, size=(16, 15, 2)), -1, 1)
            np.testing.assert_array_equal(sample, actions[cursor:cursor + 16])
            elite_indices = np.argsort(costs[cursor:cursor + 16], kind="stable")[:4]
            np.testing.assert_array_equal(elite_indices, record["elite_indices"])
            elite = sample[elite_indices]
            mean, sigma = elite.mean(axis=0), np.maximum(elite.std(axis=0), .05)
            cursor += 16
    if result["successful_indices"] != [i for i, row in enumerate(rows) if row["metrics"]["success"]]:
        raise ValueError("incorrect success list")
    base = prior.prior.helpers(ROOT)
    runtime, _ = base.load_runtime(ROOT)
    spec = importlib.util.spec_from_file_location("oracle_audit_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    env = native.PushTEnv(**base.CTOR)
    env.seed(prior.prior.SEED)
    obs, state = env.reset()
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for action in manifest["warmup"]:
        obs, _, done, info = env.step(np.asarray(action))
        if done:
            raise ValueError("warmup stopped")
        state = info["state"]
    with np.load(OUT / "best_trajectory.npz", allow_pickle=False) as trajectory:
        np.testing.assert_array_equal(trajectory["actions"], actions[result["best_index"]])
        for frame in range(16):
            np.testing.assert_array_equal(obs["visual"], trajectory["rgb"][frame])
            np.testing.assert_array_equal(state, trajectory["state"][frame])
            np.testing.assert_array_equal(obs["proprio"], trajectory["proprio"][frame])
            if frame < 15:
                obs, _, done, info = env.step(trajectory["actions"][frame])
                if done:
                    raise ValueError("replay stopped")
                state = info["state"]
    env.close()
    if result["successful_indices"]:
        raise ValueError("success requires separate witness replay")
    audit = {"status": "passed", "result_sha256": prior.sha(OUT / "result.json"),
             "audit_script_sha256": prior.sha(Path(__file__)), "candidates": len(rows),
             "best_index": result["best_index"], "best_replay_exact_frames": 16,
             "success_count": len(result["successful_indices"]),
             "scope": "full random sampling and saved arithmetic audited; best trajectory physically replayed"}
    with destination.open("x") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
