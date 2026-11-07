"""Compare spawned RGB planner with a sealed in-process feedback trace."""
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401
from sklearn.preprocessing import StandardScaler

from experiments import pc_shared_rgb_feedback_smoke_20261107_codex as source
from experiments import pc_rgb_hold_retention_search_20261107_codex as search
from experiments import pc_rgb_spawn_planner_20261107_codex as worker
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit

ROOT = source.ROOT
OLD = ROOT / "results/pc_tworoom_rgb_hold_retention_feedback_20261107_codex.json"
OUT = ROOT / "results/pc_tworoom_rgb_spawn_feedback_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RGB_SPAWN_BOUNDARY_PROTOCOL_20261107_CODEX.md"
INPUT = source.source.block.prior.inputs.INPUT
COMMON = source.source.block.prior.common


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads(OLD.read_text())
    if previous["track"] != RGB or file_hash(INPUT) != previous["input_sha256"]:
        raise ValueError("source track or input mismatch")
    if file_hash(search.__file__) != previous["search_sha256"]:
        raise ValueError("search implementation changed")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, worker.__file__, search.__file__,
                         COMMON.CHECKPOINT / "weights.pt", COMMON.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="remaining deadline, five environment steps per packet")
    with h5py.File(COMMON.DATA, "r") as data:
        action_data = data["action"][:]
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_hash = hashlib.sha256(np.ascontiguousarray(action_data).tobytes()).hexdigest()
    if action_hash != previous["normalizer_sha256"]:
        raise ValueError("normalizer data drift")
    scaler = StandardScaler().fit(action_data)
    start = time.perf_counter()
    planner = worker.SpawnPlanner(scaler.mean_, scaler.scale_, audit, previous["model_sha256"])
    startup = time.perf_counter() - start
    if planner.ready["status"] != "ready":
        raise ValueError("planner startup failed")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(INPUT, allow_pickle=False) as inputs, h5py.File(COMMON.DATA, "r") as data:
            for root in (10, 11):
                old = next(r for r in previous["rows"] if r["root"] == root)["arms"]["lookahead"]
                task = int(inputs["rows"][root])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task + 25])
                tail, decisions, trace = None, [], []
                for iteration in range(5):
                    request = {"pixels": actor.render().copy(), "goal": inputs["goal"][root].copy(),
                               "old_tail": tail, "remaining_steps": 25 - len(trace),
                               "candidate_seed": 27121000 + root * 10 + iteration,
                               "strategy_seed": 27122000 + root * 10 + iteration,
                               "strategy": "lookahead"}
                    start = time.perf_counter()
                    reply = planner.plan(request)
                    ipc_seconds = time.perf_counter() - start
                    ref = old["decisions"][iteration]
                    chosen = np.asarray(reply["actions"], dtype=np.float32)
                    np.testing.assert_array_equal(chosen, np.asarray(ref["actions"], dtype=np.float32))
                    np.testing.assert_allclose(reply["choice"]["cost"], ref["choice"]["cost"], atol=.002, rtol=0)
                    if (reply["choice"]["source"], reply["choice"]["index"], reply["queries"], reply["forward_calls"]) != \
                       (ref["choice"]["source"], ref["choice"]["index"], ref["queries"], ref["forward_calls"]):
                        raise ValueError("planner decision metadata mismatch")
                    dispatched = 0
                    for action in chosen[:5]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        reference = old["trace"][len(trace)]
                        np.testing.assert_allclose(action, reference["action"], atol=0, rtol=0)
                        np.testing.assert_allclose(info["distance_to_target"], reference["distance"], atol=1e-4, rtol=0)
                        if (bool(term), bool(trunc)) != (reference["terminated"], reference["truncated"]):
                            raise ValueError("physical outcome mismatch")
                        trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        dispatched += 1
                        if term or trunc:
                            break
                    decisions.append({"choice": reply["choice"], "queries": reply["queries"],
                                      "forward_calls": reply["forward_calls"], "ipc_seconds": ipc_seconds,
                                      "actions": chosen.tolist(), "receipts": reply["receipts"]})
                    tail = chosen[dispatched:].copy()
                    if term or trunc:
                        break
                if len(trace) != len(old["trace"]) or len(decisions) != len(old["decisions"]):
                    raise ValueError("episode length mismatch")
                rows.append({"root": root, "decisions": decisions, "trace": trace,
                             "ever_success": any(x["terminated"] for x in trace)})
                print(f"root={root}: {len(trace)} steps, exact plan parity", flush=True)
    finally:
        env.close()
        stopped = planner.close()
    if stopped["model_sha256"] != previous["model_sha256"]:
        raise ValueError("model changed in spawned planner")
    with OUT.open("x") as f:
        json.dump({"track": RGB, "scope": "two exposed roots, process boundary smoke only",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "worker_sha256": file_hash(worker.__file__), "prior_sha256": file_hash(OLD),
                   "model_sha256": previous["model_sha256"], "normalizer_sha256": action_hash,
                   "startup_seconds": startup, "rows": rows}, f, indent=2, allow_nan=False)
        f.write("\n")
    print("spawned RGB feedback parity complete", flush=True)


if __name__ == "__main__":
    main()
