"""Compare temporal selection on an existing fully replayed candidate pool."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_scored_first_pool_run_20261107_codex import BASE
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT

PROTOCOL = ROOT.parent / "docs/PC_PUSHT_FULL_POOL_PREDICTED_DEADLINE_POSTHOC_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_pusht_full_pool_predicted_deadline_20261107_codex.json"


def choose(costs, mode):
    costs = np.asarray(costs, np.float64)
    if costs.ndim != 2 or costs.shape[1] != 5 or not np.isfinite(costs).all():
        raise ValueError("expected finite [candidate, five packet] costs")
    if mode == "endpoint":
        index = int(np.argmin(costs[:, -1]))
        return index, 4
    if mode == "min_predicted":
        index, moment = np.unravel_index(np.argmin(costs), costs.shape)
        return int(index), int(moment)
    raise ValueError("unknown scoring rule")


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    rows, sources = [], [PROTOCOL, Path(__file__)]
    try:
        with h5py.File(DATA, "r") as data:
            for task in range(4):
                source = BASE / f"physical_{task}.json"
                physical = json.loads(source.read_text())
                if physical["task"] != task or physical["unique"] != 630:
                    raise ValueError("unexpected task or candidate count")
                batches = {}
                for call in range(5):
                    path = BASE / "capture" / f"call_{task * 5 + call:03}.pt"
                    batches[call] = torch.load(path, map_location="cpu", weights_only=True)
                    sources.append(path)
                sources.append(source)
                costs, identities = [], []
                goal_state = np.asarray(data["state"][physical["data_row"] + 25], dtype=np.float64)
                goal_fingerprint = None
                for i, item in enumerate(physical["records"]):
                    if item["index"] != i:
                        raise ValueError("candidate order mismatch")
                    occurrence = item["occurrences"][0]
                    batch = batches[occurrence["call"]]
                    slot = occurrence["slot"]
                    action = batch["actions"][0, slot].numpy()
                    if hashlib.sha256(action.tobytes()).hexdigest() != item["action_sha256"]:
                        raise ValueError("action hash mismatch")
                    goal = batch["goal"][0, -1].numpy()
                    goal_hash = hashlib.sha256(goal.tobytes()).hexdigest()
                    if goal_fingerprint is None:
                        goal_fingerprint = goal_hash
                    if goal_hash != goal_fingerprint:
                        raise ValueError("goal varies within candidate pool")
                    predicted = batch["prediction"][0, slot, 1:].numpy()
                    if predicted.shape != (5, goal.shape[0]):
                        raise ValueError("prediction shape mismatch")
                    timeline = np.square(predicted - goal).sum(axis=-1)
                    np.testing.assert_allclose(timeline[-1], occurrence["cost"], atol=0.002, rtol=0.0002)
                    costs.append(timeline)
                    identities.append(item["action_sha256"])
                decisions = {}
                for mode in ("endpoint", "min_predicted"):
                    index, moment = choose(costs, mode)
                    item = physical["records"][index]
                    state = np.asarray(item["packet_states"][moment], dtype=np.float64)
                    success = bool(actor.eval_state(goal_state, state)[0])
                    final = bool(actor.eval_state(goal_state, np.asarray(item["packet_states"][-1]))[0])
                    if final != item["final_success"]:
                        raise ValueError("native final success mismatch")
                    decisions[mode] = {"index": index, "action_sha256": identities[index],
                                       "predicted_packet": moment + 1, "physical_step": (moment + 1) * 5,
                                       "predicted_costs": costs[index].tolist(),
                                       "success_at_predicted_packet": success,
                                       "first_success_step": item["first_success_step"],
                                       "final_success": item["final_success"]}
                selected = physical["records"][physical["selected_index"]]
                rows.append({"task": task, "data_row": physical["data_row"], "pool_size": len(costs),
                             "goal_embedding_sha256": goal_fingerprint, "decisions": decisions,
                             "original_selected": {"index": selected["index"],
                                                   "action_sha256": selected["action_sha256"],
                                                   "first_success_step": selected["first_success_step"],
                                                   "final_success": selected["final_success"]}})
    finally:
        env.close()
    report = {"scope": "post-hoc four exposed tasks, full captured candidate pool; no online stopping or gate claim",
              "rows": rows, "sources": {str(p): file_hash(p) for p in sources},
              "dataset_path": str(DATA), "dataset_hash_scope": "goal state values are read from local HDF5; full HDF5 not rehashed"}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUTPUT), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
