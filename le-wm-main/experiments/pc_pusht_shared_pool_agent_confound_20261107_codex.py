"""Post-hoc matched-object analysis of pusher-position sensitivity."""
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE
from experiments.pc_pusht_shared_pool_real_endpoint_20261107_codex import OUT as SOURCE
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT

OUT = ROOT / "results/pc_pusht_shared_pool_agent_confound_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_SHARED_POOL_AGENT_CONFOUND_PROTOCOL_20261107_CODEX.md"


def matched_pairs(states, goal, distance=5., angle=0.1, agent_gap=20.):
    if states.ndim != 2 or states.shape[1] != 7 or goal.shape[0] < 5:
        raise ValueError("invalid state contract")
    i, j = np.triu_indices(len(states), 1)
    obj = np.linalg.norm(states[i, 2:4] - states[j, 2:4], axis=1)
    da = states[i, 4] - states[j, 4]
    wrapped = np.abs(np.arctan2(np.sin(da), np.cos(da)))
    agent = np.linalg.norm(states[:, :2] - goal[None, :2], axis=1)
    keep = (obj <= distance) & (wrapped <= angle) & (np.abs(agent[i] - agent[j]) >= agent_gap)
    i, j = i[keep], j[keep]
    closer = np.where(agent[i] < agent[j], i, j)
    farther = np.where(agent[i] < agent[j], j, i)
    return closer, farther, agent


def pair_result(cost, closer, farther):
    diff = cost[closer] - cost[farther]
    return {"closer_lower": int((diff < 0).sum()), "farther_lower": int((diff > 0).sum()),
            "ties": int((diff == 0).sum())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows, source_hashes = [], {}
    with h5py.File(DATA, "r") as data:
        for task in range(4):
            stem = SOURCE.parent / f"{SOURCE.name}_task{task}"
            source = json.loads(stem.with_suffix(".json").read_text())
            if file_hash(stem.with_suffix(".npz")) != source["arrays_sha256"]:
                raise ValueError("endpoint arrays changed")
            path = BASE / f"physical_{task}.json"
            if file_hash(path) != source["physical_sha256"]:
                raise ValueError("physical file changed")
            physical = json.loads(path.read_text())
            lookup = {item["action_sha256"]: item for item in physical["records"]}
            selected = [lookup[digest] for digest in source["hashes"]]
            states = np.asarray([item["packet_states"][-1] for item in selected], dtype=np.float64)
            goal = np.asarray(data["state"][physical["data_row"] + 25], dtype=np.float64)
            closer, farther, agent = matched_pairs(states, goal)
            final = np.asarray([item["final_success"] for item in selected], bool)
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
                if not np.array_equal(final, arrays["final"]):
                    raise ValueError("success labels changed")
                scores = {name: pair_result(arrays[name], closer, farther)
                          for name in ("actual_cost", "one_frame", "three_frame")}
            success = np.flatnonzero(final)
            rows.append({"task": task, "pair_count": int(len(closer)),
                         "success_disagreement_pairs": int((final[closer] != final[farther]).sum()),
                         "final_success_count": int(len(success)),
                         "success_pusher_distances": agent[success].tolist(),
                         "success_pair_counts": [int(((closer == s) | (farther == s)).sum()) for s in success],
                         "scores": scores})
            source_hashes[str(stem.with_suffix(".json"))] = file_hash(stem.with_suffix(".json"))
            print(json.dumps(rows[-1]), flush=True)
    payload = {"protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
               "sources_sha256": source_hashes, "rows": rows}
    with OUT.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
