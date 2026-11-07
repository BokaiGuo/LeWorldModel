"""Independently reconstruct root-zero decisions and replay the selected chains."""
import json
from pathlib import Path

import h5py
import numpy as np
import gymnasium as gym
import stable_worldmodel  # noqa: F401

from experiments import pc_tworoom_vector_retention_posthoc_20261107_codex as exp
from experiments.pc_input_permissions import file_hash

OUTPUT = Path(str(exp.OUTPUT).removesuffix(".json") + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(exp.OUTPUT.read_text())
    for key, path in (("pool_sha256", exp.source.POOL), ("plan_sha256", exp.source.PLAN),
                      ("physical_sha256", exp.source.PHYSICAL),
                      ("protocol_sha256", exp.PROTOCOL), ("runner_sha256", exp.__file__)):
        if result[key] != file_hash(path):
            raise ValueError(f"artifact drift: {key}")
    prior = json.loads(exp.source.PHYSICAL.read_text())
    row = result["rows"][0]
    if row["root"] != 0:
        raise ValueError("unexpected root")
    with np.load(exp.source.POOL, allow_pickle=False) as pool:
        a = pool["r0_a_cost"]
        b = pool["r0_b_cost"]
        centered = a - a.mean(axis=1, keepdims=True)
        first = min(range(8), key=lambda i: (min(a[i]), i))
        distance = [sum(float(x-y)**2 for x, y in zip(centered[i], centered[first]))
                    for i in range(8)]
        second = min((i for i in range(8) if i != first), key=lambda i: (-distance[i], i))
        if row["methods"]["vector"]["retained"] != [first, second]:
            raise ValueError("independent diversity calculation differs")
        for method in exp.METHODS:
            retained = row["methods"][method]["retained"]
            choices = [(float(a.flat[i]), 0, i) for i in range(64)]
            choices.extend((float(b.flat[i]), 1, i) for i in range(64) if i // 8 in retained)
            _, bank, index = min(choices)
            record = row["methods"][method]
            if (("a" if bank == 0 else "b"), index) != (record["bank"], record["index"]):
                raise ValueError("independent selection mismatch")
        ids = sorted({v["index"] for v in row["methods"].values() if v["bank"] == "b"})
        chains = pool["r0_b_actions"].reshape(64, 25, 2)[ids].copy()
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    checked = []
    try:
        with h5py.File(exp.source.source.block.prior.common.DATA, "r") as data:
            for index, chain in zip(ids, chains):
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row["row"]])
                actor._set_goal_state(data["proprio"][row["row"] + 25])
                trace = []
                for action in chain:
                    _, _, term, trunc, info = actor.step(action.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(term), "truncated": bool(trunc)})
                if trace != prior["rows"][0]["outcomes"][index]["trace"]:
                    raise ValueError(f"physical replay mismatch index={index}")
                checked.append({"index": index, "ever": any(x["terminated"] for x in trace)})
    finally:
        env.close()
    with OUTPUT.open("x") as f:
        json.dump({"result_sha256": file_hash(exp.OUTPUT), "root": 0, "replayed": checked,
                   "scope": "same simulator replay, not independent evidence"}, f, indent=2)
    print(json.dumps(checked))


if __name__ == "__main__":
    main()
