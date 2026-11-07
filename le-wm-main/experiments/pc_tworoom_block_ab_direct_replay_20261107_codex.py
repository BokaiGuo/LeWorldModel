"""Post-hoc A-versus-B direct model-score selection with equal scored counts."""
import json
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as source
from experiments.pc_input_permissions import file_hash

OUTPUT = Path(str(source.PREFIX) + "_ab_direct_physical.json")


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    previous = json.loads(source.PHYSICAL.read_text())
    if (plan["pool_sha256"] != file_hash(source.POOL)
            or previous["plan_sha256"] != file_hash(source.PLAN)
            or previous["pool_sha256"] != file_hash(source.POOL)):
        raise ValueError("sealed source mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (h5py.File(source.source.block.prior.common.DATA, "r") as data,
              np.load(source.POOL, allow_pickle=False) as pool):
            for root, (planned, old) in enumerate(zip(plan["rows"], previous["rows"], strict=True)):
                if planned["root"] != root or old["root"] != root or planned["row"] != old["row"]:
                    raise ValueError("root mismatch")
                choices = {bank: int(np.argmin(pool[f"r{root}_{bank}_cost"])) for bank in ("a", "b")}
                action = pool[f"r{root}_a_actions"].reshape(64, 25, 2)[choices["a"]]
                if not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
                    raise ValueError("illegal physical action")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][planned["row"]])
                actor._set_goal_state(data["proprio"][planned["row"] + 25])
                trace = []
                for act in action:
                    _, _, term, trunc, info = actor.step(act.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(term), "truncated": bool(trunc)})
                a = {"ever": any(x["terminated"] for x in trace),
                     "prefix_success": any(x["terminated"] for x in trace[:10]),
                     "final_distance": trace[-1]["distance"], "trace": trace}
                b = old["outcomes"][choices["b"]]
                rows.append({"root": root, "row": planned["row"], "choices": choices, "a": a, "b": b})
                print(f"root={root} direct A={a['ever']} B={b['ever']}", flush=True)
    finally:
        env.close()
    summary = {
        "a_ever": sum(r["a"]["ever"] for r in rows),
        "b_ever": sum(r["b"]["ever"] for r in rows),
        "a_only": sum(r["a"]["ever"] and not r["b"]["ever"] for r in rows),
        "b_only": sum(r["b"]["ever"] and not r["a"]["ever"] for r in rows),
        "a_mean_distance": float(np.mean([r["a"]["final_distance"] for r in rows])),
        "b_mean_distance": float(np.mean([r["b"]["final_distance"] for r in rows])),
    }
    with OUTPUT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "plan_sha256": file_hash(source.PLAN),
                   "pool_sha256": file_hash(source.POOL), "b_physical_sha256": file_hash(source.PHYSICAL),
                   "scope": "post-hoc exposed tasks; 64 scores per arm plus shared prefix generation"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
