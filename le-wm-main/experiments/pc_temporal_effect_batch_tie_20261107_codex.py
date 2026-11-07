"""Post-hoc same-prefix replay of an audited batch-sensitive selection."""
import json

import h5py
import numpy as np

from experiments import pc_temporal_effect_retention_20261107_codex as src
from experiments.pc_input_permissions import file_hash

AUDIT = src.ROOT / "results/pc_temporal_effect_integrity_audit_20261107_codex.json"
PROTOCOL = src.ROOT.parent / "docs/PC_TEMPORAL_EFFECT_BATCH_TIE_PROTOCOL_20261107_CODEX.md"
OUTPUT = src.ROOT / "results/pc_temporal_effect_batch_tie_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit, plan = json.loads(AUDIT.read_text()), json.loads(src.PLAN.read_text())
    if audit["plan_sha256"] != file_hash(src.PLAN) or plan["sha256"]["pool"] != file_hash(src.POOL):
        raise ValueError("sealed artifact drift")
    if plan["sha256"]["old_pool"] != file_hash(src.old.POOL):
        raise ValueError("candidate pool drift")
    record = next(x for x in audit["rows"] if x["root"] == 46)
    comparison = record["methods"]["gated"]
    if (comparison["sealed"] != ["a", 56] or comparison["recomputed"] != ["b", 56]
            or not comparison["choice_changed"]):
        raise ValueError("audit discrepancy changed")
    task = next(x for x in plan["rows"] if x["root"] == 46)
    with np.load(src.old.POOL, allow_pickle=False) as pool:
        actions = {name: pool[f"r46_{name}"].reshape(64, 25, 2)[56].copy() for name in ("a", "b")}
    if not np.array_equal(actions["a"][:10], actions["b"][:10]):
        raise ValueError("not the same prefix")
    if src.digest(actions["a"]) != task["decisions"]["gated"]["action_sha256"]:
        raise ValueError("sealed action drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        with h5py.File(src.old.base.source.block.prior.common.DATA, "r") as data:
            outcomes = {}
            for name, action in actions.items():
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task["row"]])
                actor._set_goal_state(data["proprio"][task["row"] + 25])
                trace = []
                for item in action:
                    _, _, terminated, truncated, info = actor.step(item.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(terminated), "truncated": bool(truncated)})
                outcomes[name] = {"action_sha256": src.digest(action),
                                  "ever": any(x["terminated"] for x in trace),
                                  "final": trace[-1]["terminated"], "trace": trace}
    finally:
        env.close()
    previous = json.loads(src.PHYS.read_text())
    if previous["plan_sha256"] != file_hash(src.PLAN):
        raise ValueError("physical source drift")
    previous_a = next(x for x in previous["rows"] if x["root"] == 46)["outcomes"]["gated"]
    if previous_a["trace"] != outcomes["a"]["trace"]:
        raise ValueError("same-action replay differs from previous")
    if outcomes["a"]["trace"][:10] != outcomes["b"]["trace"][:10]:
        raise ValueError("same prefix produced different physical trajectories")
    with OUTPUT.open("x") as f:
        json.dump({"root": 46, "outcomes": outcomes, "audit_sha256": file_hash(AUDIT),
                   "plan_sha256": file_hash(src.PLAN), "previous_physical_sha256": file_hash(src.PHYS),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "scope": "post-hoc exposed single-root physical replay"}, f, indent=2, allow_nan=False)
    print(json.dumps({k: {"ever": v["ever"], "final": v["final"],
                          "final_distance": v["trace"][-1]["distance"]} for k, v in outcomes.items()}, indent=2))


if __name__ == "__main__":
    main()
