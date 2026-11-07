"""Physical replay of action identities sealed by the saved-prefix audit."""
import json

import h5py
import numpy as np

from experiments import pc_temporal_effect_retention_20261107_codex as src
from experiments import pc_causal_prefix_saved_audit_20261107_codex as audit
from experiments.pc_input_permissions import file_hash

OUTPUT = src.ROOT / "results/pc_causal_prefix_saved_replay_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(audit.OUTPUT.read_text())
    plan = json.loads(src.PLAN.read_text())
    if data["plan_sha256"] != file_hash(src.PLAN) or data["runner_sha256"] != file_hash(audit.__file__):
        raise ValueError("audit drift")
    if plan["sha256"]["old_pool"] != file_hash(src.old.POOL):
        raise ValueError("candidate pool drift")
    root_audit = next(r for r in data["rows"] if r["root"] == 46)
    task = next(r for r in plan["rows"] if r["root"] == 46)
    if any(r["root"] != 46 and any(v["choice_changed"] for v in r["methods"].values()) for r in data["rows"]):
        raise ValueError("unexpected changed root")
    with np.load(src.old.POOL, allow_pickle=False) as pool:
        candidate_actions = np.concatenate((pool["r46_a"].reshape(64, 25, 2),
                                            pool["r46_b"].reshape(64, 25, 2)))
    targets = {m: root_audit["methods"][m]["corrected_sha256"] for m in src.METHODS}
    matches = {}
    for action in candidate_actions:
        h = src.digest(action)
        if h in targets.values():
            matches[h] = action.copy()
    if set(matches) != set(targets.values()):
        raise ValueError("corrected action not found in frozen candidates")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        with h5py.File(src.old.base.source.block.prior.common.DATA, "r") as dataset:
            outcomes = {}
            for h, action in matches.items():
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(dataset["proprio"][task["row"]])
                actor._set_goal_state(dataset["proprio"][task["row"] + 25])
                trace = []
                for packet in action:
                    _, _, terminated, truncated, info = actor.step(packet.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(terminated), "truncated": bool(truncated)})
                outcomes[h] = {"ever": any(x["terminated"] for x in trace),
                               "final": trace[-1]["terminated"], "trace": trace}
    finally:
        env.close()
    previous = json.loads(src.PHYS.read_text())
    if previous["plan_sha256"] != file_hash(src.PLAN):
        raise ValueError("physical source drift")
    rows = {r["root"]: r for r in previous["rows"]}
    summary = {}
    for method in src.METHODS:
        items = [outcomes[targets[method]] if root == 46 else rows[root]["outcomes"][method]
                 for root in range(20, 50)]
        summary[method] = {"ever": sum(x["ever"] for x in items),
                           "final": sum(x["final"] for x in items),
                           "mean_final_distance": float(np.mean([x["trace"][-1]["distance"] for x in items]))}
    with OUTPUT.open("x") as f:
        json.dump({"summary": summary, "root46": {m: {"action_sha256": targets[m],
                  "outcome": outcomes[targets[m]]} for m in src.METHODS},
                   "audit_sha256": file_hash(audit.OUTPUT), "original_physical_sha256": file_hash(src.PHYS),
                   "runner_sha256": file_hash(__file__),
                   "scope": "post-hoc exposed corrected-selection physical replay"}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
