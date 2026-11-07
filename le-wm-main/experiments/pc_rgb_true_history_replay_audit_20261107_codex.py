"""Independently replay every saved physical step and sampled RGB boundary."""
import json

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments import pc_rgb_true_history_alignment_20261107_codex as experiment

OUT = experiment.ROOT / "results/pc_rgb_true_history_replay_audit_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    trace = json.loads(experiment.TRACE.read_text())
    if (trace["input_sha256"] != file_hash(experiment.INPUT)
            or trace["truth_sha256"] != file_hash(experiment.TRUTH)
            or trace["source_physical_sha256"] != file_hash(experiment.source.PHYSICAL)
            or trace["protocol_sha256"] != file_hash(experiment.PROTOCOL)
            or trace["runner_sha256"] != file_hash(experiment.__file__)):
        raise ValueError("sealed trace dependencies changed")
    physical = json.loads(experiment.source.PHYSICAL.read_text())
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    rows = []
    try:
        with np.load(experiment.INPUT, allow_pickle=False) as inputs, np.load(experiment.TRUTH, allow_pickle=False) as truth:
            for item in trace["rows"]:
                root, side = item["root"], item["side"]
                key = f"r{root}_s{side}"
                source = physical["rows"][root]
                condition = next(c for c in source["conditions"] if c["appearance"] == "white" and c["side"] == side)
                env.reset(seed=881001 + root, options={"state": np.asarray(condition["requested_state"]),
                                                       "goal_state": np.asarray(source["goal"])})
                actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                np.testing.assert_array_equal(actor.render(), inputs[key+"_frames"][0])
                max_error = 0.
                for i, (action, saved) in enumerate(zip(inputs[key+"_actions"].reshape(15, 2), item["steps"], strict=True)):
                    _, _, terminated, truncated, _ = actor.step(action.copy())
                    max_error = max(max_error, float(np.max(np.abs(snapshot(actor) - np.asarray(saved["snapshot"])))))
                    if (bool(terminated), bool(truncated)) != (saved["terminated"], saved["truncated"]):
                        raise ValueError("terminal flags differ")
                    if i in (4, 9):
                        np.testing.assert_array_equal(actor.render(), inputs[key+"_frames"][(i+1)//5])
                    if i == 14:
                        np.testing.assert_array_equal(actor.render(), truth[key+"_future"])
                if max_error > 1e-9:
                    raise ValueError("physical state trajectory differs")
                rows.append({"root": root, "side": side, "max_state_error": max_error, "steps": 15})
    finally:
        env.close()
    if len(rows) != 8 or len({(r["root"], r["side"]) for r in rows}) != 8:
        raise ValueError("incomplete physical panel")
    output = {"rows": rows, "steps": sum(r["steps"] for r in rows),
              "max_state_error": max(r["max_state_error"] for r in rows),
              "trace_sha256": file_hash(experiment.TRACE), "runner_sha256": file_hash(__file__),
              "scope": "exposed RGB-only development physical replay, not independent tasks"}
    with OUT.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
