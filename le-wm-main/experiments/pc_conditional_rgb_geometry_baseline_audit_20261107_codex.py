"""Independently reconstruct and replay the sealed RGB geometry baseline."""
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401

from experiments import pc_conditional_rgb_geometry_baseline_20261107_codex as run
from experiments import pc_conditional_selected_alignment_20261107_codex as source
from experiments.pc_input_permissions import file_hash

OUTPUT = Path(str(run.BASE) + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, _, inputs_path = source.source_records()
    sealed, result = json.loads(run.PLAN.read_text()), json.loads(run.RESULT.read_text())
    if result["plan_sha256"] != file_hash(run.PLAN) or sealed["actions_sha256"] != file_hash(run.ACTIONS):
        raise ValueError("artifact drift")
    common = source.src.old.base.source.block.prior.common
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    replay_steps, max_pixel_error = 0, 0.
    try:
        with (np.load(run.ACTIONS, allow_pickle=False) as saved,
              np.load(inputs_path, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as data):
            for task, sealed_row, recorded in zip(plan["rows"], sealed["rows"], result["rows"], strict=True):
                root = task["root"]
                if (root != sealed_row["root"] or root != recorded["root"]
                        or sealed_row["abstained"] != recorded["abstained"]):
                    raise ValueError("root mismatch")
                chain, reason = run.generate(inputs["current"][root], inputs["goal"][root])
                if chain is None:
                    if not sealed_row["abstained"] or reason != sealed_row["reason"] or recorded["trace"]:
                        raise ValueError("abstention mismatch")
                    continue
                np.testing.assert_array_equal(chain, saved[f"r{root}"])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task["row"]])
                actor._set_goal_state(data["proprio"][task["row"] + 25])
                error = float(np.abs(actor.render().astype(float) - inputs["current"][root].astype(float)).mean())
                if abs(error - recorded["initial_pixel_mean_absolute_error"]) > 1e-12:
                    raise ValueError("initial image mismatch")
                max_pixel_error = max(max_pixel_error, error)
                if len(recorded["trace"]) != 25:
                    raise ValueError("incomplete trace")
                for action, expected in zip(chain, recorded["trace"], strict=True):
                    _, _, terminated, truncated, info = actor.step(action.copy())
                    if (abs(float(info["distance_to_target"]) - expected["distance"]) > 1e-5
                            or bool(terminated) != expected["terminated"]
                            or bool(truncated) != expected["truncated"]):
                        raise ValueError(f"physical replay mismatch: {root}")
                    replay_steps += 1
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"tasks": len(result["rows"]), "replayed_steps": replay_steps,
                   "max_initial_pixel_mean_absolute_error": max_pixel_error,
                   "result_sha256": file_hash(run.RESULT), "actions_sha256": file_hash(run.ACTIONS),
                   "scope": "same generator and simulator; independently replayed, not independent data"},
                  stream, indent=2, allow_nan=False)
    print(f"Reconstructed {len(result['rows'])} actions; replayed {replay_steps} steps.")


if __name__ == "__main__":
    main()
