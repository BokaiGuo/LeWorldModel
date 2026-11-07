"""Evaluator-side creation of a label-free planning input package."""
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments import pc_tworoom_precise_spectral_development_20261107_codex as source
from experiments.pc_input_permissions import file_hash

PUBLIC = source.ROOT / "results/pc_rgb_gradient_public_input_20261107_codex.npz"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PUBLIC.exists():
        raise FileExistsError(PUBLIC)
    plan = json.loads(source.PLAN.read_text())
    physical = json.loads(source.PHYSICAL.read_text())
    if plan["dependencies"] != source.dependencies() or physical["plan_sha256"] != file_hash(source.PLAN):
        raise ValueError("source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    current, goal, nominal, episodes = [], [], [], []
    try:
        with np.load(source.source.INPUT, allow_pickle=False) as inputs, h5py.File(source.source.common.DATA, "r") as db:
            actions = np.asarray(db["action"][:])
            actions = actions[~np.isnan(actions).any(axis=1)]
            scaler = StandardScaler().fit(actions)
            for row in plan["rows"]:
                index, = np.flatnonzero(inputs["episodes"] == row["episode"])
                if int(inputs["rows"][index]) != row["row"]:
                    raise ValueError("task identity mismatch")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(db["proprio"][row["row"]])
                actor._set_goal_state(db["proprio"][row["row"] + 25])
                image = actor.render().copy()
                if (source.img_hash(image) != row["current_rgb_sha256"]
                        or source.img_hash(inputs["goal"][index]) != row["goal_rgb_sha256"]):
                    raise ValueError("RGB identity mismatch")
                episodes.append(row["episode"])
                current.append(image)
                goal.append(inputs["goal"][index])
                nominal.append(row["arms"]["spectral"]["candidates"][0])
    finally:
        env.close()
    with PUBLIC.open("xb") as stream:
        np.savez_compressed(stream, episodes=np.asarray(episodes, np.int64),
                            current=np.asarray(current, np.uint8), goal=np.asarray(goal, np.uint8),
                            nominal=np.asarray(nominal, np.float32),
                            action_mean=scaler.mean_, action_scale=scaler.scale_)
    print(f"Sealed {len(episodes)} RGB-only planning inputs without physical labels.")


if __name__ == "__main__":
    main()
