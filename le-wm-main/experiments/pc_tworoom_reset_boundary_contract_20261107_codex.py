"""Fail-closed TwoRoom initial border check and no-op replay diagnostic."""
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as source

OUTPUT = source.ROOT / "results/pc_tworoom_reset_boundary_contract_20261107_codex.json"


def border_contract(position, border, radius, image_size):
    position = np.asarray(position, dtype=float)
    values = np.r_[position, border, radius, image_size]
    if position.shape != (2,) or not np.isfinite(values).all() or border < 0 or radius < 0:
        raise ValueError("invalid boundary inputs")
    low, high = float(border + radius), float(image_size - border - radius)
    if low > high:
        raise ValueError("agent cannot fit in image")
    return {"valid": bool(np.all(position >= low) and np.all(position <= high)),
            "lower": low, "upper": high, "position": position.tolist()}


def require_legal_border(position, border, radius, image_size):
    report = border_contract(position, border, radius, image_size)
    if not report["valid"]:
        raise ValueError("reset position lies outside the step collision border")
    return report


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = json.loads(source.PHYSICAL.read_text())
    if original["runner_sha256"] != file_hash(source.__file__):
        raise ValueError("experiment source changed")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        actor = env.unwrapped
        env_path = Path(inspect.getfile(type(actor)))
        for seed in source.SEEDS:
            _, info = env.reset(seed=seed)
            position = actor.agent_position.numpy().copy()
            radius = float(actor.variation_space["agent"]["radius"].value.item())
            contract = border_contract(position, actor.BORDER_SIZE, radius, actor.IMG_SIZE)
            _, _, terminated, truncated, step_info = actor.step(np.zeros(2, dtype=np.float32))
            after = actor.agent_position.numpy().copy()
            saved = next(x for x in original["rows"] if x["seed"] == seed and x["prefix"] == 0 and x["tail"] == 0)
            if (abs(float(step_info["distance_to_target"]) - saved["trace"][0]["distance"]) > 1e-6
                    or (bool(terminated), bool(truncated)) !=
                    (saved["trace"][0]["terminated"], saved["trace"][0]["truncated"])):
                raise ValueError("no-op differs from sealed candidate")
            rows.append({"seed": seed, "initial": contract, "after_zero": after.tolist(),
                         "zero_displacement": float(np.linalg.norm(after-position)),
                         "initial_goal_distance": float(info["distance_to_target"]),
                         "after_goal_distance": float(step_info["distance_to_target"])})
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "physical_sha256": file_hash(source.PHYSICAL), "env_sha256": file_hash(env_path),
                   "env_path": str(env_path), "runner_sha256": file_hash(__file__),
                   "scope": "existing development roots; border check is not full reset validity"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    audit()
