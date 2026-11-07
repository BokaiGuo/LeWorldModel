"""Qualify the fixed zero-tail reference at previously evaluated prefix states."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as source
from experiments.pc_input_permissions import file_hash

PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_PREFIX_HOLD_PHYSICAL_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.ROOT / "results/pc_tworoom_prefix_hold_physical_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(source.PHYSICAL.read_text())
    if prior["prediction_sha256"] != file_hash(source.PRED) or prior["runner_sha256"] != file_hash(source.__file__):
        raise ValueError("sealed source mismatch")
    previous = {(x["seed"], x["prefix"], x["tail"]): x["trace"] for x in prior["rows"]}
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    results = []
    print("Replay physical prefixes and measure subsequent zero-action drift.", flush=True)
    try:
        with np.load(source.INPUT, allow_pickle=False) as images:
            for seed in source.SEEDS:
                for prefix_index, action in enumerate(source.ACTIONS):
                    env.reset(seed=seed)
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), images[f"s{seed}_current"])
                    np.testing.assert_array_equal(source.image_goal(actor), images[f"s{seed}_goal"])
                    distances, flags = [], []
                    for _ in range(5):
                        _, _, terminated, truncated, info = actor.step(action.copy())
                        distances.append(float(info["distance_to_target"]))
                        flags.append((bool(terminated), bool(truncated)))
                    start = actor.agent_position.detach().cpu().numpy().copy()
                    positions = []
                    for _ in range(5):
                        _, _, terminated, truncated, info = actor.step(np.zeros(2, dtype=np.float32))
                        positions.append(actor.agent_position.detach().cpu().numpy().copy())
                        distances.append(float(info["distance_to_target"]))
                        flags.append((bool(terminated), bool(truncated)))
                    old = previous[(seed, prefix_index, 0)]
                    delta = max(abs(distance - item["distance"]) for distance, item in zip(distances, old, strict=True))
                    if delta > 1e-6 or flags != [(x["terminated"], x["truncated"]) for x in old]:
                        raise ValueError("physical trace drift")
                    drift = np.linalg.norm(np.asarray(positions) - start, axis=1)
                    results.append({"seed": seed, "prefix": prefix_index,
                                    "prefix_action": action.tolist(), "position_after_prefix": start.tolist(),
                                    "positions_after_zero": np.asarray(positions).tolist(),
                                    "max_hold_drift": float(drift.max()), "stationary_at_1e-6": bool(drift.max() <= 1e-6),
                                    "distance_trace": distances, "trace_max_error": delta})
    finally:
        env.close()
    output = {"scope": "post-prefix physical evaluation only; exposed seeds; same backend",
              "rows": results, "source_physical_sha256": file_hash(source.PHYSICAL),
              "input_sha256": file_hash(source.INPUT), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(Path(__file__))}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({"stationary": sum(x["stationary_at_1e-6"] for x in results),
                      "total": len(results), "max_drift": max(x["max_hold_drift"] for x in results)}, indent=2))


if __name__ == "__main__":
    main()
