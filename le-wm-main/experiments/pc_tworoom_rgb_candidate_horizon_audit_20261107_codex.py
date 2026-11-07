"""Audit the physical reach and collision displacement of the sealed 10-step bank."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_tworoom_rgb_candidate_horizon_audit_20261107_codex.json"


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    recorded = json.loads(run.PHYSICAL.read_text())
    if (recorded["prediction_sha256"] != file_hash(run.PRED)
            or recorded["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("source mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(run.INPUT, allow_pickle=False) as images:
            for seed in run.SEEDS:
                env.reset(seed=seed)
                actor = env.unwrapped
                current, goal = actor.agent_position.numpy().copy(), actor.target_position.numpy().copy()
                speed = float(actor.variation_space["agent"]["speed"].value.item())
                if not np.isfinite(speed) or speed <= 0:
                    raise ValueError("invalid speed")
                np.testing.assert_array_equal(actor.render(), images[f"s{seed}_current"])
                np.testing.assert_array_equal(run.image_goal(actor), images[f"s{seed}_goal"])
                best_distance, longest_step, path_length, pushbacks, successes = np.inf, 0., 0., 0, 0
                for i, a in enumerate(run.ACTIONS):
                    for j, b in enumerate(run.ACTIONS):
                        env.reset(seed=seed)
                        prior = actor.agent_position.numpy().copy()
                        saved = next(r for r in recorded["rows"] if (r["seed"], r["prefix"], r["tail"]) == (seed, i, j))
                        total = 0.
                        for action, step in zip([a] * 5 + [b] * 5, saved["trace"], strict=True):
                            _, _, terminated, truncated, info = actor.step(action.copy())
                            position = actor.agent_position.numpy().copy()
                            displacement = float(np.linalg.norm(position - prior))
                            command = speed * float(np.linalg.norm(action))
                            longest_step = max(longest_step, displacement)
                            total += displacement
                            pushbacks += int(displacement > command + 1e-4)
                            if abs(float(info["distance_to_target"]) - step["distance"]) > 1e-6:
                                raise ValueError("trajectory mismatch")
                            if (bool(terminated), bool(truncated)) != (step["terminated"], step["truncated"]):
                                raise ValueError("termination mismatch")
                            prior = position
                        best_distance = min(best_distance, float(info["distance_to_target"]))
                        path_length = max(path_length, total)
                        successes += int(any(s["terminated"] for s in saved["trace"]))
                rows.append({"seed": seed, "initial_position": current.tolist(), "goal_position": goal.tolist(),
                             "initial_distance": float(np.linalg.norm(current-goal)), "speed": speed,
                             "best_bank_final_distance": best_distance, "max_observed_step": longest_step,
                             "max_observed_path_length": path_length, "steps_exceeding_command": pushbacks,
                             "successful_chains": successes})
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "physical_sha256": file_hash(run.PHYSICAL),
                   "source_sha256": file_hash(__file__),
                   "scope": "post-hoc exact replay of fixed bank; not a global reachability certificate"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    audit()
