"""Controlled localization of the dataset/reset initial-frame discrepancy."""
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from experiments.pc_prefix_novel_actual_encoding_20261105_codex import native_world_pixels
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    out = root / "pc_initial_image_attribution_dev_v2_20261105_codex.json"
    if out.exists():
        raise FileExistsError(out)
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        row = int(panel_rows(data)[0])
        state = np.asarray(data["state"][row], dtype=np.float64)
        goal = np.asarray(data["state"][row + 25], dtype=np.float64)
        original = np.asarray(data["pixels"][row])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            env.reset(seed=771001)
            actor = env.unwrapped
            actor._set_state(state.copy())
            actor._set_goal_state(goal.copy())
            observed_state = actor._get_obs()
            render = native_world_pixels(actor.render())
            changed = np.any(original != render, axis=-1)
            coordinates = np.argwhere(changed)
            pairs = Counter(zip(map(tuple, original[changed]), map(tuple, render[changed])))
            # Frames from adjacent positions are exploratory and not used to select methods.
            nearby = []
            episode = data["episode_idx"] if "episode_idx" in data else data["ep_idx"]
            for offset in range(-5, 6):
                if row + offset < 0 or row + offset >= len(episode) or episode[row + offset] != episode[row]:
                    continue
                frame = np.asarray(data["pixels"][row + offset], dtype=np.int16)
                nearby.append({"offset": offset, "mean_absolute_pixel_error": float(
                    np.abs(frame - render.astype(np.int16)).mean())})
            result = {
                "dataset_row": row, "state_input": state.tolist(), "state_after_set": observed_state.tolist(),
                "state_max_abs_change": float(np.max(np.abs(state - observed_state))),
                "changed_pixels": int(changed.sum()), "total_pixels": int(changed.size),
                "difference_bbox_yx": [coordinates.min(0).tolist(), coordinates.max(0).tolist()]
                    if len(coordinates) else None,
                "most_common_rgb_transitions": [{"dataset": list(map(int, a)), "render": list(map(int, b)), "count": n}
                                                for (a, b), n in pairs.most_common(12)],
                "neighbor_frame_errors": nearby,
                "render_action": bool(actor.render_action), "with_target": bool(actor.with_target),
                "render_goal": bool(actor.variation_space["rendering"]["render_goal"].value),
                "goal_pose": np.asarray(actor.goal_pose).tolist(),
                "goal_state": np.asarray(actor.goal_state).tolist(),
            }
        finally:
            env.close()
    encoded = json.dumps(result, indent=2, allow_nan=False)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
