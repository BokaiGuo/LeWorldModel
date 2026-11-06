"""Evaluation-only ablation of PushT _set_state's implicit physics tick."""
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from experiments.pc_prefix_novel_actual_encoding_20261105_codex import native_world_pixels
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    out = Path(__file__).resolve().parents[1] / "results/pc_initial_image_state_step_ablation_v3_20261105_codex.json"
    if out.exists():
        raise FileExistsError(out)
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        row = int(panel_rows(data)[0])
        state, goal, pixels = (np.asarray(data["state"][row], dtype=np.float64),
                               np.asarray(data["state"][row + 25], dtype=np.float64),
                               np.asarray(data["pixels"][row]))
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        result = {}
        try:
            for method in ("direct_without_step", "direct_reindex_without_step", "native_set_state"):
                env.reset(seed=771001)
                actor = env.unwrapped
                if method != "native_set_state":
                    actor.agent.velocity = tuple(state[-2:])
                    actor.agent.position = state[:2].tolist()
                    actor.block.angle = float(state[4])
                    actor.block.position = state[2:4].tolist()
                    if method == "direct_reindex_without_step":
                        actor.space.reindex_shapes_for_body(actor.agent)
                        actor.space.reindex_shapes_for_body(actor.block)
                else:
                    actor._set_state(state.copy())
                actor._set_goal_state(goal.copy())
                frame = native_world_pixels(actor.render())
                err = np.abs(frame.astype(np.int16) - pixels.astype(np.int16))
                result[method] = {
                    "state_max_abs_change": float(np.max(np.abs(actor._get_obs() - state))),
                    "image_mean_absolute_error": float(err.mean()),
                    "changed_pixels": int(np.any(err != 0, axis=-1).sum()),
                    "maximum_pixel_error": int(err.max()),
                }
                jpeg = []
                for quality in (75, 85, 90, 95, 100):
                    buffer = io.BytesIO()
                    Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
                    decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
                    jpeg.append({"quality": quality, "mean_absolute_error": float(
                        np.abs(decoded.astype(np.int16) - pixels.astype(np.int16)).mean())})
                result[method]["exploratory_jpeg_roundtrip"] = jpeg
        finally:
            env.close()
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
