"""Compare model's captured initial input against evaluation-side reset rendering."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from eval import img_transform
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import native_world_pixels
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    root = Path(__file__).resolve().parents[1] / "results"
    output = root / "pc_branch_full_pool_initial_observation_audit_20261105_codex.json"
    if output.exists():
        raise FileExistsError(output)
    capture = root / "pc_branch_full_pool_lookahead_seed42_20261105_codex.capture/call_00.pt"
    inputs = torch.load(capture, map_location="cpu", weights_only=False)
    actual_model_input = inputs["context"]["pixels"][0, 0, -1]
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        row = panel_rows(data)[0]
        dataset_input = transform(np.asarray(data["pixels"][row]))
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            env.reset(seed=771001)
            actor = env.unwrapped
            actor._set_state(np.asarray(data["state"][row], dtype=np.float64))
            actor._set_goal_state(np.asarray(data["state"][row + 25], dtype=np.float64))
            reset_input = transform(native_world_pixels(actor.render()))
        finally:
            env.close()
    def diff(a, b):
        delta = (a - b).abs()
        return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                "different_elements": int((delta > 1e-6).sum()), "elements": delta.numel()}
    result = {"model_vs_dataset": diff(actual_model_input, dataset_input),
              "model_vs_reset_render": diff(actual_model_input, reset_input),
              "dataset_vs_reset_render": diff(dataset_input, reset_input)}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
