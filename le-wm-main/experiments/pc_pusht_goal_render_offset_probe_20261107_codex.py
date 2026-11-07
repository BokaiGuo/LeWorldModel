"""Evaluator-only, fixed local state-time and goal-overlay image diagnostic."""
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401
import stable_worldmodel.envs.pusht.env as pusht

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import native_world_pixels
from experiments.pc_pusht_goal_render_alignment_20261107_codex import set_render_pose
from experiments import pc_pusht_first_plan_alignment_20261107_codex as source

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_GOAL_RENDER_OFFSET_PROBE_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.RESULTS / "pc_pusht_goal_render_offset_probe_20261107_codex.json"
OFFSETS = (-2, -1, 0, 1, 2)


def image_error(left, right):
    if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("incompatible uint8 RGB images")
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {"mean_abs": float(delta.mean()), "max_abs": int(delta.max()),
            "exact_equal": bool(np.array_equal(left, right))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    record = json.loads(source.OUT.with_suffix(".json").read_text())
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
    rows = []
    try:
        with h5py.File(source.DATA, "r") as data:
            for row in record["rows"]:
                idx, target_row = row["index"], row["data_row"] + 25
                env.reset(seed=871001 + idx)
                actor = env.unwrapped
                reference = np.asarray(data["pixels"][target_row])
                episode = data["episode_idx"][target_row]
                trials = []
                for offset in OFFSETS:
                    j = target_row + offset
                    if j < 0 or j >= len(data["state"]) or data["episode_idx"][j] != episode:
                        trials.append({"offset": offset, "excluded": "outside_episode"})
                        continue
                    state = np.asarray(data["state"][j], dtype=np.float64)
                    set_render_pose(actor, state)
                    readback = actor._get_obs()
                    error = float(np.max(np.abs(state[:5] - readback[:5])))
                    if error > 1e-4:
                        raise ValueError(f"pose mismatch task={idx}, offset={offset}: {error}")
                    item = {"offset": offset, "row": j, "pose_error": error,
                            "state_sha256": hashlib.sha256(state.tobytes()).hexdigest()}
                    for enabled in (True, False):
                        actor.with_target = enabled
                        rendered = native_world_pixels(actor.render())
                        item["overlay_on" if enabled else "overlay_off"] = image_error(reference, rendered)
                    trials.append(item)
                rows.append({"index": idx, "target_row": target_row,
                             "episode_idx": int(episode), "trials": trials})
    finally:
        env.close()
    paths = (PROTOCOL, Path(__file__), source.OUT.with_suffix(".json"),
             ROOT / "experiments/pc_pusht_goal_render_alignment_20261107_codex.py", Path(pusht.__file__))
    report = {"scope": "exploratory evaluator-only; no gate promotion",
              "data": str(source.DATA), "data_size": source.DATA.stat().st_size,
              "data_mtime_ns": source.DATA.stat().st_mtime_ns,
              "source_sha256": {str(path): file_hash(path) for path in paths}, "rows": rows}
    with OUTPUT.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"completed {len(rows)} paired goal-image trials")


if __name__ == "__main__":
    main()
