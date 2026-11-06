"""Run seeded trace with a first-observation-only data/render comparison."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.wrapper.default import AddPixelsWrapper

from experiments.pc_eval_seeded_trace_20261105_codex import main as run_trace
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pixels_wrapper(env):
    current = env
    while True:
        if isinstance(current, AddPixelsWrapper):
            return current
        if not hasattr(current, "env"):
            raise ValueError("AddPixelsWrapper missing from actual environment")
        current = current.env


def replace_initial(world, mode):
    if mode not in ("dataset", "render"):
        raise ValueError("unknown observation mode")
    pixels = world.infos["pixels"]
    goals = world.infos["goal"]
    if pixels.ndim != 5 or pixels.shape[0] != world.num_envs or pixels.shape[1] != 1:
        raise ValueError("expected single-frame image history")
    old_goal = goals.copy()
    records = []
    for i, env in enumerate(world.envs.envs):
        actor = env.unwrapped
        before = snapshot(actor)
        rendered, _ = pixels_wrapper(env)._get_pixels()
        frame = np.asarray(rendered["pixels"])
        after = snapshot(actor)
        if frame.shape != pixels[i, 0].shape or frame.dtype != pixels.dtype:
            raise ValueError("rendered pixel layout or dtype differs")
        if not np.array_equal(before, after):
            raise ValueError("render changed physical snapshot")
        previous = pixels[i, 0].copy()
        if mode == "render":
            pixels[i, 0] = frame
        records.append({"index": i, "data_sha256": digest(previous),
                        "render_sha256": digest(frame), "selected_sha256": digest(pixels[i, 0]),
                        "goal_sha256": digest(goals[i]), "pixel_max_abs": int(
                            np.abs(previous.astype(np.int16) - frame.astype(np.int16)).max()),
                        "physical_snapshot": before.tolist(), "render_snapshot_unchanged": True})
    if not np.array_equal(old_goal, world.infos["goal"]):
        raise ValueError("goal changed")
    return records


def main():
    mode = os.environ["PC_INITIAL_OBSERVATION_MODE"]
    audit_path = Path(os.environ["PC_INITIAL_OBSERVATION_AUDIT"])
    if audit_path.exists():
        raise FileExistsError(audit_path)
    get_actions = swm.World._get_actions

    def patched(world):
        if not getattr(world, "_pc_initial_observation_audited", False):
            records = replace_initial(world, mode)
            with audit_path.open("x") as stream:
                json.dump({"mode": mode, "records": records}, stream, indent=2, allow_nan=False)
                stream.write("\n")
            world._pc_initial_observation_audited = True
        return get_actions(world)

    swm.World._get_actions = patched
    print(f"Controlled first-observation mode={mode}; goal and physical state unchanged.", flush=True)
    run_trace()


if __name__ == "__main__":
    main()
