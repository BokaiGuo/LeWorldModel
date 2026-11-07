"""Replay recorded feedback episodes and verify RGB scoring receipt clocks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash, tensor_receipt
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments import pc_rgb_true_history_alignment_20261107_codex as source

RECORD = source.ROOT / "results/pc_rgb_history_feedback_20261107_codex.json"
OUTPUT = source.ROOT / "results/pc_rgb_feedback_frame_clock_replay_20261107_codex.json"


def check_inputs(receipts, frames, goal, mode, transform):
    selected = frames[-1:] if mode == "single" else frames[-3:]
    pixels = torch.stack([transform(frame) for frame in selected])[None, None]
    target = transform(goal)[None, None, None]
    expected_pixels, expected_goal = tensor_receipt(pixels), tensor_receipt(target)
    for receipt in receipts:
        if receipt["track_id"] != "primary_rgb" or receipt["operation"] != "score":
            raise ValueError("unexpected track or model operation")
        if receipt["inputs"]["pixels"] != expected_pixels or receipt["inputs"]["goal"] != expected_goal:
            raise ValueError("rendered image or goal differs from scoring receipt")


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from eval import img_transform
    from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import PHYSICAL, IMAGES

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    recorded = json.loads(RECORD.read_text())
    runner = Path(__file__).with_name("pc_rgb_history_feedback_20261107_codex.py")
    if (recorded["runner_sha256"] != file_hash(runner)
            or recorded["source_physical_sha256"] != file_hash(PHYSICAL)
            or recorded["source_images_sha256"] != file_hash(IMAGES)):
        raise ValueError("recorded source mismatch")
    physical = json.loads(PHYSICAL.read_text())
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    rows = []
    try:
        with np.load(IMAGES, allow_pickle=False) as images:
            for row in recorded["rows"]:
                root, mode = row["root"], row["mode"]
                task = physical["rows"][root]
                condition = next(c for c in task["conditions"] if c["side"] == 1 and c["appearance"] == "white")
                env.reset(seed=881001 + root, options={"state": np.asarray(condition["requested_state"]),
                                                       "goal_state": np.asarray(task["goal"])})
                actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                np.testing.assert_allclose(snapshot(actor), condition["actions"]["plus"]["initial_snapshot"],
                                           atol=1e-9, rtol=0)
                frame = actor.render().copy()
                np.testing.assert_array_equal(frame, images[f"r{root}_white_1_plus_initial"])
                frames = [frame]
                goal = images[f"r{root}_white_goal"].copy()
                max_error, max_reward_error, count = 0., 0., 0
                if len(row["receipts"]) != 3 * len(row["plans"]):
                    raise ValueError("unexpected scoring calls")
                for iteration, plan in enumerate(row["plans"]):
                    check_inputs(row["receipts"][iteration * 3:(iteration + 1) * 3],
                                 frames, goal, mode, transform)
                    for item in plan["executed_steps"]:
                        _, reward, terminated, truncated, _ = actor.step(np.asarray(item["action"], np.float32))
                        max_error = max(max_error, float(np.max(np.abs(snapshot(actor) - item["snapshot"]))))
                        max_reward_error = max(max_reward_error, abs(float(reward) - item["reward"]))
                        if (bool(terminated), bool(truncated)) != (item["terminated"], item["truncated"]):
                            raise ValueError("termination mismatch")
                        count += 1
                    if len(plan["executed_steps"]) == 5:
                        frames.append(actor.render().copy())
                        frames = frames[-3:]
                if max_error > 1e-9 or max_reward_error > 1e-9:
                    raise ValueError("physical replay mismatch")
                rows.append({"root": root, "mode": mode, "executed_steps": count,
                             "plans": len(row["plans"]), "scoring_calls": len(row["receipts"]),
                             "max_state_error": max_error, "max_reward_error": max_reward_error})
    finally:
        env.close()
    if {(row["root"], row["mode"]) for row in rows} != {(0, "single"), (0, "history"), (8, "single"), (8, "history")}:
        raise ValueError("panel incomplete")
    result = {"rows": rows, "source_sha256": file_hash(RECORD), "audit_sha256": file_hash(__file__),
              "scope": "recorded development feedback replay, not independent closed-loop performance"}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
