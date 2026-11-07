"""Controlled appearance/state/action pilot; no privileged planner inputs."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_prefix_heldout_allocation_dev_20261106_codex import score
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results/pc_controlled_effect_equivalence_pilot_20261106_codex"
OUT = PREFIX.with_suffix(".json")
IMAGES = PREFIX.with_suffix(".npz")
PROTOCOL = ROOT.parent / "docs/PC_CONTROLLED_EFFECT_EQUIVALENCE_PILOT_PROTOCOL_20261106_CODEX.md"
COLORS = {"white": (255, 255, 255), "gray": (245, 245, 245)}
CONTROLS = {"zero": (0., 0.), "plus": (.5, 0.), "minus": (-.5, 0.)}


def sha_bytes(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def effects(values):
    return [float(values["plus"] - values["zero"]), float(values["minus"] - values["zero"])]


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists() or IMAGES.exists():
        raise FileExistsError(PREFIX)
    print("Controlled PushT appearance and action consequences: four exposed tasks, no camera claim.", flush=True)
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    images, rows = {}, []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        all_actions = np.asarray(data["action"][:])
        scaler = StandardScaler().fit(all_actions[~np.isnan(all_actions).any(axis=1)])
        panel = panel_rows(data)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                       relative=True, render_action=False, with_target=True)
        try:
            for task in range(4):
                original = np.asarray(data["state"][panel[task]], dtype=np.float64)
                goal = np.asarray(data["state"][panel[task] + 25], dtype=np.float64).copy()
                goal[-2:] = 0
                configurations = []
                for side in (-1, 1):
                    state = original.copy()
                    state[:2] = state[2:4] + [60 * side, 0]
                    state[-2:] = 0
                    if np.any(state[:2] < 20) or np.any(state[:2] > 492):
                        raise ValueError("candidate agent outside declared safe region")
                    configurations.append(state)
                task_data = {"task": task, "dataset_row": int(panel[task]), "goal": goal.tolist(), "conditions": []}
                for color_name, color in COLORS.items():
                    env.reset(seed=771001 + task, options={"state": goal.copy(), "goal_state": goal.copy()})
                    actor = env.unwrapped
                    actor.variation_space["background"]["color"].set_value(np.asarray(color, dtype=np.uint8))
                    before_goal = snapshot(actor)
                    goal_frame = native_world_pixels(actor.render())
                    np.testing.assert_array_equal(snapshot(actor), before_goal)
                    images[f"t{task}_{color_name}_goal"] = goal_frame
                    goal_tensor = transform(goal_frame).to("cuda")
                    with torch.inference_mode():
                        goal_embedding = model.encode({"pixels": goal_tensor[None, None]})["emb"][0, 0]
                    for side, state in zip((-1, 1), configurations, strict=True):
                        condition = {"side": side, "appearance": color_name, "requested": state.tolist(),
                                     "goal_hash": sha_bytes(goal_frame), "actions": {}}
                        initial_frame = None
                        for name, control in CONTROLS.items():
                            env.reset(seed=771001 + task, options={"state": state.copy(), "goal_state": goal.copy()})
                            actor = env.unwrapped
                            actor.variation_space["background"]["color"].set_value(np.asarray(color, dtype=np.uint8))
                            initial = snapshot(actor)
                            frame = native_world_pixels(actor.render())
                            np.testing.assert_array_equal(initial, snapshot(actor))
                            if initial_frame is not None:
                                np.testing.assert_array_equal(frame, initial_frame)
                            initial_frame = frame
                            physical_action = np.repeat(np.asarray(control, dtype=np.float32)[None], 5, axis=0)
                            packet = torch.tensor(scaler.transform(physical_action).reshape(1, 1, 10),
                                                  device="cuda", dtype=torch.float32)
                            if not np.allclose(scaler.inverse_transform(packet.cpu().numpy().reshape(5, 2)),
                                               physical_action, atol=1e-6):
                                raise ValueError("action normalization mismatch")
                            pixel_tensor = transform(frame)
                            context = {"pixels": pixel_tensor[None, None, None],
                                       "goal": goal_tensor.cpu()[None, None, None]}
                            with torch.inference_mode():
                                predicted = float(score(model, context, packet)[0].cpu())
                            trace = []
                            for action in physical_action:
                                obs, _, terminated, truncated, info = actor.step(action.copy())
                                if truncated:
                                    raise ValueError("unexpected truncation")
                                trace.append({"state": np.asarray(obs["state"]).tolist(), "snapshot": snapshot(actor).tolist(),
                                              "contacts": int(info["n_contacts"]), "success": bool(terminated)})
                            final_frame = native_world_pixels(actor.render())
                            with torch.inference_mode():
                                real_tensor = transform(final_frame).to("cuda")
                                real_embedding = model.encode({"pixels": real_tensor[None, None]})["emb"][0, 0]
                                actual = float((real_embedding - goal_embedding).square().sum().cpu())
                            physical_cost = float(np.linalg.norm(np.asarray(trace[-1]["state"]) - goal))
                            key = f"t{task}_{color_name}_{side}_{name}"
                            images[key + "_initial"] = frame
                            images[key + "_final"] = final_frame
                            condition["actions"][name] = {
                                "physical_action": physical_action.tolist(), "packet": packet.cpu().tolist(),
                                "initial_snapshot": initial.tolist(), "initial_hash": sha_bytes(frame),
                                "final_hash": sha_bytes(final_frame), "trace": trace,
                                "predicted_cost": predicted, "actual_encoded_cost": actual,
                                "physical_state_distance": physical_cost}
                        for metric in ("predicted_cost", "actual_encoded_cost", "physical_state_distance"):
                            condition[metric + "_effect"] = effects(
                                {name: row[metric] for name, row in condition["actions"].items()})
                        task_data["conditions"].append(condition)
                for side in (-1, 1):
                    white, gray = [x for x in task_data["conditions"] if x["side"] == side]
                    for name in CONTROLS:
                        w, g = white["actions"][name], gray["actions"][name]
                        np.testing.assert_allclose(w["initial_snapshot"], g["initial_snapshot"], atol=1e-9, rtol=0)
                        if w["trace"] != g["trace"]:
                            raise ValueError("appearance changed physics")
                for color_name in COLORS:
                    left, right = [x for x in task_data["conditions"] if x["appearance"] == color_name]
                    # Snapshot indices 5:7 and 9 are block position and angle.
                    for name in CONTROLS:
                        a = np.asarray(left["actions"][name]["initial_snapshot"])
                        b = np.asarray(right["actions"][name]["initial_snapshot"])
                        np.testing.assert_allclose(a[[5, 6, 9]], b[[5, 6, 9]], atol=1e-8, rtol=0)
                rows.append(task_data)
                print(f"task {task}: 12 controlled outcomes", flush=True)
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with IMAGES.open("xb") as file:
        np.savez_compressed(file, **images)
    with OUT.open("x") as file:
        json.dump({"protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                   "model_sha256": fingerprint, "images_sha256": hashlib.sha256(IMAGES.read_bytes()).hexdigest(),
                   "scope": "four exposed tasks, appearance not camera, diagnostic only",
                   "rows": rows}, file, indent=2, allow_nan=False)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
