"""Evaluation-only static position/angle/pusher factorial on exposed task 1."""
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE, OUT as PRIOR, SOURCE
from experiments.pc_pusht_shared_pool_real_endpoint_20261107_codex import OUT as ENDPOINT
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT
from experiments.pc_pusht_task1_endpoint_factor_20261107_codex import OUT as PREVIOUS

OUT = ROOT / "results/pc_pusht_task1_pose_angle_factor_20261107_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_TASK1_POSE_ANGLE_FACTOR_PROTOCOL_20261107_CODEX.md"
LABELS = ("failure", "success")


def effects(rows):
    """Return paired success-minus-failure changes; invalid pairs have no effect."""
    table = {(r["pusher"], r["position"], r["angle"]): r for r in rows}
    result = []
    for factor in range(3):
        for fixed in itertools.product(LABELS, repeat=2):
            keys = []
            for level in LABELS:
                key = list(fixed)
                key.insert(factor, level)
                keys.append(tuple(key))
            left, right = (table[k] for k in keys)
            result.append({"factor": ("pusher", "position", "angle")[factor],
                           "failure_key": keys[0], "success_key": keys[1],
                           "delta_cost": (right["goal_cost"] - left["goal_cost"]
                                          if left["valid"] and right["valid"] else None)})
    return result


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError(OUT)
    prior = json.loads(PRIOR.with_suffix(".json").read_text())
    alignment = json.loads(SOURCE.with_suffix(".json").read_text())
    stem = ENDPOINT.parent / f"{ENDPOINT.name}_task1"
    saved = json.loads(stem.with_suffix(".json").read_text())
    physical_path = BASE / "physical_1.json"
    previous = json.loads(PREVIOUS.with_suffix(".json").read_text())
    if (file_hash(stem.with_suffix(".npz")) != saved["arrays_sha256"]
            or file_hash(physical_path) != saved["physical_sha256"]
            or file_hash(SOURCE.with_suffix(".npz")) != alignment["arrays_sha256"]
            or file_hash(PREVIOUS.with_suffix(".npz")) != previous["images_sha256"]):
        raise ValueError("source artifact changed")
    physical = json.loads(physical_path.read_text())
    candidates = {r["action_sha256"]: r for r in physical["records"]}
    original = candidates[prior["plans"][1]["selected_sha256"]]
    successes = [candidates[h] for h in saved["hashes"] if candidates[h]["final_success"]]
    if original["final_success"] or len(successes) != 1:
        raise ValueError("unexpected selected chain or success count")
    endpoints = {"failure": np.asarray(original["packet_states"][-1], np.float64),
                 "success": np.asarray(successes[0]["packet_states"][-1], np.float64)}
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as data:
        i = next(i for i, r in enumerate(alignment["rows"]) if r["task"] == 1 and r["selected"])
        goal_embedding = data["goal"][i].copy()
    previous_cost = {r["name"]: r["goal_cost"] for r in previous["measurements"]}
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != saved["model_sha256"] or fingerprint != previous["model_sha256"]:
        raise ValueError("checkpoint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows, images = [], {}
    with h5py.File(DATA, "r") as data, np.load(PREVIOUS.with_suffix(".npz"), allow_pickle=False) as old:
        goal_state = np.asarray(data["state"][physical["data_row"] + 25], np.float64)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for pusher, position, angle in itertools.product(LABELS, repeat=3):
                env.reset(seed=871002)
                actor = env.unwrapped
                desired = endpoints[position].copy()
                desired[:2] = endpoints[pusher][:2]
                desired[4] = endpoints[angle][4]
                desired[5:] = 0
                actor._set_state(desired)
                actor._set_goal_state(goal_state)
                observed = np.asarray(actor._get_obs(), np.float64)
                error = float(np.abs(observed[:5] - desired[:5]).max())
                rgb = native_world_pixels(actor.render())
                key = f"pusher_{pusher}_position_{position}_angle_{angle}"
                images[key] = rgb
                with torch.inference_mode():
                    emb = model.encode({"pixels": transform(rgb)[None, None].to("cuda")})["emb"][0, 0]
                cost = float(np.square(emb.cpu().numpy() - goal_embedding).sum())
                reference = f"object_{position}_pusher_{pusher}"
                same_pixels = bool(np.array_equal(rgb, old[reference])) if position == angle else None
                cost_error = abs(cost - previous_cost[reference]) if position == angle else None
                valid = error <= 1e-4 and (same_pixels is None or (same_pixels and cost_error <= 1e-5))
                success, _ = actor.eval_state(goal_state, observed)
                angle_diff = abs(goal_state[4] - observed[4])
                rows.append({"name": key, "pusher": pusher, "position": position, "angle": angle,
                             "requested": desired.tolist(), "observed": observed.tolist(),
                             "reset_max_error": error, "prior_pixels_equal": same_pixels,
                             "prior_cost_abs_error": cost_error, "valid": bool(valid), "goal_cost": cost,
                             "joint_position_distance": float(np.linalg.norm(goal_state[:4] - observed[:4])),
                             "angle_distance": float(min(angle_diff, 2 * np.pi - angle_diff)),
                             "physical_success": bool(success)})
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUT.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, **images)
    output = {"protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
              "previous_sha256": file_hash(PREVIOUS.with_suffix(".json")),
              "physical_sha256": file_hash(physical_path), "images_sha256": file_hash(OUT.with_suffix(".npz")),
              "model_sha256": fingerprint, "goal_state": goal_state.tolist(),
              "measurements": rows, "effects": effects(rows)}
    with OUT.with_suffix(".json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({"measurements": rows, "effects": output["effects"]}, indent=2))


if __name__ == "__main__":
    main()
