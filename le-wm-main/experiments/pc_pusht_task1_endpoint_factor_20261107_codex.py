"""Static endpoint factorial for the exposed task 1 selection failure."""
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

OUT = ROOT / "results/pc_pusht_task1_endpoint_factor_20261107_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_TASK1_ENDPOINT_FACTOR_PROTOCOL_20261107_CODEX.md"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError(OUT)
    prior = json.loads(PRIOR.with_suffix(".json").read_text())
    align = json.loads(SOURCE.with_suffix(".json").read_text())
    stem = ENDPOINT.parent / f"{ENDPOINT.name}_task1"
    saved = json.loads(stem.with_suffix(".json").read_text())
    physical_path = BASE / "physical_1.json"
    if file_hash(stem.with_suffix(".npz")) != saved["arrays_sha256"] or \
       file_hash(physical_path) != saved["physical_sha256"] or \
       file_hash(SOURCE.with_suffix(".npz")) != align["arrays_sha256"]:
        raise ValueError("source artifact changed")
    physical = json.loads(physical_path.read_text())
    candidates = {r["action_sha256"]: r for r in physical["records"]}
    plan = prior["plans"][1]
    original = candidates[plan["selected_sha256"]]
    successes = [candidates[h] for h in saved["hashes"] if candidates[h]["final_success"]]
    if original["final_success"] or len(successes) != 1:
        raise ValueError("unexpected selection or success count")
    success = successes[0]
    endpoints = {"failure": np.asarray(original["packet_states"][-1], np.float64),
                 "success": np.asarray(success["packet_states"][-1], np.float64)}
    idx = {h: i for i, h in enumerate(saved["hashes"])}
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
        saved_embeddings = {name: arrays["embeddings"][idx[record["action_sha256"]]].copy()
                            for name, record in (("failure", original), ("success", success))}
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as arrays:
        row_index = next(i for i, r in enumerate(align["rows"]) if r["task"] == 1 and r["selected"])
        goal = arrays["goal"][row_index].copy()
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != saved["model_sha256"]:
        raise ValueError("checkpoint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    images, measurements = {}, []
    with h5py.File(DATA, "r") as data:
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for obj in ("failure", "success"):
                for agent in ("failure", "success"):
                    env.reset(seed=871002)
                    actor = env.unwrapped
                    desired = endpoints[obj].copy()
                    desired[:2] = endpoints[agent][:2]
                    desired[5:] = 0
                    actor._set_state(desired)
                    actor._set_goal_state(np.asarray(data["state"][physical["data_row"] + 25], np.float64))
                    observed = np.asarray(actor._get_obs(), np.float64)
                    position_error = float(np.abs(observed[:5] - desired[:5]).max())
                    rgb = native_world_pixels(actor.render())
                    name = f"object_{obj}_pusher_{agent}"
                    images[name] = rgb
                    with torch.inference_mode():
                        embedding = model.encode({"pixels": transform(rgb)[None, None].to("cuda")})["emb"][0, 0]
                    emb = embedding.cpu().numpy()
                    base_error = float(np.abs(emb - saved_embeddings[obj]).max()) if obj == agent else None
                    valid = position_error <= 1e-4 and (base_error is None or base_error <= 1e-5)
                    measurements.append({"name": name, "object": obj, "pusher": agent,
                                         "requested": desired.tolist(), "observed": observed.tolist(),
                                         "position_max_error": position_error, "base_embedding_max_error": base_error,
                                         "valid_static_reconstruction": valid,
                                         "goal_cost": float(np.square(emb - goal).sum())})
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUT.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, **images)
    output = {"protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
              "endpoint_sha256": file_hash(stem.with_suffix(".json")), "images_sha256": file_hash(OUT.with_suffix(".npz")),
              "model_sha256": fingerprint, "measurements": measurements}
    with OUT.with_suffix(".json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
