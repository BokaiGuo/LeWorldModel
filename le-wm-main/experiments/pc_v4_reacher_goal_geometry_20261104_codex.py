"""Privileged static target-image and local geometry diagnostic."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_reacher_final_packet_latent_gap_20261104_codex as old
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

prev, live, ROOT = old.prev, old.live, old.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_REACHER_GOAL_GEOMETRY_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_reacher_goal_geometry_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
ARRAYS = PREFIX.with_name(PREFIX.name + "_arrays.npz")
LATENT_PREFIX = ROOT / "results/pc_v4_reacher_final_packet_latent_gap_v2_20261104_codex"
LATENT_RESULT = LATENT_PREFIX.with_name(LATENT_PREFIX.name + "_result.json")
LATENT_ARRAYS = LATENT_PREFIX.with_name(LATENT_PREFIX.name + "_arrays.npz")
LATENT_AUDIT = LATENT_PREFIX.with_name(LATENT_PREFIX.name + "_audit.json")


def main():
    if any(p.exists() for p in (MANIFEST, RESULT, ARRAYS)):
        raise FileExistsError("existing or partial output")
    prev.v4.configure()
    host, setup, p0 = live.host, live.setup, live.p0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("GPU UUID required")
    sha = host.bridge.sha_file
    previous = json.loads(prev.RESULT.read_text())
    latent = json.loads(LATENT_RESULT.read_text())
    latent_audit = json.loads(LATENT_AUDIT.read_text())
    if (latent["arrays_sha256"] != sha(LATENT_ARRAYS) or latent_audit["result_sha256"] != sha(LATENT_RESULT)
            or latent_audit["arrays_sha256"] != sha(LATENT_ARRAYS)
            or previous["manifest_sha256"] != sha(prev.MANIFEST)):
        raise ValueError("source binding mismatch")
    seal.verify(json.loads(prev.MANIFEST.read_text())["seal"])
    source = json.loads(live.run.SOURCE.read_text())
    dependencies = seal.capture([Path(__file__), PROTOCOL, prev.MANIFEST, prev.RESULT, prev.CONTEXT0,
                                 prev.CONTEXT2, LATENT_RESULT, LATENT_ARRAYS, LATENT_AUDIT])
    host.write_new(MANIFEST, {"seal": dependencies, "scope": "post-hoc privileged static rendering"})
    print("Render target configuration and 15 privileged static interpolation points.", flush=True)
    with np.load(LATENT_ARRAYS, allow_pickle=False) as arrays:
        saved_frames = arrays["frames"].copy()
        saved_observed = arrays["observed"].copy()
    with np.load(prev.CONTEXT2, allow_pickle=False) as c:
        context = {k: c[k].copy() for k in c.files}
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        setup.install_missing_field_compat()
        env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                            target_delta=(.65, -.65))
        frames, rows = [], []
        try:
            _, info = env.reset(seed=20263160)
            root_goal = np.asarray(info["goal"], np.uint8)
            if live.context_lib.rgb_digest(root_goal) != live.context_lib.rgb_digest(context["goal"]):
                raise ValueError("reset goal image mismatch")
            inner = env.unwrapped.env
            physics = inner.physics
            target = np.asarray(inner.task.target_qpos, np.float64).copy()
            np.testing.assert_allclose(target, previous["rows"][0]["before"]["target_qpos"], rtol=0, atol=1e-10)
            specs = [("target", None, 0., target)]
            for branch in previous["rows"]:
                endpoint = np.asarray(branch["trajectory"][-1]["qpos"], np.float64)
                delta = (endpoint - target + np.pi) % (2 * np.pi) - np.pi
                for alpha in (.25, .5, 1.):
                    specs.append((branch["name"], branch["name"], alpha, target + alpha * delta))
            for name, branch_name, alpha, qpos in specs:
                physics.data.qpos[:] = qpos
                physics.data.qvel[:] = 0
                physics.forward()
                image = np.asarray(env.render(), np.uint8).copy()
                actual = np.asarray(physics.data.qpos).copy()
                distance = float(p0._qpos_distance(actual, target))
                rows.append({"name": name, "branch": branch_name, "alpha": alpha,
                             "qpos": actual.tolist(), "distance": distance,
                             "frame_sha256": live.context_lib.rgb_digest(image)})
                frames.append(image)
        finally:
            env.close()
        device = torch.device("cuda:0")
        stats = Path(source["action_stats_path"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        model = setup.load_compatible_pretrained(Path(source["weights_path"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        model_hash = setup._parameter_hash(model)
        if model_hash != source["model_parameter_hash_expected"]:
            raise ValueError("model hash mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        _, goal, _ = p0._prepare(processor, context["frames"], context["root"], context["goal"],
                                  context["history"], device)
        embeddings = []
        with torch.no_grad():
            goal_latent = p0._goal_latent(model, goal).cpu().numpy()
            for frame in frames:
                _, img, _ = p0._prepare(processor, context["frames"], context["root"],
                                         np.ascontiguousarray(frame), context["history"], device)
                embeddings.append(p0._goal_latent(model, img).cpu().numpy())
        embeddings = np.stack(embeddings)
        if setup._parameter_hash(model) != model_hash:
            raise ValueError("model changed")
    seal.verify(dependencies)
    images = np.stack(frames)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, frames=images, latent=embeddings, goal_latent=goal_latent,
                            goal_rgb=context["goal"], prior_frames=saved_frames, prior_latent=saved_observed)
        stream.flush()
        os.fsync(stream.fileno())
    for i, row in enumerate(rows):
        row["latent_goal_mse"] = float(np.mean((embeddings[i] - goal_latent) ** 2))
        row["pixel_goal_mse"] = float(np.mean((images[i].astype(np.float64) - context["goal"]) ** 2))
        if row["branch"] is not None and row["alpha"] == 1:
            branch_index = next(j for j, b in enumerate(previous["rows"]) if b["name"] == row["branch"])
            row["matches_saved_frame"] = bool(np.array_equal(images[i], saved_frames[branch_index]))
            row["prior_latent_max_abs"] = float(np.max(np.abs(embeddings[i] - saved_observed[branch_index])))
    host.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "arrays_sha256": sha(ARRAYS),
                            "target_qpos": target.tolist(), "target_image_equal": bool(np.array_equal(images[0], context["goal"])),
                            "rows": rows, "model_hash": model_hash, "cuda_allocation": allocation,
                            "scope": "privileged static render, not control"})
    print(f"Target image exact: {np.array_equal(images[0], context['goal'])}.", flush=True)


if __name__ == "__main__":
    main()
