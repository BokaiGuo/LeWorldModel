"""Replay v4's recorded central-difference probes in Reacher."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_reacher_final_packet_counterfactual_20261104_codex as prev
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

live, ROOT = prev.live, prev.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_REACHER_ACTUAL_FD_PHYSICAL_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_reacher_actual_fd_physical_20261104_codex"
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
    prior_latent = json.loads(LATENT_RESULT.read_text())
    if (prior_latent["arrays_sha256"] != sha(LATENT_ARRAYS)
            or json.loads(LATENT_AUDIT.read_text())["result_sha256"] != sha(LATENT_RESULT)
            or previous["manifest_sha256"] != sha(prev.MANIFEST)):
        raise ValueError("prior binding mismatch")
    seal.verify(json.loads(prev.MANIFEST.read_text())["seal"])
    decision = json.loads(prev.DECISION.read_text())
    trace = decision["trace"]
    batches = []
    for index, info in enumerate(decision["result"]["first_response"]):
        nominal = np.asarray(info["nominal"], dtype=float)
        axes = np.eye(10).reshape(10, 5, 2)
        probes = np.concatenate((nominal[None], nominal[None] + .05*axes, nominal[None] - .05*axes))
        matches = [b for b in trace if np.array_equal(np.asarray(b["requested"]), probes)]
        if len(matches) != 1:
            raise ValueError(f"unique recorded probe batch required: {index}")
        batch = matches[0]
        pos = np.asarray(batch["positions"])
        np.testing.assert_allclose(((pos[1:11]-pos[11:21])/.1).T, info["matrix"], rtol=0, atol=1e-12)
        batches.append(batch)
    actions = np.asarray([b["raw"] for b in batches], np.float32)
    source = json.loads(live.run.SOURCE.read_text())
    dependencies = seal.capture([Path(__file__), PROTOCOL, prev.MANIFEST, prev.RESULT, prev.CONTEXT0,
                                 prev.CONTEXT2, prev.EXEC, prev.DECISION, LATENT_RESULT, LATENT_ARRAYS, LATENT_AUDIT])
    host.write_new(MANIFEST, {"seal": dependencies, "scope": "one exposed root, actual probe replay"})
    print("Replay 42 actual v4 finite-difference probes with same physical state.", flush=True)
    with np.load(prev.CONTEXT0, allow_pickle=False) as c:
        warmup = c["history"].reshape(-1, 2).astype(np.float32)
    with np.load(prev.CONTEXT2, allow_pickle=False) as c:
        context = {k: c[k].copy() for k in c.files}
    with np.load(LATENT_ARRAYS, allow_pickle=False) as c:
        reference_frames = c["frames"].copy()
    execution = json.loads(prev.EXEC.read_text())
    frames, states = [], []
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        setup.install_missing_field_compat()
        env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                            target_delta=(.65, -.65))
        try:
            def execute(packet):
                for action in packet:
                    _, _, terminated, truncated, _ = env.step(np.asarray(action, np.float32))
                    if terminated or truncated:
                        raise ValueError("unexpected termination")

            for index, action in enumerate(actions.reshape(-1, 5, 2)):
                _, info = env.reset(seed=20263160)
                if live.context_lib.rgb_digest(info["goal"]) != live.context_lib.rgb_digest(context["goal"]):
                    raise ValueError("goal mismatch")
                execute(warmup)
                for row in execution["rows"][:2]:
                    execute(row["dispatch"])
                if live.context_lib.rgb_digest(np.asarray(env.render(), np.uint8)) != live.context_lib.rgb_digest(context["root"]):
                    raise ValueError("root RGB mismatch")
                before = prev.physical(env)
                for key in ("qpos", "qvel", "target_qpos"):
                    np.testing.assert_allclose(before[key], previous["rows"][0]["before"][key], rtol=0, atol=1e-10)
                execute(action)
                frame = np.asarray(env.render(), np.uint8).copy()
                frames.append(frame)
                states.append(prev.physical(env))
                if index in (0, 21):
                    idx = 3 if index == 0 else 4
                    if not np.array_equal(frame, reference_frames[idx]):
                        raise ValueError("nominal frame mismatch")
            print("Physical replay complete; encoding 42 actual successor frames.", flush=True)
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
            raise ValueError("checkpoint mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        _, goal, _ = p0._prepare(processor, context["frames"], context["root"], context["goal"],
                                  context["history"], device)
        with torch.no_grad():
            target = p0._goal_latent(model, goal).cpu().numpy()
            encoded = []
            for frame in frames:
                _, image, _ = p0._prepare(processor, context["frames"], context["root"],
                                           np.ascontiguousarray(frame), context["history"], device)
                encoded.append(p0._goal_latent(model, image).cpu().numpy())
        encoded = np.stack(encoded).reshape(2, 21, -1)
        chart = np.asarray(decision["chart"])
        positions = encoded @ chart
        if setup._parameter_hash(model) != model_hash:
            raise ValueError("model changed")
    seal.verify(dependencies)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, frames=np.asarray(frames).reshape(2, 21, *frames[0].shape),
                            actions=actions, observed=encoded, chart=chart, target=target,
                            model_positions=np.asarray([b["positions"] for b in batches]))
        stream.flush()
        os.fsync(stream.fileno())
    rows = []
    for k, info in enumerate(decision["result"]["first_response"]):
        rows.append({"nominal": k, "model_matrix": info["matrix"],
                     "observed_matrix": ((positions[k, 1:11] - positions[k, 11:21])/.1).T.tolist(),
                     "observed_goal_mse": np.mean((encoded[k] - target)**2, axis=1).tolist(),
                     "model_goal_mse": batches[k]["scores"],
                     "states": states[k*21:(k+1)*21]})
    host.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "arrays_sha256": sha(ARRAYS), "rows": rows,
                            "model_hash": model_hash, "cuda_allocation": allocation,
                            "scope": "same backend and checkpoint, one root and one scale"})
    print("Saved physical probe trajectories, frames and observed latents.", flush=True)


if __name__ == "__main__":
    main()
