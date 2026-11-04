"""Compare frozen-model predictions to encodings of replayed successor RGB."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_reacher_final_packet_counterfactual_20261104_codex as prev
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

live, v4, ROOT = prev.live, prev.v4, prev.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_REACHER_FINAL_PACKET_LATENT_GAP_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_reacher_final_packet_latent_gap_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
ARRAYS = PREFIX.with_name(PREFIX.name + "_arrays.npz")
PREV_AUDIT = prev.PREFIX.with_name(prev.PREFIX.name + "_audit.json")


def main():
    if any(p.exists() for p in (MANIFEST, RESULT, ARRAYS)):
        raise FileExistsError("existing or partial result")
    v4.configure()
    host, setup, p0, adapter = live.host, live.setup, live.p0, live.adapter
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = host.bridge.sha_file
    previous = json.loads(prev.RESULT.read_text())
    old_manifest = json.loads(prev.MANIFEST.read_text())
    old_audit = json.loads(PREV_AUDIT.read_text())
    if previous["manifest_sha256"] != sha(prev.MANIFEST) or old_audit["result_sha256"] != sha(prev.RESULT):
        raise ValueError("counterfactual binding mismatch")
    seal.verify(old_manifest["seal"])
    source = json.loads(live.run.SOURCE.read_text())
    dependencies = seal.capture([Path(__file__), PROTOCOL, prev.MANIFEST, prev.RESULT, PREV_AUDIT,
                                 prev.CONTEXT0, prev.CONTEXT2, prev.EXEC, prev.DECISION])
    host.write_new(MANIFEST, {"seal": dependencies, "scope": "post-hoc same-model same-backend RGB diagnostic"})
    print("Compare five sealed predictions with encodings of replayed RGB outcomes.", flush=True)
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        setup.install_missing_field_compat()
        env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                            target_delta=(.65, -.65))
        frames, end_states = [], []
        try:
            with np.load(prev.CONTEXT0, allow_pickle=False) as c:
                warmup = c["history"].reshape(-1, 2).astype(np.float32)
                goal_sha = live.context_lib.rgb_digest(c["goal"])
            with np.load(prev.CONTEXT2, allow_pickle=False) as c:
                context = {k: c[k].copy() for k in c.files}
            execution = json.loads(prev.EXEC.read_text())

            def execute(actions):
                for action in actions:
                    _, _, terminated, truncated, _ = env.step(np.asarray(action, np.float32))
                    if terminated or truncated:
                        raise ValueError("unexpected termination")

            for branch in previous["rows"]:
                _, info = env.reset(seed=20263160)
                if live.context_lib.rgb_digest(info["goal"]) != goal_sha:
                    raise ValueError("goal RGB mismatch")
                execute(warmup)
                for item in execution["rows"][:2]:
                    execute(item["dispatch"])
                if live.context_lib.rgb_digest(np.asarray(env.render(), np.uint8)) != live.context_lib.rgb_digest(context["root"]):
                    raise ValueError("initial RGB mismatch")
                before = prev.physical(env)
                for key in ("qpos", "qvel", "target_qpos"):
                    np.testing.assert_allclose(before[key], branch["before"][key], rtol=0, atol=1e-10)
                execute(branch["raw"])
                final = prev.physical(env)
                for key in ("qpos", "qvel", "target_qpos"):
                    np.testing.assert_allclose(final[key], branch["trajectory"][-1][key], rtol=0, atol=1e-10)
                frames.append(np.asarray(env.render(), np.uint8))
                end_states.append(final)
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
            raise ValueError("model mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        lo, hi = setup._normalized_action_bounds(mean, std, np.full(2, -1), np.full(2, 1))
        lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
        pixels, goal, history = p0._prepare(processor, context["frames"], context["root"], context["goal"],
                                            context["history"], device)
        raw = np.asarray([branch["raw"] for branch in previous["rows"]], np.float32)
        with torch.no_grad():
            encoded = p0._latent_context(model, pixels, history)
            target = p0._goal_latent(model, goal)
            predicted = adapter.cached_terminal(model, encoded, history,
                                                adapter.raw_to_packets(raw, mean, std, device=device), lower, upper)
            observed = []
            for image in frames:
                _, final_image, _ = p0._prepare(processor, context["frames"], context["root"], image,
                                                 context["history"], device)
                observed.append(p0._goal_latent(model, final_image))
            observed = torch.stack(observed)
            predicted, observed, target = predicted.cpu().numpy(), observed.cpu().numpy(), target.cpu().numpy()
        model_scores = np.mean((predicted - target) ** 2, axis=1)
        for value, branch in zip(model_scores, previous["rows"], strict=True):
            if min(abs(float(value) - x) for x in branch["model_scores"]) > 2e-6:
                raise ValueError("prediction score does not match sealed trace")
        if setup._parameter_hash(model) != model_hash:
            raise ValueError("model changed")
    seal.verify(dependencies)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, frames=np.stack(frames), predicted=predicted, observed=observed,
                            target=target, actions=raw)
        stream.flush()
        os.fsync(stream.fileno())
    rows = []
    for i, branch in enumerate(previous["rows"]):
        rows.append({"name": branch["name"], "frame_sha256": live.context_lib.rgb_digest(frames[i]),
                     "model_goal_mse": float(model_scores[i]),
                     "observed_goal_mse": float(np.mean((observed[i] - target) ** 2)),
                     "prediction_observation_mse": float(np.mean((predicted[i] - observed[i]) ** 2)),
                     "physical_distance": end_states[i]["distance"]})
    host.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "arrays_sha256": sha(ARRAYS),
                            "model_hash": model_hash, "rows": rows, "cuda_allocation": allocation,
                            "scope": "same-backend single-root post-hoc replay"})
    print("Saved frames, frozen-model latents and action-level comparisons.", flush=True)


if __name__ == "__main__":
    main()
