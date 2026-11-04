"""Direct physical and model probes of v4's saved spectral directions."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_reacher_actual_fd_physical_20261104_codex as old
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

prev, live, ROOT = old.prev, old.live, old.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_REACHER_DIRECT_SPECTRAL_PROBE_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_reacher_direct_spectral_probe_v3_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
ARRAYS = PREFIX.with_name(PREFIX.name + "_arrays.npz")


def make_actions(decision):
    actions, requested, directions, metadata = [], [], [], []
    bank = [np.asarray(x["prefix"], np.float32) for x in decision["result"]["first_bank"]]
    for k, info in enumerate(decision["result"]["first_response"]):
        nominal = np.asarray(info["nominal"], np.float64)
        _, _, right = np.linalg.svd(np.asarray(info["matrix"]), full_matrices=False)
        directions.append(np.stack([v.reshape(5, 2) / np.max(np.abs(v)) for v in right]))
        variants = [(None, 0., 0, nominal)]
        for mode, direction in enumerate(directions[-1]):
            for scale in (.05, .5):
                for sign in (-1, 1):
                    variants.append((mode, scale, sign, nominal + sign * scale * direction))
        for mode, scale, sign, raw in variants:
            clipped = np.clip(raw, -1, 1).astype(np.float32)
            matches = [i for i, b in enumerate(bank) if np.allclose(b, clipped, atol=1e-7, rtol=0)]
            if scale == .5 and not matches:
                raise ValueError("large-scale mode not in sealed first bank")
            metadata.append({"nominal": k, "mode": mode, "scale": scale, "sign": sign,
                             "clipped_coordinates": int(np.count_nonzero(raw != np.clip(raw, -1, 1))),
                             "bank_indices": matches})
            requested.append(raw)
            actions.append(clipped)
    return np.asarray(actions), np.asarray(requested), np.asarray(directions), metadata


def main():
    if any(p.exists() for p in (MANIFEST, RESULT, ARRAYS)):
        raise FileExistsError("existing or partial output")
    prev.v4.configure()
    host, setup, p0, adapter = live.host, live.setup, live.p0, live.adapter
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("GPU UUID required")
    sha = host.bridge.sha_file
    prior = json.loads(old.RESULT.read_text())
    reference_state = json.loads(prev.RESULT.read_text())["rows"][0]["before"]
    if (prior["manifest_sha256"] != sha(old.MANIFEST) or prior["arrays_sha256"] != sha(old.ARRAYS)
            or json.loads(old.PREFIX.with_name(old.PREFIX.name + "_audit.json").read_text())["result_sha256"] != sha(old.RESULT)
            or json.loads(old.PREFIX.with_name(old.PREFIX.name + "_reencode.json").read_text())["result_sha256"] != sha(old.RESULT)):
        raise ValueError("prior probe binding mismatch")
    seal.verify(json.loads(old.MANIFEST.read_text())["seal"])
    decision = json.loads(prev.DECISION.read_text())
    actions, requested, directions, metadata = make_actions(decision)
    deps = seal.capture([Path(__file__), PROTOCOL, old.MANIFEST, old.RESULT, old.ARRAYS,
                         old.PREFIX.with_name(old.PREFIX.name + "_audit.json"),
                         old.PREFIX.with_name(old.PREFIX.name + "_reencode.json"),
                         prev.DECISION, prev.CONTEXT0, prev.CONTEXT2, prev.EXEC,
                         Path(adapter.__file__)])
    host.write_new(MANIFEST, {"seal": deps, "scope": "one exposed root, direct spectral probes"})
    print("Direct spectral probes: 18 physical trajectories and frozen-model JVP.", flush=True)
    with np.load(prev.CONTEXT0, allow_pickle=False) as c:
        warmup = c["history"].reshape(-1, 2).astype(np.float32)
    with np.load(prev.CONTEXT2, allow_pickle=False) as c:
        context = {k: c[k].copy() for k in c.files}
    execution = json.loads(prev.EXEC.read_text())
    source = json.loads(live.run.SOURCE.read_text())
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

            for action in actions:
                _, info = env.reset(seed=20263160)
                if live.context_lib.rgb_digest(info["goal"]) != live.context_lib.rgb_digest(context["goal"]):
                    raise ValueError("goal mismatch")
                execute(warmup)
                for row in execution["rows"][:2]:
                    execute(row["dispatch"])
                if live.context_lib.rgb_digest(np.asarray(env.render(), np.uint8)) != live.context_lib.rgb_digest(context["root"]):
                    raise ValueError("root mismatch")
                before = prev.physical(env)
                for key in ("qpos", "qvel", "target_qpos"):
                    np.testing.assert_allclose(before[key], reference_state[key], rtol=0, atol=1e-10)
                execute(action)
                frames.append(np.asarray(env.render(), np.uint8).copy())
                states.append(prev.physical(env))
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
        pixels, goal, history = p0._prepare(processor, context["frames"], context["root"],
                                             context["goal"], context["history"], device)
        lo, hi = setup._normalized_action_bounds(mean, std, np.full(2, -1), np.full(2, 1))
        lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
        chart = torch.as_tensor(np.asarray(decision["chart"], np.float32), device=device)
        with torch.no_grad():
            encoded = p0._latent_context(model, pixels, history)
            target = p0._goal_latent(model, goal)
            observed = []
            for frame in frames:
                _, image, _ = p0._prepare(processor, context["frames"], context["root"],
                                           np.ascontiguousarray(frame), context["history"], device)
                observed.append(p0._goal_latent(model, image).cpu().numpy())
            predicted = adapter.cached_terminal(model, encoded, history,
                adapter.raw_to_packets(actions, mean, std, device=device), lower, upper).cpu().numpy()
        mean_t = torch.as_tensor(mean, device=device)
        std_t = torch.as_tensor(std, device=device)
        def project(raw):
            future = (raw.reshape(1, 1, 10) - mean_t) / std_t
            return (adapter.cached_terminal(model, encoded, history, future, lower, upper) @ chart).reshape(-1)
        jvps = []
        for k in range(2):
            raw = torch.as_tensor(actions[k * 9], device=device).clone().requires_grad_(True)
            output = project(raw)
            jacobian = torch.stack([torch.autograd.grad(output[i], raw, retain_graph=i == 0)[0]
                                    for i in range(2)])
            jvps.append([(jacobian * torch.as_tensor(directions[k, m], device=device,
                dtype=raw.dtype)).sum(dim=(1, 2)).detach().cpu().numpy() for m in range(2)])
        observed = np.stack(observed)
        if setup._parameter_hash(model) != model_hash:
            raise ValueError("model drift")
    seal.verify(deps)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, actions=actions, requested=requested, directions=directions,
                            frames=np.asarray(frames), observed=observed, predicted=predicted,
                            target=target.cpu().numpy(), chart=chart.cpu().numpy(), jvps=np.asarray(jvps))
        stream.flush()
        os.fsync(stream.fileno())
    host.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "arrays_sha256": sha(ARRAYS),
                            "metadata": metadata, "states": states, "model_hash": model_hash,
                            "cuda_allocation": allocation, "scope": "post-hoc one-root diagnostic"})
    print("Saved direct physical and model spectral probe results.", flush=True)


if __name__ == "__main__":
    main()
