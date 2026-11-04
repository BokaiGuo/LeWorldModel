"""Replay recorded v7 Reacher finite differences from the actual third boundary."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v7_reacher_rgb_live_20261104_codex as configured
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = configured.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_REACHER_ACTUAL_FD_PHYSICAL_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_reacher_actual_fd_physical_dev_20261104_codex"
MANIFEST, RESULT, ARRAYS = [PREFIX.with_name(PREFIX.name + s) for s in ("_manifest.json", "_result.json", "_arrays.npz")]


def physical(env):
    inner = env.unwrapped.env
    qpos = np.asarray(inner.physics.data.qpos).copy()
    qvel = np.asarray(inner.physics.data.qvel).copy()
    goal = np.asarray(inner.task.target_qpos).copy()
    return {"qpos": qpos.tolist(), "qvel": qvel.tolist(), "target_qpos": goal.tolist()}


def main():
    if any(path.exists() for path in (MANIFEST, RESULT, ARRAYS)):
        raise FileExistsError("existing or partial output")
    configured.configure()
    live = configured.live
    host, setup, p0 = live.host, live.setup, live.p0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = host.bridge.sha_file
    audit = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
    prediction = configured.PREFIX.with_name(configured.PREFIX.name + "_prediction_replay.json")
    execution_path = live.path("response", "execution.json")
    decision_path = live.path("response", "decision2.json")
    ctx0, ctx2 = (live.path("response", f"context{i}.npz") for i in (0, 2))
    if (json.loads(audit.read_text())["manifest_sha256"] != sha(live.MANIFEST)
            or json.loads(prediction.read_text())["audit_sha256"] != sha(audit)):
        raise ValueError("live record or prediction audit mismatch")
    decision, execution = json.loads(decision_path.read_text()), json.loads(execution_path.read_text())
    batches = []
    for info in decision["result"]["first_response"]:
        nominal = np.asarray(info["nominal"], dtype=float)
        axes = np.eye(10).reshape(10, 5, 2)
        expected = np.concatenate((nominal[None], nominal[None]+.05*axes, nominal[None]-.05*axes))
        matches = [entry for entry in decision["trace"] if np.array_equal(np.asarray(entry["requested"]), expected)]
        if len(matches) != 1:
            raise ValueError("expected one actual nominal probe batch")
        batch = matches[0]
        positions = np.asarray(batch["positions"])
        np.testing.assert_allclose(((positions[1:11]-positions[11:21])/.1).T, info["matrix"], atol=1e-12, rtol=0)
        batches.append(batch)
    if len(batches) != 2:
        raise ValueError("unexpected nominal count")
    actions = np.asarray([b["raw"] for b in batches], dtype=np.float32)
    source = json.loads(live.run.SOURCE.read_text())
    dependencies = seal.capture([Path(__file__), PROTOCOL, live.MANIFEST, audit, prediction, execution_path,
                                 decision_path, ctx0, ctx2, *map(Path, json.loads(live.MANIFEST.read_text())["dependencies"])])
    host.write_new(MANIFEST, {"seal": dependencies, "scope": "42 actual probes at one exposed RGB root"})
    with np.load(ctx0, allow_pickle=False) as data:
        warmup = data["history"].reshape(-1, 2).astype(np.float32)
    with np.load(ctx2, allow_pickle=False) as data:
        context = {key: data[key].copy() for key in data.files}
    print("Replaying 42 recorded v7 probe packets from the same physical boundary.", flush=True)
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
                        raise ValueError("unexpected physical termination")

            first_state = None
            for packet in actions.reshape(-1, 5, 2):
                _, info = env.reset(seed=20263160)
                if live.context_lib.rgb_digest(info["goal"]) != live.context_lib.rgb_digest(context["goal"]):
                    raise ValueError("goal mismatch")
                execute(warmup)
                for row in execution["rows"][:2]:
                    execute(row["dispatch"])
                frame = np.asarray(env.render(), np.uint8)
                if not np.array_equal(frame, context["root"]):
                    raise ValueError("root RGB mismatch")
                before = physical(env)
                if first_state is None:
                    first_state = before
                for key in ("qpos", "qvel", "target_qpos"):
                    np.testing.assert_allclose(before[key], first_state[key], rtol=0, atol=1e-10)
                execute(packet)
                frames.append(np.asarray(env.render(), np.uint8).copy())
                states.append(physical(env))
        finally:
            env.close()
        stats = Path(source["action_stats_path"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        model = setup.load_compatible_pretrained(Path(source["weights_path"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        model_hash = setup._parameter_hash(model)
        if model_hash != source["model_parameter_hash_expected"]:
            raise ValueError("checkpoint mismatch")
        device = torch.device("cuda:0")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        _, goal, _ = p0._prepare(processor, context["frames"], context["root"], context["goal"],
                                  context["history"], device)
        with torch.no_grad():
            target = p0._goal_latent(model, goal).cpu().numpy()
            observed = []
            for frame in frames:
                _, image, _ = p0._prepare(processor, context["frames"], context["root"],
                                           np.ascontiguousarray(frame), context["history"], device)
                observed.append(p0._goal_latent(model, image).cpu().numpy())
            torch.cuda.synchronize()
        observed = np.stack(observed).reshape(2, 21, -1)
        if setup._parameter_hash(model) != model_hash:
            raise ValueError("model parameters changed")
    seal.verify(dependencies)
    chart = np.asarray(decision["chart"])
    projected = observed @ chart
    model_positions = np.asarray([b["positions"] for b in batches])
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, frames=np.asarray(frames).reshape(2, 21, *frames[0].shape),
                            actions=actions, observed=observed, chart=chart, target=target,
                            model_positions=model_positions)
        stream.flush()
        os.fsync(stream.fileno())
    rows = []
    for i in range(2):
        model_matrix = ((model_positions[i, 1:11]-model_positions[i, 11:21])/.1).T
        actual_matrix = ((projected[i, 1:11]-projected[i, 11:21])/.1).T
        delta = model_matrix-actual_matrix
        norm_model, norm_actual = np.linalg.norm(model_matrix), np.linalg.norm(actual_matrix)
        model_scores = np.asarray(batches[i]["scores"])
        physical_scores = np.mean((observed[i]-target)**2, axis=1)
        upper = np.triu_indices(21, k=1)
        m_delta = model_scores[:, None]-model_scores[None, :]
        p_delta = physical_scores[:, None]-physical_scores[None, :]
        non_ties = (m_delta[upper] != 0) & (p_delta[upper] != 0)
        concordance = float(np.mean(np.sign(m_delta[upper][non_ties]) == np.sign(p_delta[upper][non_ties]))) if np.any(non_ties) else None
        rows.append({"nominal": i, "model_matrix": model_matrix.tolist(), "observed_matrix": actual_matrix.tolist(),
                     "matrix_abs_frobenius_error": float(np.linalg.norm(delta)),
                     "model_matrix_norm": float(norm_model), "observed_matrix_norm": float(norm_actual),
                     "relative_to_observed": float(np.linalg.norm(delta)/norm_actual) if norm_actual > 1e-12 else None,
                     "matrix_cosine": float(np.sum(model_matrix*actual_matrix)/(norm_model*norm_actual))
                         if min(norm_model, norm_actual) > 1e-12 else None,
                     "model_goal_mse": model_scores.tolist(), "observed_goal_mse": physical_scores.tolist(),
                     "pairwise_order_agreement_non_ties": concordance, "pair_count": int(non_ties.sum()),
                     "states": states[i*21:(i+1)*21]})
    host.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "arrays_sha256": sha(ARRAYS),
                            "model_hash": model_hash, "cuda_allocation": allocation,
                            "common_before": first_state, "rows": rows,
                            "scope": "same backend, one exposed root and one 5-step scale"})
    print("Saved physical probe images, states, task-chart responses and ordering.", flush=True)


if __name__ == "__main__":
    main()
