"""Matched model/physical RGB replay of sealed guarded spectral programs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability.reacher_d2p_context import advance_executed_context, restore_execution_snapshot

ARMS = ("selected", "zero", "same_bank_greedy")


def direct_rollout(model, embeddings, history, action_tokens):
    """Predict each token boundary using exact normalized executed actions."""
    if embeddings.ndim != 3 or history.ndim != 3 or action_tokens.ndim != 3:
        raise ValueError("rank-three contexts and actions required")
    if embeddings.shape[:2] != history.shape[:2] or embeddings.shape[0] != 1 or action_tokens.shape[0] != 1:
        raise ValueError("contexts must share a single batch and temporal extent")
    n = embeddings.shape[1]
    actions = torch.cat((history[:, 1:], action_tokens[:, :1]), dim=1)
    result = []
    for i in range(action_tokens.shape[1]):
        predicted = model.predict(embeddings[:, -n:], model.action_encoder(actions[:, -n:]))[:, -1]
        result.append(predicted)
        if i + 1 < action_tokens.shape[1]:
            embeddings = torch.cat((embeddings, predicted[:, None]), dim=1)
            actions = torch.cat((actions, action_tokens[:, i + 1:i + 2]), dim=1)
    return torch.stack(result, dim=1)


def validate_actions(method, mean, std):
    raw = np.asarray(method["action_witness_raw"], dtype=np.float32)
    normalized = np.asarray(method["action_witness_normalized"], dtype=np.float32)
    if raw.shape != (15, 2) or normalized.shape != (3, 10) or not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise ValueError("invalid action witness")
    decoded = setup.p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)
    if not np.allclose(decoded, raw, atol=1e-6, rtol=0):
        raise ValueError("saved raw and normalized actions differ")
    return raw, normalized


def prepare(setup_processor, context, device):
    return setup.p0._prepare(setup_processor, context["frames"], context["root_frame"], context["goal"],
                             np.asarray(context["history_action_blocks"], dtype=np.float32), device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", required=True, type=Path)
    parser.add_argument("--model-report", required=True, type=Path)
    parser.add_argument("--root-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    physical, generated, source = (json.loads(path.read_text()) for path in
                                   (args.physical, args.model_report, args.root_source))
    if (physical["input_report_sha256"] != setup._sha256_file(args.model_report)
            or generated["source_report_sha256"] != setup._sha256_file(args.root_source)):
        raise ValueError("input report binding mismatch")
    policy, stats = Path(source["protocol"]["policy"]), Path(source["protocol"]["action_stats"])
    if (generated["weights_sha256"] != setup._sha256_file(policy / "weights.pt")
            or generated["action_stats_sha256"] != setup._sha256_file(stats)):
        raise ValueError("model input hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != generated["parameter_hash_before"]:
        raise ValueError("model parameter mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65),
    )
    device = torch.device("cuda")
    rows = []
    try:
        if [x["seed"] for x in physical["rows"]] != [x["seed"] for x in generated["rows"]]:
            raise ValueError("root list mismatch")
        for physical_row, model_row in zip(physical["rows"], generated["rows"], strict=True):
            seed = physical_row["seed"]
            packets, _ = setup.sample_history_packets(histories, seed=seed)
            root = setup.collect_executed_context(env, seed=seed, history_len=3, action_block=5,
                        warmup_packets=packets, warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if root["root_rgb_sha256"] != model_row["root_rgb_sha256"] or root["goal_rgb_sha256"] != model_row["goal_rgb_sha256"]:
                raise RuntimeError("root binding failed")
            arms = {}
            for arm in ARMS:
                method = physical_row["methods"][arm]
                raw, normalized = validate_actions(method, mean, std)
                pixels, goal, history = prepare(processor, root, device)
                tokens = torch.as_tensor(normalized[None], device=device)
                with torch.no_grad():
                    goal_z = setup.p0._goal_latent(model, goal)
                    embeddings = setup.p0._latent_context(model, pixels, history)
                    open_predictions = direct_rollout(model, embeddings, history, tokens)[0]
                restored, restored_goal = restore_execution_snapshot(env, seed=seed, snapshot=root["execution_snapshot"])
                setup.p0._verify_execution_identity(restored, restored_goal, root["root_frame"], root["goal"])
                context = root
                checkpoints = []
                for i in range(3):
                    pix, _, hist = prepare(processor, context, device)
                    with torch.no_grad():
                        latent = setup.p0._latent_context(model, pix, hist)
                        refreshed = direct_rollout(model, latent, hist, tokens[:, i:i + 1])[0, 0]
                    context = advance_executed_context(env, context=context, raw_action_block=raw[i * 5:(i + 1) * 5],
                                action_block=5, verified_root_frame=context["root_frame"])
                    pix_actual, _, hist_actual = prepare(processor, context, device)
                    with torch.no_grad():
                        observed = setup.p0._latent_context(model, pix_actual, hist_actual)[0, -1]
                    qpos = np.asarray(env.unwrapped.env.physics.data.qpos)
                    target = np.asarray(env.unwrapped.env.task.target_qpos)
                    qdist = setup.p0._qpos_distance(qpos, target)
                    if i == 0:
                        if context["root_rgb_sha256"] != method["observed_successor_context"]["root_rgb_sha256"]:
                            raise RuntimeError("first actual successor RGB mismatch")
                        expected = method["physical"]["after_first_chunk_qpos_distance"]
                        if not np.isclose(qdist, expected, rtol=0, atol=1e-8):
                            raise RuntimeError("first physical successor mismatch")
                    if i == 2 and not np.isclose(qdist, method["physical"]["final_qpos_distance"], rtol=0, atol=1e-8):
                        raise RuntimeError("final physical successor mismatch")
                    def values(x):
                        return {"latent": x.cpu().tolist(), "goal_mse": float((x - goal_z).square().mean())}
                    checkpoints.append({"physical_step": 5 * (i + 1), "rgb_sha256": context["root_rgb_sha256"],
                                        "qpos_distance": qdist, "observed": values(observed),
                                        "open_prediction": values(open_predictions[i]),
                                        "refreshed_prediction": values(refreshed),
                                        "open_mse_to_observed": float((open_predictions[i] - observed).square().mean()),
                                        "refreshed_mse_to_observed": float((refreshed - observed).square().mean())})
                arms[arm] = checkpoints
                print(json.dumps({"seed": seed, "arm": arm, "checkpoints": len(checkpoints)}), flush=True)
            rows.append({"seed": seed, "arms": arms})
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("model parameters changed")
    report = {"scope": "known-root same-action model/RGB/physical replay; not an independent test or gate",
              "input_physical_sha256": setup._sha256_file(args.physical),
              "input_model_sha256": setup._sha256_file(args.model_report),
              "root_source_sha256": setup._sha256_file(args.root_source),
              "script_sha256": setup._sha256_file(Path(__file__)),
              "model_parameter_hash_before": before, "model_parameter_hash_after": after,
              "model_cost_scope": "3 open and 3 refreshed prediction samples per arm; encoding and wall time excluded",
              "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
