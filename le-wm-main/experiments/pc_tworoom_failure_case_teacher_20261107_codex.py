"""Evaluation-only packet-aligned teacher forcing on sealed RGB chains."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_failure_case_temporal_20261107_codex as prior
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PREFIX = prior.b.q.ROOT / "results/pc_tworoom_failure_case_teacher_20261107_codex"
OUTPUT = Path(str(PREFIX) + "_result.json")
ARRAYS = Path(str(PREFIX) + "_arrays.npz")
PROTOCOL = prior.b.q.ROOT.parent / "docs/PC_TWOROOM_FAILURE_CASE_TEACHER_PROTOCOL_20261107_CODEX.md"


def packets(raw, scaler):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (25, 2) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1):
        raise ValueError("invalid physical action clock")
    encoded = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 5, 10).astype(np.float32)
    recovered = scaler.inverse_transform(encoded.reshape(-1, 2)).reshape(raw.shape)
    error = float(np.max(np.abs(recovered - raw)))
    if error > 1e-5:
        raise ValueError("action scaler round trip failed")
    return encoded, error


def teacher(model, observations, action):
    if observations.ndim != 3 or action.ndim != 3 or observations.shape[1] != 6 or action.shape[1:] != (5, 10):
        raise ValueError("observation/action packet mismatch")
    if observations.shape[0] != action.shape[0]:
        raise ValueError("candidate mismatch")
    predictions = []
    for t in range(5):
        start = max(0, t - 2)
        predictions.append(model.predict(observations[:, start:t + 1],
                                         model.action_encoder(action[:, start:t + 1]))[:, -1])
    return torch.stack(predictions, dim=1)


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("output exists")
    prior.verify()
    previous = json.loads(prior.RESULT.read_text())
    if previous["frame_sha256"] != file_hash(prior.FRAMES) or previous["features_sha256"] != file_hash(prior.FEATURES):
        raise ValueError("prior artifact drift")
    plan, model, scaler, transform = prior.setup()
    common = prior.b.q.old.base.source.block.prior.common
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    all_arrays, rows = {}, []
    print("Comparing recorded autoregressive rollout with evaluator-only true-history packet prediction.", flush=True)
    try:
        with np.load(prior.b.q.old.POOL, allow_pickle=False) as bank, \
                np.load(prior.b.q.POOL, allow_pickle=False) as old, \
                np.load(prior.PRED, allow_pickle=False) as pred, \
                np.load(prior.FEATURES, allow_pickle=False) as features, \
                np.load(prior.b.q.old.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inp, \
                h5py.File(common.DATA, "r") as data, torch.inference_mode():
            for root in prior.ROOTS:
                task = next(row for row in plan["rows"] if row["root"] == root)
                raw = prior.actions(root, bank, old, plan)
                normalized, roundtrip = packets(raw, scaler)
                action = torch.as_tensor(normalized, device="cuda")
                actual = torch.as_tensor(features[f"r{root}_actual"], device="cuda")
                predicted = torch.as_tensor(pred[f"r{root}_pred"], device="cuda")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task["row"]])
                actor._set_goal_state(data["proprio"][task["row"] + 25])
                rendered = actor.render().copy()
                original = inp["current"][root]
                initial = torch.stack([transform(original), transform(rendered)])[:, None].cuda()
                encoded = model.encode({"pixels": initial})["emb"][:, 0]
                initial_diff = float((encoded[0] - encoded[1]).square().mean())
                initial_vs_rollout = float((encoded[0] - predicted[0, 0]).abs().max())
                if initial_vs_rollout > 1e-5:
                    raise ValueError("planning image embedding mismatch")
                obs = torch.cat([encoded[0][None, None].expand(65, 1, -1), actual], dim=1)
                forced = teacher(model, obs, action)
                first_diff = float((forced[:, 0] - predicted[:, 1]).abs().max())
                if first_diff > 1e-5:
                    raise ValueError("first packet predictor mismatch")
                auto_mse = (predicted[:, 1:] - actual).square().mean(-1).cpu().numpy()
                teacher_mse = (forced - actual).square().mean(-1).cpu().numpy()
                case = next(row for row in previous["rows"] if row["root"] == root)["cases"]
                selected = {name: {"index": entry["index"], "autoregressive_mse": auto_mse[entry["index"]].tolist(),
                                   "teacher_mse": teacher_mse[entry["index"]].tolist()}
                            for name, entry in case.items()}
                rows.append({"root": root, "roundtrip_max_abs": roundtrip, "initial_render_feature_mse": initial_diff,
                             "initial_input_vs_rollout_max_abs": initial_vs_rollout, "first_packet_max_abs": first_diff,
                             "mean_auto_mse": auto_mse.mean(axis=0).tolist(),
                             "mean_teacher_mse": teacher_mse.mean(axis=0).tolist(),
                             "median_auto_mse": np.median(auto_mse, axis=0).tolist(),
                             "median_teacher_mse": np.median(teacher_mse, axis=0).tolist(), "cases": selected})
                all_arrays[f"r{root}_teacher"] = forced.cpu().numpy()
                all_arrays[f"r{root}_auto_mse"] = auto_mse
                all_arrays[f"r{root}_teacher_mse"] = teacher_mse
                all_arrays[f"r{root}_normalized_actions"] = normalized
                print(f"root={root} packet and first-prediction checks passed", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with ARRAYS.open("xb") as f:
        np.savez_compressed(f, **all_arrays)
    bindings = [Path(__file__), PROTOCOL, prior.RESULT, prior.PRED, prior.META, prior.FRAMES,
                prior.FEATURES, prior.b.q.PLAN, prior.b.q.POOL, prior.b.q.old.POOL,
                prior.b.q.old.base.source.block.prior.inputs.INPUT, common.DATA]
    with OUTPUT.open("x") as f:
        json.dump({"rows": rows, "sha256": {str(p): file_hash(p) for p in bindings},
                   "arrays_sha256": file_hash(ARRAYS), "model_sha256": plan["model_sha256"],
                   "scope": "posthoc selected cases; evaluation-only teacher forcing"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
