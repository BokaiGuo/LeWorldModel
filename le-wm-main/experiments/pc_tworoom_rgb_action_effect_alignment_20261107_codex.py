"""Fixed action-effect matrix on frozen TwoRoom RGB model and physical replay."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RGB_ACTION_EFFECT_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
INPUT = ROOT / "results/pc_tworoom_rgb_action_effect_input_20261107_codex.npz"
PRED = ROOT / "results/pc_tworoom_rgb_action_effect_prediction_20261107_codex.json"
PRED_FEATURES = ROOT / "results/pc_tworoom_rgb_action_effect_prediction_features_20261107_codex.npz"
FRAMES = ROOT / "results/pc_tworoom_rgb_action_effect_real_frames_20261107_codex.npz"
PHYSICAL = ROOT / "results/pc_tworoom_rgb_action_effect_physical_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_rgb_action_effect_alignment_20261107_codex.json"
FEATURES = ROOT / "results/pc_tworoom_rgb_action_effect_features_20261107_codex.npz"
DATA = Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5")
CHECKPOINT = Path("/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")
SEEDS = (2717100, 2717101, 2717102, 2717103)
ACTIONS = np.asarray([[0, 0], [.6, 0], [-.6, 0], [0, .6], [0, -.6]], dtype=np.float32)


def model_setup():
    import h5py
    from sklearn.preprocessing import StandardScaler
    from eval import img_transform
    from experiments.checkpoint_compat import load_compatible_pretrained
    with h5py.File(DATA) as file:
        actions = np.asarray(file["action"][:])
    actions = actions[~np.isnan(actions).any(axis=1)]
    scaler = StandardScaler().fit(actions)
    model = load_compatible_pretrained("tworoom/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    action_hash = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
    return model, scaler, img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224))), action_hash


def image_goal(actor):
    return actor._target_img.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8, copy=True)


def prepare():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if INPUT.exists():
        raise FileExistsError(INPUT)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    images = {}
    try:
        for seed in SEEDS:
            env.reset(seed=seed)
            images[f"s{seed}_current"] = env.unwrapped.render().copy()
            images[f"s{seed}_goal"] = image_goal(env.unwrapped)
    finally:
        env.close()
    with INPUT.open("xb") as stream:
        np.savez_compressed(stream, **images)
    print(f"sealed {len(SEEDS)} RGB roots", flush=True)


def score():
    if PRED.exists() or PRED_FEATURES.exists():
        raise FileExistsError("prediction output exists")
    model, scaler, transform, action_hash = model_setup()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, INPUT,
                         CHECKPOINT / "weights.pt", CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom dataset action scaler",
                         clock="two 5-step action packets")
    chains = np.stack([np.stack([np.tile(a, (5, 1)), np.tile(b, (5, 1))])
                       for a in ACTIONS for b in ACTIONS])
    normalized = scaler.transform(chains.reshape(-1, 2)).reshape(1, 25, 2, 10)
    if np.max(np.abs(scaler.inverse_transform(normalized.reshape(-1, 2)) - chains.reshape(-1, 2))) > 1e-6:
        raise ValueError("normalizer roundtrip")
    candidates = torch.tensor(normalized, dtype=torch.float32, device="cuda")
    rows, features = [], {}
    with np.load(INPUT, allow_pickle=False) as images, torch.inference_mode():
        for seed in SEEDS:
            pixels = transform(images[f"s{seed}_current"])[None, None, None].cuda()
            goal = transform(images[f"s{seed}_goal"])[None, None, None].cuda()
            wrapper = RGBLeWM(model, audit)
            values, context = wrapper.score({"pixels": pixels, "goal": goal,
                                              "action": torch.empty((1, 1, 0, 10), device="cuda")}, candidates)
            features[f"s{seed}_predicted"] = context["predicted_emb"][0, :, -1].cpu().numpy()
            rows.append({"seed": seed, "cost": values[0].cpu().reshape(5, 5).tolist(),
                         "receipt": wrapper.receipts[0]})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weight drift")
    with PRED_FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **features)
    with PRED.open("x") as stream:
        json.dump({"rows": rows, "input_sha256": file_hash(INPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "model_sha256": fingerprint,
                   "features_sha256": file_hash(PRED_FEATURES), "normalizer_action_sha256": action_hash}, stream, allow_nan=False)
    print("sealed 100 model predictions", flush=True)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists() or FRAMES.exists():
        raise FileExistsError("physical output exists")
    predicted = json.loads(PRED.read_text())
    if (predicted["input_sha256"] != file_hash(INPUT) or predicted["runner_sha256"] != file_hash(__file__)
            or predicted["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("prediction source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    frames, rows = {}, []
    try:
        with np.load(INPUT, allow_pickle=False) as images:
            for seed in SEEDS:
                for i, a in enumerate(ACTIONS):
                    for j, b in enumerate(ACTIONS):
                        env.reset(seed=seed)
                        np.testing.assert_array_equal(env.unwrapped.render(), images[f"s{seed}_current"])
                        np.testing.assert_array_equal(image_goal(env.unwrapped), images[f"s{seed}_goal"])
                        trace = []
                        for action in [a] * 5 + [b] * 5:
                            _, _, terminated, truncated, info = env.unwrapped.step(action.copy())
                            trace.append({"distance": float(info["distance_to_target"]),
                                          "terminated": bool(terminated), "truncated": bool(truncated)})
                        frames[f"s{seed}_{i}_{j}"] = env.unwrapped.render().copy()
                        rows.append({"seed": seed, "prefix": i, "tail": j, "trace": trace})
                print(f"seed {seed}: 25 physical chains", flush=True)
    finally:
        env.close()
    with FRAMES.open("xb") as stream:
        np.savez_compressed(stream, **frames)
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "frames_sha256": file_hash(FRAMES), "prediction_sha256": file_hash(PRED),
                   "runner_sha256": file_hash(__file__)}, stream, allow_nan=False)


def pair_agreement(first, second):
    x, y = np.asarray(first).reshape(-1), np.asarray(second).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("effect arrays must match")
    sx, sy = np.sign(np.where(np.abs(x) <= 1e-6, 0, x)), np.sign(np.where(np.abs(y) <= 1e-6, 0, y))
    return {"n": len(x), "matching_sign": int((sx == sy).sum()), "ties_either": int(((sx == 0) | (sy == 0)).sum())}


def ranking_agreement(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != (5, 5) or second.shape != (5, 5):
        raise ValueError("5x5 action matrices required")
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    return pair_agreement(np.stack([first[:, i] - first[:, j] for i, j in pairs], axis=1),
                          np.stack([second[:, i] - second[:, j] for i, j in pairs], axis=1))


def analyze():
    if RESULT.exists() or FEATURES.exists():
        raise FileExistsError("analysis output exists")
    physical_data, predicted = json.loads(PHYSICAL.read_text()), json.loads(PRED.read_text())
    if (physical_data["frames_sha256"] != file_hash(FRAMES) or physical_data["prediction_sha256"] != file_hash(PRED)
            or predicted["features_sha256"] != file_hash(PRED_FEATURES)
            or physical_data["runner_sha256"] != file_hash(__file__)):
        raise ValueError("physical source drift")
    model, _, transform, action_hash = model_setup()
    if model_fingerprint(model) != predicted["model_sha256"] or action_hash != predicted["normalizer_action_sha256"]:
        raise ValueError("model or normalizer mismatch")
    rows, features = [], {}
    with np.load(INPUT, allow_pickle=False) as inputs, np.load(FRAMES, allow_pickle=False) as frames, \
            np.load(PRED_FEATURES, allow_pickle=False) as predictions, torch.inference_mode():
        for row in predicted["rows"]:
            seed = row["seed"]
            goal = model.encode({"pixels": transform(inputs[f"s{seed}_goal"])[None, None].cuda()})["emb"][0, 0]
            batch = torch.stack([transform(frames[f"s{seed}_{i}_{j}"]) for i in range(5) for j in range(5)])
            true = model.encode({"pixels": batch[:, None].cuda()})["emb"][:, 0]
            encoded = (true - goal).square().sum(-1).cpu().numpy().reshape(5, 5)
            pred = np.asarray(row["cost"])
            distance = np.asarray([next(r["trace"][-1]["distance"] for r in physical_data["rows"]
                                        if (r["seed"], r["prefix"], r["tail"]) == (seed, i, j))
                                   for i in range(5) for j in range(5)]).reshape(5, 5)
            def effects(x):
                return x[:, :1] - x[:, 1:]
            features[f"s{seed}_actual"] = true.cpu().numpy()
            features[f"s{seed}_goal"] = goal.cpu().numpy()
            rows.append({"seed": seed, "prediction": pred.tolist(), "encoded": encoded.tolist(),
                         "distance": distance.tolist(),
                         "predicted_vs_encoded": pair_agreement(effects(pred), effects(encoded)),
                         "predicted_vs_distance": pair_agreement(effects(pred), effects(distance)),
                         "encoded_vs_distance": pair_agreement(effects(encoded), effects(distance)),
                         "predicted_vs_encoded_ranking": ranking_agreement(pred, encoded),
                         "predicted_vs_distance_ranking": ranking_agreement(pred, distance),
                         "encoded_vs_distance_ranking": ranking_agreement(encoded, distance),
                         "mean_feature_mse": float(np.square(predictions[f"s{seed}_predicted"] -
                                                             true.cpu().numpy()).mean()),
                         "mean_abs_cost_error": float(np.abs(pred - encoded).mean())})
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **features)
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "features_sha256": file_hash(FEATURES), "physical_sha256": file_hash(PHYSICAL),
                   "prediction_sha256": file_hash(PRED), "runner_sha256": file_hash(__file__),
                   "scope": "four exposed RGB development roots; no mechanism or independent gate"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "score", "physical", "analyze"):
        raise SystemExit("expected prepare|score|physical|analyze")
    {"prepare": prepare, "score": score, "physical": physical, "analyze": analyze}[sys.argv[1]]()
