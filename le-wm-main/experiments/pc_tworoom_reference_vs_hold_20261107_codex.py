"""Sealed RGB prediction followed by evaluation-only physical replay."""
import json
import sys

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_standard_tworoom_reference_replay_20261105_codex import select_rows
from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as common

ROOT = common.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_REFERENCE_VS_HOLD_PROTOCOL_20261107_CODEX.md"
REFERENCE = ROOT / "results/pc_standard_tworoom_reference_replay_20261105_codex/report.json"
INPUT = ROOT / "results/pc_tworoom_reference_vs_hold_input_20261107_codex.npz"
PRED = ROOT / "results/pc_tworoom_reference_vs_hold_prediction_20261107_codex.json"
PRED_FEATURES = ROOT / "results/pc_tworoom_reference_vs_hold_prediction_features_20261107_codex.npz"
PHYSICAL = ROOT / "results/pc_tworoom_reference_vs_hold_physical_20261107_codex.json"
FRAMES = ROOT / "results/pc_tworoom_reference_vs_hold_frames_20261107_codex.npz"
RESULT = ROOT / "results/pc_tworoom_reference_vs_hold_result_20261107_codex.json"
FEATURES = ROOT / "results/pc_tworoom_reference_vs_hold_features_20261107_codex.npz"


def prepare():
    if INPUT.exists():
        raise FileExistsError(INPUT)
    ref = json.loads(REFERENCE.read_text())
    with h5py.File(common.DATA, "r") as data:
        rows = select_rows(data)
        if rows.tolist() != [x["row"] for x in ref["records"]]:
            raise ValueError("reference panel mismatch")
        current, goal, actions = [], [], []
        for row, entry in zip(rows, ref["records"], strict=True):
            row = int(row)
            if not entry["completed"]:
                raise ValueError("reference not complete")
            current.append(data["pixels"][row])
            goal.append(data["pixels"][row + 25])
            actions.append(data["action"][row:row + 25])
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (50, 25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
            raise ValueError("reference actions must be legal")
    with INPUT.open("xb") as stream:
        np.savez_compressed(stream, rows=rows, current=np.asarray(current), goal=np.asarray(goal), actions=actions)
    print("sealed 50 RGB pairs and legal reference actions")


def score():
    if PRED.exists() or PRED_FEATURES.exists():
        raise FileExistsError("prediction output already exists")
    model, scaler, transform, action_hash = common.model_setup()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, INPUT, common.CHECKPOINT / "weights.pt",
                                      common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="5 packets, 5 steps each")
    output, features = [], []
    with np.load(INPUT, allow_pickle=False) as data, torch.inference_mode():
        for i, row in enumerate(data["rows"]):
            actions = np.stack([data["actions"][i], np.zeros((25, 2), np.float32)])
            normalized = scaler.transform(actions.reshape(-1, 2)).reshape(1, 2, 5, 10)
            np.testing.assert_allclose(scaler.inverse_transform(normalized.reshape(-1, 2)),
                                       actions.reshape(-1, 2), atol=1e-6, rtol=0)
            pixels = transform(data["current"][i])[None, None, None].cuda()
            goal = transform(data["goal"][i])[None, None, None].cuda()
            wrapper = RGBLeWM(model, audit)
            cost, context = wrapper.score({"pixels": pixels, "goal": goal,
                                           "action": torch.empty((1, 1, 0, 10), device="cuda")},
                                          torch.tensor(normalized, dtype=torch.float32, device="cuda"))
            output.append({"row": int(row), "cost": cost[0].cpu().tolist(), "receipt": wrapper.receipts[0]})
            features.append(context["predicted_emb"][0, :, -1].cpu().numpy())
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with PRED_FEATURES.open("xb") as stream:
        np.savez_compressed(stream, predicted=np.asarray(features))
    with PRED.open("x") as stream:
        json.dump({"rows": output, "model_sha256": fingerprint, "normalizer_action_sha256": action_hash,
                   "input_sha256": file_hash(INPUT), "features_sha256": file_hash(PRED_FEATURES),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__)},
                  stream, allow_nan=False)
    print("sealed 100 model predictions")


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists() or FRAMES.exists():
        raise FileExistsError("physical output already exists")
    pred = json.loads(PRED.read_text())
    if (pred["input_sha256"] != file_hash(INPUT) or pred["runner_sha256"] != file_hash(__file__)
            or pred["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("sealed input drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    output, frames = [], []
    try:
        with np.load(INPUT, allow_pickle=False) as inputs, h5py.File(common.DATA, "r") as data:
            for i, row in enumerate(inputs["rows"]):
                row = int(row)
                pair = []
                for actions in (inputs["actions"][i], np.zeros((25, 2), np.float32)):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["proprio"][row]))
                    actor._set_goal_state(np.asarray(data["proprio"][row + 25]))
                    trace = []
                    for action in actions:
                        _, _, terminated, truncated, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(terminated), "truncated": bool(truncated)})
                    pair.append({"trace": trace})
                    frames.append(actor.render().copy())
                output.append({"row": row, "candidates": pair})
    finally:
        env.close()
    with FRAMES.open("xb") as stream:
        np.savez_compressed(stream, frames=np.asarray(frames).reshape(50, 2, 224, 224, 3))
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": output, "prediction_sha256": file_hash(PRED), "frames_sha256": file_hash(FRAMES),
                   "runner_sha256": file_hash(__file__)}, stream, allow_nan=False)
    print("replayed 100 physical chains, 2500 steps")


def analyze():
    if RESULT.exists() or FEATURES.exists():
        raise FileExistsError("analysis output already exists")
    pred, phys = json.loads(PRED.read_text()), json.loads(PHYSICAL.read_text())
    if (phys["prediction_sha256"] != file_hash(PRED) or phys["frames_sha256"] != file_hash(FRAMES)
            or phys["runner_sha256"] != file_hash(__file__) or pred["features_sha256"] != file_hash(PRED_FEATURES)):
        raise ValueError("sealed artifact drift")
    model, _, transform, action_hash = common.model_setup()
    if model_fingerprint(model) != pred["model_sha256"] or action_hash != pred["normalizer_action_sha256"]:
        raise ValueError("model or normalizer drift")
    rows, all_true, all_goal = [], [], []
    with np.load(INPUT, allow_pickle=False) as inputs, np.load(FRAMES, allow_pickle=False) as frames, \
            np.load(PRED_FEATURES, allow_pickle=False) as forecast, torch.inference_mode():
        for i, (p, q) in enumerate(zip(pred["rows"], phys["rows"], strict=True)):
            if p["row"] != q["row"] or p["row"] != int(inputs["rows"][i]):
                raise ValueError("candidate identity mismatch")
            goal = model.encode({"pixels": transform(inputs["goal"][i])[None, None].cuda()})["emb"][0, 0]
            batch = torch.stack([transform(x) for x in frames["frames"][i]])
            true = model.encode({"pixels": batch[:, None].cuda()})["emb"][:, 0]
            cost = (true - goal).square().sum(-1).cpu().numpy()
            forecast_array = forecast["predicted"][i]
            traces = [candidate["trace"] for candidate in q["candidates"]]
            rows.append({"row": p["row"], "prediction": p["cost"], "encoded": cost.tolist(),
                         "distance": [t[-1]["distance"] for t in traces],
                         "ever_success": [any(s["terminated"] for s in t) for t in traces],
                         "final_success": [t[-1]["terminated"] for t in traces],
                         "feature_mse": float(np.square(forecast_array - true.cpu().numpy()).mean())})
            all_true.append(true.cpu().numpy())
            all_goal.append(goal.cpu().numpy())
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, actual=np.asarray(all_true), goal=np.asarray(all_goal))
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "prediction_sha256": file_hash(PRED), "physical_sha256": file_hash(PHYSICAL),
                   "features_sha256": file_hash(FEATURES), "runner_sha256": file_hash(__file__),
                   "scope": "exposed RGB development panel; reference actions evaluation only"},
                  stream, indent=2, allow_nan=False)
    print("encoded physical endpoints and saved 50 paired comparisons")


if __name__ == "__main__":
    stages = {"prepare": prepare, "score": score, "physical": physical, "analyze": analyze}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        raise SystemExit("expected prepare|score|physical|analyze")
    stages[sys.argv[1]]()
