"""Diagnostic comparison of a goal-image chart on predicted and actual successor images."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_dual_rgb_goal_tangent_20261107_codex as source

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_GOAL_CHART_ACTUAL_SUCCESSOR_PROTOCOL_20261107_CODEX.md"
PHYSICAL = ROOT / "results/pc_tworoom_goal_chart_actual_successor_physical_20261107_codex.json"
FRAMES = ROOT / "results/pc_tworoom_goal_chart_actual_successor_frames_20261107_codex.npz"
RESULT = ROOT / "results/pc_tworoom_goal_chart_actual_successor_result_20261107_codex.json"
FEATURES = ROOT / "results/pc_tworoom_goal_chart_actual_successor_features_20261107_codex.npz"


def make_chart(probes):
    probes = np.asarray(probes, dtype=np.float64)
    if probes.ndim != 2 or probes.shape[0] != 4 or not np.isfinite(probes).all():
        raise ValueError("four finite goal perturbation features required")
    jacobian = np.stack(((probes[0] - probes[1]) / 8, (probes[2] - probes[3]) / 8), axis=1)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-6:
        return None
    return np.linalg.pinv(jacobian, rcond=1e-6)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists() or FRAMES.exists():
        raise FileExistsError("physical output exists")
    plan, previous = json.loads(source.PLAN.read_text()), json.loads(source.PHYSICAL.read_text())
    if (previous["plan_sha256"] != file_hash(source.PLAN)
            or plan["features_sha256"] != file_hash(source.FEATURES)
            or plan["runner_sha256"] != file_hash(source.__file__)):
        raise ValueError("source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, frames = [], {}
    try:
        with h5py.File(source.old.SOURCE.common.DATA, "r") as data:
            for root, (task, prior) in enumerate(zip(plan["rows"], previous["rows"], strict=True)):
                if task["row"] != prior["row"]:
                    raise ValueError("task identity changed")
                outputs = []
                for arm in source.ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task["row"]])
                    actor._set_goal_state(data["proprio"][task["row"] + 25])
                    actions = np.asarray(task["outcomes"][arm]["actions"], dtype=np.float32)
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    if trace != prior["outcomes"][arm]["trace"]:
                        raise ValueError("physical replay differs")
                    frames[f"root_{root}_{arm}"] = actor.render().copy()
                    delta = (actor.agent_position - actor.target_position).detach().cpu().numpy()
                    outputs.append({"arm": arm, "candidate_index": task["outcomes"][arm]["candidate_index"],
                                    "true_displacement": delta.tolist(), "final_distance": trace[-1]["distance"],
                                    "ever_success": any(x["terminated"] for x in trace)})
                rows.append({"row": task["row"], "root": root, "basis_valid": task["basis_valid"], "outputs": outputs})
    finally:
        env.close()
    with FRAMES.open("xb") as stream:
        np.savez_compressed(stream, **frames)
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(source.PLAN), "prior_physical_sha256": file_hash(source.PHYSICAL),
                   "frames_sha256": file_hash(FRAMES), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__))}, stream, indent=2, allow_nan=False)
    print(f"replayed {len(rows)*3} chains and matched saved traces")


def analyze():
    if RESULT.exists() or FEATURES.exists():
        raise FileExistsError("analysis output exists")
    physical_data, plan = json.loads(PHYSICAL.read_text()), json.loads(source.PLAN.read_text())
    if (physical_data["plan_sha256"] != file_hash(source.PLAN)
            or physical_data["frames_sha256"] != file_hash(FRAMES)
            or physical_data["protocol_sha256"] != file_hash(PROTOCOL)
            or physical_data["runner_sha256"] != file_hash(__file__)):
        raise ValueError("physical artifact drift")
    model, _, transform, action_hash = source.old.SOURCE.common.model_setup()
    if (model_fingerprint(model) != plan["model_sha256"]
            or action_hash != json.loads(source.old.previous.OUTPUT.read_text())["normalizer_action_sha256"]):
        raise ValueError("model or normalizer changed")
    rows, features = [], {}
    with np.load(FRAMES, allow_pickle=False) as frames, \
            np.load(source.FEATURES, allow_pickle=False) as chart_features, \
            np.load(source.old.previous.FEATURES, allow_pickle=False) as forecast_features, torch.inference_mode():
        for row in physical_data["rows"]:
            root = row["root"]
            if not row["basis_valid"]:
                rows.append({"row": row["row"], "root": root, "valid": False, "reason": "image_or_rank_abstention"})
                continue
            chart = make_chart(chart_features[f"root_{root}_probe_embedding"])
            if chart is None:
                raise ValueError("chart unexpectedly rank deficient")
            images = torch.stack([transform(frames[f"root_{root}_{arm}"]) for arm in source.ARMS])
            real = model.encode({"pixels": images[:, None].cuda()})["emb"][:, 0].cpu().numpy()
            goal = forecast_features[f"root_{root}_goal"].astype(np.float64)
            predicted = forecast_features[f"root_{root}_predicted"]
            outcomes = []
            for i, output in enumerate(row["outputs"]):
                actual = np.asarray(output["true_displacement"])
                latent = predicted[output["candidate_index"]].astype(np.float64)
                estimate_true = chart @ (real[i].astype(np.float64) - goal)
                estimate_pred = chart @ (latent - goal)
                outcomes.append({**output, "chart_true_encoded": estimate_true.tolist(),
                                 "chart_model_prediction": estimate_pred.tolist(),
                                 "zero_error": float(np.linalg.norm(actual)),
                                 "true_encoded_error": float(np.linalg.norm(estimate_true - actual)),
                                 "predicted_error": float(np.linalg.norm(estimate_pred - actual))})
            rows.append({"row": row["row"], "root": root, "valid": True, "outputs": outcomes})
            features[f"root_{root}_actual"] = real
            features[f"root_{root}_chart"] = chart
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **features)
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "physical_sha256": file_hash(PHYSICAL), "features_sha256": file_hash(FEATURES),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "post-hoc diagnostics only; simulator truth excluded from planner"},
                  stream, indent=2, allow_nan=False)
    good = [r for r in rows if r["valid"]]
    a = np.asarray([[x["true_encoded_error"], x["zero_error"], x["predicted_error"]]
                    for r in good for x in r["outputs"]])
    print(f"valid roots={len(good)} errors mean true/zero/predicted={a.mean(axis=0).tolist()}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("physical", "analyze"):
        raise SystemExit("expected physical|analyze")
    {"physical": physical, "analyze": analyze}[sys.argv[1]]()
