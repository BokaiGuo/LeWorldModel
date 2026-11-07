"""Goal-image-only local encoder tangent, tested on an already sealed candidate pool."""
import json
import sys
import time

import cv2
import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_action_sensitive_metric_20261107_codex as previous
from experiments.pc_tworoom_failed_root_full_pool_20261107_codex import candidates_for

SOURCE = previous.source
PROTOCOL = SOURCE.ROOT.parent / "docs/PC_TWOROOM_GOAL_TANGENT_METRIC_DEV_PROTOCOL_20261107_CODEX.md"
PLAN = SOURCE.ROOT / "results/pc_tworoom_goal_tangent_metric_plan_20261107_codex.json"
FEATURES = SOURCE.ROOT / "results/pc_tworoom_goal_tangent_metric_features_20261107_codex.npz"
PHYSICAL = SOURCE.ROOT / "results/pc_tworoom_goal_tangent_metric_physical_20261107_codex.json"
ARMS = ("original", "goal_tangent", "random_tangent")


def make_goal_probes(image):
    image = np.asarray(image)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("expected uint8 224 RGB")
    signed = image.astype(np.int16)
    red = np.maximum(signed[..., 0] - np.maximum(signed[..., 1], signed[..., 2]), 0)
    mask = red > 3
    ys, xs = np.nonzero(mask)
    info = {"red_pixels": int(mask.sum()), "valid": False, "reason": None}
    if not 100 <= mask.sum() <= 5000:
        info["reason"] = "red_support"
        return None, info
    info["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    if xs.min() < 4 or ys.min() < 4 or xs.max() >= 220 or ys.max() >= 220:
        info["reason"] = "touches_shift_boundary"
        return None, info
    alpha = (red.astype(np.float32) / 255.) * mask
    background = cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    def composite(layer):
        return np.clip(np.rint(background.astype(np.float32) * (1 - layer[..., None])
                               + np.asarray([255., 0., 0.]) * layer[..., None]), 0, 255).astype(np.uint8)

    original = composite(alpha)
    info["reconstruction_mae"] = float(np.abs(original.astype(np.int16) - image.astype(np.int16))[mask].mean())
    if info["reconstruction_mae"] > 4:
        info["reason"] = "inpainting_reconstruction"
        return None, info
    probes = []
    for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
        shifted = cv2.warpAffine(alpha, np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
                                 (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        probes.append(composite(shifted))
    info["valid"] = True
    return np.stack(probes), info


def tangent_basis(probes):
    probes = np.asarray(probes, dtype=np.float64)
    if probes.ndim != 2 or probes.shape[0] != 4 or not np.isfinite(probes).all():
        raise ValueError("expected four finite probe embeddings")
    derivative = np.stack((probes[0] - probes[1], probes[2] - probes[3]), axis=1)
    u, singular, _ = np.linalg.svd(derivative, full_matrices=False)
    valid = len(singular) == 2 and singular[0] > 0 and singular[1] > singular[0] * 1e-6
    return (u[:, :2] if valid else None), singular


def plan():
    if PLAN.exists() or FEATURES.exists():
        raise FileExistsError("output exists")
    prior, original = json.loads(previous.OUTPUT.read_text()), json.loads(SOURCE.PLAN.read_text())
    if (prior["features_sha256"] != file_hash(previous.FEATURES)
            or prior["source_sha256"] != file_hash(SOURCE.PLAN)):
        raise ValueError("previous data drift")
    model, scaler, transform, action_hash = SOURCE.common.model_setup()
    model_id = model_fingerprint(model)
    if model_id != prior["model_sha256"] or action_hash != prior["normalizer_action_sha256"]:
        raise ValueError("model or normalizer changed")
    rows, artifacts = [], {}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with np.load(SOURCE.inputs.INPUT, allow_pickle=False) as data, \
            np.load(previous.FEATURES, allow_pickle=False) as features, torch.inference_mode():
        for root, old in enumerate(original["rows"]):
            if int(data["rows"][root]) != old["row"]:
                raise ValueError("task mismatch")
            forecast = features[f"root_{root}_predicted"].astype(np.float64)
            goal = features[f"root_{root}_goal"].astype(np.float64)
            probe_images, inspection = make_goal_probes(data["goal"][root])
            basis, singular, probe_embedding = None, [], np.empty((0, len(goal)))
            if probe_images is not None:
                tensors = torch.stack([transform(image) for image in probe_images])[:, None].cuda()
                probe_embedding = model.encode({"pixels": tensors})["emb"][:, 0].cpu().numpy()
                basis, values = tangent_basis(probe_embedding)
                singular = values.tolist()
                if basis is None:
                    inspection["valid"] = False
                    inspection["reason"] = "rank_deficient"
            delta = forecast - goal
            raw_score = np.square(delta).sum(axis=1)
            random, _ = np.linalg.qr(np.random.default_rng(2719300 + root).normal(size=(len(goal), 2)), mode="reduced")
            scores = {"original": raw_score,
                      "goal_tangent": raw_score if basis is None else np.square(delta @ basis).sum(axis=1),
                      "random_tangent": np.square(delta @ random).sum(axis=1)}
            chains = [item[1] for item in candidates_for(old)]
            for ref in old["refinements"]:
                chains.extend(ref["actions"])
            if len(chains) != 656:
                raise ValueError("candidate count mismatch")
            outcomes = {}
            for arm in ARMS:
                index = int(np.argmin(scores[arm]))
                outcomes[arm] = {"candidate_index": index, "score": float(scores[arm][index]),
                                 "actions": SOURCE.to_physical(np.asarray(chains[index], dtype=np.float32), scaler).tolist()}
            rows.append({"row": old["row"], "root": root, "inspection": inspection, "singular_values": singular,
                         "extra_goal_encodings": 0 if probe_images is None else 4, "outcomes": outcomes})
            artifacts[f"root_{root}_scores"] = np.stack([scores[arm] for arm in ARMS])
            artifacts[f"root_{root}_probes"] = np.empty((0, 224, 224, 3), np.uint8) if probe_images is None else probe_images
            artifacts[f"root_{root}_probe_embedding"] = probe_embedding
            artifacts[f"root_{root}_basis"] = np.empty((len(goal), 0)) if basis is None else basis
            artifacts[f"root_{root}_random"] = random
            print(f"{root+1}/10 row={old['row']} valid={inspection['valid']} reason={inspection['reason']}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if model_fingerprint(model) != model_id:
        raise ValueError("model weight drift")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **artifacts)
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "features_sha256": file_hash(FEATURES), "source_sha256": file_hash(SOURCE.PLAN),
                   "previous_features_sha256": file_hash(previous.FEATURES), "model_sha256": model_id,
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "elapsed_seconds": elapsed, "scope": "post-hoc exposed RGB-only fixed-pool diagnostic"},
                  stream, separators=(",", ":"), allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    if (plan["runner_sha256"] != file_hash(__file__) or plan["features_sha256"] != file_hash(FEATURES)
            or plan["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("plan drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(SOURCE.common.DATA, "r") as data:
            for row in plan["rows"]:
                outcomes = {}
                for arm in ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    actions = np.asarray(row["outcomes"][arm]["actions"], dtype=np.float32)
                    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                        raise ValueError("invalid action")
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[arm] = {"trace": trace, "ever_success": any(x["terminated"] for x in trace),
                                     "final_distance": trace[-1]["distance"]}
                rows.append({"row": row["row"], "outcomes": outcomes})
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "runner_sha256": file_hash(__file__)},
                  stream, indent=2, allow_nan=False)
    for arm in ARMS:
        print(f"{arm}: {sum(x['outcomes'][arm]['ever_success'] for x in rows)}/{len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
