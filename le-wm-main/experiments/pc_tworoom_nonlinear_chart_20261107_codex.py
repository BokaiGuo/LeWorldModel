"""Nonlinear RGB template chart: sealed model predictions, then truth evaluation."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_goal_chart_actual_successor_20261107_codex as actual
from experiments import pc_tworoom_dual_rgb_goal_tangent_20261107_codex as prior
from experiments import pc_tworoom_dual_rgb_goal_reconstruction_20261107_codex as recon

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RGB_NONLINEAR_CHART_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_nonlinear_chart_prediction_20261107_codex.json"
DICTIONARY = ROOT / "results/pc_tworoom_nonlinear_chart_dictionary_20261107_codex.npz"
RESULT = ROOT / "results/pc_tworoom_nonlinear_chart_result_20261107_codex.json"
OFFSETS = [(x, y) for y in range(-64, 65, 8) for x in range(-64, 65, 8)]


def templates(current, goal, *, offsets=OFFSETS):
    _, info = recon.make_probes(current, goal)
    if not info["valid"]:
        return None, None, info
    current_alpha, current_mask = recon.red_alpha(current)
    goal_alpha, goal_mask = recon.red_alpha(goal)
    goal_alpha *= goal_mask
    background = goal.astype(np.float32).copy()
    visible = goal_mask & ~current_mask
    background[visible] = current[visible]
    overlap = goal_mask & current_mask
    if overlap.any():
        alpha = current_alpha[overlap, None]
        background[overlap] = np.clip((current[overlap].astype(np.float32)
                                      - alpha * np.asarray([255., 0., 0.])) / (1 - alpha), 0, 255)
    ys, xs = np.nonzero(goal_mask)
    images, retained = [], []
    for dx, dy in offsets:
        if (xs.min() + dx < 0 or xs.max() + dx >= 224 or ys.min() + dy < 0 or ys.max() + dy >= 224):
            continue
        moved = cv2.warpAffine(goal_alpha, np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
                               (224, 224), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        images.append(np.clip(np.rint(background * (1 - moved[..., None])
                                      + np.asarray([255., 0., 0.]) * moved[..., None]), 0, 255).astype(np.uint8))
        retained.append((dx, dy))
    if (0, 0) not in retained:
        raise ValueError("dictionary must include zero")
    return np.stack(images), np.asarray(retained), info


def nearest(dictionary, query, offsets):
    dictionary, query, offsets = np.asarray(dictionary, float), np.asarray(query, float), np.asarray(offsets)
    if (dictionary.ndim != 2 or query.ndim != 2 or dictionary.shape[1] != query.shape[1]
            or offsets.shape != (len(dictionary), 2) or not len(dictionary)
            or not np.isfinite(dictionary).all() or not np.isfinite(query).all()):
        raise ValueError("incompatible finite dictionary and query")
    distances = ((query[:, None] - dictionary[None]) ** 2).sum(axis=2)
    return offsets[np.argmin(distances, axis=1)]


def predict():
    if PLAN.exists() or DICTIONARY.exists():
        raise FileExistsError("output exists")
    old = json.loads(prior.PLAN.read_text())
    previous = json.loads(prior.old.previous.OUTPUT.read_text())
    if (old["source_sha256"] != file_hash(prior.old.SOURCE.PLAN)
            or old["prediction_features_sha256"] != file_hash(prior.old.previous.FEATURES)
            or old["runner_sha256"] != file_hash(prior.__file__)
            or old["reconstruction_sha256"] != file_hash(recon.OUTPUT)):
        raise ValueError("prior source drift")
    model, _, transform, action_hash = prior.old.SOURCE.common.model_setup()
    if model_fingerprint(model) != old["model_sha256"] or action_hash != previous["normalizer_action_sha256"]:
        raise ValueError("checkpoint or normalizer mismatch")
    rows, arrays = [], {}
    with np.load(prior.old.SOURCE.inputs.INPUT, allow_pickle=False) as inputs, \
            np.load(prior.old.previous.FEATURES, allow_pickle=False) as forecasts, torch.inference_mode():
        for root, task in enumerate(old["rows"]):
            if task["row"] != int(inputs["rows"][root]):
                raise ValueError("root mismatch")
            images, offsets, info = templates(inputs["current"][root], inputs["goal"][root])
            if not info["valid"]:
                rows.append({"root": root, "row": task["row"], "valid": False, "reason": info["reason"]})
                continue
            torch.cuda.synchronize()
            started = time.perf_counter()
            embedding = []
            for offset in range(0, len(images), 32):
                batch = torch.stack([transform(image) for image in images[offset:offset + 32]])[:, None].cuda()
                embedding.append(model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy())
            embedding = np.concatenate(embedding)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            forecast = forecasts[f"root_{root}_predicted"]
            indexes = [task["outcomes"][arm]["candidate_index"] for arm in prior.ARMS]
            displacement = nearest(embedding, forecast[indexes], offsets)
            rows.append({"root": root, "row": task["row"], "valid": True, "count": len(images),
                         "candidate_indexes": indexes, "predicted_offsets": displacement.tolist(),
                         "encoding_seconds": elapsed})
            arrays[f"root_{root}_embedding"] = embedding
            arrays[f"root_{root}_offsets"] = offsets
            print(f"root {root}: encoded {len(images)} RGB templates", flush=True)
    if model_fingerprint(model) != old["model_sha256"]:
        raise ValueError("model weight drift")
    with DICTIONARY.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "dictionary_sha256": file_hash(DICTIONARY), "prior_plan_sha256": file_hash(prior.PLAN),
                   "forecast_sha256": file_hash(prior.old.previous.FEATURES), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "model_sha256": old["model_sha256"],
                   "scope": "exposed post-hoc RGB template development; no physical truth used"},
                  stream, indent=2, allow_nan=False)


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    pred, truth = json.loads(PLAN.read_text()), json.loads(actual.RESULT.read_text())
    if (pred["dictionary_sha256"] != file_hash(DICTIONARY) or pred["prior_plan_sha256"] != file_hash(prior.PLAN)
            or pred["protocol_sha256"] != file_hash(PROTOCOL) or pred["runner_sha256"] != file_hash(Path(__file__))
            or truth["features_sha256"] != file_hash(actual.FEATURES)):
        raise ValueError("artifact drift")
    rows = []
    with np.load(DICTIONARY, allow_pickle=False) as dictionaries, \
            np.load(actual.FEATURES, allow_pickle=False) as true_features:
        for p, t in zip(pred["rows"], truth["rows"], strict=True):
            if (p["root"], p["row"]) != (t["root"], t["row"]):
                raise ValueError("task mismatch")
            if not p["valid"] or not t["valid"]:
                rows.append({"root": p["root"], "row": p["row"], "valid": False})
                continue
            dictionary, offsets = dictionaries[f"root_{p['root']}_embedding"], dictionaries[f"root_{p['root']}_offsets"]
            decoded = nearest(dictionary, true_features[f"root_{p['root']}_actual"], offsets)
            predicted = np.asarray(p["predicted_offsets"])
            physical = np.asarray([o["true_displacement"] for o in t["outputs"]])
            if [o["candidate_index"] for o in t["outputs"]] != p["candidate_indexes"]:
                raise ValueError("candidate mismatch")
            rows.append({"root": p["root"], "row": p["row"], "valid": True,
                         "zero_error": float(np.linalg.norm(physical, axis=1).mean()),
                         "actual_error": float(np.linalg.norm(decoded - physical, axis=1).mean()),
                         "predicted_error": float(np.linalg.norm(predicted - physical, axis=1).mean()),
                         "decoded_actual": decoded.tolist(), "predicted": predicted.tolist(),
                         "dictionary_size": p["count"]})
    valid = [r for r in rows if r["valid"]]
    if not valid:
        raise ValueError("no valid roots")
    zero = np.mean([r["zero_error"] for r in valid])
    true_error = np.mean([r["actual_error"] for r in valid])
    predicted_error = np.mean([r["predicted_error"] for r in valid])
    true_better = sum(r["actual_error"] < r["zero_error"] for r in valid)
    predicted_better = sum(r["predicted_error"] < r["zero_error"] for r in valid)
    summary = {"valid_roots": len(valid), "abstained": len(rows) - len(valid), "zero_error": float(zero),
               "actual_error": float(true_error), "predicted_error": float(predicted_error),
               "actual_better_roots": true_better, "predicted_better_roots": predicted_better,
               "passes_development_exit": bool(len(valid) >= 6 and zero > 0 and true_error <= .75 * zero
                                                and predicted_error <= .75 * zero and true_better >= 6
                                                and predicted_better >= 6)}
    with RESULT.open("x") as stream:
        json.dump({"summary": summary, "rows": rows, "prediction_sha256": file_hash(PLAN),
                   "truth_sha256": file_hash(actual.RESULT), "truth_features_sha256": file_hash(actual.FEATURES),
                   "protocol_sha256": file_hash(PROTOCOL), "scope": "post-hoc exposed diagnostic only"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "analyze"):
        raise SystemExit("expected predict|analyze")
    {"predict": predict, "analyze": analyze}[sys.argv[1]]()
