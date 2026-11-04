"""Training-free affine inverse of saved, frozen-model proprio predictions.

Scoring and outcome evaluation are separate commands. No neural inference or
simulator access is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/pc_pointmaze_velocity_score_crosstab_RUN_20261103_codexexclusive_02"
CHECKPOINT = ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PROPRIO_AFFINE_INVERSE_PROTOCOL_20261103.md"
OUT = ROOT / "results/pc_pointmaze_proprio_affine_inverse_20261103"
REPORT_SHA = "e853d368712e3e09c3e8610c0a64f93b60261a0a94f7f97eb9dfefbd0051a029"
CHECKPOINT_SHA = "a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc"
MEAN = np.array([1.8130563497543335, 1.9377127885818481,
                 -0.0039275349117815495, -0.019777977839112282])
STD = np.array([1.0055010318756104, 0.9630089402198792,
                1.6802769899368286, 1.9319307804107666])
MODES = ("zero", "causal", "privileged")
GROUPS = tuple(f"r{root}_{method}" for root in (1884, 1885, 1886, 1887)
               for method in ("random_cem", "spectral_cem"))


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_encoder():
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("checkpoint SHA drift")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    encoder = checkpoint["proprio_encoder"]
    weight = encoder["module.patch_embed.weight"].numpy().astype(np.float64).squeeze(-1)
    bias = encoder["module.patch_embed.bias"].numpy().astype(np.float64)
    if weight.shape != (16, 4) or bias.shape != (16,) or not (
            np.isfinite(weight).all() and np.isfinite(bias).all()):
        raise ValueError("encoder parameters not affine 4 to 16")
    singular = np.linalg.svd(weight, compute_uv=False)
    if np.linalg.matrix_rank(weight) != 4:
        raise ValueError("encoder must have full column rank")
    return weight, bias, singular


def project_tokens(predicted, goal_token, weight, std):
    predicted, goal_token, weight, std = map(np.asarray, (predicted, goal_token, weight, std))
    if (predicted.shape != (48, 256, 16) or goal_token.shape != (256, 16)
            or weight.shape != (16, 4) or std.shape != (4,)
            or not all(np.isfinite(x).all() for x in (predicted, goal_token, weight, std))
            or np.any(std <= 0)):
        raise ValueError("invalid fixed-bank token contract")
    mean = predicted.mean(axis=1, dtype=np.float64)
    goal = goal_token.mean(axis=0, dtype=np.float64)
    pinv = np.linalg.pinv(weight)
    delta = (mean - goal) @ pinv.T
    scores = np.linalg.norm(delta[:, :2] * std[:2], axis=1)
    error = mean - goal - delta @ weight.T
    residual = np.linalg.norm(error, axis=1)
    patch_spread = np.sqrt(np.mean((predicted - mean[:, None, :]) ** 2, axis=(1, 2)))
    return scores, residual, patch_spread


def ranking(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != (48,) or second.shape != (48,) or not (
            np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("48 finite paired scores required")
    i, j = np.triu_indices(48, 1)
    x, y = first[i] - first[j], second[i] - second[j]
    comparable = (x != 0) & (y != 0)
    return {"comparable": int(comparable.sum()), "reversed": int(np.sum(x * y < 0)),
            "first_ties": int(np.sum(x == 0)), "second_ties": int(np.sum(y == 0))}


def source_report():
    report_path = RUN / "report.json"
    if sha(report_path) != REPORT_SHA:
        raise ValueError("saved CUDA report drift")
    report = json.loads(report_path.read_text())
    if report["model_before"] != report["model_after"] or set(report["group_sha256"]) != set(GROUPS):
        raise ValueError("frozen model or roster mismatch")
    return report


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def score():
    scores_path = OUT.with_name(OUT.name + "_scores.npz")
    receipt_path = OUT.with_name(OUT.name + "_scores.json")
    if scores_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing overwrite")
    report, (weight, bias, singular) = source_report(), get_encoder()
    arrays, checks = {}, []
    for key in GROUPS:
        path = RUN / (key + ".npz")
        if sha(path) != report["group_sha256"][key]:
            raise ValueError("saved group drift")
        with np.load(path, allow_pickle=False) as group:
            goal = group["goal_proprio_token"].reshape(256, 16)
            state = group["goal_state_rgb_public"]
            if state.shape != (4,) or not np.isfinite(state).all():
                raise ValueError("missing public RGB goal")
            reconstructed = ((state - MEAN) / STD) @ weight.T + bias
            goal_error = float(np.max(np.abs(goal - reconstructed)))
            if goal_error > 3e-5:
                raise ValueError(f"goal token/normalization mismatch: {key}: {goal_error}")
            check = {"group": key, "goal_encoding_max_abs_error": goal_error, "modes": {}}
            for mode in MODES:
                values, residual, spread = project_tokens(group[mode + "_predicted_proprio"], goal, weight, STD)
                arrays[f"{key}_{mode}_score"] = values
                arrays[f"{key}_{mode}_residual"] = residual
                arrays[f"{key}_{mode}_patch_spread"] = spread
                check["modes"][mode] = {"score_finite": bool(np.isfinite(values).all()),
                                       "projection_residual_mean": float(residual.mean()),
                                       "patch_spread_rms_mean": float(spread.mean())}
            checks.append(check)
    with scores_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(receipt_path, {"schema": "pc_affine_inverse_scores.v1", "source_sha256": REPORT_SHA,
                              "checkpoint_sha256": CHECKPOINT_SHA, "score_sha256": sha(scores_path),
                              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
                              "singular_values": singular.tolist(),
                              "condition_number": float(singular[0] / singular[-1]),
                              "normalizer_mean": MEAN.tolist(), "normalizer_std": STD.tolist(),
                              "checks": checks, "physical_outcomes_accessed": False})
    print(f"Saved fixed affine inverse scores, condition number={singular[0] / singular[-1]:.6g}")


def evaluate():
    score_path = OUT.with_name(OUT.name + "_scores.npz")
    receipt_path = OUT.with_name(OUT.name + "_scores.json")
    output = OUT.with_name(OUT.name + "_evaluation.json")
    if output.exists():
        raise FileExistsError("refusing overwrite")
    receipt, report = json.loads(receipt_path.read_text()), source_report()
    if (sha(score_path) != receipt["score_sha256"] or sha(Path(__file__)) != receipt["script_sha256"]
            or sha(PROTOCOL) != receipt["protocol_sha256"]
            or sha(RUN / "evaluation.npz") != report["evaluation_sha256"]):
        raise ValueError("sealed score or outcome drift")
    rows = []
    with np.load(score_path, allow_pickle=False) as scores, \
            np.load(RUN / "evaluation.npz", allow_pickle=False) as labels:
        for key in GROUPS:
            group_path = RUN / (key + ".npz")
            if sha(group_path) != report["group_sha256"][key]:
                raise ValueError("group drift")
            with np.load(group_path, allow_pickle=False) as group:
                physical = labels[key + "_physical_distance"]
                actual_rgb = group["actual_rgb_cost"]
                row = {"group": key, "modes": {}}
                for mode in MODES:
                    variants = {"affine_inverse": scores[f"{key}_{mode}_score"],
                                "visual": group[mode + "_visual"], "proprio": group[mode + "_proprio"]}
                    row["modes"][mode] = {
                        name: {"vs_physical": ranking(values, physical),
                               "vs_actual_rgb": ranking(values, actual_rgb),
                               "choice": int(np.argmin(values)),
                               "choice_physical_distance": float(physical[np.argmin(values)])}
                        for name, values in variants.items()}
                rows.append(row)
    totals = {}
    for mode in MODES:
        totals[mode] = {}
        for name in ("affine_inverse", "visual", "proprio"):
            totals[mode][name] = {
                ref: {k: sum(r["modes"][mode][name][ref][k] for r in rows)
                      for k in ("comparable", "reversed", "first_ties", "second_ties")}
                for ref in ("vs_physical", "vs_actual_rgb")}
    write_json(output, {"schema": "pc_affine_inverse_evaluation.v1", "rows": rows, "totals": totals,
                        "score_receipt_sha256": sha(receipt_path), "source_report_sha256": REPORT_SHA,
                        "scope": "exposed dependent fixed candidates; oracle cp5; not planner or gate"})
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("score", "evaluate"))
    args = parser.parse_args()
    print("Offline checkpoint affine inverse; no learned-model inference or physical execution.")
    (score if args.stage == "score" else evaluate)()
