"""Two-stage saved-prediction task-score and outcome-only rank diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h

ROOT = h.ROOT
PRED = ROOT / "results/pc_pointmaze_cp12_three_frame_same_action_20261103_CODEX.json"
REFRESH = ROOT / "results/pc_causal_mixed_cp12_refresh_20261103_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP12_HISTORY_TASK_RANK_PROTOCOL_20261103_CODEX.md"
SCORE = ROOT / "results/pc_pointmaze_cp12_history_task_rank_scores_20261103_CODEX.json"
SCORE_NPZ = SCORE.with_suffix(".npz")
EVAL = ROOT / "results/pc_pointmaze_cp12_history_task_rank_evaluation_20261103_CODEX.json"
EXPECTED = {
    PRED: "e27017d25a31b7d35dc50d0ea9062d10c3541dbdfe4a12f3fc073b0a2af0aad7",
    REFRESH: "a2a967567427e2cf0d94a222547697505d7e595adea7a932b0f7ce9ad5bea1a3",
    PROTOCOL: "aac9dd65bb05ce5ffc4881431f83c30982c1be0d27a0bfc42e85c435ad46160d",
    h.OBSTACLE: h.OBSTACLE_SHA, h.CAMERA: h.CAMERA_SHA, h.OLD: h.OLD_SHA,
    h.DEFAULT_CHECKPOINT: "a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc",
}
ROOTS = (8343, 8345, 8349, 8351)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path):
    if sha(path) != EXPECTED[path]:
        raise ValueError(f"source hash changed: {path}")


def ranking(first, second):
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != (8,) or b.shape != (8,) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("eight finite scores per root required")
    i, j = np.triu_indices(8, 1)
    x, y = a[i] - a[j], b[i] - b[j]
    return {"pairs": 28, "comparable": int(((x != 0) & (y != 0)).sum()),
            "reversed": int((x * y < 0).sum()), "left_ties": int((x == 0).sum()),
            "right_ties": int((y == 0).sum())}


def score_stage():
    if any(p.exists() for p in (SCORE, SCORE_NPZ, EVAL)):
        raise FileExistsError("refusing overwrite")
    for p in (PRED, PROTOCOL, h.OBSTACLE, h.CAMERA, h.OLD, h.DEFAULT_CHECKPOINT):
        check(p)
    pred, obstacle, camera, old = (json.loads(p.read_text()) for p in (PRED, h.OBSTACLE, h.CAMERA, h.OLD))
    if sha(PRED.with_suffix(".npz")) != pred["arrays_sha256"] or sha(h.OBSTACLE.with_suffix(".npz")) != obstacle["arrays_sha256"]:
        raise ValueError("input array hash mismatch")
    if pred["clock"]["observations"] != [2, 7, 12] or pred["clock"]["prediction_target"] != 17:
        raise ValueError("prediction clock mismatch")
    if pred["model_before"] != pred["model_after"] or len(pred["rows"]) != 32:
        raise ValueError("prediction roster/model mismatch")
    weight, bias, _ = h.get_encoder()
    checkpoint = torch.load(h.DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    w = checkpoint["proprio_encoder"]["module.patch_embed.weight"].squeeze(-1)
    b = checkpoint["proprio_encoder"]["module.patch_embed.bias"]
    if not np.array_equal(w.numpy(), weight.astype(np.float32)) or not np.array_equal(b.numpy(), bias.astype(np.float32)):
        raise ValueError("checkpoint encoder mismatch")
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as data:
        state, _ = h.estimate_state(data["goal_rgb"], camera["metadata"], old["body_offset_from_static_scene"])
    state = torch.as_tensor(state, dtype=torch.float32)
    normalized = (state - torch.as_tensor(h.MEAN, dtype=torch.float32)) / torch.as_tensor(h.STD, dtype=torch.float32)
    goal_token = torch.nn.functional.linear(normalized, w, b).reshape(1, 16).expand(256, 16).contiguous()
    pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32)
    pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32)
    rows, outputs = [], {"goal_rgb_state": state.numpy(), "goal_token": goal_token.numpy()}
    print("Seal scores from saved same-action predictions; physical outcomes not opened.", flush=True)
    with np.load(PRED.with_suffix(".npz"), allow_pickle=False) as arrays:
        for index, branch in enumerate(pred["rows"]):
            root = branch["root"]
            if root != ROOTS[index // 8]:
                raise ValueError("root order mismatch")
            name = f"r{root}_a{('raw', 'spectral').index(branch['arm'])}_p{branch['candidate']}"
            if name in outputs:
                raise ValueError("duplicate branch")
            outputs[name] = np.array([root, index], dtype=np.int64)
            scores = {}
            for arm in ("one", "three", "target"):
                token = torch.from_numpy(arrays[f"{name}_{arm}_proprio"])
                if token.numel() != 256 * 16 or not torch.isfinite(token).all():
                    raise ValueError("invalid saved proprio tokens")
                scores[arm] = float(h.projected_cost(token.reshape(1, 256, 16), goal_token, pinv, pos_std)[0])
            rows.append({"root": root, "arm": branch["arm"], "candidate": branch["candidate"], **scores})
    with SCORE_NPZ.open("xb") as f:
        np.savez_compressed(f, **outputs)
    score_report = {"schema": "cp12_history_task_scores.v1", "rows": rows,
                    "goal_state": state.tolist(), "score_arrays_sha256": sha(SCORE_NPZ),
                    "prediction_sha256": sha(PRED), "prediction_npz_sha256": sha(PRED.with_suffix(".npz")),
                    "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
                    "files_sha256": {str(p): sha(p) for p in
                                     (h.OBSTACLE, h.OBSTACLE.with_suffix(".npz"), h.CAMERA, h.OLD, h.DEFAULT_CHECKPOINT)},
                    "counts": {"projected_scores": 96, "goal_affine_encodes": 1, "model_prediction_samples": 0,
                               "image_encodes": 0, "physical_steps": 0, "candidate_selections": 0}}
    with SCORE.open("x") as f:
        json.dump(score_report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"Sealed {SCORE}, sha256={sha(SCORE)}", flush=True)


def eval_stage():
    if EVAL.exists():
        raise FileExistsError("refusing overwrite")
    for path in (PRED, REFRESH, PROTOCOL):
        check(path)
    scored = json.loads(SCORE.read_text())
    if (scored["score_arrays_sha256"] != sha(SCORE_NPZ) or scored["prediction_sha256"] != sha(PRED)
            or scored["prediction_npz_sha256"] != sha(PRED.with_suffix(".npz"))
            or scored["protocol_sha256"] != sha(PROTOCOL) or scored["script_sha256"] != sha(Path(__file__))):
        raise ValueError("score receipt binding mismatch")
    prior = json.loads(REFRESH.read_text())
    if len(scored["rows"]) != 32 or len(prior["rows"]) != 4:
        raise ValueError("incomplete score roster")
    results, max_error = [], {"one": 0.0, "target": 0.0}
    print("Open old receipt for score parity and evaluation-only RGB/physical rank comparisons.", flush=True)
    for k, root in enumerate(ROOTS):
        old = prior["rows"][k]
        group = scored["rows"][8*k:8*k+8]
        if old["root"] != root or len(old["branches"]) != 8:
            raise ValueError("old roster mismatch")
        for new, old_branch in zip(group, old["branches"], strict=True):
            if (new["root"], new["arm"], new["candidate"]) != (root, old_branch["arm"], old_branch["candidate"]):
                raise ValueError("branch mismatch")
            for arm, field in (("one", "refresh_score"), ("target", "actual_score")):
                np.testing.assert_allclose(new[arm], old_branch[field], rtol=3e-5, atol=1e-6)
                max_error[arm] = max(max_error[arm], abs(new[arm] - old_branch[field]))
        # Use the original reference numbers to preserve the old tie semantics.
        refs = {"rgb": [x["actual_score"] for x in old["branches"]],
                "physical": [x["physical_distance"] for x in old["branches"]]}
        rankings = {}
        for arm in ("one", "three"):
            for ref, target in refs.items():
                rankings[f"{arm}_vs_{ref}"] = ranking([x[arm] for x in group], target)
        for ref in ("rgb", "physical"):
            actual = rankings[f"one_vs_{ref}"]
            expected = old["rankings"][f"refresh_vs_{ref}"]
            if any(actual[key] != expected[key] for key in ("pairs", "reversed", "left_ties", "right_ties")):
                raise ValueError(f"old rank parity failed for {root} {ref}: {actual} != {expected}")
        results.append({"root": root, "rankings": rankings,
                        "actual_rgb_scores": refs["rgb"], "physical_distance_eval_only": refs["physical"]})
    result = {"schema": "cp12_history_task_rank_evaluation.v1", "score_receipt_sha256": sha(SCORE),
              "old_refresh_sha256": sha(REFRESH), "protocol_sha256": sha(PROTOCOL),
              "one_and_target_max_abs_error_vs_old": max_error, "rows": results,
              "physical_data_use": "offline evaluation only; no planning or selection",
              "scope": "exposed dependent fixed-action branches, not independent confirmation"}
    with EVAL.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"Saved {EVAL}, sha256={sha(EVAL)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("score", "evaluate"))
    args = parser.parse_args()
    (score_stage if args.stage == "score" else eval_stage)()


if __name__ == "__main__":
    main()
