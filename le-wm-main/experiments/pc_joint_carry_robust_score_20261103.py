"""Two-stage fixed-bank perturbation scoring and offline outcome evaluation."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h

SOURCE = h.ROOT / "results/pc_joint_carry_all_roots_20261103_DEV01.json"
SOURCE_SHA = "f5977bd71e636a316935ecfce1045968adbcf6c70b100de5c825bfc20f2fe849"
PROTOCOL = h.ROOT.parent / "docs/PC_JOINT_CARRY_ROBUST_SCORE_PROTOCOL_20261103.md"
SCORE = h.ROOT / "results/pc_joint_carry_robust_scores_20261103_DEV01.json"
SCORE_NPZ = SCORE.with_suffix(".npz")
EVAL = h.ROOT / "results/pc_joint_carry_robust_evaluation_20261103_DEV01.json"
DELTAS = np.asarray([[0., 0.], [.05, 0.], [-.05, 0.], [0., .05], [0., -.05]], dtype=np.float32)


def perturb(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (2, 10, 2) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1):
        raise ValueError("invalid fixed tails")
    return np.clip(raw[:, None, :, :] + DELTAS[None, :, None, :], -1, 1).astype(np.float32)


def decisions(scores):
    scores = np.asarray(scores)
    if scores.shape != (2, 5) or not np.isfinite(scores).all():
        raise ValueError("invalid score matrix")
    return {"nominal": int(scores[1, 0] < scores[0, 0]),
            "robust_mean": int(scores[1].mean(dtype=np.float64) < scores[0].mean(dtype=np.float64))}


def score():
    if any(p.exists() for p in (SCORE, SCORE_NPZ, EVAL)):
        raise FileExistsError("refusing overwrite")
    for p, expected in ((SOURCE, SOURCE_SHA), (h.OBSTACLE, h.OBSTACLE_SHA),
                        (h.CAMERA, h.CAMERA_SHA), (h.OLD, h.OLD_SHA),
                        (h.DEFAULT_DATASET / "seq_lengths.pth", h.LENGTHS_SHA)):
        if h.sha256_file(p) != expected:
            raise ValueError(f"source changed: {p}")
    source, obstacle, camera, old = (json.loads(p.read_text()) for p in (SOURCE, h.OBSTACLE, h.CAMERA, h.OLD))
    for p, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                        (h.OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                        (h.DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if h.sha256_file(p) != expected:
            raise ValueError(f"source changed: {p}")
    if len(source["rows"]) != 51:
        raise ValueError("source roster")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    output, rows, started = {}, [], monotonic()
    print("Score fixed old/carry tails and fixed perturbations; no outcome-based decisions.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as data, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            h.lock.gpu_lock(h.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if before != source["model_before"] or before != source["model_after"]:
            raise ValueError("model mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not (np.array_equal(mean.cpu().numpy(), np.asarray(source["action_mean"], dtype=np.float32))
                and np.array_equal(std.cpu().numpy(), np.asarray(source["action_std"], dtype=np.float32))):
            raise ValueError("action normalizer mismatch")
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for i in range(0, len(source["rows"]), 3):
            group = source["rows"][i:i+3]
            if [r["mode"] for r in group] != ["old", "zero_init", "carry_init"]:
                raise ValueError("source ordering")
            root, role = group[0]["root"], group[0]["role"]
            if any((r["root"], r["role"]) != (root, role) for r in group):
                raise ValueError("source identity")
            prefix = f"r{root}_{role}"
            old_actions, new_actions = data[prefix+"_old_actions"], data[prefix+"_carry_init_actions"]
            frames = data[prefix+"_old_history"]
            if (old_actions.shape != (17, 2) or new_actions.shape != (17, 2)
                    or frames.shape != (18, 224, 224, 3)
                    or not np.array_equal(old_actions[:7], new_actions[:7])
                    or not np.array_equal(frames[:8], data[prefix+"_carry_init_history"][:8])):
                raise ValueError("RGB/action context binding")
            state = h.causal_state(frames[6], frames[7], metadata, offset, .05)
            context = h.encode_with_state(model, frames[7], state, device)
            raw = perturb(np.stack([old_actions[7:], new_actions[7:]]))
            normalized = (torch.as_tensor(raw, device=device) - mean) / std
            packets = normalized.reshape(10, 2, 10).permute(1, 0, 2)
            with torch.no_grad():
                pred = model.unroll(context, act_suffix=packets)["proprio"]
                if pred.shape[:2] != (3, 10):
                    raise ValueError("prediction clock")
                scores = h.projected_cost(pred[2].reshape(10, 256, 16), target, pinv, pos_std).reshape(2, 5)
            scores = scores.cpu().numpy()
            for j, ref in enumerate((group[0], group[2])):
                expected = ref["refined_score"] if ref["refined"] else ref["prior_score"]
                np.testing.assert_allclose(scores[j, 0], expected, rtol=3e-5, atol=1e-6)
            output[prefix+"_raw"] = raw
            output[prefix+"_scores"] = scores
            rows.append({"root": root, "role": role, "nominal_scores": scores[:, 0].tolist(),
                         "mean_scores": scores.mean(axis=1, dtype=np.float64).tolist(),
                         "decisions": decisions(scores)})
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise ValueError("model parameters changed")
    with SCORE_NPZ.open("xb") as f:
        np.savez_compressed(f, **output)
    report = {"schema": "pc_joint_carry_robust_scores.v1", "rows": rows, "source_sha256": SOURCE_SHA,
              "source_arrays_sha256": source["arrays_sha256"], "score_arrays_sha256": h.sha256_file(SCORE_NPZ),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "cp7_encodes": len(rows), "two_packet_samples": len(rows)*10,
                         "physical_commands": 0}, "wall_seconds": monotonic()-started,
              "scope": "exposed development fixed action probes; no outcome-selected rule"}
    with SCORE.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"Sealed {SCORE} sha256={h.sha256_file(SCORE)}", flush=True)


def evaluate():
    if EVAL.exists():
        raise FileExistsError("refusing overwrite")
    source, receipt = json.loads(SOURCE.read_text()), json.loads(SCORE.read_text())
    if (h.sha256_file(SOURCE) != SOURCE_SHA or receipt["source_sha256"] != SOURCE_SHA
            or receipt["score_arrays_sha256"] != h.sha256_file(SCORE_NPZ)
            or receipt["protocol_sha256"] != h.sha256_file(PROTOCOL)
            or receipt["script_sha256"] != h.sha256_file(Path(__file__))):
        raise ValueError("score receipt binding")
    rows = []
    for i, scored in enumerate(receipt["rows"]):
        group = source["rows"][3*i:3*i+3]
        if len(group) != 3 or [x["mode"] for x in group] != ["old", "zero_init", "carry_init"]:
            raise ValueError("outcome roster")
        if any((r["root"], r["role"]) != (scored["root"], scored["role"]) for r in group):
            raise ValueError("outcome identity")
        old, carry = group[0]["cp17_distance"], group[2]["cp17_distance"]
        rows.append({"root": scored["root"], "role": scored["role"], "old_distance": old,
                     "carry_distance": carry, "decisions": scored["decisions"],
                     "nominal_distance": (old, carry)[scored["decisions"]["nominal"]],
                     "robust_distance": (old, carry)[scored["decisions"]["robust_mean"]]})
    if len(rows) != 17:
        raise ValueError("outcome count")
    report = {"schema": "pc_joint_carry_robust_evaluation.v1", "rows": rows,
              "score_sha256": h.sha256_file(SCORE), "source_sha256": SOURCE_SHA,
              "protocol_sha256": h.sha256_file(PROTOCOL),
              "scope": "post-outcome dependent development; not physical counterfactuals for perturbations"}
    with EVAL.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"Saved {EVAL}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("score", "evaluate"))
    args = parser.parse_args()
    (score if args.stage == "score" else evaluate)()


if __name__ == "__main__":
    main()
