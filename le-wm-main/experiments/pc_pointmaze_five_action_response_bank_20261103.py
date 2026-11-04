"""Task-projected five-action response bank on exposed RGB obstacle roots."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.pc_pointmaze_affine_feedback_cem_20261103 import projected_cost
from experiments.pc_pointmaze_affine_inverse_20261103 import MEAN, STD, get_encoder
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import LENGTHS_SHA
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import SOURCE_SHA
from experiments.pc_pointmaze_obstacle_fixed_chain_decomp_20261103 import rank_counts
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import DATA, GOAL, ROOTS, OUTPUT as SOURCE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_FIVE_ACTION_RESPONSE_BANK_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_five_action_response_bank_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def make_bank(jacobian):
    """Return fixed ±task-response and ±nullspace bounded action vectors."""
    jac = np.asarray(jacobian, np.float64)
    if jac.shape != (2, 10) or not np.isfinite(jac).all():
        raise ValueError("finite 2x10 response Jacobian required")
    _, singular, vh = np.linalg.svd(jac, full_matrices=True)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-5:
        raise ValueError(f"task response rank below two: singular values {singular}")
    directions = vh[[0, 1, 2, 3]]
    if np.any(np.max(np.abs(directions), axis=1) <= 0):
        raise ValueError("zero basis vector")
    directions = directions / np.max(np.abs(directions), axis=1, keepdims=True)
    bank = np.stack([np.zeros(10)] + [sign * .5 * d for d in directions for sign in (1., -1.)])
    if not np.isfinite(bank).all() or np.any(np.abs(bank) > .5 + 1e-12):
        raise ValueError("invalid bank")
    return bank.astype(np.float32).reshape(9, 5, 2), singular, directions


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA),
                           (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"input changed: {path}")
    source, camera, old = (json.loads(p.read_text()) for p in (SOURCE, CAMERA, OLD))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DATA / "states.pth", source["states_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError(f"input changed: {path}")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu")
    starts, sequences, rows, arrays = [], [], [], {}
    tic = time.monotonic()
    print("Measure full five-action task response and execute the fixed bank on exposed roots.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("model hash mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        action_mean, action_std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(action_std).all() and (action_std > 0).all()
                    and torch.isfinite(action_mean).all()):
            raise ValueError("invalid action normalizer")
        weight, _, _ = get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = saved["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, metadata, offset)
        target = encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for root in ROOTS:
            initial = states[root, 40].numpy().astype(np.float64)
            initial_rgb = saved[f"r{root}_random_frames"][0]
            state0, _ = estimate_state(initial_rgb, metadata, offset)
            context = encode_with_state(model, initial_rgb, state0, device)

            def predict(raw):
                packet = ((raw.reshape(5, 2) - action_mean) / action_std).reshape(1, 1, 10)
                predicted = model.unroll(context, act_suffix=packet)["proprio"]
                if predicted.shape[:2] != (2, 1):
                    raise ValueError("prediction action clock mismatch")
                return predicted[-1].reshape(1, 256, 16)

            def task_output(raw):
                pred = predict(raw)[0]
                return (pred.mean(dim=0) @ pinv.T)[:2] * pos_std

            zero = torch.zeros(10, dtype=torch.float32, device=device)
            basis = torch.eye(10, dtype=torch.float32, device=device)
            derivatives = [torch.autograd.functional.jvp(task_output, zero, basis[i])[1].detach()
                           for i in range(10)]
            jac = torch.stack(derivatives, dim=1).cpu().numpy()
            with torch.no_grad():
                y0 = task_output(zero).cpu().numpy()
                fd = np.stack([((task_output(zero + .001 * basis[i])
                                 - task_output(zero - .001 * basis[i])) / .002).cpu().numpy()
                               for i in range(10)], axis=1)
            bank, singular, directions = make_bank(jac)
            arrays[f"r{root}_jacobian"] = jac
            arrays[f"r{root}_fd_jacobian"] = fd
            arrays[f"r{root}_directions"] = directions
            arrays[f"r{root}_bank"] = bank
            abs_error = float(np.linalg.norm(jac - fd))
            relative_error = abs_error / max(float(np.linalg.norm(jac)), 1e-12)
            predicted_rows = []
            with torch.no_grad():
                for action in bank:
                    pred = predict(torch.as_tensor(action, device=device))
                    score = float(projected_cost(pred, target, pinv, pos_std).item())
                    y = ((pred[0].mean(dim=0) @ pinv.T)[:2] * pos_std).cpu().numpy()
                    predicted_rows.append({"predicted_score": score, "predicted_response": (y - y0).tolist(),
                                           "linear_response": (jac @ action.reshape(10)).tolist()})
            images, actual_scores = [], []
            with torch.no_grad():
                for action in bank:
                    with NativeRGBBridge(initial) as env:
                        if not np.array_equal(env.initial_rgb, initial_rgb):
                            raise RuntimeError("initial RGB mismatch")
                        history = [env.initial_rgb]
                        for single in action:
                            history.append(env.step(single[None]))
                        if env.step_count != 5:
                            raise RuntimeError("physical clock mismatch")
                    velocity_state = causal_state(history[4], history[5], metadata, offset, .05)
                    actual = encode_with_state(model, history[5], velocity_state, device)["proprio"].reshape(1, 256, 16)
                    actual_scores.append(float(projected_cost(actual, target, pinv, pos_std).item()))
                    images.append(np.stack(history))
                    starts.append(initial)
                    sequences.append(action)
            arrays[f"r{root}_frames"] = np.stack(images)
            rows.append({"root": root, "singular_values": singular.tolist(), "fd_abs_error": abs_error,
                         "fd_rel_error": relative_error, "candidates": predicted_rows,
                         "actual_rgb_derived_scores": actual_scores})
            print(f"root {root}: FD relative error {relative_error:.6g}; 9 trajectories", flush=True)
        torch.cuda.synchronize()
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp5 = physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"], arrays["cp5_eval_only"] = starts, sequences, cp5
    for i, row in enumerate(rows):
        distances = np.linalg.norm(cp5[9*i:9*i+9, :2] - GOAL[:2], axis=1)
        predicted = [c["predicted_score"] for c in row["candidates"]]
        row["initial_physical_distance"] = float(np.linalg.norm(starts[9*i, :2] - GOAL[:2]))
        row["physical_distances"] = distances.tolist()
        row["ranking"] = {"predicted_vs_actual_rgb": rank_counts(predicted, row["actual_rgb_derived_scores"]),
                          "predicted_vs_physical": rank_counts(predicted, distances),
                          "actual_rgb_vs_physical": rank_counts(row["actual_rgb_derived_scores"], distances)}
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_five_action_response_bank.v1", "rows": rows,
              "scope": "exposed fixed bank development; no independent planning or gate",
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "source_sha256": SOURCE_SHA, "arrays_sha256": sha256_file(ARRAYS),
              "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
              "config_sha256": sha256_file(DEFAULT_CONFIG), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "initial_encodes": 4, "cp5_encodes": 36,
                         "jvp_calls": 40, "finite_difference_samples": 80,
                         "zero_reference_predictions": 4, "candidate_predictions": 36,
                         "rgb_physical_steps": 180, "position_backend_steps": 180},
              "wall_seconds": time.monotonic() - tic}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT} sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
