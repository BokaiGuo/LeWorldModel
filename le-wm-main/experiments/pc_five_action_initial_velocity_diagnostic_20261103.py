"""Privileged initial-qvel diagnostic on a fixed, exposed five-action bank."""
import argparse
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
from experiments.pc_pointmaze_five_action_response_bank_20261103 import OUTPUT as SOURCE
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import SOURCE_SHA as OBSTACLE_SHA
from experiments.pc_pointmaze_obstacle_fixed_chain_decomp_20261103 import rank_counts
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS, OUTPUT as OBSTACLE
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "2e3de9eafe9c26f464927534ac5a20360ed62b6eb8bbf1f2504aa71fee670456"
PROTOCOL = ROOT.parent / "docs/PC_FIVE_ACTION_INITIAL_VELOCITY_DIAGNOSTIC_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_five_action_initial_velocity_diagnostic_20261103_DEV01.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (OBSTACLE, OBSTACLE_SHA),
                           (CAMERA, CAMERA_SHA), (OLD, OLD_SHA),
                           (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"input changed: {path}")
    source, obstacle, camera, old = (json.loads(p.read_text()) for p in (SOURCE, OBSTACLE, CAMERA, OLD))
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                         (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError(f"input changed: {path}")
    if [r["root"] for r in source["rows"]] != list(ROOTS):
        raise ValueError("root order changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, tic = [], time.monotonic()
    print("Compare fixed-action predictions under zero and privileged diagnostic initial velocity.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            np.load(OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("frozen model changed")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalization mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalizer")
        weight, _, _ = get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        rgb_goal = obstacle_arrays["goal_rgb"]
        goal_state, _ = estimate_state(rgb_goal, metadata, offset)
        with torch.no_grad():
            target = encode_with_state(model, rgb_goal, goal_state, device)["proprio"].reshape(256, 16)
            starts = saved["starts"]
            if starts.shape != (36, 4) or not np.isfinite(starts).all():
                raise ValueError("invalid reset states")
            for k, root in enumerate(ROOTS):
                frames, bank = saved[f"r{root}_frames"], saved[f"r{root}_bank"]
                if frames.shape != (9, 6, 224, 224, 3) or bank.shape != (9, 5, 2):
                    raise ValueError("invalid saved observations or bank")
                if not np.array_equal(starts[k*9:k*9+9], np.broadcast_to(starts[k*9], (9, 4))):
                    raise ValueError("initial reset not shared")
                state, _ = estimate_state(frames[0, 0], metadata, offset)
                zero_state = np.asarray(state, dtype=np.float32)
                if not np.array_equal(zero_state[2:], [0., 0.]):
                    raise ValueError("baseline estimator changed")
                privileged = zero_state.copy()
                privileged[2:] = starts[k*9, 2:]
                predictions = {}
                for name, input_state in (("zero", zero_state), ("privileged_qvel", privileged)):
                    context = encode_with_state(model, frames[0, 0], input_state, device)
                    values = []
                    for actions in bank:
                        packet = ((torch.as_tensor(actions, device=device) - mean) / std).reshape(1, 1, 10)
                        prediction = model.unroll(context, act_suffix=packet)["proprio"]
                        if prediction.shape[:2] != (2, 1):
                            raise ValueError("action clock mismatch")
                        values.append(float(projected_cost(prediction[-1].reshape(1, 256, 16),
                                                           target, pinv, pos_std).item()))
                    predictions[name] = values
                actual = []
                for history in frames:
                    state5 = causal_state(history[4], history[5], metadata, offset, .05)
                    encoded = encode_with_state(model, history[5], state5, device)["proprio"].reshape(1, 256, 16)
                    actual.append(float(projected_cost(encoded, target, pinv, pos_std).item()))
                prior = source["rows"][k]
                np.testing.assert_allclose(predictions["zero"], [c["predicted_score"] for c in prior["candidates"]],
                                           rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(actual, prior["actual_rgb_derived_scores"], rtol=1e-5, atol=1e-6)
                rows.append({"root": root, "zero_state": zero_state.tolist(),
                             "privileged_state": privileged.tolist(), "predictions": predictions,
                             "actual_rgb_derived_scores": actual})
                print(f"root {root}: both velocity modes and fixed scores reproduced", flush=True)
        torch.cuda.synchronize()
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
        cp5 = saved["cp5_eval_only"]
        if cp5.shape[0] != 36 or cp5.shape[1] < 2 or not np.isfinite(cp5).all():
            raise ValueError("invalid saved endpoints")
        for k, row in enumerate(rows):
            physical = np.linalg.norm(cp5[k*9:k*9+9, :2] - GOAL[:2], axis=1)
            np.testing.assert_allclose(physical, source["rows"][k]["physical_distances"], rtol=0, atol=1e-12)
            row["physical_distances"] = physical.tolist()
            row["ranking"] = {
                f"{mode}_vs_{ref}": rank_counts(row["predictions"][mode], reference)
                for mode in ("zero", "privileged_qvel")
                for ref, reference in (("actual", row["actual_rgb_derived_scores"]), ("physical", physical))}
    report = {"schema": "pc_five_action_initial_velocity_diagnostic.v1", "rows": rows,
              "scope": "exposed fixed-action offline diagnostic; simulator qvel privileged, no planner",
              "source_sha256": SOURCE_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "initial_encodes": 8, "cp5_encodes": 36,
                         "prediction_samples": 72, "new_physical_steps": 0},
              "wall_seconds": time.monotonic() - tic}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
