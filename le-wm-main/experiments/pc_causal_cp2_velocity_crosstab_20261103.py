"""Fixed-action cp2 velocity cross-tab; simulator qvel is offline only."""
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
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import SOURCE_SHA as OBSTACLE_SHA
from experiments.pc_pointmaze_obstacle_fixed_chain_decomp_20261103 import rank_counts
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS, OUTPUT as OBSTACLE
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/pc_causal_history_full_bank_replan_20261103_DEV01.json"
SOURCE_SHA = "d04c8b56024dbe3cbb923bdc5a76f930f9af1e5438b5d5e1c87d72ae41fa316a"
PROTOCOL = ROOT.parent / "docs/PC_CAUSAL_CP2_VELOCITY_CROSSTAB_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_causal_cp2_velocity_crosstab_20261103_DEV01.json"
MODES = ("causal_rgb", "zero_velocity", "privileged_qvel")


def main():
    if OUTPUT.exists():
        raise FileExistsError("refusing overwrite")
    for path, digest in ((SOURCE, SOURCE_SHA), (OBSTACLE, OBSTACLE_SHA),
                         (CAMERA, CAMERA_SHA), (OLD, OLD_SHA),
                         (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA)):
        if sha256_file(path) != digest:
            raise ValueError(f"input changed: {path}")
    source, obstacle, camera, old = (json.loads(p.read_text()) for p in (SOURCE, OBSTACLE, CAMERA, OLD))
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                         (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError(f"input changed: {path}")
    if [r["root"] for r in source["rows"]] != list(ROOTS):
        raise ValueError("root roster changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    started, rows = time.monotonic(), []
    print("Compare fixed-prefix predictions under causal, zero and offline privileged qvel inputs.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as arrays, \
            np.load(OBSTACLE.with_suffix(".npz"), allow_pickle=False) as src, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("checkpoint model changed")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalization mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalization")
        weight, _, _ = get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_state, _ = estimate_state(src["goal_rgb"], metadata, offset)
        with torch.no_grad():
            goal = encode_with_state(model, src["goal_rgb"], goal_state, device)["proprio"].reshape(256, 16)
            for k, root in enumerate(ROOTS):
                histories = [arrays[f"r{root}_p{i}_history"] for i in range(9)]
                if any(h.shape != (13, 224, 224, 3) for h in histories):
                    raise ValueError("invalid RGB histories")
                if any(not np.array_equal(h[:3], histories[0][:3]) for h in histories[1:]):
                    raise ValueError("prelude frames differ")
                bank = arrays[f"r{root}_bank"]
                cp2 = arrays["cp2_eval_only"][k*9:k*9+9]
                if bank.shape != (9, 5, 2) or cp2.shape != (9, 4):
                    raise ValueError("invalid action bank or cp2 state")
                if not np.allclose(cp2, cp2[:1], rtol=0, atol=1e-12):
                    raise ValueError("cp2 reset differs across branches")
                causal = causal_state(histories[0][1], histories[0][2], metadata, offset, .05)
                np.testing.assert_allclose(causal, source["rows"][k]["initial_rgb_state_after_prelude"],
                                           rtol=0, atol=1e-7)
                zero, privileged = causal.copy(), causal.copy()
                zero[2:] = 0.
                privileged[2:] = cp2[0, 2:]
                inputs = {"causal_rgb": causal, "zero_velocity": zero, "privileged_qvel": privileged}
                predictions = {}
                for mode, state in inputs.items():
                    context = encode_with_state(model, histories[0][2], state, device)
                    scores = []
                    for action in bank:
                        packet = ((torch.as_tensor(action, device=device) - mean) / std).reshape(1, 1, 10)
                        prediction = model.unroll(context, act_suffix=packet)["proprio"]
                        if prediction.shape[:2] != (2, 1):
                            raise ValueError("prediction clock mismatch")
                        scores.append(float(projected_cost(prediction[-1].reshape(1, 256, 16),
                                                           goal, pinv, pos_std).item()))
                    predictions[mode] = scores
                np.testing.assert_allclose(predictions["causal_rgb"], arrays[f"r{root}_first_scores"],
                                           rtol=1e-5, atol=1e-6)
                actual = []
                for history in histories:
                    state7 = causal_state(history[6], history[7], metadata, offset, .05)
                    encoded = encode_with_state(model, history[7], state7, device)["proprio"].reshape(1, 256, 16)
                    actual.append(float(projected_cost(encoded, goal, pinv, pos_std).item()))
                rows.append({"root": root, "input_states": {key: value.tolist() for key, value in inputs.items()},
                             "predictions": predictions, "actual_rgb_derived_scores": actual})
                print(f"root {root}: fixed causal scores reproduced, all velocity modes complete", flush=True)
        torch.cuda.synchronize()
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
        cp7 = arrays["cp7_eval_only"]
        if cp7.shape != (36, 4) or not np.isfinite(cp7).all():
            raise ValueError("invalid physical endpoint")
        for k, row in enumerate(rows):
            physical = np.linalg.norm(cp7[k*9:k*9+9, :2] - GOAL[:2], axis=1)
            np.testing.assert_allclose(physical, [b["cp7_distance"] for b in source["rows"][k]["branches"]],
                                       rtol=0, atol=1e-12)
            row["physical_distances"] = physical.tolist()
            row["ranking"] = {
                f"{mode}_vs_{name}": rank_counts(row["predictions"][mode], ref)
                for mode in MODES
                for name, ref in (("actual_rgb", row["actual_rgb_derived_scores"]), ("physical", physical))}
    report = {"schema": "pc_causal_cp2_velocity_crosstab.v1", "rows": rows,
              "scope": "exposed fixed-action diagnosis; qvel privileged, not planner input",
              "source_sha256": SOURCE_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "initial_encodes": 12, "actual_encodes": 36,
                         "prediction_samples": 108, "new_physical_steps": 0},
              "wall_seconds": time.monotonic() - started}
    with OUTPUT.open("x") as out:
        json.dump(report, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
