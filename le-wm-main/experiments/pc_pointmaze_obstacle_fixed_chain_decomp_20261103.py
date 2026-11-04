"""Offline fixed-chain RGB/model/physical decomposition on exposed roots."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.pc_pointmaze_affine_feedback_cem_20261103 import projected_cost
from experiments.pc_pointmaze_affine_inverse_20261103 import MEAN, STD, get_encoder
from experiments.pc_pointmaze_affine_nonprogress_lookahead_20261103 import OUTPUT as LOOKAHEAD
from experiments.pc_pointmaze_affine_nonprogress_oracle_20261103 import OUTPUT as ORACLE
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import LENGTHS_SHA
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import BANK_SHA, SOURCE_SHA
from experiments.pc_pointmaze_obstacle_nonprogress_bank_20261103 import OUTPUT as BANK
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS, OUTPUT as SOURCE
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_OBSTACLE_FIXED_CHAIN_DECOMPOSITION_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_obstacle_fixed_chain_decomp_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ORACLE_SHA = "6884f717ab6fabc6fa96614c466eb168406c1360b5967041cf46025d44af8a6b"
LOOKAHEAD_SHA = "a4c1c2a7083195e4951eb26ed559d90af583d6b414eb06125d628c0e9ea17f71"


def rank_counts(left, right):
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != (9,) or b.shape != (9,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nine finite paired outcomes required")
    i, j = np.triu_indices(9, 1)
    x, y = a[i] - a[j], b[i] - b[j]
    return {"pairs": len(i), "reversed": int((x * y < 0).sum()),
            "left_ties": int((x == 0).sum()), "right_ties": int((y == 0).sum())}


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    checked = ((ORACLE, ORACLE_SHA), (LOOKAHEAD, LOOKAHEAD_SHA), (BANK, BANK_SHA),
               (SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA),
               (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA))
    for path, digest in checked:
        if sha256_file(path) != digest:
            raise ValueError(f"source changed: {path}")
    oracle, lookahead, bank_doc, source, camera, old = (
        json.loads(p.read_text()) for p in (ORACLE, LOOKAHEAD, BANK, SOURCE, CAMERA, OLD))
    for path, digest in ((ORACLE.with_suffix(".npz"), oracle["arrays_sha256"]),
                         (LOOKAHEAD.with_suffix(".npz"), lookahead["arrays_sha256"]),
                         (BANK.with_suffix(".npz"), bank_doc["arrays_sha256"]),
                         (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError(f"source changed: {path}")
    expected = [(r, i) for r in ROOTS for i in range(9)]
    if [(r["root"], r["bank_index"]) for r in oracle["rows"]] != expected:
        raise ValueError("oracle roster mismatch")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    begun, rows, images = time.monotonic(), [], []
    print("Replay fixed chains and decompose open versus RGB-refreshed predictions.", flush=True)
    with np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as data, \
            np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as src, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        starts, sequences = data["starts"].copy(), data["sequences"].copy()
        saved_cp10 = data["cp10"].copy()
        if starts.shape != (36, 4) or sequences.shape != (36, 10, 2) or saved_cp10.shape[0] != 36:
            raise ValueError("unexpected fixed chain shapes")
        if not np.isfinite(starts).all() or not np.isfinite(sequences).all() or np.any(np.abs(sequences) > 1):
            raise ValueError("invalid fixed chains")
        replay_cp10 = physical.replay_position_endpoints(starts, sequences)
        np.testing.assert_allclose(replay_cp10, saved_cp10, rtol=0, atol=1e-12)
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != lookahead["model_before"]:
            raise ValueError("model parameter hash changed")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalizer")
        weight, _, _ = get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = src["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, metadata, offset)
        with torch.no_grad():
            target = encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
            for k, (root, idx) in enumerate(expected):
                with NativeRGBBridge(starts[k]) as env:
                    history = [env.initial_rgb]
                    if not np.array_equal(history[0], src[f"r{root}_random_frames"][0]):
                        raise RuntimeError("initial RGB mismatch")
                    for action in sequences[k]:
                        history.append(env.step(action[None]))
                    if not np.array_equal(history[5], bank[f"r{root}_rgb5"][idx]):
                        raise RuntimeError("cp5 RGB mismatch")
                    if env.step_count != 10:
                        raise RuntimeError("action clock mismatch")
                state0, _ = estimate_state(history[0], metadata, offset)
                state5 = causal_state(history[4], history[5], metadata, offset, .05)
                state10 = causal_state(history[9], history[10], metadata, offset, .05)
                c0 = encode_with_state(model, history[0], state0, device)
                c5 = encode_with_state(model, history[5], state5, device)
                c10 = encode_with_state(model, history[10], state10, device)
                packets = ((torch.as_tensor(sequences[k], device=device) - mean) / std).reshape(2, 1, 10)
                open_prediction = model.unroll(c0, act_suffix=packets)["proprio"]
                refreshed = model.unroll(c5, act_suffix=packets[1:])["proprio"]
                if open_prediction.shape[:2] != (3, 1) or refreshed.shape[:2] != (2, 1):
                    raise RuntimeError("prediction clock mismatch")
                first = open_prediction[1].reshape(1, 256, 16)
                opened = open_prediction[2].reshape(1, 256, 16)
                fresh = refreshed[1].reshape(1, 256, 16)
                actual = c10["proprio"].reshape(1, 256, 16)
                score = lambda x: float(projected_cost(x, target, pinv, pos_std).item())
                row = {"root": root, "bank_index": idx, "first_score": score(first),
                       "open_score": score(opened), "refresh_score": score(fresh),
                       "actual_rgb_score": score(actual),
                       "open_proprio_mse": float((opened - actual).square().mean().item()),
                       "refresh_proprio_mse": float((fresh - actual).square().mean().item()),
                       "physical_distance": float(np.linalg.norm(saved_cp10[k, :2] - GOAL[:2]))}
                if not np.isclose(row["open_score"], oracle["rows"][k]["model_terminal_score"],
                                  rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"stored open score mismatch: {root}/{idx}")
                if not all(np.isfinite(v) for v in row.values()):
                    raise ValueError("nonfinite output")
                rows.append(row)
                images.append(np.stack((history[0], history[4], history[5], history[9], history[10])))
                print(f"root {root} candidate {idx}: fixed-chain replay complete", flush=True)
        torch.cuda.synchronize()
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    rankings = {}
    for root in ROOTS:
        group = [r for r in rows if r["root"] == root]
        rankings[str(root)] = {
            f"{left}_vs_{right}": rank_counts([r[left] for r in group], [r[right] for r in group])
            for left, right in (("open_score", "actual_rgb_score"), ("refresh_score", "actual_rgb_score"),
                                ("open_score", "physical_distance"), ("refresh_score", "physical_distance"),
                                ("actual_rgb_score", "physical_distance"))}
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, frames=np.stack(images), starts=starts, sequences=sequences,
                            cp10_eval_only=saved_cp10)
    report = {"schema": "pc_fixed_obstacle_decomposition.v1", "rows": rows, "rankings": rankings,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "oracle_sha256": ORACLE_SHA, "lookahead_sha256": LOOKAHEAD_SHA,
              "arrays_sha256": sha256_file(ARRAYS), "checkpoint_sha256": sha256_file(DEFAULT_CHECKPOINT),
              "config_sha256": sha256_file(DEFAULT_CONFIG), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "state_encodes": 108, "open_predictions": 36,
                         "refresh_predictions": 36, "rgb_physical_steps": 360,
                         "position_backend_replay_steps": 360},
              "wall_seconds": time.monotonic() - begun,
              "scope": "exposed fixed-action offline diagnosis; no selection or independent admission"}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
