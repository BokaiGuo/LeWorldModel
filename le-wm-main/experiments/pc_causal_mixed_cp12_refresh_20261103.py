"""Evaluate open and refreshed same-action predictions on saved RGB histories."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h
from experiments.pc_causal_mixed_rejected_physical_20261103 import rank_counts

SOURCE = h.ROOT / "results/pc_causal_mixed_rejected_physical_20261103_DEV01.json"
SOURCE_SHA = "07ada31f9a58c3472fd84778f749c2df6cfb14db6442531954e5e7003ab587f8"
PROTOCOL = h.ROOT.parent / "docs/PC_CAUSAL_MIXED_CP12_REFRESH_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_causal_mixed_cp12_refresh_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def mse(left, right):
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("invalid comparison tensors")
    return float(torch.mean((left - right).square()).item())


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (h.OBSTACLE, h.OBSTACLE_SHA),
                           (h.CAMERA, h.CAMERA_SHA), (h.OLD, h.OLD_SHA),
                           (h.DEFAULT_DATASET / "seq_lengths.pth", h.LENGTHS_SHA)):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    source, obstacle, camera, old = (json.loads(p.read_text()) for p in (SOURCE, h.OBSTACLE, h.CAMERA, h.OLD))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (h.OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                           (h.DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    if [r["root"] for r in source["rows"]] != list(h.ROOTS):
        raise ValueError("invalid root roster")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    arrays, rows = {}, []
    started = time.monotonic()
    print("Compare identical executed tails with open and real-RGB-refreshed predictions.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as old_arrays, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            h.lock.gpu_lock(h.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("frozen model mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalizer")
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for row in source["rows"]:
            root, branches = row["root"], []
            for branch in row["branches"]:
                arm = ("raw", "spectral").index(branch["arm"])
                name = f"r{root}_a{arm}_p{branch['candidate']}"
                frames, actions = old_arrays[f"{name}_history"], old_arrays[f"{name}_actions"]
                if frames.shape != (18, 224, 224, 3) or actions.shape != (17, 2) or np.any(np.abs(actions) > 1):
                    raise ValueError("invalid saved trajectory")
                encoded = []
                for t in (7, 12, 17):
                    state = h.causal_state(frames[t-1], frames[t], metadata, offset, .05)
                    encoded.append(h.encode_with_state(model, frames[t], state, device))
                norm = (torch.as_tensor(actions[7:17], device=device) - mean) / std
                with torch.no_grad():
                    opened = model.unroll(encoded[0], act_suffix=norm.reshape(2, 1, 10))
                    refreshed = model.unroll(encoded[1], act_suffix=norm[5:].reshape(1, 1, 10))
                if opened["proprio"].shape[:2] != (3, 1) or refreshed["proprio"].shape[:2] != (2, 1):
                    raise ValueError("prediction clock mismatch")
                open_p, refresh_p = opened["proprio"][-1], refreshed["proprio"][-1]
                open_v, refresh_v = opened["visual"][-1], refreshed["visual"][-1]
                actual_p, actual_v = encoded[2]["proprio"][0], encoded[2]["visual"][0]
                scores = [float(h.projected_cost(v.reshape(1, 256, 16), target, pinv, pos_std).item())
                          for v in (open_p, refresh_p, actual_p)]
                prior_score = min(branch["feedback_prior_score"], branch["feedback_refined_score"])
                np.testing.assert_allclose(scores[0], prior_score, rtol=3e-5, atol=1e-6)
                np.testing.assert_allclose(scores[2], branch["actual_cp17_rgb_score"], rtol=3e-5, atol=1e-6)
                for label, value in (("open_visual", open_v), ("refresh_visual", refresh_v),
                                     ("actual_visual", actual_v), ("open_proprio", open_p),
                                     ("refresh_proprio", refresh_p), ("actual_proprio", actual_p)):
                    arrays[f"{name}_{label}"] = value.detach().cpu().numpy()
                branches.append({"arm": branch["arm"], "candidate": branch["candidate"],
                                 "open_score": scores[0], "refresh_score": scores[1], "actual_score": scores[2],
                                 "physical_distance": branch["cp17_distance"],
                                 "open_visual_mse": mse(open_v, actual_v),
                                 "refresh_visual_mse": mse(refresh_v, actual_v),
                                 "open_proprio_mse": mse(open_p, actual_p),
                                 "refresh_proprio_mse": mse(refresh_p, actual_p)})
            actual = [b["actual_score"] for b in branches]
            physical = [b["physical_distance"] for b in branches]
            comparisons = {}
            for mode in ("open", "refresh"):
                predicted = [b[f"{mode}_score"] for b in branches]
                comparisons[f"{mode}_vs_rgb"] = rank_counts(predicted, actual)
                comparisons[f"{mode}_vs_physical"] = rank_counts(predicted, physical)
            rows.append({"root": root, "branches": branches, "rankings": comparisons})
            print(f"root {root}: completed 8 matched tail comparisons", flush=True)
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_mixed_cp12_refresh.v1", "rows": rows,
              "scope": "saved exposed trajectories; no replanning or independent confirmation",
              "source_sha256": SOURCE_SHA, "arrays_sha256": h.sha256_file(ARRAYS),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "context_encodes": 96, "open_two_packet_samples": 32,
                         "refresh_one_packet_samples": 32, "physical_steps": 0},
              "wall_seconds": time.monotonic() - started}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
