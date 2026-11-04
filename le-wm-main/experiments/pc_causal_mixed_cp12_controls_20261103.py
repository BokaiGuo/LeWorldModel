"""Apply matching cp12 real-RGB replanning to saved zero and greedy controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h

SOURCE = h.ROOT / "results/pc_causal_mixed_prefix_search_20261103_DEV01.json"
SOURCE_SHA = "f9d8f3014b44a95dfd14b0afd3d49c0b2d4b776a8112d9fa20370cb7f52fe368"
PROTOCOL = h.ROOT.parent / "docs/PC_CAUSAL_MIXED_CP12_CONTROLS_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_causal_mixed_cp12_controls_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


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
    arrays, rows, starts, sequences = {}, [], [], []
    started = time.monotonic()
    print("Give zero and greedy controls matching cp12 RGB feedback and suffix budget.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
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
        lower, upper = ((-torch.ones_like(mean)-mean)/std).repeat(5), ((torch.ones_like(mean)-mean)/std).repeat(5)
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for k, row in enumerate(source["rows"]):
            root, branches = row["root"], []
            for i, label in enumerate(("zero", "greedy")):
                name = f"r{root}_{label}"
                original, frames = saved[f"{name}_actions"], saved[f"{name}_history"]
                initial = saved["starts"][4*k+i].astype(np.float64)
                if (original.shape != (17, 2) or frames.shape != (18, 224, 224, 3)
                        or not np.isfinite(original).all() or np.any(np.abs(original) > 1)):
                    raise ValueError("invalid stored trajectory")
                with h.NativeRGBBridge(initial) as env:
                    history = [env.initial_rgb]
                    for action in original[:12]:
                        history.append(env.step(action[None]))
                    if not np.array_equal(np.stack(history), frames[:13]):
                        raise RuntimeError("cp12 physical RGB replay mismatch")
                    state12 = h.causal_state(history[11], history[12], metadata, offset, .05)
                    context = h.encode_with_state(model, history[12], state12, device)
                    prior = ((torch.as_tensor(original[12:], device=device) - mean) / std).reshape(1, 1, 10)
                    batches, count = [], 0

                    def evaluate(proposals):
                        nonlocal count
                        n = proposals.shape[1]
                        with torch.no_grad():
                            pred = model.unroll(context, act_suffix=proposals[0, :, 0].reshape(1, n, 10))["proprio"]
                            if pred.shape[:2] != (2, n):
                                raise ValueError("feedback clock mismatch")
                            score = h.projected_cost(pred[1].reshape(n, 256, 16), target, pinv, pos_std)
                        count += n
                        batches.append((proposals.detach().cpu().numpy(), score.cpu().numpy()))
                        return score.unsqueeze(0)

                    rng = torch.Generator(device=device).manual_seed(2026110410 + root)
                    chosen, prior_score, refined_score, refined = h.packet_cem(
                        evaluate, prior, lower, upper, rng)
                    if count != 50 or len(batches) != 4:
                        raise RuntimeError("CEM budget mismatch")
                    tail = np.clip((chosen.reshape(5, 2) * std + mean).cpu().numpy(), -1, 1).astype(np.float32)
                    for action in tail:
                        history.append(env.step(action[None]))
                    if env.step_count != 17:
                        raise RuntimeError("physical clock mismatch")
                state17 = h.causal_state(history[16], history[17], metadata, offset, .05)
                actual = h.encode_with_state(model, history[17], state17, device)
                actual_score = float(h.projected_cost(actual["proprio"].reshape(1, 256, 16),
                                                       target, pinv, pos_std).item())
                actions = np.concatenate((original[:12], tail))
                arrays[f"{name}_actions"], arrays[f"{name}_history"] = actions, np.stack(history)
                arrays[f"{name}_normalized_tail"] = chosen.cpu().numpy()
                for j, (p, s) in enumerate(batches):
                    arrays[f"{name}_feedback{j}_candidates"] = p
                    arrays[f"{name}_feedback{j}_scores"] = s
                starts.append(initial)
                sequences.append(actions)
                branches.append({"name": label, "prior_score": prior_score, "refined_score": refined_score,
                                 "refined": refined, "same_tail": bool(np.array_equal(tail, original[12:])),
                                 "actual_cp17_score": actual_score, "old_cp17_distance": row["branches"][i]["cp17_distance"]})
            rows.append({"root": root, "branches": branches})
            print(f"root {root}: matched zero and greedy branches completed", flush=True)
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp17 = h.physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"], arrays["cp17_eval_only"] = starts, sequences, cp17
    for k, row in enumerate(rows):
        for i, branch in enumerate(row["branches"]):
            branch["cp17_distance"] = float(np.linalg.norm(cp17[2*k+i, :2] - h.GOAL[:2]))
            branch["distance_delta_new_minus_old"] = branch["cp17_distance"] - branch["old_cp17_distance"]
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_mixed_cp12_controls.v1", "rows": rows,
              "scope": "exposed development; matching cp12 feedback, upstream budgets unequal",
              "source_sha256": SOURCE_SHA, "arrays_sha256": h.sha256_file(ARRAYS),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "cem_kernel_sha256": h.sha256_file(Path(h.packet_cem.__code__.co_filename)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "cp12_encodes": 8, "cp17_encodes": 8,
                         "feedback_one_packet_samples": 400, "rgb_backend_steps": 136,
                         "position_backend_replay_steps": 136},
              "inherited_upstream_cost": source["counts"], "wall_seconds": time.monotonic() - started}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
