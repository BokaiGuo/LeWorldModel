"""Compare saved joint-plan carry with zero-initialized real-RGB suffix CEM."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h

SOURCE = h.ROOT / "results/pc_causal_joint_prefix_gradient_20261103_DEV01.json"
SOURCE_SHA = "931c7e09fd95617c9a3ce37549e561f7510df311b4b369acf66ad9ca3a21a123"
PROTOCOL = h.ROOT.parent / "docs/PC_CAUSAL_JOINT_CARRY_TAIL_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_causal_joint_carry_tail_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ROOT_ID = 8349


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
    root_row = source["rows"][list(h.ROOTS).index(ROOT_ID)]
    if root_row["root"] != ROOT_ID or root_row["decisions"][0]["qualified"] is None:
        raise ValueError("qualified source missing")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences = {}, [], [], []
    begun = time.monotonic()
    print("Re-score and compare old joint tails against equal-budget zero/carry feedback CEM.", flush=True)
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
            raise ValueError("invalid action normalization")
        lower, upper = ((-torch.ones_like(mean)-mean)/std).repeat(10), ((torch.ones_like(mean)-mean)/std).repeat(10)
        zero_prior = ((torch.zeros_like(mean)-mean)/std).repeat(10).reshape(1, 1, 20)
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        root_index = list(h.ROOTS).index(ROOT_ID)
        idx = sum(len(r["branches"]) for r in source["rows"][:root_index])
        initial = saved["starts"][idx].astype(np.float64)
        plans = saved[f"r{ROOT_ID}_plans"]
        positions = [(0, root_row["control_iterations"][0]), (1, root_row["control_iterations"][1]),
                     tuple(root_row["decisions"][0]["qualified"])]
        for label, (candidate, iteration) in zip(("zero", "greedy", "raw_qualified"), positions):
            src_actions, src_history = saved[f"r{ROOT_ID}_{label}_actions"], saved[f"r{ROOT_ID}_{label}_history"]
            plan = plans[iteration, candidate]
            if not np.array_equal(plan[:5], src_actions[2:7]) or np.any(np.abs(plan) > 1):
                raise ValueError("joint plan binding mismatch")
            state7 = h.causal_state(src_history[6], src_history[7], metadata, offset, .05)
            context = h.encode_with_state(model, src_history[7], state7, device)
            original_prior = ((torch.as_tensor(plan[5:], device=device)-mean)/std).reshape(1, 1, 20)
            generated = {}
            for mode, prior in (("zero_init", zero_prior), ("carry_init", original_prior)):
                batches, count = [], 0

                def evaluate(proposals):
                    nonlocal count
                    n = proposals.shape[1]
                    actions = proposals[0, :, 0].reshape(n, 2, 10).permute(1, 0, 2)
                    with torch.no_grad():
                        pred = model.unroll(context, act_suffix=actions)["proprio"]
                        if pred.shape[:2] != (3, n):
                            raise ValueError("feedback clock mismatch")
                        scores = h.projected_cost(pred[2].reshape(n, 256, 16), target, pinv, pos_std)
                    count += n
                    batches.append((proposals.detach().cpu().numpy(), scores.cpu().numpy()))
                    return scores.unsqueeze(0)

                rng = torch.Generator(device=device).manual_seed(2026110380 + ROOT_ID)
                chosen, score0, score1, refined = h.packet_cem(evaluate, prior, lower, upper, rng)
                if count != 50 or len(batches) != 4:
                    raise RuntimeError("CEM budget mismatch")
                generated[mode] = (np.clip((chosen.reshape(10, 2)*std+mean).cpu().numpy(), -1, 1).astype(np.float32),
                                   score0, score1, refined)
                for j, (p, s) in enumerate(batches):
                    arrays[f"{label}_{mode}_feedback{j}_candidates"] = p
                    arrays[f"{label}_{mode}_feedback{j}_scores"] = s
            original_score = generated["carry_init"][1]
            for mode in ("old", "zero_init", "carry_init"):
                tail = plan[5:] if mode == "old" else generated[mode][0]
                actions = np.concatenate((src_actions[:7], tail))
                with h.NativeRGBBridge(initial) as env:
                    history = [env.initial_rgb]
                    for action in actions:
                        history.append(env.step(action[None]))
                    if env.step_count != 17 or not np.array_equal(np.stack(history[:8]), src_history[:8]):
                        raise RuntimeError("physical prefix mismatch")
                if mode == "zero_init" and (not np.array_equal(actions, src_actions)
                                             or not np.array_equal(np.stack(history), src_history)):
                    raise RuntimeError("zero-initialized source replay mismatch")
                name = f"{label}_{mode}"
                arrays[f"{name}_actions"], arrays[f"{name}_history"] = actions, np.stack(history)
                starts.append(initial)
                sequences.append(actions)
                rows.append({"prefix": label, "mode": mode, "old_tail_score": original_score,
                             "prior_score": original_score if mode == "old" else generated[mode][1],
                             "refined_score": original_score if mode == "old" else generated[mode][2],
                             "refined": False if mode == "old" else generated[mode][3],
                             "same_as_old": bool(np.array_equal(tail, plan[5:]))})
            print(f"prefix {label}: old score {original_score:.6g}; carry refined={generated['carry_init'][3]}", flush=True)
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp2 = h.physical.replay_position_endpoints(starts, sequences[:, :2])
    cp7 = h.physical.replay_position_endpoints(starts, sequences[:, :7])
    cp17 = h.physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"] = starts, sequences
    arrays["cp2_eval_only"], arrays["cp7_eval_only"], arrays["cp17_eval_only"] = cp2, cp7, cp17
    for i, row in enumerate(rows):
        for tag, pos in (("cp2", cp2[i]), ("cp7", cp7[i]), ("cp17", cp17[i])):
            row[f"{tag}_distance"] = float(np.linalg.norm(pos[:2]-h.GOAL[:2]))
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_joint_carry_tail.v1", "root": ROOT_ID, "rows": rows,
              "scope": "exposed single-root development; no independent, total-cost or mechanism claim",
              "source_sha256": SOURCE_SHA, "arrays_sha256": h.sha256_file(ARRAYS),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "cem_kernel_sha256": h.sha256_file(Path(h.packet_cem.__code__.co_filename)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "cp7_encodes": 3, "two_packet_model_samples": 300,
                         "rgb_backend_steps": 153, "position_backend_replay_steps": 234},
              "inherited_upstream_cost": source["counts"], "wall_seconds": time.monotonic()-begun}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
