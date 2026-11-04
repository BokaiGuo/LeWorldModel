"""Physically test all sealed rejected mixed proposals on exposed roots."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h
SOURCE = h.ROOT / "results/pc_causal_mixed_prefix_search_20261103_DEV01.json"
SOURCE_SHA = "f9d8f3014b44a95dfd14b0afd3d49c0b2d4b776a8112d9fa20370cb7f52fe368"
PROTOCOL = h.ROOT.parent / "docs/PC_CAUSAL_MIXED_REJECTED_PHYSICAL_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_causal_mixed_rejected_physical_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def rank_counts(left, right):
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != (8,) or b.shape != (8,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("eight finite paired outcomes required")
    i, j = np.triu_indices(8, 1)
    x, y = a[i] - a[j], b[i] - b[j]
    return {"pairs": len(i), "reversed": int((x * y < 0).sum()),
            "left_ties": int((x == 0).sum()), "right_ties": int((y == 0).sum())}


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (h.SOURCE, h.SOURCE_SHA),
                           (h.OBSTACLE, h.OBSTACLE_SHA), (h.CAMERA, h.CAMERA_SHA),
                           (h.OLD, h.OLD_SHA),
                           (h.DEFAULT_DATASET / "seq_lengths.pth", h.LENGTHS_SHA)):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    source, oldsource, obstacle, camera, old = (json.loads(p.read_text())
                                               for p in (SOURCE, h.SOURCE, h.OBSTACLE, h.CAMERA, h.OLD))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (h.SOURCE.with_suffix(".npz"), oldsource["arrays_sha256"]),
                           (h.OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                           (h.DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    if [r["root"] for r in source["rows"]] != list(h.ROOTS):
        raise ValueError("root roster mismatch")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences = {}, [], [], []
    begun = time.monotonic()
    print("Execute all sealed mixed proposals and test physical consequences of rejected actions.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(h.SOURCE.with_suffix(".npz"), allow_pickle=False) as previous, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            h.lock.gpu_lock(h.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("model hash mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalization")
        lower, upper = ((-torch.ones_like(mean)-mean)/std).repeat(10), ((torch.ones_like(mean)-mean)/std).repeat(10)
        prior = ((torch.zeros_like(mean)-mean)/std).repeat(10).reshape(1, 1, 20)
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for k, root in enumerate(h.ROOTS):
            initial = bank["starts"][4*k].astype(np.float64)
            prelude = previous[f"r{root}_p0_history"][:3]
            banks = bank[f"r{root}_banks"]
            if banks.shape != (2, 6, 5, 2) or np.any(np.abs(banks) > 1) or not np.isfinite(banks).all():
                raise ValueError("invalid fixed bank")
            branches = []
            for arm in range(2):
                for candidate in range(2, 6):
                    name = f"r{root}_a{arm}_p{candidate}"
                    prefix = banks[arm, candidate]
                    with h.NativeRGBBridge(initial) as env:
                        history = [env.initial_rgb]
                        for action in np.concatenate((np.zeros((2, 2), np.float32), prefix)):
                            history.append(env.step(action[None]))
                        if not np.array_equal(np.stack(history[:3]), prelude):
                            raise RuntimeError("prelude replay mismatch")
                        state7 = h.causal_state(history[6], history[7], metadata, offset, .05)
                        context = h.encode_with_state(model, history[7], state7, device)
                        actual7 = float(h.projected_cost(context["proprio"].reshape(1, 256, 16),
                                                        target, pinv, pos_std).item())
                        batches, count = [], 0

                        def evaluate(proposals):
                            nonlocal count
                            n = proposals.shape[1]
                            tail = proposals[0, :, 0].reshape(n, 2, 10).permute(1, 0, 2)
                            with torch.no_grad():
                                pred = model.unroll(context, act_suffix=tail)["proprio"]
                                if pred.shape[:2] != (3, n):
                                    raise ValueError("feedback clock mismatch")
                                score = h.projected_cost(pred[2].reshape(n, 256, 16), target, pinv, pos_std)
                            count += n
                            batches.append((proposals.detach().cpu().numpy(), score.cpu().numpy()))
                            return score.unsqueeze(0)

                        generator = torch.Generator(device=device).manual_seed(2026110380 + root)
                        chosen, prior_score, refined_score, refined = h.packet_cem(
                            evaluate, prior, lower, upper, generator)
                        if count != 50 or len(batches) != 4:
                            raise RuntimeError("feedback budget mismatch")
                        tail = np.clip((chosen.reshape(10, 2) * std + mean).cpu().numpy(), -1, 1).astype(np.float32)
                        for action in tail:
                            history.append(env.step(action[None]))
                        if env.step_count != 17:
                            raise RuntimeError("physical clock mismatch")
                    state17 = h.causal_state(history[16], history[17], metadata, offset, .05)
                    final = h.encode_with_state(model, history[17], state17, device)
                    actual17 = float(h.projected_cost(final["proprio"].reshape(1, 256, 16),
                                                      target, pinv, pos_std).item())
                    actions = np.concatenate((np.zeros((2, 2), np.float32), prefix, tail))
                    arrays[f"{name}_actions"], arrays[f"{name}_history"] = actions, np.stack(history)
                    arrays[f"{name}_normalized_tail"] = chosen.cpu().numpy()
                    for j, (p, s) in enumerate(batches):
                        arrays[f"{name}_feedback{j}_candidates"] = p
                        arrays[f"{name}_feedback{j}_scores"] = s
                    starts.append(initial)
                    sequences.append(actions)
                    branches.append({"arm": ("raw", "spectral")[arm], "candidate": candidate,
                                     "predicted_cp7_score": float(bank[f"r{root}_first_scores"][arm, candidate]),
                                     "actual_cp7_rgb_score": actual7, "actual_cp17_rgb_score": actual17,
                                     "feedback_prior_score": prior_score, "feedback_refined_score": refined_score,
                                     "feedback_refined": refined})
            rows.append({"root": root, "branches": branches})
            print(f"root {root}: executed all 8 sealed proposals", flush=True)
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
    for k, row in enumerate(rows):
        reference = {b["name"]: b for b in source["rows"][k]["branches"]}
        for i, branch in enumerate(row["branches"]):
            for tag, position in (("cp2", cp2[8*k+i]), ("cp7", cp7[8*k+i]), ("cp17", cp17[8*k+i])):
                branch[f"{tag}_distance"] = float(np.linalg.norm(position[:2] - h.GOAL[:2]))
            branch["physical_nonprogress_cp7"] = branch["cp7_distance"] >= branch["cp2_distance"]
            branch["outcome_exposed_beats_zero_and_greedy"] = (
                branch["cp17_distance"] < reference["zero"]["cp17_distance"]
                and branch["cp17_distance"] < reference["greedy"]["cp17_distance"])
        row["cp7_ranking"] = {
            "predicted_vs_actual_rgb": rank_counts([b["predicted_cp7_score"] for b in row["branches"]],
                                                  [b["actual_cp7_rgb_score"] for b in row["branches"]]),
            "predicted_vs_physical": rank_counts([b["predicted_cp7_score"] for b in row["branches"]],
                                                 [b["cp7_distance"] for b in row["branches"]])}
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_mixed_rejected_physical.v1", "rows": rows,
              "scope": "outcome-exposed diagnostics, not model selection or independent evidence",
              "source_sha256": SOURCE_SHA, "arrays_sha256": h.sha256_file(ARRAYS),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "cem_kernel_sha256": h.sha256_file(Path(h.packet_cem.__code__.co_filename)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "actual_cp7_encodes": 32, "actual_cp17_encodes": 32,
                         "feedback_two_packet_samples": 1600, "physical_rgb_steps": 544,
                         "position_backend_replay_steps": 832},
              "inherited_upstream_cost": source["counts"],
              "wall_seconds": time.monotonic() - begun}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
