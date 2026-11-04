"""Match cp12 CEM from imagined rather than real-RGB context on fixed branches."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from experiments import pc_causal_history_h10_suffix_20261103 as h

SOURCES = (
    (h.ROOT / "results/pc_causal_mixed_rejected_physical_20261103_DEV01.json",
     "07ada31f9a58c3472fd84778f749c2df6cfb14db6442531954e5e7003ab587f8"),
    (h.ROOT / "results/pc_causal_mixed_prefix_search_20261103_DEV01.json",
     "f9d8f3014b44a95dfd14b0afd3d49c0b2d4b776a8112d9fa20370cb7f52fe368"),
)
REAL = (
    (h.ROOT / "results/pc_causal_mixed_cp12_replan_20261103_DEV01.json",
     "882cf9d46da6901cf8ca1c7f70ed942ea22349e01a03d380beccf9f5b50e14b9"),
    (h.ROOT / "results/pc_causal_mixed_cp12_controls_20261103_DEV01.json",
     "f03d384a7e18a345219a8e151d12d31e62e1ac0f57461671887656d99e566491"),
)
PROTOCOL = h.ROOT.parent / "docs/PC_CAUSAL_MIXED_CP12_IMAGINED_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_causal_mixed_cp12_imagined_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def last_context(prediction, device):
    fields = {}
    for name in ("visual", "proprio"):
        value = prediction[name]
        if value.shape[:2] != (2, 1) or not torch.isfinite(value).all():
            raise ValueError("invalid imagined context")
        fields[name] = value[-1:]
    return TensorDict(fields, batch_size=[1, 1], device=device)


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in (*SOURCES, *REAL, (h.OBSTACLE, h.OBSTACLE_SHA),
                           (h.CAMERA, h.CAMERA_SHA), (h.OLD, h.OLD_SHA),
                           (h.DEFAULT_DATASET / "seq_lengths.pth", h.LENGTHS_SHA)):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    source_docs = [json.loads(p.read_text()) for p, _ in SOURCES]
    real_docs = [json.loads(p.read_text()) for p, _ in REAL]
    obstacle, camera, old = (json.loads(p.read_text()) for p in (h.OBSTACLE, h.CAMERA, h.OLD))
    for (path, _), doc in zip(SOURCES + REAL, source_docs + real_docs):
        if h.sha256_file(path.with_suffix(".npz")) != doc["arrays_sha256"]:
            raise ValueError("source arrays changed")
    if (h.sha256_file(h.OBSTACLE.with_suffix(".npz")) != obstacle["arrays_sha256"]
            or h.sha256_file(h.DEFAULT_CHECKPOINT) != source_docs[0]["checkpoint_sha256"]):
        raise ValueError("checkpoint or obstacle array changed")
    if any([r["root"] for r in doc["rows"]] != list(h.ROOTS) for doc in source_docs + real_docs):
        raise ValueError("invalid root roster")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences = {}, [], [], []
    started = time.monotonic()
    print("Compare imagined versus actual cp12 context with the same physical actions and CEM budget.", flush=True)
    with np.load(SOURCES[0][0].with_suffix(".npz"), allow_pickle=False) as candidates, \
            np.load(SOURCES[1][0].with_suffix(".npz"), allow_pickle=False) as controls, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            h.lock.gpu_lock(h.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if any(before != doc["model_before"] for doc in source_docs + real_docs):
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
        for k, root in enumerate(h.ROOTS):
            branches = []
            for group, saved, per_root in ((0, candidates, 8), (1, controls, 2)):
                source_row, real_row = source_docs[group]["rows"][k], real_docs[group]["rows"][k]
                for i in range(per_root):
                    orig_branch, real_branch = source_row["branches"][i], real_row["branches"][i]
                    if group == 0:
                        arm = ("raw", "spectral").index(orig_branch["arm"])
                        name = f"r{root}_a{arm}_p{orig_branch['candidate']}"
                        identity = {"arm": orig_branch["arm"], "candidate": orig_branch["candidate"]}
                        if (real_branch["arm"], real_branch["candidate"]) != (orig_branch["arm"], orig_branch["candidate"]):
                            raise ValueError("real comparison binding mismatch")
                    else:
                        label = ("zero", "greedy")[i]
                        name, identity = f"r{root}_{label}", {"name": label}
                        if real_branch["name"] != label:
                            raise ValueError("real control binding mismatch")
                    original, frames = saved[f"{name}_actions"], saved[f"{name}_history"]
                    initial_index = (8*k+i) if group == 0 else (4*k+i)
                    initial = saved["starts"][initial_index].astype(np.float64)
                    if original.shape != (17, 2) or frames.shape != (18, 224, 224, 3) or np.any(np.abs(original) > 1):
                        raise ValueError("invalid source trajectory")
                    state7 = h.causal_state(frames[6], frames[7], metadata, offset, .05)
                    initial_context = h.encode_with_state(model, frames[7], state7, device)
                    first_packet = ((torch.as_tensor(original[7:12], device=device)-mean)/std).reshape(1, 1, 10)
                    with torch.no_grad():
                        open_prediction = model.unroll(initial_context, act_suffix=first_packet)
                        if open_prediction["proprio"].shape[:2] != (2, 1):
                            raise ValueError("imagined clock mismatch")
                        context = last_context(open_prediction, device)
                    with h.NativeRGBBridge(initial) as env:
                        history = [env.initial_rgb]
                        for action in original[:12]:
                            history.append(env.step(action[None]))
                        if not np.array_equal(np.stack(history), frames[:13]):
                            raise RuntimeError("cp12 physical RGB replay mismatch")
                        prior = ((torch.as_tensor(original[12:], device=device)-mean)/std).reshape(1, 1, 10)
                        batches, count = [], 0

                        def evaluate(proposals):
                            nonlocal count
                            n = proposals.shape[1]
                            with torch.no_grad():
                                prediction = model.unroll(context, act_suffix=proposals[0, :, 0].reshape(1, n, 10))["proprio"]
                                if prediction.shape[:2] != (2, n):
                                    raise ValueError("feedback clock mismatch")
                                score = h.projected_cost(prediction[1].reshape(n, 256, 16), target, pinv, pos_std)
                            count += n
                            batches.append((proposals.detach().cpu().numpy(), score.cpu().numpy()))
                            return score.unsqueeze(0)

                        rng = torch.Generator(device=device).manual_seed(2026110410 + root)
                        chosen, prior_score, refined_score, refined = h.packet_cem(evaluate, prior, lower, upper, rng)
                        if count != 50 or len(batches) != 4:
                            raise RuntimeError("CEM budget mismatch")
                        tail = np.clip((chosen.reshape(5, 2)*std+mean).cpu().numpy(), -1, 1).astype(np.float32)
                        for action in tail:
                            history.append(env.step(action[None]))
                        if env.step_count != 17:
                            raise RuntimeError("physical clock mismatch")
                    actions = np.concatenate((original[:12], tail))
                    arrays[f"{name}_actions"], arrays[f"{name}_history"] = actions, np.stack(history)
                    arrays[f"{name}_normalized_tail"] = chosen.cpu().numpy()
                    arrays[f"{name}_imagined_proprio"] = context["proprio"].detach().cpu().numpy()
                    for j, (p, s) in enumerate(batches):
                        arrays[f"{name}_feedback{j}_candidates"] = p
                        arrays[f"{name}_feedback{j}_scores"] = s
                    starts.append(initial)
                    sequences.append(actions)
                    branches.append({**identity, "prior_score": prior_score, "refined_score": refined_score,
                                     "refined": refined, "same_tail": bool(np.array_equal(tail, original[12:])),
                                     "original_cp17_distance": orig_branch["cp17_distance"],
                                     "real_refresh_cp17_distance": real_branch["cp17_distance"]})
            rows.append({"root": root, "branches": branches})
            print(f"root {root}: 8 candidates and 2 controls completed", flush=True)
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp17 = h.physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"], arrays["cp17_eval_only"] = starts, sequences, cp17
    for k, row in enumerate(rows):
        for i, branch in enumerate(row["branches"]):
            branch["imagined_cp17_distance"] = float(np.linalg.norm(cp17[10*k+i, :2] - h.GOAL[:2]))
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_mixed_cp12_imagined.v1", "rows": rows,
              "scope": "exposed development; matched suffix budget, unequal context computation and no independent claim",
              "source_sha256": [d for _, d in SOURCES], "real_sha256": [d for _, d in REAL],
              "arrays_sha256": h.sha256_file(ARRAYS), "protocol_sha256": h.sha256_file(PROTOCOL),
              "script_sha256": h.sha256_file(Path(__file__)), "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT),
              "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "cp7_encodes": 40, "imagined_one_packet_samples": 40,
                         "feedback_one_packet_samples": 2000, "rgb_backend_steps": 680,
                         "position_backend_replay_steps": 680},
              "inherited_upstream_cost": [d["counts"] for d in source_docs + real_docs],
              "wall_seconds": time.monotonic() - started}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
