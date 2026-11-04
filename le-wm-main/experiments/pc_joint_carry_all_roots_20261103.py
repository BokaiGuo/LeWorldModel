"""All-source-branch joint-tail carry versus zero-initialized feedback CEM."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h

SOURCE = h.ROOT / "results/pc_causal_joint_prefix_gradient_20261103_DEV01.json"
SOURCE_SHA = "931c7e09fd95617c9a3ce37549e561f7510df311b4b369acf66ad9ca3a21a123"
PROTOCOL = h.ROOT.parent / "docs/PC_JOINT_CARRY_ALL_ROOTS_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_joint_carry_all_roots_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ROOTS = (8343, 8345, 8349, 8351)


def indices(row, role):
    if role == "zero":
        return (0, row["control_iterations"][0])
    if role == "greedy":
        return (1, row["control_iterations"][1])
    if role not in ("raw_winner", "spectral_winner", "raw_qualified", "spectral_qualified"):
        raise ValueError("unrecognized role")
    arm, kind = role.split("_")
    pair = row["decisions"][("raw", "spectral").index(arm)][
        "unconditional" if kind == "winner" else "qualified"]
    if pair is None or len(pair) != 2:
        raise ValueError("role lacks source decision")
    return tuple(pair)


def roster(source, arrays):
    if [r["root"] for r in source["rows"]] != list(ROOTS):
        raise ValueError("unexpected source roots")
    out, offset = [], 0
    for row in source["rows"]:
        root = row["root"]
        roles = [b["name"] for b in row["branches"]]
        expected = ["zero", "greedy", "raw_winner", "spectral_winner"]
        fixed = [indices(row, role) for role in expected]
        for arm in ("raw", "spectral"):
            pair = row["decisions"][("raw", "spectral").index(arm)]["qualified"]
            if pair is not None and tuple(pair) not in fixed:
                expected.append(f"{arm}_qualified")
                fixed.append(tuple(pair))
        if roles != expected:
            raise ValueError("unexpected source branch roster")
        for branch in row["branches"]:
            role = branch["name"]
            candidate, iteration = indices(row, role)
            plan = arrays[f"r{root}_plans"][iteration, candidate]
            actions = arrays[f"r{root}_{role}_actions"]
            history = arrays[f"r{root}_{role}_history"]
            if (plan.shape != (15, 2) or actions.shape != (17, 2) or history.shape != (18, 224, 224, 3)
                    or np.any(np.abs(plan) > 1) or not np.array_equal(plan[:5], actions[2:7])):
                raise ValueError("source plan binding mismatch")
            out.append({"root": root, "role": role, "plan": plan.copy(),
                        "actions": actions.copy(), "history": history.copy(),
                        "start": arrays["starts"][offset].astype(np.float64)})
            offset += 1
    if len(out) != 17:
        raise ValueError("unexpected total branch count")
    return out


def physical_prefix(entry):
    with h.NativeRGBBridge(entry["start"]) as env:
        frames = [env.initial_rgb]
        for action in entry["actions"][:7]:
            frames.append(env.step(action[None]))
        if env.step_count != 7 or not np.array_equal(np.stack(frames), entry["history"][:8]):
            raise ValueError("source RGB preflight failed")


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
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    rows, arrays, starts, sequences = [], {}, [], []
    begun = monotonic()
    print("All 17 saved branches: verify RGB, then compare zero and own-plan carry CEM.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            h.lock.gpu_lock(h.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        entries = roster(source, saved)
        for entry in entries:
            physical_prefix(entry)
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if before != source["model_before"]:
            raise ValueError("model mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("action normalizer mismatch")
        lower, upper = ((-torch.ones_like(mean)-mean)/std).repeat(10), ((torch.ones_like(mean)-mean)/std).repeat(10)
        zero = ((torch.zeros_like(mean)-mean)/std).repeat(10).reshape(1, 1, 20)
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        planned = []
        for entry in entries:
            root, role, plan, history = entry["root"], entry["role"], entry["plan"], entry["history"]
            name = f"r{root}_{role}"
            state = h.causal_state(history[6], history[7], metadata, offset, .05)
            context = h.encode_with_state(model, history[7], state, device)
            old_prior = ((torch.as_tensor(plan[5:], device=device)-mean)/std).reshape(1, 1, 20)
            generated = {}
            for mode, prior in (("zero_init", zero), ("carry_init", old_prior)):
                batches, count = [], 0

                def evaluate(proposals):
                    nonlocal count
                    n = proposals.shape[1]
                    actions = proposals[0, :, 0].reshape(n, 2, 10).permute(1, 0, 2)
                    with torch.no_grad():
                        pred = model.unroll(context, act_suffix=actions)["proprio"]
                        if pred.shape[:2] != (3, n):
                            raise ValueError("prediction clock mismatch")
                        scores = h.projected_cost(pred[2].reshape(n, 256, 16), target, pinv, pos_std)
                    count += n
                    batches.append((proposals.detach().cpu().numpy(), scores.cpu().numpy()))
                    return scores.unsqueeze(0)

                rng = torch.Generator(device=device).manual_seed(2026110380 + root)
                chosen, prior_score, refined_score, refined = h.packet_cem(evaluate, prior, lower, upper, rng)
                if count != 50 or len(batches) != 4:
                    raise ValueError("CEM sample budget")
                raw = np.clip((chosen.reshape(10, 2)*std+mean).cpu().numpy(), -1, 1).astype(np.float32)
                generated[mode] = (raw, prior_score, refined_score, refined)
                for j, (proposals, scores) in enumerate(batches):
                    arrays[f"{name}_{mode}_batch{j}_candidates"] = proposals
                    arrays[f"{name}_{mode}_batch{j}_scores"] = scores
            planned.append((entry, generated))
            print(f"planned {name}", flush=True)
        # Only after every plan is fixed, open the physical suffix outcomes.
        for entry, generated in planned:
            root, role, plan = entry["root"], entry["role"], entry["plan"]
            old_score = generated["carry_init"][1]
            for mode in ("old", "zero_init", "carry_init"):
                tail = plan[5:] if mode == "old" else generated[mode][0]
                actions = np.concatenate((entry["actions"][:7], tail))
                with h.NativeRGBBridge(entry["start"]) as env:
                    history = [env.initial_rgb]
                    for action in actions:
                        history.append(env.step(action[None]))
                    history = np.stack(history)
                    if env.step_count != 17 or not np.array_equal(history[:8], entry["history"][:8]):
                        raise ValueError("execution RGB preflight failed")
                if mode == "zero_init" and (not np.array_equal(actions, entry["actions"])
                                             or not np.array_equal(history, entry["history"])):
                    raise ValueError("source zero-init did not replay")
                name = f"r{root}_{role}_{mode}"
                arrays[name+"_actions"], arrays[name+"_history"] = actions, history
                starts.append(entry["start"])
                sequences.append(actions)
                rows.append({"root": root, "role": role, "mode": mode,
                             "prior_score": old_score if mode == "old" else generated[mode][1],
                             "refined_score": old_score if mode == "old" else generated[mode][2],
                             "refined": False if mode == "old" else generated[mode][3],
                             "same_as_old": bool(np.array_equal(tail, plan[5:]))})
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise ValueError("model parameters changed")
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    for t in (2, 7, 17):
        positions = h.physical.replay_position_endpoints(starts, sequences[:, :t])
        arrays[f"cp{t}_eval_only"] = positions
        for row, pos in zip(rows, positions, strict=True):
            row[f"cp{t}_distance"] = float(np.linalg.norm(pos[:2] - h.GOAL[:2]))
    arrays["starts"], arrays["sequences"] = starts, sequences
    with ARRAYS.open("xb") as f:
        np.savez_compressed(f, **arrays)
    report = {"schema": "pc_joint_carry_all_roots.v1", "rows": rows,
              "scope": "exposed dependent development, not P1 or independent confirmation",
              "source_sha256": SOURCE_SHA, "arrays_sha256": h.sha256_file(ARRAYS),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "cem_kernel_sha256": h.sha256_file(Path(h.packet_cem.__code__.co_filename)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "action_mean": mean.cpu().tolist(), "action_std": std.cpu().tolist(),
              "counts": {"goal_encodes": 1, "cp7_encodes": len(entries),
                         "two_packet_model_samples": len(entries)*2*50,
                         "preflight_rgb_commands": len(entries)*7,
                         "execution_rgb_commands": len(rows)*17,
                         "position_replay_commands": len(rows)*(2+7+17)},
              "inherited_upstream_cost": source["counts"], "wall_seconds": monotonic()-begun}
    with OUTPUT.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
