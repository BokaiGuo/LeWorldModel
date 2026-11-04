"""Compare sealed same-action affine predictions with RGB encodings and physics."""
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
from experiments.pc_pointmaze_affine_nonprogress_lookahead_20261103 import OUTPUT as LOOK
from experiments.pc_pointmaze_affine_nonprogress_oracle_20261103 import OUTPUT as ORACLE, SOURCE_DIGEST
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import LENGTHS_SHA
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import BANK_SHA, SOURCE_SHA
from experiments.pc_pointmaze_obstacle_nonprogress_bank_20261103 import OUTPUT as BANK
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import DATA, GOAL, ROOTS, OUTPUT as SOURCE
from experiments.pc_pointmaze_safe_ranking_20261103 import ranking
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

ORACLE_SHA = "6884f717ab6fabc6fa96614c466eb168406c1360b5967041cf46025d44af8a6b"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AFFINE_SAME_ACTION_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_affine_same_action_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, digest in ((LOOK, SOURCE_DIGEST), (ORACLE, ORACLE_SHA), (BANK, BANK_SHA),
                         (SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA),
                         (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA)):
        if sha256_file(path) != digest:
            raise ValueError(f"changed source: {path}")
    look, oracle, bank_doc, source, camera, old = (json.loads(p.read_text()) for p in
                                                   (LOOK, ORACLE, BANK, SOURCE, CAMERA, OLD))
    for path, digest in ((LOOK.with_suffix(".npz"), look["arrays_sha256"]),
                         (ORACLE.with_suffix(".npz"), oracle["arrays_sha256"]),
                         (BANK.with_suffix(".npz"), bank_doc["arrays_sha256"]),
                         (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (DATA / "states.pth", source["states_sha256"]),
                         (DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError(f"changed input: {path}")
    if [r["root"] for r in look["rows"]] != list(ROOTS) or len(oracle["rows"]) != 36:
        raise ValueError("unexpected source roster")
    weight, _, _ = get_encoder()
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, starts, actions_all = {}, [], [], []
    begun = time.monotonic()
    print("Replay 36 fixed chains: open prediction, RGB refresh, RGB encoding and physics.", flush=True)
    with np.load(LOOK.with_suffix(".npz"), allow_pickle=False) as sealed, \
            np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as physical_bank, \
            np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as src, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != look["model_before"]:
            raise RuntimeError("model identity mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not (np.array_equal(mean.cpu().numpy(), np.asarray(look["action_mean"], np.float32))
                and np.array_equal(std.cpu().numpy(), np.asarray(look["action_std"], np.float32))):
            raise ValueError("action normalizer mismatch")
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(STD[:2], dtype=torch.float32, device=device)
        goal_rgb = src["goal_rgb"]
        goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
        target = encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        arrays["target_token"] = target.cpu().numpy()
        for root_i, (root, parent) in enumerate(zip(ROOTS, look["rows"])):
            initial_rgb = src[f"r{root}_random_frames"][0]
            initial_state, _ = estimate_state(initial_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            context = encode_with_state(model, initial_rgb, initial_state, device)
            frame_list, token_lists = [], {key: [] for key in ("open5", "open10", "refresh10", "actual5", "actual10")}
            branches = []
            for index in range(9):
                slot = root_i * 9 + index
                if (oracle["rows"][slot]["root"], oracle["rows"][slot]["bank_index"]) != (root, index):
                    raise ValueError("physical oracle slot mismatch")
                initial = physical_bank["starts"][slot]
                actions = physical_bank["sequences"][slot].astype(np.float32)
                if actions.shape != (10, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                    raise ValueError("invalid physical actions")
                if not np.array_equal(actions[:5], np.repeat(bank[f"r{root}_bank"][index][None], 5, axis=0)):
                    raise ValueError("prefix action mismatch")
                actual_norm = (torch.as_tensor(actions, device=device) - mean) / std
                sealed_tail = torch.as_tensor(sealed[f"r{root}_lookahead_tails"][index], device=device).reshape(5, 2)
                roundtrip = float((actual_norm[5:] - sealed_tail).abs().max().cpu())
                if roundtrip > 2e-6:
                    raise RuntimeError("action normalization roundtrip mismatch")
                packets = torch.stack((actual_norm[:5].reshape(10), actual_norm[5:].reshape(10))).unsqueeze(1)
                with torch.no_grad():
                    rollout = model.unroll(context, act_suffix=packets)["proprio"]
                    if rollout.shape[:2] != (3, 1):
                        raise ValueError("open rollout clock mismatch")
                    open5, open10 = rollout[1, 0].reshape(256, 16), rollout[2, 0].reshape(256, 16)
                    score5 = float(projected_cost(open5.unsqueeze(0), target, pinv, pos_std).item())
                    score10 = float(projected_cost(open10.unsqueeze(0), target, pinv, pos_std).item())
                expected5 = float(sealed[f"r{root}_first_scores"][index])
                expected10 = float(sealed[f"r{root}_terminal_scores"][index].min())
                if not (np.isclose(score5, expected5, atol=1e-6, rtol=1e-4)
                        and np.isclose(score10, expected10, atol=1e-6, rtol=1e-4)):
                    raise RuntimeError("sealed open score mismatch")
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, initial_rgb):
                        raise RuntimeError("initial RGB mismatch")
                    frames = [env.initial_rgb]
                    for action in actions:
                        frames.append(env.step(action[None]))
                    if env.step_count != 10:
                        raise RuntimeError("physical clock mismatch")
                if not np.array_equal(frames[5], bank[f"r{root}_rgb5"][index]):
                    raise RuntimeError("cp5 RGB mismatch")
                state5 = causal_state(frames[4], frames[5], camera["metadata"], old["body_offset_from_static_scene"], .05)
                state10 = causal_state(frames[9], frames[10], camera["metadata"], old["body_offset_from_static_scene"], .05)
                actual5 = encode_with_state(model, frames[5], state5, device)
                actual10 = encode_with_state(model, frames[10], state10, device)
                with torch.no_grad():
                    refresh = model.unroll(actual5, act_suffix=actual_norm[5:].reshape(1, 1, 10))["proprio"][-1, 0].reshape(256, 16)
                    token5, token10 = actual5["proprio"].reshape(256, 16), actual10["proprio"].reshape(256, 16)
                    score = lambda token: float(projected_cost(token.unsqueeze(0), target, pinv, pos_std).item())
                    branch = {"index": index, "open5": score5, "open10": score10,
                              "refresh10": score(refresh), "actual5": score(token5), "actual10": score(token10),
                              "open10_token_mse": float((open10-token10).square().mean().item()),
                              "refresh10_token_mse": float((refresh-token10).square().mean().item()),
                              "normalized_roundtrip_max": roundtrip,
                              "state5_rgb_estimate": state5.tolist(), "state10_rgb_estimate": state10.tolist()}
                for key, token in (("open5", open5), ("open10", open10), ("refresh10", refresh),
                                   ("actual5", token5), ("actual10", token10)):
                    token_lists[key].append(token.cpu().numpy())
                frame_list.append(np.stack(frames))
                starts.append(initial)
                actions_all.append(actions)
                branches.append(branch)
            arrays[f"r{root}_frames"] = np.stack(frame_list)
            for key, tokens in token_lists.items():
                arrays[f"r{root}_{key}_tokens"] = np.stack(tokens)
            rows.append({"root": root, "branches": branches})
            print(f"root {root}: nine fixed chains reproduced and encoded", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    starts, actions_all = np.asarray(starts), np.asarray(actions_all)
    cp5 = physical.replay_position_endpoints(starts, actions_all[:, :5])
    cp10 = physical.replay_position_endpoints(starts, actions_all)
    if not np.array_equal(cp5[:, :2], physical_bank["cp5"][:, :2]) or not np.array_equal(cp10[:, :2], physical_bank["cp10"][:, :2]):
        raise RuntimeError("sealed physical endpoint mismatch")
    for i, row in enumerate(rows):
        for j, branch in enumerate(row["branches"]):
            branch["physical5"] = float(np.linalg.norm(cp5[9*i+j, :2] - GOAL[:2]))
            branch["physical10"] = float(np.linalg.norm(cp10[9*i+j, :2] - GOAL[:2]))
        b = row["branches"]
        values = lambda key: [x[key] for x in b]
        row["ranking"] = {f"{a}_vs_{c}": ranking(values(a), values(c)) for a, c in
                          (("open5", "actual5"), ("actual5", "physical5"),
                           ("open10", "actual10"), ("refresh10", "actual10"),
                           ("actual10", "physical10"), ("open10", "physical10"),
                           ("refresh10", "physical10"))}
    arrays["starts"], arrays["actions"], arrays["cp5"], arrays["cp10"] = starts, actions_all, cp5, cp10
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "pc_pointmaze_affine_same_action.v1", "rows": rows,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "look_sha256": SOURCE_DIGEST, "oracle_sha256": ORACLE_SHA,
              "bank_sha256": BANK_SHA, "source_sha256": SOURCE_SHA, "model_before": before, "model_after": after,
              "physical_chains": len(actions_all), "open_two_packet_predictions": len(actions_all),
              "refreshed_predictions": len(actions_all), "real_rgb_encodes": 2*len(actions_all),
              "wall_seconds": time.monotonic() - begun, "scope": "exposed roots; oracle cp5; not P1"}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
