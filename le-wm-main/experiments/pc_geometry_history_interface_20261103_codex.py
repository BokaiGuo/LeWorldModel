"""Compare fixed-chain one-frame and real three-frame action histories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_geometry_rgb_fixed_bank_select_20261103_codex as old
from experiments import pc_geometry_rgb_fixed_bank_physical_20261103_codex as physical
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

h, ROOT = old.h, old.ROOT
PROTOCOL = ROOT.parent / "docs/PC_GEOMETRY_HISTORY_INTERFACE_AUDIT_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_geometry_history_interface_20261103_codex.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    saved = json.loads(old.OUTPUT.read_text())
    phys = json.loads(physical.OUTPUT.read_text())
    if saved["arrays_sha256"] != old.prior.sha(old.ARRAYS) or phys["decision_sha256"] != old.prior.sha(old.OUTPUT):
        raise ValueError("source receipt mismatch")
    camera, prior = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], prior["body_offset_from_static_scene"]
    arrays, rows = {}, []
    with np.load(old.ARRAYS, allow_pickle=False) as original, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        before = h.parameter_hash(model)
        if before != saved["model_before"] or old.prior.sha(h.DEFAULT_CHECKPOINT) != saved["checkpoint_sha256"]:
            raise ValueError("model drift")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid action normalizer")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        goal_state, _ = h.estimate_state(goal, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
        for root, start in enumerate(old.STARTS):
            with ShadowRGBBridge(np.array([*start, 0., 0.])) as env:
                frames = [env.initial_rgb.copy()]
                for _ in range(15):
                    frames.append(env.step(np.zeros((1, 2))).copy())
            if not np.array_equal(frames[-1], original[f"root{root}_current_rgb"]):
                raise ValueError("current RGB mismatch")
            if any(not np.array_equal(frame, frames[0]) for frame in frames):
                raise ValueError("zero-history RGB changed")
            arrays[f"root{root}_history_rgb"] = np.stack(frames)
            enc = []
            with torch.no_grad():
                for tick in (5, 10, 15):
                    state = h.causal_state(frames[tick-1], frames[tick], metadata, offset, .05)
                    enc.append(h.encode_with_state(model, frames[tick], state, device))
            context3 = {key: torch.cat([e[key] for e in enc], dim=1) for key in ("visual", "proprio")}
            context1 = {key: value[:, -1:] for key, value in context3.items()}
            candidates = saved["rows"][root]["candidates"]
            raw = np.asarray([c["prefix_raw"] + c["suffix_raw"] for c in candidates], dtype=np.float32)
            if raw.shape != (12, 15, 2) or np.max(np.abs(raw)) > 1.000001:
                raise ValueError("invalid actions")
            actions = (torch.as_tensor(raw, device=device) - mean) / std
            packets = actions.reshape(12, 3, 10).transpose(0, 1)
            history = ((-mean/std).repeat(5)).reshape(1, 1, 10).expand(2, 12, 10).contiguous()
            if not torch.allclose((history.reshape(2, 12, 5, 2)*std + mean), torch.zeros((2, 12, 5, 2), device=device), atol=1e-7):
                raise ValueError("raw/normalized history mismatch")
            predictions = {}
            with torch.no_grad():
                for label, context, hist, window in (("single", context1, packets[:0], 1),
                                                      ("history3", context3, history, 3)):
                    pred = unroll_with_action_history(model, context, hist, packets, window)["proprio"][-1]
                    scores = h.projected_cost(pred.reshape(12, 256, 16), target, pinv, pos_std)
                    predictions[label] = scores.cpu().numpy()
            recorded = np.asarray([c["terminal_score"] for c in candidates])
            error = float(np.max(np.abs(predictions["single"]-recorded)))
            if error > 1e-5:
                raise ValueError(f"single-frame source score mismatch {error}")
            actual = np.asarray([x["terminal_distance"] for x in phys["rows"] if x["root_index"] == root])
            row = {"root_index": root, "single_source_max_error": error, "actual_physical_distance": actual.tolist(),
                   "raw_history_actions": np.zeros((10, 2)).tolist(),
                   "normalized_history_packets": history[:, 0].cpu().numpy().tolist(), "arms": {}}
            for label, scores in predictions.items():
                reversals = ties = 0
                for i in range(12):
                    for j in range(i+1, 12):
                        a, b = scores[i]-scores[j], actual[i]-actual[j]
                        if a == 0 or b == 0:
                            ties += 1
                        elif a*b < 0:
                            reversals += 1
                row["arms"][label] = {"predicted_cost": scores.tolist(), "selected_index": int(np.argmin(scores)),
                                      "absolute_proxy_vs_physical_error": np.abs(scores-actual).tolist(),
                                      "pairwise_reversals": reversals, "pairwise_ties": ties}
            rows.append(row)
            print(f"root {root} single/history indices {row['arms']['single']['selected_index']}/"
                  f"{row['arms']['history3']['selected_index']}", flush=True)
        if before != h.parameter_hash(model):
            raise ValueError("model changed")
    with ARRAYS.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    result = {"schema": "pc_geometry_history_interface.v1", "rows": rows,
              "protocol_sha256": old.prior.sha(PROTOCOL), "decision_sha256": old.prior.sha(old.OUTPUT),
              "physical_sha256": old.prior.sha(physical.OUTPUT), "arrays_sha256": old.prior.sha(ARRAYS),
              "runner_sha256": old.prior.sha(Path(__file__)), "model_hash": before}
    with OUTPUT.open("x") as handle:
        json.dump(result, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
