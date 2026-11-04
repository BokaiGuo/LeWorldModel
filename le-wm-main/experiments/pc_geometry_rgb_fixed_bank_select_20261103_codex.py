"""Model-only action-bank decision; physical outcome files are never read."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_geometry_rgb_action_manifest_20261103_codex as manifest
from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

h, ROOT = prior.h, prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_GEOMETRY_RGB_FIXED_BANK_SELECTION_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_geometry_rgb_fixed_bank_selection_20261103_codex.json"
ARRAYS = OUTPUT.with_suffix(".npz")
STARTS = ((.8, .8), (.8, 1.8), (2.8, .8), (2.8, 1.8))


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("decision artifacts already exist")
    action_doc = json.loads(manifest.OUTPUT.read_text())
    if set(action_doc) != {"schema", "source_sha256", "rows"} or len(action_doc["rows"]) != 48:
        raise ValueError("invalid action-only manifest")
    camera, old = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
    arrays, rows = {}, []
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        before = h.parameter_hash(model)
        if prior.sha(h.DEFAULT_CHECKPOINT) != prior.EXPECTED[h.DEFAULT_CHECKPOINT]:
            raise ValueError("checkpoint hash mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not bool(torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("invalid normalization")
        if not np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, atol=1e-7, rtol=0) or not np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, atol=1e-7, rtol=0):
            raise ValueError("proprio normalization drift")
        mean10, std10 = mean.repeat(10), std.repeat(10)
        lower, upper = (-1-mean10)/std10, (1-mean10)/std10
        zero = -mean10/std10
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        goal_state, _ = h.estimate_state(goal, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
        for root, start in enumerate(STARTS):
            fixed = [r for r in action_doc["rows"] if r["root_index"] == root]
            if len(fixed) != 12 or [r["prefix_name"] for r in fixed[:2]] != ["zero", "direct"]:
                raise ValueError("bank roster changed")
            raw = np.asarray([r["prefix_actions"] for r in fixed], dtype=np.float32)
            if raw.shape != (12, 5, 2) or not np.isfinite(raw).all() or (np.abs(raw) > 1.000001).any():
                raise ValueError("invalid bank actions")
            with ShadowRGBBridge(np.asarray([*start, 0., 0.])) as env:
                previous = env.initial_rgb.copy()
                current = env.step(np.zeros((1, 2))).copy()
            arrays[f"root{root}_previous_rgb"] = previous
            arrays[f"root{root}_current_rgb"] = current
            state = h.causal_state(previous, current, metadata, offset, .05)
            with torch.no_grad():
                context = h.encode_with_state(model, current, state, device)
                initial = float(h.projected_cost(context["proprio"].reshape(1, 256, 16), target, pinv, pos_std)[0])
            bank = (torch.as_tensor(raw, device=device).reshape(12, 10)-mean.repeat(5))/std.repeat(5)

            def predict(packets):
                with torch.no_grad():
                    prediction = unroll_with_action_history(model, context, packets[:0], packets, 1)["proprio"][-1]
                    return h.projected_cost(prediction.reshape(packets.shape[1], 256, 16), target, pinv, pos_std)
            first = predict(bank.unsqueeze(0)).cpu().numpy()
            generator = torch.Generator(device="cpu").manual_seed(2026110320+root)
            noises = torch.randn((2, 3, 16, 20), generator=generator).to(device)
            candidate_rows = []
            t0 = time.monotonic()
            for index, prefix in enumerate(bank):
                evaluations = []
                for restart in range(2):
                    loc, scale = zero.clone(), torch.full_like(zero, 1/3)
                    for generation in range(3):
                        samples = (loc + noises[restart, generation]*scale).clamp(lower, upper)
                        samples[0] = loc.clamp(lower, upper)
                        packets = torch.cat((prefix.expand(16, 10), samples), dim=1).reshape(16, 3, 10).transpose(0, 1)
                        scores = predict(packets)
                        evaluations.extend(zip(samples.cpu().numpy().tolist(), scores.cpu().numpy().tolist()))
                        elite = torch.argsort(scores, stable=True)[:4]
                        loc = samples[elite].mean(0)
                        scale = samples[elite].std(0, correction=1).clamp_min(1e-5)
                    finals = torch.stack((zero, loc)).clamp(lower, upper)
                    packets = torch.cat((prefix.expand(2, 10), finals), dim=1).reshape(2, 3, 10).transpose(0, 1)
                    scores = predict(packets)
                    evaluations.extend(zip(finals.cpu().numpy().tolist(), scores.cpu().numpy().tolist()))
                values = np.asarray([v for _, v in evaluations])
                best = int(np.argmin(values))
                suffix_norm = torch.as_tensor(evaluations[best][0], device=device)
                suffix_raw = (suffix_norm*std10+mean10).reshape(10, 2).cpu().numpy()
                candidate_rows.append({"name": fixed[index]["prefix_name"], "prefix_raw": raw[index].tolist(),
                                       "prefix_score": float(first[index]), "terminal_score": float(values[best]),
                                       "suffix_raw": suffix_raw.tolist(), "best_index": best,
                                       "evaluations": [{"suffix_normalized": a, "score": b} for a, b in evaluations]})
            scores = np.asarray([r["terminal_score"] for r in candidate_rows])
            selected = int(np.argmin(scores))
            qualified = bool(first[selected] >= initial and scores[selected] < min(scores[:2]))
            rows.append({"root_index": root, "initial_score": initial, "rgb_derived_state": state.tolist(),
                         "selected_index": selected, "selected_is_qualified": qualified,
                         "abstained_reconfiguration": not qualified, "candidates": candidate_rows,
                         "suffix_predicted_samples": 1200, "elapsed_seconds": time.monotonic()-t0})
            print(f"root {root}: selected {candidate_rows[selected]['name']}, qualified={qualified}", flush=True)
        after = h.parameter_hash(model)
        if before != after:
            raise ValueError("model changed")
    with ARRAYS.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    result = {"schema": "pc_geometry_rgb_fixed_bank_selection.v1", "rows": rows,
              "protocol_sha256": prior.sha(PROTOCOL), "action_manifest_sha256": prior.sha(manifest.OUTPUT),
              "checkpoint_sha256": prior.sha(h.DEFAULT_CHECKPOINT), "runner_sha256": prior.sha(Path(__file__)),
              "arrays_sha256": prior.sha(ARRAYS), "model_before": before, "model_after": after,
              "physical_outcomes_read": False, "development_panel_previously_exposed": True}
    with OUTPUT.open("x") as handle:
        json.dump(result, handle, allow_nan=False)
        handle.write("\n")
    print("decision sealed", hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
