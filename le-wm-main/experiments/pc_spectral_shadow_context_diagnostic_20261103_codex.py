"""Replay two saved RGB contexts with and without spectral CEM initialization."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability.pc_spectral_seeded_feedback_cem_20261103_codex import (
    seeded_cem, spectral_candidates,
)

h = prior.h
ROOT = prior.ROOT
SOURCE = ROOT / "results/pc_pointmaze_shadow_closed_loop_dev_20261103_CODEX.json"
OUTPUT = ROOT / "results/pc_spectral_shadow_context_diagnostic_20261103_CODEX.json"
PROTOCOL = ROOT.parent / "docs/PC_SPECTRAL_SHADOW_CONTEXT_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text())
    baseline_source, archive, _ = prior.files()
    prior.load_sealed(prior.CONTEXT)
    rows = source["rows"]
    if len(rows) != 2:
        raise ValueError("unexpected branch roster")
    snapshots = []
    for row in rows:
        step = row["steps"][0]
        path = Path(step["decision_path"])
        if prior.sha(path) != step["decision_sha256"]:
            raise ValueError("decision snapshot hash mismatch")
        snapshots.append((row["name"], path))

    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        before = h.parameter_hash(model)
        if before != baseline_source["model_before"]:
            raise ValueError("model hash mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if (mean.cpu().tolist() != archive["provenance"]["action_mean"]
                or std.cpu().tolist() != archive["provenance"]["action_std"]
                or not bool(torch.isfinite(std).all() and (std > 0).all())):
            raise ValueError("action normalization mismatch")
        if (not np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, atol=1e-7, rtol=0)
                or not np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, atol=1e-7, rtol=0)):
            raise ValueError("proprio normalization mismatch")
        lower, upper = ((-1-mean)/std).repeat(5), ((1-mean)/std).repeat(5)
        zero = ((-mean)/std).repeat(5)
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        camera, old = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        with torch.no_grad():
            target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        result_rows = []
        for name, path in snapshots:
            with np.load(path, allow_pickle=False) as saved:
                estimate = h.causal_state(saved["previous_rgb"], saved["current_rgb"], metadata, offset, .05)
                np.testing.assert_allclose(estimate, saved["rgb_derived_state"], rtol=0, atol=1e-7)
                with torch.no_grad():
                    context = h.encode_with_state(model, saved["current_rgb"], estimate, device)
                noise = torch.as_tensor(saved["noise"], dtype=torch.float32, device=device)

                def response(action):
                    future = action.reshape(1, 1, 10)
                    pred = unroll_with_action_history(model, context, future[:0], future, 1)["proprio"][-1]
                    return (pred.reshape(256, 16).mean(dim=0) @ pinv.T)[:2] * pos_std

                def evaluate(candidate):
                    with torch.no_grad():
                        future = candidate.unsqueeze(0)
                        pred = unroll_with_action_history(model, context, future[:0], future, 1)["proprio"][-1]
                        return h.projected_cost(pred.reshape(len(candidate), 256, 16), target, pinv, pos_std)

                torch.cuda.synchronize()
                started = time.monotonic()
                baseline = seeded_cem(evaluate, zero, lower, upper, noise)
                torch.cuda.synchronize()
                baseline_seconds = time.monotonic() - started
                np.testing.assert_allclose(baseline["chosen"].cpu().numpy(), saved["chosen"], rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(baseline["final_scores"].cpu().numpy(), saved["final_scores"], rtol=1e-5, atol=1e-6)

                torch.cuda.synchronize()
                started = time.monotonic()
                structure = spectral_candidates(response, mean, std)
                torch.cuda.synchronize()
                structure_seconds = time.monotonic() - started
                columns = []
                for index in range(10):
                    displacement = torch.zeros_like(zero)
                    displacement[index] = .01 / std.repeat(5)[index]
                    with torch.no_grad():
                        columns.append((response(zero + displacement) - response(zero - displacement)) / .02)
                fd = torch.stack(columns, dim=1)
                difference = fd - structure["jacobian_raw"]
                torch.cuda.synchronize()
                started = time.monotonic()
                spectral = seeded_cem(evaluate, zero, lower, upper, noise, structure["normalized"])
                torch.cuda.synchronize()
                spectral_seconds = time.monotonic() - started
                result_rows.append({
                    "name": name, "snapshot": str(path), "snapshot_sha256": prior.sha(path),
                    "baseline_final_scores": baseline["final_scores"].cpu().tolist(),
                    "spectral_final_scores": spectral["final_scores"].cpu().tolist(),
                    "baseline_chosen_normalized": baseline["chosen"].cpu().tolist(),
                    "spectral_chosen_normalized": spectral["chosen"].cpu().tolist(),
                    "spectral_raw_candidates": structure["raw"].cpu().tolist(),
                    "jacobian_raw": structure["jacobian_raw"].cpu().tolist(),
                    "fd_jacobian_raw": fd.cpu().tolist(),
                    "fd_max_abs_error": float(difference.abs().max()),
                    "fd_relative_norm_error": float(difference.norm() / fd.norm().clamp_min(1e-12)),
                    "singular_values": structure["singular_values"].cpu().tolist(),
                    "structure_counts": structure["structure_counts"],
                    "baseline_cem_counts": baseline["cem_counts"], "spectral_cem_counts": spectral["cem_counts"],
                    "fd_model_forwards": 20,
                    "baseline_seconds": baseline_seconds, "structure_seconds": structure_seconds,
                    "spectral_cem_seconds": spectral_seconds,
                })
        after = h.parameter_hash(model)
        if after != before:
            raise RuntimeError("model parameters changed")
        report = {"schema": "pc_spectral_shadow_context_diagnostic.v1", "rows": result_rows,
                  "source_sha256": prior.sha(SOURCE), "protocol_sha256": prior.sha(PROTOCOL),
                  "model_hash_before": before, "model_hash_after": after,
                  "device": torch.cuda.get_device_name(device), "goal_encodes": 1, "feedback_encodes": 2,
                  "physical_execution": False, "equal_total_compute": False}
        with OUTPUT.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
