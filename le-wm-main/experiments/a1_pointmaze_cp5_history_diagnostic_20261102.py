"""Compare fixed CP5 tail predictions with and without actual action history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode, goal_latent
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import (
    ARRAYS, OUTPUT as SOURCE, SOURCE_SHA as PREFIX_SOURCE_SHA, image_hash,
)
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file, terminal_latent,
)

SOURCE_SHA = "8f0bba93fa1f093cae4550da317d9b20a6cc018756e37402419f384eabdf6446"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP5_HISTORY_DIAGNOSTIC_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a1_pointmaze_cp5_history_diagnostic_20261102_DEV01.json"
PREDICTIONS = ROOT / "results/a1_pointmaze_cp5_history_diagnostic_20261102_DEV01.npz"


def history_prediction(model, first, second, prefix, tails):
    """Condition two real contexts on preceding packet and each next packet."""
    if prefix.shape != (10,) or tails.shape != (8, 10):
        raise ValueError("expected one historical packet and eight future packets")
    visual = torch.cat((first["visual"], second["visual"]), dim=1).expand(8, -1, -1, -1, -1, -1)
    proprio = torch.cat((first["proprio"], second["proprio"]), dim=1).expand(8, -1, -1, -1)
    actions = torch.stack((prefix.expand(8, -1), tails), dim=1)
    action_features = model.model.encode_act(actions)
    prediction, _, _ = model.model.forward_pred(visual, action_features, proprio)
    return prediction[:, -1].mean(dim=(1, 2, 3))


def main():
    if OUTPUT.exists() or PREDICTIONS.exists() or sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("output exists or source changed")
    source = json.loads(SOURCE.read_text())
    if source["source_sha256"] != PREFIX_SOURCE_SHA or sha256_file(ARRAYS) != source["arrays_sha256"]:
        raise RuntimeError("source ancestry or arrays changed")
    if sha256_file(DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    bank = np.asarray(source["bank"], dtype=np.float32)
    if bank.shape != (8, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
        raise ValueError("invalid bank")
    began, rows, saved = time.monotonic(), [], {}
    print("Comparing actual-history and single-frame predictions on sealed actions.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"), np.load(ARRAYS, allow_pickle=False) as data:
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid normalizer")
        tails = ((torch.as_tensor(bank, device=device) - mean) / std).repeat(1, 5)
        original = json.loads((ROOT / "results/a1_pointmaze_native_spectral_20261102_DEV01.json").read_text())
        old_rows = {r["root"]: r for r in original["rows"]}
        for row in source["rows"]:
            root = row["root"]
            state = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            with NativeRGBBridge(state) as env:
                rgb0 = env.initial_rgb
            if image_hash(rgb0) != old_rows[root]["initial_rgb_sha256"]:
                raise RuntimeError("initial RGB mismatch")
            first = encode(model, rgb0, device)
            goal = torch.as_tensor(data[f"r{root}_goal_latent"], device=device)
            branches = []
            for branch in row["branches"]:
                method = branch["method"]
                key = f"r{root}_{method}"
                rgb5 = data[f"{key}_rgb5"]
                if image_hash(rgb5) != branch["rgb5_sha256"]:
                    raise RuntimeError("feedback RGB mismatch")
                second = encode(model, rgb5, device)
                prefix = torch.as_tensor(branch["prefix_action"], device=device)
                prefix = ((prefix - mean) / std).repeat(5)
                with torch.no_grad():
                    single = terminal_latent(model, second, tails.unsqueeze(0))
                    history = history_prediction(model, first, second, prefix, tails)
                    single_cost = ((single - goal) ** 2).mean(dim=1)
                    history_cost = ((history - goal) ** 2).mean(dim=1)
                cost_np = single_cost.cpu().numpy()
                if not np.allclose(cost_np, branch["tail_costs"], rtol=1e-4, atol=1e-6):
                    raise RuntimeError("source single-frame scores cannot be reproduced")
                chosen = branch["tail_index"]
                actual = np.asarray(data[f"{key}_latent10"])
                pred_single, pred_history = single[chosen].cpu().numpy(), history[chosen].cpu().numpy()
                saved[f"{key}_single"] = single.cpu().numpy()
                saved[f"{key}_history"] = history.cpu().numpy()
                branches.append({
                    "method": method, "executed_tail": chosen,
                    "single_costs": cost_np.tolist(), "history_costs": history_cost.cpu().tolist(),
                    "history_choice_not_executed": int(torch.argmin(history_cost).cpu()),
                    "source_max_abs_cost_difference": float(np.max(np.abs(cost_np - branch["tail_costs"]))),
                    "single_executed_latent_mse": float(np.mean((pred_single - actual) ** 2)),
                    "history_executed_latent_mse": float(np.mean((pred_history - actual) ** 2)),
                })
            rows.append({"root": root, "branches": branches})
            print(f"history diagnostic completed root {root}", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    with PREDICTIONS.open("xb") as stream:
        np.savez_compressed(stream, **saved)
    report = {
        "schema": "a1_pointmaze_cp5_history_diagnostic.v1",
        "scope": "fixed-action exposed-root diagnostic; history-selected actions not executed",
        "protocol_sha256": sha256_file(PROTOCOL), "source_sha256": SOURCE_SHA,
        "source_arrays_sha256": source["arrays_sha256"], "script_sha256": sha256_file(Path(__file__)),
        "predictions_sha256": sha256_file(PREDICTIONS), "model_before": before, "model_after": after,
        "device": torch.cuda.get_device_name(device), "rows": rows, "wall_seconds": time.monotonic() - began,
        "budget": {"goal_encodes": 0, "initial_rgb_encodes": 4, "cp5_rgb_encodes": 16,
                   "single_prediction_samples": 128, "history_prediction_samples": 128},
    }
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
