"""Audit validation goal encoding as well as terminal prediction on host CUDA.

The terminal-only alignment audit does not measure batched goal-encoder
numerics. This companion quantifies that remaining path without changing a
checkpoint, selecting an epoch, or altering the P1 acceptance thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability import cube_d2p_training_source as source
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as runtime
from experiments.spectral_reachability.train_cube_d2p_same_root_ranking_repair import (
    _Groups, _normalise, _rgb, _sha256, _terminal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("score alignment output must be new")
    manifest = json.loads((args.checkpoint / "repair_manifest.json").read_text())
    if manifest.get("status") != "trained_same_root_ranking_repair":
        raise ValueError("score alignment requires a complete same-root repair")
    host = require_cuda_allocation(args.device)
    device = torch.device("cuda")
    receipt = source.validate_receipt(Path(manifest["receipt"]))
    mean_np = np.asarray(receipt["calibration_action_mean"], np.float32)
    std_np = np.asarray(receipt["calibration_action_std"], np.float32)
    mean = torch.as_tensor(mean_np, device=device).view(1, 1, -1)
    std = torch.as_tensor(std_np, device=device).view(1, 1, -1)
    dataset = _Groups(Path(manifest["pool"]["path"]), split="validation")
    model = load_compatible_pretrained(args.checkpoint, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = _image_transform()
    low = torch.full((5,), -1.0, device=device)
    before = runtime._parameter_hash(model)
    rows = []
    # Use precisely the saved validation batch size, including its last batch.
    batch_size = int(manifest["optimization"]["batch_size"])
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            successors = torch.stack([item["successors"] for item in items]).flatten(0, 1)
            targets = model.encode({"pixels": _rgb(successors, transform=transform, device=device)[:, None]})["emb"][:, 0]
            goals = targets.reshape(len(items), dataset.candidates, -1)[:, 0]
            for offset, item in enumerate(items):
                context = _rgb(item["context"][None], transform=transform, device=device)
                prefix = _normalise(item["prefix"][None], mean=mean, std=std, device=device)
                future = _normalise(item["candidates"][None], mean=mean, std=std, device=device)
                count = dataset.candidates
                actions = torch.cat((prefix[:, 1:].expand(count, -1, -1), future[0, :, None]), dim=1)
                predicted = _terminal(model, context.expand(count, -1, -1, -1, -1).contiguous(), actions)
                validation_costs = (predicted - goals[offset]).square().mean(dim=-1)
                frames = item["context"].permute(0, 2, 3, 1).numpy()
                goal_rgb = item["successors"][0].permute(1, 2, 0).numpy()
                deployed_goal = runtime._goal_latent(model, goal_rgb, transform=transform, device=device)
                costs, errors = [], []
                for candidate in range(count):
                    deployed = runtime._terminal_prediction(
                        model, list(frames[:3]), frames[3], item["prefix"].numpy(),
                        item["candidates"][candidate:candidate + 1].to(device),
                        mean=mean_np, std=std_np, action_block=5, action_dim=5,
                        transform=transform, device=device, low=low, high=-low, future_packets_are_raw=True,
                    )
                    costs.append((deployed - deployed_goal).square().mean())
                    errors.append(float((deployed - predicted[candidate]).abs().max()))
                deployed_costs = torch.stack(costs)
                rows.append({
                    "validation_root_seed": int(dataset.root_seeds[dataset.indices[start + offset]]),
                    "goal_embedding_max_abs_error": float((goals[offset] - deployed_goal).abs().max()),
                    "terminal_max_abs_error": max(errors),
                    "goal_cost_max_abs_error": float((validation_costs - deployed_costs).abs().max()),
                    "validation_argmin": int(validation_costs.argmin()),
                    "deployed_argmin": int(deployed_costs.argmin()),
                })
    gates = {"cuda_only": True, "parameters_unchanged": before == runtime._parameter_hash(model),
             "all_terminal_errors_le_2e_5": all(row["terminal_max_abs_error"] <= 2e-5 for row in rows),
             "all_candidate_argmins_equal": all(row["validation_argmin"] == row["deployed_argmin"] for row in rows)}
    result = {"schema": "cube_d2p_same_root_score_alignment.v1", "status": "pass" if all(gates.values()) else "fail",
              "cuda_host": host, "gates": gates, "rows": rows,
              "checkpoint_weights_sha256": _sha256(args.checkpoint / "weights.pt"),
              "aggregate": {"validation_roots": len(rows),
                  "max_goal_embedding_error": max(row["goal_embedding_max_abs_error"] for row in rows),
                  "max_goal_cost_error": max(row["goal_cost_max_abs_error"] for row in rows),
                  "argmin_disagreement_count": sum(row["validation_argmin"] != row["deployed_argmin"] for row in rows)},
              "claim_boundary": "Numerical diagnostic on existing validation roots only; not fresh P1 or efficacy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "gates": gates}, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("validation/runtime score alignment rejected")


if __name__ == "__main__":
    main()
