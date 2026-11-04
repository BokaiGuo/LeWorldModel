"""Evaluate frozen PointMaze scores with state-independent proprio input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from experiments.a0_pointmaze_archive_bank_alignment_20261102 import score
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)


def constant_observation(visual, fixed_proprio):
    """Construct input from RGB and a global constant, never episode proprio."""
    if visual.ndim != 4 or fixed_proprio.ndim != 1:
        raise ValueError("unexpected visual or constant proprio shape")
    visual = visual.float()
    if not torch.isfinite(visual).all() or visual.min() < -1e-6 or visual.max() > 1 + 1e-6:
        raise ValueError("visual must be finite RGB in [0,1]")
    if not torch.isfinite(fixed_proprio).all():
        raise ValueError("non-finite constant proprio")
    return {"visual": visual.unsqueeze(0) * 255.0,
            "proprio": fixed_proprio.reshape(1, 1, -1).expand(1, len(visual), -1).clone()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or not args.output.is_absolute() or args.batch_size < 1:
        raise ValueError("exclusive absolute output and positive batch size required")
    if sha256_file(args.source) != "93cdc7067ca623f1eb80439d1e41ff18b3be4468d8befdc3d9c6cecad26d3291":
        raise ValueError("unexpected sealed source report")
    source = json.loads(args.source.read_text())
    if source["protocol"]["roots"] != [1880, 1881, 1882, 1883]:
        raise ValueError("unexpected development roots")
    for key, path in (("checkpoint_sha256", args.checkpoint), ("config_sha256", args.config),
                      ("states_sha256", args.dataset_root / "states.pth"),
                      ("actions_sha256", args.dataset_root / "actions.pth")):
        if sha256_file(path) != source["provenance"][key]:
            raise ValueError(f"source identity mismatch: {key}")
    device = require_cuda()
    model, dataset = load_model_and_dataset(args, device)
    before = parameter_hash(model)
    if before != source["provenance"]["model_hash_before"]:
        raise ValueError("model mismatch")
    mean, std = dataset.action_mean.cpu().numpy(), dataset.action_std.cpu().numpy()
    if not (np.array_equal(mean, np.asarray(source["provenance"]["action_mean"], dtype=np.float32)) and
            np.array_equal(std, np.asarray(source["provenance"]["action_std"], dtype=np.float32))):
        raise ValueError("action normalizer mismatch")
    fixed = dataset.proprio_mean.to(device).float().reshape(-1)
    bank = np.asarray(source["protocol"]["bank"], dtype=np.float32)
    normalized = (bank - mean) / std
    one = np.repeat(normalized[:, None, :], 5, axis=1).reshape(8, 10)
    pairs = np.stack((np.repeat(one, 8, axis=0), np.tile(one, (8, 1))), axis=0)
    pairs_t = torch.as_tensor(pairs, device=device)
    scored = []
    for row in source["rows"]:
        root = row["root"]
        image_path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
        if sha256_file(image_path) != row["archive_rgb_sha256"]:
            raise ValueError("RGB identity mismatch")
        context_raw, _, _, _, _ = dataset.get_frames(root, [0])
        goal_raw, _, _, _, _ = dataset.get_frames(root, [10])
        context_obs = constant_observation(context_raw["visual"].to(device), fixed)
        goal_obs = constant_observation(goal_raw["visual"].to(device), fixed)
        with torch.no_grad():
            encoded_context = model.encode(context_obs)
            encoded_goal = model.encode(goal_obs)
        context = TensorDict({"visual": encoded_context["visual"], "proprio": encoded_context["proprio"]},
                             batch_size=[1, 1], device=device)
        goal = encoded_goal["visual"][0, 0].mean(dim=(0, 1, 2)).detach()
        scores = score(model, context, goal, pairs_t, args.batch_size).reshape(8, 8)
        serial = score(model, context, goal, pairs_t[:, :1], 1)[0]
        if not np.isclose(serial, scores.flat[0], rtol=1e-4, atol=1e-6):
            raise ValueError("batch and serial disagree")
        choice = np.unravel_index(int(np.argmin(scores)), (8, 8))
        scored.append({"root": root, "model_scores": scores.tolist(), "model_pair": list(map(int, choice)),
                       "serial_batch_absolute_difference": float(abs(serial - scores.flat[0]))})
        print(json.dumps({"root": root, "selected_pair": list(map(int, choice))}), flush=True)
    after = parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen parameters changed")
    # Outcome arrays are used only after every model decision is fixed.
    for result, original in zip(scored, source["rows"], strict=True):
        costs = np.asarray(original["physical_terminal_costs"])
        selected = tuple(result["model_pair"])
        result.update({"physical_cost_selected": float(costs[selected]),
                       "physical_oracle_cost": float(costs.min()),
                       "physical_regret": float(costs[selected] - costs.min()),
                       "native_proprio_model_pair": original["model_pair"],
                       "native_proprio_regret": original["physical_regret"]})
    report = {"scope": "known roots, fixed-global-proprio diagnostic; no PC/P1 gate",
              "source_sha256": sha256_file(args.source), "runner_sha256": sha256_file(Path(__file__)),
              "parameter_hash_before": before, "parameter_hash_after": after,
              "constant_proprio": fixed.cpu().tolist(),
              "protocol": {"inputs": "RGB, fixed global proprio constant, action history",
                           "per_episode_proprio_used": False, "model_samples_per_root": 64,
                           "distribution_shift": "constant proprio input to model trained with proprio",
                           "state_and_oracle_use": "offline evaluation after model choices",
                           "provenance_of_global_normalization": "unknown"},
              "rows": scored}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
