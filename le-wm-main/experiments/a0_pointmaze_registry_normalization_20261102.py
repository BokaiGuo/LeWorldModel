"""Known-root frozen-model contrast using JEPA-WM's fixed PointMaze stats."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from experiments.a0_pointmaze_archive_bank_alignment_20261102 import score
from experiments.a0_pointmaze_constant_proprio_20261102 import constant_observation
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "93cdc7067ca623f1eb80439d1e41ff18b3be4468d8befdc3d9c6cecad26d3291"


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
    if sha256_file(args.source) != SOURCE_SHA:
        raise ValueError("unexpected sealed source")
    source = json.loads(args.source.read_text())
    if source["protocol"]["roots"] != [1880, 1881, 1882, 1883]:
        raise ValueError("unexpected roots")
    for key, path in (("checkpoint_sha256", args.checkpoint), ("config_sha256", args.config),
                      ("states_sha256", args.dataset_root / "states.pth"),
                      ("actions_sha256", args.dataset_root / "actions.pth")):
        if sha256_file(path) != source["provenance"][key]:
            raise ValueError(f"source identity mismatch: {key}")
    started = time.monotonic()
    device = require_cuda()
    model, dataset = load_model_and_dataset(args, device)
    from app.plan_common.datasets import DATA_STATS

    stats = DATA_STATS["pointmaze"]
    for name in ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std"):
        value = torch.tensor(stats[name], dtype=torch.float32)
        if not torch.isfinite(value).all() or (name.endswith("_std") and not (value > 0).all()):
            raise ValueError(f"invalid registry {name}")
        setattr(model.preprocessor, name, value)
    before = parameter_hash(model)
    if before != source["provenance"]["model_hash_before"]:
        raise ValueError("model mismatch")
    mean = np.asarray(stats["action_mean"], dtype=np.float32)
    std = np.asarray(stats["action_std"], dtype=np.float32)
    fixed = torch.tensor(stats["proprio_mean"], device=device, dtype=torch.float32)
    bank = np.asarray(source["protocol"]["bank"], dtype=np.float32)
    one = np.repeat(((bank - mean) / std)[:, None, :], 5, axis=1).reshape(8, 10)
    pairs = np.stack((np.repeat(one, 8, axis=0), np.tile(one, (8, 1))), axis=0)
    pairs_t = torch.as_tensor(pairs, device=device)
    scored = []
    for row in source["rows"]:
        root = row["root"]
        image = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
        if sha256_file(image) != row["archive_rgb_sha256"]:
            raise ValueError("RGB identity mismatch")
        current, _, _, _, _ = dataset.get_frames(root, [0])
        target, _, _, _, _ = dataset.get_frames(root, [10])
        with torch.no_grad():
            enc = model.encode(constant_observation(current["visual"].to(device), fixed))
            goal_enc = model.encode(constant_observation(target["visual"].to(device), fixed))
        context = TensorDict({"visual": enc["visual"], "proprio": enc["proprio"]},
                             batch_size=[1, 1], device=device)
        goal = goal_enc["visual"][0, 0].mean(dim=(0, 1, 2)).detach()
        scores = score(model, context, goal, pairs_t, args.batch_size).reshape(8, 8)
        serial = score(model, context, goal, pairs_t[:, :1], 1)[0]
        if not np.isclose(serial, scores.flat[0], rtol=1e-4, atol=1e-6):
            raise ValueError("batch and serial disagree")
        choice = np.unravel_index(int(np.argmin(scores)), (8, 8))
        scored.append({"root": root, "model_scores": scores.tolist(),
                       "model_pair": list(map(int, choice)),
                       "serial_batch_absolute_difference": float(abs(serial - scores.flat[0]))})
        print(json.dumps({"root": root, "pair": list(map(int, choice))}), flush=True)
    after = parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen parameters changed")
    for result, original in zip(scored, source["rows"], strict=True):
        costs = np.asarray(original["physical_terminal_costs"])
        result["physical_cost_selected"] = float(costs[tuple(result["model_pair"])])
        result["physical_regret"] = float(result["physical_cost_selected"] - costs.min())
    report = {
        "scope": "known-root registry-normalization sensitivity; not independent",
        "source_sha256": sha256_file(args.source),
        "runner_sha256": sha256_file(Path(__file__)),
        "registry_sha256": sha256_file(ROOT / "external/jepa-wms/app/plan_common/datasets/__init__.py"),
        "registry_stats": stats, "parameter_hash_before": before, "parameter_hash_after": after,
        "model_scores_per_root": 64, "additional_serial_scores_per_root": 1,
        "encodes_per_root": 2, "wall_seconds": time.monotonic() - started,
        "checkpoint_registry_association": "unverified",
        "training_root_overlap": "unknown", "rows": scored,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
