"""Exploratory same-start archive RGB/proprio model vs position-only physics."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT,
    load_model_and_dataset, make_root_context, parameter_hash, require_cuda,
    sha256_file, terminal_cost,
)

ROOTS = (1880, 1881, 1882, 1883)
SOURCE = ROOT / "results/a0_pointmaze_packet_oracle_20261102_DEV01.json"
SOURCE_SHA = "3b38090507edc8f787cf0981e4d93a0d9d76636a593e2356e77386a4953a7b2e"
NAME = "a0_pointmaze_archive_bank_alignment_20261102"


def score(model, context, goal, packets, batch_size):
    chunks = []
    with torch.no_grad():
        for start in range(0, packets.shape[1], batch_size):
            chunks.append(terminal_cost(model, context, packets[:, start:start + batch_size], goal).cpu())
    return torch.cat(chunks).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.parent != ROOT / "results"
            or not args.output.name.startswith(NAME + "_") or args.output.suffix != ".json"
            or args.output.exists() or args.batch_size < 1):
        raise ValueError("exclusive output path or batch size")
    if sha256_file(SOURCE) != SOURCE_SHA:
        raise ValueError("candidate bank source hash mismatch")
    original = json.loads(SOURCE.read_text())
    # First magnitude, all eight directions. Selected before inspecting this experiment.
    bank = np.asarray(original["rows"][0]["candidate_bank"][:8], dtype=np.float32)
    if bank.shape != (8, 2) or not np.isfinite(bank).all():
        raise ValueError("invalid bank")
    device = require_cuda()
    start_time = time.monotonic()
    model, dataset = load_model_and_dataset(args, device)
    if model.action_dim != 10 or model.model.frameskip != 5:
        raise ValueError("unexpected packet contract")
    before = parameter_hash(model)
    mean = dataset.action_mean.cpu().numpy()
    std = dataset.action_std.cpu().numpy()
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid action normalization")
    normalized = (bank - mean) / std
    one = np.repeat(normalized[:, None, :], 5, axis=1).reshape(8, 10)
    pairs = np.stack((np.repeat(one, 8, axis=0), np.tile(one, (8, 1))), axis=0)
    one_t = torch.as_tensor(one[None], device=device)
    pairs_t = torch.as_tensor(pairs, device=device)
    rows = []
    for root in ROOTS:
        image_path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        context, goal, _ = make_root_context(model, dataset, root, 0, 2, device)
        model_one = score(model, context, goal, one_t, args.batch_size)
        model_two = score(model, context, goal, pairs_t, args.batch_size).reshape(8, 8)
        serial = score(model, context, goal, pairs_t[:, :1], 1)[0]
        if not np.isclose(serial, model_two.flat[0], rtol=1e-4, atol=1e-6):
            raise ValueError("batched/serial prediction mismatch")
        state = dataset.states[root, 0].cpu().numpy().astype(np.float64)
        target = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
        starts_one = np.repeat(state[None], 8, axis=0)
        starts_two = np.repeat(state[None], 64, axis=0)
        actions_one = np.repeat(bank[:, None], 5, axis=1)
        actions_two = np.concatenate((np.repeat(np.repeat(bank, 8, axis=0)[:, None], 5, axis=1),
                                      np.repeat(np.tile(bank, (8, 1))[:, None], 5, axis=1)), axis=1)
        p1 = physical.replay_position_endpoints(starts_one, actions_one)[:, :2]
        p2 = physical.replay_position_endpoints(starts_two, actions_two)[:, :2].reshape(8, 8, 2)
        c1 = np.linalg.norm(p1 - target, axis=-1)
        c2 = np.linalg.norm(p2 - target, axis=-1)
        model_choice = np.unravel_index(np.argmin(model_two), model_two.shape)
        physics_choice = np.unravel_index(np.argmin(c2), c2.shape)
        rows.append({
            "root": root, "archive_rgb_sha256": sha256_file(image_path),
            "state_start_evaluation_only": state.tolist(), "goal_position_evaluation_only": target.tolist(),
            "model_prefix_scores": model_one.tolist(), "model_terminal_scores": model_two.tolist(),
            "physical_prefix_positions": p1.tolist(), "physical_terminal_positions": p2.tolist(),
            "physical_prefix_costs": c1.tolist(), "physical_terminal_costs": c2.tolist(),
            "model_pair": list(map(int, model_choice)), "oracle_pair": list(map(int, physics_choice)),
            "model_pair_physical_cost": float(c2[model_choice]),
            "oracle_physical_cost": float(c2[physics_choice]),
            "physical_regret": float(c2[model_choice] - c2[physics_choice]),
            "serial_batch_absolute_difference": float(abs(serial - model_two.flat[0])),
        })
        print(f"root {root} completed", flush=True)
    after = parameter_hash(model)
    if after != before:
        raise RuntimeError("model parameters changed")
    result = {
        "schema": NAME + ".v1",
        "protocol": {
            "roots": list(ROOTS), "development_roots": True, "training_membership": "unknown",
            "candidate_source_sha256": SOURCE_SHA, "bank": bank.tolist(),
            "bank_rule": "first eight directions of root 1900 source bank, fixed before outcomes",
            "start_frame": 0, "goal_rgb_frame": 10, "physical_goal": "archive frame 10 position",
            "packet": "five repetitions of one physical candidate; action normalized once",
            "model_input": "recorded RGB AND proprio; not RGB-only",
            "goal_model_input": "recorded RGB and proprio through existing model encoder",
            "simulator_state": "only physics initialization and offline evaluation",
            "model_samples_per_root": 72, "physical_sequences_per_root": 72,
            "physical_backend": physical.position_backend_receipt(),
            "physical_state_equivalence": "position-only; full state not certified",
            "planner_selection": "terminal latent distance to frame 10 RGB",
            "no_independent_confirmation": True,
        },
        "provenance": {
            "checkpoint_sha256": sha256_file(args.checkpoint), "config_sha256": sha256_file(args.config),
            "states_sha256": sha256_file(args.dataset_root / "states.pth"),
            "actions_sha256": sha256_file(args.dataset_root / "actions.pth"),
            "sequence_lengths_sha256": sha256_file(args.dataset_root / "seq_lengths.pth"),
            "model_hash_before": before, "model_hash_after": after,
            "code_sha256": sha256_file(Path(__file__)),
            "action_mean": mean.tolist(), "action_std": std.tolist(),
            "action_normalizer_training_provenance": "unknown",
            "wall_seconds": time.monotonic() - start_time,
        },
        "rows": rows, "p1_admission": False,
    }
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "physical_regrets": [r["physical_regret"] for r in rows]}))


if __name__ == "__main__":
    main()
