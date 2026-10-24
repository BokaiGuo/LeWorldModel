"""CUDA-only audit of PointMaze predicted-rollout / encoded-target alignment.

P1 scores a predicted terminal visual latent against a latent encoding of an
observed future RGB frame.  This audit establishes the otherwise implicit
contract behind that comparison: one model action packet is exactly five raw
PointMaze actions and targets frame ``start + 5``; two packets target
``start + 10``.  It deliberately uses *recorded* actions, never physical
outcomes, and does not claim that a small prediction error is necessary for a
planner to be useful.  Its purpose is to catch a packet-layout, timestep, or
latent-coordinate defect before interpreting P1/P2's negative results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    ROOT,
    load_model_and_dataset,
    parameter_hash,
    prepare_archive_observation_for_model,
    require_cuda,
    sha256_file,
    terminal_latent,
    write_json,
)
from experiments.spectral_reachability.run_sra_p1_pointmaze_cuda import predicted_chart_context


def _roots(value: str) -> list[int]:
    roots = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(roots) < 4 or len(roots) != len(set(roots)):
        raise ValueError("need at least four distinct roots")
    return roots


def _pooled_visual(encoded: Any) -> torch.Tensor:
    """Pool the same visual tensor that P1 uses for its RGB goal latent."""

    visual = encoded["visual"]
    if visual.ndim != 6 or visual.shape[:2] != (1, 1):
        raise RuntimeError(f"unexpected encoded visual shape: {tuple(visual.shape)}")
    return visual[0, 0].mean(dim=(0, 1, 2)).detach()


def _encode_frame(model: Any, dataset: Any, root: int, frame: int, device: torch.device) -> torch.Tensor:
    raw, _, _, _, _ = dataset.get_frames(root, [frame])
    observation = {name: value.unsqueeze(0).to(device) for name, value in prepare_archive_observation_for_model(dataset, raw).items()}
    with torch.no_grad():
        return _pooled_visual(model.encode(observation))


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _l2_cosine(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise RuntimeError(f"predicted/encoded latent shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    l2 = torch.linalg.vector_norm(prediction - target)
    cosine = torch.nn.functional.cosine_similarity(prediction.unsqueeze(0), target.unsqueeze(0)).item()
    if not torch.isfinite(l2) or not np.isfinite(cosine):
        raise RuntimeError("non-finite predicted/encoded target comparison")
    return {"l2": float(l2), "cosine": float(cosine)}


def _neighbor_metrics(model: Any, dataset: Any, root: int, start: int, expected_frame: int, prediction: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Report +/- one-frame distances without treating model error as a layout gate."""

    offsets = [expected_frame - 1, expected_frame, expected_frame + 1]
    targets = {frame: _encode_frame(model, dataset, root, frame, device) for frame in offsets}
    comparisons = {str(frame): _l2_cosine(prediction, target) for frame, target in targets.items()}
    closest_frame = min(offsets, key=lambda frame: comparisons[str(frame)]["l2"])
    return {
        "expected_frame": expected_frame,
        "expected_offset_from_start": expected_frame - start,
        "target_sha256": _tensor_sha256(targets[expected_frame]),
        "prediction_vs_expected": comparisons[str(expected_frame)],
        "neighbor_comparisons": comparisons,
        "closest_of_expected_plus_minus_one": closest_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roots", default="1880,1881,1882,1883")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-packets", type=int, default=2, help="Audit every recorded-action rollout horizon from one through this many five-action packets.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    args = parser.parse_args()
    roots = _roots(args.roots)
    if args.output_dir.exists():
        raise FileExistsError(f"immutable audit output exists: {args.output_dir}")
    if args.start_frame < 0 or not 2 <= args.max_packets <= 4:
        raise ValueError("start-frame must be nonnegative and max-packets must be in [2, 4]")

    device = require_cuda()
    args.output_dir.mkdir(parents=True)
    model, dataset = load_model_and_dataset(args, device)
    before = parameter_hash(model)
    raw_action_dim = int(dataset.actions.shape[-1])
    if int(model.action_dim) % raw_action_dim:
        raise RuntimeError("model action packet is not divisible into raw actions")
    actions_per_packet = int(model.action_dim) // raw_action_dim
    configured_frameskip = int(model.model.frameskip)
    if actions_per_packet != configured_frameskip:
        raise RuntimeError(
            f"action packet has {actions_per_packet} raw actions but model frameskip is {configured_frameskip}"
        )

    rows: list[dict[str, Any]] = []
    for root in roots:
        required_last_frame = args.start_frame + args.max_packets * actions_per_packet + 1
        if required_last_frame >= int(dataset.get_seq_length(root)):
            raise ValueError(f"root {root} is too short for two-packet target audit")
        raw_context, _, _, _, _ = dataset.get_frames(root, [args.start_frame])
        model_context_raw = prepare_archive_observation_for_model(dataset, raw_context)
        context_obs = {name: value.unsqueeze(0).to(device) for name, value in model_context_raw.items()}
        with torch.no_grad():
            encoded_context = model.encode(context_obs)
        from tensordict import TensorDict

        context = TensorDict(
            {"visual": encoded_context["visual"], "proprio": encoded_context["proprio"]},
            batch_size=[1, 1],
            device=device,
        )
        raw_actions = dataset.actions[root, args.start_frame : args.start_frame + args.max_packets * actions_per_packet].to(device)
        packets = raw_actions.reshape(args.max_packets, -1)
        if packets.shape != (args.max_packets, int(model.action_dim)):
            raise RuntimeError(f"wrong packet shape: {tuple(packets.shape)}")
        horizons: dict[str, Any] = {}
        for horizon in range(1, args.max_packets + 1):
            with torch.no_grad():
                direct_prediction = terminal_latent(model, context, packets[:horizon].unsqueeze(1))[0]
                chart = context
                for packet_index in range(horizon):
                    sequential_prediction = terminal_latent(model, chart, packets[packet_index].reshape(1, 1, -1))[0]
                    if packet_index + 1 < horizon:
                        chart, _ = predicted_chart_context(model, chart, packets[packet_index])
            sequential_error = float((direct_prediction - sequential_prediction).abs().max())
            if not np.isfinite(sequential_error):
                raise RuntimeError(f"non-finite direct/sequential comparison at horizon {horizon}")
            horizons[str(horizon)] = {
                "packet_count": horizon,
                "target": _neighbor_metrics(model, dataset, root, args.start_frame, args.start_frame + horizon * actions_per_packet, direct_prediction, device),
                "direct_vs_sequential_max_abs_error": sequential_error,
            }
        row = {
            "root": root,
            "context_frame": args.start_frame,
            "context_rgb_sha256": _tensor_sha256(raw_context["visual"]),
            "model_input_rgb_range": [float(model_context_raw["visual"].min()), float(model_context_raw["visual"].max())],
            "model_input_proprio_sha256": _tensor_sha256(model_context_raw["proprio"]),
            "normalized_recorded_action_sha256": _tensor_sha256(raw_actions),
            "packet_shape": list(packets.shape),
            "packet_horizons": horizons,
        }
        # Keep v1 consumers readable while exposing every P3 horizon above.
        row["one_packet"] = horizons["1"]["target"]
        row["two_packet"] = horizons["2"]["target"]
        row["two_packet_direct_vs_sequential_max_abs_error"] = horizons["2"]["direct_vs_sequential_max_abs_error"]
        rows.append(row)
    after = parameter_hash(model)
    maxima = {
        "two_packet_direct_vs_sequential_max_abs_error": max(row["two_packet_direct_vs_sequential_max_abs_error"] for row in rows),
        "all_horizons_direct_vs_sequential_max_abs_error": max(
            item["direct_vs_sequential_max_abs_error"] for row in rows for item in row["packet_horizons"].values()
        ),
    }
    closest_counts = {
        "one_packet_expected_frame_closest_count": sum(row["one_packet"]["closest_of_expected_plus_minus_one"] == row["one_packet"]["expected_frame"] for row in rows),
        "two_packet_expected_frame_closest_count": sum(row["two_packet"]["closest_of_expected_plus_minus_one"] == row["two_packet"]["expected_frame"] for row in rows),
        "by_packet_horizon": {
            str(horizon): sum(
                row["packet_horizons"][str(horizon)]["target"]["closest_of_expected_plus_minus_one"]
                == row["packet_horizons"][str(horizon)]["target"]["expected_frame"]
                for row in rows
            )
            for horizon in range(1, args.max_packets + 1)
        },
    }
    gates = {
        "cuda_only": True,
        "frozen_parameter_hash_unchanged": before == after,
        "five_raw_actions_per_model_packet": actions_per_packet == 5,
        "packet_count_matches_model_frameskip": actions_per_packet == configured_frameskip,
        "direct_every_packet_horizon_equals_sequential_rollout_le_1e-6": maxima["all_horizons_direct_vs_sequential_max_abs_error"] <= 1e-6,
        "all_predicted_and_encoded_latents_finite": True,
    }
    report = {
        "schema": "sra_pointmaze_rollout_target_alignment.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "code/protocol target-coordinate and timestep audit; prediction fidelity is reported, not thresholded as a planner-effect result",
        "device": {"type": device.type, "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "inputs": {"roots": roots, "start_frame": args.start_frame, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "dataset": str(args.dataset_root)},
        "packet_contract": {"raw_action_dim": raw_action_dim, "model_action_dim": int(model.action_dim), "actions_per_packet": actions_per_packet, "model_frameskip": configured_frameskip, "max_packets": args.max_packets, "one_packet_target_offset": actions_per_packet, "two_packet_target_offset": 2 * actions_per_packet},
        "frozen_contract": {"parameter_hash_before": before, "parameter_hash_after": after},
        "maxima": maxima,
        "temporal_diagnostics": closest_counts,
        "gates": gates,
        "rows": rows,
        "claim_boundary": "A nearest-neighbor frame mismatch is a frozen-model temporal-fidelity observation, not by itself evidence of a packet or target-indexing bug. Any failed gate is a code/protocol stop for P1/P2 interpretation.",
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"output": str(args.output_dir), "status": report["status"], "gates": gates, "temporal_diagnostics": closest_counts, "maxima": maxima}, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise RuntimeError("rollout/target alignment audit failed; do not interpret P1/P2 effects")


if __name__ == "__main__":
    main()
