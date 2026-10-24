"""CUDA-gated equivalence audit for PointMaze SRA P1/P2 planner plumbing.

This audit is deliberately not a method comparison.  It makes silent
coordinate, batch-order, or one-step-vs-two-step evaluator errors fail before
any P1/P2 negative result can be interpreted.  It checks that:

* archive actions round-trip through the planner normalization unchanged;
* archived first-block endpoints match the receipt-bound physical evaluator;
* P1's two-call physical tree is exactly the same as direct ten-action replay;
* batched model rollouts agree with per-candidate rollouts; and
* a predicted intermediate chart composes to the same terminal latent as one
  two-step unroll (the released wrapper has a one-frame context window).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.spectral_reachability.pointmaze_physical_evaluator import (
    position_backend_receipt,
    replay_position_endpoints,
)
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    ROOT,
    load_model_and_dataset,
    make_root_context,
    parameter_hash,
    require_cuda,
    sha256_file,
    terminal_latent,
    write_json,
)
from experiments.spectral_reachability.run_sra_p1_pointmaze_cuda import (
    action_bank,
    block_to_physical,
    continuation_bank,
    predicted_chart_context,
)
from experiments.spectral_reachability.run_sra_p2_action_order_cuda import primitive_bank


def _roots(value: str) -> list[int]:
    roots = [int(item) for item in value.split(",") if item]
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("need at least two unique roots")
    return roots


def _p1_tree_equivalence(source: np.ndarray, first: list[np.ndarray], continuation: list[np.ndarray]) -> float:
    """Check P1's repeat/tile ordering against direct full-sequence replay."""

    first_array, continuation_array = np.stack(first), np.stack(continuation)
    middle = replay_position_endpoints(
        np.repeat(source.reshape(1, 4), len(first_array), axis=0), first_array
    )
    staged = replay_position_endpoints(
        np.repeat(middle, len(continuation_array), axis=0),
        np.tile(continuation_array, (len(first_array), 1, 1)),
    )
    direct_sequences = np.concatenate(
        (
            np.repeat(first_array, len(continuation_array), axis=0),
            np.tile(continuation_array, (len(first_array), 1, 1)),
        ),
        axis=1,
    )
    direct = replay_position_endpoints(
        np.repeat(source.reshape(1, 4), len(direct_sequences), axis=0), direct_sequences
    )
    return float(np.max(np.abs(staged - direct)))


def _model_batch_serial_error(model: Any, context: Any, blocks: torch.Tensor) -> float:
    batched = terminal_latent(model, context, blocks.unsqueeze(0))
    serial = torch.cat(
        [terminal_latent(model, context, block.reshape(1, 1, -1)) for block in blocks], dim=0
    )
    return float((batched - serial).abs().max())


def _model_chart_composition_error(model: Any, context: Any, first: torch.Tensor, continuation: torch.Tensor) -> float:
    chart, _ = predicted_chart_context(model, context, first)
    via_chart = terminal_latent(model, chart, continuation.reshape(1, 1, -1))[0]
    direct = terminal_latent(model, context, torch.stack((first, continuation)).reshape(2, 1, -1))[0]
    return float((via_chart - direct).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roots", default="1832,1840")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--physical-action-amplitude", type=float, default=0.8)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    args = parser.parse_args()
    roots = _roots(args.roots)
    if args.output_dir.exists():
        raise FileExistsError(f"immutable audit output exists: {args.output_dir}")
    if args.start_frame < 0:
        raise ValueError("start frame must be nonnegative")

    device = require_cuda()
    backend = position_backend_receipt()
    args.output_dir.mkdir(parents=True)
    model, dataset = load_model_and_dataset(args, device)
    before = parameter_hash(model)
    tolerance = float(backend["position_label_contract"]["position_tolerance"])
    rows = []
    for root in roots:
        context, _goal, _observed = make_root_context(model, dataset, root, args.start_frame, 2, device)
        labels, first_blocks = action_bank(dataset, root, args.start_frame, args.physical_action_amplitude, device)
        _continuation_labels, continuation_blocks = continuation_bank(dataset, args.physical_action_amplitude, device)
        _primitive_labels, primitives = primitive_bank(dataset, args.physical_action_amplitude, device)
        observed_normalized = dataset.actions[root, args.start_frame : args.start_frame + 5]
        observed_round_trip = block_to_physical(first_blocks[-1], dataset)
        observed_re_normalized = (observed_round_trip - dataset.action_mean.cpu().numpy()) / dataset.action_std.cpu().numpy()
        source = dataset.states[root, args.start_frame].detach().cpu().numpy().astype(np.float64)
        archived_target = dataset.states[root, args.start_frame + 5].detach().cpu().numpy().astype(np.float64)
        observed_endpoint = replay_position_endpoints(source.reshape(1, 4), observed_round_trip.reshape(1, 5, 2))[0]
        first_physical = [block_to_physical(block, dataset) for block in first_blocks]
        continuation_physical = [block_to_physical(block, dataset) for block in continuation_blocks]
        p2_sequences = torch.stack(
            [torch.stack((primitives[0], primitives[2])), torch.stack((primitives[2], primitives[0]))],
            dim=1,
        )
        p2_batch = terminal_latent(model, context, p2_sequences)
        p2_serial = torch.cat(
            [terminal_latent(model, context, p2_sequences[:, index : index + 1]) for index in range(p2_sequences.shape[1])],
            dim=0,
        )
        composition_errors = [
            _model_chart_composition_error(model, context, first, continuation_blocks[index])
            for first in first_blocks
            for index in (0, 5, 13)
        ]
        rows.append({
            "root": root,
            "candidate_labels": labels,
            "observed_action_normalized_round_trip_max_abs_error": float(np.max(np.abs(observed_re_normalized - observed_normalized.detach().cpu().numpy()))),
            "observed_first_block_position_l2_error": float(np.linalg.norm(observed_endpoint[:2] - archived_target[:2])),
            "p1_two_call_vs_direct_physical_max_abs_error": _p1_tree_equivalence(source, first_physical, continuation_physical),
            "p1_model_batch_vs_serial_terminal_max_abs_error": _model_batch_serial_error(model, context, first_blocks),
            "p1_chart_vs_direct_two_step_terminal_max_abs_error": max(composition_errors),
            "p2_model_batch_vs_serial_terminal_max_abs_error": float((p2_batch - p2_serial).abs().max()),
        })
    after = parameter_hash(model)
    maxima = {key: max(float(row[key]) for row in rows) for key in rows[0] if key not in {"root", "candidate_labels"}}
    gates = {
        "cuda_only": True,
        "frozen_parameter_hash_unchanged": before == after,
        "observed_action_normalization_round_trip_le_1e-6": maxima["observed_action_normalized_round_trip_max_abs_error"] <= 1e-6,
        "observed_archive_position_replay_le_contract": maxima["observed_first_block_position_l2_error"] <= tolerance,
        "p1_two_call_tree_equals_direct_replay_le_1e-10": maxima["p1_two_call_vs_direct_physical_max_abs_error"] <= 1e-10,
        # P1 production uses serial candidate scoring; this only verifies that
        # optional batching is numerically equivalent within CUDA float32
        # reduction noise, not bitwise identical.
        "p1_model_batch_equals_serial_le_5e-6": maxima["p1_model_batch_vs_serial_terminal_max_abs_error"] <= 5e-6,
        "p1_chart_composition_equals_direct_le_1e-6": maxima["p1_chart_vs_direct_two_step_terminal_max_abs_error"] <= 1e-6,
        "p2_model_batch_equals_serial_le_1e-6": maxima["p2_model_batch_vs_serial_terminal_max_abs_error"] <= 1e-6,
    }
    report = {
        "schema": "sra_pointmaze_planner_physical_alignment.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "code/protocol equivalence audit; not a method-effect or planning-success experiment",
        "device": {"type": device.type, "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "inputs": {"roots": roots, "start_frame": args.start_frame, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "dataset": str(args.dataset_root), "physical_position_backend": backend},
        "frozen_contract": {"parameter_hash_before": before, "parameter_hash_after": after},
        "maxima": maxima,
        "gates": gates,
        "rows": rows,
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"output": str(args.output_dir), "status": report["status"], "maxima": maxima, "gates": gates}, indent=2, sort_keys=True))
    if not all(gates.values()):
        raise RuntimeError("planner/physical alignment audit failed; do not interpret P1/P2 effects")


if __name__ == "__main__":
    main()
