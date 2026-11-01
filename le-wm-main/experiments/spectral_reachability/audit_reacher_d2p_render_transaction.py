"""Diagnose root RGB before/after snapshot's temporary goal-render transaction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, install_missing_field_compat
from experiments.spectral_reachability.audit_reacher_d2p_runtime_support import _source_state_context
from experiments.spectral_reachability.reacher_d2p_context import restore_execution_snapshot, rgb_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1:
        raise ValueError("positive repetitions and new output required")
    host = require_cuda_allocation("cuda")
    install_missing_field_compat()
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    rows = []
    try:
        context = _source_state_context(env, dataset=args.dataset, seed=args.seed)
        raw = env.unwrapped
        physics = raw.env.physics
        for repetition in range(args.repetitions):
            returned, goal = restore_execution_snapshot(env, seed=args.seed, snapshot=context["execution_snapshot"])
            state_before = np.asarray(physics.get_state()).copy()
            frames = [np.asarray(env.render(), np.uint8).copy() for _ in range(3)]
            state_after = np.asarray(physics.get_state()).copy()
            reference = np.asarray(context["root_frame"], np.uint8)
            rows.append({"repetition": repetition, "returned_root_sha256": rgb_digest(returned),
                "post_transaction_rgb_sha256": [rgb_digest(frame) for frame in frames],
                "returned_root_pixel_max_error": int(np.abs(returned.astype(np.int16) - reference).max()),
                "post_transaction_pixel_max_error": [int(np.abs(frame.astype(np.int16) - reference).max()) for frame in frames],
                "post_transaction_changed_pixel_components": [int(np.count_nonzero(frame != reference)) for frame in frames],
                "goal_matches": bool(np.array_equal(goal, context["goal"])),
                "rendering_physics_state_max_error": float(np.abs(state_before - state_after).max()),
                "restored_physics_state_max_error": float(np.abs(state_before - np.asarray(context["execution_snapshot"]["physics_state"])).max())})
    finally:
        env.close()
    result = {"schema": "reacher_d2p_render_transaction.v1", "status": "completed", "cuda_host": host,
        "seed": args.seed, "source_episode": context["source_episode"], "source_start_row": context["source_start_row"],
        "aggregate": {"max_returned_error": max(row["returned_root_pixel_max_error"] for row in rows),
                      "max_post_transaction_error": max(value for row in rows for value in row["post_transaction_pixel_max_error"]),
                      "max_physics_state_error": max(row["restored_physics_state_max_error"] for row in rows)},
        "rows": rows, "claim_boundary": "Render/restore code diagnostic only; no learned-model or efficacy claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
