"""Host-CUDA physical audit of complete integration state before/after restore."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, install_missing_field_compat
from experiments.spectral_reachability.audit_reacher_d2p_runtime_support import _source_state_context
from experiments.spectral_reachability import reacher_d2p_context as context_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1:
        raise ValueError("positive repetitions and new output required")
    host = require_cuda_allocation("cuda")
    install_missing_field_compat()
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    records = []
    try:
        context = _source_state_context(env, dataset=args.dataset, seed=args.seed)
        expected = context["execution_snapshot"]
        physics = env.unwrapped.env.physics
        spec = expected["integration_state_spec"]
        sections = []
        start = 0
        for index in range(20):
            flag = 1 << index
            if flag & spec:
                size = mujoco.mj_stateSize(physics.model.ptr, flag)
                sections.append((mujoco.mjtState(flag).name, start, start + size))
                start += size
        for repetition in range(args.repetitions):
            root, goal = context_module.restore_execution_snapshot(env, seed=args.seed, snapshot=expected)
            actual = context_module.capture_execution_snapshot(env)
            old, new = np.asarray(expected["physics_integration_state"]), np.asarray(actual["physics_integration_state"])
            changed = {name: {"max_abs_error": float(np.abs(old[a:b] - new[a:b]).max()),
                              "expected": old[a:b].tolist(), "actual": new[a:b].tolist()}
                       for name, a, b in sections if not np.array_equal(old[a:b], new[a:b])}
            records.append({"repetition": repetition, "mismatched_snapshot_keys": [key for key in expected if expected[key] != actual.get(key)],
                "changed_integration_sections": changed, "root_rgb_exact": bool(np.array_equal(root, context["root_frame"])),
                "goal_rgb_exact": bool(np.array_equal(goal, context["goal"]))})
    finally:
        env.close()
    result = {"schema": "reacher_d2p_snapshot_state_audit.v1", "status": "completed", "seed": args.seed,
        "cuda_host": host, "rows": records,
        "all_integration_states_exact": all(not row["mismatched_snapshot_keys"] for row in records),
        "context_module_sha256": hashlib.sha256(Path(context_module.__file__).read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "claim_boundary": "Execution restore diagnostic, not model accuracy or P1 efficacy."}
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
