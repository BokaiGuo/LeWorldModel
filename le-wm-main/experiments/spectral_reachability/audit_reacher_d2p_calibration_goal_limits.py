"""Check whether the goal-limit bug explains the ORIGINAL six margin roots.

Only simulator/goal metadata are replayed here; existing RGB ranking evidence
is hash-linked, not recomputed or relabeled. CUDA allocation is mandatory,
but no neural efficacy or CPU neural experiment is claimed by this audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference, fidelity = (json.loads(p.read_text()) for p in (args.reference, args.fidelity))
    evidence = {r["seed"]: r for r in fidelity["rows"]}
    seeds = sorted(r["seed"] for r in reference["rows"])
    if (len(seeds) != 6 or len(set(seeds)) != 6 or sorted(evidence) != seeds or len(evidence) != len(fidelity["rows"])
            or reference["protocol"]["warmup_policy"] != execution.WARMUP_POLICY):
        raise ValueError("the original six complete unique uniform-warmup roots required")
    print("Replay original6 root/goal RGB and actual wrist limits. Existing ranking is not rerun; no mechanism or theory verdict. CUDA allocation mandatory; no CPU neural fallback.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(reference["protocol"]["target_delta"]))
    rows = []
    try:
        for old in sorted(reference["rows"], key=lambda r: r["seed"]):
            seed = old["seed"]
            context = execution.collect_executed_context(env, seed=seed, history_len=3, action_block=5)
            if any(context[k] != old["executed_context"][k] for k in ("root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")):
                raise RuntimeError("original margin calibration RGB/action context no longer exact")
            physics = env.unwrapped.env.physics
            limits, limited = np.asarray(physics.model.jnt_range), np.asarray(physics.model.jnt_limited, dtype=bool)
            target = np.asarray(env.unwrapped.env.task.target_qpos)
            invalid = [int(j) for j in range(2) if limited[j] and not limits[j, 0] <= target[j] <= limits[j, 1]]
            e = evidence[seed]
            rows.append({"seed": seed, "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"],
                "target_qpos": target.tolist(), "limited_joints": limited.tolist(), "joint_ranges": limits.tolist(),
                "invalid_goal_joints": invalid, "original_model_margin": e["model_candidate_minus_zero_goal_cost"],
                "original_observed_margin": e["observed_candidate_minus_zero_goal_cost"],
                "original_order_agrees": e["model_to_observation_goal_order_agrees"], "exact_context_replay": True})
    finally:
        env.close()
    invalid = [r["seed"] for r in rows if r["invalid_goal_joints"]]
    reversed_legal = [r["seed"] for r in rows if not r["invalid_goal_joints"] and not r["original_order_agrees"]]
    write_new(args.output, {"schema": "reacher_d2p_original_calibration_goal_limits.v1", "status": "completed", "cuda_host": host,
        "reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "fidelity_path": str(args.fidelity.resolve()), "fidelity_sha256": digest(args.fidelity), "auditor_sha256": digest(Path(__file__)),
        "rows": rows, "invalid_goal_roots": invalid, "reversed_legal_goal_roots": reversed_legal,
        "scope": "Exact simulator/RGB-root metadata replay; historical ranking linked, no new neural efficacy result; do not overgeneralize goal bug."})
    print(json.dumps({"invalid_goal_roots": invalid, "reversed_legal_goal_roots": reversed_legal}), flush=True)


if __name__ == "__main__":
    main()
