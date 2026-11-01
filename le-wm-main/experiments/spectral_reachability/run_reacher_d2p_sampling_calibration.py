"""Known two-root frozen Reacher sampling calibration, never P1 or efficacy.

Run as a module from le-wm-main. Imports of neural/runtime dependencies are
deferred to run(); importing this module and testing orchestration is CPU-safe.
No HDF5 access, fitting, downloads, readouts, or outcome-root selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SEEDS = (20261040, 20261041)
HORIZON, HISTORY, ACTION_BLOCK = 2, 3, 5
TARGET_DELTA = (.65, -.65)
PARAM_LIMIT = 3.


def sample_parameters(method, cost, *, action_width, seed, kernels):
    """The kernel box is PRE-TANH parameters, NOT raw/normalized actions."""
    common = dict(initial=np.zeros((HORIZON, action_width)), lower=-PARAM_LIMIT,
                  upper=PARAM_LIMIT, samples=32, seed=seed)
    if method == "clipped_gaussian_mppi_heuristic":
        return kernels.mppi(cost, **common, iterations=4, temperature=1., noise_scale=1.)
    if method == "bestofn_random":
        return kernels.random_shooting(cost, **common)
    raise ValueError("unsupported sampling method")


def timed_call(call, *, synchronize, clock=time.perf_counter):
    synchronize()
    started = clock()
    result = call()
    synchronize()
    return result, clock() - started


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_new(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(raw)


def run(*, policy, adapter_artifact, output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

    policy, adapter_artifact, output = Path(policy), Path(adapter_artifact), Path(output)
    inputs = {str(p.resolve()): _digest(p) for p in
              (policy / "weights.pt", policy / "config.json", adapter_artifact)}
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *[root / name for name in (
        "spectral_reachability/d2p_sampling_baselines.py",
        "spectral_reachability/d2p_cuda_sampling_cost.py",
        "spectral_reachability/d2p_model_call_instrumentation.py",
        "spectral_reachability/d2p_prerun_source_seal.py",
        "spectral_reachability/run_reacher_d2p_p0_goal_progress.py",
        "spectral_reachability/preflight_reacher_d2p_p0.py",
        "spectral_reachability/reacher_d2p_context.py",
        "spectral_reachability/reacher_d2p_packet_support.py",
        "spectral_reachability/reacher_d2p_qpos_readout.py",
        "spectral_reachability/d2p_stage_admission.py",
        "spectral_reachability/build_reacher_d2p_training_action_stats.py",
        "audit_live_reacher_shared_candidate_ranking.py", "checkpoint_compat.py",
        "online_planning_protocol.py", "run_live_cube_action_adapter_planning.py",
        "run_live_reacher_planning.py", "run_live_reacher_sequence_candidate_planning.py",
        "gpu_preflight.py")]]
    protocol = {"seeds": list(SEEDS), "history_len": HISTORY, "future_horizon": HORIZON,
                "action_block": ACTION_BLOCK, "target_delta": list(TARGET_DELTA),
                "split": "known_calibration_only", "context_support": "uniform",
                "sampling_parameter_box": [-PARAM_LIMIT, PARAM_LIMIT],
                "sampling_coordinate": "pre_tanh_parameters_not_raw_action_box",
                "cem_parameter_domain": "P0_unbounded_Gaussian_pre_tanh_parameters",
                "methods": ["clipped_gaussian_mppi_heuristic", "bestofn_random", "cem"],
                "samples": 32, "mppi_iterations": 4, "temperature": 1., "noise_scale": 1.,
                "cem_steps": 4, "cem_elite": 4, "inputs": inputs,
                "P1_admission": False, "efficacy_claim": False, "fair_budget_claim": False,
                "readout": "none; frozen original latent objective",
                "mppi_claim_boundary": "clipped Gaussian heuristic without importance correction; not faithful MPPI; negative results cannot exclude MPPI",
                "budget_policy": "count all initial/proposal/updated-nominal scoring; no candidate-count matching"}
    output.mkdir(exist_ok=False)
    source_seal = create_seal(files=sources, args_json=json.dumps(protocol),
                              output_dir=output / "prerun_source_seal")
    _write_new(output / "protocol.json", protocol)
    # Only after the source seal exists may runtime allocation/model loading occur.
    import torch
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
    from experiments.spectral_reachability import d2p_sampling_baselines as kernels
    from experiments.spectral_reachability.d2p_cuda_sampling_cost import CudaSamplingCost
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation

    host = require_cuda_allocation("cuda")
    device = torch.device("cuda")
    p0.install_missing_field_compat()
    p0.validate_training_semantic_receipt(adapter_artifact)
    mean, std = p0._load_action_stats(adapter_artifact, "swm/ReacherDMControl-v0")
    model = p0.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = p0._parameter_hash(model)
    buffers = {key: value.detach().clone() for key, value in model.named_buffers()}
    preprocessor = p0._Preprocessor(p0.ActionNormalizer(mean, std))
    env = p0.ReacherGoalImageWrapper(p0.gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                    target_delta=TARGET_DELTA)
    rows = []
    try:
        p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=ACTION_BLOCK)
        low, high = p0._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = (torch.as_tensor(v, device=device) for v in (low, high))
        for seed in SEEDS:
            context = p0._root_context(env, seed, HISTORY, ACTION_BLOCK)
            pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"],
                context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
            methods = {}
            for name in protocol["methods"]:
                bridge = CudaSamplingCost(model, pixels, goal, history, action_lower=lower,
                                          action_upper=upper, future_horizon=HORIZON)
                def plan():
                    if name == "cem":
                        return p0._cem(model, pixels, goal, history, future_horizon=HORIZON,
                            action_lower=lower, action_upper=upper, samples=32, steps=4, elite=4, seed=seed)
                    params, receipt = sample_parameters(name, bridge, action_width=history.shape[-1],
                                                       seed=seed, kernels=kernels)
                    return torch.as_tensor(params[None], dtype=history.dtype, device=device), receipt
                with PredictInstrumentation(model) as counter:
                    with counter.phase("planning", method=name, horizon_tokens=HORIZON):
                        (params, receipt), seconds = timed_call(plan, synchronize=lambda: torch.cuda.synchronize(device))
                receipt = {**receipt, "actual_planning_predict": counter.records[0],
                           "planner_synchronized_wall_seconds": seconds,
                           "candidate_bridge": bridge.receipt() if name != "cem" else None,
                           "accounting_scope": "planning predict only; encoder/backward excluded; no equal-compute claim"}
                # Persist the action witness before any physical evaluator labels.
                raw, normalized = p0._to_raw_actions(params, mean, std, lower, upper,
                                                      env.action_space.low, env.action_space.high)
                witness = {"seed": seed, "method": name, "params": params.detach().cpu().tolist(),
                           "raw_actions": raw.tolist(), "normalized_actions": normalized.tolist(),
                           "receipt": receipt, "root_rgb_sha256": context["root_rgb_sha256"],
                           "goal_rgb_sha256": context["goal_rgb_sha256"]}
                _write_new(output / f"seed_{seed}.{name}.plan.json", witness)
                physical, identity = p0._execute(env, seed, raw, expected_frame=context["root_frame"],
                    expected_goal=context["goal"], execution_snapshot=context["execution_snapshot"])
                if not identity or not all(np.isfinite(v) for v in physical.values()):
                    raise RuntimeError("physical execution identity or finite-label check failed")
                methods[name] = {**witness, "physical": physical, "execution_identity": identity}
            row = {"seed": seed, "split": "calibration", "execution_snapshot": context["execution_snapshot"],
                   "executed_context": {k: context[k] for k in ("root_rgb_sha256", "goal_rgb_sha256",
                       "history_rgb_sha256", "warmup_actions_sha256", "warmup_policy")}, "methods": methods}
            _write_new(output / f"seed_{seed}.json", row)
            rows.append(row)
    finally:
        env.close()
    current_buffers = dict(model.named_buffers())
    gates = {"parameters_unchanged": before == p0._parameter_hash(model),
             "buffers_unchanged": set(buffers) == set(current_buffers) and
                 all(torch.equal(value, current_buffers[key]) for key, value in buffers.items()),
             "all_eval": all(not module.training for module in model.modules()),
             "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
             "input_hashes_unchanged": all(_digest(path) == digest for path, digest in inputs.items()),
             "all_predict_scopes_completed": all(m["receipt"]["actual_planning_predict"]["completed"]
                 and m["receipt"]["actual_planning_predict"]["predict_samples"] > 0
                 for row in rows for m in row["methods"].values()),
             "sampling_candidate_counts_match_bridge": all(
                 m["receipt"]["cost_evaluations"] == m["receipt"]["candidate_bridge"]["candidate_cost_samples"]
                 and m["receipt"]["cost_calls"] == m["receipt"]["candidate_bridge"]["candidate_cost_calls"]
                 for row in rows for name, m in row["methods"].items() if name != "cem")}
    verification = verify_seal(output_dir=source_seal["output_dir"],
                               expected_manifest_sha256=source_seal["manifest_sha256"])
    report = {"schema": "reacher_d2p_sampling_calibration.v1", "status": "calibration_completed"
              if all(gates.values()) else "integrity_fail", "protocol": protocol, "host": host,
              "gates": gates, "rows": rows, "P1_admission": False, "efficacy_claim": False,
              "fair_budget_claim": False, "prerun_source_seal": source_seal,
              "source_verification": verification, "comparisons": {}}
    _write_new(output / "report.json", report)
    if not all(gates.values()):
        raise RuntimeError("sampling calibration integrity failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(policy=args.policy, adapter_artifact=args.adapter_artifact, output=args.output)


if __name__ == "__main__":
    main()
