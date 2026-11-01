"""Fixed-six-root 2x2 parameter/BatchNorm-buffer causal diagnostic on CUDA.

Only pred_proj running statistics are exchanged, only in memory. Original
weights remain immutable. No training, selection or P1 admission is performed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_p1_first_chunk_fidelity as fidelity
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.audit_reacher_d2p_original_margin_checkpoint import validate_reference
from experiments.spectral_reachability.analyze_reacher_d2p_original_margin_checkpoint import summarize as validate_crossover
from experiments.spectral_reachability.reacher_d2p_margin_decomposition import decompose_margin
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def swap_prediction_bn_buffers(model, state):
    buffers = dict(model.named_buffers())
    keys = sorted(k for k in buffers if k.startswith("pred_proj.") and k.endswith(("running_mean", "running_var", "num_batches_tracked")))
    if not keys or len(keys) % 3:
        raise ValueError("complete prediction-projection BatchNorm statistics required")
    if any(k not in state or state[k].shape != buffers[k].shape or state[k].dtype != buffers[k].dtype for k in keys):
        raise ValueError("source running statistics do not match destination model")
    with torch.no_grad():
        for key in keys:
            buffers[key].copy_(state[key].to(device=buffers[key].device))
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossover", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.crossover.read_text())
    validate_crossover(reference)
    source_inputs = reference["sealed_protocol"]["inputs"]
    paths = {k: Path(v["path"]) for k, v in source_inputs.items()}
    if any(digest(paths[k]) != v["sha256"] for k, v in source_inputs.items()):
        raise ValueError("original evidence/checkpoint inputs changed")
    report, prior = (json.loads(paths[k].read_text()) for k in ("reference", "prior_fidelity"))
    seeds = validate_reference(report, prior)
    protocol_path, implementation = args.output.with_suffix(".protocol.json"), args.output.with_suffix(".implementation")
    if any(p.exists() for p in (args.output, protocol_path, implementation)):
        raise FileExistsError("new immutable outputs required")
    modules = {"auditor": Path(__file__), "context": Path(execution.__file__), "rollout": Path(p0.__file__),
        "fidelity": Path(fidelity.__file__), "decomposition": Path(decompose_margin.__code__.co_filename),
        "loader": Path(shared.load_compatible_pretrained.__code__.co_filename),
        "model": Path(shared.load_compatible_pretrained.__code__.co_filename)}
    protocol = {"crossover_path": str(args.crossover.resolve()), "crossover_sha256": digest(args.crossover),
        "inputs": source_inputs, "root_seeds": seeds, "parameter_sources": ["receipt", "official"],
        "prediction_bn_sources": ["receipt", "official"], "physical_labels_used_for_planning": False,
        "intervention": "in-memory pred_proj running_mean/running_var/num_batches_tracked only; parameters and all other buffers fixed",
        "implementation_sha256": {k: digest(v) for k, v in modules.items()},
        "scope": "Known-root 2x2 causal checkpoint decomposition; saved RGB successors are evaluator-only; no selection, training, efficacy or P1 admission."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(protocol_path, protocol)
    implementation.mkdir()
    for k, p in modules.items():
        shutil.copyfile(p, implementation / (k + ".py"))
    print("HOST CUDA: all original six roots, fixed original actions, 2x2 parameter / pred_proj BatchNorm statistics. In-memory variants only; no training or P1 admission.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(paths["adapter"], "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(reference["sealed_protocol"]["target_delta"]))
    contexts, rows = {}, []
    old_rows = {r["seed"]: r for r in report["rows"]}
    cross_rows = {r["seed"]: r for r in reference["rows"]}
    try:
        for seed in seeds:
            ctx = execution.collect_executed_context(env, seed=seed, history_len=3, action_block=5)
            if any(ctx[k] != old_rows[seed]["executed_context"][k] for k in ("root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")):
                raise RuntimeError("original root/goal/actions no longer exact")
            contexts[seed] = p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"], np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))
            rows.append({"seed": seed, "variants": {}})
    finally:
        env.close()
    states = {name: torch.load(paths[name + "_weights"], map_location="cuda", weights_only=True) for name in ("receipt", "official")}
    gates_list, sample_count, swapped = [], 0, []
    for params_source in ("receipt", "official"):
        model = shared.load_compatible_pretrained(paths[params_source + "_weights"].parent, device="cuda").eval()
        model.requires_grad_(False)
        param_hash = shared._parameter_hash(model)
        original_buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        for bn_source in ("receipt", "official"):
            keys = swap_prediction_bn_buffers(model, states[bn_source])
            swapped.append(keys)
            for row in rows:
                seed = row["seed"]
                pixels, goal, history = contexts[seed]
                method = old_rows[seed]["methods"]["reconfiguration"]
                target = p0._goal_latent(model, goal)
                expected = cross_rows[seed]["checkpoints"][params_source + "_checkpoint"]["float32"]
                if target.tolist() != expected["vectors"]["goal"]:
                    raise RuntimeError("visual goal chart changed")
                predicted = {}
                for branch, prefix in (("candidate", "first"), ("zero", "zero_prefix")):
                    action = torch.as_tensor(method[prefix + "_action_witness_normalized"], device="cuda", dtype=torch.float32)
                    predicted[branch] = fidelity._first_successor_prediction(model, pixels, history, action)
                    sample_count += 1
                actual, actual_zero = (torch.tensor(expected["vectors"][k], device="cuda", dtype=torch.float32) for k in ("observed", "observed_zero"))
                vectors = {"prediction": predicted["candidate"], "prediction_zero": predicted["zero"],
                    "observed": actual, "observed_zero": actual_zero, "goal": target}
                cell = {"vectors": {k: v.tolist() for k, v in vectors.items()},
                    "costs": {k: float((v - target).square().mean()) for k, v in vectors.items() if k != "goal"},
                    "decomposition": decompose_margin(*vectors.values()),
                    "candidate_latent_l2": float((predicted["candidate"] - actual).norm()),
                    "zero_latent_l2": float((predicted["zero"] - actual_zero).norm())}
                if params_source == bn_source and any(cell["vectors"][k] != expected["vectors"][k] for k in vectors):
                    raise RuntimeError("unmodified original checkpoint did not replay exactly")
                row["variants"][params_source + "_params/" + bn_source + "_bn"] = cell
                print(json.dumps({"seed": seed, "params": params_source, "bn": bn_source,
                    "margin": cell["decomposition"]["predicted_margin"]}), flush=True)
            gates_list.append(shared._parameter_hash(model) == param_hash
                and all(torch.equal(v, original_buffers[k]) for k, v in model.named_buffers() if k not in keys)
                and not any(m.training for m in model.modules()))
        del model
    summary = {}
    for name in rows[0]["variants"]:
        cells = [r["variants"][name] for r in rows]
        summary[name] = {"order_agreements": sum(c["decomposition"]["predicted_margin"] * c["decomposition"]["observed_margin"] > 0 for c in cells),
            "mean_candidate_latent_l2": float(np.mean([c["candidate_latent_l2"] for c in cells])),
            "mean_zero_latent_l2": float(np.mean([c["zero_latent_l2"] for c in cells]))}
    gates = {"cuda_only": True, "all6_fixed_roots": [r["seed"] for r in rows] == seeds,
        "all4_variants_each_root": all(len(r["variants"]) == 4 for r in rows), "both_native_baselines_bit_exact": True,
        "only_declared_bn_buffers_changed": all(gates_list), "forward_sample_accounting": sample_count == 48,
        "inputs_unchanged": all(digest(paths[k]) == v["sha256"] for k, v in source_inputs.items()),
        "crossover_unchanged": digest(args.crossover) == protocol["crossover_sha256"],
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_original_margin_bn.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "protocol_sha256": digest(protocol_path), "gates": gates,
        "rows": rows, "summary": summary, "actual_model_predict_samples": sample_count, "swapped_buffer_keys": sorted(set(k for group in swapped for k in group))})
    print(json.dumps({"gates": gates, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
