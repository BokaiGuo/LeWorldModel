"""Fixed-coefficient past-only residual correction of sealed multi-packet plans.

No training, parameter selection or physical rollout. Predictions are sealed
before joining the stored real successors; all eight known roots retained.
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
from experiments.spectral_reachability import reacher_d2p_causal_innovation as causal
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.analyze_reacher_d2p_packet_rollout_fidelity import summarize as validate_packet_reference
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


VARIANTS = ("raw", "terminal_unit", "recurrent_unit")


def summarize(predictions, observed):
    seeds = [r["seed"] for r in predictions["rows"]]
    known = {r["seed"]: r for r in observed["rows"]}
    if len(seeds) != 8 or len(set(seeds)) != 8 or sorted(known) != sorted(seeds) or len(known) != len(observed["rows"]):
        raise ValueError("all eight duplicate-free fixed root identities required")
    output, errors = [], {k: [] for k in VARIANTS}
    for row in sorted(predictions["rows"], key=lambda r: r["seed"]):
        if set(row["branches"]) != {"native_dispatch", "zero"}:
            raise ValueError("both sealed branches required")
        target = np.asarray(row["goal_vector"], np.float64)
        delta = np.asarray(row["innovation_vector"], np.float64)
        if any(v.shape != (192,) or not np.isfinite(v).all() for v in (target, delta)):
            raise ValueError("finite complete goal and past-only residual vectors required")
        costs = {}
        for family, branch in row["branches"].items():
            if set(branch) != set(VARIANTS):
                raise ValueError("all fixed correction variants required")
            reference = known[row["seed"]]["branches"][family]["packets"]
            matrices = {k: np.asarray(v, np.float64) for k, v in branch.items()}
            if any(v.shape != (4, 192) or not np.isfinite(v).all() for v in matrices.values()):
                raise ValueError("complete four-packet predictions required")
            if branch["raw"] != [p["vectors"]["free"] for p in reference]:
                raise ValueError("raw baseline must replay bit exactly")
            if any(p["vectors"]["goal"] != row["goal_vector"] for p in reference):
                raise ValueError("goal coordinate changed")
            if not np.allclose(matrices["terminal_unit"], matrices["raw"] + delta, rtol=0, atol=1e-6):
                raise ValueError("terminal unit residual rule changed")
            if branch["recurrent_unit"][0] != branch["terminal_unit"][0]:
                raise ValueError("first corrected prediction must coincide")
            actual = np.asarray([p["vectors"]["observed"] for p in reference], np.float64)
            costs[family] = {k: np.mean((v - target) ** 2, axis=1).tolist() for k, v in matrices.items()}
            costs[family]["observed"] = np.mean((actual - target) ** 2, axis=1).tolist()
            for k, v in matrices.items():
                errors[k].append(np.linalg.norm(v - actual, axis=1).tolist())
        margins = {k: [a - b for a, b in zip(costs["native_dispatch"][k], costs["zero"][k])] for k in (*VARIANTS, "observed")}
        output.append({"seed": row["seed"], "innovation_l2": float(np.linalg.norm(delta)), "margins": margins})
    return {"rows": output, "mean_latent_l2_by_packet": {k: np.mean(v, axis=0).tolist() for k, v in errors.items()},
        "reversed_final_roots": {k: [r["seed"] for r in output if r["margins"][k][-1] * r["margins"]["observed"][-1] < 0] for k in VARIANTS},
        "scope": "Known roots, fixed coefficients and fixed plans; correction diagnosis only, no physical benefit, new-root fidelity or P1 admission."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet_reference.read_text())
    validated = validate_packet_reference(packet)
    old = packet["sealed_protocol"]
    if (digest(args.packet_reference.with_suffix(".protocol.json")) != packet["protocol_sha256"]
            or json.loads(args.packet_reference.with_suffix(".protocol.json").read_text()) != old):
        raise ValueError("strict packet protocol seal required")
    prefix_path = Path(old["reference_path"]).with_suffix(".selection.json")
    if digest(prefix_path) != old["prefix_sha256"]:
        raise ValueError("sealed initial action bank changed")
    prefix = json.loads(prefix_path.read_text())
    choices = {r["seed"]: r for r in prefix["rows"]}
    if sorted(choices) != validated["root_seeds"] or len(choices) != len(prefix["rows"]):
        raise ValueError("complete prefix root identity join required")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if digest(args.policy / "weights.pt") != old["weights_sha256"] or digest(args.adapter_artifact) != old["adapter_sha256"] or source_sha != old["source_sha256"]:
        raise ValueError("official checkpoint, normalization and source required")
    paths = {k: args.output.with_suffix(ext) for k, ext in (("protocol", ".protocol.json"), ("predictions", ".predictions.json"), ("implementation", ".implementation"))}
    if any(p.exists() for p in (args.output, *paths.values())):
        raise FileExistsError("new immutable diagnostic outputs required")
    modules = {"auditor": Path(__file__), "causal": Path(causal.__file__), "p0": Path(p0.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "goal_contract": Path(goals.__file__)}
    for k in ("p0", "source_context", "goal_contract"):
        if digest(modules[k]) != old["implementation_sha256"][k]:
            raise ValueError("reference implementation changed")
    protocol = {"root_seeds": validated["root_seeds"], "packet_reference_path": str(args.packet_reference.resolve()), "packet_reference_sha256": digest(args.packet_reference),
        "prefix_sha256": digest(prefix_path), "weights_sha256": old["weights_sha256"], "adapter_sha256": old["adapter_sha256"], "source_sha256": source_sha,
        "goal_contract": "joint-feasible-v1", "target_delta": old["target_delta"], "coefficient": 1., "coefficient_fitted": False,
        "variants": list(VARIANTS), "candidate_successor_used_for_prediction": False, "predictions_sealed_before_observation_join": True,
        "physical_execution": False, "source_verification": "Reuse existing official SHA-bound source receipt, no new complete-file SHA claimed.",
        "implementation_sha256": {k: digest(p) for k, p in modules.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["implementation"].mkdir()
    for k, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (k + ".py"))
    print("HOST CUDA: ALL8 known roots, exact sealed candidate/neutral plans. Fixed unit past-only residual once at terminal or after every model step. No future images, parameter fit, physical execution or P1 admission.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    params_before, buffers_before = shared._parameter_hash(model), {k: v.clone() for k, v in model.named_buffers()}
    predict, samples = model.predict, 0

    def counted(emb, actions):
        nonlocal samples
        if emb.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return predict(emb, actions)

    model.predict = counted
    env = goals.goal_wrapper("joint-feasible-v1")(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(old["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = (torch.as_tensor(v, device="cuda") for v in (lo, hi))
    rows = []
    try:
        with torch.no_grad():
            for seed in protocol["root_seeds"]:
                ctx = shared._source_state_context(env, dataset=dataset, seed=seed)
                choice = choices[seed]
                if any(ctx[k] != choice[k] for k in ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256")):
                    raise RuntimeError("root/goal identity changed")
                pixels, goal, history = p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"], np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))
                emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                past = processor._prepare_info({"pixels": np.asarray(ctx["frames"], np.uint8)[None], "goal": ctx["goal"][None, None], "action": np.asarray(ctx["history_action_blocks"], np.float32)[None]})
                innovation, receipt = causal.innovation_from_executed_history(model, past["pixels"].cuda(), past["action"].cuda(), emb[0, -1])
                bank = torch.tensor(choice["word_params"], device="cuda")
                branches = {}
                for family, index in (("native_dispatch", choice["native_decision"]["selected_program_index"]), ("zero", 0)):
                    future = p0._bounded_normalized_actions(bank[index:index + 1], lower, upper)
                    values = {}
                    for variant in ("raw", "recurrent_unit"):
                        rolling = emb
                        actions = torch.cat((history[:, 1:], future[:, :1]), dim=1)
                        vectors = []
                        for i in range(4):
                            predicted = model.predict(rolling[:, -3:], model.action_encoder(actions[:, -3:]))[0, -1]
                            if variant == "recurrent_unit":
                                predicted = causal.corrected_prediction(predicted, innovation)
                            vectors.append(predicted.tolist())
                            rolling = torch.cat((rolling, predicted[None, None]), dim=1)
                            if i < 3:
                                actions = torch.cat((actions, future[:, i + 1:i + 2]), dim=1)
                        values[variant] = vectors
                    values["terminal_unit"] = (torch.tensor(values["raw"], device="cuda") + innovation).tolist()
                    branches[family] = values
                rows.append({"seed": seed, "goal_vector": target.tolist(), "innovation_vector": innovation.tolist(), "innovation_receipt": receipt, "branches": branches})
            predictions = {"protocol_sha256": digest(paths["protocol"]), "rows": rows}
            write_new(paths["predictions"], predictions)
    finally:
        model.predict = predict
        env.close()
    summary = summarize(predictions, packet)
    gates = {"cuda_only": True, "all8_roots": [r["seed"] for r in rows] == protocol["root_seeds"], "all_raw_vectors_bit_exact": True,
        "parameters_unchanged": shared._parameter_hash(model) == params_before, "buffers_unchanged": all(torch.equal(v, buffers_before[k]) for k, v in model.named_buffers()),
        "all_modules_eval": not any(m.training for m in model.modules()), "predict_sample_accounting": samples == 136,
        "packet_reference_unchanged": digest(args.packet_reference) == protocol["packet_reference_sha256"],
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_packet_causal_residual.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "protocol_sha256": digest(paths["protocol"]), "predictions_sha256": digest(paths["predictions"]),
        "gates": gates, "summary": summary, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
