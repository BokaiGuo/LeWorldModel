"""Sealed encoded-cp5, equal-budget V13-kernel planner; no physical actions."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_cp5_dual_encode_20261102_exclusive as encode
from . import p1_dev303_cp5_equal_v13_replan_20261102_exclusive as parent

NAME = "p1_dev303_cp5_encoded_v13_replan_20261102_exclusive"
ROOT = parent.ROOT
ENCODE_PREP = ROOT / "results/p1_dev303_cp5_dual_encode_20261102_exclusive_PREP_DEV01.json"
ENCODE_DIR = ROOT / "results/p1_dev303_cp5_dual_encode_20261102_exclusive_RUN_DEV01"
ENCODE_PREP_SHA = "4798b92775f9685ab7f7cf2f80804c526bb16a6efb16604a00224f3c88a1b5ad"
ENCODE_REPORT_SHA = "c2bf59aac476d93d9232122d6e66ae19d32481844e252222c261394f94273c45"
CONTEXTS_SHA = "4c77d1870474816c733b67d28311b55e076975974894c7a8bbed25918da796e8"
ARMS = parent.ARMS
engine = parent.engine
require = parent.require


def source():
    prep = parent.export.decode(parent.read_fixed(ENCODE_PREP, ENCODE_PREP_SHA))
    report = parent.export.decode(parent.read_fixed(ENCODE_DIR / "report.json", ENCODE_REPORT_SHA))
    raw = parent.read_fixed(ENCODE_DIR / "encoded_contexts.npz", CONTEXTS_SHA)
    require(prep == encode.protocol() and prep["P1_admission"] is False
            and prep["independent_confirmation"] is False and prep["planning"] is False,
            "encoded preparation binding")
    require(report["schema"] == encode.NAME + ".report.v1"
            and report["status"] == "encoded_model_only"
            and report["protocol_sha256"] == ENCODE_PREP_SHA
            and report["encoded_contexts_sha256"] == CONTEXTS_SHA
            and report["rgb_encode_calls"] == 2
            and report["host"]["device"] == "cuda:0"
            and report["model_before"] == report["model_after"]
            and report["model_before"]["original_state_dict"] == prep["model_sha256"]
            and report["planning"] is False and report["physical_execution"] is False
            and report["P1_admission"] is False and report["independent_confirmation"] is False,
            "sealed CUDA encoding report")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        expected = {arm + suffix for arm in ARMS
                    for suffix in ("_latent_history", "_normalized_action_history")}
        require(set(data.files) == expected, "exact encoded contexts roster")
        arrays = {k: data[k].copy() for k in data.files}
    task = prep["task"]
    for role in prep["roles"]:
        arm = role["role"]
        states = arrays[arm + "_latent_history"]
        actions = arrays[arm + "_normalized_action_history"]
        require(states.dtype == actions.dtype == np.float32
                and states.shape == (3, task["latent_dim"]) and actions.shape == (2, 10)
                and np.isfinite(states).all() and np.isfinite(actions).all(), "encoded arrays")
        expected_states, expected_actions = encode.contexts(task, role["actual_prefix"], states[-1])
        require(np.array_equal(states, expected_states) and np.array_equal(actions, expected_actions),
                f"{arm}: encoded task and actual action history")
    return prep, report, arrays


def protocol():
    encoded_prep, encoded_report, _arrays = source()
    original = parent.protocol()
    paths = [Path(__file__), Path(encode.__file__), ENCODE_PREP,
             ENCODE_DIR / "report.json", ENCODE_DIR / "encoded_contexts.npz"]
    bindings = dict(original["bindings"])
    for path in paths:
        bindings[str(path.resolve())] = parent.export.sha(parent.export.read_regular(path))
    require(all(parent.export.sha(parent.export.read_regular(path)) == expected
                for path, expected in bindings.items()), "source binding closure")
    return {
        "schema": NAME + ".protocol.v1",
        "classification": "development_root_not_original_holdout_confirmation",
        "encode_preparation_sha256": ENCODE_PREP_SHA,
        "encode_report_sha256": ENCODE_REPORT_SHA,
        "encoded_contexts_sha256": CONTEXTS_SHA,
        "model_hash": encoded_report["model_before"],
        "policy": encoded_prep["policy"], "task": encoded_prep["task"],
        "roles": original["roles"], "initial_mean": original["initial_mean"],
        "seed_tails": original["seed_tails"], "random_seed": original["random_seed"],
        "algorithm": original["algorithm"],
        "per_arm_budget": {"rgb_encode_calls": 0, "action_encoder_calls": 10,
                           "predict_calls": 10, "predict_samples": 3200,
                           "cem_iterations": 5, "candidates": 320, "variants": 1600},
        "equal_budget_required": True,
        "model_input": "sealed cp5 encoded latent and actual action history",
        "selection_input": "encoded history, frozen goal, actual cp5 RGB feasibility only",
        "simulator_state_used_for_selection": False,
        "physical_outcomes_used_for_selection": False,
        "physical_execution": False, "physical_actions_authorized": False,
        "P1_admission": False, "independent_confirmation": False,
        "bindings": bindings,
    }


def plan_one(direction, score, *, rng_seed=parent.SEED):
    """Pure V13-kernel search; score receives a (64,5,2,5,2) action bank."""
    mean = np.zeros((2, 5, 2), np.float32)
    std = np.full_like(mean, engine.INITIAL_STD)
    seeds = np.stack([mean, mean])
    rng = np.random.default_rng(rng_seed)
    records, iterations = [], []
    initial_bank_sha = None
    for iteration in range(engine.ITERATIONS):
        bank = engine.base.common.make_bank(rng, mean, std, seeds, iteration)
        if iteration == 0:
            initial_bank_sha = parent.export.sha(bank.astype("<f4").tobytes())
        feasible = engine._projection(bank, direction) >= engine.MIN_PROJECTION
        variants = np.stack([engine.base.common.action_variants(candidate) for candidate in bank])
        scores = np.asarray(score(variants), dtype=np.float64)
        require(scores.shape == (engine.SAMPLES, engine.VARIANTS) and np.isfinite(scores).all(),
                "finite V13 variant scores")
        robust = engine.base.common.robust_scores(scores)
        ranked = robust.copy()
        ranked[~feasible] = np.inf
        require(np.isfinite(ranked).any(), "no feasible CEM tail")
        records.extend({"iteration": iteration, "candidate": i,
                        "robust_score": float(robust[i]), "actions": bank[i].tolist()}
                       for i in range(engine.SAMPLES) if feasible[i])
        order = [int(i) for i in np.argsort(ranked, kind="stable") if np.isfinite(ranked[i])]
        while len(order) < engine.ELITES:
            order.append(order[-1])
        elite = bank[np.asarray(order[:engine.ELITES])]
        mean = elite.mean(axis=0, dtype=np.float32)
        std = np.clip(elite.std(axis=0, dtype=np.float32), engine.base.MIN_STD, 1)
        iterations.append({"iteration": iteration, "feasible_candidates": int(feasible.sum()),
                           "elite_indices": order[:engine.ELITES]})
    require(len(iterations) == engine.ITERATIONS and records, "complete CEM")
    chosen = min(records, key=lambda row: (row["robust_score"], row["iteration"], row["candidate"]))
    return {"selected": chosen, "iterations": iterations, "initial_bank_sha256": initial_bank_sha}


def run_cuda(prepared, prepared_sha, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    prepared, output = Path(prepared), Path(output)
    value = protocol()
    require(parent.export.sha(parent.export.read_regular(prepared)) == prepared_sha
            and parent.export.decode(parent.export.read_regular(prepared)) == value,
            "presealed protocol")
    require(not os.path.lexists(output), "exclusive output")
    output.mkdir()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cuda:0")
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(value["policy"], device=device).eval().requires_grad_(False)
    before = engine.base.blind.host.signal.model_hash(model)
    require(before == value["model_hash"], "exact encoder model and buffer hashes")
    _prep, _report, arrays = source()
    task = value["task"]
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean_action, std_action = tensor(task["action_mean"]), tensor(task["action_std"])
    goal = tensor(task["goal"])[None]
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        for role in value["roles"]:
            arm = role["role"]
            states = tensor(arrays[arm + "_latent_history"])[None]
            history = tensor(arrays[arm + "_normalized_action_history"])[None]
            counts = {"action_encoder_calls": 0, "predict_calls": 0, "predict_samples": 0}
            hooks = {"action_encoder": 0, "predict": 0}
            handles = [
                model.action_encoder.register_forward_pre_hook(
                    lambda _m, _a: hooks.__setitem__("action_encoder", hooks["action_encoder"] + 1)),
                model.predictor.register_forward_pre_hook(
                    lambda _m, _a: hooks.__setitem__("predict", hooks["predict"] + 1)),
            ]
            def score(variants):
                flat = tensor(variants.reshape(engine.SAMPLES * engine.VARIANTS, 2, 5, 2))
                batch = engine.SAMPLES * engine.VARIANTS
                z, past = states.repeat(batch, 1, 1), history.repeat(batch, 1, 1)
                for packet in range(engine.HORIZON):
                    context = torch.cat((past, ((flat[:, packet] - mean_action) / std_action)
                                         .reshape(batch, 1, 10)), dim=1)
                    counts["action_encoder_calls"] += 1
                    encoded = model.action_encoder(context)
                    counts["predict_calls"] += 1
                    counts["predict_samples"] += batch
                    predicted = model.predict(z, encoded)
                    require(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()),
                            "finite model prediction")
                    z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                    past = context[:, -2:]
                return ((z[:, -1] - goal).square().mean(dim=1)
                        .reshape(engine.SAMPLES, engine.VARIANTS).cpu().numpy())
            try:
                result = plan_one(role["rgb_contact_geometry"]["contact_direction_raw_xy"], score)
            finally:
                for handle in handles:
                    handle.remove()
            require(counts == {"action_encoder_calls": 10, "predict_calls": 10,
                               "predict_samples": 3200}
                    and hooks == {"action_encoder": 10, "predict": 10}, "per-arm budget/hooks")
            rows.append({"role": arm, "cp5_pixel_sha256": role["cp5_pixel_sha256"],
                         "actual_prefix_sha256": role["actual_prefix_sha256"],
                         "budget": counts, "hooks": hooks, **result})
    torch.cuda.synchronize(device)
    after = engine.base.blind.host.signal.model_hash(model)
    require(before == after and rows[0]["initial_bank_sha256"] == rows[1]["initial_bank_sha256"],
            "frozen model and equal initial random bank")
    require(all(parent.export.sha(parent.export.read_regular(p)) == d
                for p, d in value["bindings"].items()), "postrun source closure")
    report = {
        "schema": NAME + ".report.v1", "status": "model_only_complete_not_admitted",
        "protocol_sha256": prepared_sha, "encode_report_sha256": ENCODE_REPORT_SHA,
        "encoded_contexts_sha256": CONTEXTS_SHA, "host": host,
        "model_before": before, "model_after": after, "rows": rows,
        "physical_execution": False, "physical_actions_authorized": False,
        "P1_admission": False, "independent_confirmation": False,
    }
    return engine.write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--authorize-model-cuda", action="store_true")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "ASCII alphanumeric tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.prepare_sha256 and not args.authorize_model_cuda
                and not os.path.lexists(output), "prepare only")
        digest = engine.write_new(prepared, protocol())
        print(json.dumps({"prepared": str(prepared), "sha256": digest}, sort_keys=True))
    else:
        require(args.prepare_sha256 and args.authorize_model_cuda, "CUDA authorization and SHA")
        with engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            digest = run_cuda(prepared, args.prepare_sha256, output)
        print(json.dumps({"output": str(output), "report_sha256": digest,
                          "physical_actions_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
