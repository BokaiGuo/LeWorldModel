"""Host-CUDA full-program selection over the fixed dev302 mixed prefix bank."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import codex_p1_blind_uphill_bank_20261101_exclusive as blind
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_dev302_mixed_prefix_bank_20261101_codex as bank_source
from . import p1_dev302_rgb_cp5_cem_20261101_codex as cp5
from . import p1_unified_cp5_feedback_cem_20261101_codex as common
from . import preflight_pusht_d2p_p0 as p0

NAME = "p1_dev302_mixed_bank_cuda_20261101_codex"
ROOT = cp5.ROOT
ITERATIONS, SAMPLES, ELITES, VARIANTS, HORIZON = 5, 64, 8, 5, 2
INITIAL_STD, MIN_STD, SEED = 0.25, 0.03, 30220261102


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def _task():
    require(sha(cp5.TASK) == cp5.TASK_SHA, "task SHA")
    with gzip.open(cp5.TASK, "rt") as stream:
        return json.load(stream)["task"]


def _images():
    directory = cp5.physical.source.source_directory()
    current_path = directory / "rgb_10.png"
    goal_path = directory / "goal.png"
    current = cp5.physical.source.base.image(current_path)
    goal = cp5.physical.source.base.image(goal_path)
    return current_path, goal_path, current, goal


def _plans():
    _prep, _report, plans = cp5._plan_material()
    return plans


def protocol():
    current_path, goal_path, current, goal = _images()
    plans = _plans()
    bank = bank_source.build(
        current, goal,
        plans["full10_conditional_D2P"][:5],
        plans["line_search_GN"][:5],
    )
    task = _task()
    prep = json.loads(cp5.PLANNER_PREP.read_bytes())
    seeds = [
        plans["full10_conditional_D2P"][5:][None],
        plans["line_search_GN"][5:][None],
        np.zeros((1, 5, 2), np.float32),
    ]
    # Existing plans supply one packet; pad to the common two-packet tail ABI.
    seed_tails = [
        np.concatenate((seed, np.zeros((1, 5, 2), np.float32)), axis=0)
        for seed in seeds
    ]
    bindings = {
        str(path.resolve()): sha(path.resolve())
        for path in (
            Path(__file__), Path(bank_source.__file__), Path(cp5.__file__),
            Path(common.__file__), Path(blind.__file__), Path(p0.__file__),
            Path(locking.__file__), current_path, goal_path, cp5.TASK,
            cp5.PLANNER_PREP, cp5.PLANNER_REPORT,
        )
    }
    return {
        "schema": NAME + ".protocol.v1",
        "prefix_bank": bank,
        "tail_seeds": [seed.tolist() for seed in seed_tails],
        "tail_seed_labels": ["old_d2p_plus_zero", "old_gn_plus_zero", "zero"],
        "task": task,
        "policy": prep["policy"],
        "model_sha256": cp5.MODEL_SHA,
        "algorithm": {
            "prefix_candidates": len(bank["candidates"]),
            "tail_iterations_per_prefix": ITERATIONS,
            "tail_samples": SAMPLES,
            "tail_elites": ELITES,
            "tail_packets": HORIZON,
            "robust_variants": VARIANTS,
            "contact_filter": "prefix contact_feasible_prior must be true for full-program selected",
            "same_bank_greedy": "minimum predicted cp5 latent goal MSE over all fixed prefixes",
            "selected": "minimum robust nonlinear terminal latent goal MSE after equal tail CEM",
        },
        "budget": {
            "action_encoder_calls": len(bank["candidates"]) * (1 + ITERATIONS * HORIZON),
            "predict_calls": len(bank["candidates"]) * (1 + ITERATIONS * HORIZON),
            "predict_samples": len(bank["candidates"]) * (
                1 + ITERATIONS * HORIZON * SAMPLES * VARIANTS
            ),
            "candidate_evaluations": len(bank["candidates"]),
            "wall_seconds": 1800,
        },
        "physical_outcomes_used": False,
        "development_only": True,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def write_new(path: Path, value) -> str:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        require(stream.write(payload) == len(payload), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def run(prepared: Path, prepared_sha: str, output: Path) -> str:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    value = protocol()
    require(sha(prepared) == prepared_sha and json.loads(prepared.read_bytes()) == value, "protocol")
    output.mkdir()
    started = time.monotonic()
    device = torch.device("cuda:0")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(value["policy"], device=device).eval().requires_grad_(False)
    before = blind.host.signal.model_hash(model)
    require(before["original_state_dict"] == cp5.MODEL_SHA, "model identity")
    attempts = {"action_encoder": 0, "predict": 0}
    hooks = {"action_encoder": 0, "predict": 0}
    handles = [
        model.action_encoder.register_forward_pre_hook(
            lambda _m, _a: hooks.__setitem__("action_encoder", hooks["action_encoder"] + 1)
        ),
        model.predictor.register_forward_pre_hook(
            lambda _m, _a: hooks.__setitem__("predict", hooks["predict"] + 1)
        ),
    ]
    rows = []
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            task = value["task"]
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            root_states = tensor(task["root_context"]["latent_history"])[None]
            root_history = tensor(task["root_context"]["action_history"])[None]
            mean_action, std_action = tensor(task["action_mean"]), tensor(task["action_std"])
            goal = tensor(task["goal"])[None]
            seed_tails = np.asarray(value["tail_seeds"], np.float32)
            for row in value["prefix_bank"]["candidates"]:
                require(time.monotonic() - started < value["budget"]["wall_seconds"], "wall cap")
                prefix = np.asarray(row["actions"], np.float32)
                prefix_tensor = tensor(prefix)
                prefix_token = ((prefix_tensor - mean_action) / std_action).reshape(1, 1, 10)
                prefix_context = torch.cat((root_history, prefix_token), dim=1)
                attempts["action_encoder"] += 1
                encoded = model.action_encoder(prefix_context)
                attempts["predict"] += 1
                predicted = model.predict(root_states, encoded)
                cp5_latent = predicted[:, -1]
                immediate_q = float((cp5_latent - goal).square().mean().cpu())
                states = torch.cat((root_states[:, -2:], cp5_latent[:, None]), dim=1)
                history = torch.cat((root_history[:, -1:], prefix_token), dim=1)
                rng = np.random.default_rng(SEED + row["index"] * 101)
                mean = seed_tails[0].copy()
                std = np.full_like(mean, INITIAL_STD)
                records, iterations = [], []
                for iteration in range(ITERATIONS):
                    bank = np.clip(
                        rng.normal(mean, std, (SAMPLES, HORIZON, 5, 2)), -1, 1
                    ).astype(np.float32)
                    bank[0] = mean
                    if iteration == 0:
                        bank[: len(seed_tails)] = seed_tails
                    variants = np.stack(
                        [common.action_variants(candidate) for candidate in bank]
                    )
                    require(
                        variants.shape == (SAMPLES, VARIANTS, HORIZON, 5, 2),
                        "variant bank",
                    )
                    flat = tensor(variants.reshape(SAMPLES * VARIANTS, HORIZON, 5, 2))
                    batch = SAMPLES * VARIANTS
                    z = states.repeat(batch, 1, 1)
                    past = history.repeat(batch, 1, 1)
                    for packet in range(HORIZON):
                        context = torch.cat(
                            (
                                past,
                                ((flat[:, packet] - mean_action) / std_action).reshape(
                                    batch, 1, 10
                                ),
                            ),
                            dim=1,
                        )
                        attempts["action_encoder"] += 1
                        action_encoded = model.action_encoder(context)
                        attempts["predict"] += 1
                        next_state = model.predict(z, action_encoded)
                        z = torch.cat((z, next_state[:, -1:]), dim=1)[:, -3:]
                        past = context[:, -2:]
                    scores = (z[:, -1] - goal).square().mean(dim=1).reshape(SAMPLES, VARIANTS)
                    variant_scores = scores.cpu().numpy()
                    robust = common.robust_scores(variant_scores)
                    for candidate in range(SAMPLES):
                        records.append({
                            "iteration": iteration,
                            "candidate": candidate,
                            "nominal_q": float(variant_scores[candidate, 0]),
                            "variant_q": variant_scores[candidate].tolist(),
                            "robust_score": float(robust[candidate]),
                            "actions": bank[candidate].tolist(),
                        })
                    mean, std, elite = common.update(bank, robust)
                    iterations.append({
                        "iteration": iteration,
                        "elite_indices": elite,
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                    })
                selected_tail = min(
                    records, key=lambda item: (
                        item["robust_score"], item["iteration"], item["candidate"]
                    )
                )
                rows.append({
                    **{key: row[key] for key in (
                        "index", "label", "actions", "actions_sha256",
                        "nearest_contact_projection_first2", "contact_feasible_prior",
                        "constructed_from_rgb_contact_geometry",
                    )},
                    "predicted_cp5_q": immediate_q,
                    "tail_selected": selected_tail,
                    "iterations": iterations,
                    "full_program_actions": (
                        prefix.tolist() + selected_tail["actions"][0] + selected_tail["actions"][1]
                    ),
                })
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    selected = min(
        (row for row in rows if row["contact_feasible_prior"]),
        key=lambda row: (
            row["tail_selected"]["robust_score"], row["index"]
        ),
    )
    greedy = min(rows, key=lambda row: (row["predicted_cp5_q"], row["index"]))
    random = next(row for row in rows if row["label"] == "random_control")
    zero = next(row for row in rows if row["label"] == "zero")
    after = blind.host.signal.model_hash(model)
    expected = {
        "action_encoder": value["budget"]["action_encoder_calls"],
        "predict": value["budget"]["predict_calls"],
    }
    require(attempts == expected and hooks == expected and before == after, "resources/frozen")
    require(all(sha(Path(path)) == digest for path, digest in value["bindings"].items()), "closure")
    report = {
        "schema": NAME + ".report.v1",
        "status": "model_only_complete",
        "protocol_sha256": prepared_sha,
        "rows": rows,
        "selected_index": selected["index"],
        "same_bank_greedy_index": greedy["index"],
        "zero_index": zero["index"],
        "random_index": random["index"],
        "selected_distinct_from_greedy": selected["index"] != greedy["index"],
        "attempts": attempts,
        "hooks": hooks,
        "host": host,
        "model_before": before,
        "model_after": after,
        "physical_outcomes_used": False,
        "development_only": True,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    return write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive outputs")
    digest = write_new(prepared, protocol())
    print(json.dumps({"protocol": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            report_sha = run(prepared, digest, output)
        print(json.dumps({"output": str(output), "report_sha256": report_sha}), flush=True)


if __name__ == "__main__":
    main()
