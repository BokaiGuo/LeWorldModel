"""Actual-cp5 RGB feedback CEM for the fixed unified P1 development branches."""
from __future__ import annotations

import argparse
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
from . import p1_unified_constrained_cem_physical_20261101_codex as physical
from . import preflight_pusht_d2p_p0 as p0

ROOT = blind.ROOT
NAME = "p1_unified_cp5_feedback_cem_20261101_codex"
PHYSICAL = ROOT / "results/p1_unified_constrained_cem_physical_20261101_codex_run_MAIN03"
PHYSICAL_SHA = "a3ca843ab45686d0b23398695abc5da4af02863d3d862738f204ba295d6da4f6"
PHYSICAL_PREP = ROOT / "results/p1_unified_constrained_cem_physical_20261101_codex_PREP_MAIN03.json"
PHYSICAL_PREP_SHA = "6936879e4456a63b0fa2c5d89740ab126f023668bd85edcd70dccc6829b97faa"
ROLES = physical.ROLES
ITERATIONS = 5
SAMPLES = 64
ELITES = 8
VARIANTS = 5
HORIZON = 2
INITIAL_STD = 0.25
MIN_STD = 0.03
SEED = 20261101


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def action_variants(actions):
    raw = np.asarray(actions, dtype=np.float32)
    require(raw.shape == (HORIZON, 5, 2) and np.isfinite(raw).all()
            and np.max(np.abs(raw)) <= 1, "bounded two-packet actions")
    checker = np.tile(np.asarray([1, -1], np.float32), (HORIZON, 5, 1))
    deltas = (
        np.zeros_like(raw),
        np.ones_like(raw),
        -np.ones_like(raw),
        checker,
        -checker,
    )
    return np.stack([np.clip(raw + np.float32(0.05) * delta, -1, 1)
                     for delta in deltas]).astype(np.float32)


def robust_scores(scores):
    values = np.asarray(scores, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == VARIANTS
            and np.isfinite(values).all(), "finite variant scores")
    means = values.mean(axis=1)
    ranges = values.max(axis=1) - values.min(axis=1)
    return means + ranges


def make_bank(rng, mean, std, seeds, iteration):
    require(mean.shape == std.shape == (HORIZON, 5, 2), "CEM state")
    bank = np.clip(rng.normal(mean, std, (SAMPLES, HORIZON, 5, 2)), -1, 1).astype(np.float32)
    bank[0] = mean
    if iteration == 0:
        require(1 <= len(seeds) < SAMPLES, "seed tails")
        bank[:len(seeds)] = np.asarray(seeds, np.float32)
    return bank


def update(bank, scores):
    values = np.asarray(scores, dtype=np.float64)
    require(bank.shape == (SAMPLES, HORIZON, 5, 2)
            and values.shape == (SAMPLES,) and np.isfinite(values).all(), "CEM update")
    order = np.argsort(values, kind="stable")[:ELITES]
    elites = bank[order]
    return (elites.mean(axis=0, dtype=np.float32),
            np.clip(elites.std(axis=0, dtype=np.float32), MIN_STD, 1).astype(np.float32),
            order.tolist())


def _read_pixels(receipt):
    path = Path(receipt["path"])
    require(path.is_file() and not path.is_symlink() and sha(path) == receipt["file_sha256"],
            "cp5 PNG file")
    with Image.open(path) as image:
        pixels = np.asarray(image).copy()
    require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
            and hashlib.sha256(pixels.tobytes()).hexdigest() == receipt["pixel_sha256"],
            "cp5 pixel identity")
    return path


def protocol():
    require(sha(PHYSICAL / "report.json") == PHYSICAL_SHA, "physical report SHA")
    require(sha(PHYSICAL_PREP) == PHYSICAL_PREP_SHA, "physical prepare SHA")
    report = json.loads((PHYSICAL / "report.json").read_bytes())
    prepared = json.loads(PHYSICAL_PREP.read_bytes())
    require(report["status"] == "recorded_development" and not report["errors"]
            and report["prepare_sha256"] == PHYSICAL_PREP_SHA, "complete physical source")
    plans = {root["slot"]: root["candidates"] for root in prepared["roots"]}
    rows = []
    for row in report["rows"]:
        slot, role = row["source_slot"], row["role"]
        if row["status"] == "native_stopped" and row["control_returned"] < 5:
            require(slot == 1 and role == "same_bank_greedy"
                    and row["final_metrics"]["native_success"] is True, "only early-success skip")
            rows.append({
                "slot": slot, "role": role, "status": "early_success_no_replan",
                "stop_step": row["control_returned"],
            })
            continue
        require(row["control_returned"] >= 5 and len(row["observations"]) >= 6, "actual cp5")
        cp5 = row["observations"][5]
        path = _read_pixels(cp5)
        prefix = np.asarray(row["executed_raw"][:5], np.float32)
        planned = np.asarray(plans[slot][role], np.float32)
        require(prefix.shape == (5, 2) and planned.shape == (3, 5, 2)
                and np.array_equal(prefix, planned[0]), "actual prefix equals fixed plan")
        role_tails = [np.asarray(plans[slot][name], np.float32)[1:] for name in ROLES]
        seeds = [planned[1:], np.zeros((2, 5, 2), np.float32), *role_tails]
        unique = []
        for seed in seeds:
            if not any(np.array_equal(seed, old) for old in unique):
                unique.append(seed)
        task, _, source_paths, policy = blind.load(slot)
        model_source_closure = json.loads(
            (ROOT / "results" / blind.ITEMS[slot][0]).read_bytes()
        )["sources"]
        rows.append({
            "slot": slot,
            "role": role,
            "status": "replan",
            "prefix": prefix.tolist(),
            "old_tail": planned[1:].tolist(),
            "seed_tails": [seed.tolist() for seed in unique],
            "cp5_png": str(path),
            "cp5_file_sha256": cp5["file_sha256"],
            "cp5_pixel_sha256": cp5["pixel_sha256"],
            "task": task,
            "task_sha256": blind.host.digest(task),
            "model_policy": policy,
            "source_paths": source_paths,
            "model_source_closure": model_source_closure,
        })
    require(len(rows) == 6 and sum(row["status"] == "replan" for row in rows) == 5, "fixed six roster")
    dependencies = (Path(__file__).resolve(), Path(blind.__file__).resolve(),
                    Path(physical.__file__).resolve(), Path(p0.__file__).resolve(),
                    Path(locking.__file__).resolve(), PHYSICAL / "report.json", PHYSICAL_PREP)
    bindings = {str(path): sha(path) for path in dependencies}
    for row in rows:
        if row["status"] != "replan":
            continue
        bindings[row["cp5_png"]] = row["cp5_file_sha256"]
        for path, digest in row["source_paths"].items():
            require(path not in bindings or bindings[path] == digest, "source collision")
            bindings[path] = digest
        for path, digest in row["model_source_closure"].items():
            require(path not in bindings or bindings[path] == digest, "model source collision")
            bindings[path] = digest
    require(all(sha(Path(path)) == digest for path, digest in bindings.items()), "source closure")
    return {
        "schema": NAME + ".protocol.v1",
        "rows": rows,
        "algorithm": {
            "iterations": ITERATIONS, "samples": SAMPLES, "elites": ELITES,
            "horizon_packets": HORIZON, "seed_base": SEED,
            "initial_std": INITIAL_STD, "minimum_std": MIN_STD,
            "iteration0": "old tail, zero tail, and all same-root role tails before Gaussian samples",
            "variants": "nominal, +/-0.05 both axes, +/-0.05 checkerboard; clipped",
            "score": "mean five terminal latent goal MSEs plus max-min range",
            "selection": "global minimum robust score; stable iteration/candidate tie order",
        },
        "budget": {
            "rgb_encode_calls": 5,
            "predict_calls": 5 * ITERATIONS * HORIZON,
            "action_encoder_calls": 5 * ITERATIONS * HORIZON,
            "predict_samples": 5 * ITERATIONS * HORIZON * SAMPLES * VARIANTS,
            "wall_seconds": 1800,
        },
        "information_boundary": "only actual cp5 RGB, actual first-five actions, root task, and preexisting tails",
        "post_cp5_physical_outcomes_used": False,
        "development_only": True,
        "physical_execution": False,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def write_new(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        require(stream.write(payload) == len(payload), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def run(prepared, expected_sha, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    value = protocol()
    require(sha(prepared) == expected_sha and json.loads(prepared.read_bytes()) == value,
            "protocol drift")
    output.mkdir()
    started = time.monotonic()
    attempts = {"rgb_encode": 0, "predict": 0, "action_encoder": 0}
    hooks = {"predict": 0, "action_encoder": 0}
    device = torch.device("cuda:0")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    replans = [row for row in value["rows"] if row["status"] == "replan"]
    policies = {row["model_policy"] for row in replans}
    require(len(policies) == 1, "one frozen model policy")
    model = load_compatible_pretrained(next(iter(policies)), device=device).eval().requires_grad_(False)
    before = blind.host.signal.model_hash(model)
    require(before["original_state_dict"] == physical.MODEL_SHA, "model identity")
    for module in model.modules():
        filename = str(Path(inspect.getfile(type(module))).resolve())
        require(all(filename in row["model_source_closure"]
                    and sha(filename) == row["model_source_closure"][filename]
                    for row in replans), "model module source")
    handles = [
        model.predictor.register_forward_pre_hook(lambda _m, _a: hooks.__setitem__("predict", hooks["predict"] + 1)),
        model.action_encoder.register_forward_pre_hook(
            lambda _m, _a: hooks.__setitem__("action_encoder", hooks["action_encoder"] + 1)),
    ]
    results = []
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for row_index, row in enumerate(value["rows"]):
                if row["status"] != "replan":
                    results.append(dict(row))
                    continue
                require(time.monotonic() - started < value["budget"]["wall_seconds"], "wall cap")
                with Image.open(row["cp5_png"]) as image:
                    pixels = np.asarray(image).copy()
                require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["cp5_pixel_sha256"],
                        "cp5 pixels")
                image = p0._image_tensor(pixels, transform=p0.image_transform(), device=device)
                attempts["rgb_encode"] += 1
                latent = model.encode({"pixels": image})["emb"][0, 0]
                require(latent.shape == (192,) and bool(torch.isfinite(latent).all()), "cp5 latent")
                task = row["task"]
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                states = tensor(task["root_context"]["latent_history"][-2:] + [latent.cpu().tolist()])[None]
                old_history = tensor(task["root_context"]["action_history"][-1:])[None]
                mean_action, std_action = tensor(task["action_mean"]), tensor(task["action_std"])
                prefix = tensor(row["prefix"])
                prefix_token = ((prefix - mean_action) / std_action).reshape(1, 1, 10)
                history = torch.cat((old_history, prefix_token), dim=1)
                goal = tensor(task["goal"])[None]
                rng = np.random.default_rng(SEED + 17 * row["slot"] + row_index)
                mean = np.asarray(row["old_tail"], np.float32)
                std = np.full_like(mean, INITIAL_STD)
                all_rows = []
                iterations = []
                for iteration in range(ITERATIONS):
                    bank = make_bank(rng, mean, std, row["seed_tails"], iteration)
                    variant_bank = np.stack([action_variants(candidate) for candidate in bank])
                    flat = tensor(variant_bank.reshape(SAMPLES * VARIANTS, HORIZON, 5, 2))
                    batch = SAMPLES * VARIANTS
                    z = states.repeat(batch, 1, 1)
                    past = history.repeat(batch, 1, 1)
                    for packet in range(HORIZON):
                        context = torch.cat(
                            (past, ((flat[:, packet] - mean_action) / std_action).reshape(batch, 1, 10)),
                            dim=1,
                        )
                        attempts["action_encoder"] += 1
                        encoded = model.action_encoder(context)
                        attempts["predict"] += 1
                        predicted = model.predict(z, encoded)
                        require(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()),
                                "model output")
                        z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                        past = context[:, -2:]
                    scores = (z[:, -1] - goal).square().mean(dim=1).reshape(SAMPLES, VARIANTS)
                    variant_scores = scores.cpu().numpy()
                    robust = robust_scores(variant_scores)
                    for candidate in range(SAMPLES):
                        all_rows.append({
                            "iteration": iteration, "candidate": candidate,
                            "nominal_q": float(variant_scores[candidate, 0]),
                            "variant_q": variant_scores[candidate].tolist(),
                            "robust_score": float(robust[candidate]),
                            "actions": bank[candidate].tolist(),
                        })
                    mean, std, elite_indices = update(bank, robust)
                    iterations.append({
                        "iteration": iteration, "elite_indices": elite_indices,
                        "mean": mean.tolist(), "std": std.tolist(),
                    })
                selected = min(all_rows, key=lambda item: (
                    item["robust_score"], item["iteration"], item["candidate"]))
                results.append({
                    "slot": row["slot"], "role": row["role"], "status": "replanned",
                    "cp5_pixel_sha256": row["cp5_pixel_sha256"],
                    "prefix": row["prefix"], "old_tail": row["old_tail"],
                    "selected": selected, "iterations": iterations,
                })
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    after = blind.host.signal.model_hash(model)
    expected_calls = {
        "rgb_encode": value["budget"]["rgb_encode_calls"],
        "predict": value["budget"]["predict_calls"],
        "action_encoder": value["budget"]["action_encoder_calls"],
    }
    require(attempts == expected_calls
            and hooks == {"predict": attempts["predict"], "action_encoder": attempts["action_encoder"]}
            and before == after, "complete calls and frozen model")
    require(all(sha(Path(path)) == digest for path, digest in value["bindings"].items()),
            "post-run source drift")
    report = {
        "schema": NAME + ".report.v1", "status": "model_only_complete",
        "protocol_sha256": expected_sha, "rows": results, "attempts": attempts,
        "module_hooks": hooks, "predict_samples": value["budget"]["predict_samples"],
        "host": host, "model_before": before, "model_after": after,
        "post_cp5_physical_outcomes_used": False, "physical_execution": False,
        "development_only": True, "independent_confirmation": False, "P1_admission": False,
    }
    return write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
            result_sha = run(prepared, digest, output)
        print(json.dumps({"output": str(output), "report_sha256": result_sha}), flush=True)


if __name__ == "__main__":
    main()
