"""Host-CUDA cp5 tail replanning for the dev302 RGB contact prefix.

This is a development diagnosis designed after the open-loop dev302 failure.
RGB contact geometry is a fail-closed feasibility filter, not the final task
objective.  Feasible tails remain ranked by robust nonlinear frozen-model
terminal latent goal cost.
"""
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
from . import p1_dev302_rgb_contact_feedback_20261101_codex as physical
from . import p1_unified_cp5_feedback_cem_20261101_codex as common
from . import preflight_pusht_d2p_p0 as p0

NAME = "p1_dev302_rgb_cp5_cem_20261101_codex"
ROOT = physical.ROOT
PHYSICAL = ROOT / "results/p1_dev302_rgb_contact_feedback_20261101_codex_RUN_MAIN02"
PHYSICAL_SHA = "aacb4d296e0a9aacba9bff9e1803bc82d48345beaddc2035b1e72128fd74e0a0"
PHYSICAL_PREP = ROOT / "results/p1_dev302_rgb_contact_feedback_20261101_codex_PREP_MAIN02.json"
PHYSICAL_PREP_SHA = "615e45c2b0a7e9409ca22f50f2729351564e1f401ea6c81ea6ae542af71d3dc1"
PLANNER_PREP = physical.source.PLANNER_DIR.parent / "p1_dev302_blind_planners_20261101_codex_PREP_MAIN01.json"
PLANNER_PREP_SHA = "7dce84ad7b26a306bc5644369916dc2837f224cc1c9c75ce6a615b14b0bbae81"
PLANNER_REPORT = physical.source.PLANNER_DIR / "report.json"
PLANNER_REPORT_SHA = physical.source.PLANNER_REPORT_SHA
TASK = ROOT / "results/task_p1_dev302_goal_blind_legacy_20261101_codex.json.gz"
TASK_SHA = "ac95f202a0149a35eef14aebce9fd78408cbfcf64e54416d146532237221730f"
MODEL_SHA = "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546"
ITERATIONS, SAMPLES, ELITES, VARIANTS = 5, 64, 8, 5
INITIAL_STD, MIN_STD, SEED = 0.25, 0.03, 30220261101
FEASIBILITY_STEPS = 2
MIN_CONTACT_PROJECTION = 0.05


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def _read_task() -> dict:
    require(sha(TASK) == TASK_SHA, "task SHA")
    with gzip.open(TASK, "rt") as stream:
        wrapper = json.load(stream)
    task = wrapper["task"]
    require(
        task["model_sha256"] == "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
        and len(task["goal"]) == 192
        and len(task["action_mean"]) == len(task["action_std"]) == 2,
        "task ABI",
    )
    return task


def _physical_material():
    require(sha(PHYSICAL / "report.json") == PHYSICAL_SHA, "physical report SHA")
    require(sha(PHYSICAL_PREP) == PHYSICAL_PREP_SHA, "physical prep SHA")
    report = json.loads((PHYSICAL / "report.json").read_bytes())
    prepared = json.loads(PHYSICAL_PREP.read_bytes())
    require(
        prepared == physical.build()
        and report["status"] == "recorded_development_diagnosis"
        and not report["errors"],
        "physical source",
    )
    rows = {row["role"]: row for row in report["rows"]}
    row = rows["rgb_contact_then_d2p_tail"]
    require(row["plan_returned"] == 10 and len(row["rgb_decisions"]) == 5, "complete RGB prefix")
    cp5 = next(
        point for point in row["observations"]
        if point["phase"] == "plan" and point["step"] == 5
    )
    path = (PHYSICAL / cp5["file"]).resolve(strict=True)
    require(path.is_relative_to(PHYSICAL) and sha(path) == cp5["file_sha256"], "cp5 frame")
    with Image.open(path) as handle:
        pixels = np.asarray(handle.convert("RGB"), np.uint8)
    require(hashlib.sha256(pixels.tobytes()).hexdigest() == cp5["pixel_sha256"], "cp5 pixels")
    prefix = np.asarray([item["action"] for item in row["transitions"][:5]], np.float32)
    require(prefix.shape == (5, 2) and np.isfinite(prefix).all(), "actual prefix")
    return report, row, cp5, path, pixels, prefix


def _plan_material():
    require(sha(PLANNER_PREP) == PLANNER_PREP_SHA, "planner prep SHA")
    require(sha(PLANNER_REPORT) == PLANNER_REPORT_SHA, "planner report SHA")
    prep = json.loads(PLANNER_PREP.read_bytes())
    report = json.loads(PLANNER_REPORT.read_bytes())
    require(
        report["status"] == "model_plans_sealed_not_admitted"
        and report["model_hash_before"] == report["model_hash_after"] == MODEL_SHA
        and prep["policy"] == "/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm",
        "planner/model identity",
    )
    plans = {}
    for arm in physical.source.base.ARMS[:2]:
        path = physical.source.PLANNER_DIR / f"root_0_{arm}.plan.json"
        plan = json.loads(path.read_bytes())
        require(
            report["artifact_sha256"][path.name] == sha(path)
            and plan["task_sha256"] == TASK_SHA,
            "sealed plan",
        )
        plans[arm] = np.asarray(plan["commands"], np.float32)
    return prep, report, plans


def _geometry(pixels: np.ndarray) -> dict:
    _, pusher = physical.rgb._mask(pixels, physical.rgb.BLUE, 8, single=False)
    x0, y0, x1, y1 = pusher["bounds_xyxy"]
    require(
        pusher["components"] <= physical.MOVING_PUSHER_COMPONENTS_MAX
        and x1 - x0 + 1 <= physical.MOVING_PUSHER_BBOX_MAX_PIXELS
        and y1 - y0 + 1 <= physical.MOVING_PUSHER_BBOX_MAX_PIXELS,
        "pusher geometry",
    )
    object_mask, obj = physical.rgb._mask(pixels, physical.rgb.BLOCK, 8, single=False)
    py, px = np.nonzero(object_mask)
    pusher_xy = np.asarray(pusher["centroid_xy"], np.float64)
    points = np.stack((px, py), axis=1).astype(np.float64)
    deltas = points - pusher_xy
    distances = np.linalg.norm(deltas, axis=1)
    index = int(np.argmin(distances))
    gap = float(distances[index])
    require(math_is_finite_positive(gap), "contact gap")
    direction = deltas[index] / gap
    return {
        "pusher_centroid_xy": pusher_xy.tolist(),
        "object_centroid_xy": obj["centroid_xy"],
        "nearest_object_pixel_xy": points[index].tolist(),
        "gap_pixels": gap,
        "contact_direction_raw_xy": direction.tolist(),
    }


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def contact_seed(direction) -> np.ndarray:
    unit = np.asarray(direction, np.float32)
    require(unit.shape == (2,) and np.isfinite(unit).all(), "contact direction")
    unit /= np.linalg.norm(unit)
    packet = np.zeros((5, 2), np.float32)
    packet[:3] = unit * np.float32(0.25)
    return packet


def contact_projection(bank: np.ndarray, direction) -> np.ndarray:
    values = np.asarray(bank, np.float32)
    require(values.ndim == 3 and values.shape[1:] == (5, 2), "one-packet bank")
    unit = np.asarray(direction, np.float32)
    unit /= np.linalg.norm(unit)
    return np.einsum("sij,j->s", values[:, :FEASIBILITY_STEPS], unit)


def action_variants(packet):
    raw = np.asarray(packet, np.float32)
    require(raw.shape == (5, 2) and np.isfinite(raw).all() and np.max(np.abs(raw)) <= 1, "packet")
    checker = np.tile(np.asarray([1, -1], np.float32), (5, 1))
    deltas = (np.zeros_like(raw), np.ones_like(raw), -np.ones_like(raw), checker, -checker)
    return np.stack([np.clip(raw + np.float32(0.05) * delta, -1, 1) for delta in deltas])


def protocol():
    _report, _row, cp5, path, pixels, prefix = _physical_material()
    prep, planner_report, plans = _plan_material()
    task = _read_task()
    geometry = _geometry(pixels)
    contact = contact_seed(geometry["contact_direction_raw_xy"])
    seeds = [
        plans["full10_conditional_D2P"][5:],
        plans["line_search_GN"][5:],
        np.zeros((5, 2), np.float32),
        contact,
    ]
    bindings = {
        str(path.resolve()): sha(path.resolve())
        for path in (
            Path(__file__), Path(physical.__file__), Path(common.__file__),
            Path(blind.__file__), Path(p0.__file__), Path(locking.__file__),
            PHYSICAL / "report.json", PHYSICAL_PREP, PLANNER_PREP, PLANNER_REPORT, TASK, path,
        )
    }
    require(all(sha(Path(path)) == digest for path, digest in bindings.items()), "source closure")
    return {
        "schema": NAME + ".protocol.v1",
        "cp5_png": str(path),
        "cp5_file_sha256": cp5["file_sha256"],
        "cp5_pixel_sha256": cp5["pixel_sha256"],
        "actual_prefix": prefix.tolist(),
        "actual_prefix_sha256": hashlib.sha256(prefix.astype("<f4").tobytes()).hexdigest(),
        "seed_packets": [seed.tolist() for seed in seeds],
        "seed_labels": ["old_d2p_tail", "old_gn_tail", "zero_tail", "rgb_contact_tail"],
        "rgb_geometry": geometry,
        "task": task,
        "policy": prep["policy"],
        "model_sha256": MODEL_SHA,
        "algorithm": {
            "iterations": ITERATIONS,
            "samples": SAMPLES,
            "elites": ELITES,
            "variants": VARIANTS,
            "initial_std": INITIAL_STD,
            "minimum_std": MIN_STD,
            "contact_feasibility_steps": FEASIBILITY_STEPS,
            "minimum_contact_projection": MIN_CONTACT_PROJECTION,
            "feasibility_role": "reject obvious motion away from current RGB contact boundary",
            "final_objective": "robust nonlinear frozen-model terminal latent goal MSE",
            "selection": "minimum robust score among feasible candidates; stable ties",
        },
        "budget": {
            "rgb_encode_calls": 1,
            "predict_calls": ITERATIONS,
            "action_encoder_calls": ITERATIONS,
            "predict_samples": ITERATIONS * SAMPLES * VARIANTS,
            "candidate_feasibility_evaluations": ITERATIONS * SAMPLES,
            "wall_seconds": 900,
        },
        "planner_source_status": planner_report["status"],
        "post_cp5_physical_outcomes_used": False,
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
    attempts = {"rgb_encode": 0, "predict": 0, "action_encoder": 0, "feasibility": 0}
    device = torch.device("cuda:0")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(value["policy"], device=device).eval().requires_grad_(False)
    before = blind.host.signal.model_hash(model)
    require(before["original_state_dict"] == MODEL_SHA, "model identity")
    hooks = {"predict": 0, "action_encoder": 0}
    handles = [
        model.predictor.register_forward_pre_hook(
            lambda _m, _a: hooks.__setitem__("predict", hooks["predict"] + 1)
        ),
        model.action_encoder.register_forward_pre_hook(
            lambda _m, _a: hooks.__setitem__("action_encoder", hooks["action_encoder"] + 1)
        ),
    ]
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            with Image.open(value["cp5_png"]) as handle:
                pixels = np.asarray(handle.convert("RGB"), np.uint8)
            image = p0._image_tensor(pixels, transform=p0.image_transform(), device=device)
            attempts["rgb_encode"] += 1
            latent = model.encode({"pixels": image})["emb"][0, 0]
            task = value["task"]
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            states = tensor(task["root_context"]["latent_history"][-2:] + [latent.cpu().tolist()])[None]
            mean_action, std_action = tensor(task["action_mean"]), tensor(task["action_std"])
            prefix = tensor(value["actual_prefix"])
            prefix_token = ((prefix - mean_action) / std_action).reshape(1, 1, 10)
            history = torch.cat(
                (tensor(task["root_context"]["action_history"][-1:])[None], prefix_token), dim=1
            )
            goal = tensor(task["goal"])[None]
            rng = np.random.default_rng(SEED)
            seeds = np.asarray(value["seed_packets"], np.float32)
            mean = seeds[-1].copy()
            std = np.full_like(mean, INITIAL_STD)
            records, iterations = [], []
            for iteration in range(ITERATIONS):
                bank = np.clip(rng.normal(mean, std, (SAMPLES, 5, 2)), -1, 1).astype(np.float32)
                bank[0] = mean
                if iteration == 0:
                    bank[: len(seeds)] = seeds
                projection = contact_projection(bank, value["rgb_geometry"]["contact_direction_raw_xy"])
                feasible = projection >= MIN_CONTACT_PROJECTION
                attempts["feasibility"] += SAMPLES
                variants = np.stack([action_variants(candidate) for candidate in bank])
                flat = tensor(variants.reshape(SAMPLES * VARIANTS, 5, 2))
                batch = SAMPLES * VARIANTS
                context = torch.cat(
                    (
                        history.repeat(batch, 1, 1),
                        ((flat - mean_action) / std_action).reshape(batch, 1, 10),
                    ),
                    dim=1,
                )
                attempts["action_encoder"] += 1
                encoded = model.action_encoder(context)
                attempts["predict"] += 1
                predicted = model.predict(states.repeat(batch, 1, 1), encoded)
                scores = (predicted[:, -1] - goal).square().mean(dim=1).reshape(SAMPLES, VARIANTS)
                variant_scores = scores.cpu().numpy()
                robust = common.robust_scores(variant_scores)
                ranked = robust.copy()
                ranked[~feasible] = np.inf
                require(np.isfinite(ranked).any(), "at least one contact-feasible candidate")
                for candidate in range(SAMPLES):
                    records.append({
                        "iteration": iteration,
                        "candidate": candidate,
                        "contact_projection": float(projection[candidate]),
                        "contact_feasible": bool(feasible[candidate]),
                        "nominal_q": float(variant_scores[candidate, 0]),
                        "variant_q": variant_scores[candidate].tolist(),
                        "robust_score": float(robust[candidate]),
                        "rank_score": None if not feasible[candidate] else float(ranked[candidate]),
                        "actions": bank[candidate].tolist(),
                    })
                order = np.argsort(ranked, kind="stable")[:ELITES]
                finite_order = [int(index) for index in order if np.isfinite(ranked[index])]
                require(finite_order, "finite elites")
                while len(finite_order) < ELITES:
                    finite_order.append(finite_order[-1])
                elite = bank[np.asarray(finite_order[:ELITES])]
                mean = elite.mean(axis=0, dtype=np.float32)
                std = np.clip(elite.std(axis=0, dtype=np.float32), MIN_STD, 1).astype(np.float32)
                iterations.append({
                    "iteration": iteration,
                    "feasible_candidates": int(feasible.sum()),
                    "elite_indices": finite_order[:ELITES],
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                })
            selected = min(
                (row for row in records if row["contact_feasible"]),
                key=lambda row: (row["robust_score"], row["iteration"], row["candidate"]),
            )
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    after = blind.host.signal.model_hash(model)
    require(
        attempts == {
            "rgb_encode": 1,
            "predict": ITERATIONS,
            "action_encoder": ITERATIONS,
            "feasibility": ITERATIONS * SAMPLES,
        }
        and hooks == {"predict": ITERATIONS, "action_encoder": ITERATIONS}
        and before == after,
        "resource/frozen model",
    )
    require(all(sha(Path(path)) == digest for path, digest in value["bindings"].items()), "closure")
    report = {
        "schema": NAME + ".report.v1",
        "status": "model_only_complete",
        "protocol_sha256": prepared_sha,
        "selected": selected,
        "iterations": iterations,
        "attempts": attempts,
        "hooks": hooks,
        "host": host,
        "model_before": before,
        "model_after": after,
        "post_cp5_physical_outcomes_used": False,
        "physical_execution": False,
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
            result_sha = run(prepared, digest, output)
        print(json.dumps({"output": str(output), "report_sha256": result_sha}), flush=True)


if __name__ == "__main__":
    main()
