"""Four-prefix actual-RGB tail replanning with contact-preserving feasibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import p1_dev302_rgb_cp5_cem_20261101_codex as contact
from . import p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar as base

NAME = "p1_dev302_v5_contact_tail_replan_20261101_codex"
ROOT = base.ROOT
PHYSICAL = ROOT / "results/p1_dev302_mixed_prefix_physical_v4_20261101_codex_RUN_MAIN01"
PHYSICAL_SHA = "3c49e8ca83dce02b71bb44a6d41d38eab3490ef8adb9fd32119c062563186759"
ITERATIONS, SAMPLES, ELITES, VARIANTS, HORIZON = 5, 64, 8, 5, 2
SEED, INITIAL_STD, MIN_PROJECTION = 30253021101, 0.25, 0.15
ROLES = base.ROLES
require, sha256, write_new = base.require, base.sha256, base.write_new


def _tail_seed(direction):
    unit = np.asarray(direction, np.float32)
    unit /= np.linalg.norm(unit)
    return np.tile(unit * np.float32(0.25), (10, 1)).reshape(2, 5, 2)


def _projection(bank, direction):
    unit = np.asarray(direction, np.float32)
    unit /= np.linalg.norm(unit)
    return np.einsum("sij,j->s", np.asarray(bank)[:, 0, :3], unit)


def _geometry(pixels):
    _mask, pusher = contact.physical.rgb._mask(
        pixels, contact.physical.rgb.BLUE, 8, single=False
    )
    x0, y0, x1, y1 = pusher["bounds_xyxy"]
    require(
        pusher["count"] >= 20
        and x1 - x0 + 1 <= contact.physical.MOVING_PUSHER_BBOX_MAX_PIXELS
        and y1 - y0 + 1 <= contact.physical.MOVING_PUSHER_BBOX_MAX_PIXELS,
        "compact pusher geometry",
    )
    object_mask, obj = contact.physical.rgb._mask(
        pixels, contact.physical.rgb.BLOCK, 8, single=False
    )
    py, px = np.nonzero(object_mask)
    pusher_xy = np.asarray(pusher["centroid_xy"], np.float64)
    points = np.stack((px, py), axis=1).astype(np.float64)
    deltas = points - pusher_xy
    distances = np.linalg.norm(deltas, axis=1)
    index = int(np.argmin(distances))
    gap = float(distances[index])
    require(np.isfinite(gap) and gap > 0, "contact gap")
    return {
        "pusher_centroid_xy": pusher_xy.tolist(),
        "pusher_components": pusher["components"],
        "object_centroid_xy": obj["centroid_xy"],
        "nearest_object_pixel_xy": points[index].tolist(),
        "gap_pixels": gap,
        "contact_direction_raw_xy": (deltas[index] / gap).tolist(),
    }


def protocol():
    value = base.protocol(PHYSICAL, PHYSICAL_SHA)
    rows = []
    for row in value["roles"]:
        with Image.open(row["cp5_png"]) as handle:
            pixels = np.asarray(handle.convert("RGB"), np.uint8)
        geometry = _geometry(pixels)
        enriched = dict(row)
        enriched["rgb_contact_geometry"] = geometry
        enriched["contact_seed_tail"] = _tail_seed(
            geometry["contact_direction_raw_xy"]
        ).tolist()
        rows.append(enriched)
    value["schema"] = NAME + ".protocol.v1"
    value["roles"] = rows
    value["algorithm"] = {
        **value["algorithm"],
        "initial_mean": "per-arm RGB contact-preserving tail",
        "contact_feasibility_steps": 3,
        "minimum_contact_projection": MIN_PROJECTION,
        "feasibility_role": "reject first-packet motion away from actual cp5 object boundary",
        "final_objective": "robust nonlinear frozen-model terminal latent goal MSE",
    }
    value["information_boundary"] = (
        "actual cp5 RGB and action history; RGB contact direction is feasibility only; "
        "no post-cp5 physical outcome"
    )
    value["P1_admission"] = False
    value["bindings"][str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    value["bindings"][str(Path(contact.__file__).resolve())] = sha256(
        Path(contact.__file__).resolve()
    )
    return value


def run(prepared: Path, prepared_sha: str, output: Path) -> str:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    value = protocol()
    require(
        sha256(prepared) == prepared_sha
        and json.loads(prepared.read_bytes()) == value,
        "protocol",
    )
    output.mkdir()
    device = torch.device("cuda:0")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(value["policy"], device=device).eval().requires_grad_(False)
    before = base.blind.host.signal.model_hash(model)
    require(before["original_state_dict"] == base.MODEL_SHA256, "model")
    task = value["task"]
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean_action, std_action = tensor(task["action_mean"]), tensor(task["action_std"])
    goal = tensor(task["goal"])[None]
    root_seed_tails = np.asarray(value["seed_tails"], np.float32)
    rows, attempts = [], {"rgb_encode": 0, "action_encoder": 0, "predict": 0}
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(
        SDPBackend.MATH
    ):
        for role_row in value["roles"]:
            with Image.open(role_row["cp5_png"]) as handle:
                pixels = np.asarray(handle.convert("RGB"), np.uint8)
            image = base.p0._image_tensor(
                pixels, transform=base.p0.image_transform(), device=device
            )
            attempts["rgb_encode"] += 1
            latent = model.encode({"pixels": image})["emb"][0, 0]
            states = tensor(
                task["root_context"]["latent_history"][-2:] + [latent.cpu().tolist()]
            )[None]
            prefix = tensor(role_row["actual_prefix"])
            prefix_token = ((prefix - mean_action) / std_action).reshape(1, 1, 10)
            history = torch.cat(
                (tensor(task["root_context"]["action_history"][-1:])[None], prefix_token),
                dim=1,
            )
            contact_seed = np.asarray(role_row["contact_seed_tail"], np.float32)
            seeds = np.concatenate((root_seed_tails, contact_seed[None]), axis=0)
            mean, std = contact_seed.copy(), np.full_like(contact_seed, INITIAL_STD)
            rng = np.random.default_rng(SEED)
            records, summaries = [], []
            direction = role_row["rgb_contact_geometry"]["contact_direction_raw_xy"]
            for iteration in range(ITERATIONS):
                bank = base.common.make_bank(rng, mean, std, seeds, iteration)
                projection = _projection(bank, direction)
                feasible = projection >= MIN_PROJECTION
                variants = np.stack(
                    [base.common.action_variants(candidate) for candidate in bank]
                )
                flat = tensor(
                    variants.reshape(SAMPLES * VARIANTS, HORIZON, 5, 2)
                )
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
                    encoded = model.action_encoder(context)
                    attempts["predict"] += 1
                    predicted = model.predict(z, encoded)
                    z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                    past = context[:, -2:]
                scores = (
                    (z[:, -1] - goal)
                    .square()
                    .mean(dim=1)
                    .reshape(SAMPLES, VARIANTS)
                    .cpu()
                    .numpy()
                )
                robust = base.common.robust_scores(scores)
                ranked = robust.copy()
                ranked[~feasible] = np.inf
                require(np.isfinite(ranked).any(), "feasible tail")
                for candidate in range(SAMPLES):
                    records.append(
                        {
                            "iteration": iteration,
                            "candidate": candidate,
                            "contact_projection": float(projection[candidate]),
                            "contact_feasible": bool(feasible[candidate]),
                            "nominal_q": float(scores[candidate, 0]),
                            "variant_q": scores[candidate].tolist(),
                            "robust_score": float(robust[candidate]),
                            "actions": bank[candidate].tolist(),
                        }
                    )
                order = [int(i) for i in np.argsort(ranked, kind="stable") if np.isfinite(ranked[i])]
                while len(order) < ELITES:
                    order.append(order[-1])
                elite = bank[np.asarray(order[:ELITES])]
                mean = elite.mean(axis=0, dtype=np.float32)
                std = np.clip(elite.std(axis=0, dtype=np.float32), base.MIN_STD, 1)
                summaries.append(
                    {
                        "iteration": iteration,
                        "feasible_candidates": int(feasible.sum()),
                        "elite_indices": order[:ELITES],
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                    }
                )
            selected = min(
                (row for row in records if row["contact_feasible"]),
                key=lambda row: (
                    row["robust_score"],
                    row["iteration"],
                    row["candidate"],
                ),
            )
            rows.append(
                {
                    "role": role_row["role"],
                    "candidate_index": role_row["candidate_index"],
                    "candidate_label": role_row["candidate_label"],
                    "actual_prefix": role_row["actual_prefix"],
                    "actual_prefix_sha256": role_row["actual_prefix_sha256"],
                    "cp5_pixel_sha256": role_row["cp5_pixel_sha256"],
                    "rgb_contact_geometry": role_row["rgb_contact_geometry"],
                    "selected": selected,
                    "iterations": summaries,
                    "status": "replanned",
                }
            )
    torch.cuda.synchronize(device)
    after = base.blind.host.signal.model_hash(model)
    require(before == after, "frozen")
    require(all(sha256(Path(p)) == d for p, d in value["bindings"].items()), "closure")
    report = {
        "schema": NAME + ".report.v1",
        "status": "model_only_complete_not_admitted",
        "protocol_sha256": prepared_sha,
        "rows": rows,
        "attempts": attempts,
        "host": host,
        "model_before": before,
        "model_after": after,
        "wall_seconds": time.monotonic() - started,
        "post_cp5_physical_outcomes_used": False,
        "physical_execution": False,
        "development_only": True,
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
    require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive")
    digest = write_new(prepared, protocol())
    print(json.dumps({"prepared": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        with base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            report_sha = run(prepared, digest, output)
        print(json.dumps({"output": str(output), "report_sha256": report_sha}), flush=True)


if __name__ == "__main__":
    main()
