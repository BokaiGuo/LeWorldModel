"""Goal-only frozen RGB encoder translation subspace on fixed action pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import reacher_fixed_pair_goal_q_20261102 as previous

host, setup, ROOT = previous.host, previous.setup, previous.ROOT
REPORT = ROOT / "results/reacher_fixed_pair_goal_q_RUN_20261102_v1.json"
REPORT_SHA = "f8d77261c34340af5c9ad4ec3e940351a5b54005bade4a59c8b6ff7d282f435d"
AUDIT = ROOT / "results/reacher_fixed_pair_goal_q_SAVED_20261102_v1.json"
AUDIT_SHA = "7cc6139381613e8fa4dcdc499540e7413956c0223c1dc545d6bd60fcc6409d68"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_GOAL_TRANSLATION_SUBSPACE_PROTOCOL_20261102.md"
AUTH = "REACHER_GOAL_TRANSLATION_SUBSPACE_DEVELOPMENT"
SHIFTS = ((-8, 0), (8, 0), (0, -8), (0, 8))


def shift_goal(image, dy, dx):
    if image.ndim != 5 or image.shape[:2] != (1, 1) or image.shape[2] != 3:
        raise ValueError("expected singleton [batch,time,RGB,height,width] goal")
    h, w = image.shape[-2:]
    if abs(dy) >= h or abs(dx) >= w:
        raise ValueError("translation outside image")
    flat = image.reshape(1, 3, h, w)
    padded = F.pad(flat, (abs(dx), abs(dx), abs(dy), abs(dy)), mode="replicate")
    top, left = abs(dy) - dy, abs(dx) - dx
    return padded[:, :, top:top+h, left:left+w].reshape_as(image)


def prepare():
    previous.prepare()
    host.require(setup._sha256_file(REPORT) == REPORT_SHA and setup._sha256_file(AUDIT) == AUDIT_SHA,
                 "saved source hash")
    checked = json.loads(AUDIT.read_text())
    host.require(checked["status"] == "saved_consistency_complete" and checked["report_sha256"] == REPORT_SHA,
                 "goal-Q saved receipt")
    original = json.loads(previous.old.source.PLAN.read_text())
    files = [Path(__file__), PROTOCOL, REPORT, AUDIT, Path(setup.p0.__file__),
             Path(original["weights"]), Path(original["stats"])]
    return {"schema": "reacher_goal_translation_subspace_v1",
            "files": {str(p): setup._sha256_file(p) for p in files},
            "model_hash": original["model_parameter_hash"], "weights": original["weights"],
            "stats": original["stats"], "shifts": [list(x) for x in SHIFTS], "rank": 4,
            "relative_singular_minimum": 1e-6, "pairs": 26, "P1_admission": False}


def run(plan_path, digest, output, authority):
    from experiments.gpu_preflight import require_cuda_allocation

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == digest and json.loads(plan_path.read_text()) == prepare(),
                 "plan drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_goal_translation_subspace_RUN_")
                 and not os.path.lexists(output), "exclusive output")
    old = previous.old
    events = old.replay.pair.audit.saved.verify_journal(old.replay.pair.prior.L2_JOURNAL)
    source_rows = json.loads(REPORT.read_text())["rows"]
    with host.lock.gpu_lock(host.LOCK):
        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        original = json.loads(old.source.PLAN.read_text())
        stats = Path(original["stats"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        model = setup.load_compatible_pretrained(Path(original["weights"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        before = setup._parameter_hash(model)
        host.require(before == original["model_parameter_hash"], "model identity")
        rows, bases = [], []
        for seed in (20263160, 20263161):
            source = [r for r in source_rows if r["seed"] == seed]
            contexts = [e["value"] for e in events if e["kind"] == "context"
                        and e.get("seed") == seed and e.get("arm") == "prefix_00"
                        and e["role"] == "successor"]
            host.require(len(source) == 13 and len(contexts) == 1, "root/goal context")
            value = contexts[0]
            goal = old.replay.context.unpack_rgb_observation(value["goal"])
            frames = [old.replay.context.unpack_rgb_observation(f) for f in value["frames"]]
            _, goal_tensor, _ = old.source.sensitivity.prepare_saved_frames(
                setup.p0._prepare, processor, frames, goal, np.asarray(value["history"], np.float32),
                torch.device("cuda"))
            with torch.no_grad():
                target = setup.p0._planning_coordinate(
                    model, setup.p0._goal_latent(model, goal_tensor), role="goal")
                shifted = torch.stack([setup.p0._planning_coordinate(
                    model, setup.p0._goal_latent(model, shift_goal(goal_tensor, *shift)), role="goal")
                                       for shift in SHIFTS])
                matrix = (shifted - target).double()
                _, singular, vectors = torch.linalg.svd(matrix, full_matrices=False)
            host.require(singular.shape == (4,) and torch.isfinite(singular).all()
                         and float(singular[-1]) >= float(singular[0]) * 1e-6
                         and float(singular[0]) > 0, "translation subspace rank")
            target_np, basis = target.cpu().numpy(), vectors.cpu().numpy()
            bases.append({"seed": seed, "goal_sha256": old.replay.context.rgb_digest(goal),
                          "target": target_np.tolist(), "shifted": shifted.cpu().tolist(),
                          "singular": singular.cpu().tolist(), "basis": basis.tolist()})
            for row in source:
                host.require(np.allclose(target_np, row["target"], rtol=0, atol=1e-6), "goal target binding")
                predicted, actual = np.asarray(row["predicted"]), np.asarray(row["actual"])
                model_score = np.mean(((predicted - target_np) @ basis.T)**2, axis=1)
                real_score = np.mean(((actual - target_np) @ basis.T)**2, axis=1)
                rows.append({"seed": seed, "prefix": row["prefix"], "model_score": model_score.tolist(),
                             "real_score": real_score.tolist(), "physical_final": row["physical_final"],
                             "actions_sha256": row["actions_sha256"], "final_rgb_sha256": row["final_rgb_sha256"]})
        host.require(len(rows) == 26 and setup._parameter_hash(model) == before
                     and json.loads(plan_path.read_text()) == prepare(), "postcheck")
        host.write_new(output, {"schema": "reacher_goal_translation_subspace_result_v1",
                                "plan_sha256": digest, "cuda_allocation": allocation,
                                "model_before": before, "model_after": before, "goal_encodes": 10,
                                "bases": bases, "rows": rows, "P1_admission": False})
        print(json.dumps({"pairs": len(rows), "sha256": setup._sha256_file(output)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "run"))
    for arg in ("plan", "sha256", "output", "authority"):
        p.add_argument("--" + arg)
    args = p.parse_args()
    if args.command == "prepare":
        host.write_new(args.plan, prepare())
        print(json.dumps({"sha256": setup._sha256_file(Path(args.plan))}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
