"""Frozen RGB-only, goal-image rotation sensitivity on a sealed candidate pool."""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE, OUT as PRIOR, SOURCE
from experiments.pc_pusht_shared_prefix_rescore_20261107_codex import conditional_rollout
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows

PROTOCOL = ROOT.parent / "docs/PC_PUSHT_RGB_GOAL_ROTATION_SCORE_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_pusht_rgb_goal_rotation_score_20261107_codex"


def goal_variants(rgb):
    """Only RGB operations. Synthetic variants are not certified physical states."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (224, 224, 3):
        raise ValueError("expected 224x224 uint8 RGB")
    image = rgb.astype(np.int16)
    r, g, b = image.transpose(2, 0, 1)
    mask = ((g-r >= 8) & (g-r <= 30) & (b-g >= 8) & (b-g <= 30) & (r < 190) & (b < 220)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return [rgb.copy()], {"abstained": True, "reason": "no_component"}
    component = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = map(int, stats[component])
    if not (300 <= area <= 5000 and 10 <= width <= 120 and 10 <= height <= 120
            and x >= 6 and y >= 6 and x + width <= 218 and y + height <= 218):
        return [rgb.copy()], {"abstained": True, "reason": "component_geometry"}
    support = (labels == component).astype(np.uint8)
    support = cv2.dilate(support, np.ones((3, 3), dtype=np.uint8), iterations=1)
    background = cv2.inpaint(rgb, support, 3, cv2.INPAINT_TELEA)
    center = tuple(float(v) for v in centroids[component])
    variants = [rgb.copy()]
    for angle in (-8.0, 8.0):
        affine = cv2.getRotationMatrix2D(center, angle, 1.0)
        patch = cv2.warpAffine(rgb, affine, (224, 224), flags=cv2.INTER_LINEAR)
        rotated = cv2.warpAffine(support, affine, (224, 224), flags=cv2.INTER_NEAREST)
        result = background.copy()
        result[rotated != 0] = patch[rotated != 0]
        variants.append(result)
    return variants, {"abstained": False, "area": area, "bbox": [x, y, width, height],
                      "center": list(center), "angles_degrees": [0, -8, 8]}


def main():
    json_path, array_path = OUT.with_suffix(".json"), OUT.with_suffix(".npz")
    if json_path.exists() or array_path.exists():
        raise FileExistsError(OUT)
    prior = json.loads(PRIOR.with_suffix(".json").read_text())
    source = json.loads(SOURCE.with_suffix(".json").read_text())
    if file_hash(PRIOR.with_suffix(".npz")) != prior["arrays_sha256"] or \
       file_hash(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise ValueError("source arrays changed")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"] or fingerprint != source["model_sha256"]:
        raise ValueError("checkpoint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    results, arrays = [], {}
    with (np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as base,
          np.load(PRIOR.with_suffix(".npz"), allow_pickle=False) as old_scores,
          h5py.File(DATA, "r") as data, torch.inference_mode()):
        panel = panel_rows(data, seed=85, count=4)
        for task in range(4):
            plan = prior["plans"][task]
            index = next(i for i, row in enumerate(source["rows"]) if row["task"] == task and row["selected"])
            rgb = np.asarray(data["pixels"][int(panel[task]) + 25])
            variants, detection = goal_variants(rgb)
            torch.cuda.synchronize()
            start = time.perf_counter()
            goal_emb = model.encode({"pixels": torch.stack([transform(x) for x in variants]).cuda()[:, None]})["emb"][:, 0]
            torch.cuda.synchronize()
            encoding_seconds = time.perf_counter() - start
            original = torch.as_tensor(base["goal"][index], device="cuda")
            original_error = float((goal_emb[0] - original).abs().max())
            if original_error > .001:
                raise ValueError(f"goal mismatch task {task}: {original_error}")
            batches = [torch.load(BASE / "capture" / f"call_{task * 5 + i:03}.pt",
                                  map_location="cpu", weights_only=True) for i in range(5)]
            actions = []
            for candidate in plan["candidates"]:
                action = batches[candidate["call"]]["actions"][0, candidate["slot"]].numpy()
                if hashlib.sha256(action.tobytes()).hexdigest() != candidate["sha256"]:
                    raise ValueError("candidate mismatch")
                actions.append(action)
            actions = torch.as_tensor(np.stack(actions), device="cuda")
            observed = torch.as_tensor(base["actual"][index, 1:2], device="cuda")
            prediction = conditional_rollout(model, observed, actions[:, 2:], 0)[:, -1]
            costs = (prediction[:, None, :] - goal_emb[None]).square().sum(-1)
            original_scores = costs[:, 0].cpu().numpy()
            parity = float(np.max(np.abs(original_scores - old_scores[f"task_{task}"])))
            if parity > .002:
                raise ValueError(f"score parity task {task}: {parity}")
            tolerant = costs.min(dim=1).values.cpu().numpy()
            orig_winner, new_winner = int(np.argmin(original_scores)), int(np.argmin(tolerant))
            arrays[f"original_{task}"] = original_scores
            arrays[f"tolerant_{task}"] = tolerant
            arrays[f"goals_{task}"] = goal_emb.cpu().numpy()
            arrays[f"variants_{task}"] = np.stack(variants)
            results.append({"task": task, "detection": detection, "original_error": original_error,
                            "score_parity_max_abs": parity, "goal_encoder_seconds": encoding_seconds,
                            "goal_encoder_images": len(variants), "candidate_count": len(actions),
                            "original_winner": orig_winner, "tolerant_winner": new_winner,
                            "original_sha256": plan["candidates"][orig_winner]["sha256"],
                            "tolerant_sha256": plan["candidates"][new_winner]["sha256"]})
            print(f"task={task} abstained={detection['abstained']} winners={orig_winner}/{new_winner}", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with array_path.open("xb") as f:
        np.savez_compressed(f, **arrays)
    paths = [PROTOCOL, Path(__file__), PRIOR.with_suffix(".json"), PRIOR.with_suffix(".npz"),
             SOURCE.with_suffix(".json"), SOURCE.with_suffix(".npz")]
    with json_path.open("x") as f:
        json.dump({"rows": results, "track": "primary_rgb", "model_sha256": fingerprint,
                   "scope": "exposed fixed-pool development; no scientific gate claim",
                   "sources": {str(p): file_hash(p) for p in paths}, "arrays_sha256": file_hash(array_path),
                   "data_path": str(DATA), "data_size": DATA.stat().st_size, "data_mtime_ns": DATA.stat().st_mtime_ns},
                  f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
