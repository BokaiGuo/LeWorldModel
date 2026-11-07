"""Recover goal-marker background using only current and goal RGB observations."""
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_reference_vs_hold_20261107_codex as source

PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_DUAL_RGB_GOAL_RECONSTRUCTION_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.ROOT / "results/pc_tworoom_dual_rgb_goal_reconstruction_20261107_codex.json"
IMAGES = source.ROOT / "results/pc_tworoom_dual_rgb_goal_reconstruction_images_20261107_codex.npz"


def red_alpha(image):
    signed = image.astype(np.int16)
    red = np.maximum(signed[..., 0] - np.maximum(signed[..., 1], signed[..., 2]), 0)
    return red.astype(np.float32) / 255., red > 3


def make_probes(current, goal):
    current, goal = np.asarray(current), np.asarray(goal)
    if (current.shape != (224, 224, 3) or goal.shape != current.shape
            or current.dtype != np.uint8 or goal.dtype != np.uint8):
        raise ValueError("expected matching 224x224 uint8 RGB")
    current_alpha, current_mask = red_alpha(current)
    goal_alpha, goal_mask = red_alpha(goal)
    goal_alpha *= goal_mask
    ys, xs = np.nonzero(goal_mask)
    info = {"valid": False, "reason": None, "goal_red_pixels": int(goal_mask.sum())}
    if not 100 <= goal_mask.sum() <= 5000:
        info["reason"] = "goal_red_support"
        return None, info
    info["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    if xs.min() < 4 or ys.min() < 4 or xs.max() >= 220 or ys.max() >= 220:
        info["reason"] = "touches_shift_boundary"
        return None, info
    free = ~(goal_mask | current_mask)
    if not free.any():
        info["reason"] = "no_background"
        return None, info
    background_diff = np.abs(current.astype(np.int16) - goal.astype(np.int16))[free]
    info["background_mae"] = float(background_diff.mean())
    info["background_max"] = int(background_diff.max())
    if info["background_mae"] > 1 or info["background_max"] > 5:
        info["reason"] = "background_mismatch"
        return None, info
    opaque = goal_mask & (current_alpha > 0.95)
    info["opaque_overlap_pixels"] = int(opaque.sum())
    if opaque.any():
        info["reason"] = "opaque_overlap"
        return None, info
    background = goal.astype(np.float32).copy()
    visible = goal_mask & ~current_mask
    background[visible] = current[visible]
    overlap = goal_mask & current_mask
    info["translucent_overlap_pixels"] = int(overlap.sum())
    if overlap.any():
        a = current_alpha[overlap, None]
        estimated = (current[overlap].astype(np.float32) - a * np.asarray([255., 0., 0.])) / (1 - a)
        background[overlap] = np.clip(estimated, 0, 255)

    def composite(alpha):
        return np.clip(np.rint(background * (1 - alpha[..., None])
                               + np.asarray([255., 0., 0.]) * alpha[..., None]), 0, 255).astype(np.uint8)

    reconstruction = composite(goal_alpha)
    info["reconstruction_mae"] = float(
        np.abs(reconstruction.astype(np.int16) - goal.astype(np.int16))[goal_mask].mean())
    if info["reconstruction_mae"] > 4:
        info["reason"] = "reconstruction"
        return None, info
    probes = []
    for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
        moved = cv2.warpAffine(goal_alpha, np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
                               (224, 224), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        probes.append(composite(moved))
    info["valid"] = True
    return np.stack(probes), info


def run():
    if OUTPUT.exists() or IMAGES.exists():
        raise FileExistsError("outputs exist")
    prior = json.loads(source.PRED.read_text())
    if prior["input_sha256"] != file_hash(source.INPUT):
        raise ValueError("sealed input mismatch")
    rows, images = [], {}
    with np.load(source.INPUT, allow_pickle=False) as data:
        if len(data["rows"]) != 50:
            raise ValueError("panel mismatch")
        for i, row in enumerate(data["rows"]):
            probes, inspection = make_probes(data["current"][i], data["goal"][i])
            rows.append({"row": int(row), "inspection": inspection})
            if probes is not None:
                images[f"root_{i}"] = probes
    with IMAGES.open("xb") as stream:
        np.savez_compressed(stream, **images)
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "input_sha256": file_hash(source.INPUT),
                   "input_prediction_sha256": file_hash(source.PRED), "images_sha256": file_hash(IMAGES),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "exposed RGB-only image reconstruction; no planner or physical scoring"},
                  stream, indent=2, allow_nan=False)
    print(f"{sum(r['inspection']['valid'] for r in rows)}/{len(rows)} reconstructed")


if __name__ == "__main__":
    run()
