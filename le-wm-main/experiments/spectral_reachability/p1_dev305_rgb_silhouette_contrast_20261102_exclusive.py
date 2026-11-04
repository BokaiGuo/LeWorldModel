"""Fixed, post-outcome RGB silhouette diagnostic; never a planning selector."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_rgb_visibility_20261101_main as visibility

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SOURCES = {
    "spec": ("p1_dev305_two_packet_actual_metric_20261102_PREP_DEV01.json",
             "5973ea288519667af36f1b7ba1a4c64cec672f007b14ceeb322d78c1c58eb7c3"),
    "replay": ("p1_dev305_two_packet_real_rgb_20261102_RUN_DEV01/report.json",
               "ca90c9a22202148a7eeabfd45579f8e6f9fdb21ca442162c86614137786f835a"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_json(path, digest):
    raw = path.read_bytes()
    if sha(raw) != digest:
        raise ValueError(f"source hash mismatch: {path}")
    return json.loads(raw)


def frame(ref):
    raw = Path(ref["path"]).read_bytes()
    if sha(raw) != ref["file_sha256"]:
        raise ValueError("frame file hash mismatch")
    with Image.open(ref["path"]) as im:
        pixels = np.asarray(im.convert("RGB"))
    if sha(pixels.tobytes()) != ref["pixel_sha256"]:
        raise ValueError("frame pixel hash mismatch")
    return pixels


def silhouette_iou(goal, observed):
    """An image-space score only; valid operational use requires independent screening."""
    gm = visibility.masks(goal)["object"]
    om = visibility.masks(observed)["object"]
    union = np.count_nonzero(gm | om)
    if not np.any(gm) or not np.any(om) or not union:
        raise ValueError("missing object mask")
    return float(np.count_nonzero(gm & om) / union)


def centroid_distance(goal, observed):
    coords = [np.argwhere(visibility.masks(x)["object"]) for x in (goal, observed)]
    if any(len(x) == 0 for x in coords):
        raise ValueError("missing object mask")
    return float(np.linalg.norm(coords[0].mean(axis=0) - coords[1].mean(axis=0)))


def screen(goal, observed):
    goal_screen, observation_screen = visibility.inspect(goal), visibility.inspect(observed)
    return {"goal": goal_screen, "observation": observation_screen,
            "eligible": goal_screen["eligible"] and observation_screen["eligible"],
            "scope": "renderer-specific visibility only, not occlusion certification"}


def analyze():
    spec, replay = (checked_json(RESULTS / rel, digest) for rel, digest in SOURCES.values())
    if (replay["errors"] or replay["contract_sha256"] !=
            "df32b819ca55955a06877e1d6e39736a4a6266560d84fb0d8630202f12c14d80"):
        raise ValueError("incomplete or mismatched replay")
    if len(spec["frames"]) != 867 or len(replay["rows"]) != 867:
        raise ValueError("unexpected roster")
    goal = frame(spec["goal"])
    goal_screen = visibility.inspect(goal)
    groups = {}
    for index, (item, row) in enumerate(zip(spec["frames"], replay["rows"], strict=True)):
        arm = ("candidate_02", "candidate_00", "zero")[index // 289]
        expected = (arm, (index % 289) // 17, index % 17)
        if ((item["arm"], item["first"], item["second"]) != expected or
                (row["arm"], row["first"], row["second"]) != expected or
                row["status"] != "fixed_horizon_completed" or
                row["final"]["step"] != 10 or
                item["image"]["file_sha256"] != row["final"]["file_sha256"] or
                item["image"]["pixel_sha256"] != row["final"]["pixel_sha256"]):
            raise ValueError("roster, completion, or frame binding mismatch")
        pixels = frame(item["image"])
        visible = visibility.inspect(pixels)["eligible"]
        # Diagnostic ranking is explicitly post-outcome, even when goal screen fails.
        score = silhouette_iou(goal, pixels)
        distance = centroid_distance(goal, pixels)
        groups.setdefault(arm, []).append((score, visible, item["first"], item["second"], distance))
    rows = []
    for arm, entries in groups.items():
        best = max(x[0] for x in entries)
        best_iou = [x for x in entries if x[0] == best]
        nearest = min(entries, key=lambda x: x[4])
        zero = next(x for x in entries if x[2:4] == (16, 16))
        model = next(x for x in entries if x[2:4] == ((12, 11) if arm == "zero" else (12, 10)))
        rows.append({"arm": arm, "n": len(entries),
                     "visible_terminal_count": sum(x[1] for x in entries),
                     "post_outcome_best_iou": best, "iou_best_ties": len(best_iou),
                     "post_outcome_nearest_centroid_pair": list(nearest[2:4]),
                     "nearest_centroid_pixels": nearest[4],
                     "nearest_terminal_visible": nearest[1],
                     "zero_zero_centroid_pixels": zero[4],
                     "model_selected_centroid_pixels": model[4],
                     "scores": [{"first": i, "second": j, "iou": s, "terminal_visible": v,
                                 "centroid_distance_pixels": d}
                                for s, v, i, j, d in entries]})
    return {"schema": "p1_dev305_rgb_silhouette_contrast_exclusive.v1",
            "source_sha256": {key: val[1] for key, val in SOURCES.items()},
            "goal_visibility": goal_screen, "goal_sha256": spec["goal"]["file_sha256"],
            "selector_eligible": False,
            "selection_block_reason": "post-outcome frames; goal visibility not certified",
            "scope": "fixed known-root replay; true RGB silhouettes, not predicted RGB; not deployment or independent confirmation",
            "cost": {"unique_frames": 868, "byte_hash_reads": 868, "image_decode_reads": 868,
                     "neural_calls": 0, "new_physical_steps": 0,
                     "upstream_compute_included": False},
            "p1_admission": False, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
