"""Retrospective fixed-bank RGB scoring, physically isolated from evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOC = ROOT.parent / "docs/pc_pointmaze_rgb_metric_protocol_20261103.md"
PREFIX = RESULTS / "pc_pointmaze_rgb_metric_20261103"
RGB = RESULTS / "a0_pointmaze_failure_real_rgb_metric_20261103_DEV01.json"
ORACLE = RESULTS / "a0_pointmaze_extra_cem_failure_oracle_20261103_DEV01.json"
PLANS = RESULTS / "a1_pointmaze_causal_extra_cem_budget_20261103_DEV01.json"
EXPECTED = {
    RGB: "714caf46a32abc659934870c9edfac8b442e53f95718e5fa8b29663bf2c2afc3",
    ORACLE: "dc78522fb837507e37fcf255c275e885210fb3e2fbb8dec1e2f804700f2ae8ef",
    PLANS: "9c8e00d5a5d1f1401108bbcfcd5dfda12658a9f7d49468189131a7db461bcb8c",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(path):
    if sha(path) != EXPECTED[path]:
        raise ValueError(f"receipt hash mismatch: {path}")
    obj = json.loads(path.read_text())
    if sha(path.with_suffix(".npz")) != obj["arrays_sha256"]:
        raise ValueError(f"array hash mismatch: {path}")
    return obj


def marker(image):
    image = np.asarray(image)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("invalid RGB image")
    r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
    y, x = np.where((g > 100) & (g > 1.25 * r) & (g > 1.15 * b) & (r < 100))
    if not 4 <= len(x) <= 500:
        raise ValueError("missing or excessive green-marker pixels")
    if np.ptp(x) >= 20 or np.ptp(y) >= 20:
        raise ValueError("fragmented or oversized marker")
    center = np.array([x.mean(), y.mean()], dtype=np.float64)
    if min(*center, *(223 - center)) < 14:
        raise ValueError("marker too close to image edge")
    return center


def score_images(images, goal):
    """Only RGB is accepted; errors remain explicit and scores stay missing."""
    scores = np.full(len(images), np.nan)
    positions = np.full((len(images), 2), np.nan)
    failures = []
    try:
        target = marker(goal)
    except ValueError as exc:
        return scores, positions, None, [{"index": "goal", "error": str(exc)}]
    for i, image in enumerate(images):
        try:
            positions[i] = marker(image)
            scores[i] = np.linalg.norm(positions[i] - target)
        except ValueError as exc:
            failures.append({"index": i, "error": str(exc)})
    return scores, positions, target, failures


def ranking(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("invalid paired scores")
    i, j = np.triu_indices(len(a), 1)
    x, y = a[i] - a[j], b[i] - b[j]
    eligible = (x != 0) & (y != 0)
    return {"comparable": int(eligible.sum()), "reversed": int(((x * y < 0) & eligible).sum()),
            "first_ties": int((x == 0).sum()), "second_ties": int((y == 0).sum())}


def key(row):
    return f"r{row['root']}_{row['method']}_{row['phase']}"


def write_json(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def score():
    out, meta = PREFIX.with_name(PREFIX.name + "_scores.npz"), PREFIX.with_name(PREFIX.name + "_scores.json")
    if out.exists() or meta.exists():
        raise FileExistsError("refusing overwrite")
    rgb, plans = receipt(RGB), receipt(PLANS)
    if rgb["plan_sha256"] != EXPECTED[PLANS] or rgb["position_exact_count"] != 624:
        raise ValueError("render provenance mismatch")
    groups, arrays, offset = [], {}, 0
    with np.load(RGB.with_suffix(".npz"), allow_pickle=False) as rendered, \
         np.load(PLANS.with_suffix(".npz"), allow_pickle=False) as inputs:
        images = rendered["rgb10"]
        if images.shape != (624, 224, 224, 3) or images.dtype != np.uint8:
            raise ValueError("unexpected image payload")
        for row in rgb["rows"]:
            name, n = key(row), row["n"]
            scores, positions, target, failures = score_images(images[offset:offset+n], inputs[f"r{row['root']}_goal_rgb"])
            if len(scores) != n:
                raise ValueError("incomplete RGB group")
            arrays[name + "_scores"], arrays[name + "_positions"] = scores, positions
            groups.append({k: row[k] for k in ("root", "method", "phase", "n")} |
                          {"goal_pixel": None if target is None else target.tolist(), "failures": failures})
            offset += n
    if offset != 624 or len(groups) != 12 or len({key(g) for g in groups}) != 12:
        raise ValueError("incomplete groups")
    with out.open("xb") as f:
        np.savez_compressed(f, **arrays)
    write_json(meta, {"schema": "pc_pointmaze_rgb_metric.scores.v1", "groups": groups,
                      "score_arrays_sha256": sha(out), "protocol_sha256": sha(DOC),
                      "script_sha256": sha(Path(__file__)), "rgb_sha256": sha(RGB),
                      "plans_sha256": sha(PLANS)})
    print(f"Saved {offset} RGB-only scores to {out}")


def evaluate():
    score_meta = PREFIX.with_name(PREFIX.name + "_scores.json")
    score_npz = PREFIX.with_name(PREFIX.name + "_scores.npz")
    out = PREFIX.with_name(PREFIX.name + "_evaluation.json")
    if out.exists():
        raise FileExistsError("refusing overwrite")
    scores = json.loads(score_meta.read_text())
    if (sha(score_npz) != scores["score_arrays_sha256"] or sha(DOC) != scores["protocol_sha256"]
            or sha(Path(__file__)) != scores["script_sha256"]):
        raise ValueError("frozen score provenance mismatch")
    rgb, oracle, plans = receipt(RGB), receipt(ORACLE), receipt(PLANS)
    if rgb["oracle_sha256"] != sha(ORACLE) or rgb["plan_sha256"] != sha(PLANS):
        raise ValueError("cross-receipt binding mismatch")
    if [{k: r[k] for k in ("root", "method", "phase", "n")} for r in oracle["rows"]] != [
        {k: r[k] for k in ("root", "method", "phase", "n")} for r in scores["groups"]]:
        raise ValueError("physical group order mismatch")
    rows = []
    with np.load(score_npz, allow_pickle=False) as s, np.load(RGB.with_suffix(".npz"), allow_pickle=False) as r, \
         np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as o:
        for group in scores["groups"]:
            name, n = key(group), group["n"]
            item = dict(group)
            if not group["failures"]:
                geom, physical = s[name + "_scores"], o[name + "_distances"]
                predicted, actual = o[name + "_scores"], r[name + "_real_cost"]
                if any(v.shape != (n,) or not np.isfinite(v).all() for v in (geom, physical, predicted, actual)):
                    raise ValueError(f"invalid matched bank {name}")
                item["rankings"] = {f"{label}_vs_{other}": ranking(a, b) for label, other, a, b in (
                    ("rgb", "physical", geom, physical), ("predicted_latent", "physical", predicted, physical),
                    ("actual_rgb_latent", "physical", actual, physical),
                    ("predicted_latent", "rgb", predicted, geom), ("actual_rgb_latent", "rgb", actual, geom))}
                item["choices"] = {label: {"index": int(np.argmin(v)),
                    "physical_distance_eval_only": float(physical[np.argmin(v)])}
                    for label, v in (("rgb", geom), ("predicted_latent", predicted),
                                     ("actual_rgb_latent", actual), ("physical_oracle", physical))}
            rows.append(item)
    totals = {}
    for item in rows:
        for name, values in item.get("rankings", {}).items():
            total = totals.setdefault(name, {k: 0 for k in values})
            for k, v in values.items():
                total[k] += v
    write_json(out, {"schema": "pc_pointmaze_rgb_metric.evaluation.v1", "rows": rows, "pair_weighted": totals,
                     "score_receipt_sha256": sha(score_meta), "oracle_sha256": sha(ORACLE),
                     "protocol_sha256": sha(DOC), "script_sha256": sha(Path(__file__)),
                     "scope": "retrospective exposed development banks; future RGB oracle; not independent or closed-loop"})
    print(f"Saved evaluation to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("score", "evaluate"))
    args = parser.parse_args()
    print("Fixed-bank RGB-only scoring" if args.stage == "score" else "Evaluate saved scores against fixed labels")
    score() if args.stage == "score" else evaluate()
