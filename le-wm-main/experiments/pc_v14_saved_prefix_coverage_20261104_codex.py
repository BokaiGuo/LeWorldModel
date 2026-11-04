"""Read-only analysis of previously sealed E0 candidate and selected actions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_e0_v14_carry_nominal_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V14_SAVED_PREFIX_COVERAGE_DIAGNOSTIC_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_v14_saved_prefix_coverage_20261104_codex.json"
ARMS = ("v13_global_response", "v14_carry_nominal")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(source):
    audit_path, result_path = source / "audit.json", source / "result.json"
    audit = json.loads(audit_path.read_text())
    result = json.loads(result_path.read_text())
    if audit["status"] != "passed" or audit["result_sha256"] != digest(result_path):
        raise ValueError("source audit or result hash mismatch")
    if result["manifest_sha256"] != digest(source / "manifest.json"):
        raise ValueError("source manifest hash mismatch")
    by_key = {(r["root"], r["arm"]): r for r in result["rows"]}
    if len(by_key) != len(result["rows"]):
        raise ValueError("duplicate root and arm")
    rows = []
    for root in range(8):
        ref = by_key[root, "global_cem"]
        reference = np.asarray(ref["selected"], np.float32)
        if ref["status"] != "completed" or reference.shape != (15, 2):
            raise ValueError("invalid reference")
        if not np.isfinite(reference).all() or np.max(np.abs(reference)) > 1:
            raise ValueError("invalid reference actions")
        for arm in ARMS:
            row = by_key[root, arm]
            if row["status"] != "completed" or row["task"] != ref["task"]:
                raise ValueError("incomplete or mismatched pair")
            layers = row["result"]["child"]["layers"]
            if len(layers) != 3:
                raise ValueError("invalid depth")
            for depth, layer in enumerate(layers):
                actions = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
                retained = np.asarray(layer["retained"], dtype=int)
                if (layer["depth"] != depth or actions.ndim != 3
                        or actions.shape[1:] != ((depth + 1) * 5, 2)
                        or not len(actions) or not np.isfinite(actions).all()
                        or np.max(np.abs(actions)) > 1 or retained.ndim != 1
                        or not len(retained) or len(np.unique(retained)) != len(retained)
                        or np.any(retained < 0) or np.any(retained >= len(actions))):
                    raise ValueError("invalid candidate layer")
                distances = np.sqrt(np.mean(
                    (actions.astype(np.float64) - reference[None, :actions.shape[1]]) ** 2,
                    axis=(1, 2)))
                nearest = int(np.argmin(distances))
                rows.append({"root": root, "arm": arm, "depth": depth,
                             "prefix_steps": int(actions.shape[1]),
                             "reference_final_distance": float(ref["score"]),
                             "reference_success": bool(ref["score"] < .05),
                             "candidate_count": len(actions), "retained_count": len(retained),
                             "nearest_index": nearest,
                             "nearest_retained": bool(nearest in retained),
                             "generated_min_rms": float(distances[nearest]),
                             "retained_min_rms": float(np.min(distances[retained]))})
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    inputs = [Path(__file__), PROTOCOL, SOURCE / "manifest.json",
              SOURCE / "result.json", SOURCE / "audit.json"]
    hashes = {str(p): digest(p) for p in inputs}
    rows = analyze(SOURCE)
    if hashes != {str(p): digest(p) for p in inputs}:
        raise RuntimeError("source changed during analysis")
    with OUT.open("x") as stream:
        json.dump({"scope": "post-hoc saved action comparison; no new rollout",
                   "sha256": hashes, "rows": rows}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for arm in ARMS:
        for depth in range(3):
            subset = [r for r in rows if r["arm"] == arm and r["depth"] == depth]
            print(arm, "steps", subset[0]["prefix_steps"],
                  "mean_generated_rms", np.mean([r["generated_min_rms"] for r in subset]),
                  "mean_retained_rms", np.mean([r["retained_min_rms"] for r in subset]),
                  "nearest_retained", sum(r["nearest_retained"] for r in subset))
    print("output:", OUT)


if __name__ == "__main__":
    main()
