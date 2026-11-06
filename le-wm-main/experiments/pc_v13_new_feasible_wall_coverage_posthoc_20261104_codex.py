"""Analyze scored actions from the sealed new-Wall panel without new search."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import pc_v13_new_feasible_wall_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V13_NEW_FEASIBLE_WALL_COVERAGE_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_v13_new_feasible_wall_coverage_posthoc_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plans_path, audit_path = source.OUT / "plans.json", source.OUT / "plans_audit.json"
    plans, audit = json.loads(plans_path.read_text()), json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(plans_path):
        raise ValueError("source audit mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, plans_path, audit_path,
        Path(source.prior.previous.independent.__code__.co_filename),
        *(source.OUT / row["trace"] for row in plans["rows"])])
    by_key = {(r["root"], r["arm"]): r for r in plans["rows"]}
    if len(by_key) != 8 or len(plans["rows"]) != 8:
        raise ValueError("duplicate or missing rows")
    rows, comparisons = [], []
    for root in range(4):
        reference = by_key[root, "global_cem"]
        for arm in source.ARMS:
            row = by_key[root, arm]
            if (row["status"] != "completed" or row["task"] != reference["task"]
                    or row["task"] != source.tasks()[root]
                    or seal.digest(source.OUT / row["trace"]) != row["trace_sha256"]):
                raise ValueError("plan identity or trace mismatch")
            counts = {"samples": 0, "blocked": 0, "ever_right": 0, "success": 0}
            best_cross = None
            with np.load(source.OUT / row["trace"], allow_pickle=False) as data:
                for i in range(row["batches"]):
                    actions = data[f"actions_{i}"]
                    if actions.shape[1] != 15:
                        continue
                    path, blocked = source.prior.previous.independent(row["task"]["start"], actions)
                    score = np.linalg.norm(path[:, -1, :2] - row["task"]["goal"], axis=1)
                    np.testing.assert_allclose(score, data[f"scores_{i}"], atol=1e-12, rtol=0)
                    cross = np.any(path[:, 1:, 0] > 0, axis=1)
                    counts["samples"] += len(actions)
                    counts["blocked"] += int(np.any(blocked, axis=1).sum())
                    counts["ever_right"] += int(cross.sum())
                    counts["success"] += int((score < .05).sum())
                    if np.any(cross):
                        value = float(score[cross].min())
                        best_cross = value if best_cross is None else min(best_cross, value)
            rows.append({"root": root, "arm": arm, **counts, "best_cross_distance": best_cross})
        if reference["score"] < .05:
            selected = np.asarray(reference["selected"], np.float32)
            response = by_key[root, "v13"]["details"]["child"]["layers"]
            for depth, layer in enumerate(response):
                candidates = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
                retained = np.asarray(layer["retained"], int)
                if candidates.shape[1:] != (5*(depth+1), 2) or len(np.unique(retained)) != len(retained):
                    raise ValueError("invalid recorded layer")
                if np.any(retained < 0) or np.any(retained >= len(candidates)):
                    raise ValueError("invalid retained index")
                distance = np.sqrt(np.mean((candidates.astype(np.float64) -
                                            selected[None, :candidates.shape[1]])**2, axis=(1, 2)))
                idx = int(np.argmin(distance))
                comparisons.append({"root": root, "steps": candidates.shape[1],
                    "candidate_count": len(candidates), "retained_count": len(retained),
                    "closest_rms": float(distance[idx]), "retained_rms": float(distance[retained].min()),
                    "nearest_retained": bool(idx in retained)})
    seal.verify(deps)
    source.prior.writer(OUT, {"dependencies": deps, "rows": rows, "comparisons": comparisons,
                              "scope": "post-hoc recorded actions; no new planning"})
    print(json.dumps({"rows": rows, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
