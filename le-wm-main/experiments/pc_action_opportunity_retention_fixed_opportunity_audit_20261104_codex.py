"""Fixed-candidate A-E retention audit for action-opportunity descriptors.

This uses already sealed candidate suffix-quality outcomes. It changes only
the post-hoc retention rule, so it isolates retention under a common finite
suffix opportunity. It is not an end-to-end planner run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pc_action_opportunity_stats import spearman


ROOT = Path(__file__).resolve().parents[1]
DISCOVERY = ROOT / "results/pc_discovery_attribution_dev_20261104_codex"
QUALITY = DISCOVERY / "candidate_suffix_quality_posthoc"
DESCRIPTORS = {
    "spectral": ROOT / "results/pc_action_opportunity_descriptor_dev_20261104_codex/result.json",
    "random_direction": ROOT / "results/pc_action_opportunity_random_probe_dev_v2_20261104_codex/result.json",
}
OUTPUT = ROOT / "results/pc_action_opportunity_retention_fixed_opportunity_audit_20261104_codex.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor_rows(arm: str) -> dict[tuple[int, int, int], dict]:
    payload = json.loads(DESCRIPTORS[arm].read_text())
    rows = {}
    for row in payload["rows"]:
        if row.get("arm") != arm:
            continue
        if row.get("status") == "completed":
            rows[(int(row["root"]), int(row["depth"]), int(row["index"]))] = row
    return rows


def _select(candidates: list[dict], descriptor: dict, policy: str, slots: int) -> list[int]:
    if slots < 1 or slots > len(candidates):
        raise ValueError("invalid slot count")
    immediate = np.asarray([float(row["immediate_score"]) for row in candidates])
    greedy = int(np.argmin(immediate))
    selected = [greedy]
    if slots == 1:
        return selected
    if policy == "probe_min":
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-float(descriptor[i]["a_max_improvement"]), i),
        )
        selected.extend(i for i in order if i not in selected)
    elif policy == "effect_complement":
        matrix = np.asarray([descriptor[i]["a_improvements"] for i in range(len(candidates))])
        scale = matrix.std(axis=0)
        scale[scale < 1e-12] = 1.0
        normalized = matrix / scale
        while len(selected) < slots:
            pairwise = normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :]
            distances = np.linalg.norm(pairwise, axis=2).min(axis=1)
            distances[selected] = -np.inf
            selected.append(int(np.argmax(distances)))
    else:
        raise ValueError(policy)
    return selected[:slots]


def _group_rows(arm: str) -> list[dict]:
    descriptor = _descriptor_rows(arm)
    groups = []
    for root in range(16):
        source = json.loads((QUALITY / f"root{root:02d}_{arm}.json").read_text())
        if source["status"] != "completed":
            raise ValueError(f"quality source failed: root={root} arm={arm}")
        by_depth: dict[int, list[dict]] = {}
        for item in source["candidates"]:
            # The sealed suffix-quality schema records candidate completion at
            # the source-file level; individual candidate rows predate the
            # per-row status field.  Treat an absent row status as completed,
            # but still reject an explicit non-completed status.
            if item.get("status") in (None, "completed"):
                by_depth.setdefault(int(item["depth"]), []).append(item)
        for depth, candidates in sorted(by_depth.items()):
            candidates.sort(key=lambda row: int(row["index"]))
            keys = [(root, depth, int(row["index"])) for row in candidates]
            if any(key not in descriptor for key in keys):
                raise ValueError(f"descriptor binding missing: {keys[0]}")
            drows = [descriptor[key] for key in keys]
            if any(abs(float(a["immediate_score"]) - float(b["immediate_score"])) > 1e-12
                   for a, b in zip(candidates, drows)):
                raise ValueError("immediate-score binding mismatch")
            retained_count = sum(bool(row.get("retained", False)) for row in candidates)
            if retained_count < 1:
                raise ValueError("source retention has no slots")
            for policy in ("probe_min", "effect_complement"):
                selected = _select(candidates, drows, policy, retained_count)
                selected_scores = [float(candidates[i]["best_score"]) for i in selected]
                groups.append(
                    {
                        "root": root,
                        "arm": arm,
                        "depth": depth,
                        "candidate_count": len(candidates),
                        "slots": retained_count,
                        "policy": policy,
                        "selected_indices": [int(candidates[i]["index"]) for i in selected],
                        "selected_best_score": float(min(selected_scores)),
                        "selected_success": bool(any(candidates[i]["success"] for i in selected)),
                        "pool_best_score": float(min(row["best_score"] for row in candidates)),
                        "pool_success": bool(any(row["success"] for row in candidates)),
                    }
                )
    return groups


def _summary(rows: list[dict]) -> dict:
    scores = np.asarray([row["selected_best_score"] for row in rows], dtype=np.float64)
    success = np.asarray([row["selected_success"] for row in rows], dtype=np.float64)
    pool = np.asarray([row["pool_best_score"] for row in rows], dtype=np.float64)
    return {
        "groups": len(rows),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()) if len(success) else None,
        "mean_selected_best_score": float(scores.mean()) if len(scores) else None,
        "mean_pool_best_score": float(pool.mean()) if len(pool) else None,
        "mean_gap_to_pool_best": float((scores - pool).mean()) if len(scores) else None,
        "selected_vs_pool_spearman": spearman(-scores, -pool),
    }


def main() -> None:
    rows = []
    for arm in ("random_direction", "spectral"):
        rows.extend(_group_rows(arm))
    result = {
        "schema": "pc_action_opportunity_retention_fixed_opportunity_audit.v1",
        "status": "completed",
        "source": {
            "quality_root": str(QUALITY),
            "quality_source_sha256": {
                arm: _sha256(QUALITY / f"root00_{arm}.json")
                for arm in ("random_direction", "spectral")
            },
            "descriptor_source_sha256": {
                arm: _sha256(DESCRIPTORS[arm]) for arm in DESCRIPTORS
            },
        },
        "scope": (
            "fixed candidate pool and sealed finite suffix-quality opportunity; "
            "post-hoc retention only; not end-to-end planning or independent confirmation"
        ),
        "policies": {
            "probe_min": "preserve immediate greedy slot, then maximize A max improvement",
            "effect_complement": (
                "preserve immediate greedy slot, then farthest-point select in "
                "standardized full A improvement vector"
            ),
        },
        "summaries": {
            f"{arm}_{policy}": _summary(
                [row for row in rows if row["arm"] == arm and row["policy"] == policy]
            )
            for arm in ("random_direction", "spectral")
            for policy in ("probe_min", "effect_complement")
        },
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
