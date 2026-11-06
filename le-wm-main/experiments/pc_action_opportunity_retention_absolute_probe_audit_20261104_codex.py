"""Post-hoc fixed-pool retention audit with a genuine absolute probe minimum.

This is a development reanalysis, not a preregistered or end-to-end test.
The same random probe bank is used for both candidate-source arms.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pc_action_opportunity_retention_fixed_opportunity_audit_20261104_codex import (
    QUALITY,
    ROOT,
    _select,
)

DESCRIPTORS = ROOT / "results/pc_action_opportunity_random_probe_dev_v2_20261104_codex/result.json"
OUTPUT = ROOT / "results/pc_action_opportunity_retention_absolute_probe_audit_20261104_codex.json"
ARMS = ("random_direction", "spectral")
POLICIES = ("absolute_probe_min", "relative_improvement", "effect_complement")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_absolute(candidates: list[dict], descriptors: list[dict], slots: int) -> list[int]:
    if not 1 <= slots <= len(candidates) or len(candidates) != len(descriptors):
        raise ValueError("invalid candidates or slot count")
    greedy = min(range(len(candidates)), key=lambda i: (float(candidates[i]["immediate_score"]), i))
    # improvement_j = baseline - absolute_cost_j, so the minimum absolute
    # cost is baseline - max(improvement).  Baselines differ by candidate.
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            float(descriptors[i]["baseline_score"]) - max(descriptors[i]["a_improvements"]),
            i,
        ),
    )
    return ([greedy] + [i for i in order if i != greedy])[:slots]


def load_descriptors(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload["status"] != "completed":
        raise ValueError("descriptor panel incomplete")
    indexed = {}
    for row in payload["rows"]:
        if row.get("arm") not in ARMS:
            continue
        if row.get("status") != "completed":
            raise ValueError("failed descriptor: cannot silently drop candidate")
        key = (int(row["root"]), row["arm"], int(row["depth"]), int(row["index"]))
        if key in indexed:
            raise ValueError(f"duplicate descriptor key: {key}")
        values = [row["baseline_score"], row["immediate_score"], row["best_score"], *row["a_improvements"]]
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all() or len(row["a_improvements"]) != 8:
            raise ValueError(f"invalid descriptor: {key}")
        indexed[key] = row
    return indexed


def run() -> dict:
    indexed = load_descriptors(DESCRIPTORS)
    used = set()
    rows = []
    sources = {}
    for arm in ARMS:
        for root in range(16):
            source_path = QUALITY / f"root{root:02d}_{arm}.json"
            sources[str(source_path.relative_to(ROOT))] = digest(source_path)
            source = json.loads(source_path.read_text())
            if source["status"] != "completed" or source["root"] != root or source["arm"] != arm:
                raise ValueError(f"invalid source: {source_path}")
            for depth in (0, 1):
                candidates = sorted(
                    (x for x in source["candidates"] if x["depth"] == depth),
                    key=lambda x: x["index"],
                )
                if not candidates or any(x.get("status") not in (None, "completed") for x in candidates):
                    raise ValueError(f"missing or failed candidates: {root} {arm} {depth}")
                keys = [(root, arm, depth, int(x["index"])) for x in candidates]
                if len(set(keys)) != len(keys):
                    raise ValueError(f"duplicate quality key: {root} {arm} {depth}")
                descriptors = [indexed[key] for key in keys]
                used.update(keys)
                for candidate, descriptor in zip(candidates, descriptors):
                    for field in ("immediate_score", "best_score"):
                        if not np.isclose(candidate[field], descriptor[field], rtol=0, atol=1e-12):
                            raise ValueError(f"{field} mismatch: {root} {arm} {depth}")
                    if bool(candidate["success"]) != bool(descriptor["success"]):
                        raise ValueError("success mismatch")
                slots = sum(bool(x["retained"]) for x in candidates)
                for policy in POLICIES:
                    if policy == "absolute_probe_min":
                        selected = select_absolute(candidates, descriptors, slots)
                    else:
                        old_policy = "probe_min" if policy == "relative_improvement" else policy
                        selected = _select(candidates, descriptors, old_policy, slots)
                    rows.append({
                        "root": root, "arm": arm, "depth": depth, "policy": policy,
                        "candidate_count": len(candidates), "slots": slots,
                        "selected_indices": [int(candidates[i]["index"]) for i in selected],
                        "selected_best_score": min(float(candidates[i]["best_score"]) for i in selected),
                        "pool_best_score": min(float(x["best_score"]) for x in candidates),
                        "selected_success": any(bool(candidates[i]["success"]) for i in selected),
                        "pool_success": any(bool(x["success"]) for x in candidates),
                    })
    if used != set(indexed):
        raise ValueError(f"descriptor coverage mismatch: used={len(used)} available={len(indexed)}")
    root_pairs = []
    for arm in ARMS:
        for root in range(16):
            subsets = [r for r in rows if r["arm"] == arm and r["root"] == root]
            means = {
                policy: float(np.mean([r["selected_best_score"] for r in subsets if r["policy"] == policy]))
                for policy in POLICIES
            }
            root_pairs.append({
                "arm": arm, "root": root, "mean_by_policy": means,
                "effect_minus_absolute": means["effect_complement"] - means["absolute_probe_min"],
            })
    summaries = {}
    for arm in ARMS:
        arm_pairs = [r for r in root_pairs if r["arm"] == arm]
        summaries[arm] = {
            "root_count": len(arm_pairs),
            "mean_root_score": {
                policy: float(np.mean([r["mean_by_policy"][policy] for r in arm_pairs]))
                for policy in POLICIES
            },
            "effect_minus_absolute_mean": float(np.mean([r["effect_minus_absolute"] for r in arm_pairs])),
            "effect_better_roots": sum(r["effect_minus_absolute"] < -1e-12 for r in arm_pairs),
            "effect_worse_roots": sum(r["effect_minus_absolute"] > 1e-12 for r in arm_pairs),
            "ties": sum(abs(r["effect_minus_absolute"]) <= 1e-12 for r in arm_pairs),
            "pool_success_groups": sum(r["pool_success"] for r in rows if r["arm"] == arm and r["policy"] == POLICIES[0]),
        }
    return {
        "schema": "pc_action_opportunity_retention_absolute_probe_audit.v1",
        "status": "completed",
        "scope": "post-hoc fixed pools, shared random A probes and finite sealed per-candidate suffix solver; not end-to-end or independent confirmation",
        "warning": "candidate indices and scores were matched; original descriptor records lack a prefix hash, so matching is not a cryptographic action binding",
        "source_sha256": {str(DESCRIPTORS.relative_to(ROOT)): digest(DESCRIPTORS), **sources},
        "code_sha256": {
            str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
            "experiments/pc_action_opportunity_retention_fixed_opportunity_audit_20261104_codex.py":
                digest(ROOT / "experiments/pc_action_opportunity_retention_fixed_opportunity_audit_20261104_codex.py"),
        },
        "rows": rows,
        "root_pairs": root_pairs,
        "summaries": summaries,
    }


if __name__ == "__main__":
    print("Post-hoc paired retention reanalysis; lower terminal score is better.")
    result = run()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summaries"], indent=2))
