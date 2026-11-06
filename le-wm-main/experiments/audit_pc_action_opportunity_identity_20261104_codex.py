"""Post-hoc root-held-out audit of probe identity, not a planner or physics test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.audit_pc_action_opportunity_vector_prediction_20261104_codex import _ridge_predict
from experiments.pc_action_opportunity_stats import spearman

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/pc_action_opportunity_random_probe_dev_v2_20261104_codex/result.json"
OUTPUT = ROOT / "results/pc_action_opportunity_identity_audit_20261104_codex.json"
SEED = 174219
MODES = ("scalar", "ordered", "sorted", "shuffled")
TARGETS = ("relative_improvement", "absolute_cost")


def features(vectors: np.ndarray, controls: np.ndarray, mode: str, seed: int) -> np.ndarray:
    if mode == "scalar":
        core = vectors.max(axis=1, keepdims=True)
    elif mode == "ordered":
        core = vectors
    elif mode == "sorted":
        core = np.sort(vectors, axis=1)
    elif mode == "shuffled":
        rng = np.random.default_rng(seed)
        core = np.stack([rng.permutation(row) for row in vectors])
    else:
        raise ValueError(mode)
    return np.column_stack([core, controls])


def targets(b_vectors: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    relative = b_vectors.max(axis=1)
    return {"relative_improvement": relative, "absolute_cost": baseline - relative}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    payload = json.loads(INPUT.read_text())
    if payload["status"] != "completed" or any(r.get("status") != "completed" for r in payload["rows"]):
        raise ValueError("input incomplete; no silent dropping")
    rows = payload["rows"]
    keys = [(r["root"], r["arm"], r["depth"], r["index"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate candidate key")
    roots = np.asarray([int(r["root"]) for r in rows])
    depths = np.asarray([int(r["depth"]) for r in rows])
    arms = np.asarray([r["arm"] for r in rows])
    a = np.asarray([r["a_improvements"] for r in rows], dtype=np.float64)
    b = np.asarray([r["b_improvements"] for r in rows], dtype=np.float64)
    baseline = np.asarray([r["baseline_score"] for r in rows], dtype=np.float64)
    immediate = np.asarray([r["immediate_score"] for r in rows], dtype=np.float64)
    if a.shape != (len(rows), 8) or b.shape != a.shape:
        raise ValueError("unexpected probe shape")
    controls = np.column_stack([immediate, baseline, 15 - 5 * depths])
    if not np.isfinite(np.column_stack([a, b, controls])).all():
        raise ValueError("nonfinite input")
    design = {mode: features(a, controls, mode, SEED) for mode in MODES}
    outcomes = targets(b, baseline)
    records = []
    for root in sorted(set(roots.tolist())):
        train = roots != root
        test = ~train
        for target_name in TARGETS:
            y = outcomes[target_name]
            predictions = {
                mode: _ridge_predict(design[mode][train], y[train], design[mode][test])
                for mode in MODES
            }
            # Equal-weight candidate rows within each root; duplicate prefixes
            # remain and are explicitly not independent observations.
            truth = y[test]
            item = {
                "root": int(root), "target": target_name, "candidate_count": int(test.sum()),
                "mae": {mode: float(np.mean(abs(predictions[mode] - truth))) for mode in MODES},
                "spearman": {mode: spearman(predictions[mode], truth) for mode in MODES},
                "strata": [],
            }
            for arm in sorted(set(arms[test].tolist())):
                for depth in sorted(set(depths[test].tolist())):
                    subgroup = (arms[test] == arm) & (depths[test] == depth)
                    item["strata"].append({
                        "arm": arm, "depth": int(depth), "candidate_count": int(subgroup.sum()),
                        "mae": {mode: float(np.mean(abs(predictions[mode][subgroup] - truth[subgroup]))) for mode in MODES},
                        "spearman": {mode: spearman(predictions[mode][subgroup], truth[subgroup]) for mode in MODES},
                    })
            records.append(item)
    summaries = {}
    for target in TARGETS:
        selected = [row for row in records if row["target"] == target]
        differences = [row["mae"]["ordered"] - row["mae"]["sorted"] for row in selected]
        summaries[target] = {
            "root_count": len(selected),
            "mean_root_mae": {mode: float(np.mean([r["mae"][mode] for r in selected])) for mode in MODES},
            "ordered_minus_sorted_mean_root_mae": float(np.mean(differences)),
            "ordered_lower_mae_roots": sum(d < -1e-12 for d in differences),
            "sorted_lower_mae_roots": sum(d > 1e-12 for d in differences),
            "ties": sum(abs(d) <= 1e-12 for d in differences),
        }
    result = {
        "schema": "pc_action_opportunity_identity_audit.v1", "status": "completed",
        "source": str(INPUT.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_predictor_sha256": hashlib.sha256(
            (ROOT / "experiments/audit_pc_action_opportunity_vector_prediction_20261104_codex.py").read_bytes()
        ).hexdigest(),
        "seed": SEED, "ridge_penalty": 1.0,
        "scope": "post-hoc same-panel leave-one-root-out development; unseen finite B suffixes, not independent task confirmation, physical outcome, or planner",
        "warning": "sorted/shuffled change feature distribution and model class, and shuffled uses one seed; no causal isolation of semantic action identity",
        "records": records, "summaries": summaries,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    print("Assessing identity-sensitive versus identity-free probe-vector predictions; lower MAE is better.")
    main()
