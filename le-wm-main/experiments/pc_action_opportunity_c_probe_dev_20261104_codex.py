"""Frozen-protocol fresh-C-suffix development evaluation; not a planner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_candidate_suffix_quality_posthoc as quality
from experiments import pc_discovery_attribution_dev as panel
from experiments.audit_pc_action_opportunity_identity_20261104_codex import features
from experiments.audit_pc_action_opportunity_vector_prediction_20261104_codex import _ridge_predict
from experiments.pc_action_opportunity_random_probe_dev_20261104_codex import probes

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_action_opportunity_random_probe_dev_v2_20261104_codex/result.json"
OUT = ROOT / "results/pc_action_opportunity_c_probe_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_ACTION_OPPORTUNITY_C_PROBE_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("spectral", "random_direction", "diverse")
MODES = ("scalar", "ordered", "sorted")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_probes(length: int) -> np.ndarray:
    c = probes(length, 2)
    for split in (0, 1):
        previous = probes(length, split)
        if any(np.array_equal(x, y) for x in c for y in previous):
            raise ValueError(f"C probe overlaps split={split}, length={length}")
    return c


def load_rows() -> tuple[list[dict], dict]:
    source = json.loads(SOURCE.read_text())
    if source["status"] != "completed" or any(r.get("status") != "completed" for r in source["rows"]):
        raise ValueError("descriptor source has failures")
    index = {}
    for row in source["rows"]:
        key = (int(row["root"]), row["arm"], int(row["depth"]), int(row["index"]))
        if key in index:
            raise ValueError(f"duplicate descriptor {key}")
        index[key] = row
    return source["rows"], index


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    source_rows, descriptors = load_rows()
    suffixes = {length: checked_probes(length) for length in (5, 10)}
    tasks = panel.prior.tasks()
    if len(tasks) != 16:
        raise ValueError("unexpected task count")
    OUT.mkdir()
    used = set()
    rows = []
    source_hashes = {str(SOURCE.relative_to(ROOT)): sha(SOURCE), str(PROTOCOL): sha(PROTOCOL)}
    for root, task in enumerate(tasks):
        for arm in ARMS:
            qfile = quality.OUT / f"root{root:02d}_{arm}.json"
            source_hashes[str(qfile.relative_to(ROOT))] = sha(qfile)
            source = json.loads(qfile.read_text())
            if source["status"] != "completed" or source["root"] != root or source["arm"] != arm:
                raise ValueError(f"quality input failed: {qfile}")
            recorder = panel.journal.PredictionJournal(
                OUT / f"root{root:02d}_{arm}_calls",
                panel.prior.old.previous.fixed.predictor(task),
                action_dim=2, packet_length=5, token_cap=100000,
            )
            for item in source["candidates"]:
                if item.get("status") not in (None, "completed"):
                    raise ValueError("candidate failed")
                key = (root, arm, int(item["depth"]), int(item["index"]))
                if key in used or key not in descriptors:
                    raise ValueError(f"missing or duplicate key {key}")
                used.add(key)
                descriptor = descriptors[key]
                if abs(float(item["immediate_score"]) - float(descriptor["immediate_score"])) > 1e-12:
                    raise ValueError(f"candidate mismatch {key}")
                prefix = np.asarray(item["prefix"], dtype=np.float32)
                length = 15 - len(prefix)
                c = suffixes[length]
                batch = np.stack(
                    [np.concatenate((prefix, np.zeros((length, 2), dtype=np.float32)))]
                    + [np.concatenate((prefix, suffix)) for suffix in c]
                )
                before = recorder.tokens
                _, values = recorder(batch)
                values = np.asarray(values, dtype=np.float64).reshape(-1)
                if values.shape != (9,) or not np.isfinite(values).all():
                    raise ValueError(f"bad predictor values {key}")
                if abs(values[0] - float(descriptor["baseline_score"])) > 1e-7:
                    raise ValueError(f"zero-tail mismatch {key}")
                rows.append({
                    "root": root, "arm": arm, "depth": key[2], "index": key[3],
                    "prefix_sha256_float32": hashlib.sha256(prefix.tobytes()).hexdigest(),
                    "baseline_score": float(values[0]),
                    "c_costs": values[1:].tolist(),
                    "c_max_improvement": float(values[0] - values[1:].min()),
                    "tokens": int(recorder.tokens - before),
                })
    if used != set(descriptors):
        raise ValueError("candidate coverage mismatch")
    ordered = [descriptors[(r["root"], r["arm"], r["depth"], r["index"])] for r in rows]
    roots = np.asarray([r["root"] for r in rows])
    a = np.asarray([r["a_improvements"] for r in ordered], dtype=np.float64)
    train_y = np.asarray([max(r["b_improvements"]) for r in ordered])
    c_y = np.asarray([r["c_max_improvement"] for r in rows])
    controls = np.asarray([[r["immediate_score"], r["baseline_score"], 15 - 5 * r["depth"]] for r in ordered])
    matrices = {mode: features(a, controls, mode, 174219) for mode in MODES}
    root_records = []
    for root in range(16):
        train, test = roots != root, roots == root
        predictions = {m: _ridge_predict(matrices[m][train], train_y[train], matrices[m][test]) for m in MODES}
        root_records.append({
            "root": root, "candidate_count": int(test.sum()),
            "mae": {m: float(np.mean(abs(predictions[m] - c_y[test]))) for m in MODES},
        })
    diff = [r["mae"]["ordered"] - r["mae"]["sorted"] for r in root_records]
    summary = {
        "root_count": len(root_records), "candidate_count": len(rows),
        "mean_root_mae": {m: float(np.mean([r["mae"][m] for r in root_records])) for m in MODES},
        "ordered_minus_sorted": float(np.mean(diff)),
        "ordered_better_roots": sum(x < -1e-12 for x in diff),
        "sorted_better_roots": sum(x > 1e-12 for x in diff),
        "ties": sum(abs(x) <= 1e-12 for x in diff),
        "prediction_tokens": sum(r["tokens"] for r in rows),
    }
    output = {
        "schema": "pc_action_opportunity_c_probe_dev.v1", "status": "completed",
        "protocol_sha256": sha(PROTOCOL), "code_sha256": sha(Path(__file__)),
        "source_hashes": source_hashes, "rows": rows, "root_records": root_records,
        "summary": summary,
        "scope": "B-trained held-out-root C probes; same development roots; exact dynamics only",
        "absolute_cost_mae_note": "identical to relative MAE: subtract predictions and targets from the same known baseline",
    }
    (OUT / "result.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    print("Evaluating frozen B-trained predictors on new C suffixes; no planner selection.")
    main()
