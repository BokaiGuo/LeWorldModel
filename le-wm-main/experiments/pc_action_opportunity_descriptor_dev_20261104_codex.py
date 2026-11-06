"""Stage-1 development test for finite action-opportunity descriptors."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from experiments import pc_candidate_suffix_quality_posthoc as quality
from experiments import pc_discovery_attribution_dev as panel
from experiments.pc_action_opportunity_stats import spearman


OUT = panel.ROOT / "results/pc_action_opportunity_descriptor_dev_20261104_codex"
PROTOCOL = panel.ROOT.parent / "docs/PC_ACTION_OPPORTUNITY_DESCRIPTOR_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("spectral", "random_direction", "diverse")


def _patterns(length: int, heldout: bool) -> np.ndarray:
    if length <= 0:
        return np.zeros((0, 0, 2), np.float32)
    t = np.arange(length, dtype=np.float32)
    if not heldout:
        rows = [
            np.column_stack((np.full(length, 0.5), np.zeros(length))),
            np.column_stack((np.full(length, -0.5), np.zeros(length))),
            np.column_stack((np.zeros(length), np.full(length, 0.5))),
            np.column_stack((np.zeros(length), np.full(length, -0.5))),
        ]
    else:
        rows = [
            np.column_stack((np.where((t % 2) == 0, 0.5, -0.5), np.zeros(length))),
            np.column_stack((np.zeros(length), np.where((t % 2) == 0, 0.5, -0.5))),
            np.column_stack((np.full(length, 0.5), np.where((t % 2) == 0, 0.5, -0.5))),
            np.column_stack((np.where((t % 2) == 0, 0.5, -0.5), np.full(length, -0.5))),
        ]
    return np.asarray(rows, np.float32)


def _summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["status"] == "completed"]
    if not valid:
        return {"count": 0}
    a = np.asarray([row["a_max_improvement"] for row in valid], np.float64)
    b = np.asarray([row["b_max_improvement"] for row in valid], np.float64)
    best = np.asarray([row["best_score"] for row in valid], np.float64)
    immediate = np.asarray([row["immediate_score"] for row in valid], np.float64)
    top = a >= np.quantile(a, 0.9)
    return {
        "count": len(valid),
        "successes": int(sum(row["success"] for row in valid)),
        "spearman_a_vs_b_improvement": spearman(a, b),
        "spearman_a_vs_negative_best_score": spearman(a, -best),
        "spearman_immediate_vs_negative_best_score": spearman(-immediate, -best),
        "top10_a_count": int(top.sum()),
        "top10_a_best_score_mean": float(best[top].mean()) if top.any() else None,
        "all_best_score_mean": float(best.mean()),
        "top10_a_success_rate": float(np.mean([valid[i]["success"] for i in np.flatnonzero(top)]))
        if top.any() else None,
        "all_success_rate": float(np.mean([row["success"] for row in valid])),
        "tokens": int(sum(row["tokens"] for row in valid)),
        "prediction_calls": int(sum(row["prediction_calls"] for row in valid)),
    }


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    rows = []
    for root, task in enumerate(panel.prior.tasks()):
        for arm in ARMS:
            source = json.loads((quality.OUT / f"root{root:02d}_{arm}.json").read_text())
            if source["status"] != "completed":
                continue
            recorder = panel.journal.PredictionJournal(
                OUT / f"root{root:02d}_{arm}_calls",
                panel.prior.old.previous.fixed.predictor(task),
                action_dim=2,
                packet_length=5,
                token_cap=100000,
            )
            for item in source["candidates"]:
                if item.get("status") == "failed":
                    rows.append({"root": root, "arm": arm, "depth": item["depth"],
                                 "index": item["index"], "status": "failed"})
                    continue
                prefix = np.asarray(item["prefix"], np.float32)
                length = 15 - len(prefix)
                a_suffixes = _patterns(length, heldout=False)
                b_suffixes = _patterns(length, heldout=True)
                actions = np.vstack([
                    np.concatenate((prefix, np.zeros((length, 2), np.float32)))[None],
                    np.asarray([np.concatenate((prefix, suffix)) for suffix in a_suffixes]),
                    np.asarray([np.concatenate((prefix, suffix)) for suffix in b_suffixes]),
                ]).astype(np.float32)
                before_tokens = recorder.tokens
                _positions, scores = recorder(actions)
                scores = np.asarray(scores, np.float64)
                baseline = float(scores[0])
                a = baseline - scores[1:5]
                b = baseline - scores[5:9]
                rows.append({
                    "root": root, "arm": arm, "depth": item["depth"],
                    "index": item["index"], "status": "completed",
                    "immediate_score": float(item["immediate_score"]),
                    "initial_score": float(item["initial_score"]),
                    "best_score": float(item["best_score"]),
                    "success": bool(item["success"]),
                    "a_improvements": a.tolist(), "b_improvements": b.tolist(),
                    "a_max_improvement": float(np.max(a)),
                    "b_max_improvement": float(np.max(b)),
                    "baseline_score": baseline, "prediction_calls": 1,
                    "tokens": int(recorder.tokens - before_tokens),
                })
    summaries = {"all": _summary(rows)}
    for arm in ARMS:
        summaries[arm] = _summary([row for row in rows if row["arm"] == arm])
    result = {
        "schema": "pc_action_opportunity_descriptor_dev.v2",
        "status": "completed",
        "protocol": str(PROTOCOL),
        "rows": rows,
        "summaries": summaries,
        "scope": "Stage-1 exact-predictor development diagnostic; no planner mutation",
        "not_independent_confirmation": True,
        "not_visual_or_physical": True,
        "candidate_bank_source": str(quality.OUT),
    }
    panel.journal.write_json(OUT / "result.json", result)
    return result


if __name__ == "__main__":
    value = run()
    print(json.dumps(value["summaries"], indent=2))
