"""Stage-1b random-probe coverage check for action-opportunity descriptors."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from experiments import pc_candidate_suffix_quality_posthoc as quality
from experiments import pc_discovery_attribution_dev as panel
from experiments.pc_action_opportunity_stats import spearman


OUT = panel.ROOT / "results/pc_action_opportunity_random_probe_dev_20261104_codex"
PROTOCOL = panel.ROOT.parent / "docs/PC_ACTION_OPPORTUNITY_RANDOM_PROBE_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("spectral", "random_direction", "diverse")
PROBE_COUNT = 8


def probes(length: int, split: int) -> np.ndarray:
    rng = np.random.default_rng(20411000 + 100 * length + split)
    return np.clip(rng.normal(0.0, 0.35, size=(PROBE_COUNT, length, 2)), -0.5, 0.5).astype(
        np.float32
    )


def summarize(rows: list[dict]) -> dict:
    rows = [row for row in rows if row["status"] == "completed"]
    a = np.asarray([row["a_max_improvement"] for row in rows], np.float64)
    b = np.asarray([row["b_max_improvement"] for row in rows], np.float64)
    best = np.asarray([row["best_score"] for row in rows], np.float64)
    immediate = np.asarray([row["immediate_score"] for row in rows], np.float64)
    top = a >= np.quantile(a, 0.9)
    return {
        "count": len(rows),
        "spearman_a_vs_b_improvement": spearman(a, b),
        "spearman_a_vs_negative_best_score": spearman(a, -best),
        "spearman_immediate_vs_negative_best_score": spearman(-immediate, -best),
        "top10_a_count": int(top.sum()),
        "top10_a_best_score_mean": float(best[top].mean()),
        "all_best_score_mean": float(best.mean()),
        "top10_a_success_rate": float(np.mean([rows[i]["success"] for i in np.flatnonzero(top)])),
        "all_success_rate": float(np.mean([row["success"] for row in rows])),
        "prediction_calls": int(sum(row["prediction_calls"] for row in rows)),
        "tokens": int(sum(row["tokens"] for row in rows)),
    }


def run(output_dir: Path = OUT) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir()
    rows = []
    for root, task in enumerate(panel.prior.tasks()):
        for arm in ARMS:
            source = json.loads((quality.OUT / f"root{root:02d}_{arm}.json").read_text())
            recorder = panel.journal.PredictionJournal(
                output_dir / f"root{root:02d}_{arm}_calls",
                panel.prior.old.previous.fixed.predictor(task),
                action_dim=2,
                packet_length=5,
                token_cap=100000,
            )
            for item in source["candidates"]:
                if item.get("status") == "failed":
                    rows.append({"root": root, "arm": arm, "status": "failed"})
                    continue
                prefix = np.asarray(item["prefix"], np.float32)
                length = 15 - len(prefix)
                a_suffixes, b_suffixes = probes(length, 0), probes(length, 1)
                actions = np.vstack(
                    [
                        np.concatenate((prefix, np.zeros((length, 2), np.float32)))[None],
                        np.asarray([np.concatenate((prefix, suffix)) for suffix in a_suffixes]),
                        np.asarray([np.concatenate((prefix, suffix)) for suffix in b_suffixes]),
                    ]
                ).astype(np.float32)
                before = recorder.tokens
                _positions, values = recorder(actions)
                values = np.asarray(values, np.float64)
                baseline = float(values[0])
                a, b = baseline - values[1 : 1 + PROBE_COUNT], baseline - values[1 + PROBE_COUNT :]
                rows.append(
                    {
                        "root": root,
                        "arm": arm,
                        "depth": item["depth"],
                        "index": item["index"],
                        "status": "completed",
                        "immediate_score": float(item["immediate_score"]),
                        "best_score": float(item["best_score"]),
                        "success": bool(item["success"]),
                        "a_max_improvement": float(a.max()),
                        "b_max_improvement": float(b.max()),
                        "a_improvements": a.tolist(),
                        "b_improvements": b.tolist(),
                        "baseline_score": baseline,
                        "prediction_calls": 1,
                        "tokens": int(recorder.tokens - before),
                    }
                )
    result = {
        "schema": "pc_action_opportunity_random_probe_dev.v2",
        "status": "completed",
        "protocol": str(PROTOCOL),
        "rows": rows,
        "summaries": {"all": summarize(rows), **{
            arm: summarize([row for row in rows if row["arm"] == arm]) for arm in ARMS
        }},
        "scope": "Stage-1b exact-predictor development diagnostic; no planner mutation",
        "not_independent_confirmation": True,
        "not_visual_or_physical": True,
    }
    panel.journal.write_json(output_dir / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir)["summaries"], indent=2))
