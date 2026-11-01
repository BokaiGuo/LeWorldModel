"""Independently reconstruct the sealed development-root CEM candidate trace.

This is a saved-data audit, not a model rerun or physical evaluation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/p1_cp5_cem_baseline_20261101_leibniz_RUN_MAIN01"
REPORT_SHA = "898d9a289b57259a2e9e3fec0df7f7f2a014e430f448816758a0775219ba10ca"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(report, events):
    if report["status"] != "complete" or report["failure"] is not None:
        raise ValueError("incomplete run")
    if report["model_before"] != report["model_after"] or not report["source_verified"]:
        raise ValueError("model or source mismatch")
    if [e["index"] for e in events] != list(range(9)):
        raise ValueError("event sequence")
    rng = np.random.default_rng(20261101)
    mean = np.zeros((3, 5, 2), np.float32)
    std = np.full_like(mean, 0.5)
    all_actions, all_scores = [], []
    for i in range(4):
        bank_event, update_event = events[2*i:2*i+2]
        if bank_event["kind"] != "bank" or update_event["kind"] != "update":
            raise ValueError("event kind")
        b, u = bank_event["payload"], update_event["payload"]
        if b["iteration"] != i or u["iteration"] != i:
            raise ValueError("iteration")
        if not np.array_equal(np.asarray(b["mean"], np.float32), mean) or not np.array_equal(np.asarray(b["std"], np.float32), std):
            raise ValueError("distribution mismatch")
        bank = np.clip(rng.normal(mean, std, (16, 3, 5, 2)), -1, 1).astype(np.float32)
        bank[0] = mean
        if not np.array_equal(np.asarray(b["raw"], np.float32), bank):
            raise ValueError("candidate bank mismatch")
        scores = np.asarray(u["scores"], dtype=np.float64)
        if scores.shape != (16,) or not np.isfinite(scores).all():
            raise ValueError("scores invalid")
        elites = sorted(range(16), key=lambda j: (scores[j], j))[:4]
        mean = bank[elites].mean(axis=0, dtype=np.float32)
        std = np.clip(bank[elites].std(axis=0, dtype=np.float32), .05, 1.).astype(np.float32)
        if (u["elite_indices"] != elites or
            not np.array_equal(np.asarray(u["mean"], np.float32), mean) or
            not np.array_equal(np.asarray(u["std"], np.float32), std)):
            raise ValueError("elite update mismatch")
        all_actions.extend(bank)
        all_scores.extend(scores.tolist())
    result = report["result"]
    winner = min(range(64), key=lambda j: (all_scores[j], j))
    event = events[-1]
    selected = event["payload"]
    if (event["kind"] != "selected_before_replay" or selected["index"] != winner or
        result["selected_index"] != winner or result["scores"] != all_scores or
        selected["score"] != all_scores[winner] or
        not np.array_equal(np.asarray(selected["raw"], np.float32), all_actions[winner]) or
        not np.array_equal(np.asarray(result["selected_raw"], np.float32), all_actions[winner]) or
        not np.isclose(result["replay_score"], all_scores[winner], rtol=1e-6, atol=1e-8)):
        raise ValueError("winner mismatch")
    return {"verified": True, "winner": winner, "winner_score": all_scores[winner],
            "banks": 4, "candidates": 64, "neural_calls": 0,
            "physical_execution": False, "independent_confirmation": False}


def main():
    report_file, trace_file = RUN / "report.json", RUN / "trace.jsonl"
    if sha(report_file) != REPORT_SHA:
        raise ValueError("report digest")
    report = json.loads(report_file.read_bytes())
    if sha(trace_file) != report["trace_sha256"]:
        raise ValueError("trace digest")
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    print(json.dumps(audit(report, events), sort_keys=True))


if __name__ == "__main__":
    main()
