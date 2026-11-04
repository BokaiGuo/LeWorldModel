"""Independently recompute time-alignment statistics from sealed CUDA outputs."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from . import p1_dev305_contact_time_alignment_20261102 as source

base, need, ROOT = source.base, source.need, source.ROOT
NAME = "p1_dev305_contact_time_alignment_saved_20261102"
CONTRACT_SHA = "4e731c63d88cb0ef783c811565f59d595d80fd6de7d24f16c712ae042935a160"
REPORT_SHA = "a31e9d713d5c186e9b4ce6e9c0fb12e3bad5169b9627c50a4da156072be57939"


def summarize(predicted, actual):
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if predicted.shape != (4, 192) or actual.shape != (4, 5, 192):
        raise ValueError("four candidate vectors and five real frames required")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("finite vectors required")
    model_scores = np.mean(predicted ** 2, axis=1)
    stats = []
    for t in range(5):
        real_scores = np.mean(actual[:, t] ** 2, axis=1)
        pairs = [(model_scores[i] - model_scores[j], real_scores[i] - real_scores[j])
                 for i, j in combinations(range(4), 2)]
        comparable = [(x, y) for x, y in pairs if abs(x) > 1e-12 and abs(y) > 1e-12]
        stats.append({"step": t + 1, "model_choice": int(np.argmin(model_scores)),
                      "real_choice": int(np.argmin(real_scores)),
                      "comparable_pairs": len(comparable),
                      "reversed_pairs": int(sum(x * y < 0 for x, y in comparable)),
                      "forecast_vs_real_mse_mean": float(np.mean((predicted - actual[:, t]) ** 2)),
                      "real_goal_mse": real_scores.tolist()})
    return stats


def analyze():
    contract = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(ROOT / "results" / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    need(contract == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == contract["model_hash"]
         and report["counts"] == {
             "encoder": {"calls": 61, "samples": 61}, "projector": {"calls": 61, "samples": 61}}
         and len(contract["rows"]) == len(report["rows"]) == 12, "contract")
    grouped = {}
    for spec, row in zip(contract["rows"], report["rows"], strict=True):
        need((spec["arm"], spec["candidate"]) == (row["arm"], row["candidate"])
             and len(row["steps"]) == 5, "binding")
        predicted = np.asarray(spec["predicted_goal_delta"], dtype=np.float64)
        vectors = []
        for step, entry in enumerate(row["steps"], 1):
            delta = np.asarray(entry["real_goal_delta"], dtype=np.float64)
            need(entry["step"] == step and delta.shape == (192,)
                 and np.isclose(entry["real_goal_mse"], np.mean(delta**2), rtol=1e-12)
                 and np.isclose(entry["forecast_vs_real_mse"], np.mean((predicted-delta)**2), rtol=1e-12),
                 "step recomputation")
            vectors.append(delta)
        grouped.setdefault(spec["arm"], []).append((spec["candidate"], predicted, vectors))
    rows = []
    for arm, values in grouped.items():
        need([x[0] for x in values] == list(source.LABELS), "candidate order")
        rows.append({"arm": arm, "candidates": list(source.LABELS),
                     "steps": summarize([x[1] for x in values], [x[2] for x in values])})
    return {"schema": NAME + ".v1", "rows": rows,
            "scope": "same five-step model prediction compared to real step 1-5 latents; earlier steps are not forecasts for that time; one known root with correlated candidates",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
