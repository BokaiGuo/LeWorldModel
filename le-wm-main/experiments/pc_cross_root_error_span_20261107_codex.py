"""Exploratory cross-root calibration of shift-invariant opportunity margins."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_block_b_reencode_alignment_20261107_codex as source
from experiments.pc_input_permissions import file_hash
from experiments.research_runtime.artifacts import atomic_write_json

PROTOCOL = source.ROOT.parent / "docs/PC_CROSS_ROOT_ERROR_SPAN_PROTOCOL_20261107_CODEX.md"
OUT = source.ROOT / "results/pc_cross_root_error_span_20261107_codex.json"
FOLDS = (tuple(range(5)), tuple(range(5, 10)))


def span(predicted, actual):
    p, a = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if p.shape != (8, 8) or a.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError("expected finite 8x8 prediction and encoded costs")
    return float(np.ptp(p-a))


def assess(predicted, actual, outcomes, threshold):
    p, a = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    error_range = span(p, a)
    if len(outcomes) != 64 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid evaluation inputs")
    p_best, a_best = p.min(axis=1), a.min(axis=1)
    order = np.argsort(p_best, kind="stable")
    first, second = int(order[0]), int(order[1])
    gap = float(p_best[second] - p_best[first])
    selected_index = first * 8 + int(np.argmin(p[first]))
    physically_best = np.asarray([x["final_distance"] for x in outcomes], dtype=np.float64).reshape(8, 8).min(axis=1)
    p_success = outcomes[selected_index]
    triggered = gap > threshold
    observed_bound_holds = error_range <= threshold
    encoded_best = bool(a_best[first] <= a_best.min())
    if triggered and observed_bound_holds and not encoded_best:
        raise ValueError("bounded finite-set order violated")
    return {"threshold": float(threshold), "actual_error_span": error_range,
            "empirical_span_covered": bool(observed_bound_holds), "prediction_gap": gap,
            "triggered": bool(triggered), "prediction_best_prefix": first,
            "encoding_best_prefix": int(np.argmin(a_best)), "encoding_best_match": encoded_best,
            "physical_best_distance_prefix": int(np.argmin(physically_best)),
            "physical_best_distance_match": bool(physically_best[first] <= physically_best.min()),
            "selected_chain": selected_index, "selected_chain_success": bool(p_success["ever"]),
            "selected_chain_post_prefix_success": bool(p_success["ever"] and not p_success["prefix_success"]),
            "selected_chain_final_distance": float(p_success["final_distance"])}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, physical = source.source_records()
    meta, result = json.loads(source.PRED_META.read_text()), json.loads(source.RESULT.read_text())
    for actual, path in ((meta["pred_sha256"], source.PRED), (meta["pool_sha256"], source.prior.POOL),
                         (meta["plan_sha256"], source.prior.PLAN), (result["features_sha256"], source.FEATURES),
                         (result["prediction_sha256"], source.PRED), (result["protocol_sha256"], source.PROTOCOL),
                         (result["runner_sha256"], source.__file__)):
        if actual != file_hash(path):
            raise ValueError(f"source drift: {path}")
    rows, fold_records = [], []
    with np.load(source.PRED, allow_pickle=False) as pred, np.load(source.FEATURES, allow_pickle=False) as enc:
        for calibration, evaluation in (FOLDS, FOLDS[::-1]):
            threshold = max(span(pred[f"r{i}_cost"].reshape(8, 8),
                                 enc[f"r{i}_encoded_cost"].reshape(8, 8)) for i in calibration)
            fold_records.append({"calibration_roots": list(calibration), "evaluation_roots": list(evaluation),
                                 "threshold": threshold})
            for root in evaluation:
                if physical["rows"][root]["root"] != root or plan["rows"][root]["root"] != root:
                    raise ValueError("root mismatch")
                row = assess(pred[f"r{root}_cost"].reshape(8, 8),
                             enc[f"r{root}_encoded_cost"].reshape(8, 8),
                             physical["rows"][root]["outcomes"], threshold)
                rows.append({"root": root, **row})
    rows.sort(key=lambda x: x["root"])
    if [r["root"] for r in rows] != list(range(10)):
        raise ValueError("incomplete panel")
    triggered = [r for r in rows if r["triggered"]]
    summary = {"roots": len(rows), "calibration_span_covers": sum(r["empirical_span_covered"] for r in rows),
               "triggered_roots": len(triggered),
               "all_encoding_best_match": sum(r["encoding_best_match"] for r in rows),
               "triggered_encoding_best_match": sum(r["encoding_best_match"] for r in triggered),
               "all_physical_best_distance_match": sum(r["physical_best_distance_match"] for r in rows),
               "triggered_physical_best_distance_match": sum(r["physical_best_distance_match"] for r in triggered),
               "all_selected_chain_success": sum(r["selected_chain_success"] for r in rows),
               "triggered_selected_chain_success": sum(r["selected_chain_success"] for r in triggered)}
    record = {"scope": "exploratory exposed data; calibrated heuristic, not an online guarantee",
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
              "prediction_sha256": file_hash(source.PRED), "encoding_sha256": file_hash(source.FEATURES),
              "physical_sha256": file_hash(source.prior.PHYSICAL), "folds": fold_records,
              "summary": summary, "rows": rows}
    atomic_write_json(OUT, record)
    print(json.dumps({"folds": fold_records, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
