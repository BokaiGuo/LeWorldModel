"""Post-hoc finite-support opportunity bounds and physical alignment."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_block_b_reencode_alignment_20261107_codex as src
from experiments.pc_input_permissions import file_hash
from experiments.research_runtime.artifacts import atomic_write_json

PROTOCOL = src.ROOT.parent / "docs/PC_EMPIRICAL_OPPORTUNITY_BOUND_PROTOCOL_20261107_CODEX.md"
OUT = src.ROOT / "results/pc_empirical_opportunity_bound_20261107_codex.json"


def evaluate(prediction, encoding, outcomes):
    p, q = np.asarray(prediction, dtype=np.float64), np.asarray(encoding, dtype=np.float64)
    if p.shape != (8, 8) or q.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("expected finite prefix by tail matrices")
    if len(outcomes) != 64:
        raise ValueError("expected 64 physical outcomes")
    error = np.max(np.abs(p-q), axis=1)
    pred_best, enc_best = p.min(axis=1), q.min(axis=1)
    dist = np.asarray([o["final_distance"] for o in outcomes], dtype=np.float64).reshape(8, 8)
    if not np.isfinite(dist).all():
        raise ValueError("invalid physical distance")
    phys_best = dist.min(axis=1)
    post_success = np.asarray([bool(o["ever"]) and not bool(o["prefix_success"]) for o in outcomes]).reshape(8, 8).any(axis=1)
    pairs = []
    for i, j in combinations(range(8), 2):
        gap = pred_best[i] - pred_best[j]
        bound = float(error[i] + error[j])
        global_bound = float(2 * error.max())
        certified, globally_certified = bool(abs(gap) > bound), bool(abs(gap) > global_bound)
        encoding_gap = enc_best[i] - enc_best[j]
        if certified and gap * encoding_gap <= 0:
            raise ValueError("finite-set ordering theorem violated")
        if globally_certified and not certified:
            raise ValueError("global bound inconsistency")
        physical_gap = phys_best[i] - phys_best[j]
        success_discriminable = bool(post_success[i] != post_success[j])
        pairs.append({"i": i, "j": j, "predicted_gap": float(gap),
                      "encoded_gap": float(encoding_gap), "physical_best_distance_gap": float(physical_gap),
                      "local_error_bound": bound, "global_error_bound": global_bound,
                      "certified": certified, "global_certified": globally_certified,
                      "encoded_same_sign": bool(gap * encoding_gap > 0),
                      "physical_nontie": bool(gap != 0 and physical_gap != 0),
                      "physical_same_sign": bool(gap * physical_gap > 0),
                      "post_success_discriminable": success_discriminable,
                      "pred_prefers_success": bool(success_discriminable and
                                                   ((gap < 0 and post_success[i]) or (gap > 0 and post_success[j])))})
    return {"epsilon_by_prefix": error.tolist(), "prediction_opportunity": pred_best.tolist(),
            "encoding_opportunity": enc_best.tolist(), "physical_best_distance": phys_best.tolist(),
            "post_success": post_success.tolist(), "pairs": pairs}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, physical = src.source_records()
    metadata, result = json.loads(src.PRED_META.read_text()), json.loads(src.RESULT.read_text())
    for actual, path in ((metadata["pred_sha256"], src.PRED), (metadata["pool_sha256"], src.prior.POOL),
                         (metadata["plan_sha256"], src.prior.PLAN), (result["features_sha256"], src.FEATURES),
                         (result["prediction_sha256"], src.PRED), (result["protocol_sha256"], src.PROTOCOL),
                         (result["runner_sha256"], src.__file__)):
        if actual != file_hash(path):
            raise ValueError(f"artifact drift: {path}")
    rows = []
    with np.load(src.PRED, allow_pickle=False) as pred, np.load(src.FEATURES, allow_pickle=False) as encoded:
        for root in range(10):
            if physical["rows"][root]["root"] != root or plan["rows"][root]["root"] != root:
                raise ValueError("root mismatch")
            row = evaluate(pred[f"r{root}_cost"].reshape(8, 8),
                           encoded[f"r{root}_encoded_cost"].reshape(8, 8),
                           physical["rows"][root]["outcomes"])
            rows.append({"root": root, **row})
    pairs = [pair for row in rows for pair in row["pairs"]]
    def stats(group):
        nontied = [x for x in group if x["physical_nontie"]]
        successes = [x for x in group if x["post_success_discriminable"] and x["predicted_gap"] != 0]
        return {"pairs": len(group), "encoding_order_agrees": sum(x["encoded_same_sign"] for x in group),
                "physical_nonties": len(nontied), "physical_distance_order_agrees": sum(x["physical_same_sign"] for x in nontied),
                "post_success_discriminable": len(successes), "prefers_post_success": sum(x["pred_prefers_success"] for x in successes)}
    summary = {"roots": len(rows), "all": stats(pairs),
               "local_certificate": stats([x for x in pairs if x["certified"]]),
               "global_certificate": stats([x for x in pairs if x["global_certified"]]),
               "roots_with_local_certification": sum(any(p["certified"] for p in row["pairs"]) for row in rows),
               "roots_with_global_certification": sum(any(p["global_certified"] for p in row["pairs"]) for row in rows)}
    atomic_write_json(OUT, {"scope": "post-hoc finite B pool; empirical oracle bound, not deployable certificate",
                            "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                            "prediction_sha256": file_hash(src.PRED), "encoding_sha256": file_hash(src.FEATURES),
                            "physical_sha256": file_hash(src.prior.PHYSICAL), "summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
