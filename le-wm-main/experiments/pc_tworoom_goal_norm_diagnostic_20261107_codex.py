"""Sealed predicted-only scoring before evaluation-only physical labels."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_block_b_reencode_alignment_20261107_codex as source
from experiments.pc_input_permissions import file_hash

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_GOAL_NORM_SCORE_DIAGNOSTIC_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_tworoom_goal_norm_diagnostic_20261107_codex"
PLAN = Path(str(BASE) + "_plan.json")
RESULT = Path(str(BASE) + "_result.json")


def scores(prediction, goal, reference):
    x, g, ref = (np.asarray(z, dtype=np.float64) for z in (prediction, goal, reference))
    if x.shape != (64, 192) or g.shape != (192,) or ref.shape != (64,):
        raise ValueError("unexpected latent or candidate shape")
    if not all(np.isfinite(z).all() for z in (x, g, ref)):
        raise ValueError("non-finite input")
    norms, goal_norm = np.linalg.norm(x, axis=1), float(np.linalg.norm(g))
    if np.any(norms < 1e-12) or goal_norm < 1e-12:
        raise ValueError("zero norm")
    squared = np.square(x - g).sum(axis=1)
    if np.max(np.abs(squared - ref)) > 1e-3:
        raise ValueError("original cost does not match prediction")
    cosine = 1.0 - (x @ g) / (norms * goal_norm)
    return cosine, norms, goal_norm


def seal():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    meta = json.loads(source.PRED_META.read_text())
    prev_result = json.loads(source.RESULT.read_text())
    if meta["pred_sha256"] != file_hash(source.PRED) or prev_result["features_sha256"] != file_hash(source.FEATURES):
        raise ValueError("sealed input drift")
    rows = []
    with np.load(source.PRED, allow_pickle=False) as pred, np.load(source.FEATURES, allow_pickle=False) as features:
        for root in range(10):
            original = pred[f"r{root}_cost"]
            alternative, norms, goal_norm = scores(pred[f"r{root}_predicted"], features[f"r{root}_goal"], original)
            rows.append({"root": root, "original_choice": int(np.argmin(original)),
                         "cosine_choice": int(np.argmin(alternative)), "cosine_scores": alternative.tolist(),
                         "prediction_norm_min": float(norms.min()), "prediction_norm_max": float(norms.max()),
                         "goal_norm": goal_norm})
    with PLAN.open("x") as out:
        json.dump({"protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "prediction_sha256": file_hash(source.PRED), "features_sha256": file_hash(source.FEATURES),
                   "rows": rows, "scope": "predictions and goal only; no physical labels"}, out, indent=2)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = json.loads(PLAN.read_text())
    for name, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                       ("prediction_sha256", source.PRED), ("features_sha256", source.FEATURES)):
        if plan[name] != file_hash(path):
            raise ValueError(f"sealed input drift: {name}")
    _, physical = source.source_records()
    rows = []
    for item, original in zip(plan["rows"], physical["rows"], strict=True):
        if item["root"] != original["root"] or len(original["outcomes"]) != 64:
            raise ValueError("root or candidate mismatch")
        base = original["outcomes"][item["original_choice"]]
        cosine = original["outcomes"][item["cosine_choice"]]
        rows.append({"root": item["root"], "original_choice": item["original_choice"],
                     "cosine_choice": item["cosine_choice"],
                     "original_ever": base["ever"], "cosine_ever": cosine["ever"],
                     "original_final_distance": base["final_distance"],
                     "cosine_final_distance": cosine["final_distance"]})
    wins = sum(r["cosine_ever"] and not r["original_ever"] for r in rows)
    losses = sum(r["original_ever"] and not r["cosine_ever"] for r in rows)
    original_distance = float(np.mean([r["original_final_distance"] for r in rows]))
    cosine_distance = float(np.mean([r["cosine_final_distance"] for r in rows]))
    summary = {"original_success": sum(r["original_ever"] for r in rows),
               "cosine_success": sum(r["cosine_ever"] for r in rows),
               "changed_choices": sum(r["original_choice"] != r["cosine_choice"] for r in rows),
               "wins": wins, "losses": losses, "original_mean_final_distance": original_distance,
               "cosine_mean_final_distance": cosine_distance,
               "development_exit": bool(wins >= 2 and losses == 0 and cosine_distance <= original_distance)}
    with RESULT.open("x") as out:
        json.dump({"plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(source.prior.PHYSICAL),
                   "rows": rows, "summary": summary, "scope": "exposed fixed-pool diagnostic"}, out, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "evaluate"):
        raise SystemExit("expected seal|evaluate")
    {"seal": seal, "evaluate": evaluate}[sys.argv[1]]()
