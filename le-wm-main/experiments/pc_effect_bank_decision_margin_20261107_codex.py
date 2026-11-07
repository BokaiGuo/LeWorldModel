"""Post-hoc fixed-bank decision impact, with evaluator-only oracle comparisons."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_effect_bank_three_way_alignment_20261107_codex as source
from experiments.pc_input_permissions import file_hash

PROTOCOL = source.src.prior.ROOT.parent / "docs/PC_EFFECT_BANK_DECISION_MARGIN_PROTOCOL_20261107_CODEX.md"
OUT = source.src.prior.ROOT / "results/pc_effect_bank_decision_margin_20261107_codex.json"
THRESHOLDS = (1e-6, 1.0, 5.0, 10.0)


def analyze_root(model, encoded, distance, term):
    model, encoded, distance, term = map(np.asarray, (model, encoded, distance, term))
    if model.shape != (8, 9) or encoded.shape != (8, 9) or distance.shape != (8, 9) or term.shape != (8, 9, 25):
        raise ValueError("invalid bank shape")
    if not all(np.isfinite(x).all() for x in (model, encoded, distance)):
        raise ValueError("nonfinite data")
    if term.dtype != np.bool_:
        raise ValueError("termination must be boolean")
    if not np.array_equal(term[:, :, :10], np.broadcast_to(term[:, :1, :10], (8, 9, 10))):
        raise ValueError("prefix termination mismatch")
    ever = term.any(axis=-1)
    prefix = term[:, 0, :10].any(axis=-1)
    idx = np.arange(8)
    min_dist = distance.min(axis=1)
    choices = {"model": model.argmin(axis=1), "encoded_oracle": encoded.argmin(axis=1),
               "physical_oracle": distance.argmin(axis=1), "reference": np.zeros(8, dtype=int)}
    decisions = {}
    for name, selection in choices.items():
        decisions[name] = {"indices": selection.tolist(), "distance": distance[idx, selection].tolist(),
                           "regret": (distance[idx, selection] - min_dist).tolist(),
                           "ever_success": ever[idx, selection].astype(int).tolist()}
    decisions["random_expected"] = {"distance": distance.mean(axis=1).tolist(),
                                    "regret": (distance.mean(axis=1) - min_dist).tolist(),
                                    "ever_success": ever.mean(axis=1).tolist()}
    real_effect = distance[:, :1] - distance[:, 1:]
    margins = {}
    for label, score in (("model", model), ("encoded_oracle", encoded)):
        effect = score[:, :1] - score[:, 1:]
        margins[label] = {}
        for threshold in THRESHOLDS:
            mask = (np.abs(real_effect) > threshold) & (np.abs(effect) > 1e-6)
            matches = mask & (np.sign(real_effect) == np.sign(effect))
            margins[label][str(threshold)] = [int(matches.sum()), int(mask.sum())]
    return {"decisions": decisions, "margins": margins,
            "pool_ever_success": ever.any(axis=1).astype(int).tolist(),
            "prefix_ever_success": prefix.astype(int).tolist(),
            "success_available_only_after_prefix": ((~prefix) & ever.any(axis=1)).astype(int).tolist()}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    source.src.verify()
    meta, result = json.loads(source.META.read_text()), json.loads(source.RESULT.read_text())
    for actual, path in ((meta["pred_sha256"], source.PRED),
                         (meta["runner_sha256"], source.__file__),
                         (meta["protocol_sha256"], source.PROTOCOL),
                         (meta["source_physical_sha256"], source.src.PHYS),
                         (result["features_sha256"], source.FEATURES),
                         (result["prediction_meta_sha256"], source.META),
                         (result["runner_sha256"], source.__file__)):
        if actual != file_hash(path):
            raise ValueError(f"source hash mismatch: {path}")
    rows = []
    with (np.load(source.PRED, allow_pickle=False) as pred,
          np.load(source.FEATURES, allow_pickle=False) as features,
          np.load(source.src.PHYS, allow_pickle=False) as physical):
        for root in source.src.ROOTS:
            row = analyze_root(pred[f"r{root}_cost"], features[f"r{root}_encoded_cost"],
                               physical[f"r{root}_distance"][:, :, -1],
                               physical[f"r{root}_term"])
            row["root"] = root
            rows.append(row)
    summary = {}
    for name in ("model", "encoded_oracle", "physical_oracle", "reference", "random_expected"):
        summary[name] = {metric: float(np.mean([np.mean(row["decisions"][name][metric]) for row in rows]))
                         for metric in ("distance", "regret", "ever_success")}
    summary["pool_success_prefixes"] = sum(sum(row["pool_ever_success"]) for row in rows)
    summary["prefix_success_prefixes"] = sum(sum(row["prefix_ever_success"]) for row in rows)
    summary["success_available_only_after_prefix"] = sum(sum(row["success_available_only_after_prefix"]) for row in rows)
    summary["margins"] = {}
    for name in ("model", "encoded_oracle"):
        summary["margins"][name] = {}
        for threshold in THRESHOLDS:
            values = [row["margins"][name][str(threshold)] for row in rows]
            nonempty = [v for v in values if v[1]]
            summary["margins"][name][str(threshold)] = {
                "roots": len(nonempty), "matched": sum(x[0] for x in values), "pairs": sum(x[1] for x in values),
                "macro_accuracy": float(np.mean([a / b for a, b in nonempty])) if nonempty else None}
    output = {"scope": "post-hoc exposed development, evaluator oracles are not planners",
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "source_result_sha256": file_hash(source.RESULT), "summary": summary, "rows": rows}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
