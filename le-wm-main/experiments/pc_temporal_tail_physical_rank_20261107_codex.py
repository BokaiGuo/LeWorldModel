"""Post-hoc same-prefix temporal ranking against sealed physical replays."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_effect_bank_physical_alignment_20261107_codex as physical
from experiments import pc_effect_bank_three_way_alignment_20261107_codex as alignment
from experiments import pc_margin_temporal_pair_selection_20261107_codex as temporal
from experiments.pc_input_permissions import file_hash

ROOTS = range(20, 50)
PROTOCOL = physical.prior.ROOT.parent / "docs/PC_TEMPORAL_TAIL_PHYSICAL_RANK_PROTOCOL_20261107_CODEX.md"
OUTPUT = physical.prior.ROOT / "results/pc_temporal_tail_physical_rank_20261107_codex.json"
PRED_TIE = 0.005
PHYS_TIE = 1e-6


def metrics(predicted, distances, success, eligible):
    predicted, distances, success, eligible = map(np.asarray, (predicted, distances, success, eligible))
    if (predicted.shape != (8, 8) or distances.shape != (8, 8) or success.shape != (8, 8)
            or eligible.shape != (8,) or not np.isfinite(predicted).all()
            or not np.isfinite(distances).all()):
        raise ValueError("invalid score matrix")
    i, j = np.triu_indices(8, 1)
    dp = predicted[:, i] - predicted[:, j]
    dt = distances[:, i] - distances[:, j]
    considered = np.broadcast_to(eligible[:, None], dp.shape)
    valid = considered & (np.abs(dp) > PRED_TIE) & (np.abs(dt) > PHYS_TIE)
    pred_tie = considered & (np.abs(dp) <= PRED_TIE)
    phys_tie = considered & (np.abs(dt) <= PHYS_TIE)
    correct = valid & (np.sign(dp) == np.sign(dt))
    chosen = np.argmin(predicted, axis=1)
    row = np.arange(8)
    varying = eligible & success.any(axis=1) & ~success.all(axis=1)
    return {"correct_pairs": int(correct.sum()), "valid_pairs": int(valid.sum()),
            "all_pairs": int(considered.sum()), "pred_tie_pairs": int(pred_tie.sum()),
            "physical_tie_pairs": int(phys_tie.sum()), "eligible_prefixes": int(eligible.sum()),
            "chosen_distance_sum": float(distances[row, chosen][eligible].sum()),
            "mean_random_distance_sum": float(distances.mean(axis=1)[eligible].sum()),
            "oracle_distance_sum": float(distances.min(axis=1)[eligible].sum()),
            "chosen_success": int(success[row, chosen][eligible].sum()),
            "success_mixed_prefixes": int(varying.sum()),
            "chosen_success_mixed": int(success[row, chosen][varying].sum())}


def aggregate(rows, label, name):
    values = [r[label][name] for r in rows]
    valid = [x["correct_pairs"] / x["valid_pairs"] for x in values if x["valid_pairs"]]
    count = sum(x["eligible_prefixes"] for x in values)
    return {"roots_with_valid_pairs": len(valid),
            "root_macro_accuracy": float(np.mean(valid)) if valid else None,
            **{k: sum(x[k] for x in values) for k in
               ("correct_pairs", "valid_pairs", "all_pairs", "pred_tie_pairs", "physical_tie_pairs",
                "eligible_prefixes", "chosen_success", "success_mixed_prefixes", "chosen_success_mixed")},
            **{k.replace("_sum", "_mean"): (sum(x[k] for x in values) / count if count else None)
               for k in ("chosen_distance_sum", "mean_random_distance_sum", "oracle_distance_sum")}}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    physical.verify()
    meta = json.loads(alignment.META.read_text())
    tplan = json.loads(temporal.PLAN.read_text())
    if (meta["source_physical_sha256"] != file_hash(physical.PHYS)
            or tplan["scores_sha256"] != file_hash(temporal.SCORES)
            or tplan["source_meta_sha256"] != file_hash(alignment.META)):
        raise ValueError("source binding drift")
    rows = []
    with (np.load(temporal.SCORES, allow_pickle=False) as predicted,
          np.load(physical.PHYS, allow_pickle=False) as truth):
        for root in ROOTS:
            scores = predicted[f"r{root}_scores"]
            dist, term = truth[f"r{root}_distance"], truth[f"r{root}_term"]
            if scores.shape != (8, 9, 5) or dist.shape != (8, 9, 25) or term.shape != (8, 9, 25):
                raise ValueError("candidate/time shape mismatch")
            if not np.allclose(dist[:, :, :10], dist[:, :1, :10], rtol=0, atol=1e-9):
                raise ValueError("inconsistent physical prefix distance")
            if not np.array_equal(term[:, :, :10], np.broadcast_to(term[:, :1, :10], (8, 9, 10))):
                raise ValueError("inconsistent physical prefix success")
            sampled = dist[:, 1:, [14, 19, 24]].min(axis=-1)
            actual = dist[:, 1:, 10:].min(axis=-1)
            success = term[:, 1:, 10:].any(axis=-1)
            groups = {"all": np.ones(8, dtype=bool),
                      "prefix_unsuccessful": ~term[:, 0, :10].any(axis=-1)}
            row = {"root": root}
            for label, eligible in groups.items():
                row[label] = {}
                for name, pred, distance in (
                    ("temporal_sampled", scores[:, 1:, 2:].min(axis=-1), sampled),
                    ("temporal_continuous", scores[:, 1:, 2:].min(axis=-1), actual),
                    ("endpoint_continuous", scores[:, 1:, -1], actual)):
                    row[label][name] = metrics(pred, distance, success, eligible)
            rows.append(row)
    summary = {group: {name: aggregate(rows, group, name) for name in
                       ("temporal_sampled", "temporal_continuous", "endpoint_continuous")}
               for group in ("all", "prefix_unsuccessful")}
    with OUTPUT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "physical_sha256": file_hash(physical.PHYS),
                   "prediction_sha256": file_hash(temporal.SCORES), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "scope": "post-hoc exposed fixed-candidate diagnostic"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
