"""Post-hoc per-task probe-choice audit; evaluation truth never feeds planning."""
import json

import numpy as np

from experiments import pc_tworoom_actual_prefix_action_effect_20261107_codex as source
from experiments.pc_input_permissions import file_hash


def choice_metrics(distance, predicted, encoded):
    """Compare each latent-score choice to an evaluation-only physical oracle."""
    vectors = [np.asarray(x, dtype=np.float64) for x in (distance, predicted, encoded)]
    if any(v.shape != (5,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError("expected five finite, aligned probe scores")
    physical, model, observed = vectors
    model_index, encoded_index = int(np.argmin(model)), int(np.argmin(observed))
    best = float(physical.min())
    return {
        "oracle_distance": best,
        "physical_span": float(physical.max() - best),
        "zero_distance": float(physical[0]),
        "model_index": model_index,
        "encoded_index": encoded_index,
        "model_regret": float(physical[model_index] - best),
        "encoded_regret": float(physical[encoded_index] - best),
        "zero_regret": float(physical[0] - best),
        "model_matches_encoded": model_index == encoded_index,
    }


def main():
    data = json.loads(source.OUT.read_text())
    if (data["source_sha256"] != file_hash(source.source.OUTPUT)
            or data["protocol_sha256"] != file_hash(source.PROTOCOL)
            or data["runner_sha256"] != file_hash(source.__file__)
            or data["feature_sha256"] != file_hash(source.FEATURES)):
        raise ValueError("source artifact drift")
    rows = [r for r in data["rows"] if "ineligible_reason" not in r]
    if len(data["rows"]) != 40 or len(rows) != 35:
        raise ValueError("unexpected fixed panel")
    for arm in source.source.ARMS:
        records = []
        for row in rows:
            a = row["arms"][arm]
            # First five are the same ordered bank: zero, +x, -x, +y, -y.
            np.testing.assert_allclose(np.asarray(row["probes"])[:5], source.probes([]),
                                       atol=0, rtol=0)
            distances = [trace[-1]["distance"] for trace in a["trace"][:5]]
            records.append(choice_metrics(distances, a["predicted_cost"][:5], a["encoded_cost"][:5]))
        print(arm, "tasks", len(records))
        for key in ("model_regret", "encoded_regret", "zero_regret", "physical_span"):
            v = np.asarray([r[key] for r in records])
            print(key, "mean", float(v.mean()), "median", float(np.median(v)),
                  "positive_gt_1e-5", int((v > 1e-5).sum()), "max", float(v.max()))
        print("model_equals_encoded", sum(r["model_matches_encoded"] for r in records))
        print("model_better_same_worse_than_zero", *(
            sum(np.sign(r["model_regret"] - r["zero_regret"]) == sign for r in records)
            for sign in (-1, 0, 1)))


if __name__ == "__main__":
    main()
