"""Recompute three-way metrics directly from frozen feature and trace artifacts."""
import json

import numpy as np

from experiments import pc_tworoom_first_pool_triple_alignment_20261107_codex as source
from experiments.pc_input_permissions import file_hash

OUT = source.ROOT / "results/pc_tworoom_first_pool_triple_alignment_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    result = json.loads(source.RESULT.read_text())
    prior, saved = source.check_source()
    for key, path in (("source_sha256", source.physical.OUT), ("frames_sha256", source.FRAMES),
                      ("features_sha256", source.FEATURES), ("protocol_sha256", source.PROTOCOL),
                      ("runner_sha256", source.__file__)):
        if result[key] != file_hash(path):
            raise ValueError("source changed")
    max_model_error = max_actual_error = 0.
    with np.load(source.FEATURES, allow_pickle=False) as features:
        for i, (row, p, original) in enumerate(zip(result["rows"], prior["rows"], saved["rows"], strict=True)):
            if row["root"] != p["root"] or row["root"] != original["root"]:
                raise ValueError("root mismatch")
            predicted, actual, goal = features["predicted"][i], features["actual"][i], features["goal"][i]
            model_cost = np.square(predicted - goal).sum(axis=-1).reshape(3, 3)
            actual_cost = np.square(actual - goal).sum(axis=-1).reshape(3, 3)
            recorded_model, recorded_actual = np.asarray(row["model_cost"]), np.asarray(row["encoded_cost"])
            max_model_error = max(max_model_error, float(np.max(np.abs(model_cost - recorded_model))))
            max_actual_error = max(max_actual_error, float(np.max(np.abs(actual_cost - recorded_actual))))
            if max(max_model_error, max_actual_error) > 1e-3:
                raise ValueError("feature score mismatch")
            physical_cost = np.asarray([[p["physical"][x][y]["distance"][-1] for y in range(3)] for x in range(3)])
            np.testing.assert_array_equal(physical_cost, row["physical_distance"])
            a, b, c = map(source.difference, (recorded_model, recorded_actual, physical_cost))
            for key, left, right in (("model_vs_encoded", a, b), ("encoded_vs_physical", b, c),
                                     ("model_vs_physical", a, c)):
                if row[key] != source.physical.sign_report(left, right):
                    raise ValueError("sign metric mismatch")
            mse = np.square(predicted - actual).mean()
            if not np.isclose(mse, row["mean_feature_mse"], atol=1e-6, rtol=1e-6):
                raise ValueError("feature MSE mismatch")
    with OUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(source.RESULT), "roots": len(result["rows"]),
                   "model_cost_max_abs_error": max_model_error, "actual_cost_max_abs_error": max_actual_error,
                   "scope": "independent arithmetic from saved features; same model and physical backend"},
                  stream, indent=2, allow_nan=False)
    print(f"audited {len(result['rows'])} roots; max feature-cost errors "
          f"{max_model_error:.8g}, {max_actual_error:.8g}")


if __name__ == "__main__":
    main()
