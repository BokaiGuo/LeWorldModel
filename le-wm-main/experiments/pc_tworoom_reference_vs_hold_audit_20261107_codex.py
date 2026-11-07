"""Cross-check sealed latent arithmetic and earlier independently saved replay."""
import json

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_reference_vs_hold_20261107_codex as source

OUTPUT = source.ROOT / "results/pc_tworoom_reference_vs_hold_audit_20261107_codex.json"


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    pred = json.loads(source.PRED.read_text())
    phys = json.loads(source.PHYSICAL.read_text())
    result = json.loads(source.RESULT.read_text())
    reference = json.loads(source.REFERENCE.read_text())
    if (result["prediction_sha256"] != file_hash(source.PRED)
            or result["physical_sha256"] != file_hash(source.PHYSICAL)
            or result["features_sha256"] != file_hash(source.FEATURES)
            or phys["frames_sha256"] != file_hash(source.FRAMES)
            or pred["features_sha256"] != file_hash(source.PRED_FEATURES)):
        raise ValueError("artifact hash mismatch")
    max_pred_error = max_encoded_error = max_distance_error = 0.
    with np.load(source.PRED_FEATURES, allow_pickle=False) as forecast, \
            np.load(source.FEATURES, allow_pickle=False) as features:
        for i, (row, old, replay) in enumerate(zip(result["rows"], reference["records"], phys["rows"], strict=True)):
            if row["row"] != old["row"] or row["row"] != replay["row"]:
                raise ValueError("reference row mismatch")
            goal, actual = features["goal"][i], features["actual"][i]
            expected_pred = np.square(forecast["predicted"][i] - goal).sum(-1)
            expected_true = np.square(actual - goal).sum(-1)
            max_pred_error = max(max_pred_error, float(np.max(np.abs(expected_pred - row["prediction"]))))
            max_encoded_error = max(max_encoded_error, float(np.max(np.abs(expected_true - row["encoded"]))))
            for a, b in zip(old["steps"], replay["candidates"][0]["trace"], strict=True):
                max_distance_error = max(max_distance_error, abs(float(np.linalg.norm(
                    np.asarray(a["actual_position"]) - np.asarray(old["goal_position"]))) - b["distance"]))
                if (a["terminated"], a["truncated"]) != (b["terminated"], b["truncated"]):
                    raise ValueError("reference termination mismatch")
    if max_pred_error > 1e-3 or max_encoded_error > 1e-3 or max_distance_error > 1e-4:
        raise ValueError("latent or physical mismatch")
    with OUTPUT.open("x") as stream:
        json.dump({"rows": len(result["rows"]), "max_prediction_cost_error": max_pred_error,
                   "max_encoded_cost_error": max_encoded_error, "max_reference_distance_error": max_distance_error,
                   "result_sha256": file_hash(source.RESULT), "reference_sha256": file_hash(source.REFERENCE),
                   "audit_sha256": file_hash(__file__),
                   "scope": "arithmetic and earlier reference trace cross-check, not independent simulator"},
                  stream, indent=2)
    print(f"audited {len(result['rows'])} rows; max errors: prediction={max_pred_error:.8g}, "
          f"encoding={max_encoded_error:.8g}, physical={max_distance_error:.8g}")


if __name__ == "__main__":
    run()
