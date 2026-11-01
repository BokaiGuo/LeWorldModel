"""Recompute the saved diagnostic's Q/MSE without neural inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_reverse_control_20261101_main as control
from . import p1_reverse_target_cuda_20261101_main as target


def check(path):
    doc = json.loads(Path(path).read_bytes())
    if doc["counts"] != dict(encoder_calls=7, predict_calls=12, action_encode_calls=12):
        raise ValueError("fixed diagnostic count")
    if doc["model_before"] != doc["model_after"] or not doc["supplied_privileged_witness"]:
        raise ValueError("diagnostic identity")
    if any(target.encoder.base.sha(p) != h for p, h in doc["sources"].items()):
        raise ValueError("source drift")
    metric = json.loads(control.METRIC.read_bytes())
    goal = np.asarray(metric["goal_latent"], np.float32)
    matrix = np.asarray(metric["sqrt_matrix"], np.float64)
    expected = [(mode, step) for mode in ("witness_open", "witness_refreshed", "zero_open")
                for step in (5, 10, 15, 20)]
    if [(r["mode"], r["step"]) for r in doc["rows"]] != expected:
        raise ValueError("fixed twelve-row roster")
    for row in doc["rows"]:
        predicted = np.asarray(row["predicted"], np.float32)
        if predicted.shape != (192,) or not np.isfinite(predicted).all():
            raise ValueError("prediction vector")
        q = float(np.mean(((predicted - goal).astype(np.float64) @ matrix.T) ** 2))
        if not np.isclose(q, row["predicted_goal_q"], atol=1e-10, rtol=1e-8):
            raise ValueError("predicted Q mismatch")
        if row["mode"] == "zero_open":
            if "observed" in row:
                raise ValueError("fabricated zero counterfactual")
            continue
        observed = np.asarray(row["observed"], np.float32)
        mse = float(np.mean((predicted - observed) ** 2))
        q = float(np.mean(((observed - goal).astype(np.float64) @ matrix.T) ** 2))
        if observed.shape != (192,) or not np.isfinite(observed).all():
            raise ValueError("observed vector")
        if not np.isclose(mse, row["prediction_mse"], atol=1e-7, rtol=1e-6):
            raise ValueError("MSE mismatch")
        if not np.isclose(q, row["observed_goal_q"], atol=1e-10, rtol=1e-8):
            raise ValueError("observed Q mismatch")
    for i in range(4):
        if doc["rows"][i]["observed"] != doc["rows"][i+4]["observed"]:
            raise ValueError("same-time actual frame")
    return dict(status="saved_numerics_verified", rows=12, source_sha256=target.encoder.base.sha(path),
                P1_admission=False, not_new_physical_evidence=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.input)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
