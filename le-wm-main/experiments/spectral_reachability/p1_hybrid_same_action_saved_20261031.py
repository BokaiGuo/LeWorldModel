"""Independent saved-vector arithmetic check; no neural or physics calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(path, expected):
    if sha(path) != expected:
        raise ValueError("source report hash mismatch")
    data = json.loads(Path(path).read_bytes())
    if (data["status"] != "completed_known_root_diagnostic" or len(data["rows"]) != 4
            or data["model_before"] != data["model_after"] or data["P1_admission"] is not False
            or data["independent_confirmation"] is not False):
        raise ValueError("source contract mismatch")
    goal = np.asarray(data["goal"], np.float64)
    if goal.shape != (192,) or not np.isfinite(goal).all():
        raise ValueError("goal")
    for i, row in enumerate(data["rows"]):
        if (row["start"], row["end"]) != (5*i, 5*(i+1)):
            raise ValueError("timing")
        arrays = [np.asarray(row[k], np.float64)
                  for k in ("open_latent", "feedback_latent", "actual_latent")]
        if any(x.shape != (192,) or not np.isfinite(x).all() for x in arrays):
            raise ValueError("latent")
        op, fb, actual = arrays
        expected_metrics = {"open_rgb_mse": np.mean((op-actual)**2),
                            "feedback_rgb_mse": np.mean((fb-actual)**2),
                            "open_goal_mse": np.mean((op-goal)**2),
                            "feedback_goal_mse": np.mean((fb-goal)**2),
                            "actual_goal_mse": np.mean((actual-goal)**2)}
        if any(not np.isclose(row[k], v, atol=1e-6, rtol=1e-6) for k, v in expected_metrics.items()):
            raise ValueError("saved metric mismatch")
    for name, digest in data["sources"].items():
        if sha(name) != digest:
            raise ValueError("source drift")
    return {"status": "saved_arithmetic_passed", "rows": 4, "report_sha256": expected,
            "reran_neural_or_physics": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.report, args.sha256), sort_keys=True))
