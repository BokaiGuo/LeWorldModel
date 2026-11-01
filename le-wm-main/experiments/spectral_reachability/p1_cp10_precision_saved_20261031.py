"""Recompute fixed-action precision scores from saved vectors."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_cp10_precision_cuda_20261031 as experiment

RESULT = experiment.ROOT / "results/p1_cp10_precision_cuda_20261031_RUN01.json"
RESULT_SHA = "18a1fc81c9cb797b70e5aa0d227ee6e3b8b70a4af11556963b6ae9bce9cfd3aa"


def verify(result, reference):
    if result["schema"] != experiment.NAME or result["status"] != "complete" or \
            result["P1_admission"] is not False or result["independent_confirmation"] is not False:
        raise ValueError("result contract")
    if len(result["rows"]) != 2 or len(reference["rows"]) != 2:
        raise ValueError("roster")
    rows = []
    for row, prior in zip(result["rows"], reference["rows"], strict=True):
        if row["root_slot"] != prior["root_slot"]:
            raise ValueError("root identity")
        goal = np.asarray(prior["goal"], dtype=np.float64)
        if goal.shape != (192,) or not np.isfinite(goal).all():
            raise ValueError("goal")
        margins = {}
        for mode in ("fp32", "bf16"):
            scores = {}
            for arm in ("open", "feedback"):
                saved = row["modes"][mode][arm]
                old = prior["arms"][arm]
                pred = np.asarray(saved["prediction"], dtype=np.float64)
                actual = np.asarray(old["actual"], dtype=np.float64)
                if pred.shape != (192,) or actual.shape != (192,) or \
                        not np.isfinite(pred).all() or not np.isfinite(actual).all() or \
                        saved["action_sha256"] != old["action_sha256"]:
                    raise ValueError("vector/action")
                score = float(np.mean((pred - goal) ** 2))
                error = float(np.mean((pred - actual) ** 2))
                if not np.isclose(score, saved["goal_mse"], rtol=1e-5, atol=1e-7) or \
                        not np.isclose(error, saved["actual_mse"], rtol=1e-5, atol=1e-7):
                    raise ValueError("metric")
                if mode == "fp32" and not np.allclose(pred, old["prediction"], rtol=1e-6, atol=1e-6):
                    raise ValueError("reference prediction")
                scores[arm] = score
            margins[mode] = scores["feedback"] - scores["open"]
        real = {}
        for arm in ("open", "feedback"):
            old = prior["arms"][arm]
            actual = np.asarray(old["actual"], dtype=np.float64)
            real[arm] = float(np.mean((actual - goal) ** 2))
            if not np.isclose(real[arm], row["real_goal_mse"][arm], rtol=1e-5, atol=1e-7):
                raise ValueError("real score")
        rows.append({"root_slot": row["root_slot"], **margins,
                     "real": real["feedback"] - real["open"]})
    if {r["root_slot"] for r in rows} != {0, 1}:
        raise ValueError("root roster")
    return rows


def main():
    for path, expected in ((RESULT, RESULT_SHA), (experiment.REFERENCE, experiment.REFERENCE_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("source SHA")
    result = json.loads(RESULT.read_text())
    reference = json.loads(experiment.REFERENCE.read_text())
    print(json.dumps({"status": "verified", "rows": verify(result, reference)}))


if __name__ == "__main__":
    main()
