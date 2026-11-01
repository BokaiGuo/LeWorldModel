"""Independently validate saved same-cp10 prediction vectors and rankings."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_cp10_paired_prediction_cuda_20261031 as run


def verify(path, digest):
    run.need(run.sha(path) == digest, "result SHA")
    result = run.feedback_plan.source.physical.h.read(path)
    run.need(result["schema"] == run.NAME and result["status"] == "complete"
             and result["host"]["device"] == "cuda:0"
             and result["model_before"] == result["model_after"] and len(result["rows"]) == 2,
             "result identity")
    run.need(all(run.sha(p) == h for p, h in result["sources"].items()), "source drift")
    rows = []
    for index, row in enumerate(result["rows"]):
        run.need(row["root_slot"] == index and set(row["arms"]) == {"open", "feedback"}, "fixed roster")
        goal = np.asarray(row["goal"], dtype=np.float64)
        run.need(goal.shape == (192,) and np.isfinite(goal).all(), "goal vector")
        for arm in row["arms"].values():
            pred, actual = (np.asarray(arm[k], dtype=np.float64) for k in ("prediction", "actual"))
            run.need(pred.shape == actual.shape == goal.shape and np.isfinite(pred).all()
                     and np.isfinite(actual).all(), "finite vectors")
            for key, a, b in (("predicted_goal_mse", pred, goal),
                              ("real_goal_mse", actual, goal),
                              ("prediction_real_mse", pred, actual)):
                value = float(np.mean(np.square(a - b)))
                run.need(np.isclose(value, arm[key], rtol=1e-6, atol=1e-7), "MSE arithmetic")
        left, right = row["arms"]["open"], row["arms"]["feedback"]
        rows.append({"root_slot": index,
                     "predicted_feedback_minus_open": right["predicted_goal_mse"] - left["predicted_goal_mse"],
                     "real_feedback_minus_open": right["real_goal_mse"] - left["real_goal_mse"],
                     "feedback_prediction_real_mse": right["prediction_real_mse"],
                     "open_prediction_real_mse": left["prediction_real_mse"]})
    run.need(run.sha(path) == digest, "result drift")
    return {"schema": "p1_cp10_paired_prediction_saved_20261031", "status": "saved_arithmetic_passed",
            "result_sha256": digest, "rows": rows, "P1_admission": False,
            "limitations": "Recomputes saved vectors only; two known roots, no independent inference."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    run.need(output.is_absolute() and output.parent == run.ROOT / "results" and
             output.name.startswith("p1_cp10_paired_prediction_saved_20261031_") and not output.exists(),
             "exclusive output")
    result = verify(args.input, args.sha256)
    print(json.dumps({"status": result["status"], "sha256": run.base.write(output, result), "rows": result["rows"]}))


if __name__ == "__main__":
    main()
