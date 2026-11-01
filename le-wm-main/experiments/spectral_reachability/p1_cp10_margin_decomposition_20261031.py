"""Decompose a sealed fixed-action prediction ranking into exact error terms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NAME = "p1_cp10_margin_decomposition_20261031"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/p1_cp10_paired_prediction_cuda_20261031_RUN01.json"
SOURCE_SHA = "6dd3ee6f86aad55a963c065982f65d898f647ea287bebe6c659f3d198fe0a3cf"


def decompose(goal, predictions, actuals):
    """Return exact mean squared distance gap and its two residual terms."""
    g = np.asarray(goal, dtype=np.float64)
    if g.shape != (192,) or not np.isfinite(g).all():
        raise ValueError("invalid goal")
    terms = {}
    for arm in ("open", "feedback"):
        p = np.asarray(predictions[arm], dtype=np.float64)
        y = np.asarray(actuals[arm], dtype=np.float64)
        if p.shape != g.shape or y.shape != g.shape or not (
                np.isfinite(p).all() and np.isfinite(y).all()):
            raise ValueError("invalid arm")
        residual, error = y - g, p - y
        terms[arm] = {
            "prediction_goal_mse": float(np.mean((p - g) ** 2)),
            "actual_goal_mse": float(np.mean(residual ** 2)),
            "prediction_error_mse": float(np.mean(error ** 2)),
            "cross_term": float(np.mean(2 * residual * error)),
        }
        if not np.isclose(terms[arm]["prediction_goal_mse"] -
                          terms[arm]["actual_goal_mse"],
                          terms[arm]["cross_term"] + terms[arm]["prediction_error_mse"],
                          rtol=1e-10, atol=1e-12):
            raise ValueError("arm identity")
    margin = lambda key: terms["feedback"][key] - terms["open"][key]
    result = {"arms": terms, "predicted_margin": margin("prediction_goal_mse"),
              "actual_margin": margin("actual_goal_mse"),
              "cross_term_difference": margin("cross_term"),
              "error_mse_difference": margin("prediction_error_mse")}
    result["margin_error"] = result["predicted_margin"] - result["actual_margin"]
    if not np.isclose(result["margin_error"], result["cross_term_difference"] +
                      result["error_mse_difference"], rtol=1e-10, atol=1e-12):
        raise ValueError("margin identity")
    errors = [np.asarray(predictions[a], dtype=np.float64) -
              np.asarray(actuals[a], dtype=np.float64) for a in ("open", "feedback")]
    result["action_error_difference_mse"] = float(np.mean((errors[1] - errors[0]) ** 2))
    result["error_cosine"] = float(np.dot(*errors) / (np.linalg.norm(errors[0]) *
                                                     np.linalg.norm(errors[1]))) if all(
        np.linalg.norm(e) > 0 for e in errors) else None
    return result


def build(source):
    if source["schema"] != "p1_cp10_paired_prediction_cuda_20261031" or \
            source["status"] != "complete" or len(source["rows"]) != 2:
        raise ValueError("source contract")
    rows = []
    for row in source["rows"]:
        arms = row["arms"]
        result = decompose(row["goal"], {a: arms[a]["prediction"] for a in arms},
                           {a: arms[a]["actual"] for a in arms})
        for a in ("open", "feedback"):
            for name, original in (("prediction_goal_mse", "predicted_goal_mse"),
                                   ("actual_goal_mse", "real_goal_mse"),
                                   ("prediction_error_mse", "prediction_real_mse")):
                if not np.isclose(result["arms"][a][name], arms[a][original],
                                  rtol=1e-5, atol=1e-7):
                    raise ValueError("saved score does not match vectors")
        rows.append({"root_slot": row["root_slot"], **result})
    if {row["root_slot"] for row in rows} != {0, 1}:
        raise ValueError("root roster")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    if not out.is_absolute() or out.parent != ROOT / "results" or \
            not out.name.startswith(NAME + "_") or out.suffix != ".json":
        raise ValueError("output path")
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("source SHA")
    rows = build(json.loads(raw))
    payload = {"schema": NAME, "source_sha256": SOURCE_SHA, "rows": rows,
               "independent_confirmation": False, "P1_admission": False,
               "limitations": "Two known roots; fixed actions, post-outcome algebra, no new model or physics."}
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    with out.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output": str(out), "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main()
