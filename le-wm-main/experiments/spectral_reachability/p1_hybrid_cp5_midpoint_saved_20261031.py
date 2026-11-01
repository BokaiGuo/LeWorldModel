"""Independently recompute fixed-action midpoint prediction MSEs from vectors."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_hybrid_cp5_midpoint_cuda_20261031 as run


def verify(path, expected):
    run.need(run.sha(path) == expected, "external result SHA")
    result = run.physical.h.read(path)
    run.need(result["schema"] == run.NAME and result["status"] == "complete"
             and len(result["rows"]) == 2 and result["model_before"] == result["model_after"]
             and result["host"]["device"] == "cuda:0", "result identity")
    run.need(all(run.sha(p) == digest for p, digest in result["sources"].items()), "source drift")
    out = []
    for i, row in enumerate(result["rows"]):
        run.need(row["root_slot"] == i and row["status"] == "complete", "fixed root order")
        vectors = {k: np.asarray(v, dtype=np.float64) for k, v in row["vectors"].items()}
        run.need(set(vectors) == {"midpoint_predicted", "midpoint_actual", "terminal_open",
                                  "terminal_refreshed", "terminal_actual", "goal"}
                 and all(v.shape == (192,) and np.isfinite(v).all() for v in vectors.values()),
                 "finite vectors")
        pairs = {
            "prediction_goal_mse": ("terminal_open", "goal"),
            "real_rgb_goal_mse": ("terminal_actual", "goal"),
            "prediction_actual_mse": ("terminal_open", "terminal_actual"),
            "midpoint_prediction_actual_mse": ("midpoint_predicted", "midpoint_actual"),
            "refreshed_prediction_actual_mse": ("terminal_refreshed", "terminal_actual"),
            "refreshed_prediction_goal_mse": ("terminal_refreshed", "goal"),
        }
        for key, (a, b) in pairs.items():
            measured = float(np.mean(np.square(vectors[a] - vectors[b])))
            run.need(np.isclose(measured, row[key], rtol=1e-6, atol=1e-7), "MSE recomputation")
        out.append({k: row[k] for k in ("root_slot", "prediction_actual_mse",
                                        "refreshed_prediction_actual_mse")})
    run.need(run.sha(path) == expected, "result drift")
    return {"schema": "p1_hybrid_cp5_midpoint_saved_20261031", "status": "saved_arithmetic_passed",
            "result_sha256": expected, "rows": out, "P1_admission": False,
            "limitations": "Saved vectors only; not an independent inference or physical execution."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    run.need(output.is_absolute() and output.parent == run.ROOT / "results" and
             output.name.startswith("p1_hybrid_cp5_midpoint_saved_20261031_") and not output.exists(),
             "exclusive output")
    result = verify(args.input, args.sha256)
    print(json.dumps({"status": result["status"], "sha256": run.base.write(output, result)}))


if __name__ == "__main__":
    main()
