"""Recompute real-vs-predicted latent risk decomposition from saved arrays."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_third_boundary_risk_decompose_20261102 as source

base, need, ROOT = source.base, source.need, source.ROOT
NAME = "p1_dev305_third_risk_decompose_saved_20261102"
PREP_SHA = "005d76073c9ce24881e6942afa7c1ce61d56367a359543e0b56112222a5c4af5"
REPORT_SHA = "0f8e3547a6e791a190a24289ad2367d0626327c17b3ce0ae9d337bcf9e40d962"


def analyze():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    need(spec == source.protocol() and report["contract_sha256"] == PREP_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"]
         and report["counts"] == {"encoder": {"calls": 7, "samples": 7},
                                  "projector": {"calls": 7, "samples": 7}}, "sources")
    rows = []
    for item, result in zip(spec["rows"], report["rows"], strict=True):
        need(item["arm"] == result["arm"] and item["candidate"] == result["candidate"], "binding")
        real = np.asarray(result["real_goal_delta"], dtype=np.float64)
        pred = np.asarray(item["model_goal_delta"], dtype=np.float64)
        need(real.shape == pred.shape == (192,) and np.isfinite(real).all()
             and np.isclose(np.mean((pred-real)**2), result["prediction_error_mse"], rtol=1e-12), "latent")
        project = source.physical.risk.engine.risk.predict_screen
        true_projection = project(item["jacobian"], np.repeat(real[None], 4, axis=0),
                                  item["current_goal_delta"], item["source_boxes"])[0]
        model_projection = project(item["jacobian"], np.repeat(pred[None], 4, axis=0),
                                   item["current_goal_delta"], item["source_boxes"])[0]
        need(true_projection == result["real_latent_projection"]
             and model_projection == item["model_projection"], "projection")
        screen = source.physical.risk.engine.risk.visibility.inspect(
            source.run.coherent.base.frame(item["terminal"]))
        boxes = {name: screen[name]["bbox"] for name in ("object", "pusher")}
        need(boxes == result["actual_boxes"] and screen["eligible"] == result["actual_visible"], "real frame")
        rows.append({"arm": item["arm"], "candidate": item["candidate"], "selected": item["selected"],
                     "model_projection_visible": model_projection["predicted_visible"],
                     "real_latent_projection_visible": true_projection["predicted_visible"],
                     "actual_visible": result["actual_visible"],
                     "model_object_box": model_projection["estimated_boxes"]["object"],
                     "real_latent_object_box": true_projection["estimated_boxes"]["object"],
                     "actual_object_box": boxes["object"], "prediction_error_mse": result["prediction_error_mse"]})
    return {"schema": NAME + ".v1", "rows": rows,
            "scope": "same known-root actions; true RGB encoding is post-outcome diagnostic, not planner input",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
