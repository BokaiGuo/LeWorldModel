"""Recompute local tangent metric and pair selections with saved real outcomes."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_current_rgb_tangent_bank_20261102 as source
from . import p1_dev305_two_packet_metric_decompose_20261102 as previous

base, need, ROOT = source.base, source.need, source.ROOT
PREP_SHA = "41f7fcbe66e4277d61d683ddf9a5de4ac7d67ffea646324835ff4e8e5ea9d298"
REPORT_SHA = "b91e36eee625e7bb6ef7d923d6107b9260dc5dc2170ffde06c9befef9d673016"


def analyze():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    phys = base.read(ROOT / "results" / f"{previous.actual.physical.source.NAME}_RUN_DEV01/report.json",
                     previous.actual.physical.saved.REPORT_SHA)
    capture = base.read(ROOT / "results" / "p1_dev305_diverse_cem_20261102_CAPTURE_DEV01/report.json",
                        previous.CAPTURE_SHA)
    baseline = base.read(ROOT / "results" / "p1_dev305_two_packet_metric_decompose_20261102_DEV01.json",
                         "b7a40a07e7e423c651e31a31d258318dbcf53210763ce6dede49c7b1e4f5a561")
    need(spec == source.protocol() and report["contract_sha256"] == PREP_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"], "source")
    goal = np.asarray(capture["goal_state_evaluation_only"], dtype=float)
    rows = []
    for role, entry in zip(spec["roles"], report["rows"], strict=True):
        need(entry["arm"] == role["arm"], "role")
        q, stats = source.tangent.construct(entry["jacobian"])
        need(np.allclose(q, entry["q_sqrt"], atol=1e-8, rtol=1e-8)
             and np.allclose(stats["singular_values"], entry["metric"]["singular_values"], atol=1e-10),
             "metric reconstruction")
        scores = np.asarray(entry["structured_scores"], dtype=float)
        need(scores.shape == (17, 17) and np.isfinite(scores).all()
             and list(map(int, np.unravel_index(np.argmin(scores), scores.shape))) == entry["selected_pair"],
             "argmin")
        i, j = entry["selected_pair"]
        physical = next(x for x in phys["rows"] if x["prefix"] == role["arm"] and x["first"] == i and x["second"] == j)
        need(physical["status"] == "fixed_horizon_completed", "physical")
        state = np.asarray(physical["transitions"][-1]["state"], dtype=float)
        angle = abs(float((state[4] - goal[4] + np.pi) % (2*np.pi) - np.pi))
        old = next(x for x in baseline["rows"] if x["arm"] == role["arm"])["selected"]["predicted_latent"]
        rows.append({"arm": role["arm"], "structured_pair": [i, j],
                     "structured_reward": physical["transitions"][-1]["reward"],
                     "structured_block_position_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
                     "structured_block_angle_error": angle, "baseline_pair": old["pair"],
                     "baseline_reward": old["physical_reward"], "baseline_block_position_error": old["block_position_error"],
                     "baseline_block_angle_error": old["block_angle_error"],
                     "baseline_prediction_max_difference": entry["baseline_max_abs_difference"]})
    return {"schema": "p1_dev305_current_rgb_tangent_saved_20261102.v1", "rows": rows,
            "scope": "post-outcome known-root; no new physical execution, no independent confirmation",
            "p1_admission": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
