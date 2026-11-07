"""Reconstruct local chart estimates from separately saved features and physical labels."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_goal_chart_actual_successor_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_tworoom_goal_chart_actual_successor_audit_20261107_codex.json"


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result, physical = json.loads(run.RESULT.read_text()), json.loads(run.PHYSICAL.read_text())
    if (result["physical_sha256"] != file_hash(run.PHYSICAL)
            or result["features_sha256"] != file_hash(run.FEATURES)
            or physical["frames_sha256"] != file_hash(run.FRAMES)
            or result["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("sealed artifacts changed")
    max_error = 0.
    valid = 0
    with np.load(run.FEATURES, allow_pickle=False) as features, \
            np.load(run.source.FEATURES, allow_pickle=False) as source, \
            np.load(run.source.old.previous.FEATURES, allow_pickle=False) as forecast:
        for row, truth in zip(result["rows"], physical["rows"], strict=True):
            if row["row"] != truth["row"] or row["valid"] != truth["basis_valid"]:
                raise ValueError("task or abstention mismatch")
            if not row["valid"]:
                continue
            valid += 1
            root = row["root"]
            chart = run.make_chart(source[f"root_{root}_probe_embedding"])
            np.testing.assert_allclose(chart, features[f"root_{root}_chart"], atol=1e-12)
            goal = forecast[f"root_{root}_goal"].astype(np.float64)
            real = features[f"root_{root}_actual"]
            predicted = forecast[f"root_{root}_predicted"]
            for i, (reported, actual) in enumerate(zip(row["outputs"], truth["outputs"], strict=True)):
                if reported["candidate_index"] != actual["candidate_index"] or reported["arm"] != actual["arm"]:
                    raise ValueError("candidate mismatch")
                true_delta = np.asarray(actual["true_displacement"])
                real_estimate = chart @ (real[i].astype(np.float64) - goal)
                model_estimate = chart @ (predicted[actual["candidate_index"]].astype(np.float64) - goal)
                measurements = (("chart_true_encoded", real_estimate), ("chart_model_prediction", model_estimate),
                                ("zero_error", np.linalg.norm(true_delta)),
                                ("true_encoded_error", np.linalg.norm(real_estimate - true_delta)),
                                ("predicted_error", np.linalg.norm(model_estimate - true_delta)))
                for key, expected in measurements:
                    max_error = max(max_error, float(np.max(np.abs(np.asarray(reported[key]) - expected))))
                if abs(np.linalg.norm(true_delta) - actual["final_distance"]) > 1e-4:
                    raise ValueError("distance label mismatch")
    if max_error > 1e-9 or valid != 8:
        raise ValueError("audit tolerance or valid count mismatch")
    with OUTPUT.open("x") as stream:
        json.dump({"valid_roots": valid, "abstained_roots": len(result["rows"]) - valid,
                   "max_error": max_error, "result_sha256": file_hash(run.RESULT),
                   "physical_sha256": file_hash(run.PHYSICAL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "saved features and labels; same encoder and simulator"}, stream, indent=2)
    print(f"audited {valid} valid roots; maximum arithmetic error {max_error}")


if __name__ == "__main__":
    audit()
