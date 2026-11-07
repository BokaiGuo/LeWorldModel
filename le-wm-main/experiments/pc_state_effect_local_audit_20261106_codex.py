"""Recompute local descriptor results from sealed feature and physical artifacts."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments.pc_state_effect_local_dev_20261106_codex import PREFIX, BANK, ROOT


def main():
    output = PREFIX.with_name(PREFIX.name + "_audit.json")
    if output.exists():
        raise FileExistsError(output)
    source = PREFIX.with_suffix(".json")
    record = json.loads(source.read_text())
    features_path = PREFIX.with_suffix(".npz")
    if record["features_sha256"] != file_hash(features_path):
        raise ValueError("feature drift")
    for path, expected in record["dependencies"].items():
        if file_hash(path) != expected:
            raise ValueError(f"dependency drift: {path}")
    physical_path = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
    if record["task_manifest_hash"] != file_hash(physical_path):
        raise ValueError("physical input drift")
    conditions = []
    with np.load(features_path, allow_pickle=False) as features, np.load(physical_path, allow_pickle=False) as physical:
        for row in record["rows"]:
            index, camera = row["root"] - 1880, row["camera"]
            key = f"r{index}_v{camera}"
            goal = features[key + "_goal"]
            predicted = [float(np.square(features[key + f"_pred_{x}"] - goal).mean()) for x in BANK]
            np.testing.assert_allclose(predicted, row["heldout_model_costs"], atol=1e-7, rtol=1e-5)
            real = [float(np.square(features[key + f"_real_{i}"] - goal).mean()) for i in range(3)]
            np.testing.assert_allclose(real, row["real_encoded_costs_zero_plus_minus"], atol=1e-7, rtol=1e-5)
            for name, values in row["predictions"].items():
                mae = float(np.abs(np.asarray(values) - row["heldout_model_costs"]).mean())
                np.testing.assert_allclose(mae, row["model_mae"][name], atol=1e-12, rtol=0)
            target = physical["traces"][2*index+1, 0, camera, 0, :2]
            distances = np.linalg.norm(physical["traces"][2*index, :, camera, -1, :2] - target, axis=-1)
            np.testing.assert_allclose(distances, row["physical_distances_zero_plus_minus"], atol=1e-12, rtol=0)
            model_costs = np.asarray(row["predicted_costs_zero_plus_minus"])
            encoded_costs = np.asarray(real)
            regrets = {
                "model": float(distances[int(np.argmin(model_costs))] - distances.min()),
                "encoding": float(distances[int(np.argmin(encoded_costs))] - distances.min()),
                "zero": float(distances[0] - distances.min()),
            }
            for name, value in regrets.items():
                np.testing.assert_allclose(value, row[name + "_physical_regret"], atol=1e-12, rtol=0)
            mse = [float(np.square(features[key + f"_pred_{x}"] - features[key + f"_real_{i}"]).mean())
                   for x, i in ((.1, 1), (-.1, 2))]
            conditions.append({"root": row["root"], "camera": camera, "regret": regrets,
                               "selected": {"model": int(np.argmin(model_costs)),
                                            "encoding": int(np.argmin(encoded_costs)),
                                            "physical": int(np.argmin(distances))},
                               "predicted_real_feature_mse_plus_minus": mse,
                               "signed_effect": {"model": (model_costs[0] - model_costs[1:]).tolist(),
                                                 "encoding": (encoded_costs[0] - encoded_costs[1:]).tolist(),
                                                 "physical": (distances[0] - distances[1:]).tolist()}})
    means = {name: float(np.mean([row["regret"][name] for row in conditions]))
             for name in ("model", "encoding", "zero")}
    roots = []
    for root in range(1880, 1884):
        pair = [c for c in conditions if c["root"] == root]
        roots.append({"root": root, "camera_changes_model_choice":
                      pair[0]["selected"]["model"] != pair[1]["selected"]["model"],
                      "camera_changes_encoded_choice":
                      pair[0]["selected"]["encoding"] != pair[1]["selected"]["encoding"]})
    result = {"source_sha256": file_hash(source), "code_sha256": file_hash(__file__),
              "conditions": conditions, "roots": roots, "mean_physical_regret": means,
              "tie_aware_physical_best": {name: sum(c["regret"][name] <= 1e-9 for c in conditions)
                                         for name in means},
              "numerical_check_pass": record["numerical_check_pass"],
              "mean_root_mae": record["mean_root_mae"],
              "local_prediction_dev_exit": record["local_prediction_dev_exit"],
              "camera_changes_model_choice_roots": sum(r["camera_changes_model_choice"] for r in roots),
              "camera_changes_encoded_choice_roots": sum(r["camera_changes_encoded_choice"] for r in roots)}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("conditions", "roots")}, indent=2))


if __name__ == "__main__":
    main()
