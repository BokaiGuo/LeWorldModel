"""Descriptive paired budget comparisons; no treatment of rows as independent roots."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_adam_checkpoint_physical_pair_20261031 as pair

ROOT = pair.ROOT
SOURCE = ROOT / "results/p1_adam_terminal_rgb_cuda_20261031_RUN01.json"
SOURCE_SHA = "b7e64fbdb226405ad09f1086091829ba87ca365627debe7749da60ca774c2540"
NAME = "p1_adam_terminal_rgb_analysis_20261031"


def analyze(data):
    pair.pairing.need(data["status"] == "complete" and len(data["rows"]) == 64
                      and len(data["stopped"]) == 8, "panel")
    lookup = {(r["index"], r["step"]): r for r in data["rows"]}
    pair.pairing.need(len(lookup) == 64, "unique rows")
    for row in data["rows"]:
        p = np.asarray(row["predicted_latent"], dtype=np.float64)
        a = np.asarray(row["actual_latent"], dtype=np.float64)
        pair.pairing.need(p.shape == a.shape and p.size == 192
                          and abs(float(np.mean((p-a)**2))-row["prediction_rgb_mse"]) < 1e-6,
                          "latent mse reproduction")
    metrics = ["predicted_goal_mse", "actual_rgb_goal_mse",
               "object_position_error", "object_angle_error_wrapped", "pusher_position_error",
               "native_joint_xy_error"]
    output = []
    for low, high in ((5, 10), (10, 20), (5, 20)):
        for i in range(24):
            if (i, low) not in lookup or (i, high) not in lookup:
                continue
            left, right = lookup[i, low], lookup[i, high]
            pair.pairing.need(left["role"] == right["role"] and left["logical_pair"] == right["logical_pair"],
                              "pair identity")
            delta = {}
            for key in metrics:
                def value(row):
                    return row[key] if key in row else row["physical_endpoint"][key]
                delta[key] = value(right) - value(left)
            output.append({"index": i, "role": left["role"], "logical_pair": left["logical_pair"],
                           "steps": [low, high], "delta": delta})
    summary = {}
    for low, high in ((5, 10), (10, 20), (5, 20)):
        subset = [r for r in output if r["steps"] == [low, high]]
        def opposite(a, b):
            return sum(r["delta"][a] * r["delta"][b] < 0 for r in subset)
        summary[f"{low}-{high}"] = {
            "pairs": len(subset),
            "predicted_vs_rgb_opposite": opposite("predicted_goal_mse", "actual_rgb_goal_mse"),
            "rgb_vs_object_position_opposite": opposite("actual_rgb_goal_mse", "object_position_error"),
            "rgb_vs_object_angle_opposite": opposite("actual_rgb_goal_mse", "object_angle_error_wrapped"),
            "rgb_vs_joint_xy_opposite": opposite("actual_rgb_goal_mse", "native_joint_xy_error"),
        }
    return {"schema": NAME, "summary": summary, "pairs": output,
            "limitations": "Post-hoc descriptive comparisons of repeated known roots; "
                           "not independent samples, causal evidence, or P1 admission.",
            "P1_admission": False, "source_sha256": SOURCE_SHA}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    pair.pairing.need(output.is_absolute() and output.parent == ROOT / "results"
                      and output.name.startswith(NAME + "_") and not output.exists()
                      and not output.is_symlink(), "exclusive output")
    data = pair.pairing.read(SOURCE, SOURCE_SHA)
    result = analyze(data)
    result["analysis_sha256"] = pair.physical.sha(Path(__file__).resolve())
    pair.pairing.need(pair.physical.sha(SOURCE) == SOURCE_SHA, "source drift")
    digest = pair.physical.prior.physical.design.run._write_new(output, result)
    print(json.dumps({"summary": result["summary"], "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
