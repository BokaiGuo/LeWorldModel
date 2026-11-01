"""Independently recompute sealed packet fidelity and paired rankings."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_packet_objective_fidelity_20261031 as source

NAME = "p1_packet_objective_fidelity_analysis_20261031"
INPUT = source.ROOT / "results/p1_packet_objective_fidelity_20261031_RUN01.json"
INPUT_SHA = "284437414d5b40c5337e5685a5bfd814fc831755b3f5f9e961b9d3b4b68fc2f9"


def analyze(value):
    rows = value["rows"]
    if value["status"] != "complete" or len(rows) != 24:
        raise ValueError("roster")
    for i, row in enumerate(rows):
        if (row["index"] != i or row["logical_pair"] != i // 3
                or row["role"] != ("selected", "zero", "greedy")[i % 3]
                or len(row["packets"]) != 2):
            raise ValueError("identity")
        goal = np.asarray(row["goal_latent"], dtype=np.float64)
        if goal.shape != (192,) or not np.isfinite(goal).all():
            raise ValueError("goal")
        for j, packet in enumerate(row["packets"]):
            predicted = np.asarray(packet["predicted_latent"], dtype=np.float64)
            actual = np.asarray(packet["actual_latent"], dtype=np.float64)
            if (packet["step"] != (j + 1) * 5 or predicted.shape != goal.shape
                    or actual.shape != goal.shape or not np.isfinite(predicted).all()
                    or not np.isfinite(actual).all()):
                raise ValueError("packet")
            for key, number in (
                    ("predicted_goal_mse", np.mean((predicted-goal)**2)),
                    ("actual_goal_mse", np.mean((actual-goal)**2)),
                    ("predicted_actual_mse", np.mean((predicted-actual)**2))):
                if not np.isfinite(packet[key]) or abs(packet[key] - number) > 1e-6:
                    raise ValueError("numeric replay")
    pairs = []
    for pair in range(8):
        chosen, zero = rows[3 * pair:3 * pair + 2]
        comparisons = []
        for j in range(2):
            a, b = chosen["packets"][j], zero["packets"][j]
            d_model = a["predicted_goal_mse"] - b["predicted_goal_mse"]
            d_rgb = a["actual_goal_mse"] - b["actual_goal_mse"]
            d_position = a["physical"]["object_position_error"] - b["physical"]["object_position_error"]
            comparisons.append({"step": 5 * (j + 1), "model_delta": d_model,
                                "rgb_delta": d_rgb, "position_delta": d_position,
                                "model_rgb_reversal": d_model * d_rgb < 0,
                                "rgb_position_reversal": d_rgb * d_position < 0})
        pairs.append({"logical_pair": pair, "comparisons": comparisons})
    return {"schema": NAME, "status": "recomputed", "pairs": pairs,
            "mean_prediction_mse": [float(np.mean([row["packets"][j]["predicted_actual_mse"]
                                                    for row in rows])) for j in range(2)],
            "P1_admission": False, "independent_confirmation": False,
            "limitations": "Post-outcome descriptive comparisons on two known dependent roots; "
                           "not a physical mechanism intervention or independent test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    source.physical.need(out.is_absolute() and out.parent == source.ROOT / "results"
                         and out.name.startswith(NAME + "_") and not out.exists()
                         and not out.is_symlink(), "exclusive output")
    source.physical.need(source.base.sha(INPUT) == INPUT_SHA, "input hash")
    value = analyze(json.loads(INPUT.read_bytes()))
    source.physical.need(source.base.sha(INPUT) == INPUT_SHA, "input drift")
    value["input_sha256"] = INPUT_SHA
    value["code_sha256"] = source.base.sha(__file__)
    digest = source.base.write(out, value)
    print(json.dumps({"pairs": len(value["pairs"]), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
