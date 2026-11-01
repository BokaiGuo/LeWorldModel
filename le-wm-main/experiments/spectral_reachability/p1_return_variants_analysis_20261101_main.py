"""All-arm descriptive return-task comparison; no inference from native7 alone."""
import argparse
import hashlib
import json
from pathlib import Path

from . import e3_initial_goal_qualification_20261031 as qualification
from .p1_common_h10_analysis_20261101_main import metrics, delta

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("structured", "native", "random", "immediate_progress")
VARIANTS = {
    "euclidean": "p1_return_goal_shard_20261101_main_root0",
    "quadratic_winner": "p1_q_return_shard_20261101_main",
    "quadratic_equal": "p1_equal_candidate_q_20261101_main",
    "quadratic_horizon": "p1_horizon_q_20261101_main",
}


def analyze():
    sources, rows = {}, []

    def read(path):
        blob = path.read_bytes()
        sources[str(path)] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    target = read(ROOT / "results/p1_return_target_cuda_20261101_main_MAIN06.json")
    goal, initial = target["rows"][0]["goal_state"], target["rows"][0]["root_state"]
    for variant, prefix in VARIANTS.items():
        for arm in ARMS:
            directory = ROOT / "results" / f"{prefix}_{arm}_RUN_MAIN01"
            report = read(directory / "report.json")
            if report["status"] != "recorded_known_root_not_admitted" or len(report["rows"]) != 1:
                raise ValueError("incomplete or wrong roster")
            row = report["rows"][0]
            if row["arm"] != arm:
                raise ValueError("arm mismatch")
            points = []
            for cp in row["checkpoints"]:
                if cp["goal"] != goal:
                    raise ValueError("different task")
                physical = qualification.native(goal, cp["state"])
                points.append(dict(step=cp["step"], **metrics(cp["state"], goal),
                                   joint_xy=physical["joint_xy_error"],
                                   native7=physical["native7_distance"],
                                   native_success=physical["initial_success"]))
            cp0 = next(c for c in row["checkpoints"] if c["step"] == 0)
            if cp0["state"] != initial:
                raise ValueError("different initial state")
            first = next((p for p in points if p["step"] == 5), None)
            origin = next(p for p in points if p["step"] == 0)
            keys = ("pusher_xy", "object_xy", "object_angle", "joint_xy", "native7")
            immediate = delta({k: first[k] for k in keys}, {k: origin[k] for k in keys}) if first else None
            totals, decisions = {}, []
            for node in report["nodes"][row["episode_id"]]:
                if node["status"] == "not_started":
                    continue
                if node["status"] != "sealed":
                    raise ValueError("unsealed executed node")
                receipt = node["receipt"]
                for key, value in receipt["account"]["charged_upper_bounds"].items():
                    totals[key] = totals.get(key, 0) + value
                decisions.append(dict(boundary=node["boundary"],
                                      selected_index=receipt["selected_index"]))
            rows.append(dict(variant=variant, arm=arm, status=row["status"],
                             success=row["native_success"], executed_steps=row["control_returned"],
                             error=row["first_error"], prefix_delta=immediate, points=points,
                             charged_upper_bounds=totals, decisions=decisions,
                             endpoint20=next((p for p in points if p["step"] == 20), None)))
    return dict(rows=rows, sources=sources, eligible_roots=1, development_only=True,
                P1_admission=False, independent_confirmation=False,
                identical_total_compute=False,
                ledger_scope="planner nodes only; RGB encoding, physical operations and metric construction separate",
                unamortized_q_construction=dict(encoder_forwards=13, input_vjp_rows=192),
                caution="JointXY improvement cannot be relabeled non-greedy using velocity-sensitive native7.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=len(result["rows"]), output=str(args.output))))
