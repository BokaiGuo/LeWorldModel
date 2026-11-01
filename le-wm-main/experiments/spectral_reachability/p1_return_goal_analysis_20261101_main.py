"""Report every push-and-return slot, including initially ineligible tasks."""
import argparse
import hashlib
import json
from pathlib import Path

from . import e3_initial_goal_qualification_20261031 as qualification
from .p1_common_h10_analysis_20261101_main import metrics, delta

ROOT = Path(__file__).resolve().parents[2]


def analyze():
    rows, sources = [], {}

    def read(path):
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    target = read(ROOT / "results/p1_return_target_cuda_20261101_main_MAIN06.json")
    batch = read(ROOT / "results/p1_return_goal_batch_20261101_main_MAIN01.json")
    for slot in batch["rows"]:
        root, arm = slot["root"], slot["arm"]
        goal = target["rows"][root]["goal_state"]
        if not slot["executed"]:
            if not target["rows"][root]["native_initial"]["initial_success"]:
                raise ValueError("unsupported ineligible slot")
            rows.append(dict(root=root, arm=arm, status=slot["status"], executed=False,
                             method_success=None, physical_steps=0))
            continue
        directory = ROOT / f"results/p1_return_goal_shard_20261101_main_root{root}_{arm}_RUN_MAIN01"
        report = read(directory / "report.json")
        row = next(r for r in report["rows"] if r["arm"] == arm)
        points = []
        for cp in row["checkpoints"]:
            if cp["goal"] != goal:
                raise ValueError("task goal mismatch")
            original = qualification.native(goal, cp["state"])
            points.append(dict(step=cp["step"], metrics=metrics(cp["state"], goal),
                               native7_distance=original["native7_distance"],
                               joint_xy_error=original["joint_xy_error"],
                               native_success=original["initial_success"]))
        initial = next(p for p in points if p["step"] == 0)
        first = next((p for p in points if p["step"] == 5), None)
        rows.append(dict(root=root, arm=arm, executed=True, status=row["status"],
                         physical_steps=row["control_returned"], method_success=row["native_success"],
                         error=row["first_error"], points=points,
                         prefix_delta=delta(first["metrics"], initial["metrics"]) if first else None,
                         prefix_joint_xy_delta=first["joint_xy_error"]-initial["joint_xy_error"] if first else None,
                         prefix_native7_delta=first["native7_distance"]-initial["native7_distance"] if first else None))
    return dict(rows=rows, sources=sources, predefined_tasks=2, eligible_tasks=1,
                physical_control_attempts=4, ineligible_slots=4, P1_admission=False,
                independent_confirmation=False,
                caution="Native7 contains velocity; do not certify non-greedy mechanism from it alone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=len(result["rows"]), output=str(args.output))))
