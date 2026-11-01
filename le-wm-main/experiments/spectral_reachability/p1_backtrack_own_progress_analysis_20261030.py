"""Analyze own-initial and matched-control physical task progress, not P1 admission."""
import argparse
import math
from pathlib import Path
import json

from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as saved
from experiments.spectral_reachability import p1_backtrack_physical_runner_20261030 as physical
from experiments.spectral_reachability import p1_backtrack_rgb_feedback_cuda_20261030 as bound

require = physical.require


def pose_error(state, goal):
    require(len(state) == len(goal) == 7 and all(math.isfinite(float(x)) for x in state + goal),
            "finite native state and goal")
    position = math.hypot(state[2] - goal[2], state[3] - goal[3])
    diff = state[4] - goal[4]
    return position, abs(math.atan2(math.sin(diff), math.cos(diff)))


def analyze(report):
    require(len(report["rows"]) == 16, "fixed roster")
    output = []
    for index, row in enumerate(report["rows"]):
        require(row["index"] == index and [c["step"] for c in row["checkpoints"]] == [0, 5, 10, 15],
                "checkpoint time identity")
        goal = row["goal"]
        initial = pose_error(row["checkpoints"][0]["state"], goal)
        samples = []
        for checkpoint in row["checkpoints"]:
            step = checkpoint["step"]
            pos, angle = pose_error(checkpoint["state"], goal)
            if step:
                response = row["responses"][step - 1]
                require(response["step"] == step and response["state"] == checkpoint["state"], "response identity")
                m = response["native_metrics"]
                require(math.isclose(pos, m["object_position_error"], rel_tol=1e-12, abs_tol=1e-12) and
                        math.isclose(angle, m["wrapped_object_angle_error"], rel_tol=1e-12, abs_tol=1e-12),
                        "independent pose metric agreement")
                success = response["native_success"]
            else:
                success = None
            samples.append(dict(step=step, position_error=pos, angle_error=angle,
                                position_change_from_step0=pos - initial[0],
                                angle_change_from_step0=angle - initial[1], native_success=success))
        output.append(dict(index=index, root_slot=row["root_slot"], condition=row["condition"],
                           arm=row["arm"], role=row["role"], samples=samples))
    return dict(schema="p1_backtrack_own_progress_analysis_20261030", rows=output,
                source_sha256=bound.REPORT_SHA, known_roots=2, P1_admission=False,
                limitation="Exploratory fixed trajectories; repeated actions are not independent.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report_path = bound.REPORT
    require(physical.sha(report_path) == bound.REPORT_SHA, "fixed report hash")
    report = physical.native.read(report_path, cap=64 * 1024**2)
    binding = report["execution_binding"]
    prepared, _ = physical.verify(binding["prepared"], binding["prepare_sha256"])
    saved.check(report_path.parent, report, prepared)
    result = analyze(report)
    source_files = [Path(__file__), Path(saved.__file__)]
    result["analysis_code_bindings"] = {str(path.resolve()): physical.sha(path) for path in source_files}
    out = Path(args.output)
    require(out.is_absolute() and out.parent == physical.ROOT / "results" and
            out.name.startswith("p1_backtrack_own_progress_analysis_20261030_") and
            not out.exists() and not out.is_symlink(), "exclusive output")
    print(json.dumps(dict(path=str(out), sha256=bound.replay.one.saved.write(out, result))))


if __name__ == "__main__":
    main()
