"""Analyze fixed same-action RGB feedback predictions; never authorize P1."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_backtrack_rgb_feedback_cuda_20261030 as source


def array(value, shape):
    result = np.asarray(value, dtype=np.float64)
    source.require(result.shape == shape and np.isfinite(result).all(), "finite prediction shape")
    return result


def analyze(report, prepared):
    source.require(report["schema"] == source.NAME + ".report" and
                   report["fixed_rows"] == prepared["fixed_rows"] == 16 and
                   report["model_before"] == report["model_after"] == prepared["model_hash"] and
                   report["calls"] == dict(encoder=80, projector=80, predictor=32, action_encoder=32) and
                   report["training"] is False and report["P1_admission"] is False and
                   report["freshness"] is False and len(report["rows"]) == len(prepared["rows"]) == 16,
                   "report contract")
    rows = []
    for index, (r, p) in enumerate(zip(report["rows"], prepared["rows"], strict=True)):
        source.require(r["index"] == p["index"] == index and
                       all(r[k] == p[k] for k in ("slot", "arm", "role")) and
                       r["open_predictions"] == p["open_endpoints"] and
                       r["native_responses"] == p["native_responses"], "source row identity")
        actual = array(r["actual_latents"], (4, 192))
        goal = array(r["goal_latent"], (192,))
        opened = array(r["open_predictions"], (3, 192))
        feedback = array(r["feedback_predictions"], (2, 192))
        times = []
        for j, step in enumerate((5, 10, 15)):
            target = actual[j + 1]
            response = r["native_responses"][step - 1]
            source.require(response["step"] == step, "physical time identity")
            metrics = response["native_metrics"]
            item = dict(step=step, actual_goal_distance=float(np.linalg.norm(target - goal)),
                        open_goal_distance=float(np.linalg.norm(opened[j] - goal)),
                        open_mse=float(np.mean((opened[j] - target) ** 2)),
                        object_position_error=metrics["object_position_error"],
                        wrapped_object_angle_error=metrics["wrapped_object_angle_error"],
                        native_success=response["native_success"])
            if j:
                item.update(feedback_goal_distance=float(np.linalg.norm(feedback[j - 1] - goal)),
                            feedback_mse=float(np.mean((feedback[j - 1] - target) ** 2)))
            times.append(item)
        rows.append(dict(index=index, slot=r["slot"], arm=r["arm"], role=r["role"], times=times))
    pairs = []
    for i in range(0, 16, 2):
        a, b = rows[i:i + 2]
        source.require(a["slot"] == b["slot"] and a["arm"] == b["arm"] and
                       (a["role"], b["role"]) == ("incumbent", "model_selected"), "pair identity")
        times = []
        for left, right in zip(a["times"], b["times"], strict=True):
            source.require(left["step"] == right["step"], "pair time identity")
            fields = ("actual_goal_distance", "open_goal_distance", "object_position_error",
                      "wrapped_object_angle_error")
            if left["step"] > 5:
                fields += ("feedback_goal_distance",)
            times.append(dict(step=left["step"], selected_minus_incumbent={
                field: right[field] - left[field] for field in fields},
                incumbent_success=left["native_success"], selected_success=right["native_success"]))
        pairs.append(dict(slot=a["slot"], arm=a["arm"], times=times))
    return dict(schema="p1_backtrack_rgb_feedback_analysis_20261030", rows=rows, pairs=pairs,
                limitation="Known roots; repeated actions and shared trajectories are not independent.",
                P1_admission=False, independent_confirmation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = source.replay.one.saved.read_verified(args.report, args.report_sha)
    prepared = source.replay.one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    source.require(prepared == source.inputs(), "prepared source closure")
    output = Path(args.output)
    source.require(output.is_absolute() and output.parent == source.ROOT / "results" and
                   output.name.startswith("p1_backtrack_rgb_feedback_analysis_20261030_") and
                   not output.exists() and not output.is_symlink(), "exclusive output")
    result = analyze(report, prepared)
    result.update(report=str(Path(args.report).resolve()), report_sha256=args.report_sha,
                  prepared=report["prepared"], prepare_sha256=report["prepare_sha256"])
    print(json.dumps(dict(path=str(output), sha256=source.replay.one.saved.write(output, result))))


if __name__ == "__main__":
    main()
