"""Join frozen-goal and physical metrics to fixed same-action CUDA results."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import codex_p1_null_same_action_saved_audit_20261030_v1 as audit
from experiments.spectral_reachability import p1_backtrack_rgb_feedback_analysis_20261030 as old_analysis

exp = audit.exp
OLD = exp.ROOT / "results/p1_backtrack_rgb_feedback_cuda_20261030_MAIN01_report.json"
OLD_SHA = "57722787fb1a008440fdc0a0acbcca733913db9a231aa4d0cbc6f7a4012cc21e"


def join(new, physical, old, old_prepared):
    exp.require(len(new["rows"]) == len(physical["rows"]) == len(old["rows"])
                == len(old_prepared["rows"]) == 16, "fixed rows")
    rows = []
    for i, (row, phys) in enumerate(zip(new["rows"], physical["rows"], strict=True)):
        # Historical selected record shares the task and goal with both branches.
        old_row, old_p = old["rows"][(i // 2) * 2 + 1], old_prepared["rows"][(i // 2) * 2 + 1]
        exp.require(row["index"] == phys["index"] == i and row["task_id"] == phys["task_id"]
                    == old_p["task_id"] and row["slot"] == old_row["slot"] == old_p["slot"]
                    and row["arm"] == old_row["arm"] == old_p["arm"]
                    and old_row["role"] == old_p["role"] == "model_selected"
                    and phys["role"] == row["role"]
                    and phys["goal_pixel_sha256"] == old_p["goal_pixel_sha256"], "goal join")
        goal = np.asarray(old_row["goal_latent"], dtype=np.float64)
        exp.require(goal.shape == (192,) and np.isfinite(goal).all(), "goal latent")
        times = []
        for c in row["comparisons"]:
            step = c["step"]
            native = phys["responses"][step - 1]
            exp.require(native["step"] == step, "physical time")
            item = {"step": step}
            for name in ("actual_latent", "open_prediction", "refreshed_prediction"):
                if c[name] is None:
                    item[name + "_goal_distance"] = None
                else:
                    v = np.asarray(c[name], dtype=np.float64)
                    exp.require(v.shape == (192,) and np.isfinite(v).all(), "latent")
                    item[name + "_goal_distance"] = float(np.linalg.norm(v - goal))
            metrics = native["native_metrics"]
            item.update(position_error=metrics["object_position_error"],
                        angle_error=metrics["wrapped_object_angle_error"],
                        success=native["native_success"])
            times.append(item)
        rows.append(dict(index=i, task_id=row["task_id"], role=row["role"],
                         duplicate_group=row["duplicate_action_task_group"], times=times))
    pairs = []
    for i in range(0, 16, 2):
        a, b = rows[i:i + 2]
        exp.require(a["role"] == "selected_prefix" and b["role"] == "zero_prefix"
                    and a["task_id"] == b["task_id"], "pair")
        ts = []
        for x, y in zip(a["times"], b["times"], strict=True):
            exp.require(x["step"] == y["step"], "pair time")
            fields = ("actual_latent_goal_distance", "open_prediction_goal_distance",
                      "refreshed_prediction_goal_distance", "position_error", "angle_error")
            ts.append(dict(step=x["step"], selected_minus_zero={
                k: None if x[k] is None or y[k] is None else x[k] - y[k] for k in fields},
                selected_success=x["success"], zero_success=y["success"]))
        pairs.append(dict(selected=i, zero=i + 1, duplicate_group=a["duplicate_group"], times=ts))
    return dict(rows=rows, pairs=pairs, P1_admission=False, independent_confirmation=False,
                limitation="Known roots, repeated actions, post-outcome analysis; no neural or physical re-execution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    exp.require(out.is_absolute() and out.parent == exp.ROOT / "results"
                and out.name.startswith("p1_null_same_action_goal_join_20261030_main_")
                and not out.exists() and not out.is_symlink(), "exclusive output")
    prepared = exp.ROOT / "results/codex_p1_null_same_action_error_cuda_20261030_v1_prepare_MAIN01.json"
    report = exp.ROOT / "results/codex_p1_null_same_action_error_cuda_20261030_v1_run_MAIN01.json"
    psha = "cc51cba912cb67c3a99bbe479bfecf4e9848c9ec72a223396377e19b7f8ddf63"
    rsha = "56c469eb3c368b3037140bb03288b205feb49eec2aa70966e57ef6eec33a3695"
    audit.audit(report, rsha, prepared, psha)
    new = audit.read_bounded(report, rsha, "run")
    physical = exp.native.read(exp.REPORT, cap=64 * 1024**2)
    old = exp.replay.one.saved.read_verified(OLD, OLD_SHA)
    old_p = exp.replay.one.saved.read_verified(old["prepared"], old["prepare_sha256"])
    exp.require(old_p == old_analysis.source.inputs(), "historical goal source closure")
    old_analysis.analyze(old, old_p)
    exp.require(old["model_before"] == new["model_before"], "goal encoder checkpoint identity")
    result = join(new, physical, old, old_p)
    result.update(report_sha256=rsha, prepare_sha256=psha, physical_sha256=exp.REPORT_SHA,
                  historical_goal_report_sha256=OLD_SHA)
    exp.write_exclusive(out, exp.encode_json(result))
    print(json.dumps({"path": str(out), "sha256": exp.sha(out)}))


if __name__ == "__main__":
    main()
