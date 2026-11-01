"""Create a complete, fail-closed P1 H10 paired table from verified evidence."""
import argparse
import json
import math
import os
from pathlib import Path

from . import codex_p1_three_prefix_h10_analysis_20261030 as analysis
from . import codex_p1_three_prefix_h10_runner_20261030 as runner

NAME = "codex_p1_h10_paired_table_20261030_main"
need = runner.require
ROLES = analysis.ROLES


def physical_metrics(state, goal):
    if state is None:
        return None
    s, g = analysis.vector(state), analysis.vector(goal)
    joint = math.sqrt(sum((s[i] - g[i]) ** 2 for i in range(4)))
    raw_angle = abs(s[4] - g[4])
    native_angle = min(raw_angle, 2 * math.pi - raw_angle)
    return {
        "object_position_error": math.hypot(s[2] - g[2], s[3] - g[3]),
        "object_angle_error_wrapped": abs(analysis.wrapped(s[4] - g[4])),
        "pusher_position_error": math.hypot(s[0] - g[0], s[1] - g[1]),
        "native_joint_xy_error": joint,
        "native_angle_term": native_angle,
        "native_success": joint < 20 and native_angle < math.pi / 9,
    }


def paired_difference(a, b):
    """Absolute endpoint difference; missing endpoint never becomes a zero."""
    if a is None or b is None:
        return None
    keys = ("object_position_error", "object_angle_error_wrapped", "pusher_position_error",
            "native_joint_xy_error", "native_angle_term")
    return {key: a[key] - b[key] for key in keys}


def build(original, verified, states):
    need(len(original["anchors"]) == 24 and len(verified["groups"]) == 8, "roster")
    need(len(states) == 24 and all(len(shard) == 42 for shard in states), "state roster")
    rows = []
    for group in verified["groups"]:
        i = group["logical_pair"]
        anchors = original["anchors"][3*i:3*i+3]
        need([a["role"] for a in anchors] == list(ROLES), "role order")
        need(all(a["root_state"] == anchors[0]["root_state"] and a["goal"] == anchors[0]["goal"]
                 for a in anchors), "paired root state and goal")
        root = physical_metrics(anchors[0]["root_state"], anchors[0]["goal"])
        prefixes = {a["role"]: physical_metrics(a["checkpoint"]["state"], a["goal"]) for a in anchors}
        for probe in group["probes"]:
            metrics_by_role = {
                role: physical_metrics(states[3*i+k][probe["index"]], anchors[k]["goal"])
                for k, role in enumerate(ROLES)
            }
            zero_suffix_by_role = {
                role: physical_metrics(states[3*i+k][0], anchors[k]["goal"])
                for k, role in enumerate(ROLES)
            }
            for role_index, role in enumerate(ROLES):
                item = probe["roles"][role]
                endpoint = item["metrics"]
                state = states[3*i+role_index][probe["index"]]
                need((state is not None) == (item["status"] == "probe_completed"), "endpoint status")
                rows.append({
                    "root_slot": group["root_slot"], "condition": group["condition"],
                    "logical_pair": i, "duplicate_group": group["duplicate_group"],
                    "arm": group["arm"], "role": role, "probe": probe["probe"],
                    "probe_sha256": probe["raw10_sha256"], "status": item["status"],
                    "observation": item.get("observation"),
                    "root_metrics": root, "prefix_metrics": prefixes[role],
                    "prefix_position_error_change": prefixes[role]["object_position_error"] - root["object_position_error"],
                    "prefix_angle_error_change": prefixes[role]["object_angle_error_wrapped"] - root["object_angle_error_wrapped"],
                    "prefix_pusher_position_error_change": prefixes[role]["pusher_position_error"] - root["pusher_position_error"],
                    "prefix_joint_xy_error_change": prefixes[role]["native_joint_xy_error"] - root["native_joint_xy_error"],
                    "prefix_native_angle_term_change": prefixes[role]["native_angle_term"] - root["native_angle_term"],
                    "prefix_native_success_transition": {
                        "root": root["native_success"], "prefix": prefixes[role]["native_success"],
                    },
                    "common_suffix_endpoint_object_metrics": endpoint,
                    "common_suffix_endpoint_native_metrics": metrics_by_role[role],
                    "same_suffix_endpoint_minus_zero": paired_difference(metrics_by_role[role], metrics_by_role["zero"]),
                    "same_suffix_endpoint_minus_greedy": paired_difference(metrics_by_role[role], metrics_by_role["greedy"]),
                    "same_prefix_endpoint_minus_own_zero_suffix":
                        paired_difference(metrics_by_role[role], zero_suffix_by_role[role]),
                    "response_vs_own_zero_suffix": item["response_vs_own_zero"],
                    "response_vs_own_zero_suffix_minus_zero_prefix":
                        analysis.relative_difference(item["response_vs_own_zero"],
                                                     probe["roles"]["zero"]["response_vs_own_zero"]),
                    "independently_optimized_suffix": None,
                    "independently_optimized_suffix_status": "not measured by H10",
                })
    need(len(rows) == 1008, "complete table denominator")
    return {"schema": NAME, "rows": rows, "status_counts": verified["status_counts"],
            "limitations": "Known roots, duplicate trajectories, common suffix not optimized suffix; no imputation.",
            "P1_admission": False, "equal_budget_certified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("directory", "report-sha256", "prepared", "prepare-sha256", "output"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    output = Path(a.output)
    need(output.is_absolute() and output.parent == runner.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
         "exclusive output")
    analysis_code_sha256 = runner.runner.physical.sha(Path(__file__))
    verified = analysis.consume(a.directory, a.report_sha256, a.prepared, a.prepare_sha256)
    original = runner.verify_prepare(a.prepared, a.prepare_sha256)["original"]
    root = Path(a.directory)
    report = runner.native.read(root / "run_report.json", cap=4*1024**2)
    states = []
    for i, shard in enumerate(report["shards"]):
        path = root / f"anchor_{i:02d}" / "report.json"
        need(shard["anchor"] == i and runner.runner.physical.sha(path) == shard["report_sha256"], "shard hash")
        value = runner.native.read(path, cap=64*1024**2)
        states.append([r["responses"][-1]["state"] if r["status"] == "probe_completed" else None
                       for r in value["rows"]])
        need(runner.runner.physical.sha(path) == shard["report_sha256"], "shard drift")
    result = build(original, verified, states)
    need(runner.runner.physical.sha(root / "run_report.json") == a.report_sha256, "report drift")
    need(runner.runner.physical.sha(Path(__file__)) == analysis_code_sha256, "paired table code drift")
    result.update(run_report_sha256=a.report_sha256, prepare_sha256=a.prepare_sha256,
                  paired_table_code_sha256=analysis_code_sha256,
                  verified_analysis_code_sha256=verified["analysis_code_sha256"])
    with output.open("xb") as stream:
        raw = runner.native.raw(result)
        need(stream.write(raw) == len(raw), "complete table write")
        stream.flush()
        os.fsync(stream.fileno())
    runner.base.paired.fsync_dir(output.parent)
    print(json.dumps({"status": "verified_exploratory", "rows": len(result["rows"]),
                      "sha256": runner.runner.physical.sha(output)}))


if __name__ == "__main__":
    main()
