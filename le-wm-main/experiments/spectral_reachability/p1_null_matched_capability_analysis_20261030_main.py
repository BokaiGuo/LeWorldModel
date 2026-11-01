"""Analyze fixed prefix-null probe responses after full saved acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p1_prefix_null_matched_20261030 as run
from experiments.spectral_reachability import codex_p1_prefix_null_matched_saved_20261030 as saved
from experiments.spectral_reachability import p1_backtrack_object_capability_analysis_20261030 as measure


def analyze(anchors, verified):
    run.need(len(anchors) == len(verified["rows"]) == 16, "fixed anchors")
    rows = []
    for i, (anchor, row) in enumerate(zip(anchors, verified["rows"], strict=True)):
        run.need(anchor["index"] == row["anchor"] == i, "anchor order")
        rows.append(measure.measure(anchor, row["endpoints"]))
    pairs, groups = [], {}
    for i in range(0, 16, 2):
        selected, zero = rows[i:i+2]
        run.need((selected["role"], zero["role"]) == ("selected_prefix", "zero_prefix")
                 and all(selected[k] == zero[k] for k in ("root_slot", "condition", "arm")),
                 "pair identity")
        a, b = anchors[i:i+2]
        run.need(a["task_id"] == b["task_id"] and a["goal"] == b["goal"]
                 and a["root_state"] == b["root_state"]
                 and a["physical_root_sha256"] == b["physical_root_sha256"], "pair source")
        result = dict(selected_anchor=i, zero_anchor=i+1, root_slot=a["root_slot"],
                      condition=a["condition"], arm=a["arm"])
        identity = [a["task_id"], a["condition"], a["physical_root_sha256"], a["goal"],
                    a["prefix5_float32_hex"], b["prefix5_float32_hex"]]
        key = hashlib.sha256(json.dumps(identity, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        result.update(duplicate_group=groups.setdefault(key, len(groups)), duplicate_key_sha256=key)
        for key in ("fixed_bank_best_position_gain", "fixed_bank_best_angle_gain", "angle_response_l2"):
            result["selected_minus_zero_" + key] = (
                None if selected.get(key) is None or zero.get(key) is None
                else selected[key] - zero[key])
        # Compare the same probe coordinates, not only each anchor's separate
        # best probe. A censored endpoint stays missing in the paired result.
        for key in ("zero_relative_position_error_gains", "zero_relative_angle_error_gains",
                    "position_goal_response", "angle_goal_response"):
            left, right = selected.get(key), zero.get(key)
            result["selected_minus_zero_" + key] = (
                None if left is None or right is None else
                [None if x is None or y is None else x - y
                 for x, y in zip(left, right, strict=True)])
        pairs.append(result)
    return rows, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    run.need(output.is_absolute() and output.parent == run.ROOT / "results"
             and output.name.startswith("p1_null_matched_capability_analysis_20261030_main_"),
             "exclusive output namespace")
    sources = {str(Path(__file__).resolve()): run.sha(Path(__file__)),
               str(Path(measure.__file__).resolve()): run.sha(Path(measure.__file__))}
    checked = saved.consume(args.directory, args.report_sha, args.prepared, args.prepare_sha)
    p, _ = run.verify(args.prepared, args.prepare_sha)
    rows, pairs = analyze(p["anchors"], checked)
    run.need(all(run.sha(path) == digest for path, digest in sources.items()), "analysis code drift")
    result = dict(schema="p1_null_matched_capability_analysis_20261030_main.v1",
                  report_sha256=args.report_sha, prepare_sha256=args.prepare_sha,
                  source_hashes=sources, counts=checked["counts"], rows=rows, pairs=pairs,
                  fixed_branches=336, independent_confirmation=False, P1_admission=False,
                  limitations="Known roots and repeated actions; finite coordinate secants are not a causal proof.")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
