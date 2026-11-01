"""Explore paired fixed-action physical outcomes after complete saved verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.spectral_reachability import p1_matched_censored_consumer_20261030_codex as saved
from experiments.spectral_reachability import p1_matched_censored_runner_20261030_codex as runner


def errors(state, goal):
    if state is None:
        return None
    if len(state) != 7 or len(goal) != 7 or not all(math.isfinite(x) for x in (*state, *goal)):
        raise ValueError("invalid state or goal")
    d = state[4] - goal[4]
    return {"position": math.hypot(state[2] - goal[2], state[3] - goal[3]),
            "angle": abs(math.atan2(math.sin(d), math.cos(d)))}


def paired(anchors, probe_sets, rows):
    if len(anchors) != 16 or len(probe_sets) != 16 or len(rows) != 16:
        raise ValueError("fixed 16 anchors required")
    pairs, duplicate_groups = [], {}
    for i in range(0, 16, 2):
        left, right = anchors[i:i + 2]
        if (left["index"], right["index"]) != (i, i + 1) or (
            left["role"], right["role"]) != ("incumbent", "model_selected"):
            raise ValueError("anchor order")
        if any(left[k] != right[k] for k in
               ("root_slot", "condition", "arm", "task_id", "goal", "physical_root_sha256")):
            raise ValueError("paired task mismatch")
        if len(probe_sets[i]) != 21 or len(probe_sets[i + 1]) != 21:
            raise ValueError("fixed probe count")
        lrow, rrow = rows[i:i + 2]
        if lrow["anchor"] != i or rrow["anchor"] != i + 1:
            raise ValueError("verified row order")
        for row in (lrow, rrow):
            if len(row["endpoints"]) != 21 or len(row["statuses"]) != 21:
                raise ValueError("fixed endpoint count")
            for status, endpoint in zip(row["statuses"], row["endpoints"], strict=True):
                if status not in ("probe_completed", "native_stopped", "censored_out_of_domain"):
                    raise ValueError("unknown endpoint status")
                if (status == "probe_completed") != (endpoint is not None):
                    raise ValueError("status endpoint mismatch")
        lzero = errors(lrow["endpoints"][0], left["goal"])
        rzero = errors(rrow["endpoints"][0], right["goal"])
        records = []
        for j, (lp, rp) in enumerate(zip(probe_sets[i], probe_sets[i + 1], strict=True)):
            if (lp["index"], rp["index"]) != (j, j) or lp["name"] != rp["name"] or (
                lp["raw5_float32_hex"] != rp["raw5_float32_hex"]
            ):
                raise ValueError("matched probe action mismatch")
            le, re = errors(lrow["endpoints"][j], left["goal"]), errors(rrow["endpoints"][j], left["goal"])
            lgain = None if le is None or lzero is None else {
                key: lzero[key] - le[key] for key in ("position", "angle")}
            rgain = None if re is None or rzero is None else {
                key: rzero[key] - re[key] for key in ("position", "angle")}
            records.append({"probe": lp["name"], "index": j,
                            "incumbent_status": lrow["statuses"][j], "selected_status": rrow["statuses"][j],
                            "incumbent_error": le, "selected_error": re,
                            "selected_minus_incumbent": None if le is None or re is None else
                            {key: re[key] - le[key] for key in ("position", "angle")},
                            "incumbent_gain_vs_own_zero": lgain, "selected_gain_vs_own_zero": rgain,
                            "selected_minus_incumbent_gain_vs_own_zero":
                            None if lgain is None or rgain is None else
                            {key: rgain[key] - lgain[key] for key in ("position", "angle")}})
        group_material = [left["task_id"], left["condition"], left["goal"],
                          left["physical_root_sha256"], right["physical_root_sha256"],
                          left["prefix5_float32_hex"], right["prefix5_float32_hex"],
                          [p["raw5_float32_hex"] for p in probe_sets[i]]]
        identity = json.dumps(group_material, separators=(",", ":"), allow_nan=False).encode()
        duplicate_key = hashlib.sha256(identity).hexdigest()
        group = duplicate_groups.setdefault(duplicate_key, len(duplicate_groups))
        pairs.append({"root_slot": left["root_slot"], "condition": left["condition"], "arm": left["arm"],
                      "incumbent_anchor": i, "selected_anchor": i + 1, "probes": records})
        pairs[-1].update(duplicate_group=group, duplicate_key_sha256=duplicate_key)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha256", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--prepare-sha256", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if not output.is_absolute() or output.parent != runner.runner.physical.ROOT / "results":
        raise ValueError("output must be an absolute results path")
    code_sha = runner.runner.physical.sha(Path(__file__))
    verified = saved.consume(args.directory, args.report_sha256, args.prepared, args.prepare_sha256)
    preparation = runner.verify_prepare(args.prepared, args.prepare_sha256)
    original = preparation["original"]
    result = {"schema": "p1_matched_censored_analysis_20261030_main.v1",
              "run_report_sha256": args.report_sha256, "prepare_sha256": args.prepare_sha256,
              "analysis_source_sha256": code_sha,
              "fixed_branches": verified["fixed_branches"], "statuses": verified["counts"],
              "pairs": paired(original["anchors"], original["probe_sets"], verified["rows"]),
              "independent_confirmation": False, "P1_admission": False,
              "limitation": "Known roots; duplicate trajectories; fixed exploratory suffixes, not a causal capability proof."}
    if runner.runner.physical.sha(Path(__file__)) != code_sha:
        raise ValueError("analysis source changed during consumption")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
