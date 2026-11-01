"""Build an exploratory paired table from a completed, saved-verified H10 panel."""
import argparse
import json
from pathlib import Path

from . import p1_response_prefix_h10_20261031 as panel
from . import p1_response_h10_control_binding_20261031 as control
from .codex_p1_h10_paired_table_20261030_main import paired_difference, physical_metrics

need, sha = panel.need, panel.sha
NAME = "p1_response_h10_paired_table_20261031"
STATUSES = {"probe_completed", "native_stopped", "censored_out_of_domain"}
PLANNED = panel.ROOT / "results/p1_response_prefix_cp5_physical_20261031_consume_RUN01.json"
PLANNED_SHA = "7f7c67cef27a790e9d64eacde6d707c92d0cf0c027bd9901c6c0825794399f75"
NEW_PLAN = panel.ROOT / "results/p1_response_prefix_cp5_adam_20261031_RUN01.json"
NEW_PLAN_SHA = "705b75f68bfbb6663e7731428580e1639344a7afb7e9c329aa2a9ebc17d3327b"
OLD_PLAN = panel.ROOT / "results/p1_uphill_multistart_adam_20261030_main_RUN01.json"
OLD_PLAN_SHA = "018dffcb53e5e4dd40e6323fc3665ca53321cb027023c0cc4c1edb5daa06229f"


def verify_planning_protocol(prepared, new_plan, old_plan):
    for plan in (new_plan, old_plan):
        config = plan["config"]
        need(config["optimizer"] == "Adam" and config["starts"] == 4
             and config["steps_per_start"] == 20 and config["learning_rate"] == 0.05
             and config["action_box"] == [-1, 1], "optimizer configuration")
        need(plan["model_before"] == plan["model_after"], "frozen model")
    need(new_plan["model_before"] == old_plan["model_before"], "same model")
    for anchor in prepared["anchors"]:
        new = new_plan["rows"][anchor["index"]]
        pair = anchor["logical_pair"]
        need(new["index"] == anchor["index"] and new["logical_pair"] == pair
             and new["status"] == "planned", "new plan identity")
        for offset, role in ((1, "zero"), (2, "greedy")):
            old = old_plan["rows"][3 * pair + offset]
            need(old["logical_pair"] == pair and old["role"] == role
                 and old["initial_f32_sha256"] == new["initial_f32_sha256"],
                 "matched initialization")


def build(prepared, consumed):
    anchors = prepared["anchors"]
    rows = consumed["rows"]
    need(prepared["fixed_rows"] == 16 and prepared["fixed_branches"] == 588
         and len(anchors) == len(rows) == 14
         and consumed["status"] == "saved_checks_passed"
         and consumed["fixed_branches"] == 588, "panel denominator")
    by_index = {}
    for anchor, result in zip(anchors, rows, strict=True):
        need(anchor["index"] == result["anchor"] and anchor["index"] not in by_index
             and len(result["statuses"]) == len(result["endpoints"]) == 42, "anchor identity")
        for status, endpoint in zip(result["statuses"], result["endpoints"], strict=True):
            need(status in STATUSES and ((endpoint is not None) == (status == "probe_completed")),
                 "endpoint censoring")
        by_index[anchor["index"]] = (anchor, result)
    censored = prepared["censored_before_cp5"]
    need(len(censored) == 2 and set(by_index).isdisjoint(censored)
         and set(by_index) | set(censored) == set(range(16)), "fixed prefix roster")

    table = []
    for anchor, result in zip(anchors, rows, strict=True):
        pair = anchor["logical_pair"]
        bank = prepared["banks"][pair]
        need(len(bank) == 42 and bank[0]["name"] == "zero"
             and [p["index"] for p in bank] == list(range(42)), "fixed suffix bank")
        root = physical_metrics(anchor["root_state"], anchor["goal"])
        prefix = physical_metrics(anchor["checkpoint"]["state"], anchor["goal"])
        own_zero = physical_metrics(result["endpoints"][0], anchor["goal"])
        for probe, status, state in zip(bank, result["statuses"], result["endpoints"], strict=True):
            end = physical_metrics(state, anchor["goal"])
            table.append({
                "anchor": anchor["index"], "logical_pair": pair,
                "root_slot": anchor["root_slot"], "condition": anchor["condition"],
                "arm": anchor["arm"], "task_id": anchor["task_id"], "role": anchor["role"],
                "probe_index": probe["index"], "probe": probe["name"],
                "probe_action_float32_hex": probe["raw10_float32_hex"],
                "status": status, "root_metrics": root, "prefix_metrics": prefix,
                "prefix_minus_root": paired_difference(prefix, root),
                "endpoint_metrics": end, "own_zero_suffix_status": result["statuses"][0],
                "endpoint_minus_own_zero_suffix": paired_difference(end, own_zero),
                "independently_optimized_suffix_status": "not measured by this panel",
            })
    need(len(table) == 588, "table denominator")
    groups = {}
    for anchor, result in zip(anchors, rows, strict=True):
        group = groups.setdefault(anchor["logical_pair"], {})
        need(anchor["role"] not in group, "unique candidate role")
        group[anchor["role"]] = (anchor, result)
    comparisons = []
    for pair, group in sorted(groups.items()):
        need(set(group).issubset({"incumbent", "model_selected"}) and "incumbent" in group,
             "candidate roles")
        if "model_selected" not in group:
            continue
        a, ar = group["model_selected"]
        b, br = group["incumbent"]
        need(all(a[k] == b[k] for k in ("root_slot", "condition", "arm", "task_id",
                                         "root_state", "goal")), "paired task identity")
        for probe in prepared["banks"][pair]:
            i = probe["index"]
            am = physical_metrics(ar["endpoints"][i], a["goal"])
            bm = physical_metrics(br["endpoints"][i], b["goal"])
            az = physical_metrics(ar["endpoints"][0], a["goal"])
            bz = physical_metrics(br["endpoints"][0], b["goal"])
            a_response = paired_difference(am, az)
            b_response = paired_difference(bm, bz)
            comparisons.append({
                "logical_pair": pair, "probe_index": i, "probe": probe["name"],
                "model_selected_anchor": a["index"], "incumbent_anchor": b["index"],
                "model_selected_status": ar["statuses"][i], "incumbent_status": br["statuses"][i],
                "model_selected_minus_incumbent_endpoint": paired_difference(am, bm),
                "model_selected_minus_incumbent_response_vs_own_zero":
                    None if a_response is None or b_response is None else
                    {key: a_response[key] - b_response[key] for key in a_response},
            })
    return {
        "schema": NAME, "rows": table, "candidate_comparisons": comparisons,
        "censored_before_cp5": censored,
        "fixed_prefixes": 16, "fixed_branches": 588,
        "limitations": "Known roots; common suffix is not independently optimized planning; "
                       "missing endpoints are not imputed. Candidate roles are not zero/greedy controls.",
        "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
    }


def attach_controls(value, prepared, old, old_pairing, planned):
    """Attach already verified controls and separate individually planned outcomes."""
    control.verify(prepared, old, old_pairing)
    need(planned["status"] == "saved_checks_passed" and planned["fixed_rows"] == 16
         and planned["censored_before_cp5"] == prepared["censored_before_cp5"]
         and len(planned["rows"]) == 14, "planned outcome roster")
    plan_by_index = {row["index"]: row for row in planned["rows"]}
    need(len(plan_by_index) == 14, "unique planned outcome")
    anchors = {anchor["index"]: anchor for anchor in prepared["anchors"]}
    for row in value["rows"]:
        anchor = anchors[row["anchor"]]
        plan = plan_by_index[row["anchor"]]
        need(all(plan[k] == anchor[k] for k in ("index", "role", "root_slot", "condition", "arm")),
             "planning anchor identity")
        need(plan["status"] in STATUSES and
             ((plan["completed_endpoint"] is not None) == (plan["status"] == "probe_completed")),
             "planning endpoint censoring")
        pair, i = row["logical_pair"], row["probe_index"]
        historical = old_pairing["common_suffix_pairs"][pair * 42 + i]
        row["independently_optimized_suffix_status"] = plan["status"]
        row["independently_optimized_suffix_endpoint"] = plan["completed_endpoint"]
        plan_metrics = physical_metrics(plan["completed_endpoint"], anchor["goal"])
        row["independently_optimized_suffix_endpoint_metrics"] = plan_metrics
        row["control_comparisons"] = {}
        for role, offset in (("zero", 1), ("greedy", 2)):
            entry = historical["roles"][role]
            old_anchor = old["anchors"][3 * pair + offset]
            old_plan = old_pairing["anchors"][3 * pair + offset]
            need(old_plan["index"] == old_anchor["index"]
                 and old_plan["identity"]["role"] == role, "historical planning identity")
            control_response = entry["endpoint_minus_own_zero_suffix"]
            own_response = row["endpoint_minus_own_zero_suffix"]
            row["control_comparisons"][role] = {
                "status": entry["status"], "endpoint": entry["endpoint"],
                "endpoint_minus_control": paired_difference(row["endpoint_metrics"], entry["endpoint"]),
                "response_minus_control_response": None if own_response is None or control_response is None
                    else {key: own_response[key] - control_response[key] for key in own_response},
                "independently_optimized_suffix": old_plan["common_multistart_adam"],
                "independently_optimized_endpoint_minus_control":
                    paired_difference(plan_metrics, old_plan["common_multistart_adam"]["endpoint"]),
            }
    value["limitations"] += (" Historical zero/greedy controls pass source identity checks but share "
                             "known roots. Individually optimized suffixes are exploratory and "
                             "not certified equal total cost.")
    return value


def attach_verified_stops(value, anchor, source):
    """Expose last accepted observations without imputing a fixed-horizon endpoint.

    Caller must first verify the shard and bind its report hash to the consumer.
    In particular the out-of-domain returned state is not an accepted response.
    """
    rows = [row for row in value["rows"] if row["anchor"] == anchor["index"]]
    need(len(rows) == len(source["rows"]) == 42, "stop roster")
    for row, branch in zip(rows, source["rows"], strict=True):
        need(row["probe_index"] == branch["index"] and row["status"] == branch["status"],
             "stop identity")
        responses = branch["responses"]
        need(len(responses) <= 10 and
             all(response["step"] == i for i, response in enumerate(responses, 1)),
             "accepted response order")
        last = responses[-1] if responses else None
        if branch["status"] == "native_stopped":
            need(last is not None and (last["terminated"] or last["truncated"]),
                 "native stop flags")
        else:
            need(all(not item["terminated"] and not item["truncated"] for item in responses),
                 "unexpected stop flags")
        row["accepted_suffix_steps"] = len(responses)
        row["last_accepted_metrics"] = physical_metrics(last["state"], anchor["goal"]) if last else None
        row["last_accepted_terminated"] = last["terminated"] if last else None
        row["last_accepted_truncated"] = last["truncated"] if last else None
        row["last_accepted_is_fixed_horizon_endpoint"] = branch["status"] == "probe_completed"
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for option in ("prepared", "prepare-sha", "consumed", "consume-sha",
                   "directory", "report-sha", "output"):
        ap.add_argument("--" + option, required=True)
    args = ap.parse_args()
    output, directory = Path(args.output), Path(args.directory)
    need(output.is_absolute() and output.parent == panel.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    read = panel.physical.bundle.physical.prior.native.read
    for path, digest in ((args.prepared, args.prepare_sha), (args.consumed, args.consume_sha),
                         (directory / "run_report.json", args.report_sha)):
        need(sha(path) == digest, "input hash")
    prepared = read(args.prepared, cap=64 * 1024**2)
    consumed = read(args.consumed, cap=64 * 1024**2)
    report = read(directory / "run_report.json", cap=64 * 1024**2)
    need(prepared == panel.build() and report["status"] == "recorded_not_consumed"
         and report["prepare_sha256"] == args.prepare_sha
         and len(report["shards"]) == 14, "run and prepared binding")
    shard_sources = []
    for anchor, saved, shard in zip(prepared["anchors"], consumed["rows"],
                                    report["shards"], strict=True):
        path = directory / f"anchor_{anchor['index']:02d}" / "report.json"
        need(shard["index"] == anchor["index"] == saved["anchor"]
             and sha(path) == shard["sha256"], "shard binding")
        source = read(path, cap=64 * 1024**2)
        need([row["status"] for row in source["rows"]] == saved["statuses"]
             and [row["responses"][-1]["state"] if row["status"] == "probe_completed" else None
                  for row in source["rows"]] == saved["endpoints"], "consumer/report agreement")
        shard_sources.append(source)
    value = build(prepared, consumed)
    for anchor, source in zip(prepared["anchors"], shard_sources, strict=True):
        attach_verified_stops(value, anchor, source)
    sources = ((control.OLD, control.OLD_SHA), (control.PAIR, control.PAIR_SHA),
               (PLANNED, PLANNED_SHA), (NEW_PLAN, NEW_PLAN_SHA), (OLD_PLAN, OLD_PLAN_SHA))
    need(all(sha(path) == digest for path, digest in sources), "control and planning hashes")
    old, old_pairing, planned, new_plan, old_plan = (
        read(path, cap=64 * 1024**2) for path, _ in sources)
    verify_planning_protocol(prepared, new_plan, old_plan)
    value = attach_controls(value, prepared, old, old_pairing, planned)
    need(all(sha(path) == digest for path, digest in sources), "control and planning drift")
    value["input_sha256"] = {"prepare": args.prepare_sha, "consume": args.consume_sha,
                             "run_report": args.report_sha, "analysis_code": sha(__file__),
                             "old_prepare": control.OLD_SHA, "old_pairing": control.PAIR_SHA,
                             "planned_consume": PLANNED_SHA, "new_plan": NEW_PLAN_SHA,
                             "old_plan": OLD_PLAN_SHA, "control_code": sha(control.__file__)}
    for path, digest in ((args.prepared, args.prepare_sha), (args.consumed, args.consume_sha),
                         (directory / "run_report.json", args.report_sha)):
        need(sha(path) == digest, "input drift")
    for shard in report["shards"]:
        need(sha(directory / f"anchor_{shard['index']:02d}" / "report.json") == shard["sha256"],
             "shard drift")
    digest = panel.physical.bundle.physical.write(output, value)
    print(json.dumps({"rows": len(value["rows"]), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
