"""Pair a fully verified uphill H10 panel with both RGB-refreshed planning runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from . import p1_uphill_h10_consume_20261030 as consumer

ROOT = consumer.experiment.ROOT
PREP = consumer.PREP
PREP_SHA = consumer.PREP_SHA
ZERO = ROOT / "results/p1_uphill_cp5_analysis_20261030_RUN01.json"
ZERO_SHA = "04c336045d689dd4629c1c88b08653e6ba7f3402b38a72f1e42203bd0572495e"
WARM = ROOT / "results/p1_uphill_cp5_analysis_warm_20261030_RUN01.json"
WARM_SHA = "4e8f094e037a2d627306cda7c79cc42141d228bde51e88f7352a42dd64d5ac3d"
MULTISTART = ROOT / "results/p1_uphill_multistart_pairing_20261030_main_RUN01.json"
MULTISTART_SHA = "b69b9d38b7f926c31648dc00d0445e710a1014bdff2dff216a979c43ac397ff4"
TRIPLET = ROOT / "results/p1_uphill_triplet_summary_20261031_main_RUN01.json"
TRIPLET_SHA = "09ea66e72e9babe4e1869d6a80f683d8237405bb2333d25fd9ef8c0f54fcfc44"
MIDPOINT_MULTI = ROOT / "results/p1_uphill_midpoint_multistart_compare_20261031_RUN01.json"
MIDPOINT_MULTI_SHA = "5f99b60b19b88aa41b7771f406cf0bf87a2626efc0b5c35083031124f59178c6"
ROLES = ("selected", "zero", "greedy")
FIELDS = ("object_position_error", "object_angle_error_wrapped", "pusher_position_error",
          "native_joint_xy_error", "native_angle_term")
IDENTITY = ("logical_pair", "root_slot", "condition", "arm", "role", "task_id",
            "physical_root_sha256")
require = consumer.require


def read_checked(path, digest):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "source hash: " + str(path))
    return json.loads(raw)


def same_metrics(a, b):
    return (a is not None and b is not None
            and all(math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-9) for k in FIELDS)
            and a["native_success"] == b["native_success"])


def difference(a, b):
    return None if a is None or b is None else {k: a[k] - b[k] for k in FIELDS}


def same_difference(saved, expected):
    return (saved is not None and expected is not None
            and all(math.isfinite(saved[k]) and math.isfinite(expected[k])
                    and math.isclose(saved[k], expected[k], rel_tol=1e-12, abs_tol=1e-9)
                    for k in FIELDS))


def bind_midpoint_multistart(result, comparison):
    """Attach an independently verified replay without changing gate or censoring."""
    require(comparison["schema"] == "p1_uphill_midpoint_multistart_compare_20261031"
            and len(comparison["rows"]) == len(result["anchors"]) == 24,
            "midpoint multistart contract")
    for anchor, row in zip(result["anchors"], comparison["rows"], strict=True):
        identity = anchor["identity"]
        require(row["index"] == anchor["index"]
                and all(row[k] == identity[k] for k in ("logical_pair", "root_slot", "condition", "arm", "role"))
                and row["first5_exact"] is True, "midpoint multistart identity")
        opened = anchor["common_multistart_adam"]
        require(row["open_status"] == opened["status"]
                and row["open_returned_suffix"] == opened["returned_suffix"]
                and row["open_last_success"] == opened["last_native_success"]
                and ((row["open_endpoint"] is None and opened["endpoint"] is None)
                     or same_metrics(row["open_endpoint"], opened["endpoint"])),
                "midpoint multistart open binding")
        require((row["feedback_endpoint"] is not None) == (row["feedback_status"] == "probe_completed"),
                "midpoint multistart endpoint censoring")
        anchor["midpoint_feedback"]["multistart"] = {
            "status": row["feedback_status"], "returned": row["feedback_returned_suffix"],
            "success": row["feedback_last_success"], "endpoint": row["feedback_endpoint"],
        }
    return result


def join(prepared, zero, warm, panel, multistart, triplet):
    require(panel["schema"] == "p1_uphill_h10_consume_20261030"
            and panel["prepare_sha256"] == PREP_SHA and panel["P1_admission"] is False,
            "panel contract")
    require(len(prepared["anchors"]) == len(zero["rows"]) == len(warm["rows"]) == 24
            and len(multistart["rows"]) == 24 and len(panel["rows"]) == 1008, "complete roster")
    require(multistart["schema"] == "p1_uphill_multistart_pairing_20261030_main"
            and multistart["P1_admission"] is False
            and multistart["equal_budget_certified"] is False, "multistart contract")
    require(triplet["P1_admission"] is False and triplet["equal_budget_certified"] is False
            and triplet["independent_confirmation"] is False
            and len(triplet["groups"]) == 8, "triplet contract")
    anchors = {}
    for i, (source, z, w, multi) in enumerate(zip(
            prepared["anchors"], zero["rows"], warm["rows"], multistart["rows"], strict=True)):
        require(source["index"] == z["index"] == w["index"] == multi["index"] == i, "anchor index")
        require(all(source[k] == z[k] == w[k] == multi[k] for k in IDENTITY), "anchor identity")
        require(source["prefix5_sha256"] == z["prefix_action_sha256"] == w["prefix_action_sha256"]
                and source["checkpoint"]["pixel_sha256"] == z["prefix_cp5_pixel_sha256"]
                == w["prefix_cp5_pixel_sha256"], "prefix identity")
        require(same_metrics(z["root_metrics"], w["root_metrics"])
                and same_metrics(z["prefix_metrics"], w["prefix_metrics"])
                and same_metrics(z["root_metrics"], multi["root"])
                and same_metrics(z["prefix_metrics"], multi["prefix"]), "planning metric identity")
        require((multi["endpoint"] is not None) == (multi["status"] == "probe_completed"),
                "multistart endpoint censoring")
        for row in (z, w):
            require(same_difference(row["prefix_minus_root"],
                                    difference(row["prefix_metrics"], row["root_metrics"])),
                    "prefix metric difference")
            plan = row["independent_suffix_20step"]
            require((plan["endpoint"] is not None) == (plan["status"] == "probe_completed"),
                    "planning endpoint censoring")
        key = (source["logical_pair"], source["role"])
        require(key not in anchors, "duplicate anchor")
        anchors[key] = {"identity": {k: z[k] for k in IDENTITY}, "index": i,
                        "prefix_action_sha256": z["prefix_action_sha256"],
                        "prefix_cp5_pixel_sha256": z["prefix_cp5_pixel_sha256"],
                        "root_metrics": z["root_metrics"], "prefix_metrics": z["prefix_metrics"],
                        "prefix_minus_root": z["prefix_minus_root"],
                        "zero_initialization": z["independent_suffix_20step"],
                        "warm_initialization": w["independent_suffix_20step"],
                        "common_multistart_adam": multi,
                        "zero_resources": z["planning_resource_receipt"],
                        "warm_resources": w["planning_resource_receipt"]}
    rows = {}
    for row in panel["rows"]:
        key = (row["logical_pair"], row["probe"], row["role"])
        require(key not in rows, "duplicate panel row")
        rows[key] = row
    pairs = []
    for logical in range(8):
        bank = prepared["probe_banks"][logical]
        require(len(bank) == 42 and len({p["name"] for p in bank}) == 42, "probe roster")
        for probe in bank:
            roles = {}
            for role in ROLES:
                anchor = anchors[(logical, role)]
                row = rows.pop((logical, probe["name"], role))
                require(all(row[k] == anchor["identity"][k] for k in IDENTITY)
                        and row["probe_sha256"] == probe["raw10_sha256"], "panel identity")
                require(same_metrics(row["root"], anchor["root_metrics"])
                        and same_metrics(row["prefix"], anchor["prefix_metrics"]), "physical metric binding")
                require((row["endpoint"] is not None) == (row["status"] == "probe_completed"),
                        "common endpoint censoring")
                roles[role] = {"anchor_index": anchor["index"], "status": row["status"],
                               "returned_suffix_steps": row["returned_suffix_steps"],
                               "endpoint": row["endpoint"],
                               "last_accepted_metrics": row["last_accepted_metrics"],
                               "last_accepted_terminated": row["last_accepted_terminated"],
                               "last_accepted_truncated": row["last_accepted_truncated"],
                               "endpoint_minus_own_zero_suffix": row["endpoint_minus_own_zero"]}
            for role in ROLES:
                roles[role]["endpoint_minus_zero_prefix"] = difference(
                    roles[role]["endpoint"], roles["zero"]["endpoint"])
                roles[role]["response_minus_zero_prefix"] = difference(
                    roles[role]["endpoint_minus_own_zero_suffix"],
                    roles["zero"]["endpoint_minus_own_zero_suffix"])
            pairs.append({"logical_pair": logical, "probe": probe["name"],
                          "probe_sha256": probe["raw10_sha256"], "roles": roles})
    require(not rows and len(pairs) == 336, "unconsumed panel rows")
    for logical, group in enumerate(triplet["groups"]):
        require(group["logical_pair"] == logical and len(group["prefixes"]) == 3, "triplet roster")
        for role, observed in zip(ROLES, group["prefixes"], strict=True):
            anchor = anchors[(logical, role)]
            require(observed["role"] == role and observed["index"] == anchor["index"]
                    and all(group[k] == anchor["identity"][k] for k in
                            ("root_slot", "condition", "arm", "task_id", "physical_root_sha256"))
                    and same_metrics(group["root"], anchor["root_metrics"])
                    and same_metrics(observed["prefix"], anchor["prefix_metrics"]),
                    "triplet identity and physical binding")
            multi = anchor["common_multistart_adam"]
            opened = observed["open"]
            require(opened["status"] == multi["status"]
                    and opened["returned"] == multi["returned_suffix"]
                    and opened["success"] == multi["last_native_success"]
                    and ((opened["endpoint"] is None and multi["endpoint"] is None)
                         or same_metrics(opened["endpoint"], multi["endpoint"])),
                    "triplet open binding")
            for strategy in ("real", "trust"):
                item = observed[strategy]
                require((item["endpoint"] is not None) == (item["status"] == "probe_completed"),
                        "triplet feedback censoring")
            anchor["midpoint_feedback"] = {k: observed[k] for k in ("real", "trust")}
    return {"schema": "p1_uphill_complete_pairing_20261030",
            "anchors": [anchors[(i, role)] for i in range(8) for role in ROLES],
            "common_suffix_pairs": pairs, "independent_plans_per_initialization": 24,
            "common_suffix_branches": 1008, "P1_admission": False,
            "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Known roots, duplicate arms and shared warm initialization from selected. "
                           "Common multistart Adam uses same starts within each pair, but full budget uncertified. "
                           "Common suffix is not independently optimized capacity; no endpoint imputation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith("p1_uphill_complete_pairing_")
            and not output.exists() and not output.is_symlink(), "exclusive output")
    prepared = read_checked(PREP, PREP_SHA)
    zero = read_checked(ZERO, ZERO_SHA)
    warm = read_checked(WARM, WARM_SHA)
    multistart = read_checked(MULTISTART, MULTISTART_SHA)
    triplet = read_checked(TRIPLET, TRIPLET_SHA)
    midpoint_multi = read_checked(MIDPOINT_MULTI, MIDPOINT_MULTI_SHA)
    consumer_path = Path(consumer.__file__)
    metric_path = Path(consumer.metric.__file__)
    consumer_sha = hashlib.sha256(consumer_path.read_bytes()).hexdigest()
    metric_sha = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    panel = consumer.consume(args.directory, args.report_sha256)
    result = bind_midpoint_multistart(
        join(prepared, zero, warm, panel, multistart, triplet), midpoint_multi)
    for path, digest in ((PREP, PREP_SHA), (ZERO, ZERO_SHA), (WARM, WARM_SHA),
                         (MULTISTART, MULTISTART_SHA), (TRIPLET, TRIPLET_SHA),
                         (MIDPOINT_MULTI, MIDPOINT_MULTI_SHA),
                         (consumer_path, consumer_sha), (metric_path, metric_sha)):
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "source drift")
    result.update(common_run_report_sha256=args.report_sha256, prepare_sha256=PREP_SHA,
                  zero_analysis_sha256=ZERO_SHA, warm_analysis_sha256=WARM_SHA,
                  multistart_analysis_sha256=MULTISTART_SHA,
                  triplet_analysis_sha256=TRIPLET_SHA,
                  midpoint_multistart_compare_sha256=MIDPOINT_MULTI_SHA,
                  consumer_sha256=consumer_sha, metric_sha256=metric_sha,
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "anchors": 24, "common_pairs": 336}))


if __name__ == "__main__":
    main()
