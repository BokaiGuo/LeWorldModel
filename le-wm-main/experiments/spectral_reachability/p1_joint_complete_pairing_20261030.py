"""Join the complete verified common-suffix panel to 24 independent plans."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from . import p1_joint_anchor_table_20261030 as anchor_table

ROOT = anchor_table.ROOT
ANCHOR_PATH = ROOT / "results/p1_joint_anchor_table_20261030_RUN01.json"
ANCHOR_SHA = "86dd8f98beb5f4f5f7553b3479d9413d7049b66d2bc81524d34e6f8cb2333840"
PREP_PATH = ROOT / anchor_table.FILES["h10_prepare"][0]
PREP_SHA = anchor_table.FILES["h10_prepare"][1]
ROLES = ("selected", "zero", "greedy")
NUMERIC = ("object_position_error", "object_angle_error_wrapped", "pusher_position_error",
           "native_joint_xy_error", "native_angle_term")
require = anchor_table.require


def read_checked(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "source hash: " + str(path))
    return json.loads(raw)


def same_metrics(a, b):
    return (a is not None and b is not None and
            all(math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-9) for k in NUMERIC)
            and a["native_success"] == b["native_success"])


def difference(a, b):
    if a is None or b is None:
        return None
    return {key: a[key] - b[key] for key in NUMERIC}


def join(prepared, anchors, panel):
    require(panel["schema"] == "p1_joint_h10_consume_20261030"
            and panel["prepare_sha256"] == PREP_SHA
            and panel["P1_admission"] is False, "panel contract")
    require(len(prepared["anchors"]) == len(anchors["rows"]) == 24
            and len(panel["rows"]) == 1008, "complete roster")
    role_anchors = {}
    for i, (source, row) in enumerate(zip(prepared["anchors"], anchors["rows"], strict=True)):
        key = (source["logical_pair"], source["role"])
        require(source["index"] == row["index"] == i and key not in role_anchors, "anchor index")
        require(all(source[k] == row[k] for k in
                    ("logical_pair", "root_slot", "condition", "arm", "role", "task_id",
                     "physical_root_sha256")), "anchor identity")
        require(source["prefix5_sha256"] == row["prefix_action_sha256"]
                and source["checkpoint"]["pixel_sha256"] == row["prefix_cp5_pixel_sha256"],
                "anchor prefix identity")
        role_anchors[key] = row
    panel_rows = {}
    for row in panel["rows"]:
        key = (row["logical_pair"], row["probe"], row["role"])
        require(key not in panel_rows, "duplicate panel row")
        panel_rows[key] = row
    paired = []
    for logical in range(8):
        bank = prepared["probe_banks"][logical]
        require(len(bank) == 42 and len({p["name"] for p in bank}) == 42, "probe roster")
        for probe in bank:
            roles = {}
            for role in ROLES:
                anchor = role_anchors[(logical, role)]
                row = panel_rows.pop((logical, probe["name"], role))
                require(all(row[k] == anchor[k] for k in ("root_slot", "condition", "arm", "role"))
                        and row["probe_sha256"] == probe["raw10_sha256"], "panel identity")
                require(same_metrics(row["root"], anchor["root_metrics"])
                        and same_metrics(row["prefix"], anchor["prefix_metrics"]), "prefix metric binding")
                require((row["endpoint"] is not None) == (row["status"] == "probe_completed"),
                        "common endpoint censoring")
                roles[role] = {"anchor_index": anchor["index"], "status": row["status"],
                               "endpoint": row["endpoint"],
                               "endpoint_minus_own_zero_suffix": row["endpoint_minus_own_zero"]}
            for role in ROLES:
                roles[role]["same_suffix_endpoint_minus_zero_prefix"] = difference(
                    roles[role]["endpoint"], roles["zero"]["endpoint"])
                roles[role]["same_suffix_endpoint_minus_greedy_prefix"] = difference(
                    roles[role]["endpoint"], roles["greedy"]["endpoint"])
                roles[role]["response_minus_zero_prefix_response"] = difference(
                    roles[role]["endpoint_minus_own_zero_suffix"],
                    roles["zero"]["endpoint_minus_own_zero_suffix"])
            paired.append({"logical_pair": logical, "probe": probe["name"],
                           "probe_sha256": probe["raw10_sha256"], "roles": roles})
    require(not panel_rows and len(paired) == 336, "unconsumed panel rows")
    return {"schema": "p1_joint_complete_pairing_20261030",
            "anchors": anchors["rows"], "common_suffix_pairs": paired,
            "independent_suffix_observations": 24, "common_suffix_branches": 1008,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Known roots and duplicate trajectories. Independent plans are stored once, "
                           "not repeated as independent probe observations. Missing endpoints are not imputed."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith("p1_joint_complete_pairing_") and not output.exists()
            and not output.is_symlink(), "exclusive output")
    prepared = read_checked(PREP_PATH, PREP_SHA)
    anchors = read_checked(ANCHOR_PATH, ANCHOR_SHA)
    # Re-run the sealed physical and action verification instead of trusting a loose JSON table.
    from . import p1_joint_h10_consume_20261030 as consumer
    panel = consumer.consume(args.directory, args.report_sha256, PREP_PATH, PREP_SHA)
    result = join(prepared, anchors, panel)
    result.update(common_run_report_sha256=args.report_sha256, prepare_sha256=PREP_SHA,
                  anchor_table_sha256=ANCHOR_SHA,
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "anchors": 24, "common_pairs": 336}))


if __name__ == "__main__":
    main()
