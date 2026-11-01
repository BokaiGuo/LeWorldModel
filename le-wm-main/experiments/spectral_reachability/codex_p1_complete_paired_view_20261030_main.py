"""Join sealed P1 physical evidence; descriptive only, not P1 admission."""
import argparse
import json
from pathlib import Path

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap

NAME = "codex_p1_complete_paired_view_20261030_main"
TABLE = base.ROOT / "results/codex_p1_h10_paired_table_20261030_main_FULL01.json"
TABLE_SHA = "4406169f802bf40912f9ff384d6d704b645d2d6a29ddf6f19936af32d4f3ef24"
JOIN = base.ROOT / "results/codex_p1_h10_independent_join_20261030_v1_CODEX02.json"
JOIN_SHA = "bd55f5f2e07603de0a3b3d27587b942e58256e598ae7dcf5a1d8b97150ae52b5"
FIELDS = ("object_position_error", "object_angle_error_wrapped",
          "pusher_position_error", "native_joint_xy_error", "native_angle_term")
IDENTITY = ("logical_pair", "root_slot", "condition", "arm", "role")


def delta(metrics, baseline):
    if metrics is None or baseline is None:
        return None
    return {k: metrics[k] - baseline[k] for k in FIELDS}


def build(table, joined, planned):
    if (len(table["rows"]) != 1008 or len(joined["rows"]) != 24
            or len(planned["rows"]) != 24 or planned["status"] != "24_pairs_saved_verified"
            or joined["status"] != "source_bound_join_not_new_h10_verification"):
        raise ValueError("incomplete source")
    result = []
    for i, (anchor, own) in enumerate(zip(joined["rows"], planned["rows"], strict=True)):
        ident = anchor["identity"]
        if ident["index"] != i or own["index"] != i:
            raise ValueError("anchor index")
        for key in (*IDENTITY, "task_id", "physical_root_sha256", "prefix5_float32_hex",
                    "cp5_pixel_sha256", "cp5_file_sha256"):
            if ident[key] != own[key]:
                raise ValueError("anchor identity: " + key)
        common = []
        for j, item in enumerate(anchor["common_suffixes"]):
            source = table["rows"][ident["logical_pair"] * 126 + j * 3 + i % 3]
            if (any(source[k] != ident[k] for k in IDENTITY)
                    or source["probe"] != item["probe"]
                    or source["probe_sha256"] != item["raw10_sha256"]
                    or source["status"] != item["status"]
                    or source["observation"] != item["observation"]
                    or source["common_suffix_endpoint_native_metrics"] != item["endpoint_metrics"]):
                raise ValueError("common action or identity mismatch")
            common.append({k: source[k] for k in (
                "probe", "probe_sha256", "status", "observation",
                "common_suffix_endpoint_native_metrics", "same_prefix_endpoint_minus_own_zero_suffix",
                "same_suffix_endpoint_minus_zero", "same_suffix_endpoint_minus_greedy",
                "response_vs_own_zero_suffix", "response_vs_own_zero_suffix_minus_zero_prefix")})
        if len(common) != 42 or common[0]["probe"] != "zero":
            raise ValueError("fixed probe roster")
        prefix_row = table["rows"][ident["logical_pair"] * 126 + i % 3]
        two = own["two_step"]
        if (two["raw10_sha256"] != anchor["independent_suffix"]["raw10_sha256"]
                or two["horizon_metrics"] != anchor["independent_suffix"]["endpoint_metrics"]
                or two["status"] != anchor["independent_suffix"]["status"]):
            raise ValueError("independent two-step mismatch")
        zero_endpoint = common[0]["common_suffix_endpoint_native_metrics"]
        independent = {}
        for label in ("two_step", "twenty_step"):
            run = own[label]
            independent[label] = {k: run[k] for k in (
                "status", "accepted_suffix_steps", "censored", "early_native_stop",
                "native_stop_at_horizon", "raw10_sha256", "horizon_metrics", "planning_resources")}
            independent[label]["endpoint_minus_own_zero_suffix"] = delta(run["horizon_metrics"], zero_endpoint)
        result.append({"identity": ident, "duplicate_group": prefix_row["duplicate_group"],
                       "root_metrics": prefix_row["root_metrics"], "prefix_metrics": prefix_row["prefix_metrics"],
                       "prefix_change": {"object_position_error": prefix_row["prefix_position_error_change"],
                                         "object_angle_error_wrapped": prefix_row["prefix_angle_error_change"],
                                         "pusher_position_error": prefix_row["prefix_pusher_position_error_change"],
                                         "native_joint_xy_error": prefix_row["prefix_joint_xy_error_change"],
                                         "native_angle_term": prefix_row["prefix_native_angle_term_change"],
                                         "native_success_transition": prefix_row["prefix_native_success_transition"]},
                       "common_suffixes": common, "independent_suffixes": independent})
    for row in result:
        pair = row["identity"]["logical_pair"]
        zero = result[3 * pair + 1]
        if zero["identity"]["role"] != "zero":
            raise ValueError("zero identity")
        for label in ("two_step", "twenty_step"):
            own = row["independent_suffixes"][label]
            own["endpoint_minus_zero_prefix_independent_endpoint"] = delta(
                own["horizon_metrics"], zero["independent_suffixes"][label]["horizon_metrics"])
    return {"schema": NAME, "status": "descriptive_paired_view", "rows": result,
            "denominators": {"prefixes": 24, "common_suffixes": 1008,
                             "independent_suffixes_per_budget": 24},
            "sources": {str(TABLE): TABLE_SHA, str(JOIN): JOIN_SHA, str(gap.PAIRS): gap.PAIR_SHA},
            "independent_confirmation": False, "equal_total_budget_certified": False, "P1_admission": False,
            "limitations": ["two known roots and duplicate trajectories",
                            "independent suffixes once per prefix, not once per shared probe",
                            "two-step and twenty-step budgets differ",
                            "physical and planning resource accounting incomplete",
                            "native stopped common endpoints remain null per sealed contract",
                            "zero suffix and independent suffix differ in computational cost",
                            "post-outcome descriptive view does not establish mechanism or selection"] }


def run(output):
    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    table, _ = gap.read(TABLE, TABLE_SHA)
    joined, _ = gap.read(JOIN, JOIN_SHA)
    planned, _ = gap.read(gap.PAIRS, gap.PAIR_SHA)
    value = build(table, joined, planned)
    value["code_sha256"] = base.sha(Path(__file__).resolve())
    print(json.dumps({"status": value["status"], "sha256": base.write(output, value),
                      "denominators": value["denominators"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Join fixed common probes with independently optimized suffixes, without imputing stops.", flush=True)
    run(args.output)
