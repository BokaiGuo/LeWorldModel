"""Bind the new measured-resource-matched planner to prior identical actions.

No physics is executed here.  The consumer verifies that every new winner is
byte-identical to the already saved planner winner whose eight physical
trajectories were previously sealed, then reports those outcomes with explicit
known-root and no-new-physics boundaries.
"""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import p2_matched_program_host_20261101_parallel as host

NAME = "p2_matched_program_physical_reuse_20261101_parallel"
ROOT = host.ROOT
OLD_RUN = ROOT / "results/codex_p2_multistart_joint_host_20261030_v1_RUN_01"
OLD_SAVED = (
    ROOT / "results/codex_p2_multistart_joint_saved_20261030_v1_RECEIPT_01.json"
)
OLD_SAVED_SHA = "dddd5d3021c9263231cd0ade7e2ae14fd91490ad210bc33d3cdb9578c7dea8b9"
PHYSICAL_JOIN = (
    ROOT / "results/codex_p2_multistart_joint_physical_20261030_v1_JOIN_01.json"
)
PHYSICAL_JOIN_SHA = "f87186e10e18e977503dc25a44ab6b0f2405e142dbaa729fd2c4227d97414b78"
need = host.kernel.old.require


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def winner_action_hex(branch):
    winner = branch["winner"]
    if branch["arm"] == "joint_multistart":
        value = winner["raw_float32_hex"]
    elif branch["arm"] == "prefix_chain":
        value = (
            winner["first_raw_float32_hex"] + winner["second_raw_float32_hex"]
        )
    else:
        raise ValueError("unexpected arm")
    raw = bytes.fromhex(value)
    need(len(raw) == 80, "ten two-dimensional float32 actions")
    return value


def consume(new_run, new_report_sha, new_saved, new_saved_sha):
    new_run = Path(new_run)
    new_saved = Path(new_saved)
    need(
        new_run.is_absolute()
        and new_run.is_dir()
        and not new_run.is_symlink()
        and new_saved.is_absolute()
        and new_saved.is_file()
        and not new_saved.is_symlink(),
        "absolute saved inputs",
    )
    need(
        sha256(new_run / "report.json") == new_report_sha
        and sha256(new_saved) == new_saved_sha,
        "new external hashes",
    )
    need(
        sha256(OLD_SAVED) == OLD_SAVED_SHA
        and sha256(PHYSICAL_JOIN) == PHYSICAL_JOIN_SHA,
        "old planner and physical authority hashes",
    )
    saved = json.loads(new_saved.read_bytes())
    need(
        saved["schema"]
        == "p2_matched_program_saved_20261101_parallel.receipt"
        and saved["status"] == "saved_cuda_comparison_verified_not_scientific"
        and saved["report_sha256"] == new_report_sha
        and saved["measured_resource_match"] is True
        and saved["physical"] is False
        and saved["P2_admission"] is False,
        "new saved planner receipt",
    )
    old_saved = json.loads(OLD_SAVED.read_bytes())
    need(
        old_saved["schema"]
        == "codex_p2_multistart_joint_saved_20261030_v1.receipt"
        and old_saved["status"] == "complete_saved_not_scientific"
        and old_saved["physical"] is False
        and old_saved["P2_admission"] is False,
        "old saved planner receipt",
    )
    joined = json.loads(PHYSICAL_JOIN.read_bytes())
    need(
        joined["schema"]
        == "codex_p2_multistart_joint_physical_20261030_v1.joined_jobs"
        and joined["fixed_denominator"] == 8
        and joined["new_trajectories"] == 8
        and len(joined["policy_slots"]) == 8
        and joined["known_roots"] == 2
        and joined["fresh"] is False
        and joined["fresh_roots"] == 0
        and joined["physical_release"] is False
        and joined["P1_admission"] is False
        and joined["P2_admission"] is False
        and joined["P2_mechanism"] is False
        and joined["scientific_admission"] is False,
        "physical join boundary",
    )
    for authority in joined["job_authorities"]:
        path = Path(authority["path"])
        need(
            path.is_absolute()
            and path.is_file()
            and not path.is_symlink()
            and sha256(path) == authority["sha256"],
            "physical job authority",
        )

    slots = {
        (slot["base_index"], slot["policy"]): slot
        for slot in joined["policy_slots"]
    }
    need(
        len(slots) == 8
        and set(slots)
        == {
            (base, arm)
            for base in range(4)
            for arm in host.kernel.ARMS
        },
        "physical slot identities",
    )
    rows = []
    for index, (base, arm) in enumerate(host.kernel.contract()["order"]):
        new_branch_path = new_run / f"branch_{index}.json"
        old_branch_path = OLD_RUN / f"branch_{index}.json"
        need(
            new_branch_path.is_file()
            and old_branch_path.is_file()
            and not new_branch_path.is_symlink()
            and not old_branch_path.is_symlink(),
            "planner branch files",
        )
        new_branch = json.loads(new_branch_path.read_bytes())
        old_branch = json.loads(old_branch_path.read_bytes())
        need(
            (new_branch["base"], new_branch["arm"]) == (base, arm)
            and (old_branch["base"], old_branch["arm"]) == (base, arm),
            "planner branch identity",
        )
        new_action = winner_action_hex(new_branch)
        old_action = winner_action_hex(old_branch)
        need(new_action == old_action, "winner action bytes differ")
        slot = slots[(base, arm)]
        physical = slot["row"]
        need(
            slot["status"] == physical["status"] == "horizon_completed"
            and physical["own_attempted"] == physical["own_returned"] == 25
            and physical["policy"] == arm
            and physical["base_index"] == base
            and physical["final_metrics"]["native_success"] is False,
            "complete saved physical trajectory",
        )
        rows.append(
            dict(
                base=base,
                arm=arm,
                action_sha256=hashlib.sha256(bytes.fromhex(new_action)).hexdigest(),
                action_identity_verified=True,
                new_planner_winner_cost=new_branch["winner"]["cost"],
                physical_status=physical["status"],
                physical_steps=physical["own_returned"],
                final_metrics=physical["final_metrics"],
                root_slot=physical["root_slot"],
                task_id=physical["task_id"],
            )
        )

    pairs = []
    for base in range(4):
        pair = [row for row in rows if row["base"] == base]
        need(
            [row["arm"] for row in pair] == list(host.kernel.ARMS),
            "pair order",
        )
        joint, composition = pair
        pairs.append(
            dict(
                base=base,
                root_slot=joint["root_slot"],
                task_id=joint["task_id"],
                both_native_success=False,
                object_position_error_delta_composition_minus_joint=(
                    composition["final_metrics"]["object_position_error"]
                    - joint["final_metrics"]["object_position_error"]
                ),
                native_distance_delta_composition_minus_joint=(
                    composition["final_metrics"]["native_distance"]
                    - joint["final_metrics"]["native_distance"]
                ),
                angle_error_delta_composition_minus_joint=(
                    composition["final_metrics"]["wrapped_object_angle_error"]
                    - joint["final_metrics"]["wrapped_object_angle_error"]
                ),
            )
        )
    return dict(
        schema=NAME + ".receipt",
        status="saved_physical_reuse_verified_not_scientific",
        new_planner_saved_sha256=new_saved_sha,
        new_planner_report_sha256=new_report_sha,
        old_planner_saved_sha256=OLD_SAVED_SHA,
        physical_join_sha256=PHYSICAL_JOIN_SHA,
        rows=rows,
        pairs=pairs,
        measured_resource_match=True,
        action_identity_verified=True,
        reused_physical_trajectories=8,
        new_physical_calls=0,
        native_successes=0,
        known_roots=2,
        independent_confirmation=False,
        equal_total_compute=False,
        equal_flops=False,
        physical_release=False,
        P1_admission=False,
        P2_admission=False,
        P2_mechanism=False,
        scientific_admission=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run", required=True)
    parser.add_argument("--new-report-sha", required=True)
    parser.add_argument("--new-saved", required=True)
    parser.add_argument("--new-saved-sha", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    receipt = Path(args.receipt)
    need(
        receipt.is_absolute()
        and receipt.parent == ROOT / "results"
        and receipt.name.startswith(NAME + "_RECEIPT_"),
        "exclusive receipt",
    )
    value = consume(
        args.new_run,
        args.new_report_sha,
        args.new_saved,
        args.new_saved_sha,
    )
    print(
        json.dumps(
            dict(
                path=str(receipt),
                sha256=host.prep.original.previous.write_new(receipt, value),
                status=value["status"],
            )
        )
    )


if __name__ == "__main__":
    main()
