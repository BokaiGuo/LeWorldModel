"""Saved-only integrity audit of the existing matched P2 CP15 comparison.

No neural execution, environment construction, training, or gate update.
The original source verifiers are re-run before the additional paired checks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p2_matched_program_physical_reuse_20261101_parallel as reuse
from . import p2_matched_program_saved_20261101_parallel as saved

ROOT = reuse.ROOT
NEW_RUN = ROOT / "results/p2_matched_program_host_20261101_parallel_RUN_02"
PREP = ROOT / "results/p2_matched_program_host_20261101_parallel_PREP_01.json"
NEW_SAVED = ROOT / "results/p2_matched_program_saved_20261101_parallel_RECEIPT_01.json"
REUSE_SAVED = ROOT / "results/p2_matched_program_physical_reuse_20261101_parallel_RECEIPT_01.json"
PREP_SHA = "d7a1a93d6fdf208b1a965c9632dfabf081f3b0035676c0232d31766eb711adbe"
REPORT_SHA = "3db63a0eadd898283222774431157676bea698f7754ca22f0329663fb013b749"
SUPERVISOR_SHA = "52134e3f65e13c3b29c983ca2a8a08556c8ea429ed2b27aab11079713f8c1d35"
NEW_SAVED_SHA = "a364af8c5f632c8da564ef6c4389dc51764d76078124f788c881acb76323c6cd"
REUSE_SAVED_SHA = "39ad4cc27e51ea42db49e940924166a3bd8e0d9be9820ea61375c8b0c9015ca4"
NAME = "p2_cp15_pair_integrity_20261102_codexexclusive"
need = reuse.need


def inspect_pairs(joined, branches):
    """Cross-check fixed same-case CP15 inputs and each full physical plan.

    The caller must authenticate joined and branch files first. These checks
    inspect recorded commands and receipts, not a fresh physical execution.
    """
    slots = joined["policy_slots"]
    need(len(slots) == len(branches) == 8, "fixed eight-slot census")
    pairs = []
    aliases = defaultdict(list)
    for base in range(4):
        physical = slots[2 * base:2 * base + 2]
        models = branches[2 * base:2 * base + 2]
        comparable = []
        winner_hexes = []
        for slot, branch, arm in zip(physical, models, ("joint_multistart", "prefix_chain"), strict=True):
            row = slot["row"]
            plan = row["plan"]
            guard = row["feedback_guard"]
            need((slot["base_index"], slot["policy"], branch["base"], branch["arm"])
                 == (base, arm, base, arm), "fixed pair identities")
            need(row["initial_exact"] is True and row["warmup_returned"] == 10
                 and row["own_attempted"] == row["own_returned"] == 25
                 and row["status"] == "horizon_completed"
                 and row["terminated"] is False and row["truncated"] is False
                 and len(row["responses"]) == 25, "physical census")
            need(guard["base_index"] == base and guard["policy"] == arm
                 and guard["task_id"] == row["task_id"] and guard["root_slot"] == row["root_slot"]
                 and guard["history_controls"] == [5, 10, 15]
                 and guard["history_absolute_steps"] == [15, 20, 25]
                 and guard["no_state_restore"] is True
                 and guard["actual_observed_context_sha256"] == guard["compile_input_context_sha256"],
                 "observed CP15 feedback")
            need(saved.kernel.old.digest(plan["new_model_plan"]["input_context"]) == branch["context_sha256"]
                 and saved.kernel.old.digest(plan["planner_plan"]["goal"]) == branch["goal_sha256"],
                 "physical plan versus matched model context/goal")
            raw = plan["raw_float32_hex"]
            need(isinstance(raw, str) and len(raw) == 25 * 2 * 4 * 2, "physical command bytes")
            try:
                encoded = bytes.fromhex(raw)
                commands = np.asarray(plan["commands"], dtype="<f4")
            except (ValueError, TypeError) as exc:
                raise ValueError("physical command encoding") from exc
            need(commands.shape == (25, 2) and np.isfinite(commands).all()
                 and commands.tobytes() == encoded
                 and hashlib.sha256(encoded).hexdigest() == plan["raw_float32_sha256"],
                 "physical command byte identity")
            winner = reuse.winner_action_hex(branch)
            need(raw[:15 * 16] == guard["prefix_raw15_bits"]
                 and raw[15 * 16:] == winner, "CP15 prefix and winner suffix")
            comparable.append((row["root_sha256"], row["task_id"], row["root_slot"],
                               row["goal_pixel_sha256"], guard["actual_observed_context_sha256"],
                               guard["prefix_raw15_bits"], branch["context_sha256"], branch["goal_sha256"]))
            winner_hexes.append(winner)
        need(comparable[0] == comparable[1], "same root, goal, feedback and shared prefix")
        record = comparable[0]
        alias_key = (record, *winner_hexes)
        aliases[alias_key].append(base)
        pairs.append({"base": base, "root_sha256": record[0], "task_id": record[1],
                      "root_slot": record[2], "goal_pixel_sha256": record[3],
                      "cp15_observed_context_sha256": record[4],
                      "model_context_sha256": record[6], "model_goal_sha256": record[7],
                      "physical_plan_steps": 25, "shared_prefix_steps": 15,
                      "matched_planner_suffix_steps": 10,
                      "declared_plan_raw_steps": [p["row"]["plan"]["raw_steps"] for p in physical]})
    return {
        "schema": NAME + ".v1", "status": "saved_development_integrity_verified",
        "slots": len(pairs), "pairs": pairs,
        "distinct_known_roots": len({p["root_sha256"] for p in pairs}),
        "distinct_paired_contexts": len({(p["root_sha256"], p["model_context_sha256"],
                                           p["model_goal_sha256"]) for p in pairs}),
        "duplicate_bases": [items for items in aliases.values() if len(items) > 1],
        "raw_steps_field_matches_saved_commands": all(
            all(declared == pair["physical_plan_steps"] for declared in pair["declared_plan_raw_steps"])
            for pair in pairs),
        "physical_plan_suffix_matches_winner": True,
        "action_execution_proof": "inherited saved physical receipts; not independently observed",
        "measured_resource_scope": "existing eight recorded axes; not total compute",
        "equal_total_compute": False, "independent_confirmation": False,
        "P1_admission": False, "P2_admission": False, "scientific_admission": False,
        "new_neural_calls": 0, "new_physical_steps": 0,
    }


def audit_saved():
    """Re-run both existing saved consumers and bind their sealed receipts."""
    need(reuse.sha256(NEW_SAVED) == NEW_SAVED_SHA
         and reuse.sha256(REUSE_SAVED) == REUSE_SAVED_SHA, "saved receipt SHA")
    model_receipt = saved.consume(NEW_RUN, REPORT_SHA, SUPERVISOR_SHA, PREP, PREP_SHA)
    need(model_receipt == json.loads(NEW_SAVED.read_bytes()), "reconstructed model receipt")
    physical_receipt = reuse.consume(NEW_RUN, REPORT_SHA, NEW_SAVED, NEW_SAVED_SHA)
    need(physical_receipt == json.loads(REUSE_SAVED.read_bytes()), "reconstructed reuse receipt")
    joined = json.loads(reuse.PHYSICAL_JOIN.read_bytes())
    branches = [json.loads((NEW_RUN / f"branch_{i}.json").read_bytes()) for i in range(8)]
    result = inspect_pairs(joined, branches)
    result["source_receipt_sha256"] = {"model": NEW_SAVED_SHA, "physical_reuse": REUSE_SAVED_SHA,
                                        "physical_join": reuse.PHYSICAL_JOIN_SHA}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit_saved(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
