"""Conservative model-only recheck of an already proposed witness.

This is NOT a full-bank replacement selector: compare the fixed old witness
to its exact zero/greedy controls using one consistently refined suffix.
No physical successor, qpos, or outcome is an input to the decision.
"""
from __future__ import annotations

import math


def refine_witness_decision(cells, *, had_witness, non_greedy_margin=1e-6):
    if set(cells) != {"dispatch", "same_bank_greedy", "zero"} or type(had_witness) is not bool:
        raise ValueError("three complete model controls and explicit witness boolean required")
    if not math.isfinite(non_greedy_margin) or non_greedy_margin < 0:
        raise ValueError("finite nonnegative margin required")
    if any(not math.isfinite(float(cell[key])) for cell in cells.values()
           for key in ("immediate_goal_cost", "refined_goal_cost")):
        raise ValueError("finite immediate and refined model costs required")
    candidate, greedy, zero = (cells[name] for name in ("dispatch", "same_bank_greedy", "zero"))
    non_greedy = candidate["immediate_goal_cost"] >= zero["immediate_goal_cost"] + non_greedy_margin
    beats_zero = candidate["refined_goal_cost"] < zero["refined_goal_cost"]
    beats_greedy = candidate["refined_goal_cost"] < greedy["refined_goal_cost"]
    retained = had_witness and non_greedy and beats_zero and beats_greedy
    return {"policy": "fixed_old_witness_rechecked_by_common_fine8_suffix_v1",
        "selected_branch": "dispatch" if retained else "same_bank_greedy",
        "old_witness": had_witness, "witness_retained": retained, "model_non_greedy": non_greedy,
        "beats_refined_zero": beats_zero, "beats_refined_greedy": beats_greedy,
        "physical_labels_used": False, "candidate_set": "fixed old witness and zero/greedy controls, not full bank",
        "refined_greedy_gain": greedy["refined_goal_cost"] - candidate["refined_goal_cost"]}
