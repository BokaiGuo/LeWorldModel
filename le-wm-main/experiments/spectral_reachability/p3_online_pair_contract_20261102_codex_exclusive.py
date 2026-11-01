"""Prospective paired-slot contract for the existing metered P3 host.

This is a saved-semantics composition, not a launcher or a new experiment.
It never treats agreement of saved model arrays as independent model proof.
"""
from __future__ import annotations

from . import codex_p3_resolved_kernel_20261030_v1 as producer
from . import codex_p3_resolved_saved_20261030_v1 as consumer


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_verified_slots(signal, reverse, point):
    """Compare independently verified slots, excluding all outcome values."""
    _require(type(point) is int and point in producer.POINTS, "point")
    rows = []
    for result, arm in ((signal, "signal"), (reverse, "reverse")):
        _require(result.get("status") == "semantic_saved_checked_not_scientifically_admitted"
                 and result.get("scientific_pass") is False
                 and result.get("physical_certified") is False
                 and result.get("model_arrays_independently_recomputed") is False
                 and len(result.get("rows", [])) == 1, "verified single-slot result")
        row = result["rows"][0]
        _require(row["point"] == point and row["arm"] == arm, "slot identity")
        _require(row["allocation"] == producer.allocation(point, arm, row["weights"]),
                 "allocation does not follow online signal")
        _require(set(row["attempted"]) == set(producer.AXES)
                 and all(type(x) is int and x >= 0 for x in row["attempted"].values()),
                 "complete resource vector")
        rows.append(row)
    a, b = rows
    same_signal = a["weights"] == b["weights"]
    equal_axes = a["attempted"] == b["attempted"]
    distinct = a["allocation"] != b["allocation"]
    if not same_signal:
        classification = "unpaired_signal_mismatch"
    elif not distinct:
        classification = "degenerate_equal_allocation"
    else:
        classification = "paired_distinct_allocation"
    return {
        "schema": "p3_online_pair_contract_20261102_codex_exclusive.v1",
        "point": point,
        "classification": classification,
        "weights": {"signal": a["weights"], "reverse": b["weights"]},
        "allocations": {"signal": a["allocation"], "reverse": b["allocation"]},
        "same_observed_signal": same_signal,
        "distinct_allocation": distinct,
        "equal_recorded_axes": equal_axes,
        "attempted_axes": {"signal": a["attempted"], "reverse": b["attempted"]},
        "scope": "saved semantic verification; no independent model-array recomputation",
        "same_task_and_bank_required": True,
        "task_and_bank_provenance": "caller must verify host prepare and identities",
        "equal_total_compute": False,
        "outcome_comparison": False,
        "P3_admission": False,
        "C3_admission": False,
    }


def verify_pair(signal_events, reverse_events, tasks, banks, point):
    """Apply the existing strict per-slot consumer before comparing the pair.

    Both calls receive the same tasks and banks. Their upstream identities,
    host reports, journal integrity, and model identity remain caller duties.
    """
    signal = consumer.verify_events(signal_events, tasks, banks, point, "signal")
    reverse = consumer.verify_events(reverse_events, tasks, banks, point, "reverse")
    return compare_verified_slots(signal, reverse, point)


def preparation_contract():
    """Pre-run requirements only; does not authorize or initialize hardware."""
    return {
        "schema": "p3_online_pair_contract_20261102_codex_exclusive.prepare.v1",
        "producer": producer.__name__,
        "consumer": consumer.__name__,
        "points": [20, 30],
        "arms": ["signal", "reverse"],
        "requirements": [
            "Freeze source, checkpoint, task, candidate-bank and precision identities before execution.",
            "Use existing host admission, per-arm independent signal acquisition, meter and durable journal.",
            "Verify report and journal integrity, source identity and complete slot with existing host consumer.",
            "Require identical task and bank identities across slots; preserve failures without retry or padding.",
            "Classify unequal signals as unpaired and equal allocations as degenerate; do not select a budget point.",
            "Precommit independent fresh roots, resource comparison and physical endpoint contract separately.",
        ],
        "host_execution_authorized": False,
        "physical_release": False,
        "P3_admission": False,
        "C3_admission": False,
    }
