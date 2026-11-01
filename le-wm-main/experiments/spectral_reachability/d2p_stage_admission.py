"""Fail-closed P0/P1 stage evidence; no neural model or physical rerouting."""
from __future__ import annotations

import math

P0_INTEGRITY_KEYS = (
    "cuda_only", "frozen_parameter_hash_unchanged", "d2p_equals_gauss_newton_each_root",
    "all_physical_outcomes_finite", "planner_and_execution_root_rgb_identity_verified",
    "all_executed_raw_actions_within_environment_box",
    "actual_planning_predict_samples_matched_to_cem",
)

P1_INTEGRITY_KEYS = (
    "cuda_only", "upstream_p0_admission_verified", "p1_plumbing_preflight_verified",
    "frozen_parameter_hash_unchanged", "at_least_one_model_eligible_reconfiguration_root",
    "at_least_declared_eligible_calibration_roots",
    "every_eligible_reconfiguration_is_model_non_greedy",
    "every_eligible_reconfiguration_improves_declared_future_score",
    "every_ineligible_root_uses_declared_neutral_prefix",
    "all_methods_execute_exactly_three_action_tokens", "all_raw_actions_within_environment_box",
    "all_physical_outcomes_finite", "all_execution_root_identities_verified",
)

P1_MECHANISM_KEYS = (
    "every_model_eligible_reconfiguration_is_physically_non_greedy_relative_to_zero_prefix",
    "every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix",
    "every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_same_bank_greedy",
)


def favorable_interval(comparison):
    if not isinstance(comparison, dict):
        return False, "missing independent paired P0 outcome comparison"
    interval = comparison.get("ci95")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return False, "missing two-sided paired interval"
    try:
        lower, upper = map(float, interval)
    except (TypeError, ValueError, OverflowError):
        return False, "invalid paired interval"
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return False, "nonfinite or unordered paired interval"
    if upper < 0:
        return True, "independent paired interval strictly favors D2P"
    return False, "paired interval does not strictly favor D2P; ties and zero-touching intervals cannot admit later stages"


def p1_admission(comparison, *, calibration_only, implementation_gates):
    if calibration_only:
        return False, "calibration-only evidence cannot admit P1"
    if not isinstance(implementation_gates, dict) or any(implementation_gates.get(key) is not True for key in P0_INTEGRITY_KEYS):
        return False, "missing or failed P0 implementation integrity gate"
    return favorable_interval(comparison)


def p0_predict_samples_matched(rows) -> bool:
    """Validate actual D2P/CEM planning predict samples on every outcome root.

    This is NOT total-compute matching: encoder work, backward work, wall time,
    FLOPs and post-execution diagnostics are outside these planning receipts.
    Calibration rows are excluded; malformed/unknown splits fail closed.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        return False
    seeds, roots, indices = set(), set(), set()
    horizons = set()
    outcomes = 0

    def positive_int(value):
        return type(value) is int and value > 0

    def sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    def error_free(value):
        return all(value.get(key) in (None, False, "", [], {})
                   for key in ("error", "errors", "exception"))

    try:
        for row in rows:
            if not isinstance(row, dict) or row.get("split") not in ("calibration", "outcome"):
                return False
            seed, index = row["seed"], row["root_index"]
            root = row["executed_context"]["root_rgb_sha256"]
            goal = row["executed_context"]["goal_rgb_sha256"]
            if (type(seed) is not int or type(index) is not int or index < 0 or
                    not sha(root) or not sha(goal) or seed in seeds or root in roots or index in indices):
                return False
            seeds.add(seed)
            roots.add(root)
            indices.add(index)
            if row["split"] == "calibration":
                continue
            outcomes += 1
            samples = []
            for method in ("d2p_goal_progress", "cem"):
                result = row["methods"][method]
                if not error_free(row) or not error_free(result) or result.get("completed", True) is not True:
                    return False
                identity = result["execution_identity"]
                if identity["reset_rgb_sha256"] != root or identity["goal_rgb_sha256"] != goal:
                    return False
                receipt = result["receipt"]
                if (not error_free(receipt) or receipt.get("completed", True) is not True or
                        receipt.get("predict_accounting_scope") != "planner_only_excludes_post_execution_diagnostic_encoder_and_backward"):
                    return False
                actual = receipt["actual_planning_predict"]
                if (actual.get("phase") != "planning" or actual.get("method") != method or
                        actual.get("completed") is not True or
                        type(actual.get("failed_predict_calls")) is not int or actual["failed_predict_calls"] != 0 or
                        not positive_int(actual.get("horizon_tokens")) or
                        not positive_int(actual.get("predict_calls")) or
                        not positive_int(actual.get("predict_samples")) or
                        actual["predict_samples"] < actual["predict_calls"]):
                    return False
                if not error_free(actual):
                    return False
                histogram = actual.get("context_lengths")
                if not isinstance(histogram, dict) or not histogram:
                    return False
                contexts = set()
                for context, calls in histogram.items():
                    # JSON converts integer histogram keys to decimal strings.
                    if type(context) is int and context > 0:
                        normalized = context
                    elif isinstance(context, str) and context.isascii() and context.isdecimal() and str(int(context)) == context and int(context) > 0:
                        normalized = int(context)
                    else:
                        return False
                    if normalized in contexts or not positive_int(calls):
                        return False
                    contexts.add(normalized)
                if sum(histogram.values()) != actual["predict_calls"]:
                    return False
                horizons.add(actual["horizon_tokens"])
                samples.append(actual["predict_samples"])
            if samples[0] != samples[1]:
                return False
    except (KeyError, TypeError, AttributeError):
        return False
    return outcomes > 0 and len(horizons) == 1


def validate_p0_admission_report(report):
    """Do not let a historical affirmative flag bypass current scalar evidence."""
    if not isinstance(report, dict):
        raise ValueError("P0 report must be a mapping")
    protocol = report.get("protocol", {})
    comparisons = report.get("comparisons_outcome_only", {})
    if not isinstance(protocol, dict) or not isinstance(report.get("stage_gate"), dict):
        raise ValueError("P0 protocol and stage gate must be mappings")
    if report.get("schema") != "d2p_reacher_p0_goal_progress.v1" or report.get("stage_gate", {}).get("p1_unlocked") is not True:
        raise ValueError("affirmative P0 stage flag and exact report schema required")
    if not p0_predict_samples_matched(report.get("rows")):
        raise ValueError("upstream P0 actual planning predict samples are not verified matched to CEM; not a total-compute claim")
    admitted, reason = p1_admission(comparisons.get("cem") if isinstance(comparisons, dict) else None,
        calibration_only=protocol.get("calibration_only") is not False or report.get("status") == "calibration_only",
        implementation_gates=report.get("gates"))
    if not admitted or not all(value is True for value in report["gates"].values()):
        raise ValueError("upstream P0 fails strict evidence revalidation: " + reason)
    return reason


def p2_admission(comparisons, *, calibration_only, eligible_roots, min_eligible_roots,
                 implementation_gates, required_comparisons, mechanism_gate):
    if calibration_only or min_eligible_roots < 2 or eligible_roots < min_eligible_roots:
        return False, "independent outcome and declared minimum eligible roots required"
    if mechanism_gate not in P1_MECHANISM_KEYS:
        return False, "unknown P1 mechanism contract"
    gates = implementation_gates
    if not isinstance(gates, dict) or any(gates.get(key) is not True for key in (*P1_INTEGRITY_KEYS, mechanism_gate)) or not all(value is True for value in gates.values()):
        return False, "missing or failed P1 mechanism/implementation gate"
    mandatory = {"zero_prefix_two_stage_d2p", "greedy_two_stage_d2p", "cem_full_horizon"}
    if not mandatory <= set(required_comparisons) or not isinstance(comparisons, dict) or not set(required_comparisons) <= set(comparisons):
        return False, "all predeclared P1 outcome controls required"
    if not all(favorable_interval(comparison)[0] for comparison in comparisons.values()):
        return False, "every P1 paired interval must strictly favor reconfiguration; ties cannot admit P2"
    return True, "all independent P1 outcome intervals strictly favorable and integrity gates pass"
