"""Strict input wrapper for the historically sealed scalar admission module.

Existing runners do not use this wrapper unless explicitly integrated and
source-bound. The legacy module is left byte-identical for historical replay.
"""
from . import d2p_stage_admission as legacy

P0_INTEGRITY_KEYS = legacy.P0_INTEGRITY_KEYS
P1_INTEGRITY_KEYS = legacy.P1_INTEGRITY_KEYS
P1_MECHANISM_KEYS = legacy.P1_MECHANISM_KEYS


def p1_admission(comparison, *, calibration_only, implementation_gates):
    if calibration_only is not False:
        return False, "explicit non-calibration evidence required to admit P1"
    return legacy.p1_admission(comparison, calibration_only=False,
                               implementation_gates=implementation_gates)


def p2_admission(comparisons, *, calibration_only, eligible_roots, min_eligible_roots,
                 implementation_gates, required_comparisons, mechanism_gate):
    if (calibration_only is not False or type(eligible_roots) is not int
            or type(min_eligible_roots) is not int):
        return False, "explicit non-calibration and integer root counts required"
    if (not isinstance(required_comparisons, (tuple, list, set, frozenset))
            or not all(isinstance(name, str) and name for name in required_comparisons)):
        return False, "invalid predeclared comparison roster"
    return legacy.p2_admission(comparisons, calibration_only=False, eligible_roots=eligible_roots,
                               min_eligible_roots=min_eligible_roots,
                               implementation_gates=implementation_gates,
                               required_comparisons=required_comparisons,
                               mechanism_gate=mechanism_gate)
