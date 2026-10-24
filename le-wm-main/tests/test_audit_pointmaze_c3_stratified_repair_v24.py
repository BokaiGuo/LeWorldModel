from __future__ import annotations

import pytest

from experiments.decision_benchmark import audit_pointmaze_c3_stratified_repair_v24 as audit


def test_v24_audit_rejects_nominal_roots_that_repeat_one_observable_context() -> None:
    """V24's archived panel is invalid after the root-identity correction.

    A former test re-promoted this panel's numerical summaries even though its
    paired source roots duplicated the same observable state/history/goal.
    Root IDs alone are not independent calibration or outcome observations.
    """
    with pytest.raises(ValueError, match="duplicate one observable context"):
        audit.audit()
