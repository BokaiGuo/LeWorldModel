from __future__ import annotations

import inspect

from experiments.gdd_wm import audit_pointmaze_mixed_velocity_repairability_v20_outcome as audit


def test_v20_audit_requires_the_equal_budget_inverse_control() -> None:
    assert audit.POLICY["direction"] == "recurrent_when_high"
    assert audit.INVERSE_CONTROL_POLICY["direction"] == "recurrent_when_low"
    source = inspect.getsource(audit.audit)
    assert "comparison_policy=INVERSE_CONTROL_POLICY" in source
    assert "exact 50 percent recurrent budget" in source
    assert "route_beats_equal_budget_control" in source


def test_v20_audit_refuses_to_overwrite_its_sealed_artifact() -> None:
    source = inspect.getsource(audit.main)
    assert "refusing to overwrite immutable V20 outcome audit" in source
