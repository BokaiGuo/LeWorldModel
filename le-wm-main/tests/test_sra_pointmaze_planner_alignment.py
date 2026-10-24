import inspect

from experiments.spectral_reachability import audit_sra_pointmaze_planner_alignment_cuda as audit


def test_alignment_audit_checks_the_three_high_risk_equivalences():
    source = inspect.getsource(audit)
    assert "_p1_tree_equivalence" in source
    assert "_model_batch_serial_error" in source
    assert "_model_chart_composition_error" in source
    assert "require_cuda()" in source
    assert "do not interpret P1/P2 effects" in source
    assert "observed_re_normalized" in source
    assert "p1_model_batch_equals_serial_le_5e-6" in source


def test_alignment_audit_uses_receipt_bound_endpoint_evaluator():
    source = inspect.getsource(audit._p1_tree_equivalence)
    assert "replay_position_endpoints" in source
