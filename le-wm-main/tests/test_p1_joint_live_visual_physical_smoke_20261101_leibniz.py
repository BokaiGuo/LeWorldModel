import pytest

from experiments.spectral_reachability import p1_joint_live_visual_physical_smoke_20261101_leibniz as s


def rows():
    return [{"step": i, "action": [0.0, 0.0], "actual_action": [0.0, 0.0],
             "terminated": False, "truncated": False} for i in range(1, 6)]


def test_five_actual_steps_required():
    s.validate_actions(rows())
    with pytest.raises(ValueError, match="five actual"):
        s.validate_actions(rows()[:-1])
    changed = rows()
    changed[1]["actual_action"] = [0.1, 0.0]
    with pytest.raises(ValueError, match="five actual"):
        s.validate_actions(changed)
    changed = rows()
    changed[-1]["terminated"] = True
    with pytest.raises(ValueError, match="five actual"):
        s.validate_actions(changed)


def test_fixed_budget_and_no_fake_boundary():
    assert s.PCAPS["public_step"] == 5 and s.PCAPS["physics_step"] == 52
    assert s.TCAPS["source_encode_calls"] == s.TCAPS["source_encode_samples"] == 1
    assert s.TCAPS["model_load"] == 1
    assert all(s.TCAPS[k] == 0 for k in ("predict_calls", "backward_calls", "candidate_evaluations"))


def test_existing_full_host_gate_unchanged():
    with pytest.raises(ValueError, match="fail-closed"):
        s.combined.run_full_host()
