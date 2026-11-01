from experiments.spectral_reachability import p1_equal_candidate_q_20261101_main as variant


def test_allocation_preserves_extra_arm_gradient_budget(monkeypatch):
    control = variant.parent.carry.control
    monkeypatch.setattr(control, "EXTRA_ARMS", ("structured", "native", "random"))
    monkeypatch.setattr(control.base, "CONFIG", dict(control.base.CONFIG, steps=2))
    old = {arm: control.cost(arm) for arm in control.spec.ARMS}
    variant.allocation()
    assert control.EXTRA_ARMS == ()
    for arm in ("structured", "native", "random"):
        new = control.cost(arm)
        assert new["backward_calls"] == old[arm]["backward_calls"]
        assert new["vjp_rows"] == old[arm]["vjp_rows"]
        assert new["predict_calls"] == old[arm]["predict_calls"] - 3
        assert new["replay_calls"] == old[arm]["replay_calls"] - 1
    assert control.cost("immediate_progress")["backward_calls"] == 48
