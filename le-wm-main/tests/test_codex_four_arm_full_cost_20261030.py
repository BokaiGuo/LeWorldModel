import copy

import pytest

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as m


def contract():
    identity = {k: "a" * 64 for k in m.IDENTITY}
    arms = {arm: {"initializer_sha256": str(i + 1) * 64,
                  "optimizer_sha256": identity["optimizer"],
                  "optimizer_config_sha256": identity["optimizer_config"]}
            for i, arm in enumerate(m.ARMS)}
    return {"schema": m.NAME + ".contract.v1", "identity": identity, "arms": arms,
            "scope": "standalone_per_root_per_arm_including_attributable_setup",
            "phase_policy_sha256": "b" * 64,
            "levels": [{"id": "small", "caps": {k: 2 for k in m.AXES}},
                       {"id": "large", "caps": {k: 4 for k in m.AXES}}]}


def charge(**kw):
    return {k: kw.get(k, 0) for k in m.AXES}


def ledger():
    c = contract()
    return m.Ledger(c, m.digest(c), "structured", "small", "run")


def test_contract_curve_and_common_optimizer():
    c = contract()
    result = m.preflight(c, m.digest(c))
    assert result["levels"] == ["small", "large"]
    assert not result["fairness_proven"] and not result["execution_authorized"]
    c["arms"]["native"]["optimizer_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="common optimizer"):
        m.preflight(c, m.digest(c))


@pytest.mark.parametrize("mutation", ["missing_arm", "wrong_digest", "decrease", "same", "missing_axis", "boolean"])
def test_reject_bad_contract(mutation):
    c = contract()
    sha = m.digest(c)
    if mutation == "missing_arm":
        del c["arms"]["native"]
    elif mutation == "wrong_digest":
        sha = "0" * 64
    elif mutation == "decrease":
        c["levels"][1]["caps"][m.AXES[0]] = 1
    elif mutation == "same":
        c["levels"][1]["caps"] = copy.deepcopy(c["levels"][0]["caps"])
    elif mutation == "missing_axis":
        del c["levels"][0]["caps"][m.AXES[0]]
    else:
        c["levels"][0]["caps"][m.AXES[0]] = True
    if mutation != "wrong_digest":
        sha = m.digest(c)
    with pytest.raises(ValueError):
        m.preflight(c, sha)


def test_all_stages_charged_separately_and_no_fairness_assertion():
    l = ledger()
    for phase, kind, cost in [
        ("setup", "encode", charge(encode_calls=1, encode_samples=2)),
        ("initialization", "action_encode", charge(action_encode_calls=1)),
        ("jacobian", "jacobian", charge(jacobian_calls=1, vjp_rows=2)),
        ("screening", "selection", charge(selection_evaluations=1)),
        ("trial", "predict", charge(predict_calls=1, predict_samples=2)),
        ("common_suffix", "backward", charge(backward_calls=1)),
        ("verification", "replay", charge(replay_calls=1)),
        ("physical", "physical", charge(physical_steps=1)),
    ]:
        assert l.dispatch(phase, kind, cost, lambda: "ok") == "ok"
    r = l.close()
    assert len(r["events"]) == 8
    assert r["charged_upper_bounds"]["vjp_rows"] == 2
    assert r["status"] == "closed_unverified"
    assert not r["actual_usage_verified"] and not r["fairness_proven"]


def test_cap_blocks_before_callback_and_no_refund_after_failure():
    l = ledger()
    called = []
    l.dispatch("trial", "predict", charge(predict_calls=1, predict_samples=2), lambda: None)
    with pytest.raises(ValueError, match="budget exceeded"):
        l.dispatch("trial", "predict", charge(predict_calls=1, predict_samples=1),
                   lambda: called.append(True))
    assert called == []
    r = l.close()
    assert r["charged_upper_bounds"]["predict_samples"] == 2
    assert r["events"][-1]["status"] == "blocked"
    assert r["status"] == "failed_or_blocked"


def test_callback_failure_retains_reservation_and_fail_stop():
    l = ledger()
    def fail():
        raise RuntimeError("backend failure")
    with pytest.raises(RuntimeError):
        l.dispatch("common_suffix", "backward", charge(backward_calls=1), fail)
    with pytest.raises(ValueError):
        l.dispatch("trial", "selection", charge(selection_evaluations=1), lambda: None)
    r = l.close()
    assert r["charged_upper_bounds"]["backward_calls"] == 1
    assert r["events"][0]["status"] == "failed"


def test_caught_recursive_dispatch_still_fails_outer():
    l = ledger()
    def nested():
        try:
            l.dispatch("trial", "selection", charge(selection_evaluations=1), lambda: None)
        except ValueError:
            pass
    with pytest.raises(ValueError, match="nested"):
        l.dispatch("trial", "predict", charge(predict_calls=1, predict_samples=1), nested)
    assert l.close()["status"] == "failed_or_blocked"


def test_incomplete_or_false_zero_meter_poison():
    l = ledger()
    with pytest.raises(ValueError):
        l.dispatch("trial", "predict", charge(predict_calls=1), lambda: None)
    assert l.close()["status"] == "failed_or_blocked"
