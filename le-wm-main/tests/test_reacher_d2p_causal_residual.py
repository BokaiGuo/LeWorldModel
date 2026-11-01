"""Numerical/frontier fixtures only; no CPU neural or efficacy experiments."""
import copy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import audit_reacher_d2p_masked_feedback as driver
from experiments.spectral_reachability import analyze_reacher_d2p_masked_feedback as analyzer
from tests.test_audit_reacher_d2p_masked_feedback import stage
from tests.test_analyze_reacher_d2p_masked_feedback import fixture


def test_residual_is_detached_unfitted_and_does_not_mutate_inputs():
    actual = torch.ones(192, requires_grad=True)
    raw = torch.full((192,), .25, requires_grad=True)
    value = driver.causal_residual(actual, raw)
    assert torch.equal(value, torch.full((192,), .75))
    assert not value.requires_grad
    value.zero_()
    assert bool((actual == 1).all()) and bool((raw == .25).all())


@pytest.mark.parametrize("bad", ["shape", "nan", "dtype", "overflow"])
def test_invalid_residual_inputs_fail_closed(bad):
    actual, raw = torch.ones(192), torch.zeros(192)
    if bad == "shape": raw = raw[:191]
    elif bad == "nan": actual[0] = float("nan")
    elif bad == "dtype": raw = raw.double()
    else:
        actual.fill_(torch.finfo(torch.float32).max)
        raw.fill_(-torch.finfo(torch.float32).max)
    with pytest.raises(ValueError): driver.causal_residual(actual, raw)


def residual_stage():
    s = stage()
    s.update(mode="causal_residual_goal", raw_predicted_vector=[.2]*192,
        causal_residual_vector=[.3]*192, predicted_vector=[.5]*192, residual_source_packet=1)
    return s


def test_raw_plus_bias_contract():
    driver.validate_stage(residual_stage())


@pytest.mark.parametrize("bad", ["future", "double_corrected", "nan", "first"])
def test_invalid_residual_stages_fail_closed(bad):
    s = residual_stage()
    if bad == "future": s["residual_source_packet"] = 2
    elif bad == "double_corrected": s["predicted_vector"][0] += .1
    elif bad == "nan": s["causal_residual_vector"][0] = float("nan")
    else: s.update(packet=1, raw_steps_observed_before_selection=0, optimizer_receipt=None, residual_source_packet=0)
    with pytest.raises(ValueError): driver.validate_stage(s)


def residual_fixture():
    values = fixture()
    report, _, _, stages, _ = values
    report.update(schema="reacher_d2p_masked_feedback.v2", actual_predict_samples=29376, physical_branches=576)
    report["gates"].pop("all432_branches1296_packets")
    report["gates"]["all_declared_branches_packets"] = True
    report["sealed_protocol"].update(modes=list(driver.RESIDUAL_MODES),
        causal_residual_contract="unit-gain-last-observed-minus-raw-prediction-v1", residual_parameters_fit=False)
    for r in report["rows"]:
        for task, families in r["tasks"].items():
            for family, branch in families.items():
                branch["modes"]["causal_residual_goal"] = copy.deepcopy(branch["modes"]["goal_feedback"])
                for j in (1, 2, 3):
                    s = copy.deepcopy(stages[(r["seed"], task, family, "goal_feedback", j)])
                    s.update(mode="causal_residual_goal", raw_predicted_vector=[0.]*192,
                        causal_residual_vector=[0.]*192, residual_source_packet=j-1)
                    stages[(r["seed"], task, family, "causal_residual_goal", j)] = s
    return values


def test_full_residual_panel_and_old_schema_remain_supported():
    assert analyzer.summarize(*fixture())["modes"] == list(driver.MODES)
    result = analyzer.summarize(*residual_fixture())
    assert result["modes"] == list(driver.RESIDUAL_MODES)
    assert result["p1_admitted"] is False


def test_fixed_action_correction_cannot_read_current_or_later_observation():
    reference = fixture()[2]
    before = analyzer.fixed_action_lag_diagnostic(reference)
    branch = reference["rows"][0]["tasks"]["full_drive"]["common_native"]
    branch["packets"][1]["vectors"]["observed"] = [1.]*192
    after = analyzer.fixed_action_lag_diagnostic(reference)
    b = before["rows"][0]["tasks"]["full_drive"]["families"]["common_native"]
    a = after["rows"][0]["tasks"]["full_drive"]["families"]["common_native"]
    assert a[0] == b[0]
    assert a[1]["corrected_cost"] == b[1]["corrected_cost"]
    assert a[1]["observed_cost"] != b[1]["observed_cost"]
    assert a[2]["corrected_cost"] != b[2]["corrected_cost"]
    assert a[2]["residual_source_packet"] == 2


def test_fixed_action_correction_uses_raw_not_previous_corrected_error():
    reference = fixture()[2]
    branch = reference["rows"][0]["tasks"]["full_drive"]["common_native"]
    for p in branch["packets"]:
        p["vectors"]["actual_history"] = [0.]*192
        p["vectors"]["observed"] = [1.]*192
        p["vectors"]["goal"] = [1.]*192
    result = analyzer.fixed_action_lag_diagnostic(reference)["rows"][0]["tasks"]["full_drive"]["families"]["common_native"]
    assert result[0]["corrected_cost"] == 1.
    assert result[1]["corrected_cost"] == result[2]["corrected_cost"] == 0.


@pytest.mark.parametrize("bad", ["bias", "source", "fitted", "contract", "dropped", "count"])
def test_independent_chain_rejects_noncausal_residual(bad):
    values = residual_fixture()
    report, _, _, stages, _ = values
    s = stages[(0, "full_drive", "common_native", "causal_residual_goal", 2)]
    if bad == "bias":
        # Algebra still matches, but it is not the previous observed error.
        s["causal_residual_vector"] = s["predicted_vector"] = [.5]*192
    elif bad == "source": s["residual_source_packet"] = 2
    elif bad == "fitted": report["sealed_protocol"]["residual_parameters_fit"] = True
    elif bad == "contract": report["sealed_protocol"]["causal_residual_contract"] = "oracle"
    elif bad == "dropped": report["rows"].pop()
    else: report["actual_predict_samples"] -= 1
    with pytest.raises(ValueError): analyzer.summarize(*values)
