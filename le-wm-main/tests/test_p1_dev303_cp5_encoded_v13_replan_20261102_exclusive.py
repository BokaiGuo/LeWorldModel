import io

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_cp5_encoded_v13_replan_20261102_exclusive as subject,
)


def test_report_is_bound_and_contexts_match_actual_histories():
    prep, report, arrays = subject.source()
    assert report["protocol_sha256"] == subject.ENCODE_PREP_SHA
    assert report["encoded_contexts_sha256"] == subject.CONTEXTS_SHA
    assert report["host"]["device"] == "cuda:0"
    for row in prep["roles"]:
        arm = row["role"]
        states = arrays[arm + "_latent_history"]
        actions = arrays[arm + "_normalized_action_history"]
        expected_states, expected_actions = subject.encode.contexts(
            prep["task"], row["actual_prefix"], states[-1])
        assert np.array_equal(states, expected_states)
        assert np.array_equal(actions, expected_actions)


def test_protocol_is_equal_budget_no_physical_and_binds_encoder():
    value = subject.protocol()
    assert [r["role"] for r in value["roles"]] == list(subject.ARMS)
    assert value["encode_report_sha256"] == subject.ENCODE_REPORT_SHA
    assert value["encoded_contexts_sha256"] == subject.CONTEXTS_SHA
    assert value["per_arm_budget"] == {
        "rgb_encode_calls": 0, "action_encoder_calls": 10, "predict_calls": 10,
        "predict_samples": 3200, "cem_iterations": 5, "candidates": 320, "variants": 1600,
    }
    assert value["P1_admission"] is value["independent_confirmation"] is False
    assert value["physical_actions_authorized"] is False


def test_cem_both_arms_same_initial_bank_and_fixed_five_calls():
    banks = []
    results = []
    for direction in ([1.0, 0.0], [0.0, 1.0]):
        calls = []
        def score(variants):
            calls.append(variants.copy())
            return np.square(variants).mean(axis=(2, 3, 4))
        result = subject.plan_one(direction, score)
        assert len(calls) == 5
        assert all(x.shape == (64, 5, 2, 5, 2) for x in calls)
        assert len(result["iterations"]) == 5
        banks.append(calls[0])
        results.append(result)
    assert np.array_equal(banks[0], banks[1])
    assert results[0]["initial_bank_sha256"] == results[1]["initial_bank_sha256"]


def test_invalid_scores_fail_closed():
    with pytest.raises(ValueError, match="finite V13 variant scores"):
        subject.plan_one([1.0, 0.0], lambda _bank: np.full((64, 5), np.nan))


def test_changed_encode_report_sha_fails_closed(monkeypatch):
    monkeypatch.setattr(subject, "ENCODE_REPORT_SHA", "0" * 64)
    with pytest.raises(ValueError, match="fixed input SHA"):
        subject.source()


def test_changed_encoded_archive_fails_closed(monkeypatch):
    monkeypatch.setattr(subject, "CONTEXTS_SHA", "0" * 64)
    with pytest.raises(ValueError, match="fixed input SHA"):
        subject.source()


def test_duplicate_json_keys_fail_closed():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        subject.parent.export.decode(b'{"role":1,"role":2}')
