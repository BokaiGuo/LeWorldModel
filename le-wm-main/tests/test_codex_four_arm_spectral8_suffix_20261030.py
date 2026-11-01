import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_spectral8_suffix_20261030 as m
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as suffix
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as lm
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer


class Affine:
    def encode(self, info):
        return info

    def action_encoder(self, action):
        return action

    def predict(self, states, actions):
        # Non-degenerate modes with a full differentiable 192-vector output.
        result = states.clone()
        weights = torch.arange(10, 0, -1, dtype=states.dtype, device=states.device)
        result[..., :10] = result[..., :10] + actions[..., :10] * weights
        return result


def setup(tmp_path, caps_override=None, history_size=1):
    cfg = {"steps": 1, "learning_rate": 0.001, "history_size": history_size}
    policy = m.policy([0., 0.], [1., 1.], history_size)
    identity = {k: "a"*64 for k in spec.IDENTITY}
    identity.update(optimizer=m.file_sha(suffix.__file__), optimizer_config=spec.digest(cfg))
    arms = {a: {"initializer_sha256": m.file_sha(m.__file__) if a == "structured" else "b"*64,
                "optimizer_sha256": identity["optimizer"],
                "optimizer_config_sha256": identity["optimizer_config"]} for a in spec.ARMS}
    caps = m.required_caps(cfg)
    if caps_override:
        caps.update(caps_override)
    contract = {"schema": spec.NAME+".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "test", "caps": caps},
                           {"id": "larger", "caps": {k:v+1 for k,v in caps.items()}}]}
    digest = spec.digest(contract)
    driver = durable.Driver(tmp_path/"ledger", contract, digest, "structured", "test", "fixture")
    leaf = lm.LeafAdapter(Affine(), driver, contract, digest, analytic_fixture=True)
    inputs = (torch.zeros(1,history_size,192), torch.zeros(1,history_size-1,10), torch.ones(1,192),
              torch.zeros(2), torch.ones(2), torch.eye(192))
    return cfg, policy, contract, digest, driver, leaf, inputs


def consume(driver, digest):
    receipt = driver.finish()
    return consumer.consume(receipt["journal"], receipt["sha256"], digest, "structured", "test", "fixture")


def test_spectral8_optimizes_and_replays_every_postupdate_candidate(tmp_path):
    cfg, pol, contract, digest, driver, leaf, inputs = setup(tmp_path)
    events = []
    result = m.run(leaf, *inputs, cfg, pol, contract, emit=lambda k,v: events.append((k,copy.deepcopy(v))))
    r = consume(driver, digest)
    assert result["initial_raw"].shape == result["optimized_raw"].shape == (8,3,5,2)
    assert not torch.equal(result["initial_raw"], result["optimized_raw"])
    assert len(result["scores"]) == 8 and result["selected_index"] == min(range(8), key=lambda i:(result["scores"][i],i))
    assert [x[0] for x in events] == ["spectral_signal", "spectral_bank", "optimized_candidates", "final_selection"]
    assert r["charged_upper_bounds"] == m.required_caps(cfg)
    assert r["status"] == "closed_unverified"
    assert result["label"] == "spectral8_h3_extension_not_original_R_or_fairness"


def test_svd_failure_is_charged_and_no_suffix_or_selection(tmp_path, monkeypatch):
    cfg, pol, contract, digest, driver, leaf, inputs = setup(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("svd failure")
    monkeypatch.setattr(np.linalg, "svd", fail)
    with pytest.raises(RuntimeError, match="svd failure"):
        m.run(leaf, *inputs, cfg, pol, contract, emit=lambda *_: None)
    r = consume(driver, digest)
    assert r["status"] == "failed_or_blocked"
    assert r["charged_upper_bounds"]["svd_calls"] == 1
    assert r["charged_upper_bounds"]["replay_calls"] == 0


def test_reserve_shortfall_fails_before_model_or_svd(tmp_path, monkeypatch):
    cfg, pol, contract, digest, driver, leaf, inputs = setup(tmp_path, {"replay_calls": 7})
    called = []
    monkeypatch.setattr(np.linalg, "svd", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="reserve"):
        m.run(leaf, *inputs, cfg, pol, contract, emit=lambda *_: None)
    r = consume(driver, digest)
    assert not called and r["charged_upper_bounds"]["predict_calls"] == 0


def test_persistence_failure_prevents_suffix(tmp_path):
    cfg, pol, contract, digest, driver, leaf, inputs = setup(tmp_path)
    def emit(kind, payload):
        if kind == "spectral_bank":
            raise OSError("sink failure")
    with pytest.raises(OSError, match="sink failure"):
        m.run(leaf, *inputs, cfg, pol, contract, emit=emit)
    r = consume(driver, digest)
    assert r["charged_upper_bounds"]["svd_calls"] == 1
    assert r["charged_upper_bounds"]["replay_calls"] == 0


def test_spectral_bank_matches_reference_equation():
    jac = np.zeros((192,30), np.float64)
    jac[:30,:30] = np.diag(np.arange(30, 0, -1))
    bank, singular = m.spectral_bank(jac, np.eye(192))
    _, expected_singular, right = np.linalg.svd(jac, full_matrices=False)
    for row in right:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    expected = np.stack([np.tanh(sign*radius*right[mode]).astype(np.float32).reshape(3,5,2)
                         for radius in m.RADII for mode in range(2) for sign in (-1,1)])
    np.testing.assert_array_equal(bank, expected)
    np.testing.assert_array_equal(singular, expected_singular)


def test_degenerate_modes_fail_closed_after_paid_svd(tmp_path, monkeypatch):
    cfg, pol, contract, digest, driver, leaf, inputs = setup(tmp_path)
    original = np.linalg.svd
    def degenerate(*args, **kwargs):
        left, values, right = original(*args, **kwargs)
        values[1] = values[0]
        return left, values, right
    monkeypatch.setattr(np.linalg, "svd", degenerate)
    with pytest.raises(ValueError, match="degenerate"):
        m.run(leaf, *inputs, cfg, pol, contract, emit=lambda *_: None)
    r = consume(driver, digest)
    assert r["status"] == "failed_or_blocked"
    assert r["charged_upper_bounds"]["svd_calls"] == 1
    assert r["charged_upper_bounds"]["replay_calls"] == 0


def test_replay_failure_is_charged_and_produces_no_selection(tmp_path):
    cfg, pol, contract, digest, driver, leaf, inputs = setup(tmp_path)
    predict = leaf.model.predict
    calls = []
    def failing_predict(*args):
        calls.append(1)
        if len(calls) == 28:  # 3 Jacobian and 24 common-suffix predictions
            raise RuntimeError("replay failure")
        return predict(*args)
    leaf.model.predict = failing_predict
    events = []
    with pytest.raises(RuntimeError, match="replay failure"):
        m.run(leaf, *inputs, cfg, pol, contract, emit=lambda k,v: events.append(k))
    r = consume(driver, digest)
    assert r["status"] == "failed_or_blocked"
    assert r["charged_upper_bounds"]["predict_calls"] == 28
    assert r["charged_upper_bounds"]["replay_calls"] == 1
    assert "optimized_candidates" in events and "final_selection" not in events


def test_reject_nonidentity_metric_before_model(tmp_path):
    cfg, pol, contract, digest, driver, leaf, inputs = setup(tmp_path)
    values = list(inputs)
    values[-1] = 2 * values[-1]
    with pytest.raises(ValueError, match="dimensions"):
        m.run(leaf, *values, cfg, pol, contract, emit=lambda *_: None)
    assert consume(driver, digest)["charged_upper_bounds"]["predict_calls"] == 0


def test_three_state_history_is_supported(tmp_path):
    cfg, pol, contract, digest, driver, leaf, inputs = setup(tmp_path, history_size=3)
    result = m.run(leaf, *inputs, cfg, pol, contract, emit=lambda *_: None)
    assert result["optimized_raw"].shape == (8,3,5,2)
    assert consume(driver, digest)["charged_upper_bounds"] == m.required_caps(cfg)
