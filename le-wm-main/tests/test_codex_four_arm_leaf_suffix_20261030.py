import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as module


class Analytic:
    def __init__(self):
        self.calls = []
    def encode(self, info):
        self.calls.append("encode")
        return {"emb": info["pixels"]}
    def action_encoder(self, actions):
        self.calls.append("action_encoder")
        return actions
    def predict(self, states, actions):
        self.calls.append("predict")
        return states + actions


def setup(tmp_path, *, caps=None):
    config = {"steps": 2, "learning_rate": 0.1, "history_size": 1, "lower": -2.0, "upper": 2.0}
    identity = {k: "a" * 64 for k in spec.IDENTITY}
    identity["optimizer_config"] = spec.digest(config)
    arms = {a: {"initializer_sha256": str(i + 1) * 64, "optimizer_sha256": identity["optimizer"],
                "optimizer_config_sha256": identity["optimizer_config"]}
            for i, a in enumerate(spec.ARMS)}
    limits = {k: 100 for k in spec.AXES}
    if caps:
        limits.update(caps)
    contract = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": "b" * 64,
                "levels": [{"id": "a", "caps": limits},
                           {"id": "b", "caps": {k: v + 1 for k, v in limits.items()}}]}
    sha = spec.digest(contract)
    driver = durable.Driver(tmp_path / "run", contract, sha, "structured", "a", "run-1")
    model = Analytic()
    leaf = module.LeafAdapter(model, driver, contract, sha, analytic_fixture=True)
    return config, contract, sha, driver, model, leaf


def verify(driver, sha):
    r = driver.finish()
    return consumer.consume(r["journal"], r["sha256"], sha, "structured", "a", "run-1")


def optimize(leaf, config):
    return module.optimize(leaf, torch.zeros(1, 1, 1), torch.empty(1, 0, 1),
                           torch.zeros(1, 2, 1), torch.ones(1, 1), config)


def test_analytic_suffix_leaf_events_and_real_optimizer(tmp_path):
    cfg, _, sha, d, model, leaf = setup(tmp_path)
    output = optimize(leaf, cfg)
    assert output.shape == (1, 2, 1)
    assert torch.all(output > 0)
    r = verify(d, sha)
    assert model.calls == ["action_encoder", "predict"] * 4
    assert r["charged_upper_bounds"]["action_encode_calls"] == 4
    assert r["charged_upper_bounds"]["predict_calls"] == 4
    assert r["charged_upper_bounds"]["predict_samples"] == 4
    assert r["charged_upper_bounds"]["backward_calls"] == 2
    assert r["charged_upper_bounds"]["vjp_rows"] == 2
    assert not r["fairness_proven"]


def test_encode_disallows_hidden_action_and_counts_batch(tmp_path):
    _, _, sha, d, model, leaf = setup(tmp_path)
    result = leaf.encode({"pixels": torch.zeros(2, 1, 1)})
    assert result["emb"].shape[0] == 2
    with pytest.raises(ValueError):
        leaf.encode({"pixels": torch.zeros(2, 1, 1), "action": torch.zeros(2, 1, 1)})
    r = verify(d, sha)
    assert model.calls == ["encode"]
    assert r["charged_upper_bounds"]["encode_calls"] == 1
    assert r["charged_upper_bounds"]["encode_samples"] == 2
    assert r["status"] == "failed_or_blocked"


def test_budget_block_does_not_run_predict_and_preserves_previous_action_encoder(tmp_path):
    cfg, _, sha, d, model, leaf = setup(tmp_path, caps={"predict_calls": 0})
    with pytest.raises(ValueError, match="budget"):
        optimize(leaf, cfg)
    r = verify(d, sha)
    assert model.calls == ["action_encoder"]
    assert r["charged_upper_bounds"]["action_encode_calls"] == 1
    assert r["charged_upper_bounds"]["predict_calls"] == 0
    assert r["status"] == "failed_or_blocked"


def test_backend_failure_recorded_and_propagated(tmp_path):
    cfg, _, sha, d, model, leaf = setup(tmp_path)
    def fail(*args):
        raise RuntimeError("model fail")
    model.predict = fail
    with pytest.raises(RuntimeError, match="model fail"):
        optimize(leaf, cfg)
    r = verify(d, sha)
    assert r["status"] == "failed_or_blocked"
    assert r["charged_upper_bounds"]["predict_calls"] == 1


def test_invalid_return_or_config_cannot_be_marked_success(tmp_path):
    cfg, _, sha, d, model, leaf = setup(tmp_path)
    model.predict = lambda states, actions: torch.full_like(states, float("nan"))
    with pytest.raises(ValueError):
        optimize(leaf, cfg)
    assert verify(d, sha)["status"] == "failed_or_blocked"


def test_common_config_and_header_binding(tmp_path):
    cfg, contract, sha, d, _, leaf = setup(tmp_path)
    cfg["learning_rate"] = .3
    with pytest.raises(ValueError, match="config"):
        optimize(leaf, cfg)
    assert verify(d, sha)["status"] == "failed_or_blocked"


def test_no_cpu_neural_fallback(tmp_path):
    _, contract, sha, d, _, _ = setup(tmp_path)
    class Neural(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
            self.eval()
            self.requires_grad_(False)
        def encode(self, value):
            return value
        def action_encoder(self, value):
            return self.layer(value)
        def predict(self, state, action):
            return state + action
    with pytest.raises(ValueError, match="CUDA"):
        module.LeafAdapter(Neural(), d, contract, sha)
    d.finish()


def test_backward_cap_rejects_before_autograd(tmp_path, monkeypatch):
    cfg, _, sha, d, _, leaf = setup(tmp_path, caps={"backward_calls": 0})
    called = []
    monkeypatch.setattr(torch.autograd, "grad", lambda *a, **k: called.append(True))
    with pytest.raises(ValueError, match="budget"):
        optimize(leaf, cfg)
    r = verify(d, sha)
    assert called == []
    assert r["charged_upper_bounds"]["predict_calls"] == 2
    assert r["charged_upper_bounds"]["backward_calls"] == 0


def test_history_three_alignment_and_gradient(tmp_path):
    cfg, contract, sha, d, model, leaf = setup(tmp_path)
    # A separate ledger is required because the optimizer configuration is bound.
    d.finish()
    cfg["history_size"] = 3
    contract["identity"]["optimizer_config"] = spec.digest(cfg)
    for arm in contract["arms"].values():
        arm["optimizer_config_sha256"] = spec.digest(cfg)
    sha = spec.digest(contract)
    d = durable.Driver(tmp_path / "history3", contract, sha, "native", "a", "run-2")
    contexts = []
    def predict(z, a):
        contexts.append((tuple(z.shape), a.detach().clone()))
        return z + a
    model.predict = predict
    leaf = module.LeafAdapter(model, d, contract, sha, analytic_fixture=True)
    out = module.optimize(leaf, torch.zeros(1, 3, 1), torch.tensor([[[.2], [.3]]]),
                          torch.zeros(1, 2, 1), torch.ones(1, 1), cfg)
    assert out.shape == (1, 2, 1)
    assert all(shape == (1, 3, 1) for shape, _ in contexts)
    assert contexts[0][1].flatten().tolist() == pytest.approx([.2, .3, 0])
    assert contexts[1][1].flatten().tolist() == pytest.approx([.3, 0, 0])
    receipt = d.finish()
    r = consumer.consume(receipt["journal"], receipt["sha256"], sha, "native", "a", "run-2")
    assert r["charged_upper_bounds"]["predict_samples"] == 4


def test_nested_model_hook_fails_closed(tmp_path):
    cfg, _, sha, d, model, leaf = setup(tmp_path)
    def nested(z, a):
        try:
            leaf.action_encode(a, phase="common_suffix")
        except ValueError:
            pass
        return z + a
    model.predict = nested
    with pytest.raises(ValueError, match="nested"):
        optimize(leaf, cfg)
    assert verify(d, sha)["status"] == "failed_or_blocked"


def test_bad_action_encoder_exception_fail_stops(tmp_path):
    cfg, _, sha, d, model, leaf = setup(tmp_path)
    def fail(_):
        raise RuntimeError("action encoder failure")
    model.action_encoder = fail
    with pytest.raises(RuntimeError, match="action encoder"):
        optimize(leaf, cfg)
    r = verify(d, sha)
    assert r["charged_upper_bounds"]["action_encode_calls"] == 1
    assert r["charged_upper_bounds"]["predict_calls"] == 0
    assert r["status"] == "failed_or_blocked"
