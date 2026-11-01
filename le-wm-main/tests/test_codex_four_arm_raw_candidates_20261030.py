import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay
from experiments.spectral_reachability import codex_four_arm_raw_candidates_20261030 as m


class Affine:
    def encode(self, info):
        return {"emb": info["pixels"]}
    def action_encoder(self, action):
        return action
    def predict(self, states, actions):
        return states + actions[..., :1]


def make(tmp_path, arm, cap=None):
    policy = {"schema": replay.POLICY_SCHEMA, "candidate_ids": ["slot0", "slot1", "slot2"],
              "horizon": 1, "history_size": 1, "metric": "terminal_latent_mean_squared_error",
              "tie": "first_declared_candidate", "replay_batch": 1,
              "action_chart": "raw_packet_5x2_box_minus1_to1", "mean": [0., 0.], "std": [1., 1.]}
    cfg = {"schema": m.CONFIG_SCHEMA, "iterations": 2, "elite": 1, "seed": 17,
           "initial_std": .4, "std_floor": .05, "ridge": .1}
    identity = {k: "a" * 64 for k in spec.IDENTITY}
    identity["optimizer_config"] = spec.digest(cfg)
    arms = {a: {"initializer_sha256": str(i+1)*64, "optimizer_sha256": identity["optimizer"],
                "optimizer_config_sha256": identity["optimizer_config"]}
            for i, a in enumerate(spec.ARMS)}
    caps = {k: 100 for k in spec.AXES}
    if cap:
        caps.update(cap)
    contract = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "a", "caps": caps},
                           {"id": "b", "caps": {k: v+1 for k, v in caps.items()}}]}
    sha = spec.digest(contract)
    driver = durable.Driver(tmp_path / arm, contract, sha, arm, "a", arm)
    leaf = leafmod.LeafAdapter(Affine(), driver, contract, sha, analytic_fixture=True)
    return cfg, policy, contract, sha, driver, leaf


def run(leaf, cfg, policy, contract, arm):
    return m.run_arm(leaf, arm, torch.zeros(1, 1, 1), torch.empty(1, 0, 10),
                     torch.ones(1, 1), torch.zeros(2), torch.ones(2), cfg, policy, contract)


def read(d, sha, arm):
    receipt = d.finish()
    return consumer.consume(receipt["journal"], receipt["sha256"], sha, arm, "a", arm)


@pytest.mark.parametrize("arm", spec.ARMS)
def test_four_arm_cem_produces_raw_and_charges_all_model_work(tmp_path, arm):
    cfg, policy, contract, sha, d, leaf = make(tmp_path, arm)
    out = run(leaf, cfg, policy, contract, arm)
    assert out["selected_raw"].shape == (1, 1, 5, 2)
    assert (out["selected_raw"].abs() <= 1).all()
    receipt = read(d, sha, arm)
    expected_predict = {"native": 9, "random": 9, "structured": 10, "immediate_progress": 12}
    expected_replay = 12 if arm == "immediate_progress" else 9
    assert receipt["charged_upper_bounds"]["predict_calls"] == expected_predict[arm]
    assert receipt["charged_upper_bounds"]["predict_samples"] == expected_predict[arm]
    assert receipt["charged_upper_bounds"]["action_encode_calls"] == expected_predict[arm]
    assert receipt["charged_upper_bounds"]["replay_calls"] == expected_replay
    assert receipt["charged_upper_bounds"]["selection_evaluations"] == expected_replay
    if arm == "structured":
        assert receipt["charged_upper_bounds"]["jacobian_calls"] == 1
        assert receipt["charged_upper_bounds"]["vjp_rows"] == 1
        assert receipt["charged_upper_bounds"]["solve_calls"] == 1
    assert not receipt["fairness_proven"]


def test_structured_backward_cap_blocks_before_backward(tmp_path, monkeypatch):
    cfg, policy, contract, sha, d, leaf = make(tmp_path, "structured", {"backward_calls": 0})
    calls = []
    monkeypatch.setattr(torch.autograd, "grad", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="budget"):
        run(leaf, cfg, policy, contract, "structured")
    r = read(d, sha, "structured")
    assert not calls and r["status"] == "failed_or_blocked"


def test_other_arm_cannot_claim_driver_and_config_cannot_drift(tmp_path):
    cfg, policy, contract, sha, d, leaf = make(tmp_path, "native")
    with pytest.raises(ValueError):
        run(leaf, cfg, policy, contract, "random")
    assert read(d, sha, "native")["status"] == "failed_or_blocked"


def test_structured_solver_failure_is_charged_and_no_winner(tmp_path, monkeypatch):
    cfg, policy, contract, sha, d, leaf = make(tmp_path, "structured")
    def fail(*args):
        raise RuntimeError("solve failed")
    monkeypatch.setattr(torch.linalg, "solve", fail)
    with pytest.raises(RuntimeError, match="solve failed"):
        run(leaf, cfg, policy, contract, "structured")
    r = read(d, sha, "structured")
    assert r["status"] == "failed_or_blocked"
    assert r["charged_upper_bounds"]["solve_calls"] == 1
    assert r["charged_upper_bounds"]["replay_calls"] == 0


def test_config_drift_is_rejected_before_any_model_work(tmp_path):
    cfg, policy, contract, sha, d, leaf = make(tmp_path, "native")
    cfg["iterations"] += 1
    with pytest.raises(ValueError, match="identity"):
        run(leaf, cfg, policy, contract, "native")
    assert read(d, sha, "native")["charged_upper_bounds"]["predict_calls"] == 0
