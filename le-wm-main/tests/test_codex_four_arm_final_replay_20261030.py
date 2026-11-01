import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as suffix
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as raw_suffix
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay


class Affine:
    def __init__(self):
        self.actions = []
    def encode(self, info):
        return {"emb": info["pixels"]}
    def action_encoder(self, action):
        self.actions.append(action.detach().clone())
        return action
    def predict(self, states, actions):
        return states + actions[..., :1]


def setup(tmp_path, count=2, caps=None):
    config = {"steps": 1, "learning_rate": .1, "history_size": 1}
    policy = {"schema": replay.POLICY_SCHEMA, "candidate_ids": [f"c{i}" for i in range(count)],
              "horizon": 1, "history_size": 1, "metric": "terminal_latent_mean_squared_error",
              "tie": "first_declared_candidate", "replay_batch": 1,
              "action_chart": "raw_packet_5x2_box_minus1_to1", "mean": [0.5, -0.5], "std": [0.5, 2.0]}
    identity = {k: "a" * 64 for k in spec.IDENTITY}
    identity["optimizer_config"] = spec.digest(config)
    arms = {a: {"initializer_sha256": str(i + 1) * 64, "optimizer_sha256": identity["optimizer"],
                "optimizer_config_sha256": identity["optimizer_config"]}
            for i, a in enumerate(spec.ARMS)}
    limits = {k: 50 for k in spec.AXES}
    if caps:
        limits.update(caps)
    contract = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "a", "caps": limits},
                           {"id": "b", "caps": {k: v + 1 for k, v in limits.items()}}]}
    sha = spec.digest(contract)
    d = durable.Driver(tmp_path / "run", contract, sha, "structured", "a", "run")
    model = Affine()
    leaf = suffix.LeafAdapter(model, d, contract, sha, analytic_fixture=True)
    return config, policy, contract, sha, d, model, leaf


def select(leaf, policy, contract, candidates):
    return replay.replay_and_select(leaf, torch.zeros(1, 1, 1), torch.empty(1, 0, 10), candidates,
                                    torch.ones(1, 1), torch.tensor(policy["mean"]),
                                    torch.tensor(policy["std"]), policy, contract)


def consume(d, sha):
    r = d.finish()
    return consumer.consume(r["journal"], r["sha256"], sha, "structured", "a", "run")


def test_final_post_update_raw_action_gets_fresh_replay(tmp_path):
    cfg, policy, contract, sha, d, model, leaf = setup(tmp_path, count=1)
    raw = torch.zeros(1, 1, 5, 2)
    state, history, goal = torch.zeros(1, 1, 1), torch.empty(1, 0, 10), torch.ones(1, 1)
    optimized = raw_suffix.optimize(leaf, state, history, raw, goal, torch.tensor(policy["mean"]),
                                    torch.tensor(policy["std"]), cfg)
    assert model.actions[0][0, 0, 0].item() == pytest.approx(-1.)
    assert optimized[0, 0, 0, 0].item() == pytest.approx(.8)
    result = select(leaf, policy, contract, optimized)
    assert model.actions[1][0, 0, 0].item() == pytest.approx(.6)
    assert result["scores"][0] == pytest.approx(.16)
    assert result["scores"][0] != pytest.approx(4.)
    r = consume(d, sha)
    assert r["charged_upper_bounds"]["predict_calls"] == 2
    assert r["charged_upper_bounds"]["backward_calls"] == 1
    assert r["charged_upper_bounds"]["replay_calls"] == 1
    assert r["charged_upper_bounds"]["selection_evaluations"] == 1


def test_all_candidates_paid_and_tie_stable(tmp_path):
    _, policy, contract, sha, d, model, leaf = setup(tmp_path, count=3)
    result = select(leaf, policy, contract, torch.zeros(3, 1, 5, 2))
    assert result["selected_id"] == "c0"
    assert result["scores"] == pytest.approx([4., 4., 4.])
    r = consume(d, sha)
    for k in ("predict_calls", "predict_samples", "action_encode_calls", "replay_calls", "selection_evaluations"):
        assert r["charged_upper_bounds"][k] == 3
    assert len(model.actions) == 3 and not r["fairness_proven"]


def test_all_candidates_same_mean_std_select_lower_score(tmp_path):
    _, policy, contract, sha, d, model, leaf = setup(tmp_path)
    raw = torch.zeros(2, 1, 5, 2)
    raw[1, :, :, 0] = 1.
    result = select(leaf, policy, contract, raw)
    assert result["scores"] == pytest.approx([4., 0.])
    assert result["selected_id"] == "c1"
    assert model.actions[0][0, 0, :2].tolist() == pytest.approx([-1., .25])
    assert model.actions[1][0, 0, :2].tolist() == pytest.approx([1., .25])
    assert consume(d, sha)["charged_upper_bounds"]["selection_evaluations"] == 2


def test_cap_failure_no_partial_winner(tmp_path):
    _, policy, contract, sha, d, _, leaf = setup(tmp_path, caps={"selection_evaluations": 1})
    with pytest.raises(ValueError, match="budget"):
        select(leaf, policy, contract, torch.zeros(2, 1, 5, 2))
    r = consume(d, sha)
    assert r["status"] == "failed_or_blocked"
    assert r["charged_upper_bounds"]["replay_calls"] == 2
    assert r["charged_upper_bounds"]["selection_evaluations"] == 1


@pytest.mark.parametrize("bad", ["denominator", "raw_bound", "policy", "normalizer"])
def test_invalid_contract_and_raw_fail_closed_before_model(tmp_path, bad):
    _, policy, contract, sha, d, model, leaf = setup(tmp_path)
    raw = torch.zeros(2, 1, 5, 2)
    if bad == "denominator":
        raw = raw[:1]
    elif bad == "raw_bound":
        raw[0, 0, 0, 0] = 1.1
    elif bad == "policy":
        policy["candidate_ids"].reverse()
    if bad == "normalizer":
        with pytest.raises(ValueError):
            replay.replay_and_select(leaf, torch.zeros(1, 1, 1), torch.empty(1, 0, 10), raw,
                                     torch.ones(1, 1), torch.zeros(2), torch.ones(2), policy, contract)
    else:
        with pytest.raises(ValueError):
            select(leaf, policy, contract, raw)
    assert not model.actions
    assert consume(d, sha)["status"] == "failed_or_blocked"


def test_replay_budget_blocks_before_model_and_failure_is_retained(tmp_path):
    _, policy, contract, sha, d, model, leaf = setup(tmp_path, count=1, caps={"replay_calls": 0})
    with pytest.raises(ValueError, match="budget"):
        select(leaf, policy, contract, torch.zeros(1, 1, 5, 2))
    assert not model.actions
    assert consume(d, sha)["status"] == "failed_or_blocked"


def test_predict_failure_blocks_selection(tmp_path):
    _, policy, contract, sha, d, model, leaf = setup(tmp_path, count=1)
    def fail(*args):
        raise RuntimeError("replay failed")
    model.predict = fail
    with pytest.raises(RuntimeError, match="replay failed"):
        select(leaf, policy, contract, torch.zeros(1, 1, 5, 2))
    r = consume(d, sha)
    assert r["charged_upper_bounds"]["predict_calls"] == 1
    assert r["charged_upper_bounds"]["selection_evaluations"] == 0
    assert r["status"] == "failed_or_blocked"


def test_h3_history_is_not_renormalized_and_raw_box_is_not_normalized_box(tmp_path):
    _, policy, contract, _, old, model, _ = setup(tmp_path, count=1)
    old.finish()
    policy["history_size"] = 3
    policy["horizon"] = 2
    contract["phase_policy_sha256"] = spec.digest(policy)
    sha = spec.digest(contract)
    d = durable.Driver(tmp_path / "h3", contract, sha, "native", "a", "h3")
    leaf = suffix.LeafAdapter(model, d, contract, sha, analytic_fixture=True)
    history = torch.stack((torch.full((10,), 3.), torch.full((10,), 4.))).reshape(1, 2, 10)
    raw = torch.full((1, 2, 5, 2), -1.)
    result = replay.replay_and_select(leaf, torch.zeros(1, 3, 1), history, raw,
                                      torch.ones(1, 1), torch.tensor(policy["mean"]),
                                      torch.tensor(policy["std"]), policy, contract)
    assert result["actions"].shape == (1, 2, 5, 2)
    assert model.actions[0][0, :, 0].tolist() == pytest.approx([3., 4., -3.])
    assert model.actions[1][0, :, 0].tolist() == pytest.approx([4., -3., -3.])
    receipt = d.finish()
    r = consumer.consume(receipt["journal"], receipt["sha256"], sha, "native", "a", "h3")
    assert r["charged_upper_bounds"]["predict_calls"] == 2
