import pytest
import torch

from experiments.spectral_reachability import reacher_p1_predicted_successor_cem as successor
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1


class Model:
    def encode(self, info):
        return {"emb": info["pixels"]}

    def action_encoder(self, actions):
        return actions

    def predict(self, embeddings, actions):
        return embeddings[:, -1:] + 0.2 * actions[:, -1:] + 0.1 * actions[:, -2:-1]


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_predicted_successor_matches_existing_cem_on_identical_context(horizon):
    model = Model()
    pixels = torch.tensor([[[0.1, -0.2], [0.3, 0.4], [-0.1, 0.5]]])
    history = torch.tensor([[[0.1, 0.2], [-0.2, 0.1], [0.3, -0.4]]])
    goal = torch.tensor([[[0.5, -0.3]]])
    lower, upper = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
    kwargs = dict(future_horizon=horizon, action_lower=lower, action_upper=upper,
                  samples=16, steps=3, elite=4, seed=42)
    actual, old_receipt = p0._cem(model, pixels, goal, history, **kwargs)
    predicted, receipt = successor.cem_from_predicted_successor(
        model, p0._goal_latent(model, goal), p0._latent_context(model, pixels, history),
        history, **kwargs,
    )
    assert torch.equal(actual, predicted)
    assert receipt["trace"] == pytest.approx(old_receipt["trace"])
    assert receipt["forward_rollout_evaluations"] == 48
    assert receipt["model_predict_sample_evaluations"] == 48 * horizon


@pytest.mark.parametrize("elite", [0, 1, 16])
def test_invalid_elite_count_fails_closed(elite):
    with pytest.raises(ValueError, match="budget"):
        successor.cem_from_predicted_successor(
            Model(), torch.zeros(2), torch.zeros(1, 3, 2), torch.zeros(1, 3, 2),
            future_horizon=2, action_lower=-torch.ones(2), action_upper=torch.ones(2),
            samples=16, steps=3, elite=elite, seed=1,
        )


@pytest.mark.parametrize("prefix_horizon,suffix_horizon", [(1, 1), (1, 2), (2, 3)])
def test_predicted_successor_carries_the_same_causal_history_as_full_rollout(
    prefix_horizon, suffix_horizon,
):
    model = Model()
    pixels = torch.tensor([[[0.1, -0.2], [0.3, 0.4], [-0.1, 0.5]]])
    history = torch.tensor([[[0.1, 0.2], [-0.2, 0.1], [0.3, -0.4]]])
    goal = torch.zeros(1, 1, 2)
    lower, upper = -torch.ones(2), torch.ones(2)
    params = torch.arange(2 * (prefix_horizon + suffix_horizon), dtype=torch.float32)
    params = (params / 20).reshape(1, prefix_horizon + suffix_horizon, 2)
    full = p0._terminal_batch(
        model, pixels, goal, history, params, action_lower=lower, action_upper=upper,
    )
    embeddings, actions = p1._state_after_prefix(
        model, pixels, history, params[:, :prefix_horizon],
        action_lower=lower, action_upper=upper,
    )
    stitched = p1._terminal_from_state(
        model, embeddings, actions, params[:, prefix_horizon:],
        action_lower=lower, action_upper=upper,
    )
    torch.testing.assert_close(full, stitched)


def test_selector_ranks_fixed_bank_without_physical_inputs_and_counts_all_calls(monkeypatch):
    monkeypatch.setattr(successor.p0, "_goal_latent", lambda *args: torch.zeros(1))
    monkeypatch.setattr(successor.p0, "_cost_fn", lambda *args, **kwargs: torch.tensor([1.0, 1.2, 0.9]))
    monkeypatch.setattr(successor.p1, "_state_after_prefix", lambda model, pixels, history, params, **kwargs:
                        (params, params))
    seen = []

    def fake_cem(model, goal, embeddings, actions, **kwargs):
        index = int(embeddings[0, 0, 0])
        seen.append((index, kwargs["seed"]))
        return embeddings.repeat(1, kwargs["future_horizon"], 1), {"best_cost": [3.0, 2.0, 1.0][index]}

    monkeypatch.setattr(successor, "cem_from_predicted_successor", fake_cem)
    chosen, receipt = successor.select_prefix_with_cem_suffix(
        Model(), torch.zeros(1, 3, 1), torch.zeros(1), torch.zeros(1, 3, 1),
        torch.arange(3.0).reshape(3, 1, 1), future_horizon=2,
        action_lower=-torch.ones(1), action_upper=torch.ones(1),
        samples=8, steps=3, elite=2, seed=42, non_greedy_margin=0.01, future_margin=0.01,
    )
    assert chosen.item() == 1
    assert seen == [(0, 42), (1, 42), (2, 42)]
    assert receipt["eligible_count"] == 1
    assert receipt["selected_suffix_parameters"] == [[1.0], [1.0]]
    assert [row["suffix_parameters"] for row in receipt["proposal_rows"]] == [
        [[0.0], [0.0]], [[1.0], [1.0]], [[2.0], [2.0]]]
    assert receipt["predict_sample_evaluations"] == 2 * 3 + 3 * 8 * 3 * 2


def test_selector_rejects_non_neutral_reference():
    with pytest.raises(ValueError, match="neutral"):
        successor.select_prefix_with_cem_suffix(
            Model(), torch.zeros(1, 3, 1), torch.zeros(1), torch.zeros(1, 3, 1),
            torch.ones(2, 1, 1), future_horizon=1, action_lower=-torch.ones(1),
            action_upper=torch.ones(1), samples=4, steps=1, elite=2, seed=0,
            non_greedy_margin=0, future_margin=0,
        )
