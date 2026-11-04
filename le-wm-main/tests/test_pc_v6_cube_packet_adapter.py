import numpy as np
import pytest
import torch

from experiments.spectral_reachability import pc_v6_cube_packet_adapter as adapter


class ToyModel:
    def action_encoder(self, actions):
        return actions

    def predict(self, embeddings, actions):
        # Nonlinear dependence on all causal slots makes shifts observable.
        increment = torch.stack((actions.sum(dim=(1, 2)), (actions ** 2).sum(dim=(1, 2)),
                                 actions[:, -1, 0]), dim=-1) * .001
        output = embeddings.clone()
        output[:, -1] = embeddings[:, -1] + increment
        return output


@pytest.mark.parametrize("steps", [5, 10, 15])
def test_packet_order_and_cached_root_anchored_rollout(steps):
    mean = np.tile(np.linspace(-.2, .2, 5, dtype=np.float32), 5)
    std = np.tile(np.linspace(.5, 1., 5, dtype=np.float32), 5)
    actions = np.arange(2 * steps * 5, dtype=np.float32).reshape(2, steps, 5) / 300
    tokens = adapter.raw_to_packets(actions, mean, std, device="cpu")
    np.testing.assert_allclose(tokens.numpy(), (actions.reshape(2, -1, 25) - mean) / std)

    encoded = torch.tensor([[[.1, .2, .3], [.2, .3, .4], [.3, .4, .5]]])
    history = torch.arange(75, dtype=torch.float32).reshape(1, 3, 25) / 100
    goal = torch.tensor([.7, -.1, .9])
    model = ToyModel()
    predict, target = adapter.make_predictor(model, encoded, history, goal, mean, std, chart_seed=42)
    position, score = predict(actions)
    chart = adapter.goal_chart(encoded, goal, 42)
    expected = []
    for row in tokens:
        embeddings = encoded.clone()
        sequence = torch.cat((history[0, 1:], row), dim=0)
        for index in range(len(row)):
            context = sequence[index:index + 3].unsqueeze(0)
            terminal = model.predict(embeddings[:, -3:], context)[:, -1]
            embeddings = torch.cat((embeddings, terminal[:, None]), dim=1)
        expected.append(terminal[0])
    expected = torch.stack(expected)
    np.testing.assert_allclose(position, (expected @ chart).numpy(), atol=1e-6)
    np.testing.assert_allclose(score, ((expected - goal) ** 2).mean(-1).numpy(), atol=1e-6)
    np.testing.assert_allclose(target, (goal @ chart).numpy())


@pytest.mark.parametrize("bad", [np.nan, 1.0001, -1.0001])
def test_rejects_nonfinite_or_out_of_bounds(bad):
    actions = np.zeros((1, 5, 5))
    actions[0, 0, 0] = bad
    with pytest.raises(ValueError):
        adapter.raw_to_packets(actions, np.zeros(25), np.ones(25), device="cpu")


def test_rejects_invalid_statistics():
    with pytest.raises(ValueError):
        adapter.raw_to_packets(np.zeros((1, 5, 5)), np.zeros(25), np.zeros(25), device="cpu")
