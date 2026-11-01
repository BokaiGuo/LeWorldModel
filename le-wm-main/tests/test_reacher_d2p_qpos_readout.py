import torch

from experiments.spectral_reachability.reacher_d2p_qpos_readout import ReacherQposReadoutCostModel
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def test_qpos_readout_cost_is_circularly_small_across_the_pi_boundary():
    artifact = {"feature_mean": torch.zeros(4), "feature_scale": torch.ones(4), "coefficient": torch.eye(5, 4)}
    # Reorder identity features into [sin0, sin1, cos0, cos1].
    artifact["coefficient"] = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

    class Base(torch.nn.Module):
        def get_cost(self, info, _actions):
            info["predicted_emb"] = torch.tensor([[[[0.01, 0.0, -1.0, 1.0]]]])
            info["goal_emb"] = torch.tensor([[[0.01, 0.0, -1.0, 1.0]]])
            return torch.zeros((1, 1))

    model = ReacherQposReadoutCostModel(Base(), artifact)
    value = model.get_cost({}, torch.zeros((1, 1, 1)))
    assert float(value.item()) == 0.0
    coordinate = model.reachability_coordinate(torch.tensor([[0.0, 0.0, -1.0, 1.0]]))
    torch.testing.assert_close(coordinate, torch.tensor([[0.0, 0.0, -1.0, 1.0]]))


def test_terminal_cost_uses_declared_reachability_coordinate(monkeypatch):
    class Model:
        @staticmethod
        def reachability_coordinate(value):
            return value[..., :1]

    monkeypatch.setattr(p0, "_terminal_batch", lambda *_args, **_kwargs: torch.tensor([[1.0, 999.0]]))
    monkeypatch.setattr(p0, "_goal_latent", lambda *_args: torch.tensor([3.0, -999.0]))
    cost = p0._cost_fn(Model(), torch.empty(1), torch.empty(1), torch.empty(1), torch.empty(1), action_lower=torch.empty(1), action_upper=torch.empty(1))
    assert float(cost.item()) == 4.0


def test_terminal_cost_keeps_predicted_and_goal_charts_distinct(monkeypatch):
    class Model:
        @staticmethod
        def predicted_coordinate(value):
            return value[..., :1] + 10.0

        @staticmethod
        def goal_coordinate(value):
            return value[..., :1] - 5.0

    monkeypatch.setattr(p0, "_terminal_batch", lambda *_args, **_kwargs: torch.tensor([[1.0, 999.0]]))
    monkeypatch.setattr(p0, "_goal_latent", lambda *_args: torch.tensor([3.0, -999.0]))
    cost = p0._cost_fn(Model(), torch.empty(1), torch.empty(1), torch.empty(1), torch.empty(1), action_lower=torch.empty(1), action_upper=torch.empty(1))
    assert float(cost.item()) == 169.0


def test_relinearized_d2p_diagnostic_uses_the_same_declared_chart_as_its_solver(monkeypatch):
    class Model:
        @staticmethod
        def predicted_coordinate(value):
            return value[..., :1]

        @staticmethod
        def goal_coordinate(value):
            return value[..., :1]

    def terminal_batch(_model, _pixels, _goal, _history, params, **_kwargs):
        return torch.cat((params[:, 0, :1], torch.full_like(params[:, 0, :1], 999.0)), dim=-1)

    monkeypatch.setattr(p0, "_terminal_batch", terminal_batch)
    monkeypatch.setattr(p0, "_goal_latent", lambda *_args: torch.tensor([0.75, -999.0]))
    _params, receipt = p0._d2p_goal_progress(
        Model(), torch.empty(1), torch.empty(1), torch.zeros(1, 3, 1),
        future_horizon=1, action_lower=torch.tensor([-10.0]), action_upper=torch.tensor([10.0]),
        damping=1.0, relinearization_steps=1,
    )
    assert receipt["predicted_successor_l2_to_goal_after"] < 1.0
