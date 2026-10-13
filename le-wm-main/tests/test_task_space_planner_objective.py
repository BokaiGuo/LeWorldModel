import torch

from experiments.run_causal_visual_pusht_world_model import PUSHT_RENDER_SIZE, task_space_action_cost


def test_task_space_cost_prefers_correct_orientation_without_wrap_discontinuity():
    goal = torch.tensor([[0.0, 0.0, -1.0, 0.0]])
    current = torch.tensor([[10.0, 10.0, 0.0, 0.0]])
    correct = torch.tensor([[0.0, 0.0, 0.0, 0.0, -1.0, 0.0]])
    opposite = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    assert task_space_action_cost(correct, goal, current)[0] < task_space_action_cost(opposite, goal, current)[0]


def test_task_space_cost_is_finite_for_zero_orientation_vectors():
    predicted = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]])
    goal = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    current = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    assert torch.isfinite(task_space_action_cost(predicted, goal, current)).all()


def test_push_t_render_scale_is_the_declared_environment_scale():
    assert PUSHT_RENDER_SIZE == 224.0
