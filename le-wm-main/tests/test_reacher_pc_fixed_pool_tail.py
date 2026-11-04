import torch

from experiments.spectral_reachability.reacher_pc_fixed_pool_tail import cem_fixed_pool_with_tail
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def test_tail_evaluation_does_not_change_candidate_population():
    batches = []
    tail = torch.full((1, 2, 3), 99.0)

    def cost(x):
        batches.append(x.clone())
        return x.square().sum(dim=(1, 2))

    chosen, receipt = cem_fixed_pool_with_tail(cost, tail, samples=8, steps=2, elite=3, seed=8)
    assert [len(b) for b in batches] == [8, 8, 1]
    assert torch.equal(chosen, torch.tensor(receipt["original_best_plan_params"]))
    assert not receipt["selected_old_tail"]
    assert receipt["model_predict_sample_evaluations"] == 34
    other_batches = []

    def other_cost(x):
        other_batches.append(x.clone())
        return x.square().sum(dim=(1, 2))

    cem_fixed_pool_with_tail(other_cost, torch.zeros_like(tail), samples=8, steps=2, elite=3, seed=8)
    assert all(torch.equal(a, b) for a, b in zip(batches[:2], other_batches[:2]))


def test_strict_better_tail_is_selected():
    tail = torch.zeros((1, 2, 2))

    def cost(x):
        if len(x) == 1:
            return torch.tensor([-1.0])
        return torch.arange(len(x), dtype=torch.float32)

    chosen, receipt = cem_fixed_pool_with_tail(cost, tail, samples=8, steps=2, elite=3, seed=9)
    assert torch.equal(chosen, tail)
    assert receipt["selected_old_tail"]
    assert receipt["forward_rollout_evaluations"] == 17


def test_identical_tail_is_not_reported_as_replacement_despite_score_noise():
    seen = []
    def cost(x):
        seen.append(x.clone())
        if len(x) == 1:
            return torch.tensor([-1.0])
        return torch.arange(len(x), dtype=torch.float32)

    # For the first generation index zero is the known all-zero mean.
    tail = torch.zeros((1, 2, 2))
    result, receipt = cem_fixed_pool_with_tail(cost, tail, samples=8, steps=1, elite=3, seed=9)
    assert torch.equal(result, tail)
    assert receipt["tail_equals_cem_parameters"]
    assert not receipt["selected_old_tail"]
    assert receipt["tail_model_cost"] < receipt["baseline_model_cost"]


def test_fixed_pool_matches_original_cem(monkeypatch):
    def score(batch):
        return (batch - 0.4).square().sum(dim=(1, 2))

    monkeypatch.setattr(p0, "_cost_fn", lambda model, pixels, goal, history, candidates, **kwargs: score(candidates))
    history = torch.zeros((1, 3, 4))
    original, _ = p0._cem(None, None, None, history, future_horizon=2,
                          action_lower=None, action_upper=None,
                          samples=8, steps=3, elite=3, seed=74)
    _, receipt = cem_fixed_pool_with_tail(score, torch.full((1, 2, 4), 20.0),
                                         samples=8, steps=3, elite=3, seed=74)
    assert torch.equal(original, torch.tensor(receipt["original_best_plan_params"]))
