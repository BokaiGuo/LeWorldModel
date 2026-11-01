import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_p1_diag32_cp5_own_suffix_20261101_exclusive as subject


def test_fixed_roster_greedy_and_matched_budget():
    protocol = subject.prepare()
    assert protocol["roles"] == ["selected", "zero", "same_bank_greedy"]
    assert protocol["greedy_candidate"] == 4
    assert protocol["budget_per_role"]["predict"] == 8 * 33 * 2
    assert protocol["budget_per_role"]["reverse"] == 8 * 32
    assert not np.array_equal(protocol["prefixes"]["selected"], protocol["prefixes"]["same_bank_greedy"])


def test_cp5_context_uses_own_actual_prefix_and_latent():
    base, bank, _, _ = subject.blind.load(0)
    plan = subject.prepare()
    prefix = np.asarray(plan["prefixes"]["zero"], np.float32)
    warm = np.asarray(plan["raw_warmup10"], np.float32)[5:]
    latent = np.linspace(-1, 1, 192, dtype=np.float32)
    result = subject.feedback_task(base, prefix, prefix, warm, latent, {"role": "zero"})
    assert result["root_context"]["latent_history"][-1] == latent.tolist()
    assert result["root_context"]["action_history"] == plan["cp5_action_histories_expected"]["zero"]
    assert result["goal"] == base["goal"]
    with pytest.raises(ValueError):
        subject.feedback_task(base, prefix, bank[0, 0], warm, latent, {"role": "zero"})
    with pytest.raises(ValueError):
        subject.feedback_task(base, prefix, prefix, warm, latent[:191], {"role": "zero"})


def test_equal_fixed_work_projection_and_tie():
    bank = torch.zeros((8, 2, 5, 2))
    calls, reverses, records = [], [], []
    def evaluate(candidate, iteration, action):
        calls.append((candidate, iteration))
        return -action.sum()
    def reverse(candidate, iteration, loss, action):
        reverses.append((candidate, iteration))
        return torch.autograd.grad(loss, action)[0]
    winner = subject.optimize_suffix_bank(bank, evaluate, reverse, records.append)
    assert len(calls) == len(records) == 8 * 33
    assert len(reverses) == 8 * 32
    assert (winner["candidate"], winner["iteration"]) == (0, 20)
    assert np.max(np.abs(winner["actions"])) <= 1
