import numpy as np

from experiments.spectral_reachability import p1_unified_constrained_cem_20261101_codex as subject


def row(q1, q3, *, root=1.0, iteration=0, candidate=0):
    return {
        "q1": q1,
        "q3": q3,
        "violation": max(0.0, root - q1),
        "iteration": iteration,
        "candidate": candidate,
    }


def test_candidate_key_prefers_feasible_then_terminal_cost():
    infeasible = row(0.99, 0.1)
    feasible = row(1.0, 0.8, candidate=1)
    better = row(1.2, 0.4, candidate=2)
    assert subject.candidate_key(feasible) < subject.candidate_key(infeasible)
    assert subject.candidate_key(better) < subject.candidate_key(feasible)


def test_choose_requires_non_greedy_prefix_and_terminal_improvement():
    rows = [
        row(0.9, 0.1),
        row(1.0, 1.1, candidate=1),
        row(1.1, 0.6, candidate=2),
        row(1.2, 0.5, candidate=3),
    ]
    assert subject.choose(1.0, rows)["candidate"] == 3
    assert subject.choose(1.0, rows[:2]) is None


def test_bank_is_deterministic_and_contains_authenticated_initial_bank():
    rng1 = np.random.default_rng(subject.SEED)
    rng2 = np.random.default_rng(subject.SEED)
    mean = np.zeros((subject.HORIZON, 5, 2), np.float32)
    std = np.full_like(mean, subject.INITIAL_STD)
    initial = np.arange(8 * subject.HORIZON * 5 * 2, dtype=np.float32).reshape(
        8, subject.HORIZON, 5, 2
    )
    initial = np.clip(initial / initial.max(), -1, 1)
    first = subject.make_bank(rng1, mean, std, initial, 0)
    second = subject.make_bank(rng2, mean, std, initial, 0)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], mean)
    np.testing.assert_array_equal(first[1:9], initial)
    assert first.dtype == np.float32 and np.max(np.abs(first)) <= 1


def test_update_uses_fixed_elite_count_and_bounds_std():
    bank = np.zeros((subject.SAMPLES, subject.HORIZON, 5, 2), np.float32)
    for index in range(subject.SAMPLES):
        bank[index].fill(index / subject.SAMPLES)
    rows = [row(1.1, float(subject.SAMPLES - index), candidate=index)
            for index in range(subject.SAMPLES)]
    mean, std, elites = subject.update(bank, rows)
    assert len(elites) == subject.ELITES
    assert elites[0] == subject.SAMPLES - 1
    assert mean.shape == std.shape == (subject.HORIZON, 5, 2)
    assert np.all(std >= subject.MIN_STD)
