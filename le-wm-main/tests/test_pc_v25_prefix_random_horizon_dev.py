import numpy as np
import pytest

from experiments.spectral_reachability import pc_global_reachability_v25 as original
from experiments.spectral_reachability import pc_v25_prefix_random_horizon_dev as policy
from experiments.spectral_reachability import pc_v25_prefix_random_refinement_dev as variant


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_horizon_policy_matches_declared_branch_and_budget(segments):
    nominal = tuple(tuple((0., 0.) for _ in range(5)) for _ in range(2))
    config = policy.Config(action_dim=2, segment_length=5, segments=segments,
                           nominals=nominal, token_cap=1920, seed=43, max_modes=2)
    carry = np.full((segments * 5, 2), 0.1, dtype=np.float32)

    def run(compiler):
        calls = []

        def predict(actions):
            calls.append(actions.copy())
            state = actions.sum(axis=1)
            return state, np.linalg.norm(state - [1., -1.], axis=1)

        return compiler(predict, np.array([1., -1.]), config,
                        score_scale=1., score_target=0., carry=carry), calls

    actual, calls = run(policy.compile_program)
    reference, expected = run(
        original.compile_program if segments == 1 else
        lambda *a, **kw: variant.compile_program(*a, **kw, random_refinement=True)
    )
    assert len(calls) == len(expected)
    for left, right in zip(calls, expected, strict=True):
        np.testing.assert_array_equal(left, right)
        assert left.shape[2] == 2
        assert 0 < left.shape[1] <= segments * 5
        assert left.shape[1] % 5 == 0
    np.testing.assert_array_equal(actual["selected"], reference["selected"])
    assert actual["resource"]["prediction_tokens"] == reference["resource"]["prediction_tokens"]
    assert actual["resource"]["prediction_tokens"] <= config.token_cap
    assert actual["horizon_policy"]["random_tail_enabled"] == (segments > 1)
    assert actual["horizon_policy"]["fallback_reason"] == (
        "no_tail_after_first_segment" if segments == 1 else None
    )


def test_fixed_terminal_feedback_rescores_old_tail_with_fresh_observation():
    nominal = tuple(tuple((0., 0.) for _ in range(5)) for _ in range(2))
    state = np.array([0., 0.])
    carry = None
    dispatched = []
    for step in range(3):
        config = policy.Config(action_dim=2, segment_length=5, segments=3 - step,
                               nominals=nominal, token_cap=1920, seed=43 + step, max_modes=2)
        observation = state.copy()
        calls = []

        def predict(actions):
            calls.append(actions.copy())
            successor = observation + actions.sum(axis=1)
            return successor, np.linalg.norm(successor - [1., -1.], axis=1)

        result = policy.compile_program(predict, np.array([1., -1.]), config,
                                        score_scale=1., score_target=0., carry=carry)
        if carry is not None:
            index = next(i for i, action in enumerate(calls[0]) if np.array_equal(action, carry))
            expected = np.linalg.norm(observation + carry.sum(axis=0) - [1., -1.])
            assert result["global_proposals"]["scores"][index] == pytest.approx(expected)
        chain = np.asarray(result["selected"], np.float32)
        assert chain.shape == (15 - step * 5, 2)
        dispatched.append(chain[:5].copy())
        state += chain[:5].sum(axis=0)
        carry = chain[5:].copy() if step < 2 else None
    assert sum(len(part) for part in dispatched) == 15
