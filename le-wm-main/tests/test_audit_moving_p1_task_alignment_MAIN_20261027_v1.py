import numpy as np
import pytest
from experiments.spectral_reachability import audit_moving_p1_task_alignment_MAIN_20261027_v1 as m


def test_cross_term_can_make_actual_better_than_prediction():
    result = m.alignment(np.ones(192), np.zeros(192), np.zeros(192), 1., 0.)
    assert result['prediction_MSE'] == 1.
    assert result['error_goal_cross_term'] == -2.
    assert result['actual_minus_predicted_Q'] == -1.


def test_full_residual_identity_for_nonzero_goals():
    rng = np.random.default_rng(42)
    pred, real, goal = rng.normal(size=(3, 192))
    qp, qa = np.mean((pred-goal)**2), np.mean((real-goal)**2)
    result = m.alignment(pred, real, goal, qp, qa)
    assert result['actual_minus_predicted_Q'] == pytest.approx(result['error_square_term']+result['error_goal_cross_term'])


@pytest.mark.parametrize('bad', [np.zeros(191), np.zeros((1,192)), np.full(192, np.nan), np.full(192, np.inf)])
@pytest.mark.parametrize('slot', range(3))
def test_bad_latent_shape_or_values_rejected(bad, slot):
    values = [np.zeros(192) for _ in range(3)]
    values[slot] = bad
    with pytest.raises(ValueError):
        m.alignment(*values, 0., 0.)


@pytest.mark.parametrize('costs', [(192., 1.), (1., 192.), (2., 1.), (1., 2.)])
def test_sum_mean_confusion_and_tampered_costs_rejected(costs):
    with pytest.raises(ValueError):
        m.alignment(np.ones(192), np.ones(192), np.zeros(192), *costs)
