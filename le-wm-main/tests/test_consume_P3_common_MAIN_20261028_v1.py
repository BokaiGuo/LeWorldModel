import pytest
from experiments.spectral_reachability.consume_P3_common_MAIN_20261028_v1 import independent_cost


def test_projection_precedes_squared_norm():
    task = dict(goal=[1., 2.], projection=[[2., 1.]])
    row = dict(whole=[dict(latent_history=[[3., 4.]])], cost=36.)
    assert independent_cost(task, row) == 36.


@pytest.mark.parametrize('cost', [0., 35.999, float('nan')])
def test_wrong_or_nonfinite_saved_cost_rejected(cost):
    with pytest.raises(ValueError):
        independent_cost(dict(goal=[1., 2.], projection=[[2., 1.]]),
                         dict(whole=[dict(latent_history=[[3., 4.]])], cost=cost))
