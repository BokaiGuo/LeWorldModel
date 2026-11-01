import numpy as np
import pytest
from experiments.spectral_reachability.pusht_goal_set_metric import catalog_images,distances


def test_each_equivalent_goal_has_zero_cost():
    goals=np.array([[0.,0.],[10.,10.],[20.,-20.]])
    assert distances(goals,goals)['costs']==[0.,0.,0.]


def test_nearest_goal_not_mean_goal():
    assert distances([[10.,10.]],[[0.,0.],[10.,10.]])['costs']==[0.]


def test_ties_stable():
    assert distances([[1.,0.]],[[0.,0.],[2.,0.]])['goal_indices']==[0]


def test_catalog_fixed_unique_grid():
    rgb=np.full((224,224,3),255,np.uint8);rgb[160:170,30:40]=[65,105,225]
    values=list(catalog_images(rgb))
    assert len(values)==len({x['id'] for x in values})==256
    assert len({x['rgb'].tobytes() for x in values})==256


@pytest.mark.parametrize('bad',[[],[[float('nan'),0.]],[[True,False]],[[1j,0.]]])
def test_invalid_catalog(bad):
    with pytest.raises(ValueError):distances([[0.,0.]],bad)
