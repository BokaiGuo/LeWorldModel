import numpy as np
import pytest
from experiments.spectral_reachability.pusht_scene_nuisance_metric import agent_variants,compile_metric


def test_visible_object_pixels_preserved():
    image=np.full((224,224,3),255,np.uint8);image[30:40,30:40]=[119,136,153]
    image[100:110,100:110]=[65,105,225]
    for v in agent_variants(image):
        np.testing.assert_array_equal(v['rgb'][30:40,30:40],image[30:40,30:40])


def test_missing_agent_rejected():
    with pytest.raises(ValueError):agent_variants(np.zeros((224,224,3),np.uint8))


def test_scene_nuisance_directions_removed():
    goals=np.array([[0.,0.,0.,0.],[1.,0.,0.,0.]])
    extra=np.tile([0.,1.,0.,0.],(16,1))
    objects=np.tile([0.,0.,1.,0.],(6,1));r=compile_metric(goals,objects,extra)
    q=np.array(r['projection'])
    np.testing.assert_allclose(q@extra.T,0,atol=1e-14)
    np.testing.assert_allclose(q@np.array([0.,0.,1.,0.]),[0.,0.,1.,0.],atol=1e-14)
    assert r['nuisance_rank']==2
