import numpy as np
import pytest
from experiments.spectral_reachability.pusht_goal_object_tangent_metric import variants,compile_metric


def test_variants_keep_agent_pixels_and_no_wrap():
    rgb=np.full((224,224,3),255,np.uint8);rgb[80:120,90:110]=[119,136,153]
    rgb[70:80,90:100]=[65,105,225]
    results=variants(rgb)
    assert len(results)==6
    for v in results:
        np.testing.assert_array_equal(v['rgb'][70:80,90:100],rgb[70:80,90:100])
        np.testing.assert_array_equal(v['rgb'][0],rgb[0])
    assert not np.array_equal(results[0]['rgb'],results[1]['rgb'])


def test_no_object_fails_closed():
    with pytest.raises(ValueError):variants(np.zeros((224,224,3),np.uint8))


def test_projector_preserves_task_rejects_nuisance():
    goals=np.array([[0.,0.,0.],[1.,0.,0.]])
    objects=np.array([[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.],[1.,1.,0.],[1.,0.,1.]])
    r=compile_metric(goals,objects);q=np.array(r['projection'])
    np.testing.assert_allclose(q,np.diag([0.,1.,1.]),atol=1e-14)
    np.testing.assert_allclose(q@q,q,atol=1e-14)
    assert r['task_rank']==2 and r['P1_admission'] is False


def test_nuisance_only_task_rejected():
    with pytest.raises(ValueError):compile_metric([[0.,0.],[1.,0.]],[[1.,0.]]*6)


@pytest.mark.parametrize('bad',[np.full((6,3),np.nan),np.ones((6,3),dtype=bool),np.ones((6,3))*1j])
def test_nonreal_invalid_embeddings(bad):
    with pytest.raises(ValueError):compile_metric([[0.,0.,0.],[1.,0.,0.]],bad)
