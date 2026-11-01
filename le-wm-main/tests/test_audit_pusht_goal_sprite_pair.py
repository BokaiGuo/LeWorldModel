import pytest
from experiments.spectral_reachability.audit_pusht_goal_sprite_pair import compare


def test_same_goal_ids_not_array_order():
    a=dict(root_sha256='root',ids=['b','a','unused'],vectors=[[2.,0.],[0.,0.],[1.,0.]])
    b=dict(root_sha256='root',ids=['a','b'],vectors=[[0.,0.],[2.,0.]])
    r=compare(a,b,[[1.,0.]])
    assert r['old_common_costs']==r['clean_costs']==[.5]


def test_missing_shared_goal_rejected():
    a=dict(root_sha256='root',ids=['a'],vectors=[[0.,0.]])
    b=dict(root_sha256='root',ids=['b'],vectors=[[0.,0.]])
    with pytest.raises(ValueError):compare(a,b,[[1.,0.]])
