from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability.audit_pusht_simulator_vs_edited_goal_catalog_20261027 import common_catalogs


def catalogs():
    return (dict(root_sha256='root',ids=['source_goal_0','agent_2_2','agent_1_1'],vectors=[[99,99],[2,2],[1,1]]),
            dict(root_sha256='root',ids=['agent_1_1','agent_2_2','agent_3_3'],vectors=[[10,10],[20,20],[30,30]]))


def test_stable_identity_not_array_index_or_favorable_cost():
    a,b=catalogs();saved=deepcopy((a,b));ids,x,y=common_catalogs(a,b)
    assert (a,b)==saved
    assert ids==['agent_1_1','agent_2_2']
    np.testing.assert_array_equal(x,[[1,1],[2,2]])
    np.testing.assert_array_equal(y,[[10,10],[20,20]])


@pytest.mark.parametrize('change',['root','duplicate','missing_vectors','nonfinite','dimensions','no_common'])
def test_bad_catalogs_rejected(change):
    a,b=catalogs()
    if change=='root':b['root_sha256']='other'
    elif change=='duplicate':b['ids'][1]=b['ids'][0]
    elif change=='missing_vectors':b['vectors'].pop()
    elif change=='nonfinite':b['vectors'][0][0]=float('nan')
    elif change=='dimensions':b['vectors']=[[1],[2],[3]]
    elif change=='no_common':b['ids']=['agent_4_4','agent_5_5','agent_6_6']
    with pytest.raises(ValueError):common_catalogs(a,b)
