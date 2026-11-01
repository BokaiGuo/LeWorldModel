from copy import deepcopy
import pytest
from experiments.spectral_reachability.p1_fixed40_reach_physical_join_20261029 import join


def fixture():
    panel = {'rows': []}
    original = {'rows': []}
    for i in range(8):
        panel['rows'].append(dict(anchor_index=i, anchor_id=str(i), task_id='t'+str(i),
                                  cases={k:{'solve':{'own_reach_value':i}} for k in ('predicted','observed')}))
        for j in range(5):
            original['rows'].append(dict(index=5*i+j, identity={'anchor_id':str(i),'task_id':'t'+str(i)},
                suffix_hex='00', suffix={'value':1}, observed_terminal={'value':None}, physical={'value':{}}))
    return panel, original


def test_join_preserves_missing_observation():
    p, o = fixture()
    assert join(p,o)['rows'][0]['candidates'][0]['actual_terminal'] is None


def test_join_rejects_mismatched_task_and_order():
    p, o = fixture()
    bad = deepcopy(o); bad['rows'][0]['identity']['task_id'] = 'wrong'
    with pytest.raises(ValueError): join(p,bad)
    bad = deepcopy(o); bad['rows'][0]['index'] = 3
    with pytest.raises(ValueError): join(p,bad)
