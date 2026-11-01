from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e1_action_identity_cost_pair_20261029_v1 as m


def row():
    cp=dict(step=0,state=[0.]*7,snapshot={'x':1},pixel_sha256='pixels')
    return dict(root_slot=0,arm='a',selection_digest='s',selected_plan={'commands':[[0.,0.] for _ in range(10)]},
                own_returned=0,own_attempted=1,status='native_stopped',terminated=True,truncated=False,
                first_error=None,secondary_errors=[],checkpoints=[cp])


def test_missing_checkpoint_is_not_filled_and_failure_preserved():
    a=m.identity(row(),'subgoal');b=m.identity(row(),'forced')
    assert m.classify(a,b,5)['same_observation'] is None
    assert a['attempted']==1 and a['returned']==0 and a['terminated']
    assert m.classify(a,b,0)['same_observation'] is True


def test_bitwise_identity_preserves_signed_zero():
    a=row();b=deepcopy(a);b['selected_plan']['commands'][9][1]=-0.
    x=m.identity(a,'subgoal');y=m.identity(b,'forced')
    assert x['plan_raw5_sha256']==y['plan_raw5_sha256']
    assert x['plan_raw10_sha256']!=y['plan_raw10_sha256']
    assert len(m.group_identities([x,y]))==2


def test_observation_difference_is_visible():
    a=m.identity(row(),'subgoal');b=deepcopy(a);b['checkpoints']['0']['pixel_sha256']='other'
    assert m.classify(a,b,0)['same_observation'] is False
    assert m.classify(a,b,0)['same_state'] is True


def test_checkpoint_after_stop_rejected():
    a=row();a['checkpoints'][0]['step']=5
    with pytest.raises(ValueError):m.identity(a,'subgoal')


def test_cost_join_fails_closed_on_bad_binding():
    with pytest.raises(ValueError):m.slot_costs([],{'rows':[]},{'rows':[]},{'report_sha256':'x'},
                                               {'receipt_sha256':'bad'},{'receipt_sha256':'bad'})
