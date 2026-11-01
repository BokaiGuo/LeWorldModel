"""RAM-only metadata consumer fixtures are not actual new-root evidence."""
import json
import math
import pytest
from experiments.spectral_reachability import chat_freshroots_20261027_consumer_v2 as c
from experiments.spectral_reachability import moving_root_exclusion_manifest_20261027_v1 as v
from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as r


def actuals():
    return [dict(slot=i,root_state=[100+i*10,2,3,4,0,0,0],root_goal_sha256=str(i)*64,
                 root_snapshot_sha256=str(i+4)*64,history_pixels_sha256=[str(i+1)*64]*3,
                 raw_history_sha256='a'*64) for i in range(4)]


def check(index,rows):return c.assess(index,rows,v.state_identity,v.same_pose)


def test_no_actuals_no_authorization():
    result=check([],[dict(slot=i,root_state=None) for i in range(4)])
    assert len(result)==4 and all(x['status']=='blocked_missing_actual_identity' for x in result)
    assert not any(x['source_run_authorized'] for x in result)


def test_different_poses_not_certified_and_raw_history_alone_not_duplicate():
    result=check([],actuals())
    assert all(x['status']=='no_known_duplicate_still_blocked_by_coverage' for x in result)
    assert all(not x['freshness_certified'] and not x['duplicate_evidence'] for x in result)


def test_pose_alias_velocity_angle_cannot_escape():
    rows=actuals(); state=rows[0]['root_state'][:]; state[4]=2*math.pi; state[5]=90
    old=v.state_identity(state)
    result=check([old],rows)
    assert result[0]['status']=='excluded_duplicate'


def test_batch_duplicate_both_slots_retained():
    rows=actuals(); rows[1]['root_state']=rows[0]['root_state'][:]
    result=check([],rows)
    assert result[0]['status']==result[1]['status']=='excluded_duplicate'
    assert len(result)==4


@pytest.mark.parametrize('key',['root_goal_sha256','root_snapshot_sha256','history_pixels_sha256'])
def test_historical_hash_and_batch_reject(key):
    rows=actuals();old=v.state_identity([0,0,0,0,0,0,0]);old[key]=rows[0][key]
    assert check([old],rows)[0]['status']=='excluded_duplicate'
    rows[1][key]=rows[0][key]
    out=check([],rows)
    assert out[0]['status']==out[1]['status']=='excluded_duplicate'


@pytest.mark.parametrize('field,value',[('cost',0),('success',True),('selected',True),('root_seed',203027101)])
def test_no_outcome_or_seed_reselection_input(field,value):
    rows=actuals();rows[0][field]=value
    with pytest.raises(ValueError):check([],rows)


def test_fixed_order_and_denominator():
    rows=actuals()
    with pytest.raises(ValueError):check([],rows[:3])
    with pytest.raises(ValueError):check([],rows[::-1])


@pytest.mark.parametrize('state',[[1]*6,[True]*7,[1,2,3,4,float('nan'),0,0]])
def test_invalid_actual_state(state):
    rows=actuals();rows[0]['root_state']=state
    with pytest.raises(ValueError):check([],rows)


def test_bad_actual_hash():
    rows=actuals();rows[0]['root_snapshot_sha256']='not hash'
    with pytest.raises(ValueError):check([],rows)


def test_branch_projection_keeps_outcome_opaque():
    raw=json.dumps(dict(rows=[dict(root_sha256='a'*64,branches=[dict(candidate_id='prefix0',
        prefix_state=[1,2,3,4,5,0,0],prefix_rgb_sha256='b'*64,
        final_state='OPAQUE',comparisons='OPAQUE',probes='OPAQUE')])])).encode()
    out=c.parent_prefixes(raw,'branch_prefix',r.project,v.state_identity)
    assert len(out)==1 and out[0]['ancestor_root_goal_sha256']=='a'*64


def test_probe_projection_and_no_final_state_fallback():
    raw=json.dumps(dict(rows=[dict(root_sha256='a'*64,branches=[dict(candidate_id='prefix0',probes=[
        dict(prefix_state=[1,2,3,4,5,0,0],prefix_rgb_sha256='b'*64,state='OPAQUE')])])])).encode()
    assert len(c.parent_prefixes(raw,'probe_prefix',r.project,v.state_identity))==1
    assert c.parent_prefixes(raw.replace(b'prefix_state',b'final_state'),'probe_prefix',r.project,v.state_identity)==[]


def test_join_exact_tuple_not_goal_or_filename():
    records=[dict(root_state_sha256='a'*64,prefix_rgb_sha256='b'*64)]
    parent=dict(state_sha256='a'*64,prefix_rgb_sha256='b'*64,ancestor_root_goal_sha256='c'*64,candidate_id='p')
    groups,maps=c.join_parents(records,[parent])
    assert maps[0]['status']=='explicit_ancestor_branch_associated'
    assert not maps[0]['original_sampling_state_equivalence_proved']
    assert not groups[0]['physical_root_equivalence_proved']
    _,maps=c.join_parents(records,[dict(parent,prefix_rgb_sha256='d'*64)])
    assert maps[0]['status']=='parent_mapping_missing'


def test_ambiguous_ancestors_not_resolved_by_selection():
    records=[dict(root_state_sha256='a'*64,prefix_rgb_sha256='b'*64)]
    parent=dict(state_sha256='a'*64,prefix_rgb_sha256='b'*64,ancestor_root_goal_sha256='c'*64,candidate_id='p')
    groups,maps=c.join_parents(records,[parent,dict(parent,ancestor_root_goal_sha256='d'*64)])
    assert len(groups)==2 and maps[0]['status']=='ambiguous_ancestor_mapping'
