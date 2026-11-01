"""RAM tests for metadata-only additional exclusion identities."""
import json
import math
import pytest
from experiments.spectral_reachability import chat_freshroots_20261027_prefix_identity as p
from experiments.spectral_reachability import moving_root_exclusion_manifest_20261027_v1 as v


def raw(state=None,rgb='a'*64):
    return json.dumps(dict(prefix_state=state if state is not None else [1,2,3,4,5,0,0],
                          prefix_rgb_sha256=rgb,final_state='OUTCOME_MUST_STAY_OPAQUE',
                          official={'success':'NEVER_DECODE'},cost='NOT_A_NUMBER')).encode()


def test_project_only_prefix():
    state,rgb=p.project_prefix(raw(),v.Projection)
    assert state==[1,2,3,4,5,0,0] and rgb=='a'*64


@pytest.mark.parametrize('state',[[1]*6,[1]*8,[True]*7,['1']*7])
def test_bad_state_shape_and_types(state):
    with pytest.raises(ValueError):p.project_prefix(raw(state),v.Projection)


@pytest.mark.parametrize('rgb',[None,'g'*64,'a'*63,123])
def test_bad_rgb(rgb):
    with pytest.raises(ValueError):p.project_prefix(raw(rgb=rgb),v.Projection)


def test_duplicate_metadata_key():
    data=raw().replace(b'"prefix_state":',b'"prefix_state": [], "prefix_state":')
    with pytest.raises(ValueError):p.project_prefix(data,v.Projection)


def test_trailing_data():
    with pytest.raises(ValueError):p.project_prefix(raw()+b'{}',v.Projection)


def test_no_fallback_final_state():
    data=json.dumps(dict(final_state=[1]*7,prefix_rgb_sha256='a'*64)).encode()
    with pytest.raises(ValueError):p.project_prefix(data,v.Projection)


def test_nonfinite_identity_rejected():
    state,rgb=p.project_prefix(raw([1,2,3,4,float('nan'),0,0]),v.Projection)
    with pytest.raises(ValueError):v.state_identity(state)


def test_pose_wrap_velocity_goal_independence():
    a=v.state_identity([1,2,3,4,0,0,0])
    b=v.state_identity([1,2,3,4,2*math.pi,90,80])
    assert v.same_pose(a['physical_pose'],b['physical_pose'])
    assert a['root_state_sha256']!=b['root_state_sha256']
    groups,edges=p.alias_groups([a,b],v.same_pose)
    assert groups==[[0,1]] and edges==[[0,1]]


def test_alias_components_retain_all_rows():
    a=v.state_identity([1,2,3,4,0,0,0])
    b=v.state_identity([20,2,3,4,0,0,0])
    groups,edges=p.alias_groups([a,a,b],v.same_pose)
    assert groups==[[0,1],[2]]
    assert sorted(i for group in groups for i in group)==[0,1,2]


def test_empty_alias_set():
    assert p.alias_groups([],v.same_pose)==([],[])
