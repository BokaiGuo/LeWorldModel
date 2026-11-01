from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p5_prospective_ridge_array_audit_main_20261027_v1 as a


def fixture():
    j=np.arange(200,dtype=np.float32).reshape(20,10)/200
    goal=np.ones(20,np.float32);nominal=np.zeros(20,np.float32)
    gram=j.astype(float).T@j/20;rhs=j.astype(float).T@goal/20
    delta=np.linalg.solve(gram+.01*np.eye(10),rhs)
    arrays=dict(response_columns=j,goal_features=goal,nominal_successor=nominal,
        ridge_successor=np.full(20,.5,np.float32),selected_replay=np.full(20,.5,np.float32),
        gram=gram,rhs=rhs,ridge_delta=delta)
    candidates=[dict(id='nominal',env_packet=np.zeros((5,2)).tolist(),independent_goal_mse=1.),
        dict(id='ridge',env_packet=np.full((5,2),.1).tolist(),independent_goal_mse=.25)]
    return arrays,candidates


def test_independent_numeric_solution():
    arrays,candidates=fixture();out=a.arithmetic(arrays,candidates,'ridge')
    assert out['all_checks'] and out['solve_residual_norm']<1e-12


@pytest.mark.parametrize('role',['gram','rhs','ridge_delta','selected_replay'])
def test_numeric_corruption(role):
    arrays,candidates=fixture();arrays[role]=arrays[role].copy();arrays[role].flat[0]+=.1
    assert not a.arithmetic(arrays,candidates,'ridge')['all_checks']


def test_wrong_selection_and_score_rejected():
    arrays,candidates=fixture()
    assert not a.arithmetic(arrays,candidates,'nominal')['all_checks']
    candidates[0]['independent_goal_mse']=.9
    assert not a.arithmetic(arrays,candidates,'ridge')['all_checks']


def test_exact_tie_nominal_first():
    arrays,candidates=fixture();arrays['ridge_successor']=arrays['nominal_successor'].copy()
    arrays['selected_replay']=arrays['nominal_successor'].copy();candidates[1]['independent_goal_mse']=1.
    assert a.arithmetic(arrays,candidates,'nominal')['all_checks']


def test_nonfinite_and_missing_candidate():
    arrays,candidates=fixture();arrays['response_columns'][0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):a.arithmetic(arrays,candidates,'ridge')
    arrays,candidates=fixture()
    with pytest.raises(ValueError,match='census'):a.arithmetic(arrays,candidates[:1],'nominal')


def test_bounds_and_nominal_raw_zero():
    arrays,candidates=fixture();candidates[1]['env_packet'][0][0]=1.1
    assert not a.arithmetic(arrays,candidates,'ridge')['all_checks']
    candidates[0]['env_packet'][0][0]=.01
    assert not a.arithmetic(arrays,candidates,'ridge')['checks']['native_zero_nominal']


def test_npy_trailing_payload_and_dtype(tmp_path):
    path=tmp_path/'array.npy';np.save(path,np.ones((3,2),np.float32))
    assert a.load_array(path,a.sha(path),(3,2),np.float32,tmp_path).shape==(3,2)
    with pytest.raises(ValueError,match='layout'):a.load_array(path,a.sha(path),(3,2),np.int32,tmp_path)
    with path.open('ab') as stream:stream.write(b'extra')
    with pytest.raises(ValueError,match='trailing'):a.load_array(path,a.sha(path),(3,2),np.float32,tmp_path)


def test_symlink_and_receipt_sha(tmp_path):
    path=tmp_path/'array.npy';np.save(path,np.ones((3,2),np.float32))
    link=tmp_path/'link.npy';link.symlink_to(path)
    with pytest.raises(ValueError,match='non-symlink'):a.sha(link)
    with pytest.raises(ValueError,match='SHA'):a.load_array(path,'0'*64,(3,2),np.float32,tmp_path)


def test_json_duplicate_and_exclusive_output(tmp_path):
    path=tmp_path/'dup.json';path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError,match='duplicate'):a.read_json(path)
    other=tmp_path/'out.json';a.write(other,dict(ok=True))
    with pytest.raises(FileExistsError):a.write(other,dict(ok=False))


def test_source_prepare_no_neural_import():
    prepared=a.prepare()
    assert len(prepared['seals'])==3 and prepared['neural_calls']==0 and not prepared['P5_admission']


def coordinate_fixture():
    mean=np.array([-.0087,.0068],np.float32);std=np.array([.2019,.2002],np.float32)
    delta=np.linspace(-.1,.1,10);ridge=delta.reshape(5,2)*std.astype(float)
    nominal=np.zeros((5,2));history=np.full((2,5,2),.05)
    norm=lambda x:(((100*x).astype(np.float32)/np.float32(100)-mean)/std).tolist()
    receipt=dict(action_coordinates=dict(action_mean_float32=mean.tolist(),action_std_float32=std.tolist(),
        nominal_native_u64=nominal.tolist(),nominal_normalized_f32=norm(nominal),history_native_u64=history.tolist(),
        history_normalized_f32=norm(history),ridge_native_u64=ridge.tolist(),ridge_normalized_f32=norm(ridge)),
        candidates=[dict(id='nominal',env_packet=nominal.tolist()),dict(id='ridge',env_packet=ridge.tolist())],
        selected_id='ridge',selected_env_packet=ridge.tolist())
    return receipt,delta


def test_coordinates_and_packet_selection():
    receipt,delta=coordinate_fixture()
    assert all(a.coordinates(receipt,delta).values())
    receipt['selected_env_packet'][0][0]+=.01
    assert not a.coordinates(receipt,delta)['selected_packet_exact']


def test_coordinates_reject_double_scale_and_history_corruption():
    receipt,delta=coordinate_fixture();receipt['candidates'][1]['env_packet'][0][0]/=100
    assert not a.coordinates(receipt,delta)['ridge_centered_inverse_exact']
    receipt,delta=coordinate_fixture();receipt['action_coordinates']['history_normalized_f32'][0][0][0]+=.1
    assert not a.coordinates(receipt,delta)['normalized_history_exact']


def test_replay_signed_zero_is_not_bitwise_equal():
    arrays,candidates=fixture();arrays['ridge_successor']=np.zeros(20,np.float32)
    arrays['selected_replay']=np.zeros(20,np.float32);arrays['selected_replay'][0]=-0.
    candidates[1]['independent_goal_mse']=1.
    assert not a.arithmetic(arrays,candidates,'nominal')['checks']['selected_replay_exact']


def test_saved_coordinate_arrays_are_checked_not_just_receipt():
    receipt,delta=coordinate_fixture();c=receipt['action_coordinates']
    inputs=dict(normalized_history=np.asarray(c['history_normalized_f32'],np.float32).reshape(1,2,10),
        normalized_nominal=np.asarray(c['nominal_normalized_f32'],np.float32).reshape(10),
        normalized_ridge=np.asarray(c['ridge_normalized_f32'],np.float32).reshape(10))
    assert all(a.coordinates(receipt,delta,inputs).values())
    for name in inputs:
        corrupt={k:v.copy() for k,v in inputs.items()};corrupt[name].flat[0]+=.2
        assert not all(a.coordinates(receipt,delta,corrupt).values())


@pytest.mark.parametrize('errors',[None,False,{},''])
def test_falsey_nonlist_finalization_errors_rejected(tmp_path,errors):
    receipt=dict(schema='p5_prospective_control_bridge_20261027_v1.compiler',
        status='selected_packet_replayed',backend='host_cuda_only',physical_calls=0,
        used_observed_frame15=False,partial_work_unknown=False,selected_replay_exact=True,
        finalization_errors=errors)
    path=tmp_path/'receipt.json';a.write(path,receipt)
    with pytest.raises(ValueError,match='completed prospective'):
        a.audit(a.prepare(),path,a.sha(path))
