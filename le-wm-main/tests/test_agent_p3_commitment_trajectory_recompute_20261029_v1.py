"""Independent synthetic context/action support tests; no model or producer imports."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_commitment_trajectory_recompute_20261029_v1 as m


def node(width=8):
    candidates=[];actions=[]
    for i in range(width):
        raw=np.full((15,2),np.float32(i/(width-1)*.1)).tolist();actions.append(raw)
        path=[dict(latent_history=np.ones((3,192)).tolist())]*3
        candidates.append(dict(index=i,status='returned',actions=raw,whole=path,segmented=path,cost=1.))
    result=dict(decision=dict(horizon=3,candidates=width),bank=dict(actions=actions,fractions=[i/(width-1) for i in range(width)]),
        candidates=candidates,winner=candidates[0],commands=candidates[0]['actions'],diagnostic=dict(
            spectra=[dict(short_nominal_reference=dict(chi=.1))],signed_gain=.2,scheduling_gain=.2))
    request=dict(task=dict(goal=[0.]*192,projection=np.eye(192).tolist(),root_context={'synthetic':True}),
        raw10=np.zeros((10,2)).tolist(),commitment=None)
    return dict(episode_id='s0_fixedH3',boundary=5,status='sealed',request=request,receipt=result,
        resource_receipt=dict(attempted={'predict_API':59},completed={'predict_API':59}))


def test_width_four_and_eight_are_not_assumed_nested():
    a=m.trace(node(8));b=m.trace(node(4));s=m.candidate_support(a,b)
    assert s['left_count']==8 and s['right_count']==4
    assert [(x['left_index'],x['right_index']) for x in s['all_shared_raw_pairs']]==[(0,0),(7,3)]
    assert s['selected_raw_equal'] and s['selected_Q_equal'] and s['same_context']


@pytest.mark.parametrize('damage',['cost','winner','action','count','projection'])
def test_full_support_native_Q_and_witness_failclosed(damage):
    n=node()
    if damage=='cost':n['receipt']['candidates'][1]['cost']+=1
    elif damage=='winner':n['receipt']['winner']=n['receipt']['candidates'][1]
    elif damage=='action':n['receipt']['commands']=np.ones((15,2)).tolist()
    elif damage=='count':n['receipt']['decision']['candidates']=4
    else:n['request']['task']['projection'][0][0]=2.
    with pytest.raises(ValueError):m.trace(n)


def test_active_commitment_no_schedule_or_reselection():
    n=node();n['request']['commitment']=dict(start=5,end=20,seal_sha256='a'*64)
    n['boundary']=10;n['status']='observed_commitment'
    n['receipt'].update(status='observed_commitment',decision=None,bank=None,candidates=[],winner=None,commands=None)
    trace=m.trace(n)
    assert trace['decision'] is None and trace['winner'] is None and trace['candidates']==[]
    assert trace['paid_predict_completed']==59  # Explicit synthetic paid scalar, not a protocol cost claim.


def test_original_png_bytes_and_full_state_observations(tmp_path):
    path=tmp_path/'x.png';path.write_bytes(b'fixture_png_bytes_not_decoder_evidence')
    e=dict(kind='fresh_observation',episode_id='s0_fixedH3',receipt=dict(phase='plan',step=5,state=[0.]*7,
        goal=[1.]*7,snapshot={'original':'state'},goal_pose=[1.,2.,3.],goal_rgb_sha256='a'*64,
        pixel_sha256='b'*64,path=str(path),file_sha256=m.numeric.sha(path)))
    result=m.observations(tmp_path,[e],'s0_fixedH3')
    assert result['plan',5]['png_bytes']==path.read_bytes()
    bad=deepcopy(e);bad['receipt']['file_sha256']='0'*64
    with pytest.raises(ValueError):m.observations(tmp_path,[bad],'s0_fixedH3')


def test_raw_float32_exact_not_silently_repaired():
    with pytest.raises(ValueError):m.raw([[.1,0.]])
    assert m.bits(np.zeros((5,2),np.float32))!=m.bits(np.zeros((15,2),np.float32))
