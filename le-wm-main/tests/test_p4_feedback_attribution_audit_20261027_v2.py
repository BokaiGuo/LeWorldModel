"""Pure saved-array/mock validation; no model or encoder runtime imports."""
import ast
from copy import deepcopy
import io
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p4_feedback_attribution_audit_20261027_v2 as m


def evidence():
    entries=[dict(arm=arm,kind=kind,index=i*2+j,completed=True)
             for i,arm in enumerate(m.ARMS) for j,kind in enumerate(('encoder','existing_projector'))]
    r=dict(schema=m.ENCODER+'.derived_receipt',status='two_terminal20_latents_saved',historical_runs=1,
        errors=[],first_error=None,blocked=[],forbidden_attempts={},predict_calls=0,AD_calls=0,physics_steps=0,
        training=False,P1_admission=False,P4_admission=False,efficacy_verified=False,
        model_unchanged=True,model_hash_before='a'*64,model_hash_after='a'*64,
        host=dict(device='cuda:0',allocation_sum=4.0),entries=entries,
        counts={kind:dict(attempted=2,completed=2) for kind in ('encoder','existing_projector')},
        rows=[dict(arm=arm,status='encoded_and_saved') for arm in m.ARMS])
    journal=[]
    for arm in m.ARMS:
        for e in entries:
            if e['arm']==arm:
                journal.extend([dict(e,event='attempt',completed=False),dict(e,event='return_or_error')])
        journal.append(dict(event='endpoint',row=deepcopy(next(v for v in r['rows'] if v['arm']==arm))))
    return r,journal


def test_entries_complete():
    m.validate_entries(*evidence())


@pytest.mark.parametrize('key,value',[
    ('schema','wrong'),('status','partial_or_failed'),('historical_runs',2),('errors',[{'error':'x'}]),
    ('first_error',{'error':'x'}),('blocked',[{}]),('forbidden_attempts',{'predict':1}),
    ('predict_calls',1),('AD_calls',1),('physics_steps',1),('training',True),
    ('P1_admission',True),('P4_admission',True),('efficacy_verified',True),
    ('model_unchanged',False),('model_hash_after','b'*64),('model_hash_before',None),
    ('predict_calls',False),('rows',[{'arm':'actual'},{'arm':'actual'}]),
])
def test_receipt_tamper_rejected(key,value):
    r,j=evidence(); r[key]=value
    with pytest.raises(ValueError): m.validate_entries(r,j)


@pytest.mark.parametrize('field,value',[('device','cpu'),('allocation_sum',0)])
def test_cuda_host_receipt(field,value):
    r,j=evidence(); r['host'][field]=value
    with pytest.raises(ValueError): m.validate_entries(r,j)


@pytest.mark.parametrize('change',['duplicate','missing','failed','swapped','count'])
def test_entry_census(change):
    r,j=evidence()
    if change=='duplicate': r['entries'].append(deepcopy(r['entries'][0]))
    if change=='missing': r['entries'].pop()
    if change=='failed': r['entries'][0]['completed']=False
    if change=='swapped': r['entries'][0],r['entries'][1]=r['entries'][1],r['entries'][0]
    if change=='count': r['counts']['encoder']['attempted']=3
    with pytest.raises(ValueError): m.validate_entries(r,j)


@pytest.mark.parametrize('change',['reverse','extra','lost_return','wrong_arm','wrong_endpoint'])
def test_journal_sequence(change):
    r,j=evidence()
    if change=='reverse': j.reverse()
    if change=='extra': j.append({'event':'unknown'})
    if change=='lost_return': j.pop(1)
    if change=='wrong_arm': j[0]['arm']='actual'
    if change=='wrong_endpoint': j[4]['row']['status']='failed'
    with pytest.raises(ValueError): m.validate_entries(r,j)


def endpoint():
    ep=dict(root_sha256='root',arm='imagined',primitive_step=20,event_index=160,
        rgb_path='/mock/frame.png',rgb_file_sha256='file',pixel_sha256='pixels',state=[0.]*7)
    row=dict(ep,status='encoded_and_saved',latent_shape=[192],latent_dtype='float32')
    obs=dict(ep,kind='observation',rgb_sha256='file'); obs.pop('rgb_file_sha256')
    return row,ep,obs


def test_endpoint_native_valid():
    m.validate_endpoint(*endpoint(),'root',np.zeros(192,np.float32))


@pytest.mark.parametrize('field,value',[
    ('root_sha256','wrong'),('arm','actual'),('primitive_step',15),('event_index',322),
    ('rgb_path','/wrong.png'),('pixel_sha256','wrong'),('rgb_file_sha256','wrong'),
    ('state',[1.]*7),('latent_shape',[193]),('latent_dtype','float64'),('status','failed')])
def test_endpoint_identity_rejected(field,value):
    row,ep,obs=endpoint(); row[field]=value
    with pytest.raises(ValueError): m.validate_endpoint(row,ep,obs,'root',np.zeros(192,np.float32))


@pytest.mark.parametrize('native',[np.zeros(191,np.float32),np.zeros(192,np.float64),np.full(192,np.nan,np.float32)])
def test_wrong_native(native):
    with pytest.raises(ValueError): m.validate_endpoint(*endpoint(),'root',native)


def test_observation_arm_rejected():
    row,ep,obs=endpoint(); obs['arm']='actual'
    with pytest.raises(ValueError): m.validate_endpoint(row,ep,obs,'root',np.zeros(192,np.float32))


def test_arithmetic_exact():
    result=m.terminal_metrics([0.,0.],[1.,1.],[2.,2.],np.array([3.,3.]),np.eye(2))
    assert result['predicted_Q_cost']==9 and result['observed15_Q_cost']==4
    assert result['observed20_Q_cost']==1 and result['observed_Q_cost_change']==-3
    assert result['prediction_error']['projected_mse']==4
    assert result['prediction_error']['goal_cross_term']==-12
    assert abs(result['prediction_error']['identity_error'])<1e-12


def test_nonorthogonal_Q_identity():
    goal=np.array([2.,-1.]); q=np.array([[2.,.3],[.2,1.]])
    result=m.terminal_metrics([0.,1.],[2.,3.],[1.,0.],goal,q)
    assert abs(result['prediction_error']['identity_error'])<1e-12
    assert result['observed20_Q_cost']==pytest.approx(np.mean((q@(goal-[1.,0.]))**2))


def test_zero_prediction_error_not_zero_task_cost():
    result=m.terminal_metrics([1.,1.],[0.,0.],[1.,1.],np.array([3.,3.]),np.eye(2))
    assert result['prediction_error']['raw_l2']==0 and result['observed20_Q_cost']==4


def test_latent_nan_rejected():
    with pytest.raises(ValueError): m.terminal_metrics([np.nan,0],[0,0],[1,1],np.zeros(2),np.eye(2))


def test_streamed_source_mutation(tmp_path):
    path=tmp_path/'small'; path.write_bytes(b'first')
    inputs=m.Inputs(); inputs.hash(path); path.write_bytes(b'second')
    with pytest.raises(ValueError): inputs.stable()


def test_no_runtime_neural_imports():
    tree=ast.parse(Path(m.__file__).read_text())
    modules=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): modules.extend(v.name for v in node.names)
        elif isinstance(node,ast.ImportFrom): modules.append(node.module or '')
    assert not any(any(bad in name for bad in ('torch','gym','pymunk','checkpoint_compat','terminal20_cuda_encoding','runtime')) for name in modules)


def test_new_output_only(tmp_path):
    existing=tmp_path/(m.TAG+'_exists'); existing.mkdir()
    with pytest.raises(ValueError): m.run(existing)


def test_array_pickle_not_allowed():
    stream=io.BytesIO(); np.save(stream,np.array([{'x':1}],dtype=object),allow_pickle=True)
    with pytest.raises(ValueError): np.load(io.BytesIO(stream.getvalue()),allow_pickle=False)
