"""Inert dict/NumPy ABI fixtures only; never HOST/native evidence."""
from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_six_target_rgb_roster_20261029_v1 as m


def fixture():
    plans=[];rows=[];prs=[];saved={}
    for i in range(6):
        z=np.arange(576,dtype=np.float32).reshape(1,3,192)/np.float32(1024)+np.float32(i//3)
        a=np.arange(20,dtype=np.float32).reshape(1,2,10)/np.float32(1024)
        raw=np.arange(30,dtype=np.float32).reshape(15,2)/np.float32(128);raw[0,0]=-0.
        target=np.full((1,192),np.float32(i/8));tokens=raw.reshape(1,1,3,10).copy()
        v=dict(z=z,a=a,target=target)
        p=dict(index=i,root_slot=i//3,condition='off',arm=m.ARMS[i%3],task_id=f'task{i//3}',
            selection_digest=f'seal{i}',controller_payload={k:x.tolist() for k,x in v.items()},
            payload_sha256={k:m.digest(x) for k,x in v.items()},commands=raw.tolist(),raw_float32_bits=raw.tobytes().hex(),
            raw_float32_sha256=m.digest(raw),native_goal=[10.,20.,30.,40.,.1,0.,0.])
        arrays=dict(z=z,a=a,target=target,add=raw.copy(),sealed=raw.copy(),tokens=tokens)
        # Explicit synthetic trace with saved HOST ABI fields, not a runtime trace
        # obtained by relabeling a core RAM run; no model has run here.
        tr=dict(backend='host_cuda_actual_tokens',completed=True,errors=[],blocked=None,
            inputs=dict(z='z',a='a',target='target',sealed_raw='sealed'),actual_tokens='tokens',
            arithmetic=[dict(kind=k,returned=True,output='add' if k=='add' else 'unused') for k in ('mul','add','sub','div')],replays=[])
        pred=np.full((1,1,3,192),.25+i/16,np.float32)
        for route in ('whole','segmented'):
            arrays[route+'_p']=pred.copy();checks=[]
            for j in range(3):
                zk,ak=f'{route}_z{j}',f'{route}_a{j}'
                arrays[zk]=np.concatenate((z,pred[0,:,:j+1]),axis=1)
                arrays[ak]=np.concatenate((a,tokens[0,:,:j+1]),axis=1);checks.append(dict(z=zk,a=ak))
            tr['replays'].append(dict(kind=route,returned=True,rollout=dict(prediction=route+'_p',checkpoints=checks,final=deepcopy(checks[-1]))))
        frames=[];responses=[]
        for step in range(16):
            state=[float(step),2.,3.,4.,.1,0.,0.]
            frames.append(dict(phase='plan',step=step,overlay_index=0,condition='off',png=f'slot{i}_cp{step}.png',
                file_sha256='a'*64,pixel_sha256='b'*64,state=state,snapshot=dict(state=state)))
            if step:responses.append(dict(step=step,state=state,terminated=False,truncated=False,reward=-1.))
        n=dict(index=i,root_slot=i//3,condition='off',arm=m.ARMS[i%3],task_id=p['task_id'],plan=deepcopy(p),
            selection_digest=p['selection_digest'],status='horizon_completed',initial_exact=True,first_error=None,secondary_errors=[],
            own_returned=15,goal=deepcopy(p['native_goal']),trajectory=frames,
            checkpoints=[{k:f[k] for k in m.FRAME_KEYS} for f in frames if f['step'] in m.CP],missing_checkpoints=[],responses=responses)
        plans.append(p);rows.append(n);prs.append(dict(slot=i,root=i//3,condition='off',arm=m.ARMS[i%3],status='returned',trace=dict(file=f'slot{i}_trace.json')))
        saved[i]=(tr,arrays)
    authority=dict(fixture_only=True)
    return (dict(schema=m.NATIVE+'.prepare',plans=plans,planner_authority=authority,arguments=dict(replay_manifest_sha='paid-manifest')),
        dict(schema=m.NATIVE+'.consume',fixed_denominator=6,errors=[],planner_authority=authority,rows=rows),
        dict(schema=m.PAID+'.report',completed=True,errors=[],manifest_sha256='paid-manifest',rows=prs),saved)


def build(f):return m.assemble(*f[:3],f[3].__getitem__)


def test_all24_source_join_without_authentication_or_scoring():
    f=fixture();before=deepcopy(f[:3]);out=build(f)
    assert f[:3]==before and out['denominator']==dict(slots=6,checkpoints=24)
    assert out['encoder_samples_available']==24 and not out['host_ready'] and not out['upstream_full_authentication_performed']
    assert out['new_neural_calls']==out['new_physics_calls']==0
    for i,row in enumerate(out['rows']):
        assert row['index']==i and row['slot']==i//4 and row['cp_public_step']==m.CP[i%4]
        assert row['metrics'] is row['latent'] is None
        assert row['prediction_origin']==('observed_history' if i%4==0 else 'paid_actual_deployment_tokens')
        assert row['prediction_pointer']['array']==('z' if i%4==0 else 'whole_p')
        assert row['frame']['state']==row['frame']['snapshot']['state']


def stop(f,step):
    n=f[1]['rows'][0];n['status']='native_stopped';n['own_returned']=step
    n['responses']=n['responses'][:step]
    if step:n['responses'][-1]['terminated']=True
    n['trajectory']=n['trajectory'][:step+1]
    n['checkpoints']=[c for c in n['checkpoints'] if c['step']<=step]
    n['missing_checkpoints']=[c for c in m.CP if c>step]


@pytest.mark.parametrize('step',[0,1,5,7,10,14,15])
def test_native_stop_keeps24_never_uses_last_as_cp(step):
    f=fixture();stop(f,step);out=build(f)
    assert len(out['rows'])==24 and out['encoder_samples_available']==20+sum(c<=step for c in m.CP)
    for row in out['rows'][:4]:
        if row['cp_public_step']>step:
            assert row['frame'] is row['response'] is None and row['status']=='missing_physical_checkpoint'
        else:assert row['frame']['step']==row['cp_public_step']


@pytest.mark.parametrize('fault',['denominator','missing_slot','slot_order','paid_schema','paid_manifest','paid_partial','authority',
    'plan','task','seal','failure','initial','goal','step','response','early_stop','cp','padding','off','snapshot','pixel_state','png'])
def test_native_join_mutations(fault):
    f=fixture();p,c,r,s=f;n=c['rows'][0]
    if fault=='denominator':c['fixed_denominator']=4
    elif fault=='missing_slot':r['rows'].pop()
    elif fault=='slot_order':r['rows'][0]['slot']=1
    elif fault=='paid_schema':r['schema']='normalized.report'
    elif fault=='paid_manifest':r['manifest_sha256']='wrong'
    elif fault=='paid_partial':r['completed']=False
    elif fault=='authority':c['planner_authority']={}
    elif fault=='plan':n['plan']['raw_steps']=10
    elif fault=='task':n['task_id']='other'
    elif fault=='seal':n['selection_digest']='other'
    elif fault=='failure':n['first_error']='failed'
    elif fault=='initial':n['initial_exact']=False
    elif fault=='goal':n['goal'][0]+=1
    elif fault=='step':n['own_returned']=10
    elif fault=='response':n['responses'][0]['step']=2
    elif fault=='early_stop':n['responses'][1]['terminated']=True
    elif fault=='cp':n['checkpoints'].pop()
    elif fault=='padding':stop(f,7);n['missing_checkpoints']=[]
    elif fault=='off':n['trajectory'][5]['overlay_index']=1
    elif fault=='snapshot':n['trajectory'][5]['snapshot']=dict(state=[0.]*7)
    elif fault=='pixel_state':n['responses'][4]['state']=[1.]*7
    else:n['trajectory'][5]['png']='../other.png'
    with pytest.raises(ValueError):build(f)


@pytest.mark.parametrize('fault',['backend','blocked','incomplete','input','hash','raw','signedzero','actual_add','sealed',
    'arithmetic','tokens','replaycount','kind','returned','prediction','history_z','history_a','final'])
def test_actual_token_prediction_mutations(fault):
    f=fixture();tr,a=f[3][0]
    if fault=='backend':tr['backend']='normalized'
    elif fault=='blocked':tr['blocked']='domain'
    elif fault=='incomplete':tr['completed']=False
    elif fault=='input':a['z']=a['z']+np.float32(1)
    elif fault=='hash':f[0]['plans'][0]['payload_sha256']['z']='wrong';f[1]['rows'][0]['plan']=deepcopy(f[0]['plans'][0])
    elif fault=='raw':a['add']=a['add']+np.float32(1)
    elif fault=='signedzero':a['add']=a['add'].copy();a['add'][0,0]=0.
    elif fault=='actual_add':tr['arithmetic'][1]['output']='tokens'
    elif fault=='sealed':a['sealed']=a['sealed']+np.float32(1)
    elif fault=='arithmetic':tr['arithmetic'][1]['kind']='mul'
    elif fault=='tokens':a['tokens']=a['tokens']+np.float32(1)
    elif fault=='replaycount':tr['replays'].pop()
    elif fault=='kind':tr['replays'][0]['kind']='old_mean'
    elif fault=='returned':tr['replays'][1]['returned']=False
    elif fault=='prediction':a['segmented_p']=a['segmented_p']+np.float32(1)
    elif fault=='history_z':a['whole_z0']=a['whole_z0']+np.float32(1)
    elif fault=='history_a':a['segmented_a1']=a['segmented_a1']+np.float32(1)
    else:tr['replays'][0]['rollout']['final']['z']='z'
    with pytest.raises(ValueError):build(f)


def test_cold_import_has_no_runtime_or_neural_import():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert not any("six_target_physical" in k for k in sys.modules)'
    subprocess.run([sys.executable,'-B','-c',code],check=True)
