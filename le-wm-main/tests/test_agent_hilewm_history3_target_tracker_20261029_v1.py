"""RAM arithmetic and saved ABI only: no torch/model/weights/physics execution."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_hilewm_history3_target_tracker_20261029_v1 as m


def spec():
    return m.Spec(horizon=3,samples=4,iterations=2,topk=2,var_scale=.25,seed=1729,
        segments=(1,1,1),initialization='full_horizon_zero_no_witness',
        action_domain='unclipped_normalized_gaussian',objective='identity192_terminal_meanMSE_float64',
        predict_api_cap=12,predict_sample_cap=30,array_bytes_cap=8*1024**2,wall_seconds=30.)


def inputs():
    return (np.arange(576,dtype=np.float32).reshape(1,3,192)/np.float32(1024),
            np.arange(20,dtype=np.float32).reshape(1,2,10)/np.float32(1024),
            np.full((1,192),.25,np.float32))


def kernels(fail=None,mutate=False):
    def check(x):
        if type(x) is not np.ndarray or x.dtype!=np.float32 or not np.isfinite(x).all():
            raise ValueError('finite RAM float32')
    def action(a):
        if fail=='action_encoder': raise RuntimeError('synthetic AE failure')
        return np.repeat(a.sum(axis=-1,keepdims=True,dtype=np.float32),192,axis=-1)
    def predictor(z,a):
        if fail=='low_predictor': raise RuntimeError('synthetic predictor failure')
        if mutate: z[:]=0
        return z+a/np.float32(16)
    def project(x):
        if fail=='low_pred_proj': raise RuntimeError('synthetic projection failure')
        return x.copy()
    return m.low.Kernels(lambda xs,dim:np.concatenate(xs,axis=dim),
        lambda xs,dim:np.stack(xs,axis=dim),check,action,predictor,project,'ram_fixture')


def run(s=None,k=None): return m.track_fixture(*inputs(),spec=s or spec(),kernels=k or kernels())


def test_complete_full_mean_two_paid_replays_and_saved_check():
    r,a=run(); assert r['completed'],r['errors']
    c=m.validate_saved(r,a)
    assert c['all_mean_replays_verified'] and not c['authority_certified']
    assert r['predict_API_debited']==12 and r['predictor_samples_debited']==30
    assert len(r['events'])==36 and len(r['searches'])==2 and len(r['replays'])==2
    assert [e['phase'] for e in r['replays']]==['mean_whole','mean_segmented']
    assert all(len(e['rollout']['checkpoints'])==3 for e in r['searches']+r['replays'])
    assert a[r['selected']['mean']].shape==(1,3,10)
    assert not r['real_meter_authenticated'] and not r['raw_inverse'] and not r['scientific_admission']
    assert r['resources']['AD_primal']==r['resources']['backward']==0
    elite=a[r['selected']['elite_average_not_mean_cost']]
    paid=a[r['replays'][0]['cost']].reshape(1)
    assert not np.array_equal(elite,paid), 'fixture must expose elite-average != actualmean cost'


def test_actual_modules_debit_and_token_axes():
    r,_=run()
    ae,pred,proj=(r['modules'][k] for k in ('action_encoder','low_predictor','low_pred_proj'))
    assert ae['attempts']==ae['dispatched']==ae['module_returns']==ae['captured_returns']==12
    assert ae['rows_dispatched']==30 and ae['tokens_dispatched']==120
    assert pred['rows_dispatched']==30 and pred['tokens_dispatched']==90
    assert proj['rows_dispatched']==90 and proj['tokens_dispatched']==0


@pytest.mark.parametrize('fault',['debit','module_count','AD','authority'])
def test_resource_metadata_cannot_claim_new_evidence(fault):
    r,a=run()
    if fault=='debit': r['events'][0]['budget_debited']=True
    elif fault=='module_count': r['modules']['action_encoder']['module_returns']-=1
    elif fault=='AD': r['resources']['AD_primal']=12
    else: r['real_meter_authenticated']=True
    with pytest.raises(ValueError):m.validate_saved(r,a)


@pytest.mark.parametrize('parts',[(3,),(1,2),(2,1),(1,1,1)])
def test_whole_segmented_fixed_partitions_exact(parts):
    r,a=run(replace(spec(),segments=parts)); assert r['completed']; m.validate_saved(r,a)


def test_planned_counts_no_axis_sum_and_full_ae_history():
    p=spec().planned()
    assert p['predict_API']==12 and p['predictor_samples']==30
    assert p['action_encoder_tokens']==120 and p['low_predictor_context_tokens']==90
    assert p['mean_paid_replays']==2 and p['FLOPs'] is None and p['AD_primal']==0
    r,a=run()
    for row in r['searches']+r['replays']:
        for i in range(3):
            ev=r['events'][row['first_event']+3*i]
            assert a[ev['inputs'][0]].shape[1]==i+3


@pytest.mark.parametrize('field,bad',[
    ('horizon',True),('horizon',17),('samples',1),('samples',65),('iterations',0),
    ('topk',1),('topk',5),('var_scale',float('nan')),('var_scale',0),('seed',None),
    ('segments',(1,1)),('segments',(True,2)),('initialization','compiled_witness'),
    ('action_domain','clipped_raw'),('objective','learnedQ'),('wall_seconds',901),
    ('predict_api_cap',0),('predict_sample_cap',False)])
def test_explicit_spec_rejects(field,bad):
    with pytest.raises(ValueError): replace(spec(),**{field:bad}).checked()


def test_all_fields_required_no_hidden_production_defaults():
    d=asdict(spec()); assert m.spec_from_dict(d)==spec()
    for key in d:
        bad=dict(d); bad.pop(key)
        with pytest.raises(ValueError): m.spec_from_dict(bad)
    with pytest.raises(ValueError):m.spec_from_dict(dict(d,witness=[]))


@pytest.mark.parametrize('capfield,value,paid',[
    ('predict_api_cap',1,1),('predict_sample_cap',4,1),
    ('predict_api_cap',6,6), # all candidates paid; first mean replay refused
    ('predict_sample_cap',24,6)])
def test_cap_refusal_keeps_partial_and_no_refund(capfield,value,paid):
    r,a=run(replace(spec(),**{capfield:value}))
    assert not r['completed'] and r['errors'] and r['predict_API_debited']==paid
    assert r['events'][-1]['kind']=='low_predictor' and not r['events'][-1]['dispatched']
    assert len(a)>0
    with pytest.raises(ValueError,match='partial'):m.validate_saved(r,a)


@pytest.mark.parametrize('kind',['action_encoder','low_predictor','low_pred_proj'])
def test_failed_module_return_kept(kind):
    r,a=run(k=kernels(fail=kind))
    assert not r['completed'] and r['events'][-1]['kind']==kind
    assert r['events'][-1]['dispatched'] and not r['events'][-1]['returned']
    assert ('synthetic' in r['errors'][0]) and a
    if kind!='action_encoder': assert r['predict_API_debited']==1 and r['predictor_samples_debited']==4


def test_array_cap_and_wall_are_failclosed_partial_not_padding():
    for s in (replace(spec(),array_bytes_cap=1024),replace(spec(),wall_seconds=1e-20)):
        r,a=run(s); assert not r['completed'] and r['errors']
        assert len(r['replays'])==0


@pytest.mark.parametrize('which',['z','a','target'])
def test_invalid_context_no_synthetic_history(which):
    z,a,g=inputs(); vals=dict(z=z,a=a,target=g); vals[which]=vals[which][:,:1]
    with pytest.raises(ValueError):m.track_fixture(vals['z'],vals['a'],vals['target'],spec=spec(),kernels=kernels())


@pytest.mark.parametrize('fault',['array_hash','context','target','future','projection','checkpoint','mean',
    'cost','elites','topk','budget','extra_array','missing_replay','event_order','raw_gate'])
def test_saved_mutations(fault):
    r,arr=run(); r=deepcopy(r); a={k:v.copy() for k,v in arr.items()}
    def change(name):
        a[name].flat[0]+=1
        r['arrays'][name]['sha256']=m.digest(a[name].tobytes())
    if fault=='array_hash': a[next(iter(a))].flat[0]+=1
    elif fault=='context':change(r['searches'][0]['context_a'])
    elif fault=='target':change(r['searches'][0]['target'])
    elif fault=='future':change(r['searches'][0]['future'])
    elif fault=='projection':change(r['events'][2]['inputs'][0])
    elif fault=='checkpoint':change(r['replays'][0]['rollout']['checkpoints'][1]['z'])
    elif fault=='mean':change(r['selected']['mean'])
    elif fault=='cost':change(r['replays'][0]['cost'])
    elif fault=='elites':change(r['cem'][0]['arrays']['mean'])
    elif fault=='topk':change(r['cem'][0]['arrays']['topk_inds'])
    elif fault=='budget':r['predictor_samples_debited']-=1
    elif fault=='extra_array':a['array_unreferenced']=np.zeros(1,np.float32)
    elif fault=='missing_replay':r['replays'].pop()
    elif fault=='event_order':r['events'][0]['kind']='low_predictor'
    else:r['raw_inverse']=True
    with pytest.raises((ValueError,IndexError)):m.validate_saved(r,a)


def test_actual_cem_math_is_source_pinned_no_foreign_execution():
    source=m.source_closure(); assert source[str(m.CEM)]==m.PINS[m.CEM]
    body=m.CEM.read_text()
    assert 'batch_mean = topk_candidates.mean(dim=1)' in body
    assert 'batch_var = topk_candidates.std(dim=1)' in body
    assert 'outputs[\'actions\'] = mean.detach().cpu()' in body
    assert '@torch.inference_mode()' in body


def test_unauthorized_host_does_not_import_torch_or_access_fake_root():
    script='''
import sys
from experiments.spectral_reachability import agent_hilewm_history3_target_tracker_20261029_v1 as m
assert 'torch' not in sys.modules
try: m.track_cuda(object(),None,None,None,spec=None,expected_state={})
except ValueError as e: assert 'authorization' in str(e)
else: raise AssertionError('unauthorized')
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True,cwd=m.ROOT)


def test_fixture_cannot_relabel_itself_as_host():
    with pytest.raises(ValueError):m.track_fixture(*inputs(),spec=spec(),kernels=replace(kernels(),kind='host_cuda'))
    with pytest.raises(ValueError,match='raw inverse'):m.raw_inverse(np.zeros((3,10),np.float32),authority={'certified':True})


def test_cli_source_and_explicit_spec(tmp_path):
    prefix=[sys.executable,'-B','-m','experiments.spectral_reachability.'+m.TAG]
    out=subprocess.run(prefix+['verify-sources'],check=True,capture_output=True,text=True,cwd=m.ROOT)
    assert not json.loads(out.stdout)['production_run']
    path=tmp_path/'spec.json'; path.write_text(json.dumps(asdict(spec())))
    out=subprocess.run(prefix+['validate-spec','--spec',str(path)],check=True,capture_output=True,text=True,cwd=m.ROOT)
    assert json.loads(out.stdout)['planned']['predict_API']==12
