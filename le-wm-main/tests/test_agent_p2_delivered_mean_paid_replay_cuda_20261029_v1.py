"""RAM equations and saved-file ABI only, never torch/model/full strict."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_delivered_mean_paid_replay_cuda_20261029_v1 as m


class Ops:
    mul=staticmethod(lambda a,b:np.multiply(a,b,dtype=np.float32))
    add=staticmethod(lambda a,b:np.add(a,b,dtype=np.float32))
    sub=staticmethod(lambda a,b:np.subtract(a,b,dtype=np.float32))
    div=staticmethod(lambda a,b:np.divide(a,b,dtype=np.float32))
    def capture(self,x):return x.copy()
    def layout(self,x):return dict(shape=list(x.shape),stride=[s//x.itemsize for s in x.strides],dtype='numpy.float32',device='ram_fixture',requires_grad=False)


def values():
    mean=np.linspace(-1,1,30,dtype=np.float32).reshape(1,3,10)
    mu=np.asarray([-.0087,.0068],np.float32);std=np.asarray([.2019,.2002],np.float32)
    rr=np.add(np.multiply(mean.reshape(15,2),std,dtype=np.float32),mu,dtype=np.float32)
    return dict(z=np.arange(576,dtype=np.float32).reshape(1,3,192)/np.float32(1024),
        a=np.arange(20,dtype=np.float32).reshape(1,2,10)/np.float32(1024),target=np.full((1,192),.25,np.float32),
        mean=mean,mean_xy=mu,std=std,sealed_raw=rr)


def kernels(fail=None):
    def check(a):m.need(type(a) is np.ndarray and a.dtype==np.float32 and np.isfinite(a).all(),'RAM f32')
    def ae(a):
        if fail=='action_encoder':raise RuntimeError('RAM failed AE')
        return np.repeat(a.sum(axis=-1,keepdims=True,dtype=np.float32),192,axis=-1)
    def predict(z,a):
        if fail=='low_predictor':raise RuntimeError('RAM failed predictor')
        return z+a/np.float32(16)
    def project(x):
        if fail=='low_pred_proj':raise RuntimeError('RAM failed projection')
        return x.copy()
    return m.low.Kernels(lambda xs,dim:np.concatenate(xs,axis=dim),lambda xs,dim:np.stack(xs,axis=dim),check,ae,predict,project,'ram_fixture')


def execute(v=None,k=None):return m.execute(v or values(),k or kernels(),Ops(),'ram_fixture')


def test_protocol_fixed_budget_and_no_deployment():
    p=m.protocol()
    assert p['active']==[0,1,3,4] and p['missing']==[2,5] and p['denominator']==6
    assert (p['H'],p['S'],p['segments'])==(3,1,[1,2])
    assert p['planned']==dict(predict_API=24,predictor_samples=24,action_encoder=24,low_projection=24,AE_tokens=96,predictor_context_tokens=72,projection_rows=72)
    assert all(p[k]==0 for k in ('encoder','high','AD_primal','J','VJP','backward','physics'))
    assert not any(p[k] for k in ('search','reselection','clip','tanh','raw_deployment_authorized','scientific_admission','original_Hi_scaler_certified'))
    assert p['FLOPs'] is None and (p['wall'],p['per_slot_wall'])==(900,180)
    assert (p['output_cap'],p['reserve'],p['floor'])==tuple(x*1024**2 for x in (512,64,200))


def test_actual_authority_small_metadata_only():
    # No build/verify/consume, no model NPY reads: fixed actual metadata ABI only.
    d=m.read(m.AUTH,m.AUTH_SHA);ref=d['authorities']['prepare'];cm=m.read(ref['path'],ref['sha256'])
    v=m.payloads(d,cm)
    assert tuple(v)==m.ACTIVE
    assert [len(d['rows'][i]['checks']['roundtrip_differences']) for i in m.ACTIVE]==[4,8,6,10]
    assert all(x['z'].shape==(1,3,192) and x['a'].shape==(1,2,10) and x['mean'].shape==(1,3,10) for x in v.values())
    bad=deepcopy(d);bad['rows'][0]['mean_source']['normalized_value_sha256']='0'*64
    with pytest.raises(ValueError):m.payloads(bad,cm)
    bad=deepcopy(d);bad['rows'][2]['status']='returned'
    with pytest.raises(ValueError):m.payloads(bad,cm)


def test_complete_replay_full_arrays_no_score():
    r,a=execute();assert r['completed'],r['errors']
    c=m.validate_trace(r,a,values(),host=False)
    assert c['predict_API']==c['predictor_samples']==6 and c['blocked'] is None
    assert len(r['events'])==18 and len(r['arithmetic'])==4 and len(r['replays'])==2
    assert [r['modules'][k]['tokens_dispatched'] for k in ('action_encoder','low_predictor','low_pred_proj')]==[24,18,0]
    assert [r['modules'][k]['rows_dispatched'] for k in ('action_encoder','low_predictor','low_pred_proj')]==[6,6,18]
    assert all(len(x['rollout']['checkpoints'])==3 for x in r['replays'])
    assert 'scores' not in r and not any('cost' in e for e in r['replays'])
    assert any(not x['bitexact'] for x in c['old_to_actual'])
    with pytest.raises(ValueError,match='backend'):m.validate_trace(r,a,values())


@pytest.mark.parametrize('fault',['domain','raw_bytes','signedzero'])
def test_blocked_no_model_no_fallback(fault):
    v=values()
    if fault=='domain':
        v['mean'].fill(10);v['sealed_raw']=np.add(np.multiply(v['mean'].reshape(15,2),v['std'],dtype=np.float32),v['mean_xy'],dtype=np.float32)
    elif fault=='raw_bytes':v['sealed_raw'][0,0]=np.nextafter(v['sealed_raw'][0,0],np.float32(1))
    else:
        v['mean'].fill(0);v['mean_xy'].fill(0);v['sealed_raw'].fill(0);v['sealed_raw'][0,0]=np.float32(-0.)
    r,a=execute(v);c=m.validate_trace(r,a,v,host=False)
    assert c['blocked'] is not None and c['predict_API']==0
    assert r['events']==r['replays']==[] and len(r['arithmetic'])==2


@pytest.mark.parametrize('fail,api',[('action_encoder',0),('low_predictor',1),('low_pred_proj',1)])
def test_failure_partial_attempts_not_refunded(fail,api):
    r,a=execute(k=kernels(fail));assert not r['completed'] and r['errors']
    assert r['predict_API_debited']==api
    e=r['events'][-1];assert e['dispatched'] and not e['module_returned'] and not e['returned']
    assert e['kind']==fail and len(a)>0
    with pytest.raises(ValueError):m.validate_trace(r,a,values(),host=False)


@pytest.mark.parametrize('fault',['debit','AD','count','context','mean','AE','target','projection','whole','layout','arithmetic','raw_guard','extra_event'])
def test_trace_mutations_fail(fault):
    r,a=execute();v=values()
    if fault=='debit':r['events'][1]['budget_debited']=False
    elif fault=='AD':r['resources']['AD_primal']=1
    elif fault=='count':r['modules']['action_encoder']['tokens_dispatched']-=1
    elif fault=='context':r['replays'][1]['rollout']['checkpoints'][1]['a']=r['replays'][1]['rollout']['checkpoints'][0]['a']
    elif fault=='mean':v['mean'][0,0,0]+=1
    elif fault=='AE':r['events'][0]['inputs'][0]=r['inputs']['a']
    elif fault=='target':v['target'][0,0]+=1
    elif fault=='projection':r['events'][2]['inputs'][0]=r['events'][1]['output']
    elif fault=='whole':r['replays'][0]['rollout']['prediction']=r['inputs']['z']
    elif fault=='layout':r['events'][0]['input_layouts'][0]['device']='cpu'
    elif fault=='arithmetic':r['arithmetic'][2]['kind']='fused'
    elif fault=='raw_guard':r['raw_byte_identity']=False
    else:r['events'].append(deepcopy(r['events'][-1]))
    with pytest.raises(ValueError):m.validate_trace(r,a,v,host=False)


def test_signed_zero_ulp_preserved():
    a=np.zeros((15,2),np.float32);b=a.copy();b[0,0]=-0.
    d=m.token_differences(a,b)
    assert not d[0]['bitexact'] and d[0]['ULP']==0 and d[0]['zero_signs']==[False,True]


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    return m.Store(tmp_path/'results'/(m.TAG+'_fixture'))


def test_store_arrays_exclusive_and_exact_eof(store):
    v=values()['mean'];d=store.array('slot0_array_0.npy',v,0)
    assert m.core.exact(m.controller.load_array(store.path,d),v)
    with pytest.raises(ValueError):store.array('slot0_array_0.npy',v,0)
    p=store.path/d['file'];p.write_bytes(p.read_bytes()+b'junk')
    with pytest.raises(ValueError):m.controller.load_array(store.path,d)


def test_store_cap_reserve_failure_footer(store):
    store.used=m.CAP-m.RESERVE
    with pytest.raises(ValueError):store.guard(1)
    store.guard(1,footer=True)
    with pytest.raises(ValueError):store.guard(m.RESERVE+1,final=True)
    store.used=0;store.start-=m.WALL+1
    with pytest.raises(ValueError):store.guard()
    store.guard(1,footer=True)


def test_partial_persist_keeps_debits_and_trace_when_array_write_fails(store,monkeypatch):
    tr,arrays=execute(k=kernels('low_predictor'));row=dict(slot=0,root=0)
    def fail(*args,**kwargs):raise ValueError('fixture output array cap')
    monkeypatch.setattr(store,'array',fail)
    with pytest.raises(ValueError):m.persist_trace(store,row,tr,arrays)
    saved=m.read(store.path/row['trace']['file'],row['trace']['file_sha256'])
    assert saved['predict_API_debited']==1 and not saved['completed']
    assert saved['events'][-1]['dispatched'] and not saved['events'][-1]['returned']
    assert row['arrays']=={}


def fixture_saved(store,monkeypatch):
    """Saved orchestration fixture; explicitly retain RAM backend, never promote."""
    original_validate=m.validate_trace
    monkeypatch.setattr(m,'validate_trace',lambda tr,a,v:original_validate(tr,a,v,host=False))
    data={i:values() for i in m.ACTIVE}
    monkeypatch.setattr(m,'payloads',lambda d,cm:data)
    expected={str(i):dict(shape=[1],dtype='torch.float32',sha256='f'*64) for i in range(425)}
    construction=dict(own_sha256={},tensor_copy_attempts=425,tensor_copy_returns=425,state_key_count=425,forward_calls=0,physics=0)
    manifest=dict(expected_state=expected,expected_construction=construction,sources={},raw_artifact={},controller_prepare={})
    store.put('dispatch_manifest.json',m.encode(manifest))
    rows=[]
    for i in range(6):
        row=dict(slot=i,root=i//3,arm=m.controller.ARMS[i%3],status='missing',debits=dict(known=True,predict_API=0,predictor_samples=0));rows.append(row)
        if i not in m.ACTIVE:continue
        v=data[i];row['status']='predictions_sealed'
        row['inputs']={k:store.array(f'slot{i}_array_{9000+j}.npy',a,i//3) for j,(k,a) in enumerate(v.items())}
        tr,a=execute(v);m.persist_trace(store,row,tr,a)
        row.update(local_check=original_validate(tr,a,v,host=False),model_before=expected,model_after=expected,wall_seconds=1.,
            debits=dict(known=True,predict_API=6,predictor_samples=6))
    seal=m.seal_predictions(store,rows)
    for i in m.ACTIVE:
        row=rows[i];tr=m.read(store.path/row['trace']['file'],row['trace']['file_sha256'])
        row['scores']=[]
        for j,replay in enumerate(tr['replays']):
            p=m.controller.load_array(store.path,row['arrays'][replay['rollout']['prediction']])[:,:,-1].astype(np.float64)
            cost=((p-data[i]['target'][:,None].astype(np.float64))**2).mean(-1)
            row['scores'].append(store.array(f'slot{i}_array_{8000+j}.npy',cost,i//3))
        row['status']='returned'
    report=dict(schema=m.TAG+'.report',protocol=m.protocol(),completed=True,errors=[],rows=rows,files=store.files,
        write_journal=store.journal,model_before=expected,model_after=expected,construction=construction,sources_after={},
        raw_deployment_authorized=False,scientific_admission=False,prediction_seal=seal,
        whole_wall_seconds=2.,cuda_resources=dict(max_allocated=1,max_reserved=2),
        debits=m.controller.totals(rows),root_file_bytes=store.root_used,file_bytes_before_report=store.used)
    store.put('report.json',m.encode(report),final=True)
    # Real JSON roundtrip: dict order MUST NOT be confused with write order.
    return manifest,m.read(store.path/'report.json',m.sha(store.path/'report.json'))


def test_complete_saved_fixture_barrier_and_resource_join(store,monkeypatch):
    manifest,r=fixture_saved(store,monkeypatch)
    m.validate_files(r,store.path);c=m.analyze_saved(manifest,r,store.path)
    assert c['denominator']==dict(total=6,returned=4,blocked=0,missing=2)
    assert c['predict_API']==c['predictor_samples']==24
    assert c['module_resources']==dict(action_encoder=24,low_projection=24,AE_tokens=96,predictor_context_tokens=72,projection_rows=72)
    assert not c['raw_deployment_authorized'] and not c['scientific_admission']


@pytest.mark.parametrize('fault',['score','barrier','model','missing','slots','receipt','journal','bytecap','wall','resources','source','debits'])
def test_full_saved_mutations(store,monkeypatch,fault):
    manifest,r=fixture_saved(store,monkeypatch)
    if fault=='score':r['rows'][0]['scores'][0]=r['rows'][1]['scores'][0]
    elif fault=='barrier':
        order=r['write_journal'];j=next(i for i,e in enumerate(order) if e['name']=='prediction_seal.json');order[j],order[-1]=order[-1],order[j]
    elif fault=='model':r['model_after']={}
    elif fault=='missing':r['rows'][2]['status']='returned'
    elif fault=='slots':r['rows'][0]['root']=1
    elif fault=='receipt':r['construction']['forward_calls']=1
    elif fault=='journal':r['write_journal'].pop()
    elif fault=='bytecap':r['file_bytes_before_report']+=1
    elif fault=='wall':r['whole_wall_seconds']=901
    elif fault=='resources':r['cuda_resources']['max_allocated']=3*1024**3+1
    elif fault=='source':r['sources_after']={'fake':'0'*64}
    else:r['debits']['actual_predict_API_debited']=23
    with pytest.raises(ValueError):
        m.validate_files(r,store.path);m.analyze_saved(manifest,r,store.path)


def test_barrier_refuses_unfinished_four_slots(store):
    rows=[dict(status='predictions_sealed') for _ in range(6)];rows[4]['status']='partial'
    with pytest.raises(ValueError):m.seal_predictions(store,rows)
    assert not (store.path/'prediction_seal.json').exists()


def test_prepare_verify_fixture_exclusive_cli(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    expected=dict(explicit='RAM_prepare_fixture_not_HOST',protocol=m.protocol())
    monkeypatch.setattr(m,'build',lambda:deepcopy(expected))
    p=tmp_path/'results'/(m.TAG+'_prepare_fixture.json');v=m.prepare(p)
    assert m.verify(p,v['sha256'])==expected
    m.main(['verify','--manifest',str(p),'--manifest-sha256',v['sha256']]);assert json.loads(capsys.readouterr().out)['verified']
    with pytest.raises(FileExistsError):m.prepare(p)
    expected['bad']=True
    with pytest.raises(ValueError):m.verify(p,v['sha256'])


def test_unauthorized_run_before_io_and_cold_cli(tmp_path):
    p=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='authorization'):m.run_cuda('missing','0'*64,p)
    assert not p.exists()
    code='from experiments.spectral_reachability import '+m.TAG+' as m; import sys; assert "torch" not in sys.modules; m.main(["protocol"])'
    r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)['planned']['predict_API']==24
