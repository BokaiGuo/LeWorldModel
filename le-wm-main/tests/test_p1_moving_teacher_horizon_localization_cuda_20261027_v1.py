"""RAM arrays only, no temp directories, model weights, GPU, neural CPU, or physics."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
import base64
import gzip
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_teacher_horizon_localization_cuda_20261027_v1 as m
from test_p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2 import RAM,fixture as suffix_fixture,native_fixture


def fixture():
    p,arrays=suffix_fixture();p.update(schema=m.NAME+'.prepare',contract=m.contract(),root_inputs=[],arguments={})
    warm=m.core.moving.fixed_raw();prefix=np.full((5,2),np.float32(.05),np.float32)
    for r in p['inputs']:
        task=dict(root_context=dict(latent_history=np.zeros((3,192),np.float32).tolist(),
            action_history=m.core.normalized(warm,r['mean'],r['std']).tolist()))
        p['root_inputs'].append(m.root_request(r,task,warm,prefix))
    return p,arrays


def fake_forward(l,s,fail_at=None,extra=False):
    def forward(r):
        n=l.current;tokens=m.core.validate_request(r)
        ae=l.begin('action_encoder');ae['inputs']=[s.array(f'm{n}_a_in.npy',tokens[None])]
        ae['output']=s.array(f'm{n}_a_out.npy',np.zeros((1,3,192),np.float32));l.end(ae)
        api=l.begin('predict_API')
        if n==fail_at:raise RuntimeError('artificial forward failure')
        z=np.asarray(r['latents'],np.float32)[None]
        # Artificial array fixture, not a model approximation or research prediction.
        out=z+np.float32(.25)
        pred=l.begin('predictor');pred['inputs']=[s.array(f'm{n}_p_in.npy',z),ae['output']]
        pred['output']=s.array(f'm{n}_p_out.npy',out);l.end(pred)
        proj=l.begin('pred_proj');proj['inputs']=[s.array(f'm{n}_q_in.npy',out.reshape(3,192))]
        proj['output']=s.array(f'm{n}_q_out.npy',out.reshape(3,192));l.end(proj);l.end(api)
        if extra:l.begin('action_encoder')
        return out[0,-1].copy()
    return forward


def execute(monkeypatch,fail_at=None,label_error=False):
    p,arrays=fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p);loads=[]
    monkeypatch.setattr(m,'load_array',lambda root,e:s.load(e))
    def labels(e):
        assert 'predictions_seal.json' in s.files
        assert l.counts()=={k:dict(attempted=36,completed=36) for k in m.RESOURCES}
        loads.append(e['path'])
        if label_error:raise FileNotFoundError('artificial missing label')
        return arrays[e['path']].copy()
    seal=m.execute(p,s,l,fake_forward(l,s,fail_at),rows,label_loader=labels)
    return p,arrays,s,l,rows,loads,seal


def test_fixed36_three_branches_seal_then_score_and_context_isolation(monkeypatch):
    p,arrays,s,l,rows,loads,seal=execute(monkeypatch)
    assert len(rows)==12 and len(loads)==24 and len(l.entries)==144
    assert all(r['status']=='scored' for r in rows)
    requests=[m.core.saved.parse(s.files[f'input_{n:02d}.json']) for n in range(36)]
    m.check_trace(l.entries,requests,s.load)
    for i in range(12):
        root,op,tf=requests[3*i:3*i+3]
        assert np.array_equal(np.asarray(op['latents'])[-1],np.full(192,.25,np.float32))
        assert np.array_equal(np.asarray(tf['latents'])[-1],np.ones(192,np.float32))
        assert op['history']==tf['history'] and op['raw5']==tf['raw5']
        assert root==p['root_inputs'][i]
        met=rows[i]['metrics']
        assert met['open_minus_teacher_factual_mse']==met['propagation_gap_mse']+met['propagation_cross_term']
    events=[m.core.saved.parse(line) for line in s.files['events.jsonl'].splitlines()]
    seal_event=next(e for e in events if e['kind']=='all_predictions_sealed')
    assert all(e['index']>seal_event['index'] for e in events if e['kind']=='episode_scored')


@pytest.mark.parametrize('n',[0,1,2,17,35])
def test_partial_no_labels_no_seal_no_retry_full12_retained(monkeypatch,n):
    p,_=fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p)
    with pytest.raises(RuntimeError):m.execute(p,s,l,fake_forward(l,s,n),rows,label_loader=lambda e:pytest.fail('future label'))
    assert len(rows)==len(m.core.saved.parse(s.files['initial_rows.json']))==12
    assert 'predictions_seal.json' not in s.files
    assert l.counts()['predict_API']==dict(attempted=n+1,completed=n)
    assert all(r['status']=='not_started' for r in rows[n//3+1:])


def test_label_failure_preserves_whole36_predictions(monkeypatch):
    p,_=fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p)
    with pytest.raises(FileNotFoundError):m.execute(p,s,l,fake_forward(l,s),rows,label_loader=lambda e:(_ for _ in ()).throw(FileNotFoundError('label')))
    assert all(r['status']=='predicted' for r in rows)
    assert len([n for n in s.files if n.startswith('prediction_')])==36 and 'predictions_seal.json' in s.files


def test_extra_call_and_cross_episode_rejected_before_more_work():
    p,_=fixture();s=RAM();l=m.Ledger(s)
    with pytest.raises(ValueError):m.execute(p,s,l,fake_forward(l,s,extra=True),m.initial_rows(p))
    assert l.counts()['predict_API']==dict(attempted=1,completed=1)
    p['evaluators'][0],p['evaluators'][1]=p['evaluators'][1],p['evaluators'][0]
    s=RAM();l=m.Ledger(s)
    with pytest.raises(ValueError):m.execute(p,s,l,lambda r:pytest.fail('wrong identity dispatched'),m.initial_rows(p))


@pytest.mark.parametrize('key',['goal','label','terminal','state','future_token','path'])
def test_shared_narrow_request_contract_rejects_privileged_fields(key):
    p,_=fixture();r=p['root_inputs'][0];r[key]=[1]
    with pytest.raises(ValueError):m.core.validate_request(r)


def test_action_chronology_root_and_suffix_full2_actual_packets():
    p,_=fixture();root=p['root_inputs'][0];suffix=p['inputs'][0]
    actual=m.core.validate_request(root)
    assert np.array_equal(actual[1:],suffix['history'])
    assert actual.shape==(3,10) and np.array_equal(np.asarray(root['raw5']),np.full((5,2),np.float32(.05)))
    bad=deepcopy(suffix);bad['history'][0][0]+=1
    with pytest.raises(ValueError,match='chronology'):m.open_request(root,bad,np.zeros(192,np.float32))
    bad=deepcopy(suffix);bad['identity']['mode']='R'
    with pytest.raises(ValueError,match='episode'):m.open_request(root,bad,np.zeros(192,np.float32))


def test_actual_native_hooks_full36_no_cpu_and_restores(monkeypatch):
    p,arrays=fixture();model,torch,device=native_fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p);old=model.predict
    monkeypatch.setattr(m,'load_array',lambda root,e:s.load(e))
    with m.core.Native(model,torch,device,l,s) as forward:
        m.execute(p,s,l,forward,rows,label_loader=lambda e:arrays[e['path']].copy())
    assert model.predict==old and 'predict' not in model.__dict__ and len(l.entries)==144
    assert all(not x.pre and not x.post for x in (model.action_encoder,model.predictor,model.pred_proj,model.encoder,model.projector))
    with pytest.raises(ValueError):m.core.Native(None,None,NS(type='cpu'),l,s)


def test_forbidden_ad_and_image_guard_without_neural():
    f=lambda:None;torch=NS(autograd=NS(grad=f,functional=NS(jacobian=f)),Tensor=NS(backward=f),func=NS(vjp=f),linalg=NS(svd=f))
    s=RAM();l=m.Ledger(s);errors=[]
    with pytest.raises(ValueError):
        with m.core.no_ad(torch,l,errors):torch.autograd.grad()
    assert torch.autograd.grad is f and not errors
    model,torch,dev=native_fixture()
    with pytest.raises(ValueError):
        with m.core.Native(model,torch,dev,l,s):model.encoder(None)
    assert not model.encoder.pre


def census_fixture():
    from PIL import Image
    stream=BytesIO();image=np.zeros((224,224,3),np.uint8);Image.fromarray(image).save(stream,format='PNG');png=stream.getvalue()
    events=[];rows=[];files={}
    for i in range(12):
        ep=f'e{i//3:02}_{("O","I","R")[i%3]}';cps=[]
        for phase in ('warmup','plan'):
            for step in range(11):
                path=f'/RAM/live/{ep}_{phase}_{step:02}.png';files[path]=png
                c=dict(phase=phase,step=step,path=path,file_sha256=m.core.saved.sha_bytes(png),
                    pixel_sha256=m.core.saved.sha_bytes(image.tobytes()),state=[0.]*7)
                events.append(dict(kind='fresh_observation',episode_id=ep,receipt=c))
                events.append(dict(kind='guard_passed',episode_id=ep,phase=phase,step=step))
                if phase=='plan' and step in (0,5,10):cps.append(c)
                if step:
                    raw=np.array([.125,-.25],np.float32)
                    events.append(dict(kind='warmup_return' if phase=='warmup' else 'control_return',episode_id=ep,
                        step=step,raw=raw.astype(float).tolist(),command=raw.astype(float).tolist(),raw_bits=raw.tobytes().hex()))
        rows.append(dict(episode_id=ep,checkpoints=cps))
    return events,rows,files


@pytest.mark.parametrize('fault',[None,'missing_png','missing_guard','missing_step','bad_raw','bad_pixel','missing_checkpoint'])
def test_durable_source_census_all_raw_frames_not_fake_packets(monkeypatch,fault):
    from PIL import Image
    events,rows,files=census_fixture();oldopen=Image.open
    monkeypatch.setattr(Image,'open',lambda path:oldopen(BytesIO(files[str(path)])))
    monkeypatch.setattr(m.core.saved,'local',lambda base,name:Path(name))
    if fault=='missing_png':files.pop(next(iter(files)))
    if fault=='missing_guard':events.pop(next(i for i,e in enumerate(events) if e['kind']=='guard_passed'))
    if fault=='missing_step':events.pop(next(i for i,e in enumerate(events) if e['kind']=='fresh_observation'))
    if fault=='bad_raw':next(e for e in events if e['kind']=='control_return')['raw_bits']='00000000'
    if fault=='bad_pixel':events[0]['receipt']['pixel_sha256']='f'*64
    if fault=='missing_checkpoint':rows[0]['checkpoints'].pop()
    def bind(path,h):
        assert m.core.saved.sha_bytes(files[str(path)])==h
        return path
    if fault:
        with pytest.raises((ValueError,KeyError,AssertionError)):m.source_census(events,rows,bind,Path('/RAM/live'))
    else:
        c=m.source_census(events,rows,bind,Path('/RAM/live'))
        assert c['PNG']==264 and c['maximum_model_horizon']==2 and c['intermediate_raw_frames_are_not_model_steps']


@pytest.mark.parametrize('fault',[None,'second','trailing','truncated','sha','bomb'])
def test_prepare_transport_caps_semantics_and_single_member(fault):
    p,_=fixture();raw=m.encode_prepare(p);assert len(raw)<=262144
    if fault is None:assert m.decode_prepare(raw)==p;return
    e=m.core.saved.parse(raw);packed=base64.b64decode(e['payload'])
    if fault=='second':packed+=gzip.compress(b'{}',mtime=0)
    if fault=='trailing':packed+=b'x'
    if fault=='truncated':packed=packed[:-1]
    if fault=='sha':e['semantic_sha256']='0'*64
    if fault=='bomb':packed=gzip.compress(b'x'*(m.SEMANTIC_CAP+1),mtime=0)
    e['payload']=base64.b64encode(packed).decode()
    with pytest.raises((ValueError,m.zlib.error)):m.decode_prepare(m.canonical(e))


def test_complete_saved_consume_and_reject_tampered_open_context(monkeypatch):
    p,arrays,s,l,rows,loads,seal=execute(monkeypatch);actual_load=m.load_array
    r=dict(schema=m.NAME+'.report',status='horizon_localization_recorded_not_admitted',prepared='/RAM/prep',prepare_sha256='a'*64,
        errors=[],blocked=[],source_verified=True,model_unchanged=True,model_hash_before=p['expected_model_hash'],model_hash_after=p['expected_model_hash'],
        contract=m.contract(),fixed_denominator=12,prediction_denominator=36,rows=rows,resources=l.counts(),ledger=l.entries,
        predictions_seal_sha256=seal,artifact_sha256={k:m.core.saved.sha_bytes(v) for k,v in s.files.items()})
    s.files['report.json']=m.canonical(r)
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p));monkeypatch.setattr(m,'sha',lambda path:m.core.saved.sha_bytes(s.files[Path(path).name]))
    monkeypatch.setattr(m.core.saved,'read_bytes',lambda path,*a:s.files[Path(path).name])
    monkeypatch.setattr(Path,'read_bytes',lambda path:s.files[Path(path).name])
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([s.output/k for k in s.files]));monkeypatch.setattr(Path,'is_symlink',lambda path:False)
    monkeypatch.setattr(m.core,'receipt_array',lambda e:arrays[e['path']].copy())
    out=m.consume(s.output,m.core.saved.sha_bytes(s.files['report.json']));assert len(out['rows'])==12
    # Resealing an input file is insufficient: semantic causal construction is independently rebuilt.
    inp=m.core.saved.parse(s.files['input_01.json']);inp['latents']=p['inputs'][0]['latents'];inp=m.reseal(inp)
    s.files['input_01.json']=m.canonical(inp)
    r['artifact_sha256']['input_01.json']=m.core.saved.sha_bytes(s.files['input_01.json'])
    ss=m.core.saved.parse(s.files['predictions_seal.json']);ss['forward_inputs'][1]['sha256']=r['artifact_sha256']['input_01.json']
    s.files['predictions_seal.json']=m.canonical(ss);r['predictions_seal_sha256']=m.core.saved.sha_bytes(s.files['predictions_seal.json'])
    r['artifact_sha256']['predictions_seal.json']=r['predictions_seal_sha256'];s.files['report.json']=m.canonical(r)
    with pytest.raises(ValueError):m.consume(s.output,m.core.saved.sha_bytes(s.files['report.json']))


def test_mse_decomposition_not_necessarily_monotone():
    zero=np.zeros(192,np.float32);one=np.ones(192,np.float32)
    result=m.score(zero,zero,one,zero,zero,zero,zero)
    assert result['open_minus_teacher_factual_mse']==-1 and result['propagation_gap_mse']==1
    assert result['propagation_cross_term']==-2
    for bad in (np.zeros(191,np.float32),np.zeros(192,np.float64)):
        with pytest.raises(ValueError):m.score(bad,zero,zero,zero,zero,zero,zero)
