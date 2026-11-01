"""Saved zero8 schema + RAM tensor/guard doubles only. No torch/environment."""
from copy import deepcopy
import os
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_safeguard_factorial_terminal_rgb_20261028_v1 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as tensors


def authority():
    return m.physical.read(m.ROOT/'results/agent_p2_safeguard_factorial_physical_20261028_v1_consume_MAIN01.json')


def prepared(missing=()):
    s=authority()
    for i in missing:
        row=s['rows'][i];row['rgb_encoding_inputs']=[x for x in row['rgb_encoding_inputs'] if x['step']!=10]
        row['missing_checkpoints']=[10];row['own_returned']=9;row['full_horizon']=False
    rows=m.fixed_rows(s,m.ROOT/'results/agent_p2_safeguard_factorial_physical_20261028_v1_MAIN01')
    return dict(schema=m.NAME+'.prepare',rows=rows,contract=m.contract(),
        expected_calls={k:8-len(missing) if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def execute(monkeypatch,missing=(),tokens=257,forbidden=None):
    p=prepared(missing);rows=deepcopy(p['rows']);s=tensors.old.Memory()
    model=tensors.Model(tokens);torch=tensors.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',tensors.pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(2,lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(tensors.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(tensors.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,s,encode)
    assert not errors and not guard.blocked
    r=dict(schema=m.NAME+'.report',rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,
        modules=trace.entries,guards=guard.entries,counts=m.bridge.counts(trace.entries))
    return p,r,s


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_real_saved_schema_exact_all8_no_old_receipt_projection():
    s=authority();assert m.sha(m.ROOT/'results/agent_p2_safeguard_factorial_physical_20261028_v1_consume_MAIN01.json')==m.RECEIPT_SHA
    p=prepared();assert len(p['rows'])==8
    for spec,plan in zip(p['rows'],s['plans'],strict=True):
        assert spec['new_suffix_Q']==plan['new_suffix_Q']
        assert spec['raw_float32_bits']==plan['raw_float32_bits']
        assert spec['source']['condition']==plan['condition']
        assert spec['physical_row']['selection_digest']==plan['selection_digest']
    assert p['expected_calls']['encoder']==p['expected_calls']['projector']==8


def test_full_native_trace_preprocess_CLS_projector_Q(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',tensors.pixels)
    rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==8 and len(r['modules'])==len(r['guards'])==16
    for spec,result,row in zip(p['rows'],rows,r['rows'],strict=True):
        z=s.load(row['latent']);q=m.bridge.mse(z,spec['goal192'])
        assert result['actual_Q']==q and result['Q_error']==spec['new_suffix_Q']-q
        assert result['endpoint_MSE']==m.bridge.mse(z,spec['predicted192'])
        assert result['native_metrics']==spec['physical_row']['final_metrics']
        assert not result['physical_benefit_claim']


@pytest.mark.parametrize('missing',[(0,),(7,),(0,3,7),tuple(range(8))])
def test_missing_keeps_all8_without_padding(monkeypatch,missing):
    p,r,s=execute(monkeypatch,missing);rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==8 and sum(x['latent_missing'] for x in rows)==len(missing)
    for i in missing:assert rows[i]['actual_Q'] is rows[i]['endpoint_MSE'] is rows[i]['Q_error'] is None


@pytest.mark.parametrize('bad',['predict','AD','action_encoder'])
def test_forbidden_API_blocked(monkeypatch,bad):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=bad)


def test_patch16_hidden197_not_accepted(monkeypatch):
    with pytest.raises(ValueError,match='hidden257'):execute(monkeypatch,tokens=197)


@pytest.mark.parametrize('fault',['schema','seal','cost','context','raw','goal','history','condition','model','early','drop','duplicate'])
def test_authority_and_literal_binding_mutations_rejected(fault):
    s=authority();p=s['plans'][0];r=s['rows'][0];w=p['new_suffix_witness']
    if fault=='schema':s['schema']=m.core.intervention.NAME+'.physical_consume'
    elif fault=='seal':p['selection_digest']='0'*64
    elif fault=='cost':p['new_suffix_Q']+=1
    elif fault=='context':p['new_suffix_context']['latent_history'][-1][0]+=.1
    elif fault=='raw':w['raw5'][0][0]=.5
    elif fault=='goal':w['goal'][0]+=1
    elif fault=='history':r['rgb_encoding_inputs'][-1]['raw_action_history_bits']='00'*80
    elif fault=='condition':r['rgb_encoding_inputs'][-1]['condition']='on'
    elif fault=='model':r['rgb_encoding_inputs'][-1]['encoder_identity']={}
    elif fault=='early':r['own_returned']=9
    elif fault=='drop':s['rows'].pop()
    else:r['rgb_encoding_inputs'].append(deepcopy(r['rgb_encoding_inputs'][-1]))
    if fault in ('cost','context','raw','goal'):
        p['selection_digest']=m.physical.saved.digest({k:v for k,v in p.items() if k!='selection_digest'})
        r['selection_digest']=p['selection_digest']
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(s,'/RAM_ONLY')


@pytest.mark.parametrize('fault',['event','drop_attempt','order','shape','preprocessed','pixel','native_output','plan','guard','missing','count','blocked','schema'])
def test_complete_array_and_dual_ledger_validation(replay,monkeypatch,fault):
    p,original,store=replay;r=deepcopy(original);events=deepcopy(store.events);s=tensors.old.Memory();s.files=dict(store.files)
    monkeypatch.setattr(m.bridge,'load_pixels',tensors.pixels);row=r['rows'][0]
    if fault=='event':events.insert(0,dict(kind='UNKNOWN'))
    elif fault=='drop_attempt':events.remove(next(e for e in events if e['kind']=='module_attempt'))
    elif fault=='order':events[2],events[3]=events[3],events[2]
    elif fault=='shape':row['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='preprocessed':row['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixel':row['pixels']=s.array('badpix',np.zeros((224,224,3),np.uint8))
    elif fault=='native_output':row['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='plan':row['new_suffix_Q']+=1
    elif fault=='guard':r['guards'].pop()
    elif fault=='missing':row['source']=None
    elif fault=='count':r['counts']['encoder']['completed']-=1
    elif fault=='schema':r['schema']=m.core.NAME+'.report'
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


def test_AUTH_before_any_model_or_saved_IO(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must not read'))
    with pytest.raises(ValueError):m.run('/NONE','0'*64,'/NONE','wrong')


def test_pinned_authority_rejects_before_io():
    with pytest.raises(ValueError,match='only actual pinned'):m.build('/NONE','0'*64,'/NONE',m.RECEIPT_SHA)


def test_whole_native_strict_reconstruction_not_just_status(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda p:m.PRODUCER_SHA if str(p)==m.producer.__file__ else m.CORE_SHA if str(p)==m.core.__file__ else m.RECEIPT_SHA)
    monkeypatch.setattr(m.producer,'consume',lambda *a:{'rows':['actual']})
    monkeypatch.setattr(m.physical,'read',lambda *a:{'rows':['wrong']})
    with pytest.raises(ValueError,match='entire actual zero8'):m.build('/RAM',m.REPORT_SHA,'/RAM',m.RECEIPT_SHA)


def test_metadata_fsync_file_open_before_directory(monkeypatch,tmp_path):
    path=tmp_path/(m.NAME+'_RAM.json');calls=[];real=m.os.fsync
    def sync(fd):
        assert os.fstat(fd).st_size==len(m.raw({'x':1}));calls.append(fd);real(fd)
    def directory(p):
        assert p==tmp_path
        with pytest.raises(OSError):os.fstat(calls[0])
        calls.append('directory')
    monkeypatch.setattr(m.os,'fsync',sync);monkeypatch.setattr(m.bridge.rgb,'syncdir',directory)
    m.write(path,{'x':1});assert calls[-1]=='directory'
    with pytest.raises(ValueError):m.write(path,{'x':1})


def test_saved_build_schema_seam_reuses_accepted_native_for_fixture_only(monkeypatch):
    # Do not rerun MAIN's expensive accepted native consume here. Production
    # build always invokes it; this test checks real saved schema wiring only.
    monkeypatch.setattr(m.producer,'consume',lambda *a:authority())
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda *a:None)
    p=m.build(m.ROOT/'results/agent_p2_safeguard_factorial_physical_20261028_v1_MAIN01',m.REPORT_SHA,
        m.ROOT/'results/agent_p2_safeguard_factorial_physical_20261028_v1_consume_MAIN01.json',m.RECEIPT_SHA)
    assert p['expected_model_hash']==m.bridge.rgb.MODEL_TENSOR_SHA and p['shapes']['hidden']==[1,257,192]
    assert p['expected_calls']['encoder']==p['expected_calls']['projector']==8 and not p['full_upstream_rebuilt_here']
    assert all(str(f) in p['bindings'] for f in m.own_files())


def test_saved_consume_real_artifact_census_arrays_and_schema(monkeypatch,tmp_path,replay):
    p,r,s=deepcopy(replay);p.update(expected_model_hash='a'*64,bindings={},actual_new_physical_counts={},
        inherited_factorial_model_compute={},guarded_reuse=[])
    store=m.Store(tmp_path/(m.NAME+'_RAM'),m.bridge.Budget())
    for name,data in s.files.items():store.write(name,data)
    store.put('binding.json',dict(prepared='/RAM_ONLY',prepare_sha256='b'*64));store.put('initial_roster.json',p['rows'])
    for event in s.events:store.event(event)
    seal={k:r[k] for k in ('rows','modules','guards','blocked','errors')}
    h=store.put('native_seal.json',seal,final=True)
    r.update(status='zero8_terminal_encoded_science_blocked',prepared='/RAM_ONLY',prepare_sha256='b'*64,
        model_unchanged=True,source_verified=True,model_hash_before='a'*64,model_hash_after='a'*64,
        P1_admission=False,P2_admission=False,physical_release=False,elapsed_seconds=1.,native_seal_sha256=h,
        artifact_sha256=deepcopy(store.hashes),event_count=store.index)
    store.put('report.json',r,final=True);rh=m.sha(store.path/'report.json')
    monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m.bridge,'load_pixels',tensors.pixels)
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda *a:None)
    result=m.consume(store.path,rh)
    assert len(result['rows'])==8 and result['physical_report_sha256']==m.REPORT_SHA
    assert result['counts']['encoder']==dict(attempted=8,completed=8)
    (store.path/'unexpected.json').write_bytes(b'{}')
    with pytest.raises(ValueError,match='census'):m.consume(store.path,rh)
