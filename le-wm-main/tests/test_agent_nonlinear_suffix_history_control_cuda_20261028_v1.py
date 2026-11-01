"""RAM tensor journals and metadata only; no torch/neural/physics production."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
import test_agent_nonlinear_suffix_midpoint_rgb_20261028_v1 as previous
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as native
from experiments.spectral_reachability import agent_nonlinear_suffix_history_control_cuda_20261028_v1 as m


@pytest.fixture(scope='module')
def upstream():
    with pytest.MonkeyPatch.context() as patch:
        p,r,s=previous.execute(patch)
        r['status']='actual_midpoint_suffix4_predicted_not_admitted'
        accepted=dict(schema=m.mid.NAME+'.consume',status='actual_midpoint_suffix4_verified_not_admitted',
            fixed_denominator=4,known_roots=2,fresh_roots=0,P1_admission=False,freshness=False,scientific_admission=False,
            rows=m.mid.validate(p,r,s.events,s.load))
    _,_,_,pp,c=previous.fixture()
    p.update(expected_model_hash=c['expected_model_hash'],physical_authority=pp['planner_authority'],bindings={},
        prediction_abi=dict(predict_output=[1,3,192]),policy='/RAM/policy',inherited_resources={})
    return p,r,accepted,c,s


def prepared(upstream):
    p,r,a,c,s=upstream
    rows=m.material(p,r,a,c,s.load)
    return dict(schema=m.NAME+'.prepare',contract=m.contract(),rows=rows)


class RAM(native.RAM):
    def call(self,resource,fn,inputs,shape=None):
        if resource=='action_encoder':fn=lambda:np.repeat(inputs['0'].sum(-1,keepdims=True),192,axis=-1).astype(np.float32)
        return super().call(resource,fn,inputs,shape)


class MemoryStore:
    def __init__(self,path):self.path=path;self.path.mkdir()
    def namespace(self,name):p=self.path/name;p.mkdir();return p
    def check(self,*a):return None
    def put(self,name,value,**kw):
        p=self.path/name;p.write_bytes(m.raw(value));return m.sha(p)


@pytest.fixture(scope='module')
def complete(upstream,tmp_path_factory):
    p=prepared(upstream);store=MemoryStore(tmp_path_factory.mktemp('hybrid')/'output');rows=m.roster()
    m.dispatch(p,store,rows,lambda:None,lambda task,journal,ledger:RAM(task,journal,ledger))
    journals=[m.suffix.load_journal(store.path,row) for row in rows]
    r=dict(schema=m.NAME+'.report',contract=m.contract(),rows=rows,errors=[],blocked=[],
        precision_before=m.mid.terminal.precision_contract(),precision_after=m.mid.terminal.precision_contract())
    return p,r,journals,store


def test_exact_explicit_hybrid_no_last_only_relabel(upstream):
    p=prepared(upstream)
    for row in p['rows']:
        old=np.asarray(row['original_root_context']['latent_history'],np.float32)
        context=np.asarray(row['hybrid_context']['latent_history'],np.float32)
        observed=np.asarray(row['observed_full3'],np.float32)
        np.testing.assert_array_equal(context[:2],old[-2:])
        np.testing.assert_array_equal(context[-1],observed[-1])
        assert row['hybrid_context']['action_history']==upstream[4].load(upstream[1]['rows'][row['index']]['feedback']['inputs']['incoming'])[0].tolist()
        assert row['suffix5']==row['full10'][5:] and row['task']['goal']==upstream[0]['rows'][row['index']]['goal192']
    assert m.contract()['new_encoder']==m.contract()['new_AD']==m.contract()['new_physics']==0
    assert 'NOT full RGB history' in m.contract()['intervention']


@pytest.mark.parametrize('fault',['schema','partial','rowcount','task','goal','raw','context','normalizer','actualQ',
    'missing_history','oldtoken','actualfull','incoming','final','admission'])
def test_material_join_refuses_untied_inputs(upstream,fault):
    p,r,a,c,s=upstream;p,r,a,c=deepcopy((p,r,a,c));load=s.load
    if fault=='schema':r['schema']='old.report'
    elif fault=='partial':r['status']='partial_or_failed'
    elif fault=='rowcount':a['rows'].pop()
    elif fault=='task':a['rows'][0]['task_sha256']='0'*64
    elif fault=='goal':c['slots'][0]['task']['goal'][0]+=.1
    elif fault=='raw':r['rows'][0]['raw_float32_sha256']='0'*64
    elif fault=='context':p['rows'][0]['original_root_context']['latent_history'][0][0]+=.1
    elif fault=='normalizer':p['rows'][0]['action_std'][0]+=.1
    elif fault=='actualQ':a['rows'][0]['actual_final_Q']+=.1
    elif fault=='missing_history':p['rows'][0]['eligible']=False
    elif fault=='oldtoken':c['slots'][0]['task']['root_context']['action_history'][1][0]+=.1
    elif fault=='actualfull':r['rows'][0]['actual_full3']=r['rows'][0]['prediction']
    elif fault=='incoming':r['rows'][0]['feedback']['inputs']['incoming']=r['rows'][0]['feedback']['inputs']['outgoing']
    elif fault=='final':a['rows'][0]['conditioned_final192'][0]+=.1
    else:a['P1_admission']=True
    with pytest.raises((ValueError,KeyError)):m.material(p,r,a,c,load)


def test_actual_dispatch_native_census_full4(complete):
    p,r,journals,store=complete;rows,counts=m.validate(p,r,journals)
    assert len(rows)==4 and counts==m.suffix.caps(4,0)
    assert counts['predict_API']==4 and counts['predict_context_tokens']==12 and counts['pred_proj_rows']==12
    assert counts['encoder_calls']==counts['VJP_directions']==0
    assert all(row['diagnosis'].startswith('explicit hybrid') and not row['physical_benefit_claim'] for row in rows)
    assert all(row['resource_receipt']['FLOPs'] is None for row in r['rows'])


@pytest.mark.parametrize('fault',['unknown','candidate_event','extra','drop','reorder','scope','input','raw','shape',
    'nativeoutput','result','count','precision','blocked','request','drop_row','relabel_row'])
def test_complete_saved_native_mutations_refused(complete,fault):
    p0,r0,j0,_=complete;p,r,j=deepcopy((p0,r0,j0));es=j[0]
    if fault=='unknown':es.insert(0,dict(kind='unknown'))
    elif fault=='candidate_event':es.append(dict(kind='library_selection',prefix_index=0,commands=[],Q=0.))
    elif fault=='extra':es.append(deepcopy(es[-1]))
    elif fault=='drop':es.pop(3)
    elif fault=='reorder':es[2],es[3]=es[3],es[2]
    elif fault=='scope':next(e for e in es if e['kind']=='io_input')['scope']['label']='fake'
    elif fault=='input':next(e for e in es if e['kind']=='program_packet')['arrays']['latent']=m.signal.blob(np.zeros((1,3,192),np.float32))
    elif fault=='raw':es[0]['commands'][0][0]+=.125
    elif fault=='shape':
        e=next(e for e in es if e['kind']=='io_return' and 'action_encoder:output' in e['arrays'])
        e['arrays']['action_encoder:output']=m.signal.blob(np.zeros((1,3,10),np.float32))
    elif fault=='nativeoutput':
        e=next(e for e in es if e['kind']=='io_return' and 'predict:output' in e['arrays'])
        e['arrays']['predict:output']=m.signal.blob(np.zeros((1,3,192),np.float32))
    elif fault=='result':r['rows'][0]['result']['latent_history'][-1][0]+=.1
    elif fault=='count':r['rows'][0]['resource_receipt']['completed']['predict_API']+=1
    elif fault=='precision':r['precision_after']['matmul_TF32']=True
    elif fault=='blocked':r['blocked']=['AD']
    elif fault=='request':p['rows'][0]['hybrid_context']['latent_history'][0][0]+=.1
    elif fault=='drop_row':r['rows'].pop()
    else:r['rows'][0]['condition']='on'
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,j)


def test_failure_keeps_four_denominator_and_real_attempts(upstream,tmp_path):
    p=prepared(upstream);rows=m.roster();store=MemoryStore(tmp_path/'partial')
    class Fail(RAM):
        def step(self,c,a,label):
            self.ledger.scope=dict(forward='f000',label=label)
            return self.ledger.call('predict',lambda:(_ for _ in ()).throw(RuntimeError('RAM fail')),shape=[1,3,192])
    with pytest.raises(RuntimeError,match='RAM fail'):m.dispatch(p,store,rows,lambda:None,lambda t,j,l:Fail(t,j,l))
    assert len(rows)==4 and rows[0]['status']=='failed_partial' and all(r['status']=='not_started' for r in rows[1:])
    assert rows[0]['resource_receipt']['attempted']['predict_API']==1
    assert rows[0]['resource_receipt']['completed']['predict_API']==0
    assert (store.path/'row0_result.json').is_file()


def test_endpoint_effect_and_cancellation_are_not_dropped(complete):
    p,r,_,_=complete;spec=deepcopy(p['rows'][0]);row=deepcopy(r['rows'][0])
    spec['observed_prediction3'][-1]=[.5]*192;spec['actual_terminal192']=[.25]*192
    row['result']['latent_history'][-1]=[.125]*192
    out=m.analyze(spec,row)
    assert out['endpoint_MSE_change']==out['endpoint_error_cross_term']+out['terminal_response_MSE']
    assert out['endpoint_error_cross_term']<0 and out['terminal_response_MSE']>0
    assert out['hybrid_goalQ_absolute_error']>=0 and not out['drift_cause_attributed']


def setup_build(upstream,monkeypatch):
    p,r,a,c,s=upstream;p,r,a,c=deepcopy((p,r,a,c))
    args=dict(midpoint_output='/RAM/mid',midpoint_report_sha='1'*64,
        midpoint_receipt='/RAM/mid_receipt.json',midpoint_receipt_sha='2'*64)
    r.update(prepared='/RAM/mid_prepare.json',prepare_sha256='3'*64,artifact_sha256={})
    source=p['physical_authority']['producer_prepared'];source_sha=p['physical_authority']['producer_prepare_sha256']
    p['bindings']={source:source_sha}
    records={'/RAM/mid/report.json':r,'/RAM/mid_receipt.json':a,'/RAM/mid_prepare.json':p,source:c}
    hashes={'/RAM/mid/report.json':'1'*64,'/RAM/mid_receipt.json':'2'*64,'/RAM/mid_prepare.json':'3'*64,source:source_sha}
    hashes.update({str(Path(mod.__file__).resolve()):h for mod,h in m.FROZEN})
    calls=[]
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),hashlib.sha256(str(path).encode()).hexdigest()))
    monkeypatch.setattr(m,'read',lambda path,*a:deepcopy(records[str(path)]))
    monkeypatch.setattr(m.mid,'consume',lambda out,h:calls.append((str(out),h)) or deepcopy(a))
    monkeypatch.setattr(m.mid,'load_saved_array',lambda out,ref:s.load(ref))
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:None)
    return args,records,hashes,calls


def test_prepare_consumes_whole_upstream_and_rebuilds_own(upstream,monkeypatch):
    args,records,hashes,calls=setup_build(upstream,monkeypatch);p=m.build(**args)
    assert calls==[('/RAM/mid','1'*64)] and len(p['rows'])==4
    assert p['contract']['resource_caps']==m.suffix.caps(4,0)
    records['/RAM/own.json']=p;assert m.verify('/RAM/own.json',m.sha('/RAM/own.json'))==p
    assert len(calls)==2


@pytest.mark.parametrize('fault',['external_sha','receipt','prepared','compiled'])
def test_prepare_rejects_wrong_upstream_seal_or_status(upstream,monkeypatch,fault):
    args,records,hashes,calls=setup_build(upstream,monkeypatch)
    if fault=='external_sha':args['midpoint_report_sha']='4'*64
    elif fault=='receipt':records['/RAM/mid_receipt.json']['status']='fake'
    elif fault=='prepared':hashes['/RAM/mid_prepare.json']='4'*64
    else:hashes[records['/RAM/mid_prepare.json']['physical_authority']['producer_prepared']]='4'*64
    with pytest.raises(ValueError):m.build(**args)


def test_frozen_helper_hashes_and_precise_budget():
    for mod,h in m.FROZEN:assert m.sha(mod.__file__)==h
    assert m.CAP==64*1024**2 and m.FINAL==8*1024**2 and m.FLOOR==200*1024**2 and m.WALL==600


def test_authority_before_IO_or_import():
    with pytest.raises(ValueError,match='before IO'):m.run('/not/read','a'*64,'/not/write','CPU')


def test_cold_cli_never_imports_runtime():
    code="""
import builtins,sys
old=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in ('torch','gym','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return old(name,*a,**kw)
builtins.__import__=guard
from experiments.spectral_reachability import agent_nonlinear_suffix_history_control_cuda_20261028_v1 as m
assert m.contract()['new_encoder']==0 and m.contract()['resource_caps']['predict_API']==4
m.main(['--help'])
"""
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr


def saved_fixture(complete,tmp_path,monkeypatch):
    p0,r0,journals,original=complete;p,r=deepcopy((p0,r0));output=tmp_path/'saved';output.mkdir()
    for f in original.path.rglob('*'):
        target=output/f.relative_to(original.path)
        if f.is_dir():target.mkdir()
        else:target.write_bytes(f.read_bytes())
    p.update(expected_model_hash='a'*64,bindings={},inherited_resources={},arguments=dict(midpoint_report_sha='b'*64))
    prepared=tmp_path/'prepare.json';prepared.write_bytes(m.raw(p))
    r.update(prepared=str(prepared),prepare_sha256=m.sha(prepared),source_verified=True,
        model_hash_before='a'*64,model_hash_after='a'*64,elapsed_seconds=1.,**m.FLAGS)
    r['status']='hybrid_suffix4_predicted_not_admitted'
    (output/'binding.json').write_bytes(m.raw(dict(prepared=str(prepared),prepare_sha256=r['prepare_sha256'])))
    (output/'initial_roster.json').write_bytes(m.raw(m.roster()))
    seal={k:r[k] for k in ('rows','errors','blocked','precision_before','precision_after')}
    (output/'native_seal.json').write_bytes(m.raw(seal));r['native_seal_sha256']=m.sha(output/'native_seal.json')
    r['artifact_sha256']={str(f.relative_to(output)):m.sha(f) for f in output.rglob('*') if f.is_file()}
    (output/'report.json').write_bytes(m.raw(r))
    monkeypatch.setattr(m,'verify',lambda path,expected:deepcopy(p))
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:None)
    return p,r,output


def test_saved_consume_whole_hashchain_and_arrays(complete,tmp_path,monkeypatch):
    p,r,output=saved_fixture(complete,tmp_path,monkeypatch)
    accepted=m.consume(str(output),m.sha(output/'report.json'))
    assert accepted['fixed_denominator']==len(accepted['rows'])==4
    assert accepted['counts']==m.suffix.caps(4,0)
    assert accepted['consume_new_neural_calls']==accepted['consume_new_physics']==0
    assert not accepted['physical_release'] and accepted['FLOPs'] is None


@pytest.mark.parametrize('fault',['report_sha','extra_file','extra_dir','binding','roster','row','journal','seal',
    'model','admission','wall','precision','row_path','source'])
def test_full_saved_consumer_refuses_all_durable_mutations(complete,tmp_path,monkeypatch,fault):
    p,r,output=saved_fixture(complete,tmp_path,monkeypatch)
    if fault=='extra_file':(output/'extra').write_bytes(b'extra')
    elif fault=='extra_dir':(output/'extra').mkdir()
    elif fault=='binding':(output/'binding.json').write_bytes(b'{}')
    elif fault=='roster':(output/'initial_roster.json').write_bytes(b'[]')
    elif fault=='row':(output/'row0_result.json').write_bytes(b'{}')
    elif fault=='journal':
        f=output/'row0/events.jsonl';f.write_bytes(f.read_bytes()+b'{}\n')
    elif fault=='seal':r['native_seal_sha256']='0'*64
    elif fault=='model':r['model_hash_after']='0'*64
    elif fault=='admission':r['scientific_admission']=True
    elif fault=='wall':r['elapsed_seconds']=m.WALL+1.
    elif fault=='precision':r['precision_before']['autocast']=True
    elif fault=='row_path':r['rows'][0]['journal_file']='row1/events.jsonl'
    elif fault=='source':
        def fail(b):raise ValueError('source changed')
        monkeypatch.setattr(m.bridge,'verify_bindings',fail)
    (output/'report.json').write_bytes(m.raw(r));expected=m.sha(output/'report.json')
    if fault=='report_sha':expected='0'*64
    with pytest.raises(ValueError):m.consume(str(output),expected)


def test_saved_census_checks_resealed_unknown_event(complete,tmp_path,monkeypatch):
    p,r,output=saved_fixture(complete,tmp_path,monkeypatch)
    events,tail=m.signal.journal(output/'row0/events.jsonl')
    events.append(dict(kind='unknown'));tail='0'*64;records=[]
    for i,payload in enumerate(events):
        rec=dict(index=i,previous=tail,payload=payload);tail=m.digest(rec);records.append(m.transport.raw(dict(rec,sha256=tail))+b'\n')
    f=output/'row0/events.jsonl';f.write_bytes(b''.join(records));row=r['rows'][0]
    row.update(journal_sha256=m.sha(f),journal_tail=tail)
    (output/'row0_result.json').write_bytes(m.raw(row))
    seal={k:r[k] for k in ('rows','errors','blocked','precision_before','precision_after')}
    (output/'native_seal.json').write_bytes(m.raw(seal));r['native_seal_sha256']=m.sha(output/'native_seal.json')
    r['artifact_sha256']={str(f.relative_to(output)):m.sha(f) for f in output.rglob('*') if f.is_file() and f.name!='report.json'}
    (output/'report.json').write_bytes(m.raw(r))
    with pytest.raises(ValueError,match='only actual prediction events'):m.consume(str(output),m.sha(output/'report.json'))


def test_cli_passes_only_actual_source_seals(monkeypatch):
    args=dict(midpoint_output='/RAM/out',midpoint_report_sha='a'*64,midpoint_receipt='/RAM/receipt',midpoint_receipt_sha='b'*64)
    calls=[]
    monkeypatch.setattr(m,'build',lambda *v:calls.append(dict(zip(args,v,strict=True))) or dict(RAM=True))
    monkeypatch.setattr(m,'write',lambda path,v:dict(RAM=True))
    command=['prepare','--output','/RAM/own']
    for k,v in args.items():command+=['--'+k.replace('_','-'),v]
    assert m.main(command)==0 and calls==[args]
