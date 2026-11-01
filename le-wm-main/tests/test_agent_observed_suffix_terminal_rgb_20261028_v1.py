"""RAM native metadata/array doubles only; no production neural or physics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import subprocess
import sys
import numpy as np
import pytest
import test_agent_observed_suffix_physical_20261028_v1 as physical
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as h
from experiments.spectral_reachability import agent_observed_suffix_terminal_rgb_20261028_v1 as m


def fixture():
    r0,s,p0,_,_=physical.complete();r,p,events=deepcopy((r0,p0,s.events))
    _,_,compiled,_,_,_,_,_=physical.fixture()
    # The physical fixture contains literal metadata scores, not real predicted
    # arrays. Bind a consistent RAM endpoint and recalculate each OWN digest.
    for i,(plan,actual) in enumerate(zip(p['plans'],r['rows'],strict=True)):
        goal=np.asarray(plan['model_task']['goal'],np.float32)
        chosen=plan['compiler_plan'];chosen['new_terminal_context']['latent_history'][-1]=(goal+np.float32(.5)).tolist()
        chosen['new_suffix_Q']=m.bridge.mse(chosen['new_terminal_context']['latent_history'][-1],goal)
        chosen['selection_digest']=m.producer.compiler.digest({k:v for k,v in chosen.items() if k!='selection_digest'})
        plan['new_suffix_Q']=chosen['new_suffix_Q']
        plan['selection_digest']=m.producer.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        actual['plan']=deepcopy(plan);actual['selection_digest']=plan['selection_digest']
        for e in events:
            if e.get('branch')==i:e['selection_digest']=plan['selection_digest']
    p['planner_authority']['plans_sha256']=m.producer.digest(p['plans'])
    r['execution_binding']['planner_authority']=deepcopy(p['planner_authority'])
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,errors=[],rows=deepcopy(r['rows']),
        fixed_denominator=4,known_roots=2,fresh_roots=0,inherited=0,P1_admission=False,P3_admission=False,
        scientific_admission=False,freshness=False,equal_FLOPs=False,equal_total_compute=False,independent_confirmation=False,
        planner_authority=deepcopy(p['planner_authority']),counts=deepcopy(r['counts']),bindings={})
    return p,strict,compiled,r,events


@pytest.fixture(scope='module')
def inputs():return fixture()


def prepared(inputs,missing=()):
    p,s,c,r,e=deepcopy(inputs)
    for i in missing:
        a=r['rows'][i];a['checkpoints']=[f for f in a['checkpoints'] if f['step']!=10]
        a['trajectory']=[f for f in a['trajectory'] if not(f['phase']=='plan' and f['step']==10)]
        a.update(status='native_stopped',own_returned=9,missing_checkpoints=[10],terminated=True)
    s['rows']=deepcopy(r['rows']);rows=m.fixed_rows(p,s,c,r,e,'/RAM_ONLY/physical')
    return dict(schema=m.NAME+'.prepare',contract=m.contract(),rows=rows,shapes=dict(hidden=[1,257,192]),
        expected_calls={k:4-len(missing) if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def test_real_new_schema_and_compiler_digest_domains(inputs):
    p=prepared(inputs)
    assert len(p['rows'])==4 and [r['condition'] for r in p['rows']]==['off','on','off','on']
    for row in p['rows']:
        assert row['actual_cp5_guard'][0]['accepted'] is True and row['prefix_reference_exact'] is True
        assert row['source']['step']==10 and len(row['original_task']['goal'])==192
        assert row['predicted_Q']==m.bridge.mse(row['predicted192'],row['goal192'])
        assert row['compiler_selection_digest']==m.producer.compiler.digest({k:v for k,v in row['compiler_plan'].items() if k!='selection_digest'})
    sample={'literal':[1.,2.]}
    assert m.producer.digest(sample)!=m.producer.compiler.digest(sample)


@pytest.mark.parametrize('fault',['old_schema','order','drop','goal','raw','task','Q','context','own_digest',
    'wrong_compiler_digest','compiler_goal_digest','compiler_context_digest','guard','cp5','early_cp10',
    'no_stop','native_goal','frame_condition','frame_state','snapshot','prediction','normalizer','missingcp'])
def test_metadata_and_cross_schema_digest_mutations_failclosed(inputs,fault):
    p,s,c,r,e=deepcopy(inputs)
    if fault=='old_schema':s['schema']=m.template.producer.NAME+'.consume'
    elif fault=='order':r['rows'].reverse();s['rows']=deepcopy(r['rows'])
    elif fault=='drop':s['rows'].pop()
    elif fault=='goal':c['rows'][0]['request']['task']['goal'][0]+=.1
    elif fault=='raw':p['plans'][0]['commands'][0][0]+=.1
    elif fault=='task':s['rows'][0]['task_id']='0'*64
    elif fault=='Q':p['plans'][0]['model_task']['projection'][0][0]=2.
    elif fault=='context':c['rows'][0]['request']['observed_context']['latent_history'][0][0]+=.1
    elif fault=='own_digest':p['plans'][0]['selection_digest']='0'*64
    elif fault=='wrong_compiler_digest':
        q=p['plans'][0]['compiler_plan'];q['selection_digest']=m.producer.digest({k:v for k,v in q.items() if k!='selection_digest'})
        p['plans'][0]['selection_digest']=m.producer.digest({k:v for k,v in p['plans'][0].items() if k!='selection_digest'})
        r['rows'][0]['plan']=deepcopy(p['plans'][0]);s['rows']=deepcopy(r['rows'])
    elif fault=='compiler_goal_digest':p['plans'][0]['compiler_plan']['goal_sha256']=m.producer.digest(p['plans'][0]['model_task']['goal'])
    elif fault=='compiler_context_digest':p['plans'][0]['compiler_plan']['observed_context_sha256']='0'*64
    elif fault=='guard':e=[x for x in e if not(x['kind']=='cp5_guard' and x['branch']==0)]
    elif fault=='cp5':r['rows'][0]['checkpoints'][1]['snapshot']['state'][0]+=1;s['rows']=deepcopy(r['rows'])
    elif fault=='early_cp10':r['rows'][0]['own_returned']=9;s['rows']=deepcopy(r['rows'])
    elif fault=='no_stop':r['rows'][0]['checkpoints'].pop();r['rows'][0]['missing_checkpoints']=[10];s['rows']=deepcopy(r['rows'])
    elif fault=='native_goal':r['rows'][0]['goal'][0]+=1;s['rows']=deepcopy(r['rows'])
    elif fault=='frame_condition':r['rows'][0]['trajectory'][-1]['condition']='on';s['rows']=deepcopy(r['rows'])
    elif fault=='frame_state':r['rows'][0]['checkpoints'][-1]['state'][0]+=1;s['rows']=deepcopy(r['rows'])
    elif fault=='snapshot':r['rows'][0]['checkpoints'][-1]['snapshot']['state'][0]+=1;s['rows']=deepcopy(r['rows'])
    elif fault=='prediction':p['plans'][0]['compiler_plan']['new_terminal_context']['latent_history'][-1][0]+=.1
    elif fault=='normalizer':p['plans'][0]['model_task']['action_std'][0]+=.1
    else:r['rows'][0]['missing_checkpoints']=[10];s['rows']=deepcopy(r['rows'])
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,c,r,e,'/RAM_ONLY/physical')


def execute(inputs,monkeypatch,missing=(),tokens=257,forbidden=None):
    p=prepared(inputs,missing);rows=deepcopy(p['rows']);store=h.old.Memory();model=h.Model(tokens);torch=h.fake_torch()
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls']);errors=[]
    guard=m.bridge.native.Ledger(2,lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.bridge.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='AE':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(h.Tensor(pre),'pre_'+str(row['index']))
            return model.projector(model.encoder(h.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.encode_guard(model,torch,guard,errors):m.process(p,rows,store,encode)
    r=dict(schema=m.NAME+'.report',contract=m.contract(),rows=rows,errors=errors,blocked=guard.blocked,
        modules=trace.entries,guards=guard.entries,counts=m.bridge.counts(trace.entries),
        precision_before=m.precision_contract(),precision_after=m.precision_contract())
    return p,r,store


@pytest.fixture(scope='module')
def replay(inputs):
    with pytest.MonkeyPatch.context() as patch:return execute(inputs,patch)


def test_exact_native_arrays_resources_and_new_guard_provenance(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',h.pixels);rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==4 and len(r['modules'])==8 and len(r['guards'])==8
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=4,completed=4)
    assert r['counts']['predict_API']==dict(attempted=0,completed=0)
    assert all(row['actual192'] and row['actual_cp5_guard'] and not row['P1_admission'] for row in rows)


@pytest.mark.parametrize('missing',[(0,),(1,3),(0,1,2,3)])
def test_native_missing_keeps_null_four_denominator(inputs,monkeypatch,missing):
    p,r,s=execute(inputs,monkeypatch,missing);rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==4 and r['counts']['encoder']['completed']==4-len(missing)
    assert all(rows[i]['actual192'] is rows[i]['actual_Q'] is rows[i]['endpoint_MSE'] is None for i in missing)


@pytest.mark.parametrize('fault',['unknown_event','extra_event','missing_event','pixels','preprocess','hidden197','CLS','latent',
    'raw','guard','module','count','blocked','precision','goal','ownschema'])
def test_all_saved_array_event_identity_mutations_refused(replay,monkeypatch,fault):
    p0,r0,s0=replay;p,r=deepcopy((p0,r0));es=deepcopy(s0.events);s=h.old.Memory();s.files=dict(s0.files)
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels);row=r['rows'][0]
    if fault=='unknown_event':es[0]['kind']='unknown'
    elif fault=='extra_event':es.append(dict(kind='unknown'))
    elif fault=='missing_event':es.pop()
    elif fault=='pixels':row['pixels']=s.array('badpix',np.zeros((224,224,3),np.uint8))
    elif fault=='preprocess':row['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='hidden197':row['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='CLS':row['modules'][1]['inputs'][0]=s.array('badCLS',np.zeros((1,192),np.float32))
    elif fault=='latent':row['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='raw':row['full10'][0][0]+=.1
    elif fault=='guard':r['guards'].pop()
    elif fault=='module':r['modules'].pop()
    elif fault=='count':r['counts']['encoder']['completed']-=1
    elif fault=='blocked':r['blocked']=['predict']
    elif fault=='precision':r['precision_after']['matmul_TF32']=True
    elif fault=='goal':row['goal192'][0]+=.1
    else:r['schema']=m.template.NAME+'.report'
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,es,s.load)


@pytest.mark.parametrize('forbidden',['predict','AD','AE'])
def test_encoder_only_guard_refuses_extra_computation(inputs,monkeypatch,forbidden):
    with pytest.raises(ValueError):execute(inputs,monkeypatch,forbidden=forbidden)


def test_hidden197_is_rejected_not_relaxed(inputs,monkeypatch):
    with pytest.raises(ValueError,match='hidden257'):execute(inputs,monkeypatch,tokens=197)


def test_cpu_authority_rejected_before_IO():
    with pytest.raises(ValueError):m.run('/not/read','a'*64,'/not/create','CPU')


def test_frozen_helpers_only_not_future_physical_SHA():
    for module,hv in m.FROZEN:assert m.sha(module.__file__)==hv
    assert 'physical_source_sha' in m.ARGUMENTS
    assert m.CAP==64*1024**2 and m.FINAL==8*1024**2 and m.WALL==600


def test_cold_import_does_not_load_models_or_physics():
    code="""
import builtins
old=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in ('torch','gym','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return old(name,*a,**kw)
builtins.__import__=guard
from experiments.spectral_reachability import agent_observed_suffix_terminal_rgb_20261028_v1 as m
assert m.contract()['new_prediction']==0
m.main(['--help'])
"""
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr


def setup_build(inputs,monkeypatch):
    p,s,c,r,e=deepcopy(inputs);c['policy']=str(m.bridge.rgb.POLICY);c['expected_model_hash']=m.bridge.rgb.MODEL_TENSOR_SHA
    for plan,actual,spec in zip(p['plans'],r['rows'],c['rows'],strict=True):
        for binding in (plan['model_task']['binding'],plan['model_task_binding'],plan['compiler_plan']['model_task_binding'],spec['request']['task']['binding']):
            binding['model_sha256']=c['expected_model_hash']
        q=plan['compiler_plan'];q['selection_digest']=m.producer.compiler.digest({k:v for k,v in q.items() if k!='selection_digest'})
        plan['selection_digest']=m.producer.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        actual['plan']=deepcopy(plan);actual['selection_digest']=plan['selection_digest']
    p['planner_authority']['plans_sha256']=m.producer.digest(p['plans'])
    s.update(rows=deepcopy(r['rows']),planner_authority=deepcopy(p['planner_authority']))
    args=dict(physical_output='/RAM/physical',physical_report_sha='1'*64,physical_prepared='/RAM/physical_prepare.json',
        physical_prepare_sha='2'*64,physical_receipt='/RAM/physical_receipt.json',physical_receipt_sha='3'*64,
        physical_source_sha=m.FROZEN[0][1])
    s['physical_report_sha256']=args['physical_report_sha']
    r['execution_binding']=dict(prepared=args['physical_prepared'],prepare_sha256=args['physical_prepare_sha'])
    body=b''.join(m.raw(event).rstrip(b'\n')+b'\n' for event in e);r['journal_sha256']=hashlib.sha256(body).hexdigest()
    a=p['planner_authority'];compile_report=a['compile_output']+'/report.json'
    cr=dict(prepared='/RAM/compile_prepare.json',prepare_sha256='4'*64)
    hashes={args['physical_output']+'/report.json':'1'*64,args['physical_prepared']:'2'*64,args['physical_receipt']:'3'*64,
        compile_report:a['compile_report_sha'],a['compile_receipt']:a['compile_receipt_sha'],cr['prepared']:'4'*64,
        str(Path(m.producer.compiler.__file__).resolve()):a['compile_source_sha']}
    hashes.update({str(Path(mod.__file__).resolve()):h for mod,h in m.FROZEN})
    p['bindings']=dict(hashes);s['bindings']=dict(hashes)
    records={args['physical_output']+'/report.json':r,args['physical_prepared']:p,args['physical_receipt']:s,
        compile_report:cr,cr['prepared']:c}
    calls=[]
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),hashlib.sha256(str(path).encode()).hexdigest()))
    monkeypatch.setattr(m,'read',lambda path,*a:deepcopy(records[str(path)]))
    monkeypatch.setattr(m.producer,'consume',lambda output,h:calls.append((str(output),h)) or deepcopy(s))
    monkeypatch.setattr(m.bridge.rgb,'flat',lambda output,name:NS(read_bytes=lambda:body))
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:None)
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    monkeypatch.setattr(m.producer.compiler.F.source.kernel,'validate_normalizer_source',lambda root:None)
    monkeypatch.setattr(m.bridge.rgb,'read',lambda *a:dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),
        projector=dict(input_dim=192,output_dim=192)))
    return args,records,hashes,calls


def test_build_real_consume_receipt_source_and_verify(inputs,monkeypatch):
    args,records,hashes,calls=setup_build(inputs,monkeypatch);p=m.build(**args)
    assert calls==[('/RAM/physical','1'*64)] and p['expected_calls']['encoder']==4
    assert p['physical_authority']['producer']==m.producer.compiler.NAME
    assert p['arguments']['physical_source_sha']==m.FROZEN[0][1]
    assert p['inherited_recompile_resources']['completed']['predict_API']==72
    records['/RAM/own.json']=p;assert m.verify('/RAM/own.json',m.sha('/RAM/own.json'))==p and len(calls)==2


@pytest.mark.parametrize('fault',['receipt','source','report','prepare','compiled','journal'])
def test_build_rejects_actual_seal_mismatch(inputs,monkeypatch,fault):
    args,records,hashes,calls=setup_build(inputs,monkeypatch)
    if fault=='receipt':records[args['physical_receipt']]['status']='fake'
    elif fault=='source':args['physical_source_sha']='5'*64
    elif fault=='report':args['physical_report_sha']='5'*64
    elif fault=='prepare':hashes[args['physical_prepared']]='5'*64
    elif fault=='compiled':hashes['/RAM/compile_prepare.json']='5'*64
    else:records[args['physical_output']+'/report.json']['journal_sha256']='5'*64
    with pytest.raises(ValueError):m.build(**args)


def saved_fixture(replay,tmp_path,monkeypatch):
    p0,r0,s=replay;p,r=deepcopy((p0,r0));output=tmp_path/'output';output.mkdir()
    p.update(bindings={},expected_model_hash='a'*64,physical_authority={},actual_new_physical_counts={},
        inherited_recompile_resources={},inherited_resources={},arguments=dict(physical_report_sha='1'*64,
            physical_receipt_sha='2'*64,physical_prepare_sha='3'*64,physical_source_sha=m.FROZEN[0][1]))
    path=tmp_path/'prepare.json';path.write_bytes(m.raw(p))
    r.update(prepared=str(path),prepare_sha256=m.sha(path),status='observed_suffix4_encoded_not_admitted',
        model_unchanged=True,source_verified=True,model_hash_before='a'*64,model_hash_after='a'*64,
        P1_admission=False,scientific_admission=False,freshness=False,elapsed_seconds=1.,event_count=len(s.events))
    files=dict(s.files);files.update({f'event_{i:04}.json':m.raw(event) for i,event in enumerate(s.events)})
    files['binding.json']=m.raw(dict(prepared=str(path),prepare_sha256=r['prepare_sha256']))
    files['initial_roster.json']=m.raw(p['rows'])
    files['native_seal.json']=m.raw({k:r[k] for k in ('rows','modules','guards','blocked','errors','precision_before','precision_after')})
    r['native_seal_sha256']=hashlib.sha256(files['native_seal.json']).hexdigest()
    r['artifact_sha256']={name:hashlib.sha256(body).hexdigest() for name,body in files.items()}
    for name,body in files.items():(output/name).write_bytes(body)
    (output/'report.json').write_bytes(m.raw(r))
    monkeypatch.setattr(m,'verify',lambda path,expected:deepcopy(p))
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:None)
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    return output,r


def test_whole_saved_consume_own_schema_no_new_physics(replay,tmp_path,monkeypatch):
    out,r=saved_fixture(replay,tmp_path,monkeypatch);accepted=m.consume(str(out),m.sha(out/'report.json'))
    assert accepted['schema']==m.NAME+'.consume' and accepted['fixed_denominator']==len(accepted['rows'])==4
    assert accepted['physical_source_sha256']==m.FROZEN[0][1]
    assert accepted['consume_new_neural_calls']==accepted['consume_new_physics']==0


@pytest.mark.parametrize('fault',['extra','artifact','report_sha','model','seal','admission','binding','roster'])
def test_saved_consume_refuses_durable_mutations(replay,tmp_path,monkeypatch,fault):
    out,r=saved_fixture(replay,tmp_path,monkeypatch)
    if fault=='extra':(out/'extra').write_bytes(b'extra')
    elif fault=='artifact':(out/'event_0000.json').write_bytes(b'{}')
    elif fault=='model':r['model_hash_after']='0'*64
    elif fault=='seal':r['native_seal_sha256']='0'*64
    elif fault=='admission':r['P1_admission']=True
    elif fault=='binding':(out/'binding.json').write_bytes(b'{}')
    elif fault=='roster':(out/'initial_roster.json').write_bytes(b'[]')
    (out/'report.json').write_bytes(m.raw(r));expected=m.sha(out/'report.json')
    if fault=='report_sha':expected='0'*64
    with pytest.raises(ValueError):m.consume(str(out),expected)


def test_cli_accepts_real_source_sha_no_future_defaults(monkeypatch):
    args=dict(physical_output='/RAM/out',physical_report_sha='1'*64,physical_prepared='/RAM/prepare',physical_prepare_sha='2'*64,
        physical_receipt='/RAM/receipt',physical_receipt_sha='3'*64,physical_source_sha=m.FROZEN[0][1]);calls=[]
    monkeypatch.setattr(m,'build',lambda *v:calls.append(dict(zip(args,v,strict=True))) or dict(RAM=True))
    monkeypatch.setattr(m,'write',lambda path,v:dict(RAM=True))
    command=['prepare','--output','/RAM/own']
    for k,v in args.items():command+=['--'+k.replace('_','-'),v]
    assert m.main(command)==0 and calls==[args]
