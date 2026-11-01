"""Native-chain RAM doubles only, not a mock of the unavailable physical ABI."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as h
import test_agent_observed_raw_coordinate_physical_20261028_v1 as physical
from experiments.spectral_reachability import agent_observed_raw_coordinate_rgb_20261028_v1 as m


def execute(monkeypatch,available=(0,41,82,123),tokens=257,forbidden=None):
    rows=[dict(index=i,source=dict(RAM_ONLY=i) if i in available else None,
        status='not_started' if i in available else 'missing',latent=None,panel=i//41,local_index=i%41,
        probe_id=f'RAM_{i}',baseline=i%41==0,root_slot=i//82,condition='on' if i//41%2 else 'off',
        coord=None,h=None,sign=0,actual_delta=0.,task_id='a'*64,model_task_binding=dict(RAM_ONLY=True),
        selection_digest='b'*64,producer_job_sha256='c'*64,raw_float32_sha256='d'*64,goal192=[0.]*192,
        actual_cp5=[],actual_cp5_guard=[],prefix_reference_exact=None,evaluation=dict(RAM_ONLY=True),
        physical_status='RAM_ONLY_NOT_PHYSICAL_AUTHORITY') for i in range(164)]
    p=dict(schema=m.NAME+'.prepare',contract=m.contract(),rows=rows,shapes=dict(hidden=[1,257,192]),
        expected_calls=m.expected_calls(rows))
    out=deepcopy(rows);s=h.old.Memory();model=h.Model(tokens);torch=h.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(41,lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.bridge.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='AE':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(h.Tensor(pre),'pre_'+str(row['index']))
            return model.projector(model.encoder(h.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.encode_guard(model,torch,guard,errors):m.process(p,out,s,encode)
    r=dict(schema=m.NAME+'.report',contract=m.contract(),rows=out,errors=errors,blocked=guard.blocked,
        modules=trace.entries,guards=guard.entries,counts=m.bridge.counts(trace.entries),
        precision_before=m.precision_contract(),precision_after=m.precision_contract())
    return p,r,s


def test_contract_full164_not_four_selected():
    c=m.contract();assert c['fixed_denominator']==164 and c['panels']*c['jobs_per_panel']==164
    assert c['max_calls']['encoder']==c['max_calls']['projector']==164
    assert c['max_calls']['AD']==c['new_physics']==c['max_calls']['predict']==0
    assert not any(c[k] for k in ('P1_admission','freshness','scientific_admission'))


def test_native_exact_chain_full_denominator(monkeypatch):
    p,r,s=execute(monkeypatch);z=m.validate_native(p,r,s.events,s.load)
    assert len(z)==164 and sum(x is not None for x in z)==4
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=4,completed=4)


def test_all164_single_image_calls_no_dedup(monkeypatch):
    p,r,s=execute(monkeypatch,available=range(164));z=m.validate_native(p,r,s.events,s.load)
    assert len(z)==164 and all(v is not None for v in z)
    assert len(r['modules'])==len(r['guards'])==328 and len(s.events)==1640
    assert all(r['counts'][k]==dict(attempted=164 if k in ('encoder','projector') else 0,
        completed=164 if k in ('encoder','projector') else 0) for k in m.bridge.KINDS)


@pytest.mark.parametrize('fault',['drop','reorder','duplicate','unknown_event','missing_event','resource',
    'precision','hidden','preprocess','cls','latent','metadata','padded','guard'])
def test_native_mutations_rejected(monkeypatch,fault):
    p,r,s=execute(monkeypatch)
    if fault=='drop':r['rows'].pop()
    elif fault=='reorder':p['rows'].reverse()
    elif fault=='duplicate':p['rows'][1]['index']=0
    elif fault=='unknown_event':s.events.append(dict(kind='unknown'))
    elif fault=='missing_event':s.events.pop()
    elif fault=='resource':r['counts']['encoder']['completed']-=1
    elif fault=='precision':r['precision_before']['autocast']=True
    elif fault=='hidden':p['shapes']['hidden']=[1,197,192]
    elif fault in ('preprocess','cls','latent'):
        key={'preprocess':r['rows'][0]['preprocessed'],'cls':r['rows'][0]['modules'][1]['inputs'][0],
            'latent':r['rows'][0]['latent']}[fault]
        original=s.load
        def changed(ref):
            a=original(ref).copy()
            if ref==key:a.flat[0]+=np.float32(.5)
            return a
        s.load=changed
    elif fault=='metadata':r['rows'][0]['source']['RAM_ONLY']=-1
    elif fault=='padded':r['rows'][1]['latent']=deepcopy(r['rows'][0]['latent'])
    else:r['guards'][0]['image']=3
    with pytest.raises((ValueError,KeyError)):m.validate_native(p,r,s.events,s.load)


def test_no_frame_is_encoded_for_all_missing(monkeypatch):
    p,r,s=execute(monkeypatch,available=())
    assert m.validate_native(p,r,s.events,s.load)==[None]*164 and not s.events


def test_reject_197_native_hidden(monkeypatch):
    with pytest.raises(ValueError,match='hidden257'):execute(monkeypatch,tokens=197)


@pytest.mark.parametrize('forbidden',['predict','AD','AE'])
def test_forbidden_work_intercepted_before_callback(monkeypatch,forbidden):
    with pytest.raises((ValueError,RuntimeError)):execute(monkeypatch,forbidden=forbidden)


def test_budget_counted_bytes_floor_and_final_reserve():
    b=m.Budget(clock=lambda:0,free=lambda:2*m.CAP);b.used=m.CAP-m.FINAL
    with pytest.raises(ValueError):b.check(1)
    b.check(1,final=True);b.used=m.CAP
    with pytest.raises(ValueError):b.check(1,final=True)
    with pytest.raises(ValueError):b.check(-1)
    b.used=0;b.free=lambda:m.FLOOR
    with pytest.raises(ValueError):b.check(1)


def test_store_exclusive_native_codec_actual_bytes(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:type('Disk',(),dict(free=2*m.CAP))())
    b=m.Budget();output=tmp_path/'results'/(m.NAME+'_RAM');s=m.Store(output,b)
    a=np.zeros(192,np.float32);ref=s.array('native',a);s.event(dict(kind='RAM_ONLY'))
    assert b.used==sum(f.stat().st_size for f in output.iterdir())
    assert m.bridge.rgb.load_array(ref,lambda n:(output/n).read_bytes()).tobytes()==a.tobytes()
    with pytest.raises(FileExistsError):s.put('event_0000.json',{})
    with pytest.raises(ValueError):m.Store(output,m.Budget())


def test_real_config_abi_257_and_no_defaults():
    config=dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),projector=dict(input_dim=192,output_dim=192))
    assert m.bridge.rgb.config_shapes(config)['hidden']==[1,257,192]
    config['encoder']['patch_size']=16
    with pytest.raises(ValueError):m.bridge.rgb.config_shapes(config)
    del config['encoder']['patch_size']
    with pytest.raises(KeyError):m.bridge.rgb.config_shapes(config)


def test_cold_import_no_neural_or_physics_imports():
    code="""
import builtins
original=builtins.__import__
def guarded(name,*a,**k):
    assert name.split('.')[0] not in {'torch','gym','gymnasium','pymunk','stable_worldmodel','transformers'}, name
    return original(name,*a,**k)
builtins.__import__=guarded
from experiments.spectral_reachability import agent_observed_raw_coordinate_rgb_20261028_v1
"""
    subprocess.run([sys.executable,'-B','-c',code],check=True,cwd=m.ROOT)


@pytest.fixture(scope='module')
def physical_join():
    # Metadata join fixture, not a claim of164 real or simulated physical runs.
    p,_,fd,*_=physical.ram_prepare()
    old_report,old_store,*_=physical.old.complete();rows=[];events=[]
    for i,plan in enumerate(p['plans']):
        slot=i//41;row=deepcopy(old_report['rows'][slot]);row['index']=i;row['plan']=deepcopy(plan)
        for k in ('root_slot','condition','arm','task_id','selection_digest'):row[k]=plan[k]
        rows.append(row)
        # Operation events identify branch INSIDE receipt, observations/guard
        # use top-level branch. Do not drop prefix returns in this join fixture.
        def needed(e):
            if e.get('branch',e.get('receipt',{}).get('branch'))!=slot:return False
            if e['kind']=='cp5_guard':return True
            if e['kind']=='fresh_observation':return e['receipt']['phase']=='warmup' or e['receipt']['step']<=5
            op=e.get('receipt',{})
            return e['kind'] in ('operation_returned','operation_dispatch_intent') and op.get('resource')=='public_step' and op.get('phase')=='plan'
        scoped=deepcopy([e for e in old_store.events if needed(e)])
        for e in scoped:
            if 'branch' in e:
                e['branch']=i
                for k in ('root_slot','condition','arm','selection_digest'):e[k]=plan[k]
            if 'receipt' in e and 'branch' in e['receipt']:
                e['receipt']['branch']=i
                for k in ('root_slot','condition','arm','selection_digest'):e['receipt'][k]=plan[k]
            if e['kind']=='cp5_guard':e['prefix_reference_sha256']=m.producer.digest(p['prefix_reference'][i])
        events.extend(scoped)
    r=dict(schema=m.producer.NAME+'.report',status=m.producer.RECORDED,rows=rows,errors=[],journal_sha256='b'*64,
        execution_binding=dict(prepared='/RAM/physical_prepare.json',prepare_sha256='a'*64))
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,rows=deepcopy(rows),errors=[],
        fixed_denominator=164,known_roots=2,fresh_roots=0,inherited=0,planner_authority=deepcopy(p['planner_authority']),
        counts=dict(RAM_ONLY=True),bindings={},physical_report_sha256='b'*64,
        **{k:False for k in ('P1_admission','P3_admission','scientific_admission','freshness',
            'deployment_allowed','equal_total_compute','equal_FLOPs','independent_confirmation')})
    return p,strict,r,events,fd


def test_real_physical_schema_join_original_baseline_and_digest(physical_join):
    p,s,r,events,_=physical_join;rows=m.fixed_rows(p,s,r,events,'/RAM/physical')
    assert len(rows)==164 and [x['index'] for x in rows if x['baseline']]==[0,41,82,123]
    for row,plan,job in zip(rows,p['plans'],p['jobs'],strict=True):
        assert row['source']['step']==10 and row['actual_cp5_guard'][0]['accepted']
        assert row['raw_float32_sha256']==hashlib.sha256(np.asarray(row['full10'],np.float32).tobytes()).hexdigest()
        assert row['producer_job_sha256']==m.producer.producer_digest({k:v for k,v in job.items() if k!='job_sha256'})
        assert row['selection_digest']==m.producer.digest({k:v for k,v in plan.items() if k!='selection_digest'})


@pytest.mark.parametrize('fault',['old_schema','partial','drop','order','task','raw','codec','cp5','condition','state','fake_missing'])
def test_physical_join_mutations(physical_join,fault):
    p,s,r,es,_=deepcopy(physical_join)
    if fault=='old_schema':s['schema']=m.frozen.producer.NAME+'.consume'
    elif fault=='partial':s['status']='partial_or_failed'
    elif fault=='drop':s['rows'].pop()
    elif fault=='order':s['rows'].reverse()
    elif fault=='task':p['plans'][0]['model_task']['goal'][0]+=.5
    elif fault=='raw':p['plans'][0]['commands'][9][0]+=.1
    elif fault=='codec':p['plans'][0]['selection_digest']=m.producer.producer_digest({k:v for k,v in p['plans'][0].items() if k!='selection_digest'})
    elif fault=='cp5':es=[e for e in es if not(e.get('branch')==0 and e['kind']=='cp5_guard')]
    elif fault=='condition':r['rows'][0]['trajectory'][-1]['condition']='on';s['rows']=deepcopy(r['rows'])
    elif fault=='state':r['rows'][0]['checkpoints'][-1]['state'][0]+=.5;s['rows']=deepcopy(r['rows'])
    else:
        row=r['rows'][0];row['checkpoints']=[c for c in row['checkpoints'] if c['step']!=10]
        row['missing_checkpoints']=[10];s['rows']=deepcopy(r['rows'])
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,r,es,'/RAM/physical')


def test_run_authorization_before_prepare_or_torch(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no unauthorized I/O'))
    with pytest.raises(ValueError):m.run(None,None,None,None)


def test_cli_wires_actual_seven_inputs(monkeypatch,capsys):
    calls=[];monkeypatch.setattr(m,'build',lambda **kw:calls.append(kw) or {})
    monkeypatch.setattr(m,'write',lambda path,value:dict(path=path))
    args=['prepare','--output','/RAM/new.json']
    for k in m.ARGUMENTS:args.extend(['--'+k.replace('_','-'),k])
    assert m.main(args)==0 and calls==[{k:k for k in m.ARGUMENTS}]


@pytest.mark.parametrize('mismatch',[False,True])
def test_build_reconstructs_entire_physical_receipt_before_join(physical_join,monkeypatch,mismatch):
    p,s,r,events,fd=deepcopy(physical_join);code=str(Path(m.producer.__file__).resolve())
    args=dict(physical_output='/RAM/physical',physical_report_sha='b'*64,physical_prepared='/RAM/physical_prepare.json',
        physical_prepare_sha='a'*64,physical_receipt='/RAM/physical_receipt.json',physical_receipt_sha='c'*64,
        physical_source_sha=m.sha(code))
    p['bindings'][code]=s['bindings'][code]=args['physical_source_sha']
    fd.update(policy=str(m.bridge.rgb.POLICY),expected_model_hash=m.bridge.rgb.MODEL_TENSOR_SHA,
        inherited_compute=dict(RAM_ONLY=True),inherited_resources=dict(RAM_ONLY=True))
    sealed=deepcopy(s)
    if mismatch:sealed['counts']['unknown_resource']=1
    values={args['physical_prepared']:p,args['physical_receipt']:sealed,'/RAM/physical/report.json':r,
        p['arguments']['fd_prepared']:fd}
    hashes={'/RAM/physical/report.json':'b'*64,args['physical_prepared']:'a'*64,
        args['physical_receipt']:'c'*64,p['arguments']['fd_prepared']:p['arguments']['fd_prepare_sha']}
    original_sha=m.sha;calls=[]
    monkeypatch.setattr(m,'sha',lambda path:hashes[str(path)] if str(path) in hashes else original_sha(path))
    monkeypatch.setattr(m.producer,'read',lambda path,*a:deepcopy(values[str(path)]))
    monkeypatch.setattr(m.producer,'consume',lambda out,h:calls.append((str(out),h)) or deepcopy(s))
    if mismatch:
        with pytest.raises(ValueError,match='entire reconstructed164'):m.build(**args)
        assert calls==[('/RAM/physical','b'*64)];return
    class Lines:
        def __enter__(self):self.lines=iter([m.raw(e)+b'\n' for e in events]);return self
        def __exit__(self,*a):return False
        def readline(self,*a):return next(self.lines,b'')
    original_open=Path.open
    monkeypatch.setattr(Path,'open',lambda path,*a,**kw:Lines() if str(path)=='/RAM/physical/journal.jsonl' else original_open(path,*a,**kw))
    hashes['/RAM/physical/journal.jsonl']=r['journal_sha256']
    rows=[dict(index=i,source=None,model_task_binding=dict(model_sha256=m.bridge.rgb.MODEL_TENSOR_SHA)) for i in range(164)]
    monkeypatch.setattr(m,'fixed_rows',lambda *a:deepcopy(rows))
    checked=[];monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:checked.append(deepcopy(b)))
    built=m.build(**args)
    assert calls==[('/RAM/physical','b'*64)] and built['arguments']==args and len(checked)==1
    assert checked[0][args['physical_receipt']]=='c'*64 and checked[0][code]==args['physical_source_sha']
    assert built['actual_new_physical_counts']==s['counts'] and built['expected_calls']['encoder']==0


def test_verify_rebuilds_complete_prepare(monkeypatch):
    p=dict(arguments=dict(RAM_ONLY=True),extra=dict(immutable=1));monkeypatch.setattr(m,'sha',lambda path:'a'*64)
    monkeypatch.setattr(m.producer,'read',lambda *a:deepcopy(p));calls=[]
    monkeypatch.setattr(m,'build',lambda **kw:calls.append(kw) or deepcopy(p))
    assert m.verify('/RAM/prepared.json','a'*64)==p and calls==[p['arguments']]
    monkeypatch.setattr(m,'build',lambda **kw:dict(p,extra=dict(immutable=2)))
    with pytest.raises(ValueError):m.verify('/RAM/prepared.json','a'*64)


@pytest.mark.parametrize('fault',[None,'extra','bytes','model','seal','binding','unclosed_gate','events','negative_time'])
def test_saved_consumer_artifact_closure(tmp_path,monkeypatch,fault):
    p,r,s=execute(monkeypatch)
    out=tmp_path/(m.NAME+'_RAM');out.mkdir();prepared=tmp_path/'RAM_prepared.json'
    p.update(expected_model_hash='a'*64,bindings={},arguments=dict(RAM_ONLY=True),physical_authority=dict(RAM_ONLY=True),
        actual_new_physical_counts=dict(RAM_ONLY=True),inherited_compute=dict(RAM_ONLY=True),inherited_resources=dict(RAM_ONLY=True))
    prepared.write_bytes(m.raw(p));ph=m.sha(prepared)
    r.update(status='raw164_encoded_not_admitted',prepared=str(prepared),prepare_sha256=ph,
        model_unchanged=True,source_verified=True,model_hash_before='a'*64,model_hash_after='a'*64,
        elapsed_seconds=1.,P1_admission=False,scientific_admission=False,freshness=False,event_count=len(s.events))
    for i,e in enumerate(s.events):s.files[f'event_{i:04}.json']=m.raw(e)
    s.files['binding.json']=m.raw(dict(prepared=str(prepared),prepare_sha256=ph))
    s.files['initial_roster.json']=m.raw(p['rows'])
    seal={k:r[k] for k in ('rows','modules','guards','blocked','errors','precision_before','precision_after')}
    s.files['native_seal.json']=m.raw(seal);r['native_seal_sha256']=hashlib.sha256(s.files['native_seal.json']).hexdigest()
    r['artifact_sha256']={n:hashlib.sha256(b).hexdigest() for n,b in s.files.items()}
    if fault=='model':r['model_hash_after']='e'*64
    elif fault=='unclosed_gate':r['P1_admission']=True
    elif fault=='events':r['event_count']+=1
    elif fault=='negative_time':r['elapsed_seconds']=-1
    if fault=='seal':s.files['native_seal.json']+=b' '
    elif fault=='binding':s.files['binding.json']=m.raw(dict(prepared=str(prepared),prepare_sha256='f'*64))
    elif fault=='bytes':s.files[next(n for n in s.files if n.endswith('.npy.gz'))]+=b'bad'
    for n,b in s.files.items():(out/n).write_bytes(b)
    if fault=='extra':(out/'unknown').write_bytes(b'extra')
    (out/'report.json').write_bytes(m.raw(r));expected=m.sha(out/'report.json')
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p))
    if fault is not None:
        with pytest.raises((ValueError,FileNotFoundError)):m.consume(out,expected)
    else:
        accepted=m.consume(out,expected)
        assert accepted['fixed_denominator']==164 and accepted['baseline_indices']==[0,41,82,123]
        assert len(accepted['rows'])==164 and sum(not x['latent_missing'] for x in accepted['rows'])==4
        assert accepted['consume_new_neural_calls']==accepted['consume_new_physics']==0
