"""RAM/fake arrays only: no native runtime, CUDA, prepare or filesystem outputs."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import moving_budget_scheduler_rgb_cuda_20261027_v1 as m
from test_p1_moving_extended_baseline_rgb_cuda_20261027_v1 import RawRAMStore
from test_p1_moving_baseline_rgb_cuda_20261027_v1 import fixture as baseline_fixture, ALL


def fixture():
    r,events,p,manifest,tasks,oldselected=baseline_fixture(ALL)
    mapping=dict(zip(manifest['arms'],m.ARMS,strict=True))
    def rename(v):
        if isinstance(v,str):return mapping.get(v,v)
        if isinstance(v,list):return [rename(x) for x in v]
        if isinstance(v,dict):return {k:rename(x) for k,x in v.items()}
        return v
    r,events,p,manifest=map(rename,(r,events,p,manifest))
    manifest['schema']=m.physical.NAME+'.selection'
    entry=dict(authority=m.physical.AUTHORITY,output='/RAM/scheduler',report_sha256='c'*64)
    manifest['authorities']=[entry];selected={}
    for (s,a),old in oldselected.items():
        row=rename(deepcopy(old));row.update(schema=m.physical.AUTHORITY+'.model_plan',
            root_sha256=manifest['roots'][s]['root_sha256'],
            task_contract_sha256=tasks[s]['interface']['gn_task']['task_contract_sha256'],
            physical_execution=False,selection_uses_physics=False)
        row['selection_sha256']=m.physical.source.digest(row)
        row['whole_run_report_sha256']=entry['report_sha256']
        selected[s,mapping[a]]=row
    plans=[m.plan_from_consumed(selected[s,a],entry,manifest['roots']) for s in (0,1) for a in m.ARMS]
    p.update(schema=m.physical.NAME+'.prepare',contract=m.physical.contract(600),plans=plans,source_prepare_id='e'*64)
    r.update(schema=m.physical.NAME+'.report',status='moving_budget_scheduler_physical_recorded',contract=p['contract'],P2_admission=False,P3_admission=False,source_prepare_id=p['source_prepare_id'])
    r['counts'].update({k:dict(attempted=8,completed=8) for k in ('environment_construct','environment_close')})
    for b,(plan,row) in enumerate(zip(plans,r['rows'],strict=True)):
        row.update(branch_index=b,selected_plan=deepcopy(plan),selection_digest=plan['selection_digest'])
        for e in events:
            if e.get('branch')==b:e['selection_digest']=plan['selection_digest']
            if e['kind']=='branch_started' and e['branch']==b:e['plan']=deepcopy(plan)
            if e['kind']=='branch_closed' and (e['receipt']['root_slot'],e['receipt']['arm'])==(row['root_slot'],row['arm']):e['receipt']=deepcopy(row)
    augmented=[]
    for e in events:
        if e['kind']=='branch_started':
            scope={k:e[k] for k in ('branch','root_slot','arm','selection_digest')}
            augmented.extend([dict(kind=k,resource='environment_construct',**scope) for k in ('operation_attempted','operation_completed')])
        if e['kind']=='branch_closed':
            row=e['receipt'];scope=dict(branch=row['branch_index'],root_slot=row['root_slot'],arm=row['arm'],selection_digest=row['selection_digest'])
            augmented.extend([dict(kind=k,resource='environment_close',**scope) for k in ('operation_attempted','operation_completed')])
        augmented.append(e)
    events=augmented
    previous=None
    for i,e in enumerate(events):
        if e['kind']=='run_started':e['prepare']=deepcopy(p)
        if e['kind']=='run_finished':e.update(status=r['status'],counts=deepcopy(r['counts']))
        e.update(index=i,previous_sha256=previous);e.pop('event_sha256',None)
        e['event_sha256']=m.physical.source.digest(e);previous=e['event_sha256']
    r['journal_chain_tail']=previous
    return r,events,p,manifest,tasks,selected


def test_fixed8_saved_only_exact_join_and_null_prediction():
    args=fixture();before=deepcopy(args);images=m.assemble(*args)
    assert args==before and len(images)==16
    assert [(i['root_slot'],i['arm'],i['role']) for i in images]==[(s,a,k) for s in (0,1) for a in m.ARMS for k in ('prefix5','terminal10')]
    assert all(i['predicted'] is None for i in images)
    c=m.contract();assert c['encoder']==c['projector']==16 and c['branches']==8
    assert all(c[k]==0 for k in ('predict','action_encoder','AD','physics','new_independent_samples'))
    assert 'loop only' in c['wall_scope'] and c['events_per_completed_image']==7
    assert c['P1_admission'] is c['P3_admission'] is False


@pytest.mark.parametrize('bad',['schema','missing5','missing10','subset','reorder','boolslot','task','root','goal','raw',
    'command','counts','guard','sourceprepare','tail','close','partial','ownseal','modelseal','modelcontract','wholeSHA','admit'])
def test_fail_closed_saved_cohort_and_provenance(bad):
    r,e,p,mf,t,selected=fixture();a=m.ARMS[0]
    if bad=='schema':r['schema']='other.report'
    elif bad.startswith('missing'):r['rows'][0]['checkpoints']=[c for c in r['rows'][0]['checkpoints'] if c['step']!=int(bad[7:])]
    elif bad=='subset':mf['arms'].pop()
    elif bad=='reorder':mf['arms'].reverse()
    elif bad=='boolslot':r['rows'][0]['root_slot']=False
    elif bad=='task':t[0]['task_sha256']='0'*64
    elif bad=='root':p['plans'][0]['root_sha256']='0'*64
    elif bad=='goal':t[0]['interface']['gn_task']['goal_source_sha256']='0'*64
    elif bad=='raw':next(x for x in e if x['kind']=='warmup_return')['raw_float32_bits']='00'
    elif bad=='command':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['command']=[0,0]
    elif bad=='counts':r['counts']['physics_step']['completed']-=1
    elif bad=='guard':next(x for x in e if x['kind']=='guard_receipt')['accepted']=False
    elif bad=='sourceprepare':r['source_prepare_id']='0'*64
    elif bad=='tail':r['journal_chain_tail']='0'*64
    elif bad=='close':e[:]=[x for x in e if x['kind']!='branch_closed']
    elif bad=='partial':r['rows'][0]['full_horizon']=False
    elif bad=='ownseal':p['plans'][0]['own_plan_sha256']='0'*64
    elif bad=='modelseal':selected[0,a]['selection_sha256']='0'*64
    elif bad=='modelcontract':selected[0,a]['task_contract_sha256']='0'*64
    elif bad=='wholeSHA':selected[0,a]['whole_run_report_sha256']='0'*64
    else:r['P3_admission']=True
    with pytest.raises(ValueError):m.assemble(r,e,p,mf,t,selected)


def test_physical_plan_matches_final_scheduler_consumer_canonical():
    r,e,p,mf,t,selected=fixture()
    rows=list(reversed(list(selected.values())))
    data=dict(references=[dict(root_sha256=x['root_sha256']) for x in mf['roots']])
    with patch.object(m.physical,'ownconsume',return_value=(rows,{})),patch.object(m.physical,'task_bridge'):
        plans,_=m.physical.selection(mf,data)
    assert plans==p['plans']
    chosen=selected[0,m.ARMS[0]]
    assert m.physical.source.digest({k:v for k,v in chosen.items() if k not in ('selection_sha256','whole_run_report_sha256')})==chosen['selection_sha256']


def test_build_full_external_sha_bindings_in_RAM(monkeypatch):
    r,e,p,mf,t,selected=fixture();h='a'*64;provider=m.PHYSICAL_SOURCE_SHA
    ph='/RAM/physical/report.json';r.update(prepared='/RAM/prepare.json',prepare_sha256=h)
    saved={ph:r,'/RAM/physical/execution_binding.json':dict(prepared=r['prepared'],prepare_sha256=h),r['prepared']:p,p['manifest']:mf}
    for s,path in enumerate(m.original.EXPORTS):
        saved[path]=dict(source_slot=s,blind_sha256=t[s]['blind_sha256'],blind_source_sha256=t[s]['blind_sha256'],task_contract=t[s]['task_contract'])
    pins={str(m.physical.__file__):provider,str(m.moving.__file__):m.original.MOVING_SHA,
        str(m.rgb.__file__):m.original.RGB_SHA,str(m.rgb.old.__file__):m.rgb.OLD_SHA,
        **dict(zip(m.original.EXPORTS,m.original.EXPORT_SHAS)),**m.enc.MODELS,'/RAM/physical/journal.jsonl':r['journal_sha256']}
    pins.update({str(Path('/RAM/physical')/x['receipt']['png']):x['receipt']['file_sha256'] for x in e if x['kind']=='fresh_observation'})
    reads=[]
    def read(path,expected=None):
        reads.append(str(path));assert 'sourcepack' not in str(path) and 'certificate' not in str(path)
        return deepcopy(saved[str(path)])
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'sha',lambda f:pins.get(str(f),h))
    monkeypatch.setattr(m.physical,'ownconsume',lambda entry:(deepcopy(list(selected.values())),{}))
    monkeypatch.setattr(m.physical.kernel,'journal',lambda *a:deepcopy(e))
    monkeypatch.setattr(m.moving,'load_task',lambda path,_:deepcopy(t[m.original.EXPORTS.index(path)]['interface']))
    monkeypatch.setattr(m.moving,'source_files',lambda:[])
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    args=dict(physical_report=ph,physical_sha=h,physical_prepare_sha=h,physical_source_sha=provider)
    result=m.build(**args)
    assert len(result['images'])==16 and result['contract']==m.contract()
    assert set(result['arguments'])==set(args) and result['bindings'][str(m.physical.__file__)]==provider
    r['prepare_sha256']='0'*64
    with pytest.raises(ValueError,match='execution binding'):m.build(**args)


def encoding_fixture():
    images=m.assemble(*fixture());a=np.zeros((224,224,3),np.uint8);a[0,0]=[65,105,225]
    from PIL import Image
    b=BytesIO();Image.fromarray(a).save(b,format='PNG');payload=b.getvalue();h=m.enc.digest(payload)
    for image in images:
        image['png_path']='/RAM/original.png';image['checkpoint']['file_sha256']=h
        image['checkpoint']['pixel_sha256']=m.enc.digest(a.tobytes())
    p=dict(images=images,contract=m.contract());rows=[dict(status='not_started') for _ in images]
    store=RawRAMStore();ledger=m.enc.Ledger(4,store.event)
    return p,rows,store,ledger,a,payload,h


def encode_mock(ledger,z):
    def callback(a):
        assert a.dtype==np.uint8 and a.shape==(224,224,3)
        ledger.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        return ledger.dispatch('projector',NS(device=NS(type='cuda'),shape=(1,192)),lambda:z.copy())
    return callback


def test_16_images_lossless_png_native_NPY_112events_no_predictions():
    p,rows,store,ledger,a,payload,h=encoding_fixture()
    with patch.object(m.rgb.old,'png',return_value=a),patch.object(Path,'read_bytes',return_value=payload),\
        patch.object(Path,'stat',return_value=NS(st_size=len(payload))),patch.object(m,'sha',return_value=h):
        m.encode_images(p,encode_mock(ledger,np.ones(192,np.float32)),ledger,store,rows)
    assert ledger.counts()=={k:dict(attempted=16,completed=16) for k in ('encoder','projector')}
    assert len(store.events)==112
    for row in rows:
        assert row['status']=='encoded' and store.files[row['rawpixel']['file']]==payload
        z=np.load(BytesIO(store.files[int(row['latent']['file'])]),allow_pickle=False)
        assert z.shape==(192,) and z.dtype==np.float32 and np.array_equal(z,np.ones(192,np.float32))
        assert row['metrics']['predicted_Q'] is None and row['metrics']['predicted_vs_real_mse'] is None
        assert row['metrics']['actual_Q']==4 and row['metrics']['root_Q']==1


@pytest.mark.parametrize('stage',['encoder','between','projector','latent'])
def test_partial_keeps_all16_rows_and_raw_before_dispatch(stage):
    p,rows,store,ledger,a,payload,h=encoding_fixture()
    def fail():raise RuntimeError('fault '+stage)
    def callback(a):
        assert store.files['image_00.rgb.png']==payload
        ledger.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),fail if stage=='encoder' else lambda:None)
        if stage=='between':fail()
        return ledger.dispatch('projector',NS(device=NS(type='cuda'),shape=(1,192)),fail if stage=='projector' else lambda:np.ones(192,np.float32))
    if stage=='latent':store.latent=lambda *a:fail()
    with patch.object(m.rgb.old,'png',return_value=a),patch.object(Path,'read_bytes',return_value=payload),\
        patch.object(Path,'stat',return_value=NS(st_size=len(payload))),patch.object(m,'sha',return_value=h),pytest.raises(RuntimeError):
        m.encode_images(p,callback,ledger,store,rows)
    assert len(rows)==16 and rows[0]['status']=='attempted' and all(x['status']=='not_started' for x in rows[1:])
    assert ledger.counts()['encoder']==dict(attempted=1,completed=int(stage!='encoder'))
    assert store.files['image_00.rgb.png']==payload


def test_unknown_prediction_and_exact_mean192_Q():
    item=dict(root=[0.]*192,goal=[2.]*192)
    out=m.compare(item,np.ones(192,np.float32))
    assert out['actual_Q']==1 and out['root_Q']==4 and out['actual_Q_change']==-3
    assert out['predicted_Q'] is out['predicted_vs_real_mse'] is None
    for bad in (np.ones(192,np.float64),np.ones(191,np.float32),np.full(192,np.nan,np.float32)):
        with pytest.raises(ValueError):m.compare(item,bad)


def test_floor_caps_backend_and_no_runtime_imports():
    with patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR+m.CAP-1)),pytest.raises(ValueError):m.Store('/RAM/not_created')
    store=object.__new__(m.Store);store.output=Path('/RAM')
    with patch.object(Path,'iterdir',return_value=[]),patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)),pytest.raises(ValueError):store.write('no',b'1')
    assert m.contract()['numerics']==dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False)
    assert not any(k in sys.modules for k in ('torch','gymnasium','pymunk','stable_worldmodel'))


def test_external_physical_source_pin_before_reads():
    with patch.object(m,'read',side_effect=AssertionError('no reads')),pytest.raises(ValueError):
        m.build('/RAM/report','a'*64,'b'*64,'c'*64)


def saved_rgb(monkeypatch):
    p,_,_,ledger,a,payload,h=encoding_fixture();p['arguments']=dict(mock='external pinned inputs')
    class FileRAMStore(RawRAMStore):
        def latent(self,index,z):
            b=BytesIO();np.save(b,z,allow_pickle=False);name=f'image_{index:02d}.native192.npy'
            self.files[name]=b.getvalue()
            return dict(file=name,sha256=m.enc.digest(b.getvalue()),dtype='float32',shape=[192])
    store=FileRAMStore();ledger.sink=store.event
    rows=[dict(**{k:i[k] for k in ('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')},
        status='not_started',provenance=i) for i in p['images']]
    with patch.object(m.rgb.old,'png',return_value=a),patch.object(Path,'read_bytes',return_value=payload),\
        patch.object(Path,'stat',return_value=NS(st_size=len(payload))),patch.object(m,'sha',return_value=h):
        m.encode_images(p,encode_mock(ledger,np.ones(192,np.float32)),ledger,store,rows)
    store.files['events.jsonl']=b''.join(m.canonical(e) for e in store.events)
    binding=dict(prepared='/RAM/prepare.json',prepare_sha256='b'*64)
    store.files['binding.json']=m.canonical(binding)
    report=dict(schema=m.NAME+'.report',status='moving_budget_scheduler_rgb_encoded_not_admitted',errors=[],first_error=None,
        source_verified=True,model_unchanged=True,model_hash_before=m.enc.MODEL_TENSOR_SHA,model_hash_after=m.enc.MODEL_TENSOR_SHA,
        contract=m.contract(),fixed_branch_denominator=8,fixed_image_denominator=16,P1_admission=False,P2_admission=False,P3_admission=False,
        training=False,new_predict=0,new_AD=0,new_physics=0,new_independent_samples=0,resources=ledger.counts(),
        inputs=p['arguments'],rows=rows,artifact_sha256={n:m.enc.digest(b) for n,b in store.files.items()},**binding)
    report_sha='d'*64
    monkeypatch.setattr(m,'verify',lambda *args:deepcopy(p))
    monkeypatch.setattr(m,'read',lambda path,expected=None:deepcopy(report if Path(path).name=='report.json' else binding))
    monkeypatch.setattr(m,'sha',lambda path:report_sha if Path(path).name=='report.json' else m.enc.digest(store.files[Path(path).name]))
    real_stat=Path.stat
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name])
    monkeypatch.setattr(Path,'stat',lambda path,**kw:NS(st_size=len(m.canonical(report)) if path.name=='report.json' else len(store.files.get(path.name,b''))) if str(path).startswith('/RAM/') else real_stat(path,**kw))
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([path/name for name in [*store.files,'report.json']]))
    monkeypatch.setattr(Path,'is_file',lambda path:True)
    monkeypatch.setattr(Path,'is_symlink',lambda path:False)
    monkeypatch.setattr(m.rgb.old,'png',lambda *args:a.copy())
    return report,store,report_sha


def test_comparison_ownconsumer_saved_only_full16_and_null(monkeypatch):
    report,store,h=saved_rgb(monkeypatch);receipt=m.consume('/RAM/output',h)
    assert receipt['status']=='scheduler_rgb_verified_not_admitted' and len(receipt['rows'])==16
    assert receipt['P1_admission'] is receipt['P3_admission'] is False
    assert all(r['metrics']['predicted_Q'] is None for r in receipt['rows'])


@pytest.mark.parametrize('bad',['partial','row','Q','latent','events','artifact','model','prediction'])
def test_comparison_adapter_rejects_tamper_without_reencoding(monkeypatch,bad):
    report,store,h=saved_rgb(monkeypatch)
    if bad=='partial':report['status']='partial_or_failed'
    elif bad=='row':report['rows'][0]['provenance']['selection_digest']='0'*64
    elif bad=='Q':report['rows'][0]['metrics']['actual_Q']+=.01
    elif bad=='latent':store.files['image_00.native192.npy']+=b'extra'
    elif bad=='events':
        events=[m.rgb.old.planner.decode(line) for line in store.files['events.jsonl'].splitlines()]
        events[2]['entry']['completed']=True
        store.files['events.jsonl']=b''.join(m.canonical(e) for e in events)
        report['artifact_sha256']['events.jsonl']=m.enc.digest(store.files['events.jsonl'])
    elif bad=='artifact':report['artifact_sha256'].pop('image_15.rgb.png')
    elif bad=='model':report['model_hash_after']='0'*64
    else:report['rows'][0]['metrics']['predicted_Q']=0.
    with pytest.raises(ValueError):m.consume('/RAM/output',h)


def test_run_startup_error_retains16_and_final_source_error(monkeypatch):
    p,_,_,_,_,_,_=encoding_fixture();p.update(arguments={},policy='/RAM/model',expected_model_hash='a'*64)
    class FailureStore(RawRAMStore):
        def __init__(self,*a):super().__init__();self.output=Path('/RAM/output');self.reports={}
        def put(self,name,value,final=False):self.reports[name]=deepcopy(value)
    store=FailureStore();calls=[]
    def verify(*args):
        calls.append(args)
        if len(calls)>1:raise ValueError('secondary final source')
        return p
    monkeypatch.setattr(m,'verify',verify);monkeypatch.setattr(m,'Store',lambda output:store)
    monkeypatch.setitem(sys.modules,'torch',None)
    monkeypatch.setattr(Path,'iterdir',lambda path:[])
    with pytest.raises(ValueError,match='partial output'):m.run('/RAM/p','b'*64,'/RAM/output')
    report=store.reports['report.json']
    assert report['status']=='partial_or_failed' and len(report['rows'])==16
    assert report['errors'][0]['stage']=='execution' and report['errors'][1]['stage']=='final_source'
    assert all(r['status']=='not_started' for r in report['rows'])
    assert report['resources']=={k:dict(attempted=0,completed=0) for k in ('encoder','projector')}


@pytest.mark.parametrize('bad',['extra','missing','directory','symlink'])
def test_consume_exact35_files_no_other_payload(monkeypatch,bad):
    report,store,h=saved_rgb(monkeypatch)
    if bad=='extra':store.files['unlisted.bin']=b'x'
    elif bad=='missing':store.files.pop('image_00.rgb.png')
    elif bad=='directory':monkeypatch.setattr(Path,'is_file',lambda path:path.name!='image_00.rgb.png')
    else:monkeypatch.setattr(Path,'is_symlink',lambda path:path.name=='image_00.rgb.png')
    with pytest.raises(ValueError,match='census'):m.consume('/RAM/output',h)


@pytest.mark.parametrize('bad',['trailing','secondarray','float64','shape','nan'])
def test_consume_NPY_no_tail_finite_float32_native192(monkeypatch,bad):
    report,store,h=saved_rgb(monkeypatch);name='image_00.native192.npy';old=store.files[name]
    if bad=='trailing':value=old+b'tail'
    elif bad=='secondarray':value=old+old
    else:
        a=np.ones(192,np.float32)
        if bad=='float64':a=a.astype(np.float64)
        elif bad=='shape':a=a.reshape(1,192)
        else:a[0]=np.nan
        b=BytesIO();np.save(b,a,allow_pickle=False);value=b.getvalue()
    store.files[name]=value;new_sha=m.enc.digest(value)
    report['artifact_sha256'][name]=new_sha;report['rows'][0]['latent']['sha256']=new_sha
    with pytest.raises(ValueError,match='finite float32'):m.consume('/RAM/output',h)


def test_consume_CLI_has_separate_report_SHA(monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr(m,'consume',lambda output,h:calls.append((output,h)) or dict(mock=True))
    monkeypatch.setattr(sys,'argv',[m.NAME,'consume','--output','/RAM/output','--report-sha','a'*64,'--prepare-sha','b'*64])
    m.main();assert calls==[('/RAM/output','a'*64)]


def test_each_terminal_uses_own_prefix_pixel_Q_difference():
    p,rows,store,ledger,_,_,_=encoding_fixture();arrays=[];payloads=[];hashes=[]
    from PIL import Image
    for i,item in enumerate(p['images']):
        a=np.full((224,224,3),i//2+1+2*(i%2),np.uint8);arrays.append(a)
        b=BytesIO();Image.fromarray(a).save(b,format='PNG');raw=b.getvalue();payloads.append(raw)
        h=m.enc.digest(raw);hashes.append(h);item['checkpoint'].update(file_sha256=h,pixel_sha256=m.enc.digest(a.tobytes()))
    def callback(a):return encode_mock(ledger,np.full(192,ledger.image,np.float32))(a)
    with patch.object(m.rgb.old,'png',side_effect=arrays),patch.object(Path,'read_bytes',side_effect=payloads),\
        patch.object(Path,'stat',return_value=NS(st_size=1024)),patch.object(m,'sha',side_effect=hashes):
        m.encode_images(p,callback,ledger,store,rows)
    for i in range(0,16,2):
        prefix,terminal=rows[i]['metrics'],rows[i+1]['metrics']
        assert prefix['rawpixel_vs_prefix5_mse']==0 and terminal['rawpixel_vs_prefix5_mse']==4
        assert terminal['actual_Q_change_from_prefix5']==terminal['actual_Q']-prefix['actual_Q']


@pytest.mark.parametrize('times,completed',[(iter([0,301]),0),(iter([0,0,301]),1)])
def test_loop_wall_checks_before_and_after_each_image(times,completed):
    p,rows,store,ledger,a,payload,h=encoding_fixture()
    with patch.object(m.time,'monotonic',side_effect=lambda:next(times)),patch.object(m.rgb.old,'png',return_value=a),\
        patch.object(Path,'read_bytes',return_value=payload),patch.object(Path,'stat',return_value=NS(st_size=len(payload))),\
        patch.object(m,'sha',return_value=h),pytest.raises(ValueError,match='wall cap'):
        m.encode_images(p,encode_mock(ledger,np.ones(192,np.float32)),ledger,store,rows)
    assert ledger.counts()=={k:dict(attempted=completed,completed=completed) for k in ('encoder','projector')}
    assert len(rows)==16 and sum(r['status']=='encoded' for r in rows)==completed
