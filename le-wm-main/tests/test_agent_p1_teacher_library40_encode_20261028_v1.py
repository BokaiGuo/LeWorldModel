"""RAM fixtures only: never load a model, construct an env, or certify evidence."""
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace as NS
import gzip
import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_teacher_library40_encode_20261028_v1 as m


def physical_fixture():
    rows=[];jobs=[];plans=[];events=[]
    for i in range(40):
        ti,rem=divmod(i,10);ri,li=divmod(rem,5);slot,cond=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[ti]
        plan=dict(index=i,task=ti,root_slot=slot,condition=cond,role=('Z','S')[ri],library_index=li,
            library_label=m.teacher.LIBRARY[li],anchor_id=f'anchor{ti}{ri}',task_id=f'task{ti}',blind_sha256=f'blind{ti}',commands=np.zeros((10,2),np.float32).tolist())
        plan['plan_digest']=m.digest(plan);job=deepcopy(plan);job['job_digest']=m.digest(job)
        row=deepcopy(job);state=[200.,200.,250.,250.,0.,0.,0.]
        row.update(public=dict(warmup=dict(attempted=10,completed=10),prefix=dict(attempted=5,completed=5),suffix=dict(attempted=5,completed=5)),
            reference_frames_verified=51,reference_raw_verified=15,initial_exact=True,history_exact=True,start_verified=True,teacher_history_verified=True,
            last_return=dict(phase='suffix',step=5,native_success=False),goal=state,goal_pose=[256.,256.,.7],goal_pixel_sha256='goalpixels',
            status='failed' if i==12 else 'horizon_completed',missing_checkpoints=[10] if i==12 else [],
            end_state=None if i==12 else state,last_state=[200.,515.,250.,250.,0.,0.,0.] if i==12 else state,
            rgb_guard_accepted=i!=12,checkpoints=[])
        for overlay in range(1 if i==12 else 3):
            condition='on' if overlay==1 else 'off';snapshot=dict(state=state)
            invariant=dict(snapshot=snapshot,goal_state=row['goal'],goal_pose=row['goal_pose'],goal_rgb_sha256=row['goal_pixel_sha256'])
            q=dict(phase='suffix',step=5,condition=condition,overlay_index=overlay,png=f'b{i}_suffix_05_{overlay}.png',
                   file_sha256=f'file{i}{overlay%2}',pixel_sha256=f'pixels{i}{overlay%2}',snapshot=snapshot,state=state)
            events.append(dict(kind='fresh_observation',job_digest=job['job_digest'],receipt=q,pre=invariant,post=invariant))
            if i!=12 and overlay==(1 if cond=='on' else 0):
                frame=dict(file=q['png'],sha256=q['file_sha256'],pixel_sha256=q['pixel_sha256'],pre=invariant,post=invariant,state=state,step=5)
                row['checkpoints']=[dict(frame=frame,step=10,absolute_step=20)]
        if i!=12:events.append(dict(kind='checkpoint_recorded',job_digest=job['job_digest'],checkpoint=row['checkpoints'][0]))
        rows.append(row);jobs.append(job);plans.append(plan)
    counts={k:dict(attempted=v,completed=v) for k,v in [('public_step',800),('physics_step',8160),('explicit_render',2638)]}
    report=dict(schema='agent_p1_teacher_cross_library_physical_20261028_v1.report',status='partial',fixed_denominator=40,inputs_final_verified=True,rows=rows,counts=counts,errors=['real_missing'])
    receipt=dict(schema='agent_p1_teacher_cross_library_physical_20261028_v1.consume',status='partial_teacher_cross_library40_preserved_not_admitted',
                 report_sha256='physical',prepare_sha256='prepare',rows=deepcopy(rows),counts=deepcopy(counts),errors=deepcopy(report['errors']),fixed_denominator=40,P1_admission=False)
    return report,receipt,dict(jobs=jobs),dict(plans=plans),events


def roster(data):return m.roster(*data,'physical','prepare')


def test_fixed40_roster_and_exact_missing():
    rows=roster(physical_fixture())
    assert len(rows)==40 and sum(r['eligible_for_encoding'] for r in rows)==39
    assert rows[12]['actual_latent'] is rows[12]['actual_Q'] is rows[12]['actual_model_error'] is None
    assert rows[12]['canonical_blue_pixels']==8 and rows[12]['guard_failure']=='canvas_out_of_bounds'
    assert rows[12]['png'] is None and rows[12]['incident_png']=='b12_suffix_05_0.png'


@pytest.mark.parametrize('mutation', ['drop','swap','receipt','budget','seal','raw','order','history','native_return','missing','canvas','state','prepost','goal','selected_condition','checkpoint','off_replay','incident'])
def test_roster_rejects_mutations(mutation):
    d=physical_fixture();r,w,p,c,ev=d
    if mutation=='drop':r['rows'].pop();w['rows']=deepcopy(r['rows'])
    elif mutation=='swap':p['jobs'][0],p['jobs'][1]=p['jobs'][1],p['jobs'][0]
    elif mutation=='receipt':w['report_sha256']='other'
    elif mutation=='budget':r['counts']['public_step']['completed']=799;w['counts']=deepcopy(r['counts'])
    elif mutation=='seal':c['plans'][0]['plan_digest']='bad'
    elif mutation=='raw':r['rows'][0]['commands'][0][0]=.5;w['rows']=deepcopy(r['rows'])
    elif mutation=='order':r['rows'][0]['condition']='on';w['rows']=deepcopy(r['rows'])
    elif mutation=='history':r['rows'][0]['history_exact']=False;w['rows']=deepcopy(r['rows'])
    elif mutation=='native_return':r['rows'][0]['last_return']['step']=4;w['rows']=deepcopy(r['rows'])
    elif mutation=='missing':r['rows'][12]['end_state']=[0]*7;w['rows']=deepcopy(r['rows'])
    elif mutation=='canvas':r['rows'][12]['last_state'][1]=512;w['rows']=deepcopy(r['rows'])
    elif mutation=='state':ev[0]['receipt']['snapshot']['state'][0]=513
    elif mutation=='prepost':ev[0]['post']={}
    elif mutation=='goal':ev[0]['pre']['goal_pose']=[0,0,0]
    elif mutation=='selected_condition':ev[0]['receipt']['condition']='on'
    elif mutation=='checkpoint':ev[3]['checkpoint']['absolute_step']=19
    elif mutation=='off_replay':ev[2]['receipt']['file_sha256']='different'
    elif mutation=='incident':next(e for e in ev if e.get('job_digest')==r['rows'][12]['job_digest'])['receipt']['overlay_index']=1
    # Preserve the outer accepted-receipt equality so deeper joins are exercised.
    w['rows']=deepcopy(r['rows'])
    with pytest.raises((RuntimeError,AssertionError,ValueError,KeyError)):roster(d)


class RAM:
    def __init__(self):self.arrays={};self.events=[]
    def array(self,name,a):
        assert name not in self.arrays
        self.arrays[name]=np.array(a,copy=True)
        return dict(file=name,dtype=str(a.dtype),shape=list(a.shape),array_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    def event(self,e):self.events.append(deepcopy(e))
    def load(self,r):
        a=self.arrays[r['file']]
        m.require(r==dict(file=r['file'],dtype=str(a.dtype),shape=list(a.shape),array_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()),'RAM receipt exact binding')
        return a


@pytest.fixture
def encoded(monkeypatch):
    specs=roster(physical_fixture())
    for s in specs:s.update(goal192=[.125]*192,predicted192=[.5]*192,predicted_Q=.140625)
    p=dict(mode='partial_observed_only',rows=specs)
    rows=[dict(**deepcopy(s),status='not_started' if s['eligible_for_encoding'] else 'missing_condition_terminal') for s in specs]
    pix=np.zeros((224,224,3),np.uint8);pix[0,0]=[65,105,225]
    monkeypatch.setattr(m,'load_pixels',lambda s:pix.copy())
    store=RAM();modules=[];calls=[]
    def encode(pixels,row):
        calls.append(row['index']);pre=m.preprocess_pixels(pixels)
        row['preprocessed']=store.array(f'p{row["index"]}',pre)
        ein=pre.reshape(1,3,224,224);eout=np.full((1,257,192),.25,np.float32);pout=np.full((1,192),.75,np.float32)
        pair=[]
        for kind,x,y in [('encoder',ein,eout),('projector',eout[:,0],pout)]:
            e=dict(index=len(modules),resource=kind,scope=dict(image_index=row['index']),completed=False)
            store.event(dict(kind='tensor_module_attempt',entry=e))
            e['inputs']=[store.array(f'i{e["index"]}',x)];e['output']=store.array(f'o{e["index"]}',y);e['completed']=True
            store.event(dict(kind='tensor_module_completed',entry=e));modules.append(e);pair.append(e)
        row['modules']=deepcopy(pair);return pout[0].copy()
    m.execute(p,store,encode,rows)
    counts={k:dict(attempted=39 if k in ('encoder','projector') else 0,completed=39 if k in ('encoder','projector') else 0) for k in ('encoder','projector','action_encoder','predictor','pred_proj')}
    return p,dict(rows=rows,modules=modules,counts=counts),store,calls


def test_partial39_actual_dispatch_no_dedup(encoded):
    p,r,s,calls=encoded
    assert calls==list(range(12))+list(range(13,40)) and len(s.events)==234
    m.validate_encoded(p,r,s.events,s.load)
    assert r['rows'][12]['actual_Q'] is None
    assert all(r['rows'][i]['actual_Q']==.390625 for i in calls)


@pytest.mark.parametrize('mutation',['197','missing_module','reorder','scope','count','extra_input','latent','goal','missing_zero','failed','extra_event','return_first','identity','prediction','pixels','preprocess','dtype','Q','MSE','null_reason','false_completion'])
def test_encoded_saved_mutations(encoded,mutation):
    p,r,s,calls=encoded;row=r['rows'][0]
    if mutation=='197':s.arrays[row['modules'][0]['output']['file']]=np.zeros((1,197,192),np.float32)
    elif mutation=='missing_module':r['modules'].pop()
    elif mutation=='reorder':s.events[1],s.events[2]=s.events[2],s.events[1]
    elif mutation=='scope':r['modules'][0]['scope']['image_index']=1
    elif mutation=='count':r['counts']['encoder']['attempted']=40
    elif mutation=='extra_input':row['modules'][0]['inputs']*=2
    elif mutation=='latent':row['actual_latent']={}
    elif mutation=='goal':row['goal192'][0]+=1
    elif mutation=='missing_zero':r['rows'][12]['actual_Q']=0.
    elif mutation=='failed':row['status']='failed_partial'
    elif mutation=='extra_event':s.events.append(dict(kind='unknown'))
    elif mutation=='return_first':s.events=s.events[1:]+s.events[:1]
    elif mutation=='identity':s.events[0]['identity']['condition']='on'
    elif mutation=='prediction':row['predicted192'][0]+=1
    elif mutation=='pixels':s.arrays[row['pixels']['file']][2,3]=[1,2,3]
    elif mutation=='preprocess':s.arrays[row['preprocessed']['file']].flat[0]+=1
    elif mutation=='dtype':s.arrays[row['modules'][0]['output']['file']]=np.zeros((1,257,192),np.float64)
    elif mutation=='Q':row['actual_Q']+=1
    elif mutation=='MSE':row['actual_model_error']+=1
    elif mutation=='null_reason':r['rows'][12]['missing_reason']='agent_pixels_absent'
    elif mutation=='false_completion':r['modules'][0]['completed']=False
    with pytest.raises((RuntimeError,AssertionError,ValueError,KeyError)):m.validate_encoded(p,r,s.events,s.load)


def test_failed_dispatch_no_retry_keeps40(encoded):
    p,_,_,_=encoded;rows=[dict(**deepcopy(s),status='not_started') for s in p['rows']];s=RAM();calls=[]
    def fail(pix,row):calls.append(row['index']);raise RuntimeError('actual callback failure')
    with pytest.raises(RuntimeError,match='actual callback failure'):m.execute(p,s,fail,rows)
    assert calls==[0] and len(rows)==40 and rows[0]['status']=='attempted' and rows[1]['status']=='not_started'
    assert len(s.events)==1 and s.events[0]['kind']=='image_attempt' and len(s.arrays)==1


def test_full40_and_authorization_fail_before_io(monkeypatch):
    def forbidden(*a,**k):pytest.fail('must not access input/output/model')
    monkeypatch.setattr(m,'read',forbidden);monkeypatch.setattr(m,'Store',forbidden)
    with pytest.raises(ValueError,match='full40 blocked'):m.build()
    with pytest.raises(ValueError,match='authorization'):m.run('x','y','z',None)


def test_metrics_original_identity_mean192():
    z=np.arange(192,dtype=np.float32)/32;g=np.ones(192,np.float64)*.3;pred=z+.5
    got=m.metrics(z,g,pred)
    assert got['actual_Q']==float(np.mean((np.eye(192)@(z.astype(float)-g))**2))
    assert got['actual_model_error']==.25
    assert got['actual_Q']!=float(np.sum((z-g)**2))
    with pytest.raises(ValueError):m.metrics(z[:191],g,pred)


def test_numpy_preprocessing_known_values_no_neural():
    pix=np.zeros((224,224,3),np.uint8);pix[0,0]=255
    x=m.preprocess_pixels(pix)
    assert x.dtype==np.float32 and x.shape==(1,1,3,224,224)
    assert np.array_equal(x[0,0,:,0,0],(np.ones(3,np.float32)-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32))
    with pytest.raises(ValueError):m.preprocess_pixels(pix.astype(np.float32))


def test257_bridge_rejects197_independently_of_receipt_hash():
    pre=np.zeros((1,1,3,224,224),np.float32);eout=np.zeros((1,257,192),np.float32);pout=np.zeros((1,192),np.float32)
    m.teacher.overlay.visual_bridge(pre,pre.reshape(1,3,224,224),eout,eout[:,0],pout,pout[0],[1,257,192])
    with pytest.raises(ValueError):m.teacher.overlay.visual_bridge(pre,pre.reshape(1,3,224,224),eout[:,:197],eout[:,0],pout,pout[0],[1,257,192])


def test_lossless_capacity_all39_and_fail_before_execution(monkeypatch):
    rows=[dict(eligible_for_encoding=i!=12) for i in range(40)];calls=[]
    def pixels(r):calls.append(r);return np.zeros((224,224,3),np.uint8)
    c=m.capacity(rows,pixels)
    assert len(calls)==39 and c['pre_final_upper_bound_bytes']<=m.CAP-m.FINAL
    rng=np.random.default_rng(101);a=rng.integers(0,256,(224,224,3),dtype=np.uint8)
    random_capacity=m.capacity(rows,lambda r:a)
    assert random_capacity['pre_final_upper_bound_bytes']>c['pre_final_upper_bound_bytes']
    monkeypatch.setattr(m,'CAP',m.FINAL+random_capacity['pre_final_upper_bound_bytes']-1)
    with pytest.raises(ValueError,match='capacity'):m.capacity(rows,lambda r:a)


@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'])
def test_duplicate_or_nonfinite_json_rejected(raw):
    with pytest.raises((RuntimeError,ValueError)):m.decode(raw)


def test_store_array_three_level_hash_ram():
    s=object.__new__(m.Store);files={};s.write=lambda name,raw:files.setdefault(name,raw)
    a=np.arange(192,dtype=np.float32);r=s.array('latent',a)
    packed=files[r['file']];raw=gzip.decompress(packed)
    assert hashlib.sha256(packed).hexdigest()==r['file_sha256']
    assert hashlib.sha256(raw).hexdigest()==r['npy_sha256']
    assert hashlib.sha256(a.tobytes()).hexdigest()==r['array_sha256']
    assert np.array_equal(np.load(BytesIO(raw),allow_pickle=False),a)
    with pytest.raises(ValueError):s.array('bad',a.astype(np.float64))


def test_store_journal_chain_and_failure_no_retry_ram():
    s=object.__new__(m.Store);s.index=0;s.previous=None;s.broken=False;writes=[]
    s.write=lambda name,raw,**k:writes.append(raw)
    s.event(dict(kind='test'));s.event(dict(kind='second'))
    a,b=[m.decode(x) for x in writes]
    assert b['previous_sha256']==a['event_sha256'] and s.index==2
    assert b['event_sha256']==m.digest({k:v for k,v in b.items() if k!='event_sha256'})
    def fail(*a,**k):raise OSError('RAM short write')
    s.write=fail
    with pytest.raises(OSError):s.event(dict(kind='failure'))
    assert s.broken and s.index==2
    with pytest.raises(ValueError):s.event(dict(kind='no retry'))


def test_partial_file_write_is_charged_in_finally(monkeypatch):
    s=object.__new__(m.Store);s.output=m.ROOT;s.files={};s.used=0;s.check=lambda *a,**k:None
    stream=BytesIO()
    class Short:
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def write(self,b):stream.write(b[:3]);return 3
    monkeypatch.setattr(m.os,'open',lambda *a:7);monkeypatch.setattr(m.os,'close',lambda *a:None)
    monkeypatch.setattr(m.os,'fdopen',lambda *a,**k:Short())
    monkeypatch.setattr(m.os,'fstat',lambda fd:NS(st_mode=0o100600,st_dev=1,st_ino=2,st_size=len(stream.getvalue())))
    with pytest.raises(ValueError,match='short artifact write'):s.write('partial.bin',b'abcdef')
    assert s.used==3 and s.files['partial.bin']==(1,2,3)


class Module:
    def __init__(self):self.pre=[];self.post=[]
    def register_forward_pre_hook(self,h):
        self.pre.append(h);return NS(remove=lambda:self.pre.remove(h))
    def register_forward_hook(self,h):
        self.post.append(h);return NS(remove=lambda:self.post.remove(h))
    def __call__(self,a,out):
        for h in self.pre:h(self,(a,))
        for h in self.post:h(self,(a,),out)
        return out


@pytest.mark.parametrize('mutation',['none','197','forbidden','double','unscoped','wrong_input','budget40'])
def test_real_hook_wiring_ram_shapes_and_caps(mutation):
    model=NS(**{k:Module() for k in ('encoder','projector','action_encoder','predictor','pred_proj')})
    s=RAM();trace=m.Trace(model,NS(cuda=NS(synchronize=lambda:None)),s,lambda:None,[1,257,192])
    # RAM arrays stand in only at trace.save boundary; no torch or CUDA compute.
    trace.save=lambda x,name:s.array(name,x)
    with trace:
        trace.scope=None if mutation=='unscoped' else dict(image_index=0)
        if mutation=='budget40':trace.entries=[dict(resource='encoder',scope=dict(image_index=i)) for i in range(39)]
        x=np.zeros((1,3,224,224),np.float32);out=np.zeros((1,257,192),np.float32)
        if mutation=='wrong_input':x=x[:,:,:223]
        if mutation=='197':out=out[:,:197]
        if mutation=='forbidden':
            with pytest.raises(ValueError):model.predictor(x,out)
            assert not trace.entries
        elif mutation!='none':
            with pytest.raises(ValueError):
                model.encoder(x,out)
                if mutation=='double':model.encoder(x,out)
        else:
            model.encoder(x,out);model.projector(out[:,0],np.zeros((1,192),np.float32))
            assert [e['resource'] for e in trace.finish(dict(image_index=0))]==['encoder','projector']
    assert all(not getattr(model,k).pre and not getattr(model,k).post for k in vars(model))


@pytest.mark.parametrize('mutation',['none','sha','tail','partial_line','chain','nonterminal','duplicate_key','index'])
def test_physical_journal_terminal_binding_ram(monkeypatch,mutation):
    events=[];prev=None
    for k in ('run_started','fresh_observation','checkpoint_recorded','run_finished'):
        e=dict(index=len(events),previous_sha256=prev,kind=k)
        if k=='fresh_observation':e['receipt']=dict(phase='suffix',step=5)
        if k=='checkpoint_recorded':e['checkpoint']=dict(step=10)
        e['event_sha256']=m.digest(e);prev=e['event_sha256'];events.append(e)
    tail=prev;expected='original'
    if mutation=='tail':tail='other'
    if mutation=='chain':events[1]['previous_sha256']='other'
    if mutation=='index':events[1]['index']=9
    if mutation=='nonterminal':events=events[:-1];tail=events[-1]['event_sha256']
    raw=b''.join(m.canonical(e)+b'\n' for e in events)
    if mutation=='partial_line':raw=raw[:-1]
    if mutation=='duplicate_key':raw=raw.replace(b'"kind":"run_started"',b'"kind":"run_started","kind":"run_started"')
    path=NS(stat=lambda:NS(st_size=len(raw)),open=lambda *a:BytesIO(raw))
    monkeypatch.setattr(m,'Path',lambda p:p);monkeypatch.setattr(m,'sha',lambda p:'different' if mutation=='sha' else expected)
    if mutation=='none':assert [e['kind'] for e in m.terminal_events(path,expected,tail)]==['fresh_observation','checkpoint_recorded']
    else:
        with pytest.raises(ValueError):m.terminal_events(path,expected,tail)


@pytest.mark.parametrize('secondary',['none','journal','closure','report'])
def test_run_finally_retains40_without_importing_neural(monkeypatch,secondary):
    p=dict(mode='partial_observed_only',rows=roster(physical_fixture()),bindings={},expected_model_hash='frozen',
           inherited_compiler_resources=dict(predict=96),inherited_physical_counts=dict(public_step=800))
    captured={}
    class FailureStore:
        def __init__(self,*a):self.index=0;self.previous=None
        def check(self,*a):return None
        def put(self,name,value,**k):
            if name=='binding.json':raise OSError('pre-model RAM write failure')
            captured[name]=deepcopy(value)
            if name=='report.json' and secondary=='report':raise OSError('secondary report failure')
        def event(self,e):
            if secondary=='journal':raise OSError('secondary journal failure')
            self.index+=1
        def census(self):return {}
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'Store',FailureStore)
    monkeypatch.setattr(m,'sha',lambda *a:'prepare')
    def closure(*a):
        if secondary=='closure':raise ValueError('secondary closure failure')
    monkeypatch.setattr(m,'verify_bindings',closure)
    with pytest.raises(ValueError,match='partial retained'):m.run('/RAM/prepare','prepare','/RAM/output',m.AUTH)
    r=captured['report.json']
    assert len(r['rows'])==40 and r['image_attempted']==r['image_completed']==0 and r['missing']==[12]
    assert r['model_hash_before'] is r['model_hash_after'] is None and r['modules']==[]
    assert r['errors'][0]['stage']=='execution'
    if secondary in ('journal','closure'):assert len(r['errors'])==2
    if secondary=='report':assert captured['failure.json']['errors'][-1]['stage']=='final_report'
