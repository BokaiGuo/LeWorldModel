"""Synthetic real-ABI tests only: no torch, CUDA, gym or physical dispatch."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v2 as m


def request():
    return dict(schema=m.NAME+'.forward', anchor_index=0, word=0,
                latents=np.zeros((3,192),np.float32).tolist(),
                history=np.zeros((2,10),np.float32).tolist(),
                raw5_bits=np.zeros((5,2),'<f4').tobytes().hex())


def test_whitelist_rejects_physical_labels():
    m.validate_request(request())
    for key in ('state','reward','snapshot','native_goal','distance','selected_word'):
        r=request(); r[key]=0
        with pytest.raises(ValueError): m.validate_request(r)


def test_stable_model_only_selection_and_mse192():
    p=np.zeros((5,192),np.float32); p[:,0]=[2,1,1,3,4]
    s=m.selection(p.tolist(),np.zeros(192,np.float32).tolist())
    assert s['selected_word']==1
    assert s['costs'][1]==1/192
    assert s['seal_sha256']==m.digest({k:v for k,v in s.items() if k!='seal_sha256'})


def test_literal_replay_fingerprint_detects_signed_zero():
    f=dict(raw20_bits='00'*160,state7=[0.]*7,snapshot={'angle':0.},png_sha256='a'*64,pixel_sha256='b'*64)
    m.same_start(f,deepcopy(f))
    other=deepcopy(f); other['state7'][0]=-0.
    with pytest.raises(ValueError): m.same_start(f,other)


def test_real_row_initialization_and_fixed_denominator():
    row=m.new_row(dict(index=3,condition='on',job_digest='a'*64))
    assert row['missing_checkpoints']==[0,5,10]
    assert row['public']['suffix']=={'attempted':0,'completed':0}
    assert row['endpoint'] is None and row['alternate_endpoint'] is None
    assert m.CAPS['public_step']==1000 and m.CAPS['physics_step']==10160


def test_store_physical_and_model_receipts(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_fixture'))
    a=np.zeros((1,3,192),np.float32); rec=s.array('x',a)
    assert np.array_equal(m.load_array(s.path,rec),a)
    s.frame('rgb.png',np.zeros((224,224,3),np.uint8))
    s.arrays('info.npz',x=np.array([1],np.float64))
    s.emit('operation_attempted',operation_index=0,resource='reset',fields={})
    s.emit('operation_returned',operation_index=0,resource='reset',fields={})
    events=m.read_events(s.path/'journal.jsonl')
    assert m.count_operations(events)['reset']==dict(attempted=1,completed=1)
    assert len(s.census())==4


def test_failed_native_attempt_not_refunded(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_fixture')); meter=m.Meter(s)
    def fail(): raise RuntimeError('native reset failed')
    with pytest.raises(RuntimeError): meter.call('reset',fail)
    events=m.read_events(s.path/'journal.jsonl')
    assert m.count_operations(events)==meter.receipt()
    assert meter.receipt()['reset']==dict(attempted=1,completed=0)


def test_consumer_rejects_return_without_attempt():
    with pytest.raises(ValueError):
        m.count_operations([dict(kind='operation_returned',operation_index=0,resource='reset',fields={})])


def test_store_no_overwrite_or_foreign_file(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_fixture')); s.put('a.json',{})
    with pytest.raises(ValueError): s.put('a.json',{})
    (s.path/'foreign').touch()
    with pytest.raises(ValueError): s.census()


def test_journal_rejects_tampered_chain(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_fixture')); s.emit('x',value=1)
    path=s.path/'journal.jsonl'
    event=json.loads(path.read_text()); event['value']=2
    path.write_text(json.dumps(event)+'\n')
    with pytest.raises(ValueError): m.read_events(path)


def test_float32_forward_no_double_normalization():
    from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v1 as old
    z=np.arange(576,dtype=np.float32).reshape(3,192)
    raw=np.full((10,2),.1,np.float32)
    f=m.actual_forward(z.tolist(),np.ones(192,np.float32).tolist(),raw.tobytes().hex())
    assert f==old.forward20(z.tolist(),np.ones(192,np.float32).tolist(),raw.tobytes().hex(),
                           m.teacher.kernel.ACTION_MEAN.tolist(),m.teacher.kernel.ACTION_STD.tolist())


def test_real_triplet_schema_no_renderer_state_change():
    trip=[]
    for i in range(3):
        trip.append(dict(kind='fresh_observation',pre={'snapshot':{'x':1}},post={'snapshot':{'x':1}},
            receipt=dict(phase='suffix',step=10,overlay_index=i,condition='on' if i==1 else 'off',
                         png=f'v{i}.png',file_sha256='a'*64 if i!=1 else 'b'*64,
                         pixel_sha256='c'*64 if i!=1 else 'd'*64,state=[0.]*7,snapshot={'x':1})))
    m.validate_triplet(trip)
    trip[1]['post']['snapshot']['x']=2
    with pytest.raises(ValueError): m.validate_triplet(trip)


def test_foreign_scope_operation_rejected():
    r=dict(rows=[dict(index=0,anchor_index=0,word=0,job_digest='a')])
    with pytest.raises(ValueError):
        m.validate_scopes(r,[dict(kind='raw_return',job_digest='foreign',anchor_index=0,word=0)])


def test_module_caps_are_measured_not_forward_inferred():
    es=[dict(resource='encoder',completed=True),dict(resource='projector',completed=False)]
    assert m.module_counts(es)==dict(encoder=dict(attempted=1,completed=1),projector=dict(attempted=1,completed=0))


def test_native_projection_does_not_count_extension_or_neural_work():
    row=m.new_row(dict(index=0,anchor_index=0,word=0,job_digest='x',condition='off'))
    row.update(replay20=dict(state7=[0.]*7),PNG=81)
    row['trajectory']=[dict(n=i) for i in range(27)]
    events=[dict(kind='run_started'),dict(kind='replay20_ready',index=2,job_digest='x'),
            dict(kind='operation_attempted',index=3,job_digest='x',resource='predict_API'),
            dict(kind='fresh_observation',index=4,job_digest='x')]
    rows,filtered=m.native20_projection([row],events)
    assert rows[0]['PNG']==66 and len(rows[0]['trajectory'])==22
    assert rows[0]['status']=='horizon_completed'
    assert all(e.get('index',0)<=2 for e in filtered)
    assert row['PNG']==81  # Never mutate actual25 authority.


def test_real_orchestration_fixed_order_selection_precedes_all_outcomes(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    calls=[];packet=np.zeros((5,2),'<f4').tobytes().hex()
    js=[dict(anchor_index=i,source_slot=i//4,condition='off',first_selected_original_index=i*5,
             prefix5_bits=packet,first_suffix5_bits=packet,words=[dict(raw5_bits=packet) for _ in range(5)]) for i in range(8)]
    parent=dict(teacher_inputs=[dict(context=dict(latent_history=np.zeros((3,192),np.float32).tolist())) for _ in range(8)],
                anchors=[dict(goal=np.zeros(192,np.float32).tolist()) for _ in range(8)])
    rows=[m.new_row(dict(index=i,anchor_index=i//5,word=i%5,condition='off',job_digest=str(i))) for i in range(40)]
    store=m.Store(tmp_path/(m.NAME+'_fixture'));meter=m.Meter(store)
    monkeypatch.setattr(m.phys,'validate_env',lambda _:None)
    monkeypatch.setattr(m.phys,'native_hooks',lambda *args:nullcontext())
    def replay(env,parent,p,data,j,row,meter,store):
        f=dict(raw20_bits='00'*160,state7=[0.]*7,snapshot={},png_sha256='a'*64,pixel_sha256='b'*64)
        row['replay20']=f;return np.zeros((224,224,3),np.uint8),f
    monkeypatch.setattr(m,'replay20',replay)
    def extend(env,j,w,row,meter,store,backend,seal):
        assert len([x for x in calls if x==('predict',j['anchor_index'])])==5
        assert (store.path/f'seal_{j["anchor_index"]}.json').is_file()
        calls.append(('outcome',j['anchor_index'],w));row.update(status='terminal25_encoded')
    monkeypatch.setattr(m,'extend25',extend)
    class Backend:
        def encode(self,*args):return np.zeros(192,np.float32).tolist()
        def predict(self,r):calls.append(('predict',r['anchor_index']));return np.zeros(192,np.float32).tolist()
    def factory():
        env=SimpleNamespace(unwrapped=SimpleNamespace(render_action=False,render_size=224,render_mode='rgb_array'),close=lambda:None)
        return env
    anchors=[dict(anchor_index=i,seal=None) for i in range(8)]
    m.execute_panel(dict(jobs=js),parent,{}, {},store,meter,Backend(),rows,anchors,factory)
    assert len([x for x in calls if x[0]=='outcome'])==40
    assert all(a['seal']['selected_word']==0 for a in anchors)
    assert meter.receipt()['constructor']==meter.receipt()['close']==dict(attempted=40,completed=40)


def test_failed_leader_retains_five_unknown_slots_no_retry(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    j=dict(anchor_index=0,condition='off');rows=[m.new_row(dict(index=i,anchor_index=0,word=i,condition='off',job_digest=str(i))) for i in range(5)]
    store=m.Store(tmp_path/(m.NAME+'_fixture'));meter=m.Meter(store)
    monkeypatch.setattr(m.phys,'validate_env',lambda _:None)
    monkeypatch.setattr(m.phys,'native_hooks',lambda *args:nullcontext())
    def fail(*args):raise ValueError('real replay20 mismatch')
    monkeypatch.setattr(m,'replay20',fail)
    factory=lambda:SimpleNamespace(unwrapped=SimpleNamespace(render_action=False,render_size=224,render_mode='rgb_array'),close=lambda:None)
    m.execute_panel(dict(jobs=[j]),{},{},{},store,meter,None,rows,[dict(anchor_index=0,seal=None)],factory)
    assert len(rows)==5 and all(r['endpoint'] is None for r in rows)
    assert meter.receipt()['constructor']['attempted']==1
    assert len(list(store.path.glob('branch_*.json')))==5


def test_saved_native_metric_preserves_velocity_and_success_separation():
    goal=[0.]*7;state=[0.]*5+[3.,4.]
    assert m.native_labels(goal,state)==dict(native_success=True,native_distance=5.)


def test_journal_requires_canonical_complete_line(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_fixture'));s.emit('x',value=1)
    path=s.path/'journal.jsonl';path.write_bytes(path.read_bytes().rstrip(b'\n'))
    with pytest.raises(ValueError):m.read_events(path)


def test_module_tensor_consumption_rejects_changed_dtype(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_fixture'));r=s.array('x',np.ones((192,),np.float32))
    r['dtype']='float64'
    with pytest.raises(ValueError):m.load_array(s.path,r)


class Tensor:
    """NumPy tensor ABI fake, NOT a CPU neural implementation."""
    def __init__(self,a):
        from types import SimpleNamespace
        self.a=np.asarray(a);self.device=SimpleNamespace(type='cuda')
    @property
    def shape(self):return self.a.shape
    @property
    def dtype(self):return self.a.dtype
    def __getitem__(self,index):return Tensor(self.a[index])
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return Tensor(np.ascontiguousarray(self.a))
    def numpy(self):return self.a


class Module:
    def __init__(self,fn):self.fn=fn;self.pre=[];self.post=[]
    def handle(self,items,fn):
        from types import SimpleNamespace
        items.append(fn);return SimpleNamespace(remove=lambda:items.remove(fn))
    def register_forward_pre_hook(self,fn):return self.handle(self.pre,fn)
    def register_forward_hook(self,fn):return self.handle(self.post,fn)
    def __call__(self,*args):
        for fn in self.pre:fn(self,args)
        out=self.fn(*args)
        for fn in self.post:fn(self,args,out)
        return out


def fake_cuda_backend(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys
    from experiments import spectral_reachability as package
    p0=SimpleNamespace(image_transform=lambda:None,_image_tensor=lambda *a,**kw:Tensor(np.zeros((1,1,3,224,224),np.float32)))
    monkeypatch.setitem(sys.modules,package.__name__+'.preflight_pusht_d2p_p0',p0)
    monkeypatch.setattr(package,'preflight_pusht_d2p_p0',p0,raising=False)
    torch=SimpleNamespace(float32=np.dtype('float32'),inference_mode=nullcontext,device=lambda d:d,
        tensor=lambda a,**kw:Tensor(np.asarray(a,np.float32)),cuda=SimpleNamespace(synchronize=lambda:None))
    model=SimpleNamespace()
    model.encoder=Module(lambda x:Tensor(np.zeros((1,257,192),np.float32)))
    model.projector=Module(lambda x:x)
    model.action_encoder=Module(lambda x:Tensor(np.zeros((1,3,192),np.float32)))
    model.predictor=Module(lambda z,a:z)
    model.pred_proj=Module(lambda x:x)
    model.encode=lambda d:dict(emb=model.projector(model.encoder(d['pixels'][:,0])[:,0])[:,None])
    model.predict=lambda z,a:model.pred_proj(Tensor(model.predictor(z,a).a.reshape(3,192)))[None]
    store=m.Store(tmp_path/(m.NAME+'_fixture'));meter=m.Meter(store)
    trace=m.NativeTrace(model,torch,store,lambda:None,[1,257,192])
    return m.CudaModel(model,torch,trace,store,meter),trace,model,p0


def test_real_cuda_adapter_full_module_schema_with_no_neural_fake(tmp_path,monkeypatch):
    backend,trace,model,p0=fake_cuda_backend(tmp_path,monkeypatch)
    with trace:
        assert backend.encode(np.zeros((224,224,3),np.uint8),'actual20')==[0.]*192
        for w in range(5):
            r=request();r['word']=w;assert backend.predict(r)==[0.]*192
    assert backend.meter.receipt()==dict(encode_API=dict(attempted=1,completed=1),predict_API=dict(attempted=5,completed=5))
    assert m.module_counts(trace.entries)=={k:dict(attempted=v,completed=v) for k,v in dict(encoder=1,projector=1,action_encoder=5,predictor=5,pred_proj=5).items()}
    assert m.count_operations(m.read_events(backend.store.path/'journal.jsonl'))==backend.meter.receipt()


def test_preprocessing_failure_is_not_encode_API_attempt(tmp_path,monkeypatch):
    backend,trace,model,p0=fake_cuda_backend(tmp_path,monkeypatch)
    def fail(*args,**kwargs):raise RuntimeError('preprocess before encode')
    p0._image_tensor=fail
    with pytest.raises(RuntimeError),trace:backend.encode(np.zeros((224,224,3),np.uint8),'actual20')
    assert backend.meter.receipt()=={}


def test_AE_failure_is_not_predict_API_attempt(tmp_path,monkeypatch):
    backend,trace,model,p0=fake_cuda_backend(tmp_path,monkeypatch)
    def fail(*args):raise RuntimeError('AE before predict')
    model.action_encoder.fn=fail
    with pytest.raises(RuntimeError),trace:backend.predict(request())
    assert backend.meter.receipt()=={}
    assert m.module_counts(trace.entries)==dict(action_encoder=dict(attempted=1,completed=0))


def test_consume_durable_load_failure_keeps_all80_unknown(tmp_path,monkeypatch):
    packet=np.zeros((5,2),np.float32).tolist();bits=np.zeros((5,2),'<f4').tobytes().hex()
    js=[dict(anchor_index=i,first_selected_original_index=i*5,words=[dict(raw5_bits=bits) for _ in range(5)]) for i in range(8)]
    sourcejobs=[dict(index=i,anchor_index=i//5,task=i//10,root_slot=i//20,condition='off',
                    job_digest=str(i),prefix_commands=packet,suffix_commands=packet) for i in range(40)]
    p=dict(tasks=[dict(native=dict(goal=[0.]*7)) for _ in range(4)])
    parent=dict(jobs=sourcejobs,probe_digest=m.digest(p),references=[dict(raw15=[],frames51=[]) for _ in range(8)],teacher_inputs=[])
    prepared=dict(jobs=js,expected_model_hash='a'*64)
    rows=[m.new_row(j) for j in m.production_jobs(prepared,parent)]
    monkeypatch.setattr(m,'verify',lambda *args:(prepared,parent,p,{}))
    store=m.Store(tmp_path/(m.NAME+'_fixture'));binding=dict(prepared='/not_executed.json',prepare_sha256='a'*64)
    errors=[dict(phase='execution',type='RuntimeError',message='CUDA load failed')]
    store.put('binding.json',binding);store.emit('run_started',binding=binding,contract=m.contract())
    store.emit('run_finished',status='partial',counts={},errors=errors)
    report=dict(schema=m.NAME+'.report',contract=m.contract(),P1_admission=False,rows=rows,
        anchors=[dict(anchor_index=i,seal=None) for i in range(8)],binding=binding,counts={},status='partial',
        images=[],forwards=[],native_module_attempts=[],module_counts={},errors=errors,
        artifacts=store.census(),output_bytes_before_report=store.used)
    store.put('report.json',report,final=True)
    result=m.consume(store.path,m.sha(store.path/'report.json'))
    assert result['fixed_slots']==40 and result['unknown_pairs']==80 and result['available_pairs']==0
    assert result['status']=='partial_saved_audit' and result['P1_admission'] is False


def test_alternate_encode_failure_does_not_erase_known_factual_endpoint(tmp_path,monkeypatch):
    from types import SimpleNamespace
    rgb=np.zeros((224,224,3),np.uint8);state=[0.]*7
    store=m.Store(tmp_path/(m.NAME+'_fixture'));meter=m.Meter(store)
    monkeypatch.setattr(m,'pixels',lambda *args:rgb)
    fixed=dict(snapshot=dict(state=state),goal=[0.]*7)
    monkeypatch.setattr(m.phys,'invariant',lambda *args:deepcopy(fixed))
    monkeypatch.setattr(m.phys.rgb_guard,'RgbObservabilityGuard',lambda **kw:SimpleNamespace(observe=lambda *args,**kw:None))
    def dispatch(env,a,phase,step,row,*args):
        row['last_return']=dict(state=state);return dict(state=state),False,False
    monkeypatch.setattr(m.phys,'dispatch',dispatch)
    def observe(env,obs,phase,step,row,meter,store,*args):
        trip=[dict(kind='fresh_observation',pre=deepcopy(fixed),post=deepcopy(fixed),receipt=dict(overlay_index=i,
            phase=phase,step=step,condition='on' if i==1 else 'off',state=state,snapshot=fixed['snapshot'],
            file_sha256='a'*64,pixel_sha256='b'*64,png=str(i))) for i in range(3)]
        store.observations.extend(trip);row['trajectory'].append(trip[0]['receipt'])
        return rgb,dict(state=state)
    monkeypatch.setattr(m.phys,'observe',observe)
    row=m.new_row(dict(index=0,condition='off'));row.update(goal=state,root_sha256='a'*64,trajectory=[{}])
    n=[]
    def encode(*args):
        n.append(1)
        if len(n)==2:raise RuntimeError('alternate CUDA encode failed')
        return [0.]*192
    j=dict(anchor_index=0,condition='off',words=[dict(raw5_bits=np.zeros((5,2),'<f4').tobytes().hex()) for _ in range(5)])
    m.extend25(SimpleNamespace(unwrapped=SimpleNamespace()),j,0,row,meter,store,SimpleNamespace(encode=encode),dict(seal_sha256='a'*64))
    assert row['status']=='terminal25_original_encoded' and row['endpoint']==[0.]*192
    assert row['alternate_endpoint'] is None and row['alternate_error']['type']=='RuntimeError'


def test_original_saved40_native_schema_projection_without_producer():
    """Read-only original artifacts, not a fresh trajectory/encoder experiment."""
    parent=m.teacher.read(m.PARENT,m.PARENT_SHA)
    p=m.teacher.read(parent['probe_prepared'],parent['probe_prepare_sha256'])
    directory=m.ROOT/'results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02'
    saved=m.teacher.read(directory/'report.json','464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01',m.phys.REPORT_CAP)
    original=m.phys.read_events(directory/'journal.jsonl',saved['journal_tail'])
    rows=[];bydigest={}
    for j,oldrow in zip(parent['jobs'],saved['rows'],strict=True):
        row=m.new_row(j);row.update(deepcopy(oldrow));row.update(word=j['library_index'],replay_original_index=j['index'])
        if oldrow['status']=='horizon_completed':row['replay20']=dict(state7=deepcopy(oldrow['end_state']))
        rows.append(row);bydigest[j['job_digest']]=row
    events=[]
    for event in original:
        row=bydigest.get(event.get('job_digest'))
        if row is not None and row['replay20'] is not None and event['kind']=='operation_attempted' and event['resource']=='close':
            events.append(dict(kind='replay20_ready',index=len(events),job_digest=row['job_digest']))
        e=deepcopy(event);e['index']=len(events);events.append(e)
    m.validate_native20(parent,p,dict(rows=rows),events,directory)


def test_exact_array_bridge_distinguishes_zero_sign_and_dtype():
    positive=np.array([0.],np.float32);negative=np.array([-0.],np.float32)
    assert m.exact_array(positive,positive.copy())
    assert not m.exact_array(positive,negative)
    assert not m.exact_array(positive,positive.astype(np.float64))


def test_new_jobs_are_rebuilt_from_original_selected_plan_not_report_fields():
    packet=np.zeros((5,2),'<f4').tobytes().hex()
    js=[dict(anchor_index=i,first_selected_original_index=i*5+2,words=[dict(raw5_bits=packet) for _ in range(5)]) for i in range(8)]
    parent=dict(jobs=[dict(index=i,condition='off',job_digest='old',prefix_commands=[],suffix_commands=[]) for i in range(40)])
    jobs=m.production_jobs(dict(jobs=js),parent)
    assert len(jobs)==40
    assert [j['replay_original_index'] for j in jobs]==[i*5+2 for i in range(8) for _ in range(5)]
    assert len({j['job_digest'] for j in jobs})==40
    assert all(j['extension_raw5_bits']==packet for j in jobs)
