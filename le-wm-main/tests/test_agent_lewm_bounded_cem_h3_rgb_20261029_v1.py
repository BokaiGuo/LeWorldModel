"""Synthetic NumPy/fake-hook models + real temporary disk; no torch/NN/physics production."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_lewm_bounded_cem_h3_rgb_20261029_v1 as m
import test_agent_lewm_bounded_cem_h3_physical_20261029_v1 as ph
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc


def put(path,value):
    path.write_bytes(m.raw(value));return m.sha(path)


def disk_build(tmp_path,mp,*,early=True):
    cem=tmp_path/'cem';cem.mkdir()
    pp=tmp_path/'cem_prepare.json';ur=tmp_path/'cem_ownreceipt.json'
    mp.setattr(m.producer,'PLANNER_OUTPUT',cem);mp.setattr(m.producer,'PLANNER_PREP',pp)
    mp.setattr(m.producer,'PLANNER_RECEIPT',ur)
    _,data,c,u=ph.fixture()
    policy=tmp_path/'policy';policy.mkdir()
    config=policy/'config.json';config.write_bytes((m.bridge.rgb.POLICY/'config.json').read_bytes())
    weights=policy/'fixture_weights.bin';weights.write_bytes(b'RAM FAKE model bytes, NEVER loaded')
    c['policy']=str(policy);c['expected_model_hash']=m.bridge.rgb.MODEL_TENSOR_SHA
    for i,slot in enumerate(c['slots']):
        mean=np.asarray([-.0087,.0068],np.float32);std=np.asarray([.2019,.2002],np.float32)
        emb=np.full((3,192),np.float32(.125*i),np.float32);goal=np.full(192,.5,np.float32)
        slot.update(mean=mean.tolist(),std=std.tolist(),q=np.eye(192).tolist(),
            emb=emb.tolist(),goal_emb=goal.tolist())
        warm=np.asarray(slot['warm_raw10'],np.float32)
        context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=emb.tolist(),action_history=m.normalized_history(slot,warm).tolist())
        slot['packet']=dict(root_context=context,goal=goal.tolist())
    prep_sha=put(pp,c);mp.setattr(m.producer,'PREP_SHA',prep_sha)
    cr=dict(schema=m.producer.PRODUCER+'.report',elapsed_seconds=.123,rows=[],artifacts={})
    futures=[]
    for i,slot in enumerate(c['slots']):
        f=np.zeros((1,1,6,192),np.float32);f[0,0,:3]=slot['emb']
        for j in range(3):f[0,0,3+j]=np.float32((i+j+1)/8)
        path=cem/('whole%d.npy'%i);np.save(path,f,allow_pickle=False);futures.append(f)
        rec=dict(file=path.name,sha256=m.sha(path),file_bytes=path.stat().st_size,dtype='<f4',shape=[1,1,6,192],nbytes=4608)
        cr['artifacts'][path.name]={k:rec[k] for k in ('file_bytes','sha256')}
        cr['rows'].append(dict(root=i,task_id=slot['task_id'],task_sha256=slot['task_sha256'],
            delivery=dict(root=i,status='paid_finalmean_evaluated',layouts=dict(whole=dict(predicted_emb=rec,attempted=True,returned=True)))))
    h=put(cem/'report.json',cr);mp.setattr(m.producer,'REPORT_SHA',h)
    u['report_sha256']=h;mp.setattr(m.producer,'RECEIPT_SHA',put(ur,u))
    p=m.producer.assemble({},c,u,data)
    p['bindings']={str(pp):prep_sha,str(cem/'report.json'):h,str(ur):m.sha(ur),
        str(config):m.sha(config),str(weights):m.sha(weights)}
    p['bindings'].update({str(cem/r['file']):r['sha256'] for r in [x['delivery']['layouts']['whole']['predicted_emb'] for x in cr['rows']]})
    prepared=tmp_path/(m.producer.NAME+'_prepare_RAM.json');prepare_sha=put(prepared,p)
    store=m.producer.Store(tmp_path/(m.producer.NAME+'_physics_RAM'))
    mp.setattr(m.producer,'place',ph.fake_place);mp.setattr(m.producer,'validate_env',lambda _:None)
    counter=[]
    def factory():
        i=len(counter);counter.append(i)
        return ph.Env(fault='terminal' if early and i==0 else None,at=20)
    r=m.producer.execute(p,data,store,factory,authorization=m.producer.AUTH,
        prepared=str(prepared),prepare_sha256=prepare_sha)
    assert r['status']==m.producer.RECORDED,r['errors']
    # Independent RAM-only construction of the authority that production MAIN
    # already persisted. Never call the production physical consume or verify.
    u2=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,rows=deepcopy(r['rows']),
        plans=deepcopy(p['plans']),planner_authority=deepcopy(p['planner_authority']),
        counts=r['counts'],errors=[],fixed_denominator=2,known_roots=2,fresh_roots=0,
        completion=m.read(store.path/'completion.json'),completion_sha256=m.sha(store.path/'completion.json'),
        physical_report_sha256=m.sha(store.path/'report.json'),
        saved_PNG_verified=147 if early else 162,**{k:False for k in ('P1_admission','P2_admission','scientific_admission','physical_release','fair_ranking')})
    u2['bindings']=dict(p['bindings'],**{str(f):m.sha(f) for f in store.path.iterdir()})
    u2['bindings'][str(prepared)]=prepare_sha
    receipt=tmp_path/'RAM_complete_physical_receipt.json';put(receipt,u2)
    args=dict(physical_prepared=str(prepared),physical_prepare_sha=prepare_sha,physical_output=str(store.path),
        physical_report_sha=m.sha(store.path/'report.json'),physical_completion_sha=m.sha(store.path/'completion.json'),
        physical_receipt=str(receipt),physical_receipt_sha=m.sha(receipt))
    mp.setattr(m,'ACTUAL',args)
    mp.setattr(m.producer,'consume',lambda *a:pytest.fail('must not rerun upstream fullstrict'))
    mp.setattr(m.producer,'verify',lambda *a:pytest.fail('must not rerun upstream verify/material'))
    shapes=m.bridge.rgb.config_shapes(m.bridge.rgb.read(config))
    mp.setattr(m,'model_metadata',lambda:(str(policy),shapes,{str(f):m.sha(f) for f in (config,weights)}))
    mp.setattr(m,'source_files',lambda:m.own_files()+[Path(m.producer.__file__),Path(m.primitive.__file__),Path(m.codec.__file__)])
    p2=m.build(**args);sealed=tmp_path/(m.NAME+'_prepare_RAM.json');m.write(sealed,p2)
    assert m.raw(m.verify(sealed,m.sha(sealed)))==m.raw(p2)
    return p2,sealed,p,u2,futures


def encode_fixture(p,store,*,tokens=257,forbidden=None):
    rows=deepcopy(p['rows']);model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    guard=m.Ledger(p['expected_calls']['encoder'],lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    original=model.predict
    def encode(pixels,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pixels)
            row['preprocessed']=trace.save(enc.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(enc.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,store,encode)
    assert not errors and model.predict==original
    return rows,trace.entries,guard.entries,guard.blocked


def completed_disk(tmp_path,mp,*,early=True):
    p,prepared,*_=disk_build(tmp_path,mp,early=early)
    s=m.Store(tmp_path/(m.NAME+'_RAM'),m.Budget())
    s.put('binding.json',dict(prepared=str(prepared),prepare_sha256=m.sha(prepared)))
    s.put('initial_roster.json',p['rows'])
    rows,modules,guards,blocked=encode_fixture(p,s)
    seal=dict(rows=rows,modules=modules,guards=guards,blocked=blocked,errors=[])
    h=s.put('native_seal.json',seal,final=True)
    s.event(dict(kind='all8_native_sealed',sha256=h,encoded=p['expected_calls']['encoder']),final=True)
    inv=[dict(type='RAM_ONLY',path=str(Path(m.__file__).resolve()),sha256=m.sha(m.__file__))]
    r=dict(schema=m.NAME+'.report',status=m.RECORDED,prepared=str(prepared),prepare_sha256=m.sha(prepared),
        contract=m.contract(),**seal,counts=m.bridge.counts(modules),model_hash_before=p['expected_model_hash'],
        model_hash_after=p['expected_model_hash'],model_unchanged=True,source_verified=True,native_seal_sha256=h,
        artifact_sha256=deepcopy(s.hashes),event_count=s.index,elapsed_seconds=0.,
        precision_before=m.precision_contract(),precision_after=m.precision_contract(),
        model_inventory_before=inv,model_inventory_after=inv,CUDA_peak_bytes=[1,1],
        known_roots=2,fixed_denominator=8,physical_consume_rebuilt=False,CEM_numerical_fullstrict_rebuilt=False,**m.FLAGS)
    publication=m.publish(s,r,s.budget.started)
    return p,s,r,publication


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    # Keep fixture source bindings stable for all independent saved consumers.
    with pytest.MonkeyPatch.context() as mp:
        result=completed_disk(tmp_path_factory.mktemp('RGB7_CEM'),mp)
        saved=dict(ACTUAL=deepcopy(m.ACTUAL),PLANNER_PREP=m.producer.PLANNER_PREP,PLANNER_OUTPUT=m.producer.PLANNER_OUTPUT,
            PHYSICAL_SOURCE_SHA=m.PHYSICAL_SOURCE_SHA)
        yield result,saved


def install_fixture(mp,p,saved):
    mp.setattr(m,'ACTUAL',saved['ACTUAL'])
    mp.setattr(m.producer,'PLANNER_PREP',saved['PLANNER_PREP'])
    mp.setattr(m.producer,'PLANNER_OUTPUT',saved['PLANNER_OUTPUT'])
    # Consumers use a sealed prepare reconstruction seam, never actual authority.
    mp.setattr(m,'verify',lambda *a:deepcopy(p))


def test_full_temp_build_seven_encode_eight_slots(complete,monkeypatch):
    (p,s,r,pub),saved=complete;install_fixture(monkeypatch,p,saved)
    out=m.consume(s.path,pub['report_sha256'],pub['completion_sha256'])
    assert out['fixed_denominator']==8 and out['encoded']==7 and out['missing']==1
    assert s.arrays==49 and s.index==71 and len(list(s.path.iterdir()))==125
    assert out['sample_frame_counts']=={k:dict(API=7,samples=7,frames=7) for k in ('encoder','projector')}
    missing=out['rows'][3]
    assert missing['root_slot']==0 and missing['checkpoint']==15 and missing['actual192'] is None
    assert out['comparisons'][3]['actual_encoded_Q'] is None and out['comparisons'][3]['native_success'] is None
    assert missing['planned_raw10_bits'] is not None and missing['causal_raw10_bits'] is None
    assert out['counts']['predict_API']==dict(attempted=0,completed=0)
    assert not out['physical_consume_rebuilt'] if 'physical_consume_rebuilt' in out else not out['full_upstream_rebuilt_here']
    assert all(out[k] is False for k in m.FLAGS)


def test_full_eight_fixture_remains_legal_not_hardcoded_missing(tmp_path,monkeypatch):
    p,s,r,pub=completed_disk(tmp_path,monkeypatch,early=False)
    out=m.consume(s.path,pub['report_sha256'],pub['completion_sha256'])
    assert len(out['rows'])==8 and out['encoded']==8 and out['missing']==0 and s.arrays==56


def test_fixed_mapping_causal_history_and_zero_newpredict(tmp_path,monkeypatch):
    p,_,native,u,futures=disk_build(tmp_path,monkeypatch)
    assert [(x['root_slot'],x['checkpoint'],x['absolute_step']) for x in p['rows']]==[(i,t,10+t) for i in (0,1) for t in m.CP]
    for row in p['rows']:
        i=row['root_slot'];step=row['checkpoint'];slot=native['model_slots'][i]
        raw=np.concatenate((np.asarray(slot['warm_raw10'],np.float32),m.producer.raw15(native['plans'][i]['commands'])[:step]))[-10:]
        assert row['planned_raw10_bits']==raw.tobytes().hex()
        if step==0:assert row['predicted192'] is None and row['model_Q'] is None
        else:assert np.asarray(row['predicted192'],np.float32).tobytes()==futures[i][0,0,2+step//5].tobytes()
    assert m.contract()['max_calls']=={k:8 if k in ('encoder','projector') else 0 for k in m.bridge.KINDS}
    assert m.contract()['cuda_allocated_bytes']==3*1024**3 and m.contract()['cuda_reserved_bytes']==4*1024**3


@pytest.mark.parametrize('mutation',['root','order','raw','task','time','cp','state','normalizer','goal','future','oldschema'])
def test_fixed_rows_reject_cross_time_and_source_substitution(tmp_path,monkeypatch,mutation):
    p,_,n,u,f=disk_build(tmp_path,monkeypatch);c=m.read(m.producer.PLANNER_PREP)
    if mutation=='root':u['rows'][0]['root_slot']=1
    elif mutation=='order':u['rows'].reverse()
    elif mutation=='raw':u['rows'][0]['plan']['commands'][0][0]=.25
    elif mutation=='task':u['rows'][0]['task_id']='f'*64
    elif mutation=='time':u['rows'][0]['own_returned']=True
    elif mutation=='cp':u['rows'][0]['checkpoints'][1]['step']=15
    elif mutation=='state':u['rows'][0]['checkpoints'][1]['state'][0]+=.25
    elif mutation=='normalizer':c['slots'][0]['std'][0]=1.
    elif mutation=='goal':c['slots'][0]['packet']['goal'][0]=1.
    elif mutation=='future':f[0][0,0,0,0]+=.25
    else:u['schema']='old_physical.consume'
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(n,u,c,f,Path(p['arguments']['physical_output']))


@pytest.mark.parametrize('forbidden',['predict','AD','action_encoder'])
def test_forbidden_nn_api_no_retry(complete,forbidden):
    (p,*_),_=complete
    with pytest.raises(ValueError):encode_fixture(p,enc.old.Memory(),forbidden=forbidden)


@pytest.mark.parametrize('tokens',[197,256])
def test_original_hidden257_guard(complete,tokens):
    (p,*_),_=complete
    with pytest.raises(ValueError):encode_fixture(p,enc.old.Memory(),tokens=tokens)


def test_exact_seven_guard_not_rounded_eight():
    ledger=m.Ledger(7,lambda e:None);ledger.entries=[{}]*14;ledger.image=7
    with pytest.raises(ValueError):ledger.dispatch('encoder',None,lambda:pytest.fail('8th native call'))


def test_full_noise_array_and_metadata_integer_budget(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_noise'),m.Budget());rng=np.random.default_rng(8)
    c=m.shape_census(7)
    for i in range(8):
        for j,(shape,dtype) in enumerate(m.SHAPES):
            a=rng.integers(0,256,shape,dtype=np.uint8) if dtype=='uint8' else rng.standard_normal(shape,dtype=np.float32)
            ref=s.array('r%d_a%d'%(i,j),a)
            assert m.codec.load_array(ref,lambda n:(s.path/n).read_bytes()).tobytes()==a.tobytes()
    assert s.arrays==56 and c['arrays']==49 and c['native_module_IO_bytes']==7*sum(c['per_image_array_bytes'][2:6])
    assert m.shape_census()['total_upper_bytes']<m.CAP and c['total_upper_bytes']<m.CAP
    v=dict(padding='');v['padding']='x'*(m.METADATA-len(m.raw(v)));s.put('full_metadata.json',v,final=True)
    with pytest.raises(ValueError):s.put('extra.json',{})
    with pytest.raises(ValueError):s.array('extra',np.zeros(192,np.float32))


def test_native_success_not_replaced_by_velocity_reward():
    goal=np.asarray([300,300,350,350,0,0,0],float);state=goal.copy();state[5]=256.
    metrics=m.producer.native.metrics(goal,state)
    spec=dict(index=0,root_slot=0,checkpoint=10,absolute_step=20,task_id='x',predicted192=[0.]*192,goal192=[0.]*192,
        model_Q=0.,observed_anchor192=None,native_metrics=metrics,missing_reason=None)
    result=m.score(spec,dict(actual192=[1.]*192))
    assert result['native_success'] is True and metrics['native7_distance']==256. and metrics['native7_reward']==-256.
    assert result['actual_encoded_Q']==1. and result['scientific_pass'] is False


@pytest.mark.parametrize('kind',['event','counter','native_input','CLS','output'])
def test_native_validate_independent_seams(complete,kind):
    (p,s,r,_),_=complete;r=deepcopy(r);files={f.name:f.read_bytes() for f in s.path.iterdir()}
    events=[m.producer.native.decode(files['event_%04d.json'%i]) for i in range(s.index)]
    def load(ref):return m.codec.load_array(ref,lambda n:files[n])
    if kind=='event':events.insert(0,dict(kind='unknown'))
    elif kind=='counter':r['counts']['encoder']['completed']-=1
    else:
        en,pr=r['rows'][0]['modules'];ref=en['inputs'][0] if kind=='native_input' else pr['inputs'][0] if kind=='CLS' else pr['output']
        value=load(ref);value.flat[0]+=.25;oldload=load
        def load(e):return value if e==ref else oldload(e)
    with pytest.raises(ValueError):m.validate(p,r,events,load)


@pytest.mark.parametrize('kind',['orphan','source','receipt','prepare','future'])
def test_whole_temp_build_tamper_no_fullstrict(tmp_path,monkeypatch,kind):
    p,prepared,*_=disk_build(tmp_path,monkeypatch)
    if kind=='orphan':put(Path(m.ACTUAL['physical_output'])/'orphan.json',{})
    elif kind=='source':next(Path(m.ACTUAL['physical_output']).glob('*.png')).write_bytes(b'changed')
    elif kind=='receipt':Path(m.ACTUAL['physical_receipt']).write_bytes(m.raw({}))
    elif kind=='prepare':prepared.write_bytes(prepared.read_bytes()+b' ')
    else:
        with (m.producer.PLANNER_OUTPUT/'whole0.npy').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError):m.verify(prepared,'f'*64 if kind=='prepare' else m.sha(prepared))


def test_auth_before_any_reads(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('source reads'))
    with pytest.raises(ValueError):m.run('p','s','out','bad')

