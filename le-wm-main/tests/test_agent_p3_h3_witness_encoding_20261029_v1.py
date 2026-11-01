"""RAM-only hook/guard/array callbacks and temporary saved files. No torch/physics."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_h3_witness_encoding_20261029_v1 as m
import test_agent_p3_h3_witness_physical_20261029_v1 as physical_fixture
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc


@lru_cache(maxsize=1)
def base_fixture():
    p,data,_,_=physical_fixture.fixture();plans=p['plans'];rows=[]
    modelprep=dict(schema=m.planner.NAME+'.prepare',contract=m.planner.contract(),tasks=[],layers=[],
        expected_model_hash=plans[0]['source_witness']['task']['binding']['model_sha256'])
    pix=np.full((224,224,3),64,np.uint8);pre=m.bridge.pixels.preprocess_pixels(pix)
    z=np.full(192,pre.mean(),np.float32)+np.float32(.125)
    for plan in plans:
        w=plan['source_witness'];task=w['task'];slot=plan['root_slot'];mode=w['mode']
        task['binding']['model_sha256']=modelprep['expected_model_hash']
        raw10=np.concatenate((m.producer.moving.warmup_commands(),np.asarray(plan['commands'][:5],np.float32)))[-10:]
        ctx=deepcopy(w['request']['task']['root_context']);ctx['latent_history'][-1]=z.tolist()
        ctx['action_history']=m.planner.moving.normalized(task,raw10)
        ob=w['request']['observation'];ob['observed_latent_sha256']=m.planner.old.packet_digest(z.tolist())
        w['request']=m.planner.make_request(task,ctx,slot=slot,arm=mode,boundary=5,raw10=raw10.tolist(),observation=ob)
        w['result']['request_sha256']=m.planner.digest(w['request'])
        path=[];c=deepcopy(ctx)
        for j in range(3):
            chunk=plan['suffix_commands'][5*j:5*j+5]
            token=((np.asarray(chunk,np.float32)-np.asarray(task['action_mean'],np.float32))/np.asarray(task['action_std'],np.float32)).reshape(10)
            c=dict(c,latent_history=deepcopy(c['latent_history'][1:])+[np.full(192,(j+1)*.125,np.float32).tolist()],
                action_history=[deepcopy(c['action_history'][-1]),token.tolist()])
            path.append(deepcopy(c))
        winner=w['winner'];winner['whole']=path;winner['segmented']=deepcopy(path)
        winner['cost']=m.bridge.mse(path[-1]['latent_history'][-1],task['goal'])
        panel=w['result']['panels'][1];panel['winner']=deepcopy(winner)
        panel['candidates'][winner['index']]=deepcopy(winner);panel['selected_path']=deepcopy(path)
        w['witness_sha256']=m.producer.digest({k:v for k,v in w.items() if k!='witness_sha256'})
        plan['selection_digest']=m.producer.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        if len(modelprep['tasks'])<=slot:
            modelprep['tasks'].append(deepcopy(task));modelprep['layers'].append(dict(
                commands=w['prefix']['commands'],selection_sha256=w['prefix']['source_selection_sha256']))
        row=dict(index=plan['index'],root_slot=slot,condition='off',arm=plan['arm'],task_id=plan['task_id'],
            plan=deepcopy(plan),selection_digest=plan['selection_digest'],status='horizon_completed',own_returned=20,
            checkpoints=[],trajectory=[],responses=[],missing_checkpoints=[])
        for step in m.CP:
            if step==5:cp=deepcopy(w['actual_cp5']);cp.pop('phase',None)
            else:
                state=deepcopy(plan['native_goal']);state[0]+=step
                cp=dict(step=step,png=f'p{plan["index"]}_s{step}.png',file_sha256=physical_fixture.h(('PNG',slot,step)),
                    pixel_sha256=physical_fixture.h(('pixels',slot,step)),state=state,snapshot=dict(state=state,RAM=True))
            cp.setdefault('png',f'p{plan["index"]}_s{step}.png')
            cp={k:deepcopy(cp[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
            row['checkpoints'].append(cp);row['trajectory'].append(dict(cp,phase='plan',condition='off',overlay_index=0))
            metric=m.producer.native.metrics(plan['native_goal'],cp['state'])
            row['responses'].append(dict(step=step,state=cp['state'],native_metrics=metric['native'],
                reward=metric['native7_reward'],native_success=metric['native_success'],terminated=metric['native_success'],truncated=False))
        rows.append(row)
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,fixed_denominator=4,known_roots=2,
        errors=[],plans=deepcopy(plans),rows=rows,P1_admission=False,P3_admission=False,C3=False,physical_release=False)
    return p,strict,modelprep


def fixture(stops=()):
    p,s,mp=deepcopy(base_fixture())
    for i,stop in stops:
        row=s['rows'][i];row.update(own_returned=stop,status='native_stopped')
        row['checkpoints']=[x for x in row['checkpoints'] if x['step']<=stop]
        row['trajectory']=[x for x in row['trajectory'] if x['step']<=stop]
        row['responses']=[x for x in row['responses'] if x['step']<=stop]
        row['missing_checkpoints']=[x for x in m.CP if x>stop]
    return p,s,mp


def prepared(stops=()):
    pp,strict,mp=fixture(stops);rows=m.fixed_rows(pp,strict,mp,'/RAM/physical')
    n=sum(x['source'] is not None for x in rows)
    return dict(schema=m.NAME+'.prepare',rows=rows,plans=pp['plans'],contract=m.contract(),
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def execute(monkeypatch,stops=(),tokens=257,forbidden=None):
    p=prepared(stops);rows=deepcopy(p['rows']);s=enc.old.Memory();model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(4,lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    original=model.predict
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(enc.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(enc.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,s,encode)
    assert model.predict==original and not errors and not guard.blocked
    r=dict(schema=m.NAME+'.report',rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,
        modules=trace.entries,guards=guard.entries,counts=m.bridge.counts(trace.entries))
    return p,r,s


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_all16_native_hidden_CLS_projector_and_dual_ledger(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    result=m.validate(p,r,s.events,s.load)
    assert len(result)==16 and len(r['modules'])==len(r['guards'])==32
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=16,completed=16)
    for i,row in enumerate(result):
        assert row['checkpoint']==m.CP[i%4]
        if i%4==0:
            assert row['model_Q'] is row['prediction_vs_encoded_MSE'] is row['model_minus_actual_Q'] is None
            assert row['observed_anchor_reencode_MSE']==0
        else:
            assert row['observed_anchor_reencode_MSE'] is None and row['prediction_vs_encoded_MSE'] is not None
            assert row['absolute_Q_error']==abs(row['model_Q']-row['actual_encoded_Q'])
    chains=m.context_chains(p,result)
    assert len(chains)==4 and all(not c['context_missing'] for x in chains for c in x['checkpoints'])
    assert all(not row['C3'] and not row['physical_benefit_claim'] for row in result)


@pytest.mark.parametrize('stops',(((0,14),),((0,4),(3,19)),tuple((i,0) for i in range(4))))
def test_missing_keeps_full16_with_unknown_errors_and_no_padding(monkeypatch,stops):
    p,r,s=execute(monkeypatch,stops);out=m.validate(p,r,s.events,s.load);chains=m.context_chains(p,out)
    assert len(out)==16 and len(chains)==4
    assert sum(x['actual_encoded_Q'] is not None for x in out)==p['expected_calls']['encoder']
    for row in out:
        if row['latent_missing']:
            assert row['actual192'] is row['actual_encoded_Q'] is row['prediction_vs_encoded_MSE'] is row['native_metrics'] is None
    for i,stop in stops:
        assert [c['context_missing'] for c in chains[i]['checkpoints']]==[cp>stop for cp in m.CP]


@pytest.mark.parametrize('bad',('predict','AD','action_encoder'))
def test_no_new_predict_action_or_AD(monkeypatch,bad):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=bad)


def test_hidden197_not_shape_relaxed(monkeypatch):
    with pytest.raises(ValueError):execute(monkeypatch,tokens=197)


@pytest.mark.parametrize('fault',('drop','order','model','raw','task','normalizer','Q','path','cp5','cp10state','response','missing','stop','cost'))
def test_pure_own_physical_and_original_H3_join_rejects_mutations(fault):
    p,s,mp=fixture();row=s['rows'][0];plan=p['plans'][0]
    if fault=='drop':s['rows'].pop()
    elif fault=='order':s['rows'].reverse()
    elif fault=='model':mp['expected_model_hash']='a'*64
    elif fault=='raw':plan['commands'][19][0]=.75
    elif fault=='task':row['task_id']='a'*64
    elif fault=='normalizer':mp['tasks'][0]['action_std'][0]*=2
    elif fault=='Q':mp['tasks'][0]['projection'][0][0]=2
    elif fault=='path':plan['source_witness']['winner']['whole'][0]['action_history'][1][0]=.75
    elif fault=='cp5':row['checkpoints'][0]['pixel_sha256']='a'*64
    elif fault=='cp10state':row['checkpoints'][1]['state'][0]+=1
    elif fault=='response':row['responses'][1]['reward']+=1
    elif fault=='missing':row['missing_checkpoints']=[10]
    elif fault=='stop':row['own_returned']=14
    else:plan['source_witness']['winner']['cost']+=.1
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,mp,'/RAM/output')


@pytest.mark.parametrize('fault',('event','drop_attempt','order','shape','preprocessed','pixel','latent','spec','guard','missing','count','blocked','cp5latent'))
def test_full_saved_tensors_chronology_mutations_fail(replay,monkeypatch,fault):
    p,original,store=replay;r=deepcopy(original);events=deepcopy(store.events);s=enc.old.Memory();s.files=dict(store.files)
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels);row=r['rows'][0]
    if fault=='event':events.insert(0,dict(kind='BAD'))
    elif fault=='drop_attempt':events.remove(next(e for e in events if e['kind']=='module_attempt'))
    elif fault=='order':events[2],events[3]=events[3],events[2]
    elif fault=='shape':row['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='preprocessed':row['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixel':row['pixels']=s.array('badpix',np.zeros((224,224,3),np.uint8))
    elif fault=='latent':row['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='spec':row['model_Q']=0.
    elif fault=='guard':r['guards'].pop()
    elif fault=='missing':row['source']=None
    elif fault=='count':r['counts']['encoder']['completed']-=1
    elif fault=='blocked':r['blocked']=['predict']
    else:row['observed_anchor192'][0]+=.125
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


def test_auth_before_ANY_IO(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no unauthorized IO'))
    with pytest.raises(ValueError):m.run('/absent','a'*64,'/absent','wrong')


def test_external_source_hash_before_actual_strict(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda _: 'a'*64)
    monkeypatch.setattr(m.producer,'consume',lambda *a:pytest.fail('no strict before code seal'))
    with pytest.raises(ValueError,match='source before'):m.build('/RAM/phys','a'*64,'/RAM/receipt','b'*64,m.PRODUCER_SHA)


def test_real_store_arrays_events_census_and_exclusive_write(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_codec'),m.bridge.Budget())
    s.put('binding.json',dict(x=1));s.event(dict(kind='first'));s.event(dict(kind='second'))
    a=np.arange(192,dtype=np.float32);ref=s.array('native',a)
    assert m.read(s.path/'event_0000.json')==dict(kind='first')
    assert m.read(s.path/'event_0001.json')==dict(kind='second')
    assert np.array_equal(m.bridge.rgb.load_array(ref,lambda n:(s.path/n).read_bytes()),a)
    assert s.index==2 and len(s.hashes)==4
    with pytest.raises(FileExistsError):s.put('binding.json',dict(x=2))


def test_metadata_flush_fsync_and_exclusive(tmp_path):
    f=tmp_path/(m.NAME+'_fixture.json');m.write(f,dict(RAM=True))
    assert m.read(f)==dict(RAM=True)
    with pytest.raises(ValueError):m.write(f,dict(RAM=True))


def saved_fixture(tmp_path,monkeypatch,replay):
    """Own-schema report from actual synthetic process/Trace/guard and real Store."""
    p0,r0,ram=replay;p=deepcopy(p0);r=deepcopy(r0)
    code=str(Path(m.__file__).resolve());code_sha=m.sha(code);p.update(bindings={code:code_sha},expected_model_hash='a'*64,
        inherited_physical_counts={'RAM':True},inherited_model_resources={'RAM':True},inherited_model_wall_seconds=0,
        physical_report_sha256='b'*64,physical_receipt_sha256='c'*64)
    out=m.Store(tmp_path/(m.NAME+'_saved'),m.bridge.Budget())
    prepared='/RAM/fake_prepare.json';prep_sha='d'*64
    out.put('binding.json',dict(prepared=prepared,prepare_sha256=prep_sha));out.put('initial_roster.json',p['rows'])
    for name,body in ram.files.items():out.write(name,body)
    for e in ram.events:out.event(e)
    seal={k:r[k] for k in ('rows','modules','guards','blocked','errors')}
    h=out.put('native_seal.json',seal,final=True)
    inventory=[dict(type='RAMFixture',path=code,sha256=code_sha)]
    r.update(status='terminal_encoded_science_blocked',prepared=prepared,prepare_sha256=prep_sha,
        source_verified=True,model_unchanged=True,model_hash_before='a'*64,model_hash_after='a'*64,
        precision_before=m.planner.precision_contract(),precision_after=m.planner.precision_contract(),
        model_inventory_before=inventory,model_inventory_after=deepcopy(inventory),CUDA_peak_bytes=[0,0],
        P1_admission=False,P3_admission=False,C3=False,physical_release=False,elapsed_seconds=0,
        native_seal_sha256=h,artifact_sha256=deepcopy(out.hashes),event_count=out.index)
    out.put('report.json',r,final=True)
    monkeypatch.setattr(m,'verify',lambda *args:deepcopy(p))
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    return p,r,out


def test_actual_own_saved_consumer_rebuilds_every_array_context_and_census(tmp_path,monkeypatch,replay):
    p,r,s=saved_fixture(tmp_path,monkeypatch,replay)
    receipt=m.consume(s.path,m.sha(s.path/'report.json'))
    assert receipt['fixed_denominator']==16 and receipt['plan_denominator']==4 and len(receipt['actual_context_chains'])==4
    assert receipt['consume_new_neural_calls']==receipt['consume_new_physics']==0 and not receipt['C3']
    with (s.path/'unexpected.bin').open('xb') as f:f.write(b'unknown')
    with pytest.raises(ValueError,match='census'):m.consume(s.path,m.sha(s.path/'report.json'))


@pytest.mark.parametrize('fault',('source','precision','weights','memory','event','blob'))
def test_own_saved_consumer_rejects_coherent_metadata_or_file_mutations(tmp_path,monkeypatch,replay,fault):
    p,r,s=saved_fixture(tmp_path,monkeypatch,replay)
    if fault=='source':r['model_inventory_before'][0]['sha256']='e'*64;r['model_inventory_after']=deepcopy(r['model_inventory_before'])
    elif fault=='precision':r['precision_before']['autocast']=True;r['precision_after']=deepcopy(r['precision_before'])
    elif fault=='weights':r['model_hash_after']='f'*64
    elif fault=='memory':r['CUDA_peak_bytes'][0]=m.contract()['cuda_allocated_bytes']+1
    elif fault=='event':
        event='event_0000.json';(s.path/event).write_bytes(m.raw(dict(kind='unknown')))
        r['artifact_sha256'][event]=m.sha(s.path/event)
    else:
        name=next(n for n in s.hashes if n.endswith('.npy.gz'));(s.path/name).write_bytes(b'corrupt')
        r['artifact_sha256'][name]=m.sha(s.path/name)
    (s.path/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,KeyError)):m.consume(s.path,m.sha(s.path/'report.json'))
