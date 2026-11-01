"""RAM-only hook/guard/array callbacks and temporary saved files. No torch/physics."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_two_program_full30_rgb_encode_20261029_v1 as m
import test_agent_p2_two_program_full30_physical_20261029_v1 as physical_fixture
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc


@lru_cache(maxsize=1)
def base_fixture():
    _,data,mp,u=physical_fixture.fixture()
    pix=np.full((224,224,3),64,np.uint8);pre=m.bridge.pixels.preprocess_pixels(pix)
    z=np.full(192,pre.mean(),np.float32)+np.float32(.125)
    for slot in mp['slots']:slot['root_context']['latent_history'][-1]=z.tolist()
    for w in u['plans']:
        slot=mp['slots'][w['root_slot']];w['root_context']=deepcopy(slot['root_context'])
        c=deepcopy(w['root_context']);w['contexts']=[]
        a=m.producer.raw30(w['commands'])
        for j in range(6):
            c=deepcopy(c)
            c['latent_history']=c['latent_history'][1:]+[np.full(192,(j+1)*.125,np.float32).tolist()]
            c['action_history']=c['action_history'][1:]+m.producer.normalization.normalized_history(a[5*j:5*j+5]).tolist()
            w['contexts'].append(c)
        w['predicted_Q']=m.K.objective(w['contexts'][-1],w['goal'])
        w['witness_sha256']=m.K.digest({k:v for k,v in w.items() if k!='witness_sha256'})
    p=m.producer.assemble(m.producer.PINS,mp,u,data);rows=[]
    for i,plan in enumerate(p['plans']):
        root=p['source_roots'][plan['root_slot']]
        row=dict(index=i,root_slot=plan['root_slot'],condition='off',arm=plan['arm'],
            task_id=plan['task_id'],plan=deepcopy(plan),selection_digest=plan['selection_digest'],
            status='horizon_completed',own_returned=30,warmup_returned=10,initial_exact=True,
            root_state=root['root_state'],root_sha256=root['root_sha256'],
            checkpoints=[],trajectory=[],responses=[],missing_checkpoints=[])
        for step in m.CP:
            state=deepcopy(root['root_state'] if step==0 else plan['native_goal'])
            if step:state[0]+=100+step
            cp=dict(step=step,png=f'b{i}_plan_{step:02}_0.png',
                file_sha256=physical_fixture.h(('file',i,step)),pixel_sha256=physical_fixture.h(('pix',i,step)),
                state=state,snapshot=dict(state=state,RAM=True))
            row['checkpoints'].append(cp);row['trajectory'].append(dict(cp,phase='plan',condition='off',overlay_index=0))
            if step:
                metric=m.producer.native.metrics(plan['native_goal'],state)
                row['responses'].append(dict(step=step,state=state,native_metrics=metric['native'],
                    reward=metric['native7_reward'],native_success=metric['native_success'],
                    terminated=metric['native_success'],truncated=False))
        rows.append(row)
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,fixed_denominator=4,known_roots=2,
        own_native_saved_reconstructed_here=True,errors=[],plans=deepcopy(p['plans']),rows=rows,
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False)
    return p,strict,mp


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
    return dict(schema=m.NAME+'.prepare',rows=rows,plans=pp['plans'],contract=m.contract(),array_budget=m.array_budget(),
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def execute(monkeypatch,stops=(),tokens=257,forbidden=None):
    p=prepared(stops);rows=deepcopy(p['rows']);s=enc.old.Memory();model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(7,lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
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
    assert len(result)==28 and len(r['modules'])==len(r['guards'])==56
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=28,completed=28)
    for i,row in enumerate(result):
        assert row['checkpoint']==m.CP[i%7]
        if i%7==0:
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
    assert len(out)==28 and len(chains)==4
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


@pytest.mark.parametrize('fault',('drop','order','model','raw','task','normalizer','path','cp0','cp10state','response','missing','stop','cost'))
def test_pure_H30_join_rejects_mutations(fault):
    p,s,mp=fixture();row=s['rows'][0];plan=p['plans'][0]
    if fault=='drop':s['rows'].pop()
    elif fault=='order':s['rows'].reverse()
    elif fault=='model':mp['expected_model_hash']='a'*64
    elif fault=='raw':plan['commands'][29][0]=.75
    elif fault=='task':row['task_id']='a'*64
    elif fault=='normalizer':mp['slots'][0]['programs'][0]['scope']['normalization']['std'][0]*=2
    elif fault=='path':plan['planner_plan']['contexts'][0]['action_history'][1][0]=.75
    elif fault=='cp0':row['checkpoints'][0]['state'][0]+=.125
    elif fault=='cp10state':row['checkpoints'][2]['state'][0]+=.125
    elif fault=='response':row['responses'][1]['reward']+=1
    elif fault=='missing':row['missing_checkpoints']=[10]
    elif fault=='stop':row['own_returned']=14
    else:plan['planner_plan']['predicted_Q']+=.1
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


def test_actual_pins_before_input_read(monkeypatch):
    a=deepcopy(m.ACTUAL);a['physical_report_sha']='a'*64
    monkeypatch.setattr(m,'sha',lambda _:pytest.fail('not actual authority'))
    with pytest.raises(ValueError,match='pinned'):m.build(**a)


def test_real_store_arrays_events_census_and_exclusive_write(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_codec'),m.Budget())
    s.put('binding.json',dict(x=1));s.event(dict(kind='first'));s.event(dict(kind='second'))
    a=np.arange(192,dtype=np.float32);ref=s.array('native',a)
    assert m.read(s.path/'event_0000.json')==dict(kind='first')
    assert m.read(s.path/'event_0001.json')==dict(kind='second')
    assert np.array_equal(m.load_array(ref,lambda n:(s.path/n).read_bytes()),a)
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
    out=m.Store(tmp_path/(m.NAME+'_saved'),m.Budget())
    prepared='/RAM/fake_prepare.json';prep_sha='d'*64
    out.put('binding.json',dict(prepared=prepared,prepare_sha256=prep_sha));out.put('initial_roster.json',p['rows'])
    for name,body in ram.files.items():out.write(name,body)
    for e in ram.events:out.event(e)
    seal={k:r[k] for k in ('rows','modules','guards','blocked','errors')}
    h=out.put('native_seal.json',seal,final=True)
    inventory=[dict(type='RAMFixture',path=code,sha256=code_sha)]
    r.update(status='terminal_encoded_science_blocked',prepared=prepared,prepare_sha256=prep_sha,
        source_verified=True,model_unchanged=True,model_hash_before='a'*64,model_hash_after='a'*64,
        precision_before=m.precision_contract(),precision_after=m.precision_contract(),
        model_inventory_before=inventory,model_inventory_after=deepcopy(inventory),CUDA_peak_bytes=[0,0],
        P1_admission=False,P2_admission=False,P2_mechanism=False,C3=False,physical_release=False,elapsed_seconds=0,
        native_seal_sha256=h,artifact_sha256=deepcopy(out.hashes),event_count=out.index)
    out.put('report.json',r,final=True)
    out.put('completion.json',dict(schema=m.NAME+'.completion',report_sha256=m.sha(out.path/'report.json'),elapsed_seconds=0),final=True)
    monkeypatch.setattr(m,'verify',lambda *args:deepcopy(p))
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    return p,r,out


def test_actual_own_saved_consumer_rebuilds_every_array_context_and_census(tmp_path,monkeypatch,replay):
    p,r,s=saved_fixture(tmp_path,monkeypatch,replay)
    receipt=m.consume(s.path,m.sha(s.path/'report.json'))
    assert receipt['fixed_denominator']==28 and receipt['plan_denominator']==4 and len(receipt['actual_context_chains'])==4
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
    (s.path/'completion.json').write_bytes(m.raw(dict(schema=m.NAME+'.completion',
        report_sha256=m.sha(s.path/'report.json'),elapsed_seconds=0)))
    with pytest.raises((ValueError,KeyError)):m.consume(s.path,m.sha(s.path/'report.json'))


def source_journal():
    p,strict,_=fixture();events=[dict(kind='run_started')]
    for i,(plan,row) in enumerate(zip(p['plans'],strict['rows'],strict=True)):
        events.append(dict(kind='native_history_guard',branch=i,guard=dict(
            root_context_sha256=m.K.digest(plan['planner_plan']['root_context']),task_id=plan['task_id'],
            warm_raw_float32_bits=m.producer.moving.warmup_commands().tobytes().hex(),
            history_controls=[-10,-5,0],history_absolute_steps=[0,5,10])))
        for phase,commands in (('warmup',m.producer.moving.warmup_commands()),('plan',m.producer.raw30(plan['commands']))):
            for step,cmd in enumerate(commands,1):
                if phase=='plan':events.append(dict(kind='operation_dispatch_intent',branch=i,
                    receipt=dict(resource='public_step',phase=phase,step=step)))
                events.append(dict(kind='returned_transition',branch=i,phase=phase,step=step,
                    actual_dtype=np.dtype(np.float32).str,requested_bits=cmd.tobytes().hex(),actual_bits=cmd.tobytes().hex()))
    events.append(dict(kind='run_finished'))
    report=dict(rows=strict['rows']);seal_source(events,report)
    return p,report,events


def seal_source(events,report):
    tail=None
    for i,e in enumerate(events):
        e.update(index=i,previous_sha256=tail);e.pop('event_sha256',None)
        e['event_sha256']=m.producer.source.digest(e);tail=e['event_sha256']
    report['journal_chain_tail']=tail


@pytest.mark.parametrize('fault',['none','unknown','raw','guardlate','droprecord','context','time'])
def test_all_raw_time_history_source_join(fault):
    p,r,events=source_journal()
    if fault=='unknown':events.insert(-1,dict(kind='BAD'))
    elif fault=='raw':next(e for e in events if e['kind']=='returned_transition')['actual_bits']='00'
    elif fault=='guardlate':
        guard=events.pop(1);events.insert(20,guard)
    elif fault=='droprecord':events.remove(next(e for e in events if e['kind']=='returned_transition'))
    elif fault=='context':events[1]['guard']['root_context_sha256']='a'*64
    elif fault=='time':next(e for e in events if e['kind']=='returned_transition')['step']=2
    seal_source(events,r)
    if fault=='none':m.join_actions(p,r,events)
    else:
        with pytest.raises(ValueError):m.join_actions(p,r,events)


def test_whole_inclusive_verify_timeout_no_output_model_or_dispatch(monkeypatch):
    clock=[0.];original=m.Budget
    monkeypatch.setattr(m,'Budget',lambda:original(clock=lambda:clock[0]))
    def verify(*args):clock[0]=m.WALL;return prepared()
    monkeypatch.setattr(m,'verify',verify)
    monkeypatch.setattr(m,'Store',lambda *args:pytest.fail('no output/model after verify exhausted budget'))
    with pytest.raises(ValueError,match='600s'):m.run('/RAM/p','a'*64,'/RAM/out',m.AUTH)


def test_partial_callback_retained_and_hooks_restored(monkeypatch):
    p=prepared();rows=deepcopy(p['rows']);store=enc.old.Memory();model=enc.Model();torch=enc.fake_torch()
    original=model.predict;trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(7,lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    errors=[];monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    def fail_after_encoder(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(enc.Tensor(pre),f'r{row["index"]}_preprocessed')
            model.encoder(enc.Tensor(pre.reshape(1,3,224,224)))
            raise ValueError('fixture transport failure before projector')
    with pytest.raises(ValueError,match='fixture transport'):
        with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,store,fail_after_encoder)
    assert model.predict==original and all(not getattr(model,k).pre and not getattr(model,k).post
        for k in ('encoder','projector','action_encoder','predictor','pred_proj'))
    assert m.bridge.counts(trace.entries)['encoder']==dict(attempted=1,completed=1)
    assert m.bridge.counts(trace.entries)['projector']==dict(attempted=0,completed=0)
    assert rows[0]['status']=='attempted' and rows[1]['status']=='not_started' and store.files


@pytest.mark.parametrize('fault',['tail','header','dtype','scalar','filehash','arrayhash'])
def test_real_array_codec_header_and_gzip_mutations(tmp_path,fault):
    import io,gzip
    s=m.Store(tmp_path/(m.NAME+'_arrays'),m.Budget());ref=s.array('x',np.arange(192,dtype=np.float32))
    data=(s.path/ref['file']).read_bytes()
    if fault=='tail':data+=b'x'
    elif fault in ('header','dtype','scalar'):
        v=np.zeros((197,),np.float32) if fault=='header' else (np.zeros(192,np.float64) if fault=='dtype' else np.asarray(1.,np.float32))
        stream=io.BytesIO();np.save(stream,v,allow_pickle=False);data=gzip.compress(stream.getvalue(),mtime=0)
    elif fault=='filehash':ref['sha256']='a'*64
    else:ref['array_sha256']='a'*64
    if fault!='filehash':ref['sha256']=hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):m.load_array(ref,lambda _:data)


def test_budget_group_writes_cannot_borrow_or_silently_expand(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_cap'),m.Budget())
    s.budget.groups['arrays']=m.ARRAY_CAP
    with pytest.raises(ValueError,match='group byte cap'):s.write('x.npy.gz',b'x')
    s.budget.groups['metadata']=m.EVENT_CAP
    with pytest.raises(ValueError,match='group byte cap'):s.put('x.json',{})
    s.budget.groups['final']=m.FINAL
    with pytest.raises(ValueError,match='group byte cap'):s.put('report.json',{},final=True)


def test_all28_complete_io_event_report_budget_actual_stress(tmp_path,monkeypatch,replay):
    # Full native synthetic callback chain, all arrays, all events and duplicated seal/report/roster.
    p,r,s=saved_fixture(tmp_path,monkeypatch,replay)
    normal=m.byte_census({f.name:f.read_bytes() for f in s.path.iterdir()})
    assert len([n for n in s.hashes if n.endswith('.npy.gz')])==196 and s.index==280
    assert m.array_budget()['payload']==43524096 and m.array_budget()['npy']==43549184
    assert m.array_budget()['gzip_upper_bound']==43568644<m.ARRAY_CAP
    # Separate budget-only entropy stress: no neural inference, no claim of numerical outputs.
    stress=m.Store(tmp_path/(m.NAME+'_entropy'),m.Budget());rng=np.random.default_rng(20261029)
    array_names={n for n in s.hashes if n.endswith('.npy.gz')}
    for name in sorted(array_names):
        ref=next(ref for row in r['rows'] for ref in
            [row['pixels'],row['preprocessed'],row['latent']]+[v for mod in row['modules'] for v in mod['inputs']+[mod['output']]]
            if ref['file']==name)
        a=rng.integers(0,256,size=ref['shape'],dtype=np.uint8) if ref['dtype']=='uint8' else rng.standard_normal(ref['shape'],dtype=np.float32)
        stress.array(name[:-7],a)
    final_names={'native_seal.json','report.json','completion.json'}
    for f in s.path.iterdir():
        if f.name not in array_names:stress.write(f.name,f.read_bytes(),final=f.name in final_names)
    actual=m.byte_census({f.name:f.read_bytes() for f in stress.path.iterdir()})
    assert actual['arrays']<m.ARRAY_CAP and actual['metadata']<m.EVENT_CAP and actual['final']<m.FINAL
    print('FULL28_BUDGET',dict(normal=normal,entropy_transport_only=actual,analytic=m.array_budget()),flush=True)

