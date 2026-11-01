"""RAM hook/tensor and saved-schema tests only; no NN/physics production."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_wallcapped_rgb_20261029_v1 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc
import test_agent_p2_wallcapped_physical_20261029_v1 as phys


def fixture(stops=(15,15,15,14)):
    p,_,modelprep,_=phys.fixture();actual=[]
    for slot in modelprep['slots']:
        slot.update(task_file='/RAM/task.json',blind_path='/RAM/blind.npz')
    p['model_slots']=deepcopy(modelprep['slots'])
    for i,plan in enumerate(p['plans']):
        mp=plan['planner_plan'];c=deepcopy(mp['root_context']);mp['contexts']=[]
        for j in range(3):
            out=deepcopy(c)
            out['latent_history']=c['latent_history'][1:]+[np.full(192,(j+1)*.125,np.float32).tolist()]
            tokens=m.producer.normalization.normalized_history(plan['commands'][j*5:j*5+5])
            out['action_history']=np.concatenate((np.asarray(c['action_history']),tokens))[-2:].tolist()
            mp['contexts'].append(out);c=out
        mp['model_Q']=m.bridge.mse(c['latent_history'][-1],mp['goal'])
        mp['selection_digest']=m.producer.planner_digest({k:v for k,v in mp.items() if k!='selection_digest'})
        plan['selection_digest']=m.producer.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        stop=stops[i]
        row=dict(index=i,root_slot=plan['root_slot'],condition='off',arm=plan['arm'],task_id=plan['task_id'],
            plan=deepcopy(plan),selection_digest=plan['selection_digest'],initial_exact=True,
            status='horizon_completed' if stop==15 else 'native_stopped',own_returned=stop,
            checkpoints=[],trajectory=[],responses=[],missing_checkpoints=[])
        for step in (5,10,15):
            if step>stop:row['missing_checkpoints'].append(step);continue
            state=deepcopy(plan['native_goal']);state[0]+=step
            cp=dict(step=step,png=f'p{i}_s{step}.png',file_sha256=phys.h(('file',i,step)),
                pixel_sha256=phys.h(('pixels',i,step)),state=deepcopy(state),snapshot=dict(RAM=True))
            row['checkpoints'].append(cp);row['trajectory'].append(dict(deepcopy(cp),phase='plan',overlay_index=0,condition='off'))
            metric=m.producer.native.metrics(plan['native_goal'],state)
            row['responses'].append(dict(step=step,state=deepcopy(state),native_metrics=metric['native'],
                reward=metric['native7_reward'],native_success=metric['native_success'],terminated=metric['native_success'],truncated=False))
        actual.append(row)
    p['planner_authority']['plans_sha256']=m.producer.digest(p['plans'])
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,fixed_denominator=4,known_roots=2,fresh_roots=0,
        errors=[],P1_admission=False,P2_admission=False,physical_release=False,rows=actual)
    return p,strict,modelprep


def prepared(stops=(15,15,15,14)):
    rows=m.fixed_rows(*fixture(stops),'/RAM/physical');n=sum(x['source'] is not None for x in rows)
    return dict(rows=rows,contract=m.contract(),expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def execute(monkeypatch,stops=(15,15,15,14),tokens=257,forbidden=None,fail_at=None,store=None):
    p=prepared(stops);rows=deepcopy(p['rows']);s=store if store is not None else enc.old.Memory()
    model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(3,lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    original=model.predict
    def encode(pix,row,scope,ordinal):
        if fail_at==ordinal:raise ValueError('RAM injected encode failure')
        guard.image=ordinal
        with m.visual.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(enc.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(enc.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    try:
        with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,s,encode)
    except ValueError:
        assert model.predict==original
        if fail_at is not None:return p,rows,s
        raise
    assert model.predict==original and not errors and not guard.blocked
    r=dict(rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,modules=trace.entries,guards=guard.entries,
        counts=m.bridge.counts(trace.entries))
    return p,r,s


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_own_contexts_no_edges_exact12_slots_11_calls_and_6_pairs(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    rows=m.validate(p,r,s.events,s.load);pairs=m.paired(rows)
    assert len(rows)==12 and len(pairs)==6 and len(r['modules'])==22
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=11,completed=11)
    assert all(r['counts'][k]==dict(attempted=0,completed=0) for k in m.bridge.KINDS if k not in ('encoder','projector'))
    assert [x['checkpoint'] for x in rows]==[5,10,15]*4
    assert rows[11]['latent_missing'] and rows[11]['real_Q'] is rows[11]['native_metrics'] is None
    assert rows[11]['actual_returned']==14 and rows[11]['model_Q'] is not None
    assert sum(x['real_pair_missing'] for x in pairs)==1
    assert pairs[5]['checkpoint']==15 and pairs[5]['real_Q']['joint'] is pairs[5]['real_Q']['delta'] is None
    assert pairs[5]['model_Q']['delta'] is not None
    assert all('edge' not in x for x in p['rows'])
    assert len(set(x['model_Q'] for x in rows[:3]))==3


def test_exact_model_real_physical_pair_arithmetic(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    rows=m.validate(p,r,s.events,s.load);pairs=m.paired(rows)
    for pair in pairs:
        c,j=rows[pair['composition_index']],rows[pair['joint_index']]
        assert pair['model_Q']['delta']==c['model_Q']-j['model_Q']
        if not pair['real_pair_missing']:
            assert pair['real_Q']['delta']==c['real_Q']-j['real_Q']
            assert pair['physical_metrics']['delta']['joint_xy_error']==c['native_metrics']['joint_xy_error']-j['native_metrics']['joint_xy_error']
    for row,spec in zip(rows,p['rows'],strict=True):
        if not row['latent_missing']:
            assert row['real_Q']==m.bridge.mse(s.load(r['rows'][row['index']]['latent']),spec['goal192'])
            assert row['model_Q']==m.bridge.mse(spec['output_context']['latent_history'][-1],spec['goal192'])


@pytest.mark.parametrize('stops',[(4,9,14,14),(0,0,0,0)])
def test_more_missing_retains_full_denominators(monkeypatch,stops):
    p,r,s=execute(monkeypatch,stops);rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==12 and len(m.paired(rows))==6
    assert sum(not x['latent_missing'] for x in rows)==p['expected_calls']['encoder']
    assert all(x['source'] is None and x['real_Q'] is None for x in rows if x['latent_missing'])


def test_no_twelfth_encoding_even_if_all_checkpoints_claimed():
    with pytest.raises(ValueError):prepared((15,15,15,15))


@pytest.mark.parametrize('bad',['predict','AD','action_encoder'])
def test_zero_prediction_AD_action_encoder(monkeypatch,bad):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=bad)


def test_original257_hidden_no_shape_relaxation(monkeypatch):
    with pytest.raises(ValueError):execute(monkeypatch,tokens=197)


@pytest.mark.parametrize('fault',['schema','drop','order','task','model','raw','goal','warm','context','history','cpstep','state','response','missing','early','terminalQ','root'])
def test_actual_plan_context_native_binding_mutations(fault):
    p,s,mp=fixture();a=s['rows'][0];plan=p['plans'][0]
    if fault=='schema':s['schema']='old.consume'
    elif fault=='drop':s['rows'].pop()
    elif fault=='order':s['rows'].reverse()
    elif fault=='task':a['task_id']='bad'
    elif fault=='model':mp['expected_model_hash']='bad'
    elif fault=='raw':plan['commands'][0][0]=.75
    elif fault=='goal':plan['planner_plan']['goal'][0]+=.1
    elif fault=='warm':plan['planner_plan']['warm_raw10'][0][0]=.1
    elif fault=='context':plan['planner_plan']['contexts'][0]['latent_history'][-1][0]+=.1
    elif fault=='history':plan['planner_plan']['contexts'][0]['action_history'][0][0]+=.1
    elif fault=='cpstep':a['checkpoints'][0]['step']=10
    elif fault=='state':a['checkpoints'][0]['state'][0]+=1
    elif fault=='response':a['responses'][0]['reward']+=1
    elif fault=='missing':a['missing_checkpoints']=[5]
    elif fault=='early':a['own_returned']=4
    elif fault=='root':a['initial_exact']=False
    else:plan['planner_plan']['model_Q']+=1
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,mp,'/RAM/o')


@pytest.mark.parametrize('fault',['event','drop_attempt','order','shape','preprocessed','pixel','latent','spec','guard','missing','count','blocked'])
def test_full_saved_tensor_events_mutations(replay,monkeypatch,fault):
    p,original,store=replay;r=deepcopy(original);events=deepcopy(store.events);s=enc.old.Memory();s.files=dict(store.files)
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels);row=r['rows'][0]
    if fault=='event':events.insert(0,dict(kind='UNKNOWN'))
    elif fault=='drop_attempt':events.remove(next(e for e in events if e['kind']=='module_attempt'))
    elif fault=='order':events[2],events[3]=events[3],events[2]
    elif fault=='shape':row['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='preprocessed':row['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixel':row['pixels']=s.array('badpix',np.zeros((224,224,3),np.uint8))
    elif fault=='latent':row['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='spec':row['model_Q']+=1
    elif fault=='guard':r['guards'].pop()
    elif fault=='missing':row['source']=None
    elif fault=='count':r['counts']['encoder']['completed']-=1
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


def test_partial_no_retry_missing_kept_and_no_later_dispatch(monkeypatch):
    p,rows,s=execute(monkeypatch,fail_at=2)
    assert len(rows)==12 and rows[0]['status']==rows[1]['status']=='encoded'
    assert rows[2]['status']=='attempted' and rows[2]['latent'] is None
    assert all(x['status']=='not_started' for x in rows[3:11]) and rows[11]==p['rows'][11]
    assert len([e for e in s.events if e['kind']=='image_attempt'])==3


def test_auth_before_any_reads(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no IO'))
    with pytest.raises(ValueError):m.run('/absent','a'*64,'/absent','bad')


def actual_arguments():
    return dict(physical_output=str(m.ROOT/'results/agent_p2_wallcapped_physical_20261029_v1_MAIN01'),
        physical_report_sha='d0708d3105e2b9523d6e90f92fb94c06dfd18f48b4860f78f84c98934b08a691',
        physical_receipt=str(m.ROOT/'results/agent_p2_wallcapped_physical_20261029_v1_consume_MAIN01.json'),
        physical_receipt_sha='47ee2abeb1ee5805c47a204efc7afca07e5f98cfca8144591bfd59b995dc376a',
        physical_source_sha='217c27a2f9398ee4909d91afb6c6681599f6a4791ba80253eb3c4bcfb06bb0f4')


def test_actual_saved_own_schema_smoke_no_productionprepare_or_encoding():
    args=actual_arguments();rp=Path(args['physical_output'])/'report.json';cp=Path(args['physical_receipt'])
    if not rp.exists() or not cp.exists():pytest.skip('MAIN completed actual assets unavailable')
    assert m.sha(rp)==args['physical_report_sha'] and m.sha(cp)==args['physical_receipt_sha']
    assert m.sha(m.producer.__file__)==args['physical_source_sha']
    r=m.read(rp);p=m.read(r['execution_binding']['prepared']);strict=m.read(cp)
    mr=m.read(Path(p['arguments']['planner_output'])/'report.json');mp=m.read(mr['prepared'])
    rows=m.fixed_rows(p,strict,mp,rp.parent)
    assert len(rows)==12 and sum(x['source'] is not None for x in rows)==11 and rows[11]['source'] is None
    assert [strict['rows'][i]['own_returned'] for i in range(4)]==[15,15,15,14]
    for x in rows:
        if x['source'] is not None:m.bridge.load_pixels(x['source'])


def test_real_store_arrays_events_and_consumer_disk(tmp_path,monkeypatch):
    budget=m.bridge.Budget(free=lambda:10**12)
    store=m.Store(tmp_path/(m.NAME+'_RAM'),budget)
    p,r,_=execute(monkeypatch,store=store)
    p.update(arguments=dict(RAM=True),bindings={},expected_model_hash='a'*64,physical_report_sha256='b'*64,
        physical_receipt_sha256='c'*64,inherited_physical_counts={},inherited_model_accounts=[],inherited_model_wall_seconds=1.)
    prepared=tmp_path/(m.NAME+'_prepare_RAM.json');m.write(prepared,p)
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=m.sha(prepared)));store.put('initial_roster.json',p['rows'])
    seal={k:r[k] for k in ('rows','modules','guards','blocked','errors')};h=store.put('native_seal.json',seal,final=True)
    r.update(schema=m.NAME+'.report',status='wallcapped_RGB_encoded_NOT_science',prepared=str(prepared),prepare_sha256=m.sha(prepared),
        model_unchanged=True,source_verified=True,model_hash_before='a'*64,model_hash_after='a'*64,
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False,elapsed_seconds=1.,
        peak_allocated_bytes=1024,peak_reserved_bytes=2048,native_seal_sha256=h,event_count=store.index,
        artifact_sha256=deepcopy(store.hashes))
    store.put('report.json',r,final=True)
    monkeypatch.setattr(m,'verify',lambda *a:p)  # only own prep boundary, no real authority claim
    receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert len(receipt['pairs'])==6 and receipt['missing_slots']==[11] and receipt['real_pairs_available']==5
    assert receipt['counts']['encoder']['completed']==11
    (store.path/'extra.json').write_text('{}')
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))


def test_own_metadata_codec_and_no_overwrite(tmp_path):
    assert m.producer.digest([1,2])!=m.producer.planner_digest([1,2])
    path=tmp_path/(m.NAME+'_RAM.json');m.write(path,dict(RAM=True))
    with pytest.raises(ValueError):m.write(path,{})
    assert m.META==128*1024**2 and m.contract()['max_calls']['predict_API']==0
