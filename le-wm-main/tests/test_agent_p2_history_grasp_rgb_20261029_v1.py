"""RAM native tensor/hooks, original GRASP result ABI; no NN or physics."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_history_grasp_rgb_20261029_v1 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc
import test_agent_p2_history_grasp_physical_20261029_v2 as phys


def fixture(stops=(15,15)):
    p,_,modelprep,_=phys.fixture();actual=[];results=[]
    for i,plan in enumerate(p['plans']):
        mp=plan['planner_plan'];c=deepcopy(mp['original_context']);whole=[]
        for j in range(3):
            out=deepcopy(c)
            out['latent_history']=c['latent_history'][1:]+[np.full(192,(j+1)*.125,np.float32).tolist()]
            tokens=m.normalization.normalized_history(plan['commands'][j*5:j*5+5])
            out['action_history']=np.concatenate((np.asarray(c['action_history']),tokens))[-2:].tolist()
            whole.append(out);c=out
        mp['terminal']=deepcopy(whole[-1]);mp['model_terminal_Q']=m.bridge.mse(c['latent_history'][-1],mp['goal'])
        mp['selection_digest']=m.producer.planner_digest({k:v for k,v in mp.items() if k!='selection_digest'})
        plan['selection_digest']=m.producer.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        results.append(dict(root_slot=i,task_id=mp['task_id'],commands=deepcopy(mp['commands']),logits=deepcopy(mp['logits']),
            whole=deepcopy(whole),terminal_Q=mp['model_terminal_Q']))
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
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,fixed_denominator=2,known_roots=2,fresh_roots=0,
        errors=[],P1_admission=False,P2_admission=False,physical_release=False,rows=actual)
    return p,strict,modelprep,results


def prepared(stops=(15,15)):
    rows=m.fixed_rows(*fixture(stops),'/RAM/physical');n=sum(x['source'] is not None for x in rows)
    return dict(rows=rows,contract=m.contract(),expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def execute(monkeypatch,stops=(15,15),tokens=257,forbidden=None,fail_at=None,store=None):
    p=prepared(stops);rows=deepcopy(p['rows']);store=store if store is not None else enc.old.Memory()
    model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(3,lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
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
        with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,store,encode)
    except ValueError:
        assert model.predict==original
        if fail_at is not None:return p,rows,store
        raise
    assert model.predict==original and not errors and not guard.blocked
    r=dict(rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,modules=trace.entries,guards=guard.entries,
        counts=m.bridge.counts(trace.entries))
    return p,r,store


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_six_original_whole_slots_six_calls_no_arm_pairs(replay,monkeypatch):
    p,r,store=replay;monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    rows=m.validate(p,r,store.events,store.load)
    assert len(rows)==6 and len(r['modules'])==12
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=6,completed=6)
    assert all(r['counts'][k]==dict(attempted=0,completed=0) for k in m.bridge.KINDS if k not in ('encoder','projector'))
    assert [x['checkpoint'] for x in rows]==[5,10,15]*2 and all(not x['latent_missing'] for x in rows)
    assert all('edge' not in x and 'contexts' not in x for x in p['rows'])
    for row,spec in zip(rows,p['rows'],strict=True):
        z=store.load(r['rows'][row['index']]['latent'])
        assert row['real_Q']==m.bridge.mse(z,spec['goal192'])
        assert row['model_Q']==m.bridge.mse(spec['output_context']['latent_history'][-1],spec['goal192'])
        assert row['model_minus_real_Q']==row['model_Q']-row['real_Q']
        assert row['prediction_vs_encoded_MSE']==m.bridge.mse(z,spec['predicted192'])


@pytest.mark.parametrize('stops',[(4,9),(0,0),(15,14)])
def test_missing_keeps_all_six_slots(monkeypatch,stops):
    p,r,store=execute(monkeypatch,stops);rows=m.validate(p,r,store.events,store.load)
    assert len(rows)==6 and sum(not x['latent_missing'] for x in rows)==p['expected_calls']['encoder']
    assert all(x['source'] is None and x['real_Q'] is None for x in rows if x['latent_missing'])


def test_warmup_native_stop_no_root_or_endpoint_substitution():
    p,strict,mp,results=fixture((0,15));strict['rows'][0]['initial_exact']=False
    rows=m.fixed_rows(p,strict,mp,results,'/RAM/physical')
    assert len(rows)==6 and all(x['source'] is None for x in rows[:3])
    assert all(x['predicted192'] is not None for x in rows)


@pytest.mark.parametrize('bad',['predict','AD','action_encoder'])
def test_forbidden_calls(monkeypatch,bad):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=bad)


def test_original257_hidden_no_relaxation(monkeypatch):
    with pytest.raises(ValueError):execute(monkeypatch,tokens=197)


@pytest.mark.parametrize('fault',['schema','drop','order','task','model','raw','goal','warm','context','history','cpstep','state',
    'response','missing','early','terminalQ','root','resultRoot','resultTask','resultRaw','resultLogits','resultTerminal','resultCount'])
def test_original_root_results_and_native_binding_mutations(fault):
    p,strict,mp,results=fixture();actual=strict['rows'][0];plan=p['plans'][0]
    if fault=='schema':strict['schema']='old.consume'
    elif fault=='drop':strict['rows'].pop()
    elif fault=='order':strict['rows'].reverse()
    elif fault=='task':actual['task_id']='bad'
    elif fault=='model':mp['expected_model_hash']='bad'
    elif fault=='raw':plan['commands'][0][0]=.75
    elif fault=='goal':plan['planner_plan']['goal'][0]+=.1
    elif fault=='warm':mp['slots'][0]['warm_raw10'][0][0]=.2
    elif fault=='context':results[0]['whole'][0]['latent_history'][-1][0]+=.1
    elif fault=='history':results[0]['whole'][0]['action_history'][0][0]+=.1
    elif fault=='cpstep':actual['checkpoints'][0]['step']=10
    elif fault=='state':actual['checkpoints'][0]['state'][0]+=1
    elif fault=='response':actual['responses'][0]['reward']+=1
    elif fault=='missing':actual['missing_checkpoints']=[5]
    elif fault=='early':actual['own_returned']=4
    elif fault=='root':actual['initial_exact']=False
    elif fault=='resultRoot':results[0]['root_slot']=1
    elif fault=='resultTask':results[0]['task_id']='bad'
    elif fault=='resultRaw':results[0]['commands'][0][0]=.5
    elif fault=='resultLogits':results[0]['logits'][0]=.5
    elif fault=='resultTerminal':results[0]['whole'][-1]['latent_history'][-1][0]+=.1
    elif fault=='resultCount':results[0]['whole'].pop()
    else:plan['planner_plan']['model_terminal_Q']+=1
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,strict,mp,results,'/RAM/o')


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


def test_partial_no_retry_or_later_dispatch(monkeypatch):
    _,rows,store=execute(monkeypatch,fail_at=2)
    assert len(rows)==6 and rows[0]['status']==rows[1]['status']=='encoded'
    assert rows[2]['status']=='attempted' and rows[2]['latent'] is None
    assert all(x['status']=='not_started' for x in rows[3:])
    assert len([e for e in store.events if e['kind']=='image_attempt'])==3


def test_auth_before_reads(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no IO'))
    with pytest.raises(ValueError):m.run('/absent','a'*64,'/absent','bad')


def actual_arguments():
    return dict(physical_output=str(m.ROOT/'results/agent_p2_history_grasp_physical_20261029_v2_MAIN01'),
        physical_report_sha='d71588499aae674790ba95107650116b12aae07b5c17c718d432f003451534e6',
        physical_receipt=str(m.ROOT/'results/agent_p2_history_grasp_physical_20261029_v2_consume_MAIN01.json'),
        physical_receipt_sha='17373c3d9878d2bfab9eebfa6ac16c11e8ebf2736d8b77c4d206b5a45cf4957e',
        physical_source_sha='4eddd161df17f506916db3bcb7a4ca39c20b091cc0539ef4db6276b055267411')


def test_actual_saved_ABI_readonly_not_productionprepare():
    args=actual_arguments();rp=Path(args['physical_output'])/'report.json';cp=Path(args['physical_receipt'])
    if not rp.exists() or not cp.exists():pytest.skip('MAIN actual saved inputs unavailable')
    assert m.sha(rp)==args['physical_report_sha'] and m.sha(cp)==args['physical_receipt_sha']
    assert m.sha(m.producer.__file__)==args['physical_source_sha']
    r=m.read(rp);p=m.read(r['execution_binding']['prepared']);strict=m.read(cp)
    modeldir=Path(p['arguments']['planner_output']);mr=m.read(modeldir/'report.json');mp=m.read(mr['prepared'])
    results=[m.read(modeldir/f'root{i}.json') for i in (0,1)]
    rows=m.fixed_rows(p,strict,mp,results,rp.parent)
    assert len(rows)==6 and [x['checkpoint'] for x in rows]==[5,10,15]*2
    for x in rows:
        if x['source'] is not None:m.bridge.load_pixels(x['source'])


@pytest.mark.parametrize('stops',[(15,15),(15,14),(0,0)])
def test_real_store_array_event_reader_full_consume(tmp_path,monkeypatch,stops):
    store=m.Store(tmp_path/(m.NAME+'_RAM'),m.bridge.Budget(free=lambda:10**12))
    p,r,_=execute(monkeypatch,store=store,stops=stops)
    p.update(arguments=dict(RAM=True),bindings={},expected_model_hash='a'*64,physical_report_sha256='b'*64,
        physical_receipt_sha256='c'*64,inherited_physical_counts={},inherited_model_resources={},inherited_model_wall_seconds=1.)
    prep=tmp_path/(m.NAME+'_prepare_RAM.json');m.write(prep,p)
    store.put('binding.json',dict(prepared=str(prep),prepare_sha256=m.sha(prep)));store.put('initial_roster.json',p['rows'])
    seal={k:r[k] for k in ('rows','modules','guards','blocked','errors')};h=store.put('native_seal.json',seal,final=True)
    r.update(schema=m.NAME+'.report',status='grasp_RGB_encoded_NOT_science',prepared=str(prep),prepare_sha256=m.sha(prep),
        model_unchanged=True,source_verified=True,model_hash_before='a'*64,model_hash_after='a'*64,
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False,elapsed_seconds=1.,
        peak_allocated_bytes=1024,peak_reserved_bytes=2048,native_seal_sha256=h,event_count=store.index,
        artifact_sha256=deepcopy(store.hashes))
    store.put('report.json',r,final=True);monkeypatch.setattr(m,'verify',lambda *a:p)
    receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    missing=[i for i,spec in enumerate(p['rows']) if spec['source'] is None]
    assert len(receipt['rows'])==6 and receipt['missing_slots']==missing and receipt['real_comparisons_available']==6-len(missing)
    assert receipt['counts']['encoder']['completed']==6-len(missing)
    # Extra artifact saved through real writer, not production source edits.
    store.put('extra.json',{} ,final=True)
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))


def test_own_metadata_no_overwrite(tmp_path):
    path=tmp_path/(m.NAME+'_RAM.json');m.write(path,dict(RAM=True))
    with pytest.raises(ValueError):m.write(path,{})
    assert m.META==128*1024**2 and m.contract()['max_calls']['predict_API']==0


@pytest.mark.parametrize('bad',['source','report','receiptSHA','receiptBody'])
def test_build_actual_authority_required_before_any_model_load(monkeypatch,bad):
    args=actual_arguments()
    def sh(path):
        if Path(path)==Path(m.producer.__file__):return '0'*64 if bad=='source' else args['physical_source_sha']
        if Path(path)==Path(args['physical_output'])/'report.json':return '0'*64 if bad=='report' else args['physical_report_sha']
        if Path(path)==Path(args['physical_receipt']):return '0'*64 if bad=='receiptSHA' else args['physical_receipt_sha']
        pytest.fail('unexpected further seal check')
    monkeypatch.setattr(m,'sha',sh)
    if bad=='receiptBody':
        monkeypatch.setattr(m.producer,'consume',lambda *a:dict(actual_rebuilt_receipt=True))
        monkeypatch.setattr(m,'read',lambda *a:dict(actual_rebuilt_receipt=False))
    else:monkeypatch.setattr(m.producer,'consume',lambda *a:pytest.fail('consume before external source/report/receipt seals'))
    monkeypatch.setattr(m.producer,'verify',lambda *a:pytest.fail('must reject before prepare reconstruction'))
    with pytest.raises(ValueError):m.build(**args)
