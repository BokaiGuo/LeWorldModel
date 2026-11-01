"""Only RAM arithmetic, synthetic module callbacks, and temporary journal files."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_effective_schedule_20261029_v1 as m
import agent_p3_fullcommit_live_20261028_v1_tests as f


def request(arm=m.ARMS[0],boundary=5,pending=None):
    base,old=f.request()
    req=m.make_request(base,old['task']['root_context'],slot=0,arm=arm,boundary=boundary,
        raw10=old['raw10'],observation={**old['observation'],'boundary':boundary},commitment=pending)
    return base,req


def planned(arm=m.ARMS[0],boundary=5,pending=None,backend=f.RamBackend):
    base,req=request(arm,boundary,pending);store=f.Events()
    ledger=m.core.Ledger(m.caps(arm),sink=store.event)
    r=m.plan_node(req,base,lambda t,s:backend(t,s),store,ledger,permit_ram=True)
    return base,req,r,store


@pytest.fixture(scope='module')
def nodes():return {a:planned(a) for a in m.ARMS}


def test_full_factorial_and_accounting():
    assert len(m.roster())==8 and sum(len(r['nodes']) for r in m.roster())==24
    assert set(m.BANKS['endpoints']).isdisjoint(m.BANKS['midpoints'])
    assert all(len(v)==4 for v in m.BANKS.values())
    for a in m.ARMS:
        c=m.caps(a);assert c['predict_API']==51 and c['jacobian_API']==6
        assert c['VJP_directions']==c['grad_API']==c['raw_backward_callbacks']==1152
    x=m.caps('commit15_endpoints');y=m.caps('feedback5_endpoints')
    assert x['solve_calls']==1 and y['solve_calls']==3
    assert x['replay_calls']==9 and y['replay_calls']==27
    assert x['metric_calls']==4 and y['metric_calls']==12
    c=m.contract()
    assert c['budget_match']=='notmatched' and c['total_predict_if_full20']==408
    assert not any(c[k] for k in ('P1_admission','P3_admission','C3','freshness','equal_total_compute'))
    assert all(x['actual_control_returned'] is None for x in m.roster())


@pytest.mark.parametrize('arm',m.ARMS)
def test_real_node_bank_dispatch_and_saved_arrays(nodes,arm):
    base,req,r,store=nodes[arm]
    assert r['status']=='sealed' and len(r['candidates'])==4
    assert len(r['commands'])==(15 if arm.startswith('commit15') else 5)
    assert r['resource_receipt']['attempted']==r['resource_receipt']['completed']==m.costs(arm,5)
    proof=m.audit_node(req,r,store.events,base)
    assert proof['node_verified'] and not proof['whole_job_verified']
    assert r['winner']==min(r['candidates'],key=lambda x:(x['cost'],x['index']))


@pytest.mark.parametrize('bank',m.BANKS)
def test_nonnested_generated_raw_support(nodes,bank):
    _,_,r,_=nodes['commit15_'+bank]
    assert r['bank']['fractions']==list(m.BANKS[bank])
    assert r['bank']['counterfactual_raw_support_overlap']==[]
    d=np.asarray(r['bank']['solve']['params'],np.float64)
    for fraction,a in zip(m.BANKS[bank],r['bank']['actions'],strict=True):
        assert np.asarray(a,np.float32).tobytes()==(fraction*d/r['bank']['divisor']).astype(np.float32).reshape(-1,2).tobytes()


def test_zero_direction_is_not_resampled_or_false_nonnested_raw_claim(nodes):
    _,req,r,_=nodes[m.ARMS[0]];base=req['task']
    response=deepcopy(r['responses'][1]);response['jacobian']=np.zeros((192,30),np.float32).tolist()
    ledger=m.core.Ledger(m.caps(m.ARMS[0]));ledger.scope=dict(test_synthetic=True)
    bank=m.generate(base,response,'midpoints',ledger)
    assert len(bank['counterfactual_raw_support_overlap'])==16
    assert all(np.count_nonzero(a)==0 for a in bank['actions'])
    assert ledger.completed['solve_calls']==ledger.completed['array_SVD_calls']==1


@pytest.mark.parametrize('boundary',(10,15))
def test_commit15_observes_but_does_not_select(boundary):
    base,req,r,store=planned('commit15_midpoints',boundary,dict(start=5,end=20,seal_sha256='a'*64))
    assert r['status']=='observed_commitment' and r['decision'] is r['commands'] is None
    assert r['resource_receipt']['attempted']==m.live.costs()
    assert m.audit_node(req,r,store.events,base)['node_verified']


@pytest.mark.parametrize('arm',m.ARMS)
def test_complete_session_ack_and_exact_nonrefunded_costs(arm):
    base,old=f.request();session=m.Session(base,0,arm);all_actions=[];results=[]
    for b in (5,10,15):
        raw=np.asarray(old['raw10'],np.float32)
        context=deepcopy(old['task']['root_context'])
        context['latent_history'][-1][0]=float(np.float32(context['latent_history'][-1][0]+float(b)/100))
        ob={**old['observation'],'boundary':b,
            'observed_latent_sha256':m.live.old.packet_digest(context['latent_history'][-1])}
        store=f.Events()
        req,r,packet=session.plan(context,raw,ob,lambda t,s:f.RamBackend(t,s),store,permit_ram=True)
        assert req['task']['root_context']==context
        results.append(r);all_actions.extend(packet.tolist())
        with pytest.raises(ValueError):
            session.plan(context,raw,ob,lambda t,s:f.RamBackend(t,s),store,permit_ram=True)
        session.acknowledge(packet)
    assert session.next_boundary==20 and len(all_actions)==15
    assert m.remaining_caps(arm,results)==dict.fromkeys(m.core.AXES,0)
    assert [r['status'] for r in results]==(['sealed','observed_commitment','observed_commitment']
        if arm.startswith('commit15') else ['sealed']*3)
    if arm.startswith('commit15'):
        assert np.asarray(all_actions,np.float32).tobytes()==np.asarray(results[0]['commands'],np.float32).tobytes()
    with pytest.raises(ValueError):
        session.plan(context,raw,ob,lambda t,s:f.RamBackend(t,s),f.Events(),permit_ram=True)


def test_native_short_stop_never_pads_or_replans():
    base,old=f.request();session=m.Session(base,0,'commit15_endpoints');store=f.Events()
    req,r,p=session.plan(old['task']['root_context'],old['raw10'],old['observation'],
        lambda t,s:f.RamBackend(t,s),store,permit_ram=True)
    with pytest.raises(ValueError):session.acknowledge(p[:2])
    session.acknowledge(p[:2],terminated=True)
    assert session.next_boundary==7 and session.stopped and session.active is None
    assert m.remaining_caps(session.arm,session.results)['predict_API']==16
    with pytest.raises(ValueError):session.plan(old['task']['root_context'],old['raw10'],old['observation'],None,store)


@pytest.mark.parametrize('mutation',('mode','slot','goal','raw','observation','late_idle','pending_feedback'))
def test_request_mutations_fail_closed(mutation):
    base,req=request()
    if mutation=='mode':req['mode']='adaptive'
    elif mutation=='slot':req['source_slot']=True
    elif mutation=='goal':req['task']['goal'][0]+=.1
    elif mutation=='raw':req['raw10'][0][0]+=.1
    elif mutation=='observation':req['observation']['observed_latent_sha256']='0'*64
    elif mutation=='late_idle':req['boundary']=10
    else:
        base,req=request('feedback5_endpoints',10,dict(start=5,end=20,seal_sha256='a'*64))
    with pytest.raises(ValueError):m.validate_request(req,base)


def test_no_implicit_cpu_neural_or_fixture_fallback():
    base,req=request();store=f.Events();ledger=m.core.Ledger(m.caps(m.ARMS[0]),sink=store.event)
    with pytest.raises(ValueError):m.plan_node(req,base,lambda t,s:f.RamBackend(t,s),store,ledger)
    assert ledger.events==[] and store.events==[]
    script=f'import sys; import experiments.spectral_reachability.{m.NAME}; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',script],cwd=m.ROOT,check=True)


def test_failed_replay_keeps_denominator_and_debit():
    class Broken(f.RamBackend):
        def replay(self,*args):raise RuntimeError('RAM fault')
    base,req=request();store=f.Events();ledger=m.core.Ledger(m.caps(m.ARMS[0]),sink=store.event)
    with pytest.raises(RuntimeError):
        m.plan_node(req,base,lambda t,s:Broken(t,s),store,ledger,permit_ram=True)
    result=store.events[-1]['result'];rr=result['resource_receipt']
    assert result['status']=='failed' and len(result['candidates'])==4
    assert rr['attempted']['replay_calls']==1 and rr['completed']['replay_calls']==0
    assert m.remaining_caps(m.ARMS[0],[result])['replay_calls']==8


@pytest.mark.parametrize('damage',('candidate','winner','gradient','nominal','unknown','module','decision','cost'))
def test_saved_node_mutations_reject(nodes,damage):
    base,req,result,store=nodes[m.ARMS[0]];r=deepcopy(result);events=deepcopy(store.events)
    if damage=='candidate':r['candidates'].pop()
    elif damage=='winner':r['winner']['index']=99
    elif damage=='decision':r['decision']['horizon']=1
    elif damage=='cost':r['candidates'][0]['cost']+=1
    elif damage=='unknown':events.insert(1,dict(kind='unregistered'))
    else:
        key={'gradient':'raw_backward','nominal':'nominal','module':'io_return'}[damage]
        event=next(e for e in events if e['kind']==key)
        name=next(iter(event['arrays']));a=m.signal.unblob(event['arrays'][name]).copy()
        a.flat[0]+=.1;event['arrays'][name]=m.signal.blob(a)
    # Re-seal result transport, not native array arithmetic.
    events[-1]=dict(kind=m.NAME+'.node_final',result=r)
    with pytest.raises(ValueError):m.audit_node(req,r,events,base)


def test_prepare_fixture_and_binding_drift(tmp_path,monkeypatch):
    base,_=request();source=tmp_path/'frozen';source.write_bytes(b'original')
    accepted=dict(bindings={str(source):m.sha(source)},tasks=[base,base],layers=[],policy={},
        modelbinding={},expected_model_hash='a'*64,source_manifest='RAM',authority={})
    # Real files/hash/writer, with synthetic upstream authority only.
    r=tmp_path/'seen_report';r.write_bytes(b'{}')
    s=tmp_path/'seen_receipt';s.write_bytes(m.core.canonical(dict(complete=True)))
    monkeypatch.setattr(m.previous,'build',lambda:deepcopy(accepted))
    monkeypatch.setattr(m,'SEEN_REPORT',r);monkeypatch.setattr(m,'SEEN_REPORT_SHA',m.sha(r))
    monkeypatch.setattr(m,'SEEN_STRICT',s);monkeypatch.setattr(m,'SEEN_STRICT_SHA',m.sha(s))
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'own_files',lambda:[source])
    path=tmp_path/'results'/(m.NAME+'_RAM.json')
    receipt=m.prepare(path);assert m.verify(path,receipt['sha256'])['contract']==m.contract()
    with pytest.raises(ValueError):m.prepare(path)
    source.write_bytes(b'mutated')
    with pytest.raises(ValueError):m.verify(path,receipt['sha256'])


def test_outcome_free_function_interfaces():
    import inspect
    assert set(inspect.signature(m.generate).parameters)=={'task','response','bank','ledger'}
    assert not any(k in inspect.signature(m.plan_node).parameters for k in ('reward','success','state','outcome'))
    assert m.contract()['scheduler']=='fixed ex ante; no chi/gain/outcome-dependent branching'


@pytest.mark.parametrize('arm',('commit15_midpoints','feedback5_endpoints'))
def test_real_native_episode_wiring_with_synthetic_rgb_and_transitions(monkeypatch,arm):
    # Actual moving.setup/packet/roll/encode_boundary and this episode body;
    # only synthetic Env/visual/backend/journal buffers, never a physics worker.
    import test_p1_moving_live_feedback_physical_20261027_v1 as physical
    import agent_p3_native_program_20261028_v1_test as programs
    p,data=physical.fixture();data['visual_tasks']=data.pop('tasks')
    layer=next(x for x in p['layers'] if x['source_slot']==0 and x['arm']==m.bridge.old.ARMS[0])
    base=programs.task(0);gn=data['visual_tasks'][0]['gn_task']
    for k in ('root_context','goal','action_mean','action_std'):base[k]=deepcopy(gn[k])
    base['binding'].update(context_sha256=m.digest(base['root_context']),goal_sha256=m.digest(base['goal']),
        normalizer_sha256=m.digest([base['action_mean'],base['action_std']]))
    row=next(r for r in m.native_roster() if r['source_slot']==0 and r['mode']==arm)
    store=f.RAM('/RAM_EFFECTIVE');realsha=m.sha
    def hash_any(path):
        return store.hash(path) if Path(path).is_relative_to(store.path) else realsha(path)
    monkeypatch.setattr(m,'sha',hash_any)
    monkeypatch.setattr(m.bridge.old,'sha',hash_any)
    monkeypatch.setattr(m.live,'NativeJournal',f.FullRamJournal)
    monkeypatch.setattr(m.bridge.old.source,'sample_state',lambda *a:physical.REQUEST.copy())
    with physical.visual_mock(store) as (runtime,_):
        meter=m.bridge.Meter(store,runtime['check_limits']);meter.episode=row['episode_id']
        Env=m.bridge.old.metered_class(physical.ArrayEnv,meter)
        env=meter.call('constructor',lambda:Env())
        try:
            m.episode(env,data['references'][0],data,layer,base,row,m.bridge.old.Handler(None),runtime,
                meter,store,lambda t,s:f.RamBackend(t,s),permit_ram=True)
        finally:meter.region='closing';meter.call('close',lambda:env.close())
    assert row['status']=='completed20' and row['control_returned']==20 and row['warmup_returned']==10
    assert row['initial_exact'] and row['prefix_exact'] and len(row['schedule_nodes'])==3
    assert len(row['encodings'])==4
    results=[n['result'] for n in row['schedule_nodes']]
    assert m.remaining_caps(arm,results)==dict.fromkeys(m.core.AXES,0)
    for b,node in zip((5,10,15),row['schedule_nodes'],strict=True):
        req=node['request']
        raw=np.concatenate((data['references'][0]['raw_history'],np.asarray(row['executed_raw'][:b],np.float32)))[-10:]
        assert np.asarray(req['raw10'],np.float32).tobytes()==raw.tobytes()
    assert meter.counts['public_step']==dict(attempted=30,completed=30)


def test_real_inherited_writer_journal_roundtrip(tmp_path,monkeypatch):
    import json
    (tmp_path/'results').mkdir();monkeypatch.setattr(m.bridge,'ROOT',tmp_path)
    store=m.bridge.Store(tmp_path/'results'/(m.bridge.NAME+'_effective_probe'))
    assert not m.core.canonical({'x':1}).endswith(b'\n')
    assert m.bridge.old.raw({'x':1}).endswith(b'\n')
    store.put('own_contract.json',m.contract())
    assert m.signal.read(store.path/'own_contract.json')==m.contract()
    store.emit('probe_start',contract=m.contract());store.emit('probe_end',complete=False)
    records=[json.loads(x) for x in (store.path/'journal.jsonl').read_bytes().splitlines()]
    assert len(records)==2
    previous=None
    for i,r in enumerate(records):
        assert r['index']==i and r['previous_sha256']==previous
        assert r['event_sha256']==m.bridge.old.source.digest({k:v for k,v in r.items() if k!='event_sha256'})
        previous=r['event_sha256']
    journal=m.live.NativeJournal(store,'native',lambda:None)
    events=[dict(kind=m.NAME+'.probe',value=i) for i in range(2)]
    for event in events:journal.event(event)
    journal.close();raw=journal.path.read_bytes()
    assert len(raw.splitlines())==2 and not raw.endswith(b'\n\n')
    assert m.signal.journal(journal.path)==(events,journal.tail)
    journal.path.write_bytes(raw.replace(b'\n',b'\n\n',1))
    with pytest.raises((ValueError,json.JSONDecodeError)):m.signal.journal(journal.path)


def execute_ram(*,path='/RAM_EFFECTIVE_WHOLE',fault=None,at=24):
    from contextlib import ExitStack
    from unittest.mock import patch
    import test_p1_moving_live_feedback_physical_20261027_v1 as physical
    import agent_p3_native_program_20261028_v1_test as programs
    p,data=physical.fixture();p['contract']=m.contract();p['tasks']=[]
    p['layers']=[next(x for x in p['layers'] if x['source_slot']==s and x['arm']==m.old.ARMS[0]) for s in (0,1)]
    data['visual_tasks']=data.pop('tasks')
    for s in (0,1):
        t=programs.task(s);gn=data['visual_tasks'][s]['gn_task']
        for k in ('root_context','goal','action_mean','action_std'):t[k]=deepcopy(gn[k])
        t['goal']=[1.,.5]+[0.]*190
        t['binding'].update(context_sha256=m.digest(t['root_context']),goal_sha256=m.digest(t['goal']),
            normalizer_sha256=m.digest([t['action_mean'],t['action_std']]))
        p['tasks'].append(t)
    store=f.RAM(path);envs=[];realsha=m.sha
    def hash_any(path):return store.hash(path) if Path(path).is_relative_to(store.path) else realsha(path)
    class Env(physical.ArrayEnv):
        def __init__(self):super().__init__(fault=fault,at=at)
    def factory(meter):
        e=m.old.metered_class(Env,meter)();envs.append(e);return e
    with physical.visual_mock(store) as (runtime,_),ExitStack() as stack:
        for obj,k,v in ((m.live,'NativeJournal',f.FullRamJournal),(m,'sha',hash_any),
            (m.old,'sha',hash_any),(m.old.source,'sample_state',lambda *a:physical.REQUEST.copy())):
            stack.enter_context(patch.object(obj,k,v))
        report=m.execute(p,data,store,m.old.Handler(None),runtime,factory,lambda t,s:f.RamBackend(t,s),
            authorization=m.AUTH,final_check=lambda:dict(model_unchanged=True),permit_ram=True)
    store.test_p=p;store.test_data=data
    return report,store,envs


def saved_ram(tmp_path,**kwargs):
    import io
    from PIL import Image
    import agent_p3_same_feedback_physical20_20261028_v2_tests as rgb_tests
    output=tmp_path/'RAM_SAVED';output.mkdir()
    r,store,envs=execute_ram(path=str(output),**kwargs)
    assert r['status']=='known_fullcommit_recorded',[(x['status'],x['first_error']) for x in r['rows']]
    p=store.test_p;p['sources']=rgb_tests.image_sources()['sources']
    p['modelbinding']=m.moving.suffix.modelbinding();p['expected_model_hash']=p['modelbinding']['tensor_sha256']
    prepared=str(tmp_path/'PREPARED_RAM.json');h='a'*64
    hashes=dict(original_state_dict=p['expected_model_hash'],all_buffers='b'*64)
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    r.update(prepared=prepared,prepare_sha256=h,source_manifest_sha256=m.digest(p['sources']),
        model_before=hashes,model_after=hashes,precision_before=precision,precision_after=precision,
        CUDA_peak_bytes=[0,0],per_episode_neural=m.node_accounting(r['nodes']))
    r['final_check']=dict(model_unchanged=True,before=hashes,after=hashes,source_manifest_sha256=m.digest(p['sources']))
    store.files['execution_binding.json']=m.raw(dict(prepared=prepared,prepare_sha256=h))
    for name,payload in store.files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    stats=m.harden.image_contract(p)['statistics']
    for row in r['rows']:
        for enc in row['encodings']:
            ns=row['episode_id']+f"_b{enc['step']:02}"
            with Image.open(output/(row['episode_id']+f"_plan_{enc['step']:02}.png")) as image:rgb=np.asarray(image).copy()
            b=io.BytesIO();np.save(b,m.harden.preprocess(rgb,stats),allow_pickle=False)
            (output/(ns+'_encode/preprocessed_pixels.npy')).write_bytes(b.getvalue())
    (output/'report.json').write_bytes(m.raw(r))
    return output,r,p,prepared,h,store.test_data


def consume_fixture(bundle,monkeypatch):
    out,r,p,prepared,h,data=bundle
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'source_data',lambda *a:data)
    return m.consume(out,m.sha(out/'report.json'),prepared,h)


@pytest.fixture(scope='module')
def full_saved(tmp_path_factory):return saved_ram(tmp_path_factory.mktemp('effective8'))


def test_complete_eight_arm_whole_driver_and_saved_consumer(full_saved,monkeypatch):
    out,r,p,prepared,h,data=full_saved
    assert len(r['rows'])==8 and len(r['nodes'])==24 and r['completed20']==8
    result=consume_fixture(full_saved,monkeypatch)
    assert result['complete'] and result['fixed_denominator']==8 and result['node_denominator']==24
    assert len(result['audited'])==24 and result['counts']['encoder']==dict(attempted=32,completed=32)
    for row in r['rows']:
        own=[n for n in r['nodes'] if n['episode_id']==row['episode_id']]
        assert m.remaining_caps(row['mode'],own)==dict.fromkeys(m.core.AXES,0)
    assert all(a['attempted']==a['completed']==m.caps(a['mode']) for a in r['per_episode_neural'])
    assert result['P1_admission'] is result['P3_admission'] is False


@pytest.mark.parametrize('field',('precision_after','model_after','counts','nodes','commitments'))
def test_whole_report_coherent_reseal_rejects(full_saved,monkeypatch,field):
    out,r,p,prepared,h,data=full_saved;changed=deepcopy(r)
    if field=='precision_after':changed[field]['matmul_TF32']=True
    elif field=='model_after':changed[field]={**changed[field],'all_buffers':'f'*64}
    elif field=='counts':changed[field]['predictor']=dict(attempted=0,completed=0)
    elif field=='nodes':changed[field].pop()
    else:changed['rows'][0]['commitments'][0]['commands'][0][0]+=.1
    payload=(out/'report.json').read_bytes();(out/'report.json').write_bytes(m.raw(changed))
    try:
        with pytest.raises(ValueError):consume_fixture(full_saved,monkeypatch)
    finally:(out/'report.json').write_bytes(payload)


def test_native_stop_fourteen_complete_strict_no_padding(tmp_path,monkeypatch):
    bundle=saved_ram(tmp_path,fault='truncate',at=24)
    r=bundle[1]
    assert r['completed20']==0 and all(row['control_returned']==14 for row in r['rows'])
    assert all(n['status']=='not_reached' for n in r['nodes'] if n['boundary']==15)
    result=consume_fixture(bundle,monkeypatch)
    assert result['complete'] and len(result['audited'])==16
    assert result['counts']['encoder']==dict(attempted=16,completed=16)


def test_whole_driver_authorization_before_env_or_model():
    with pytest.raises(ValueError):m.run(None,None,None,authorization='not-authorized')
    with pytest.raises(ValueError):m.execute(dict(contract=m.contract()),None,None,None,None,None,None,
        authorization='not-authorized',final_check=lambda:None)


def test_new_store_meter_full8_contract_and_readers(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    store=m.Store(tmp_path/'results'/(m.NAME+'_store_probe'))
    meter=m.Meter(store,lambda:None)
    assert meter.caps==m.contract()['physical_caps']|m.contract()['visual_caps']
    assert meter.caps['constructor']==8
    store.put('probe.json',m.contract());assert m.signal.read(store.path/'probe.json')==m.contract()
    store.emit('run_started',contract=m.contract())
    store.emit('run_finished',status='partial_or_failed',counts=meter.counts,nodes_sha256=m.digest(m.node_roster()))
    r=dict(journal_sha256=m.sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    events=m.load_events(store.path,r);assert len(events)==2
