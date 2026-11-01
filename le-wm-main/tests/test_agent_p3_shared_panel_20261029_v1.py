"""RAM arithmetic, synthetic env callbacks and saved fixtures ONLY. No neural/physics."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import io
import json
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_shared_panel_20261029_v1 as m
import agent_p3_fullcommit_live_20261028_v1_tests as f


def request(arm=m.ARMS[0],boundary=5,unsupported=False):
    base,old=f.request()
    if unsupported:
        base['goal']=[1.]*192;base['binding']['goal_sha256']=m.digest(base['goal'])
    req=m.make_request(base,old['task']['root_context'],slot=0,arm=arm,boundary=boundary,
        raw10=old['raw10'],observation={**old['observation'],'boundary':boundary})
    return base,req


def planned(arm=m.ARMS[0],boundary=5,unsupported=False,backend=f.RamBackend):
    base,req=request(arm,boundary,unsupported);store=f.Events()
    ledger=m.core.Ledger(m.caps(arm),sink=store.event)
    r=m.plan_node(req,base,lambda t,s:backend(t,s),store,ledger,permit_ram=True)
    return base,req,r,store


@pytest.fixture(scope='module')
def nodes():return {a:planned(a,boundary=15) for a in m.ARMS}


def test_denominators_and_every_budget_axis():
    assert len(m.roster())==8 and len(m.node_roster())==24
    c=m.contract();assert c['panel_slots']==48 and c['candidate_slots']==192
    for a in m.ARMS:
        x=m.caps(a)
        assert x['predict_API']==x['predict_samples']==132 and x['predict_context_tokens']==396
        assert x['jacobian_API']==6 and x['VJP_directions']==x['grad_API']==x['raw_backward_callbacks']==1152
        assert x['signal_compile_calls']==x['program_compile_calls']==x['solve_calls']==6
        assert x['array_SVD_calls']==12 and x['replay_calls']==54 and x['metric_calls']==24 and x['schedule_decisions']==3
        assert x['action_encoder_calls']==x['predictor_calls']==x['pred_proj_calls']==132
    assert c['total_predict_if_full20']==1056 and not c['natural_cheap_baseline']
    assert not any(c[k] for k in m.FLAGS)
    assert c['horizon_contract']['H3_forecast_ends']==[20,25,30]
    assert c['horizon_contract']['remaining_fit_gate'] is False


@pytest.mark.parametrize('arm',m.ARMS)
def test_b15_both_full_panels_and_only_first5(nodes,arm):
    base,req,r,store=nodes[arm]
    assert r['status']=='sealed' and r['selected_horizon']==(1 if arm.startswith('fixedH1') else 3)
    assert [len(p['winner']['actions']) for p in r['panels']]==[5,15]
    assert len(r['commands'])==5
    assert [len(p['selected_path']) for p in r['panels']]==[1,3]
    assert r['resource_receipt']['completed']==m.costs(arm,15)
    assert m.audit_node(req,r,store.events,base)['node_verified']
    assert not m.audit_node(req,r,store.events,base)['whole_job_verified']


def test_same_complete_paid_axes_different_horizon_routing(nodes):
    for bank in m.BANKS:
        a=nodes['fixedH1_'+bank][2];b=nodes['adaptive_'+bank][2]
        assert a['resource_receipt']['attempted']==b['resource_receipt']['attempted']
        # Same RAM arithmetic, distinct mode-bound local task/response SHA authorities.
        assert [p['candidates'] for p in a['panels']]==[p['candidates'] for p in b['panels']]
        assert a['panels'][0]['bank']['response_sha256']!=b['panels'][0]['bank']['response_sha256']
        assert a['policy']['decision']['program_depth_limit']==3
        assert a['policy']['applied_fields']==['horizon']
        assert a['policy']['actual_candidates_per_panel']==4
        assert a['policy']['original_width_and_depth_executed'] is False


def test_progress_policy_original_width8_is_not_relabelled(monkeypatch,nodes):
    base,req,r,_=nodes['adaptive_endpoints'];original=m.core.choose
    def width8(*args,**kwargs):return dict(original(*args,**kwargs),candidates=8)
    monkeypatch.setattr(m.core,'choose',width8)
    p=m.policy_decision(req['task'],r['responses'],r['diagnostic'],req['mode'])
    assert p['supported'] and p['decision']['candidates']==8
    assert p['actual_candidates_per_panel']==4 and not p['original_width_and_depth_executed']


@pytest.mark.parametrize('arm',('adaptive_endpoints','adaptive_midpoints'))
def test_original_reconfiguration_refusal_keeps_full_paid_panels(arm):
    base,req,r,sink=planned(arm,15,unsupported=True)
    assert r['policy']['decision']['program_type']=='reconfiguration'
    assert r['policy']['decision']['candidates']==8
    assert r['status']=='refused' and r['commands'] is r['selected_horizon'] is None
    assert len(r['panels'])==2 and all(p['status']=='returned' and len(p['candidates'])==4 for p in r['panels'])
    assert r['resource_receipt']['completed']==m.costs(arm,15)
    assert m.audit_node(req,r,sink.events,base)['node_verified']
    session=m.Session(base,0,arm)
    req2,r2,packet=session.plan(req['task']['root_context'],req['raw10'],{**req['observation'],'boundary':5},
        lambda t,s:f.RamBackend(t,s),f.Events(),permit_ram=True)
    assert r2['status']=='refused' and packet is None and session.stopped
    with pytest.raises(ValueError):session.acknowledge(np.zeros((5,2),np.float32))


@pytest.mark.parametrize('damage',('program_type','extension_type','continue_expanding'))
def test_unsupported_policy_never_maps_to_progress(monkeypatch,nodes,damage):
    _,req,r,_=nodes['adaptive_endpoints'];original=m.core.choose
    value=False if damage=='continue_expanding' else 'composition'
    monkeypatch.setattr(m.core,'choose',lambda *a,**k:dict(original(*a,**k),**{damage:value}))
    p=m.policy_decision(req['task'],r['responses'],r['diagnostic'],req['mode'])
    assert not p['supported'] and p['refusal']


@pytest.mark.parametrize('damage',('reward','state','success','future_rgb','commitment','boundary','goal','raw','latent','reserved'))
def test_model_request_whitelist(damage):
    base,req=request()
    if damage in ('reward','state','success','future_rgb','commitment'):req[damage]=0
    elif damage=='boundary':req['boundary']=20
    elif damage=='goal':req['task']['goal'][0]+=.1
    elif damage=='raw':req['raw10'][0][0]+=.125
    elif damage=='latent':req['observation']['observed_latent_sha256']='0'*64
    else:req['source_slot']=2
    with pytest.raises(ValueError):m.validate_request(req,base)


@pytest.mark.parametrize('arm',m.ARMS)
def test_actual_ack_and_full132_nonrefundable_budget(arm):
    base,first=request(arm);session=m.Session(base,0,arm)
    for b in (5,10,15):
        context=deepcopy(first['task']['root_context']);context['latent_history'][-1][0]=float(np.float32(b/100))
        ob={**first['observation'],'boundary':b,'observed_latent_sha256':m.old.packet_digest(context['latent_history'][-1])}
        req,r,packet=session.plan(context,first['raw10'],ob,lambda t,s:f.RamBackend(t,s),f.Events(),permit_ram=True)
        assert req['task']['root_context']==context and len(packet)==5
        with pytest.raises(ValueError):session.plan(context,first['raw10'],ob,None,f.Events(),permit_ram=True)
        with pytest.raises(ValueError):session.acknowledge(packet[:2])
        session.acknowledge(packet)
    assert session.next_boundary==20 and m.remaining_caps(arm,session.results)==dict.fromkeys(m.core.AXES,0)
    with pytest.raises(ValueError):session.plan(context,first['raw10'],ob,None,f.Events(),permit_ram=True)


def test_earlystop_no_padding_no_replan():
    base,req=request('adaptive_midpoints');session=m.Session(base,0,req['mode'])
    _,_,packet=session.plan(req['task']['root_context'],req['raw10'],req['observation'],lambda t,s:f.RamBackend(t,s),f.Events(),permit_ram=True)
    session.acknowledge(packet[:2],terminated=True)
    assert session.next_boundary==7 and session.stopped
    assert m.remaining_caps(req['mode'],session.results)['predict_API']==88
    with pytest.raises(ValueError):session.plan(req['task']['root_context'],req['raw10'],req['observation'],None,f.Events(),permit_ram=True)


def test_failed_program_retains_both_panel_candidate_slots_and_attempt():
    class Broken(f.RamBackend):
        def replay(self,*a):raise RuntimeError('synthetic replay failure')
    base,req=request();sink=f.Events();ledger=m.core.Ledger(m.caps(req['mode']),sink=sink.event)
    with pytest.raises(RuntimeError):m.plan_node(req,base,lambda t,s:Broken(t,s),sink,ledger,permit_ram=True)
    r=sink.events[-1]['result'];rr=r['resource_receipt']
    assert r['status']=='failed' and [len(p['candidates']) for p in r['panels']]==[4,4]
    assert rr['attempted']['replay_calls']==1 and rr['completed']['replay_calls']==0
    assert m.remaining_caps(req['mode'],[r])['replay_calls']==53


@pytest.mark.parametrize('damage',('missing_panel','missing_candidate','winner','cost','selected_path','commands15','gradient','nominal','unknown','module','decision','paid'))
def test_saved_node_mutations_reject(nodes,damage):
    base,req,result,store=nodes['adaptive_midpoints'];r=deepcopy(result);es=deepcopy(store.events)
    if damage=='missing_panel':r['panels'].pop()
    elif damage=='missing_candidate':r['panels'][0]['candidates'].pop()
    elif damage=='winner':r['panels'][0]['winner']['index']=99
    elif damage=='cost':r['panels'][0]['candidates'][0]['cost']+=.1
    elif damage=='selected_path':r['panels'][1]['selected_path'].pop()
    elif damage=='commands15':r['commands']=r['panels'][1]['winner']['actions']
    elif damage=='gradient':
        e=next(e for e in es if e['kind']=='raw_backward');a=m.signal.unblob(e['arrays']['value']).copy();a[0]+=.5;e['arrays']['value']=m.signal.blob(a)
    elif damage=='nominal':r['responses'][0]['nominal'][0]+=.25
    elif damage=='unknown':es.insert(1,dict(kind='silent_free_operation'))
    elif damage=='module':es.pop(next(i for i,e in enumerate(es) if e['kind']=='io_return'))
    elif damage=='decision':r['policy']['decision']['program_depth_limit']=99
    else:r['resource_receipt']['completed']['predict_API']-=1
    es[-1]=dict(kind=m.NAME+'.node_final',result=r)
    with pytest.raises(ValueError):m.audit_node(req,r,es,base)


def test_cold_import_cli_and_no_implicit_cpu():
    code=f'import sys; import experiments.spectral_reachability.{m.NAME}; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,check=True)
    p=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,capture_output=True,text=True,check=True)
    assert 'strict' in p.stdout and 'authorize-known-cuda-and-physics' in p.stdout
    base,req=request();sink=f.Events();ledger=m.core.Ledger(m.caps(req['mode']),sink=sink.event)
    with pytest.raises(ValueError):m.plan_node(req,base,lambda t,s:f.RamBackend(t,s),sink,ledger)
    assert not sink.events and not ledger.events
    with pytest.raises(ValueError):m.run('unused','a'*64,'unused',authorization=None)


def test_prepare_ram_source_and_envelope_drift(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();src=tmp_path/'source';src.write_bytes(b'RAM')
    base,_=request();accepted=dict(sources={str(src):m.sha(src)},tasks=[base,base],layers=[],policy={},modelbinding={},
        expected_model_hash='a'*64,source_manifest='RAM',authority={},schema='RAM.prepare')
    monkeypatch.setattr(m.frozen,'build',lambda:deepcopy(accepted));monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'own_files',lambda:[src]);monkeypatch.setattr(m,'PINS',{src:m.sha(src)})
    path=tmp_path/'results'/(m.NAME+'_RAM.json');receipt=m.prepare(path)
    assert m.verify(path,receipt['sha256'])['schema']==m.NAME+'.prepare'
    with pytest.raises(ValueError):m.prepare(path)
    with pytest.raises(ValueError):m.verify(path,'f'*64)
    src.write_bytes(b'changed')
    with pytest.raises(ValueError):m.verify(path,receipt['sha256'])


def execute_ram(path,monkeypatch,*,unsupported=False,fault=None,at=24):
    import test_p1_moving_live_feedback_physical_20261027_v1 as physical
    import agent_p3_native_program_20261028_v1_test as programs
    p,data=physical.fixture();p['contract']=m.contract();p['tasks']=[]
    p['layers']=[next(x for x in p['layers'] if x['source_slot']==s and x['arm']==m.old.ARMS[0]) for s in (0,1)]
    data['visual_tasks']=data.pop('tasks')
    for s in (0,1):
        t=programs.task(s);gn=data['visual_tasks'][s]['gn_task']
        for k in ('root_context','goal','action_mean','action_std'):t[k]=deepcopy(gn[k])
        t['goal']=([1.]*192 if unsupported else [1.,.5]+[0.]*190)
        t['binding'].update(context_sha256=m.digest(t['root_context']),goal_sha256=m.digest(t['goal']),
            normalizer_sha256=m.digest([t['action_mean'],t['action_std']]))
        p['tasks'].append(t)
    store=f.RAM(str(path));envs=[];realsha=m.sha
    def hash_any(path):return store.hash(path) if Path(path).is_relative_to(store.path) else realsha(path)
    class Env(physical.ArrayEnv):
        def __init__(self):super().__init__(fault=fault,at=at)
    def factory(meter):
        e=m.old.metered_class(Env,meter)();envs.append(e);return e
    with physical.visual_mock(store) as (runtime,_),ExitStack() as stack:
        # Synthetic RAM files, no production globals are patched by runner code.
        for obj,k,v in ((m.live,'NativeJournal',f.FullRamJournal),(m,'sha',hash_any),(m.old,'sha',hash_any),
            (m.old.source,'sample_state',lambda *a:physical.REQUEST.copy())):
            stack.enter_context(patch.object(obj,k,v))
        report=m.execute(p,data,store,m.old.Handler(None),runtime,factory,lambda t,s:f.RamBackend(t,s),
            authorization=m.AUTH,final_check=lambda:dict(model_unchanged=True),permit_ram=True)
    return report,store,envs,p,data


def saved_ram(path,monkeypatch,**kwargs):
    from PIL import Image
    import agent_p3_same_feedback_physical20_20261028_v2_tests as rgb_tests
    path.mkdir();r,store,envs,p,data=execute_ram(path,monkeypatch,**kwargs)
    assert r['status']=='shared_panel_recorded',[(x['status'],x['first_error']) for x in r['rows']]
    p['sources']=rgb_tests.image_sources()['sources'];p['modelbinding']=m.moving.suffix.modelbinding()
    p['expected_model_hash']=p['modelbinding']['tensor_sha256']
    prepared=str(path.parent/'PREPARED_RAM.json');h='a'*64
    hashes=dict(original_state_dict=p['expected_model_hash'],all_buffers='b'*64)
    r.update(prepared=prepared,prepare_sha256=h,source_manifest_sha256=m.digest(p['sources']),
        model_before=hashes,model_after=deepcopy(hashes),precision_before=m.precision_contract(),precision_after=m.precision_contract(),
        CUDA_peak_bytes=[0,0],wall_seconds=1.,per_episode_neural=m.node_accounting(r['nodes']),budget_match=m.contract()['budget_match'])
    r['final_check']=dict(model_unchanged=True,before=hashes,after=deepcopy(hashes),source_manifest_sha256=m.digest(p['sources']))
    store.files['execution_binding.json']=m.raw(dict(prepared=prepared,prepare_sha256=h))
    for name,payload in store.files.items():
        dest=path/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    stats=m.harden.image_contract(p)['statistics']
    for row in r['rows']:
        for enc in row['encodings']:
            ns=row['episode_id']+f"_b{enc['step']:02}"
            with Image.open(path/(row['episode_id']+f"_plan_{enc['step']:02}.png")) as image:rgb=np.asarray(image).copy()
            b=io.BytesIO();np.save(b,m.harden.preprocess(rgb,stats),allow_pickle=False)
            (path/(ns+'_encode/preprocessed_pixels.npy')).write_bytes(b.getvalue())
    (path/'report.json').write_bytes(m.raw(r));return path,r,p,prepared,h,data


def consume_fixture(bundle,monkeypatch):
    out,r,p,prepared,h,data=bundle
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m.bridge,'source_data',lambda *a:data)
    return m.consume(out,m.sha(out/'report.json'),prepared,h)


@pytest.fixture(scope='module')
def full_saved(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:return saved_ram(tmp_path_factory.mktemp('shared')/'saved',mp)


@pytest.fixture(scope='module')
def refused_saved(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:return saved_ram(tmp_path_factory.mktemp('refused')/'saved',mp,unsupported=True)


@pytest.fixture(scope='module')
def stopped_saved(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:return saved_ram(tmp_path_factory.mktemp('stopped')/'saved',mp,fault='truncate',at=18)


def test_ram_full8_dispatch_and_strict(full_saved,monkeypatch):
    out,r,p,prepared,h,data=full_saved
    result=consume_fixture(full_saved,monkeypatch)
    assert result['complete'] and result['all20'] and result['equal_paid_axes_full20']
    assert result['prepared']==str(Path(prepared).absolute()) and result['prepare_sha256']==h
    assert len(result['audited'])==24 and r['completed20']==8
    assert result['physical_census']['counts']['encoder']==dict(attempted=32,completed=32)
    assert sum(x['completed']['predict_API'] for x in r['per_episode_neural'])==1056
    assert all(m.remaining_caps(row['mode'],[n for n in r['nodes'] if n['episode_id']==row['episode_id']])==dict.fromkeys(m.core.AXES,0)
        for row in r['rows'])
    assert not result['natural_cheap_baseline'] and not result['C3'] and not result['P1_admission']
    assert all(x['scalar_backward_calls'] is x['AD_primal_predict_API'] is x['Jacobian_elements'] is None
        for x in result['per_episode_neural'])


def test_refused8_keeps_unreached_null_other_episodes_continue(refused_saved,monkeypatch):
    out,r,p,prepared,h,data=refused_saved;result=consume_fixture(refused_saved,monkeypatch)
    assert result['complete'] and not result['all20'] and not result['equal_paid_axes_full20']
    assert result['prepared']==str(Path(prepared).absolute())
    assert r['completed20']==4 and r['refused']==4 and len(r['rows'])==8
    assert len(result['audited'])==16
    for row in r['rows']:
        if row['mode'].startswith('adaptive'):
            assert row['status']=='contract_refused' and row['control_returned']==5
            own=[n for n in r['nodes'] if n['episode_id']==row['episode_id']]
            assert [n['status'] for n in own]==['refused','not_reached','not_reached']
            assert own[1]['receipt'] is own[2]['resource_receipt'] is None
            assert len(row['encodings'])==1 and row['actual_Q20'] is None


def test_native_earlystop_all8_saved_strict_no_padding(stopped_saved,monkeypatch):
    out,r,p,prepared,h,data=stopped_saved;result=consume_fixture(stopped_saved,monkeypatch)
    assert result['complete'] and not result['all20'] and r['completed20']==0 and r['refused']==0
    assert len(result['audited'])==8 and len(r['rows'])==8
    assert all(row['status']=='native_stopped_control' and row['control_returned']==8 and row['truncated']
        and len(row['executed_raw'])==8 and len(row['encodings'])==1 and row['actual_Q20'] is None for row in r['rows'])
    assert result['physical_census']['counts']['encoder']==dict(attempted=8,completed=8)
    assert result['physical_census']['counts']['public_step']==dict(attempted=144,completed=144)
    assert all(n['status']=='not_reached' and n['receipt'] is None for n in r['nodes'] if n['boundary']>5)


@pytest.mark.parametrize('damage',('counts','nodes','precision_after','model_after','packets','unknown_physical','preprocessed'))
def test_full_saved_mutations_reject(full_saved,monkeypatch,damage):
    out,r,p,prepared,h,data=full_saved;changed=deepcopy(r);rp=out/'report.json';before=rp.read_bytes();other=None
    if damage=='counts':changed['counts']['physics_step']['completed']-=1
    elif damage=='nodes':changed['nodes'].pop()
    elif damage=='precision_after':changed['precision_after']['autocast']=True
    elif damage=='model_after':changed['model_after']['all_buffers']='f'*64
    elif damage=='packets':changed['rows'][0]['packets'][0]['commands'][0][0]+=.125
    elif damage=='unknown_physical':
        other=out/'journal.jsonl';oldbytes=other.read_bytes();lines=[json.loads(x) for x in oldbytes.splitlines()]
        lines[1]['kind']='invented_reset';previous=None
        for i,e in enumerate(lines):
            e.update(index=i,previous_sha256=previous);e['event_sha256']=m.old.source.digest({k:v for k,v in e.items() if k!='event_sha256'});previous=e['event_sha256']
        other.write_bytes(b''.join(m.old.raw(e) for e in lines));changed.update(journal_sha256=m.sha(other),journal_chain_tail=previous)
    else:
        row=r['rows'][0];other=out/(row['episode_id']+'_b05_encode/preprocessed_pixels.npy');oldbytes=other.read_bytes()
        a=np.load(io.BytesIO(oldbytes),allow_pickle=False);a.flat[0]=np.nextafter(a.flat[0],np.float32(np.inf));b=io.BytesIO();np.save(b,a,allow_pickle=False);other.write_bytes(b.getvalue())
    try:
        rp.write_bytes(m.raw(changed))
        with pytest.raises(ValueError):consume_fixture(full_saved,monkeypatch)
    finally:
        rp.write_bytes(before)
        if other is not None:other.write_bytes(oldbytes)


def test_real_writer_and_native_journal_roundtrip(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    store=m.Store(tmp_path/'results'/(m.NAME+'_RAM'))
    store.emit('run_started',contract=m.contract());store.emit('run_finished',status='partial_or_failed',counts={},nodes_sha256='a'*64)
    journal=m.live.NativeJournal(store,'probe',lambda:None);journal.event(dict(kind='RAM_data',x=1));journal.close()
    assert m.signal.journal(journal.path)==([dict(kind='RAM_data',x=1)],journal.tail)
    store.put('metadata.json',dict(x=1));assert m.old.read(store.path/'metadata.json')==dict(x=1)
    with pytest.raises((ValueError,FileExistsError)):m.Store(store.path)
