"""Six-arm pure RAM transitions and synthetic native arrays; never model evidence."""
from copy import deepcopy
from contextlib import ExitStack
import io
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p3_commitment_controls_20261029_v2 as m
import agent_p3_fullcommit_live_20261028_v1_tests as old_tests
import test_p1_moving_live_feedback_physical_20261027_v1 as physical
import agent_p3_native_program_20261028_v1_test as programs
import agent_p3_same_feedback_physical20_20261028_v2_tests as rgb_tests


def execute_ram(*,fault=None,at=16,real_plan=False,path='/RAM_CONTROLS',compile_failure=None,
                visual_failure=None,geometry='long'):
    p,data=physical.fixture();p['contract']=m.contract();p['tasks']=[]
    p['layers']=[next(x for x in p['layers'] if x['source_slot']==s and x['arm']==m.old.ARMS[0]) for s in (0,1)]
    data['visual_tasks']=data.pop('tasks')
    for s in (0,1):
        t=programs.task(s);g=data['visual_tasks'][s]['gn_task']
        for k in ('root_context','goal','action_mean','action_std'):t[k]=deepcopy(g[k])
        t['binding'].update(context_sha256=m.digest(t['root_context']),goal_sha256=m.digest(t['goal']),
            normalizer_sha256=m.digest([t['action_mean'],t['action_std']]))
        if real_plan:
            t['goal']=[1.,.5]+[0.]*190;t['binding']['goal_sha256']=m.digest(t['goal'])
        p['tasks'].append(m.core.validate_task(t))
    store=old_tests.RAM(path);requests=[];envs=[];real_node=m.plan_node
    def plan(req,base,factory,sink,ledger,**kw):
        requests.append(deepcopy(req));b=req['boundary'];mode=req['mode']
        if compile_failure==b:raise RuntimeError('explicit RAM compile failure')
        if real_plan:
            return real_node(req,base,lambda t,s:old_tests.RamBackend(t,s),sink,ledger,permit_ram=True)
        if req['commitment'] is not None:
            result=dict(status='observed_commitment',commands=None,resource_receipt=ledger.receipt())
        else:
            h=3 if mode=='fixedH3' or mode=='adaptive' and (geometry=='long' or b>5) else 1
            if h*5>20-b:
                result=dict(status='refused',commands=None,refusal='fullH_does_not_fit',resource_receipt=ledger.receipt())
            else:
                result=dict(status='sealed',commands=np.full((5*h,2),np.float32(.02+b*.001)).tolist(),
                    resource_receipt=ledger.receipt())
        sink.event(dict(kind='RAM_plan',boundary=b));return result
    class Env(physical.ArrayEnv):
        def __init__(self):super().__init__(fault=fault,at=at)
    def factory(meter):
        env=m.old.metered_class(Env,meter)();envs.append(env);return env
    encode=m.old.Handler.encode_observation
    def visual(self,request,sink,**runtime):
        if visual_failure is not None and f'_b{visual_failure:02}' in request['episode_id']:
            raise RuntimeError('explicit RAM RGB failure')
        return encode(self,request,sink,**runtime)
    # Synthetic modules/Env ONLY; production source has no patched globals.
    with physical.visual_mock(store) as (runtime,_),ExitStack() as stack:
        for obj,key,value in ((m,'NativeJournal',old_tests.FullRamJournal if real_plan else old_tests.RamJournal),
            (m,'plan_node',plan),(m,'sha',store.hash),(m.old,'sha',store.hash),
            (m.old.Handler,'encode_observation',visual),(m.old.source,'sample_state',lambda *a:physical.REQUEST.copy())):
            stack.enter_context(patch.object(obj,key,value))
        result=m.execute(p,data,store,m.old.Handler(None),runtime,factory,None,authorization=m.AUTH,
            final_check=lambda:dict(model_unchanged=True),permit_ram=True)
    store.test_p=p;store.test_data=data
    return result,store,envs,requests


def saved_ram(tmp_path,*,fault=None):
    output=tmp_path/'RAM_SAVED';output.mkdir()
    r,store,envs,req=execute_ram(fault=fault,real_plan=True,path=str(output))
    assert r['status']=='known_fullcommit_recorded',[(x['status'],x['first_error']) for x in r['rows']]
    p=store.test_p;p['sources']=rgb_tests.image_sources()['sources']
    p['modelbinding']=m.moving.suffix.modelbinding();p['expected_model_hash']=p['modelbinding']['tensor_sha256']
    prepared=str(tmp_path/'PREPARED_RAM.json');h='a'*64
    hashes=dict(original_state_dict=p['expected_model_hash'],all_buffers='b'*64)
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    r.update(prepared=prepared,prepare_sha256=h,source_manifest_sha256=m.digest(p['sources']),
        model_before=hashes,model_after=hashes,precision_before=precision,precision_after=precision,
        CUDA_peak_bytes=[0,0],per_episode_neural=m.node_accounting(r['nodes']))
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
def full_saved(tmp_path_factory):return saved_ram(tmp_path_factory.mktemp('controls6'))


def test_six_independent_initializations_full20_and_complete_commitments():
    r,store,envs,reqs=execute_ram()
    assert r['status']=='known_fullcommit_recorded' and len(reqs)==18 and r['completed20']==6
    assert len(envs)==6 and len({id(e) for e in envs})==6
    assert all(len(e.resets)==2 and len(e.commands)==30 and e.closed for e in envs)
    assert r['counts']['reset']==dict(attempted=12,completed=12)
    assert r['counts']['set_state']==dict(attempted=24,completed=24)
    assert r['counts']['public_step']==dict(attempted=180,completed=180)
    assert r['counts']['physics_step']==dict(attempted=1824,completed=1824)
    assert r['counts']['explicit_render']==dict(attempted=192,completed=192)
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=24,completed=24)
    assert m.census(store.events,r)['complete'] and r['budget_match']=='notmatched'
    assert sum(e['kind']=='initialization_intent' for e in store.events)==6
    for row in r['rows']:
        own=[q for q in reqs if q['episode_id']==row['episode_id']]
        assert len(row['commitments'])==(3 if row['mode']=='fixedH1' else 1)
        assert [q['commitment'] is None for q in own]==([True]*3 if row['mode']=='fixedH1' else [True,False,False])
        for q in own:
            raw=np.concatenate((m.old.kernel.moving.warmup_commands(),
                np.asarray(row['executed_raw'][:q['boundary']],np.float32)))[-10:]
            assert q['raw10']==raw.tolist()
            assert not {'state','reward','env','success','goal7','snapshot'}&q.keys()


def test_full_saved_arrays_RGB_native_and_costs(full_saved,monkeypatch):
    c=consume_fixture(full_saved,monkeypatch)
    assert c['complete'] and c['fixed_denominator']==6 and c['node_denominator']==18
    assert len(c['audited'])==18 and c['RGB_preprocessing']['frames']==24
    assert c['physical_census']['complete'] and c['budget_match']=='notmatched'
    for arm in c['per_episode_neural']:
        assert arm['signal_completed']['predict_API']==24
        # Synthetic fixture adaptive chooses H3/n4. FixedH3 is explicitly H3/n8.
        assert arm['completed']['predict_API']==(75 if arm['mode']=='fixedH3' else 51)
        assert arm['FLOPs'] is None
    assert all(c[k] is False for k in ('P1_admission','P3_admission','freshness','scientific_admission'))


@pytest.mark.parametrize('fault',['truncate','throw','substitute','guard'])
def test_earlystop_or_failure_keeps_six_and_eighteen(fault):
    r,store,envs,_=execute_ram(fault=fault)
    assert len(r['rows'])==6 and len(r['nodes'])==18 and all(e.closed for e in envs)
    assert r['completed20']==0 and all(x['control_returned']<=6 for x in r['rows'])
    assert r['status']==('known_fullcommit_recorded' if fault=='truncate' else 'partial_or_failed')
    assert m.census(store.events,r)['complete']==(fault=='truncate')


def test_refusal_does_not_force_h1_or_truncate_h3():
    r,store,_,_=execute_ram(geometry='tail')
    assert r['status']=='known_fullcommit_recorded' and r['completed20']==4
    assert all(x['control_returned']==10 for x in r['rows'] if x['mode']=='adaptive')
    assert r['equal_actual_execution_horizon'] is False
    assert m.census(store.events,r)['complete']


@pytest.mark.parametrize('kind',['compile','RGB'])
def test_no_dispatch_from_stale_or_failed_node(kind):
    r,_,envs,_=execute_ram(**({'compile_failure':10} if kind=='compile' else {'visual_failure':10}))
    assert r['status']=='partial_or_failed' and all(len(e.commands)==20 and e.closed for e in envs)
    assert all(x['control_returned']==10 for x in r['rows'])


def test_saved_native_stop_is_complete_engineering_not_full20(tmp_path,monkeypatch):
    b=saved_ram(tmp_path,fault='truncate');c=consume_fixture(b,monkeypatch)
    assert c['complete'] and c['completed20']==0 and len(c['audited'])==6
    assert all(x['control_returned']==6 for x in b[1]['rows'])
    assert c['counts']['public_step']['completed']==96


@pytest.mark.parametrize('damage',['unknown_event','missing_constructor','reset_count','warmup_action','goal','commitment','frame','policy','denominator'])
def test_saved_consumer_coherent_transport_mutations(full_saved,monkeypatch,damage):
    out,original,p,prepared,h,data=full_saved;r=deepcopy(original);backup={}
    def replace(path,value):
        path=Path(path);backup.setdefault(path,path.read_bytes());path.write_bytes(value)
    if damage in ('unknown_event','missing_constructor','warmup_action','goal','reset_count'):
        events=m.load_events(out,original)
        if damage=='unknown_event':events[1]['kind']='unrecognized_native'
        elif damage=='missing_constructor':
            next(e for e in events if e['kind']=='operation_attempted' and e['resource']=='constructor')['resource']='reset'
        elif damage=='reset_count':r['counts']['reset']['completed']-=1
        elif damage=='warmup_action':next(e for e in events if e['kind']=='warmup_return')['command'][0]+=.1
        else:next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan')['receipt']['goal'][0]+=1
        prev=None
        for i,e in enumerate(events):
            e.update(index=i,previous_sha256=prev)
            e['event_sha256']=m.old.source.digest({k:v for k,v in e.items() if k!='event_sha256'});prev=e['event_sha256']
        replace(out/'journal.jsonl',b''.join(m.raw(e) for e in events))
        r.update(journal_sha256=m.sha(out/'journal.jsonl'),journal_chain_tail=prev)
    elif damage=='commitment':
        r['rows'][1]['commitments'][0]['commands'][8][0]+=.25
        replace(out/'s0_fixedH3_episode.json',m.raw(r['rows'][1]))
    elif damage=='frame':
        path=out/'s0_fixedH1_b05_encode/preprocessed_pixels.npy'
        a=np.load(path,allow_pickle=False);a.flat[0]+=.125;s=io.BytesIO();np.save(s,a,allow_pickle=False);replace(path,s.getvalue())
    elif damage=='policy':r['contract']['rule']['fixedH3']['candidates']=4
    else:r['rows'].pop()
    replace(out/'report.json',m.raw(r))
    try:
        with pytest.raises((ValueError,KeyError)):consume_fixture((out,r,p,prepared,h,data),monkeypatch)
    finally:
        for path,payload in backup.items():path.write_bytes(payload)


def test_prepare_new_closure_exclusive(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    material=dict(schema=m.NAME+'.prepare',RAM_only=True)
    monkeypatch.setattr(m,'build',lambda:deepcopy(material))
    path=tmp_path/'results'/(m.NAME+'_prepare_RAM.json');receipt=m.prepare(path)
    assert m.verify(path,receipt['sha256'])==material
    with pytest.raises(ValueError):m.prepare(path)
    path.write_bytes(b'{}')
    with pytest.raises(ValueError):m.verify(path,receipt['sha256'])


def test_cold_import_no_neural_cpu_or_implicit_worker():
    code='''
import builtins
orig=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in ('torch','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return orig(name,*a,**kw)
builtins.__import__=guard
from experiments.spectral_reachability import agent_p3_commitment_controls_20261029_v2 as m
try:m.run(None,None,None,authorization='no')
except ValueError:pass
else:raise AssertionError('authorization gate')
assert len(m.roster())==6 and len(m.node_roster())==18
'''
    r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


@pytest.fixture(scope='module')
def node_fixture():
    import test_agent_p3_commitment_controls_20261029_v1 as lower
    return lower.planned('fixedH3')


def test_frozen_node_adapter_saved_reconstruction(node_fixture):
    base,req,result,store=node_fixture
    a=m.audit_plan(req,result,store.events,result['resource_receipt'],base)
    assert a['horizon']==3 and a['candidates']==8


@pytest.mark.parametrize('damage',['J','grad','input','endpoint','cost','decision','unknown','missing_io'])
def test_all_native_array_mutations_rejected(node_fixture,damage):
    base,req,result,store=deepcopy(node_fixture);es=store.events
    if damage in ('J','grad','input','endpoint'):
        kind,key={'J':('jacobian','value'),'grad':('raw_backward','value'),
            'input':('program_packet','raw'),'endpoint':('program_boundary','latent')}[damage]
        e=next(e for e in es if e['kind']==kind and key in e['arrays']);a=m.signal.unblob(e['arrays'][key]).copy()
        a.flat[0]+=1;e['arrays'][key]=m.signal.blob(a)
    elif damage=='cost':result['candidates'][0]['cost']+=1
    elif damage=='decision':result['decision']['candidates']=4
    elif damage=='unknown':es.insert(0,dict(kind='outcome_selector'))
    else:es.pop(next(i for i,e in enumerate(es) if e['kind']=='io_return'))
    with pytest.raises((ValueError,KeyError)):m.audit_plan(req,result,es,result['resource_receipt'],base)


def test_real_inherited_store_put_emit_and_nativejournal_readers(tmp_path,monkeypatch):
    """Actual filesystem writers/readers, NOT the RAM codec or a neural worker."""
    import json
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    store=m.Store(tmp_path/'results'/(m.NAME+'_writer_probe'))
    # This distinction is the actual ABI, not an assumed strip/filter repair.
    assert not m.raw({'x':1}).endswith(b'\n')
    assert m.old.raw({'x':1}).endswith(b'\n')
    metadata={'schema':'writer_probe','values':[1,2]}
    saved=store.put('metadata.json',metadata)
    assert (store.path/'metadata.json').read_bytes()==m.old.raw(metadata)
    assert saved==m.sha(store.path/'metadata.json') and m.signal.read(store.path/'metadata.json')==metadata
    nodes=m.node_roster();counts={k:dict(attempted=0,completed=0) for k in m.contract()['physical_caps']|m.contract()['visual_caps']}
    store.emit('run_started',contract=m.contract())
    store.emit('run_finished',status='partial_or_failed',counts=counts,nodes_sha256=m.digest(nodes))
    payload=(store.path/'journal.jsonl').read_bytes()
    assert payload.endswith(b'\n') and not payload.endswith(b'\n\n')
    assert len(payload.splitlines())==2 and all(payload.splitlines())
    physical_records=[json.loads(line) for line in payload.splitlines()]
    assert payload==b''.join(m.old.raw(e) for e in physical_records)
    report=dict(journal_sha256=m.sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    assert m.load_events(store.path,report)==physical_records
    assert physical_records[0]['previous_sha256'] is None
    assert physical_records[1]['previous_sha256']==physical_records[0]['event_sha256']==m.old.source.digest(
        {k:v for k,v in physical_records[0].items() if k!='event_sha256'})
    checks=[];journal=m.NativeJournal(store,'writer_native',lambda:checks.append(True))
    native=[dict(kind='resource_probe',value=1),dict(kind='resource_probe',value=2)]
    for value in native:journal.event(value)
    journal.close();raw=journal.path.read_bytes()
    assert len(checks)==2 and raw.endswith(b'\n') and not raw.endswith(b'\n\n')
    assert len(raw.splitlines())==2 and all(raw.splitlines())
    records=[json.loads(line) for line in raw.splitlines()]
    assert raw==b''.join(m.raw(record)+b'\n' for record in records)
    values,tail=m.signal.journal(journal.path)
    assert values==native and tail==journal.tail
    assert records[0]['previous']=='0'*64 and records[1]['previous']==records[0]['sha256']
    # Reader must refuse a double-LF or bad nonempty record, never filter it.
    journal.path.write_bytes(raw.replace(b'\n',b'\n\n',1))
    with pytest.raises((ValueError,json.JSONDecodeError)):m.signal.journal(journal.path)
    journal.path.write_bytes(raw)
    records[1]['payload']['value']=3
    journal.path.write_bytes(b''.join(m.raw(x)+b'\n' for x in records))
    with pytest.raises(ValueError):m.signal.journal(journal.path)


@pytest.mark.parametrize('status',['native_stopped_control','contract_refused','completed20'])
def test_short_run_cannot_fake_clean_terminal_label(status):
    row=m.roster()[0];row.update(status=status,initial_exact=True,warmup_returned=10,
        control_returned=6,terminated=False,truncated=False)
    with pytest.raises(ValueError):m.terminal_disposition(row)
