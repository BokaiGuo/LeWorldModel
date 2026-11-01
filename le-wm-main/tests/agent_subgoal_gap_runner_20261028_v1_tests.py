"""Lightweight RAM metadata/JSON durability tests, never neural or physics."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_subgoal_gap_runner_20261028_v1 as m
from agent_subgoal_gap_contract_20261028_v1_tests import fixture as contract_fixture


def setup(tmp_path):
    score,artifacts=contract_fixture();shared=deepcopy(score['trials'][0]['shared'])
    witness=deepcopy(next(iter(artifacts.values())));root=witness['receipt']['root']
    def put(name,body):
        p=tmp_path/(name+'.json');p.write_bytes(m.c.encode(body));return dict(path=str(p),sha256=m.c.digest(body))
    assets={k:put(k,{'RAM_only':True}) for k in m.ASSETS}
    assets['adapter_source']=dict(path=str(Path(__file__).resolve()),sha256=m.file_sha(__file__))
    caps={phase:m.vector() for phase in m.PHASES}
    for cap in caps.values():
        for k in m.KNOWN:cap[k]=100
        cap['callback_calls']=1
    assets['budget']=put('budget',dict(schema=m.NAME+'.budget',phase_caps=caps))
    pair=dict(schema=m.NAME+'.pairing',method='HiLeWM_hierarchical_adaptation',prior_checkpoint_sha256=assets['learned_checkpoint']['sha256'])
    for k in ('model','encoder','normalizer','observation'):pair[k+'_sha256']=assets[k]['sha256']
    assets['learned_pairing']=put('learned_pairing',pair)
    controller=dict(schema=m.NAME+'.controller',latent_dim=2,checkpoints=[5,10],action_low=[-1.,-1.],action_high=[1.,1.],
        bindings={k:v['sha256'] for k,v in assets.items() if k not in ('controller','learned_checkpoint','learned_pairing','arbitrary_rule','compiled_rule')})
    assets['controller']=put('controller',controller)
    for k in m.COMMON:shared[k]=assets[k]['sha256']
    restore=dict(state='RAM metadata only',goal='unchanged',history='same',renderer='same',warmup='same')
    inp=put('input',dict(root=root,task_goal_sha256=shared['task_goal'],restore_binding=restore))
    trial=dict(trial_id='RAM-trial-0',shared=shared,input=inp,restore_binding_sha256=m.c.digest(restore),
        proposal_inputs={a:put(a,dict(query='RAM fixed query')) for a in m.c.ARMS})
    spec=dict(schema=m.NAME+'.input',latent_dim=2,checkpoints=[5,10],assets=assets,phase_caps=caps,trials=[trial])
    prepared=m.prepare(spec);calls=[]
    def restore_fn(inp,work):
        calls.append(('restore',id(restore_fn)))
        return work.call({'restore_calls':1},lambda:dict(shared_sha256=m.c.digest(shared),restore_binding_sha256=trial['restore_binding_sha256']))
    target=[[1.,0.],[2.,0.]]
    def proposer(arm):
        def propose(inp,query,work):
            calls.append(('proposal',arm));receipt=None;w=None
            if arm==m.c.ARMS[1]:
                work.call({'generator_calls':1},lambda:None)
                receipt=dict(schema=m.c.NAME+'.learned_pairing',method=pair['method'],prior_checkpoint_sha256=pair['prior_checkpoint_sha256'],
                    targets_sha256=m.c.digest(target),lowlevel_targets_sha256=m.c.digest(target),
                    generator_attempted=1,generator_completed=1,fallback=False)
                for k in ('model','encoder','normalizer','observation','history','task_goal'):receipt[k+'_sha256']=shared[k]
            if arm==m.c.ARMS[2]:w=deepcopy(witness)
            source=('arbitrary_rule','learned_checkpoint','compiled_rule')[m.c.ARMS.index(arm)]
            return dict(target=deepcopy(target),source_sha256=assets[source]['sha256'],learned_receipt=receipt,witness=w)
        return propose
    def controller_fn(inp,targets,steps,work):
        calls.append(('controller',id(controller_fn),m.c.digest(inp)))
        return work.call({'physics_steps':10},lambda:dict(shared_sha256=m.c.digest(shared),targets_sha256=m.c.digest(targets),
            predicted=deepcopy(targets),observed=deepcopy(targets),raw_actions=[[0.,0.]]*10))
    adapter=m.Adapter('RAM_METADATA',deepcopy(prepared['asset_hashes']),restore_fn,{a:proposer(a) for a in m.c.ARMS},
        controller_fn,lambda inp,tracking,work:None,lambda:pytest.fail('no preflight/model inspection in fixture'))
    return spec,prepared,adapter,calls,put


def test_RAM_three_arms_use_same_controller_and_restore_fixed_start(tmp_path):
    _,p,a,calls,_=setup(tmp_path);events=[]
    r=m.run_fixture(p,m.c.digest(p),a,events.append)
    assert r['status']=='callbacks_completed_NOT_scientific_pass' and r['fixture_only']
    assert r['arm_denominator']==3 and all(x['status']=='completed' for x in r['rows'])
    controllers=[x for x in calls if x[0]=='controller']
    assert len(controllers)==3 and len({x[1:] for x in controllers})==1
    assert sum(x[0]=='restore' for x in calls)==3
    assert all(x['native_success'] is None for x in r['rows'])
    assert all(x['deviation'][-1]['l2']==0 for x in r['rows'])
    assert len(r['costs'])==12 and all(x['returned']['flops'] is None for x in r['costs'])
    assert sum(x['returned']['physics_steps'] for x in r['costs'])==30  # dummy callbacks ONLY
    assert sum(x['returned']['generator_calls'] for x in r['costs'])==1
    assert r['physical_execution_authorized'] is r['scientific_pass'] is r['full_cost_instrumentation_complete'] is False
    assert events[0]['kind']=='header' and events[-1]['kind']=='terminal'


@pytest.mark.parametrize('missing',['learned_checkpoint','learned_pairing','controller','adapter_source'])
def test_missing_assets_refuse_all_callbacks_before_any_arm(tmp_path,missing):
    spec,_,a,calls,_=setup(tmp_path);spec['assets'][missing]=None;p=m.prepare(spec)
    assert p['status']=='blocked' and p['arm_denominator']==3
    with pytest.raises(ValueError,match='missing assets/controller'):m.run_fixture(p,m.c.digest(p),a,lambda e:pytest.fail('no journal'))
    assert calls==[]


@pytest.mark.parametrize('damage',['controller','pairing','budget','source_drift'])
def test_prepare_and_verify_reject_incompatible_bindings(tmp_path,damage):
    spec,p,a,calls,put=setup(tmp_path)
    if damage=='source_drift':
        Path(spec['assets']['learned_checkpoint']['path']).write_bytes(b'changed metadata asset')
        with pytest.raises(ValueError):m.verify(p,m.c.digest(p))
    else:
        key='learned_pairing' if damage=='pairing' else damage
        body=m.c.bound(spec['assets'][key]['path'],spec['assets'][key]['sha256'])
        if damage=='pairing':body['encoder_sha256']='b'*64
        elif damage=='controller':body['bindings']['model']='b'*64
        else:body['phase_caps']['controller']['physics_steps']=0
        spec['assets'][key]=put('changed_'+key,body)
        if damage in m.COMMON:
            for trial in spec['trials']:trial['shared'][damage]=spec['assets'][key]['sha256']
        with pytest.raises(ValueError):m.prepare(spec)
    assert calls==[]


def test_no_MAIN_P1_release_refuses_production_before_callbacks_or_journal(tmp_path):
    _,p,a,calls,put=setup(tmp_path);prod=replace(a,backend='HOST_CUDA')
    auth=put('authority',dict(schema=m.NAME+'.MAIN_authority',reviewer='MAIN',prepared_sha256=m.c.digest(p),
        P1_passed=False,freshness_passed=False,physical_run_authorized=False))
    path=tmp_path/'not_created.jsonl'
    with pytest.raises(ValueError,match='P1/freshness/physical'):
        m.run(p,m.c.digest(p),prod,path,auth['path'],auth['sha256'])
    assert calls==[] and not path.exists()


def test_production_rejects_unbound_controller_code_before_callbacks(tmp_path):
    _,p,a,calls,_=setup(tmp_path)
    # Another existing pure function is callable but not from the bound MAIN
    # adapter file. It must never be invoked under an asserted controller hash.
    a=replace(a,backend='HOST_CUDA',controller=m.c.digest)
    with pytest.raises(ValueError,match='bound MAIN adapter source'):
        m.run(p,m.c.digest(p),a,tmp_path/'unused.jsonl',None,None)
    assert not calls


@pytest.mark.parametrize('failure',['restore','proposal','controller','native'])
def test_failure_keeps_denominator_partial_cost_and_never_retries(tmp_path,failure):
    _,p,a,calls,_=setup(tmp_path)
    def fail(*args):raise RuntimeError('fixed '+failure+' failure')
    if failure=='proposal':a=replace(a,proposers={**a.proposers,m.c.ARMS[0]:fail})
    else:a=replace(a,**{failure:fail})
    r=m.run_fixture(p,m.c.digest(p),a,lambda e:None)
    assert r['status']=='failed_closed' and len(r['rows'])==r['arm_denominator']==3
    assert [x['status'] for x in r['rows']]==['failed','not_dispatched','not_dispatched']
    assert r['errors'][0]['message']=='fixed '+failure+' failure'
    assert any(x['pending'] for x in r['costs'])
    assert len(r['costs'])==12
    assert all(x['returned'] is None for x in r['costs'] if x['slot']>0)
    assert all(x['native_success'] is None for x in r['rows'])


def test_wrong_restore_digest_blocks_proposal_even_if_callback_returned(tmp_path):
    _,p,a,calls,_=setup(tmp_path)
    a=replace(a,restore=lambda inp,work:{'shared_sha256':'a'*64,'restore_binding_sha256':'a'*64})
    r=m.run_fixture(p,m.c.digest(p),a,lambda e:None)
    assert r['status']=='failed_closed' and not any(x[0]=='proposal' for x in calls)


def test_native_false_kept_separate_from_zero_latent_error(tmp_path):
    _,p,a,_,_=setup(tmp_path);shared=p['spec']['trials'][0]['shared']
    def native(inp,tracking,work):
        return work.call({'native_evaluator_calls':1},lambda:dict(success=False,
            evaluator_sha256=shared['native_success_evaluator'],task_goal_sha256=shared['task_goal']))
    r=m.run_fixture(p,m.c.digest(p),replace(a,native=native),lambda e:None)
    assert not r['errors'] and all(x['native_success'] is False for x in r['rows'])


@pytest.mark.parametrize('at',['reserve_intent','entry_boundary','host_return'])
def test_work_durable_failure_no_retry_and_pending_not_completion(at):
    calls=[];cap={p:{**m.vector(),'predict_calls':1} for p in m.PHASES}
    def sink(e):
        if e['kind']==at:raise OSError('durability '+at)
    w=m.Work(cap,sink);w.phase(0,'controller')
    with pytest.raises(OSError):w.call({'predict_calls':1},lambda:calls.append(1))
    assert calls==([1] if at=='host_return' else [])
    with pytest.raises(ValueError):w.call({'predict_calls':1},lambda:calls.append(2))
    assert w.accounts[(0,'controller')]['returned']['predict_calls']==0


def test_original_call_failure_not_overwritten_by_failed_failure_journal():
    def sink(e):
        if e['kind']=='failed_return':raise OSError('log secondary')
    def fail():raise RuntimeError('call primary')
    w=m.Work({p:{**m.vector(),'predict_calls':1} for p in m.PHASES},sink);w.phase(0,'proposal')
    with pytest.raises(RuntimeError,match='call primary'):w.call({'predict_calls':1},fail)
    assert str(w.primary)=='call primary' and w.secondary_errors[0]['message']=='log secondary'


def test_phase_cap_before_call_unknown_flops_stays_null():
    w=m.Work({p:m.vector() for p in m.PHASES},lambda e:None);w.phase(0,'proposal')
    with pytest.raises(ValueError,match='phase cap'):w.call({'predict_calls':1},lambda:pytest.fail('over cap'))
    assert w.accounts[(0,'proposal')]['reserved']['flops'] is None
    with pytest.raises(ValueError):w.call({'restore_calls':0},lambda:pytest.fail('no retry after cap rejection'))


@pytest.mark.parametrize('damage',['unmetered_generator','wrong_controller_target','fallback','CPU_backend','missing_query'])
def test_remaining_integration_boundaries(tmp_path,damage):
    spec,p,a,calls,_=setup(tmp_path)
    if damage=='CPU_backend':
        with pytest.raises(ValueError,match='backend'):m.run_fixture(p,m.c.digest(p),replace(a,backend='CPU'),lambda e:None)
        assert not calls;return
    if damage=='missing_query':
        spec['trials'][0]['proposal_inputs'][m.c.ARMS[1]]=None;p=m.prepare(spec)
        with pytest.raises(ValueError,match='missing assets/controller'):m.run_fixture(p,m.c.digest(p),a,lambda e:None)
        assert not calls;return
    if damage=='wrong_controller_target':
        original=a.controller
        def controller(inp,target,steps,work):
            body=original(inp,target,steps,work);body['targets_sha256']='a'*64;return body
        a=replace(a,controller=controller)
    else:
        original=a.proposers[m.c.ARMS[1]]
        def proposer(inp,query,work):
            body=original(inp,query,work)
            if damage=='fallback':body['learned_receipt']['fallback']=True
            else:work.accounts[work.current]['returned']['generator_calls']=0
            return body
        a=replace(a,proposers={**a.proposers,m.c.ARMS[1]:proposer})
    r=m.run_fixture(p,m.c.digest(p),a,lambda e:None)
    assert r['status']=='failed_closed' and len(r['rows'])==3


def test_zero_work_function_cannot_be_dispatched():
    w=m.Work({p:m.vector() for p in m.PHASES},lambda e:None);w.phase(0,'proposal')
    with pytest.raises(ValueError,match='nonzero'):w.call({'generator_calls':0},lambda:pytest.fail('free function call'))


def test_hidden_cap_error_remains_fatal_if_callback_catches_it():
    w=m.Work({p:{**m.vector(),'callback_calls':1} for p in m.PHASES},lambda e:None);w.phase(0,'proposal')
    def callback():
        try:w.call({'generator_calls':1},lambda:pytest.fail('over cap'))
        except ValueError:return 'attempt to hide failure'
    with pytest.raises(ValueError):w.call({'callback_calls':1},callback)
    assert str(w.primary)=='phase cap before dispatch' and w.poisoned


def test_Journal_metadata_hashchain_exclusive_create(tmp_path,monkeypatch):
    root=tmp_path/'sandbox';(root/'results').mkdir(parents=True);monkeypatch.setattr(m,'ROOT',root)
    p=root/'results'/(m.NAME+'_RAM.jsonl');j=m.Journal(p)
    j({'kind':'one'});j({'kind':'two'});j.close()
    lines=[m.c.parse(line) for line in p.read_bytes().splitlines()]
    assert lines[0]['previous']=='0'*64 and lines[1]['previous']==lines[0]['sha256']
    for row in lines:
        h=row.pop('sha256');assert m.c.digest(row)==h
    with pytest.raises(FileExistsError):m.Journal(p)


def test_prepare_verify_CLI_metadata_only(tmp_path,capsys):
    spec,p,_,calls,put=setup(tmp_path);s=put('spec',spec)
    assert m.main(['prepare','--input',s['path'],'--sha256',s['sha256']])==0
    raw=capsys.readouterr().out;out=m.c.parse(raw);assert out==p and not calls
    # Real stdout includes a newline: byte SHA and canonical prepare SHA differ.
    path=tmp_path/'prepared_stdout.json';path.write_text(raw)
    assert m.file_sha(path)!=m.c.digest(p)
    assert m.main(['verify','--input',str(path),'--sha256',m.file_sha(path)])==0
    assert m.c.parse(capsys.readouterr().out)==p
