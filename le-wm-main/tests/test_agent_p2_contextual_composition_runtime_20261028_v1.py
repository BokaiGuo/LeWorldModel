"""RAM analytic fixtures and saved input schema only, not neural evidence."""
from copy import deepcopy
import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_contextual_composition_runtime_20261028_v1 as m


def context():
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())


def step(c,raw):
    c=deepcopy(c); raw=np.asarray(raw,np.float32)
    z=np.asarray(c['latent_history'][-1],np.float32)
    z=z+np.float32(raw.sum()/100.)
    c['latent_history']=c['latent_history'][1:]+[z.astype(float).tolist()]
    c['action_history']=c['action_history'][1:]+[m.frontier._normalized_token(raw)]
    return c


@pytest.fixture(scope='module')
def fixture_row():
    initial=context(); initial['action_history']=[m.frontier._normalized_token(np.zeros((5,2)))]*2
    root=dict(root_id='RAM_ONLY',root_rgb_sha256='1'*64,goal_rgb_sha256='2'*64,context=initial)
    domain=dict(name='raw',units='RAM fixture',normalization_sha256='3'*64,action_block=5,low=[-1.,-1.],high=[1.,1.])
    prov={k:'4'*64 for k in m.frontier.legacy().compose.__globals__['PROVENANCE_KEYS']}
    compiled=dict(goal_vectors=[np.ones(192).tolist()],scoring=dict(metric=dict(projection=np.eye(192).tolist())))
    base=m.frontier.legacy().sra.spectral_bank(np.zeros((192,20)),np.zeros(192),compiled['goal_vectors'][0],np.eye(192))
    bank=m.expanded_bank(base,0); results=[]
    def suffix(c,prefix,i):
        n=step(c,np.zeros((5,2)))['latent_history'][-1]
        local=m.frontier.legacy().suffix_solution(prefix,np.zeros((192,10)),n,compiled['goal_vectors'][0],np.eye(192))
        return dict(local,input_context_sha256=m.digest(c))
    for arm in m.ARMS:
        results.append(m.frontier.run_arm(arm=arm,bank=bank,root=root,provenance=prov,
            action_domain=domain,target_metric=m.frontier.legacy().full20.metric_for(compiled),replay=step,
            suffix=suffix,fixture=True))
    row=dict(base=base,bank=bank,results=results,modules=[])
    ready=dict(root=root,action_domain=domain)
    return initial,ready,prov,compiled,row


def test_fixed_contract_no_science_or_fake_equality():
    p=m.protocol()
    assert p['candidate_denominator']==4*(80+40)==480 and p['plan_denominator']==8
    assert p['total_predict']==4*(4+164+164)==1328
    assert p['standalone_predict_per_arm']==[168,168]
    assert p['inherited_search_costs'] is p['total_FLOPs'] is p['backward_engine_calls'] is None
    assert not any(p[k] for k in ('P1_admission','P2_admission','fresh','physical_release','equal_FLOPs'))


@pytest.mark.parametrize('slot',range(4))
def test_bank_preserves_frontier_base_and_all80(fixture_row,slot):
    base=fixture_row[-1]['base']; bank=m.expanded_bank(base,slot)
    assert bank==m.expanded_bank(base,slot)
    assert bank['seed']==20261027+slot and bank['logits'][:40]==base['logits']
    assert bank['commands'][:40]==base['commands']
    u=np.asarray(bank['logits']); assert u.shape==(80,20)
    assert np.max(np.linalg.norm(u,axis=1))<=m.frontier.CAP*(1+1e-12)
    assert np.array_equal(np.asarray(bank['commands']),np.tanh(u).astype(np.float32).reshape(80,2,5,2))


@pytest.mark.parametrize('slot',[-1,4,True,1.5])
def test_bank_rejects_unfixed_slots(fixture_row,slot):
    with pytest.raises(ValueError): m.expanded_bank(fixture_row[-1]['base'],slot)


def test_real_frontier_orchestration_ram_only_own_suffix(fixture_row):
    _,_,_,_,row=fixture_row
    assert [len(r['attempts']) for r in row['results']]==[80,40]
    assert [len(r['ledger']) for r in row['results']]==[164,84]
    for r in row['results']:
        assert r['complete'] and r['fixture_only']
        assert r['winner']['independent_full_replay_exact']
    for c in row['results'][1]['attempts']:
        assert len(c['witness']['edges'])==2
        assert c['suffix_compilation']['input_context_sha256']==m.digest(c['witness']['edges'][0]['output_context'])


def test_fixture_cannot_export_as_actual(fixture_row):
    _,ready,prov,compiled,row=fixture_row
    with pytest.raises(ValueError,match='actual arm'):
        m.frontier.validate_result(row['results'][0],arm=m.ARMS[0],bank=row['bank'],ready=ready,prov=prov,compiled=compiled)


def test_wrong_observed_context_suffix_is_rejected(fixture_row):
    _,ready,prov,compiled,row=fixture_row
    def wrong(c,p,i): return dict(suffix_logits=[0.]*10,input_context_sha256='0'*64)
    r=m.frontier.run_arm(arm=m.ARMS[1],bank=row['bank'],root=ready['root'],provenance=prov,
        action_domain=ready['action_domain'],target_metric=m.frontier.legacy().full20.metric_for(compiled),
        replay=step,suffix=wrong,fixture=True)
    assert not r['complete'] and len(r['attempts'])==1 and 'context' in r['error']['error']


def emit_entry(gate,i):
    spec=m.frontier.expected_entries(80)[i]
    x=dict(index=i,kind=spec['kind'],scope=m.frontier.expected_scopes()[i],
        samples=1,context_tokens=3,completed=False)
    gate(dict(kind='entry_attempt',entry=x))
    gate(dict(kind='entry_terminal',entry=dict(x,completed=True)))


def test_gate_full_actual_entry_schedule_and_no_refund(fixture_row):
    events=[]; g=m.Gate(events.append)
    exercise_gate(g,fixture_row[-1])
    assert len(g.entries)==664 and all(x['completed'] for x in g.entries)
    assert sum(e['kind']=='reserve_intent' for e in events)==664+83
    assert not any(e.get('actual_completed',False) for e in events)
    with pytest.raises(ValueError,match='predispatch'):
        g(dict(kind='entry_attempt',entry=dict(index=664,kind='action_encoder',scope=None,samples=1,context_tokens=3,completed=False)))
    assert g.failed


@pytest.mark.parametrize('field,value',[('kind','predict'),('samples',2),('context_tokens',2),('scope',None),('completed',True)])
def test_bad_entry_before_dispatch(field,value):
    g=m.Gate(lambda e:None)
    x=dict(index=0,kind='action_encoder',scope=m.frontier.expected_scopes()[0],samples=1,context_tokens=3,completed=False)
    x[field]=value
    with pytest.raises(ValueError): g(dict(kind='entry_attempt',entry=x))
    assert g.failed and not g.entries


def test_fsync_reservation_failure_stops_and_poisoned():
    def broken(e): raise OSError('RAM fsync error')
    g=m.Gate(broken)
    with pytest.raises(OSError): emit_entry(g,0)
    assert not g.entries and g.failed
    with pytest.raises(ValueError,match='poisoned'): emit_entry(g,0)


def test_pending_and_failed_return_never_complete():
    g=m.Gate(lambda e:None)
    x=dict(index=0,kind='action_encoder',scope=m.frontier.expected_scopes()[0],samples=1,context_tokens=3,completed=False)
    g(dict(kind='entry_attempt',entry=x))
    with pytest.raises(ValueError):g.complete([])
    with pytest.raises(ValueError):g(dict(kind='entry_terminal',entry=dict(x,error='RAM failed')))
    assert len(g.entries)==1 and not g.entries[0]['completed'] and g.failed


def test_operation_order_and_return_cap():
    events=[]; g=m.Gate(events.append)
    for i in range(4):emit_entry(g,i)
    e=dict(kind='jacobian',shape=[192,20],vectorize=True,completed=False)
    g(e)
    for i in range(4,8):emit_entry(g,i)
    g(dict(e,completed=True))
    assert len(g.operations)==2 and g.pending is None
    with pytest.raises(ValueError):g(e)


def operation_events(row):
    result=[dict(kind='jacobian',shape=[192,20],vectorize=True,scalar_output_vjp_directions_requested=192,completed=False),
        dict(kind='bounded_ridge_svd',svd_calls_requested=1,completed=False),
        dict(kind='mode_svd',svd_calls_requested=1,completed=False)]
    for i,c in enumerate(row['results'][1]['attempts']):
        n=c['suffix_compilation']['svd_calls']
        result.extend((dict(kind='suffix_jacobian',index=i,shape=[192,10],vectorize=True,completed=False),
            dict(kind='suffix_solve',index=i,svd_calls_requested=n,completed=False)))
    for e in result:
        yield e
        terminal=dict(e,completed=True)
        if e['kind']=='suffix_solve':terminal.update(svd_calls=e['svd_calls_requested'],bounded_solve_calls=e['svd_calls_requested'])
        yield terminal


def exercise_gate(g,row):
    """Match the literal native entry/AD/save boundaries in RAM."""
    ops=list(operation_events(row))
    at={4:[dict(kind='array_saved',receipt={'file':'RAM_shared_nominal'}),ops[0]],
        8:[dict(kind='array_saved',receipt={'file':'RAM_shared_jacobian'})]+ops[1:6]}
    for i in range(40):
        at[340+8*i]=[dict(kind='array_saved',receipt={'file':f'RAM_suffix{i}_nominal'}),ops[6+4*i]]
        at[342+8*i]=[dict(kind='array_saved',receipt={'file':f'RAM_suffix{i}_jacobian'})]+ops[7+4*i:10+4*i]
    for i in range(665):
        for e in at.get(i,[]):g(e)
        if i<664:emit_entry(g,i)


def test_full_gate_resource_reconciliation(fixture_row):
    row=fixture_row[-1];g=m.Gate(lambda e:None)
    exercise_gate(g,row)
    receipt=g.complete(row['results'])
    assert receipt['actual_forward_entries']==664
    assert receipt['math']['jacobian_calls']==41
    assert receipt['math']['scalar_output_vjp_directions']==7872
    assert receipt['budget']['paired_unique_predict']==332
    assert receipt['total_FLOPs'] is None


def tensor(value):
    a=np.asarray(value,np.float32)
    return dict(dtype='float32',shape=list(a.shape),values=a.astype(float).tolist())


def modules_for(initial,row):
    """Independent analytic fixture schedule; no torch/model calls."""
    packets=[]
    def dispatch(before,raw):
        after=step(before,raw); packets.append((deepcopy(before),np.asarray(raw,np.float32),after))
        return after
    for _ in range(2):
        mid=dispatch(initial,np.zeros((5,2))); dispatch(mid,np.zeros((5,2)))
    for arm,result in zip(m.ARMS,row['results']):
        for c in result['attempts']:
            mid=dispatch(initial,c['actions'][:5])
            if arm==m.ARMS[1]:
                dispatch(mid,np.zeros((5,2))); dispatch(mid,np.zeros((5,2)))
            dispatch(mid,c['actions'][5:])
        win=next(x for x in result['attempts'] if x['candidate_id']==result['winner']['candidate_id'])
        for _ in range(2):
            mid=dispatch(initial,win['actions'][:5]); dispatch(mid,win['actions'][5:])
    assert len(packets)==332
    modules=[]
    for scope,(before,raw,after) in zip(m.frontier.expected_scopes()[::2],packets,strict=True):
        outgoing=np.asarray(before['action_history']+[m.frontier._normalized_token(raw)],np.float32)[None]
        z=np.asarray(before['latent_history'],np.float32)[None]
        # A finite fictitious embedding; only RAM bridge consistency is tested.
        embedding=np.zeros((1,3,192),np.float32)
        pred=z.copy();pred[0,-1]=np.asarray(after['latent_history'][-1],np.float32)
        for resource,inputs,out in [('action_encoder',[outgoing],embedding),
            ('predictor',[z,embedding],pred),('pred_proj',[pred.reshape(3,192)],pred.reshape(3,192))]:
            modules.append(dict(index=len(modules),resource=resource,scope=scope,completed=True,
                inputs=[tensor(x) for x in inputs],output=tensor(out)))
    return modules


@pytest.fixture(scope='module')
def tensor_row(fixture_row):
    initial,_,_,_,row=fixture_row
    row=deepcopy(row);row['modules']=modules_for(initial,row)
    return dict(packet=dict(root_context=initial)),row


def test_all332_tensor_to_action_and_full_history_bridges(tensor_row):
    m.check_tensor_chain(*tensor_row)


@pytest.mark.parametrize('which',('latent','action','output','scope','missing_module','suffix_nominal'))
def test_mutations_fail_tensor_chain(tensor_row,which):
    slot,row=deepcopy(tensor_row)
    if which=='latent':row['modules'][1]['inputs'][0]['values'][0][0][0]=1.
    if which=='action':row['modules'][0]['inputs'][0]['values'][0][2][0]=1.
    if which=='output':row['modules'][2]['output']['values'][-1][0]=1.
    if which=='scope':row['modules'][0]['scope']=None
    if which=='missing_module':row['modules'].pop()
    if which=='suffix_nominal':row['results'][1]['attempts'][0]['suffix_compilation']['nominal'][0]=1.
    with pytest.raises(ValueError):m.check_tensor_chain(slot,row)


def test_raw_tensor_rounding_not_accepted():
    receipt=tensor([1.]);receipt['values'][0]=1.0000000001
    with pytest.raises(ValueError):m.module_array(receipt)


def test_actual_cost_callback_budget_and_pending_failure(fixture_row):
    initial,_,_,compiled,_=fixture_row;events=[];metric=m.Metric(compiled,events.append)
    for arm,n in zip(m.ARMS,(80,40)):
        for _ in range(n):assert metric(arm,initial)==1.
        with pytest.raises(ValueError,match='cost evaluation cap'):metric(arm,initial)
    assert len(metric.entries)==120 and len(events)==240
    assert events[0]['receipt']['completed'] is False and events[1]['receipt']['value']==1.
    def broken(e):raise OSError('RAM cost fsync failed')
    metric=m.Metric(compiled,broken)
    with pytest.raises(OSError):metric(m.ARMS[0],initial)
    assert metric.failed and not metric.entries


def test_store_exclusive_journal_and_caps(tmp_path,monkeypatch):
    p=tmp_path/(m.NAME+'_RAM');store=m.Store(p)
    store.event(dict(RAM=True));store.put('e.json',dict(unknown=None));store.close()
    raw,eof=m.overlay.blind.inflate((p/'events.jsonl.gz').read_bytes(),1024)
    assert eof and raw==m.canonical(dict(RAM=True))+b'\n'
    with pytest.raises(ValueError):m.Store(p)
    with pytest.raises(ValueError):store.event({})
    q=tmp_path/(m.NAME+'_RAMcap');store=m.Store(q)
    monkeypatch.setattr(m,'CAP',store.output.joinpath('events.jsonl.gz').stat().st_size+8)
    with pytest.raises(ValueError):store.put('overflow.json',{'x':'y'*50})


def test_reserved_roots_and_selector_are_not_inputs():
    assert m.ORDER==((0,'off'),(0,'on'),(1,'off'),(1,'on'))
    assert 'selector' not in m.source_files.__code__.co_names
    assert 'fixture' not in m.execute_slot.__code__.co_names


def test_actual_saved_four_task_schema_no_model_execution():
    r=m.overlay.read(m.assets.PARENT/'report.json',4*1024**2)
    assert m.sha(m.assets.PARENT/'report.json')==m.assets.PARENT_SHA
    for i,key in enumerate(m.ORDER):
        path=m.assets.PARENT/f'task_{i:02}.json.gz';e=m.overlay.read(path,m.PREP_CAP)
        slot=dict(index=i,task_file=str(path),task_sha256=r['artifact_sha256'][path.name],task_id=e['task_id'],
            packet=dict(root_context=e['task']['root_context'],goal=e['task']['goal']))
        ready,compiled=m.task_bindings(slot)
        assert compiled['goal_vectors']==[e['task']['goal']]
        assert ready['root']['context']==e['task']['root_context']
        assert (e['task_contract']['root_slot'],e['task_contract']['condition'])==key


def hook_order_events():
    out=[]
    for j,scope in enumerate(m.frontier.expected_scopes()[::2]):
        for kind,resource,index in [('module_attempt','action_encoder',3*j),('entry_attempt','action_encoder',2*j),
            ('entry_terminal','action_encoder',2*j),('module_return','action_encoder',3*j),
            ('entry_attempt','predict',2*j+1),('module_attempt','predictor',3*j+1),
            ('module_return','predictor',3*j+1),('module_attempt','pred_proj',3*j+2),
            ('module_return','pred_proj',3*j+2),('entry_terminal','predict',2*j+1)]:
            if kind.startswith('module'):
                out.append(dict(kind=kind,receipt=dict(resource=resource,index=index,scope=scope)))
            else:
                out.append(dict(kind='native_event',event=dict(kind=kind,entry=dict(kind=resource,index=index,scope=scope))))
    return out


def test_exact_actual_module_hook_order():
    m.check_module_event_order(hook_order_events())


@pytest.mark.parametrize('mutation',('drop','swap','scope','extra'))
def test_hook_order_rejects_hash_only_census(mutation):
    events=hook_order_events()
    if mutation=='drop':events.pop()
    if mutation=='swap':events[1],events[2]=events[2],events[1]
    if mutation=='scope':events[0]['receipt']['scope']=None
    if mutation=='extra':events.append(deepcopy(events[0]))
    with pytest.raises(ValueError):m.check_module_event_order(events)


def test_incomplete_report_cannot_be_consumed(tmp_path,monkeypatch):
    path=tmp_path/'report.json'
    body=dict(schema=m.NAME+'.report',status='partial_or_failed',prepared='unused',prepare_sha256='0'*64)
    path.write_bytes(m.canonical(body))
    monkeypatch.setattr(m,'verify',lambda *a:dict(expected_model_hash='1'*64))
    with pytest.raises(ValueError,match='terminal complete'):m.consume(tmp_path,m.sha(path))


def test_bad_prepare_sha_fails_before_build(tmp_path,monkeypatch):
    path=tmp_path/'x.json';path.write_bytes(b'{}')
    def forbidden():raise AssertionError('must not open inherited source')
    monkeypatch.setattr(m,'build',forbidden)
    with pytest.raises(ValueError,match='prepare SHA'):m.verify(path,'0'*64)
