"""RAM metadata callbacks + tmp metadata durability; no tensor/model/physics work."""
from copy import deepcopy
from types import SimpleNamespace as NS
import json
import pytest
from experiments.spectral_reachability import agent_p2_runtime_guard_20261028_v1 as m
from test_p2_composition_predict_frontier_cuda_20261027_v1 import fake_model,tensor,dispatch


class MemoryJournal:
    def __init__(self,fail=None): self.events=[]; self.fail=fail
    def append(self,kind,payload):
        if kind==self.fail: raise OSError('intent fsync denied')
        self.events.append(dict(kind=kind,payload=deepcopy(payload)))


def make_guard(journal=None):
    journal=journal or MemoryJournal()
    g=m.DispatchGuard(journal,0,m.native.ROOTS[0]); g.begin_shared()
    return g,journal


def native_forward(g,model,replay,index):
    spec=m.native.expected_entries(80)[index]; scope=m.native.expected_scopes()[index]
    replay._p2_scope=scope
    return dispatch(model,spec['kind'],tensor(spec['kind']))


def op(g,kind,index=None,count=1):
    e=dict(kind=kind,completed=False)
    if index is not None:e['index']=index
    if kind in ('jacobian','suffix_jacobian'):
        e.update(shape=[192,20 if kind=='jacobian' else 10],vectorize=True)
    else:e['svd_calls_requested']=count
    g(e); e['completed']=True
    if kind=='suffix_solve': e.update(svd_calls=count,bounded_solve_calls=count)
    g(e)


def test_full664_GENUINE_native_method_guard_and_primitive_budgets():
    g,j=make_guard(); model,replay,calls=fake_model(); faults=m.native.Faults()
    specs=m.native.expected_entries(80); scopes=m.native.expected_scopes()
    with m.native.guarded_entries(model,replay,g,faults):
        for index,(spec,scope) in enumerate(zip(specs,scopes)):
            # Root operation brackets the real native AD call site in production.
            if index==4: g(dict(kind='jacobian',completed=False,shape=[192,20],vectorize=True))
            if index==8:
                g(dict(kind='jacobian',completed=True,shape=[192,20],vectorize=True))
                op(g,'bounded_ridge_svd'); op(g,'mode_svd')
            if scope['phase']=='planning' and scope['path'][1]==0 and spec['kind']=='action_encoder':
                g(dict(kind='candidate_request',arm=scope['arm'],index=scope['path'][0],completed=False))
            if scope['phase']=='suffix_compilation' and scope['path'][1]=='jacobian' and spec['kind']=='action_encoder':
                g(dict(kind='suffix_jacobian',index=scope['path'][0],completed=False,shape=[192,10],vectorize=True))
            native_forward(g,model,replay,index)
            if spec['kind']=='predict' and scope['phase']=='suffix_compilation' and scope['path'][1]=='jacobian':
                g(dict(kind='suffix_jacobian',index=scope['path'][0],completed=True,shape=[192,10],vectorize=True))
                op(g,'suffix_solve',scope['path'][0],count=0 if scope['path'][0]==0 else 1)
            if spec['kind']=='predict' and scope['phase']=='planning' and scope['path'][1]==1:
                g.metric(lambda c: 1.)(None)
                i=scope['path'][0]; a=scope['arm']
                g(dict(candidate_id=f'{a}_{i:02d}',index=i,completed=True))
    r=g.snapshot()
    assert len(calls)==664 and not r['pending_operations'] and not r['unknown_completion']
    assert r['observed_returned_lower_bound_unique']['predict_calls']==332
    assert r['observed_returned_lower_bound_unique']['action_encoder_calls']==332
    assert r['observed_returned_lower_bound_unique']['svd_192x10_calls']==39
    assert r['observed_returned_lower_bound_unique']['jacobian_192x10_calls']==40
    assert all(r['reservation_intent_standalone'][a]['predict_calls']==168 for a in m.native.ARMS)
    assert all(r['observed_returned_lower_bound_standalone'][a]['predict_calls']==168 for a in m.native.ARMS)
    assert all(len(r['candidates'][a])==m.contract.COUNTS[a] for a in m.native.ARMS)
    assert all(c['status']=='completed' for cs in r['candidates'].values() for c in cs)
    for gdata in r['groups'].values():
        for axis in m.C.KNOWN: assert gdata['returned'][axis]<=gdata['subreserved'][axis]<=gdata['upper'][axis]
    assert all(r['actual_complete_totals'][axis] is None for axis in m.C.UNKNOWN)


@pytest.mark.parametrize('failure',['operation_reserve_intent','entry_boundary'])
def test_native_model_not_called_if_reservation_or_entry_fsync_fails(failure):
    g,j=make_guard(); j.fail=failure; model,replay,calls=fake_model(); faults=m.native.Faults()
    with pytest.raises(OSError):
        with m.native.guarded_entries(model,replay,g,faults): native_forward(g,model,replay,0)
    assert not calls and g.poisoned
    assert g.snapshot()['observed_returned_lower_bound_unique']['action_encoder_calls']==0


def test_native_compute_return_and_durable_return_failure_separate():
    g,j=make_guard(); model,replay,calls=fake_model(); j.fail='return_observed'
    with pytest.raises(OSError):
        with m.native.guarded_entries(model,replay,g): native_forward(g,model,replay,0)
    assert calls==['action_encoder']  # Actual mock callable DID run, durability did NOT.
    r=g.snapshot(); assert r['actual_complete_totals'] is None and r['pending_operations']
    assert r['observed_returned_lower_bound_unique']['action_encoder_calls']==0


def test_native_primary_exception_retained_and_actual_completion_unknown():
    g,j=make_guard(); model,replay,calls=fake_model(); primary=RuntimeError('callable primary')
    def broken(*a): raise primary
    model.action_encoder.forward=broken
    with pytest.raises(RuntimeError) as caught:
        with m.native.guarded_entries(model,replay,g): native_forward(g,model,replay,0)
    assert caught.value is primary
    r=g.snapshot(str(primary)); assert r['actual_complete_totals'] is None and r['failed_returns']
    assert all(c['status']=='skipped' for cs in r['candidates'].values() for c in cs)


def test_root_derivative_request_is_not_claimed_as_returned_AD():
    g,j=make_guard()
    g(dict(kind='jacobian',completed=False,shape=[192,20],vectorize=True))
    r=g.snapshot('fatal before return')
    assert r['actual_complete_totals'] is None
    assert r['observed_returned_lower_bound_unique']['jacobian_192x20_calls']==0
    assert r['reservation_intent_standalone'][m.native.ARMS[0]]['jacobian_192x20_calls']==1
    assert r['backward_engine_api_calls'] is r['flops'] is None


def test_even_zero_cost_primitive_cannot_repeat():
    g,j=make_guard();a=m.native.ARMS[1]
    g(dict(kind='candidate_request',arm=a,index=0,completed=False))
    op(g,'suffix_solve',0,count=0)
    with pytest.raises(ValueError,match='unique operation'):
        g(dict(kind='suffix_solve',index=0,completed=False,svd_calls_requested=0))


def test_primitive_budget_cannot_borrow_another_group_and_retry_forbidden():
    g,j=make_guard()
    with pytest.raises(ValueError,match='allocation exceeds'):
        g.reserve('too_much',m.C.forward(5),{})
    assert not g.pending
    with pytest.raises(ValueError,match='retry'): g.begin_shared()
    g(dict(kind='candidate_request',arm=m.native.ARMS[0],index=0,completed=False))
    with pytest.raises(ValueError,match='no candidate retry'):
        g(dict(kind='candidate_request',arm=m.native.ARMS[0],index=0,completed=False))


@pytest.mark.parametrize('damage',['finite','bool'])
def test_finite_unsupported_and_bool_rejected(damage):
    g,j=make_guard()
    work=m.C.vector(); work['flops' if damage=='finite' else 'predict_calls']=0 if damage=='finite' else True
    with pytest.raises(ValueError):g.reserve('invalid',work,{})


def test_durable_real_fsync_chain_exclusive_and_mutation(tmp_path):
    path=tmp_path/'events.jsonl'; j=m.DurableJournal(path,{'scope':'RAM metadata'},floor_bytes=0)
    j.append('reserve_intent',{'computation_observed':False});j.append('return_observed',{'returned':True});j.close()
    records=m.read_journal(path); assert len(records)==3
    with pytest.raises(FileExistsError):m.DurableJournal(path,{})
    # Mutate bytes in memory; read_journal sees mocked saved bytes, never rewrite old evidence.
    raw=path.read_bytes().replace(b'"returned":true',b'"returned":false')
    from unittest.mock import patch
    with patch.object(m.Path,'read_bytes',return_value=raw):
        with pytest.raises(ValueError,match='hash-chain'):m.read_journal(path)
    with patch.object(m.Path,'read_bytes',return_value=raw[:-1]):
        with pytest.raises(ValueError,match='tail'):m.read_journal(path)


def test_fsync_failure_poison_no_retry_and_no_next_callback(tmp_path,monkeypatch):
    j=m.DurableJournal(tmp_path/'events.jsonl',{},floor_bytes=0); before=j.seq
    def fail(fd):raise OSError('fsync failed')
    monkeypatch.setattr(m.os,'fsync',fail)
    with pytest.raises(OSError):j.append('intent',{})
    assert j.poisoned and j.seq==before
    with pytest.raises(ValueError,match='poisoned'):j.append('retry',{})
    j.close()


def test_byte_and_wall_limits_at_boundary(tmp_path):
    j=m.DurableJournal(tmp_path/'events.jsonl',{},floor_bytes=0,max_bytes=1024)
    with pytest.raises(ValueError,match='byte cap'):j.append('too_big',{'x':'z'*2000})
    assert j.poisoned;j.close()
    j=m.DurableJournal(tmp_path/'wall.jsonl',{},floor_bytes=0);j.started-=901
    with pytest.raises(ValueError,match='wall cap'):j.append('intent',{})
    j.close()


def test_metric_callback_entry_return_and_error_retained():
    g,j=make_guard(); a=m.native.ARMS[0]
    g(dict(kind='candidate_request',arm=a,index=0,completed=False)); called=[]
    def fail(c):called.append(c);raise RuntimeError('metric')
    with pytest.raises(RuntimeError):g.metric(fail)('context')
    assert called==['context'] and g.poisoned and g.snapshot()['actual_complete_totals'] is None
    assert [e['kind'] for e in j.events][-3:]==['operation_reserve_intent','entry_boundary','return_observed']


def test_confirmatory_denies_before_read(monkeypatch):
    monkeypatch.setattr(m.contract.prior,'read',lambda *a:pytest.fail('must not read'))
    with pytest.raises(ValueError,match='confirmatory prohibited'):m.main(['execute-known','--confirmatory'])
    with pytest.raises(ValueError,match='confirmatory prohibited'):
        m.execute_known(None,None,None,main_review_sha256=None,confirmatory=True)


def test_wrapper_import_and_kernel_binding_no_source_patch():
    assert m.file_sha(m.native.__file__)==m.SHA_NATIVE
    assert m.file_sha(m.contract.__file__)=='c577bad4e4ff0ce5b5eeba2d606a9472245f7ae80e400600e1fa4785dbf7e942'
    assert m.native.run_arm.__module__==m.native.__name__


def test_prepare_CLI_new_stem_actual_exclusive_writer(tmp_path,monkeypatch,capsys):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    fake=dict(schema='RAM_fixture',new_neural_calls=0,new_physics_steps=0)
    monkeypatch.setattr(m,'prepare',lambda **kw:fake)
    monkeypatch.setattr(m.contract.prior,'write',lambda *a:pytest.fail('OLD-prefix writer forbidden'))
    path=tmp_path/'results'/(m.NAME+'_prepared.json')
    argv=['prepare','--native-prepared','RAM','--native-sha','a'*64,'--output',str(path)]
    assert m.main(argv)==0
    assert m.file_sha(path)==m.C.digest(fake)
    assert json.loads(capsys.readouterr().out)['prepare_sha256']==m.C.digest(fake)
    with pytest.raises(FileExistsError):m.main(argv)
    with pytest.raises(ValueError,match='prefix under results'):
        m.write_metadata(tmp_path/(m.NAME+'_outside.json'),fake)


def test_run_known_root_really_attaches_to_original_entry_guard_before_compile(tmp_path,monkeypatch):
    old=m.native.legacy(); model,replay,calls=fake_model(); primary=RuntimeError('mock compile stopped')
    monkeypatch.setattr(old.full20,'RootAnchoredCudaReplay',lambda model,device:replay)
    monkeypatch.setattr(m.native,'array_binding',lambda *a:dict(scope='RAM'))
    called=[]
    def compile_mock(model,device,ready,compiled,r,journal,save,faults,stages):
        called.append('compile');r._p2_scope=m.native.expected_scopes()[0]
        model.action_encoder.forward(tensor('action_encoder'))
        raise primary
    monkeypatch.setattr(m.native,'compile_cuda',compile_mock)
    monkeypatch.setattr(m.native,'suffix_cuda',lambda *a:pytest.fail('no suffix on compile failure'))
    monkeypatch.setattr(m.native,'run_arm',lambda **kw:pytest.fail('no arm on compile failure'))
    ready=dict(root_index=0,root_sha256=m.native.ROOTS[0])
    row=m.run_known_root(tmp_path,{}, {},'a'*64,ready,{},model,NS(type='cuda'),lambda:called.append('sync'))
    assert called==['compile','sync'] and calls==['action_encoder']
    assert not row['complete'] and row['faults']['primary']['error']=='mock compile stopped'
    assert row['guard']['observed_returned_lower_bound_unique']['action_encoder_calls']==1
    assert sum(len(v) for v in row['guard']['candidates'].values())==120
    assert all(c['status']=='skipped' for cs in row['guard']['candidates'].values() for c in cs)
    records=m.read_journal(tmp_path/'root_0_guard.jsonl')
    assert [r['kind'] for r in records][:5]==['header','group_reserve_intent','operation_reserve_intent','entry_boundary','return_observed']
    assert model.action_encoder.forward(tensor('action_encoder')) is None  # Hook restored, not poisoned leftover.


def test_actual_saved_prepare_verify_no_execution():
    directory=m.ROOT/'results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01'
    sha=m.C.digest(m.native.read(directory/'prepared.json'))
    p=m.prepare(native_dir=directory,native_sha=sha)
    assert p['resources']['cohort_unique_predict_upper']==1328 and len(p['root_order'])==4
    assert not p['confirmatory'] and p['new_neural_calls']==p['new_physics_steps']==0
    assert m.verify(p,m.C.digest(p))==p
    changed=deepcopy(p);changed['resources']['full_resource_equal']=True
    with pytest.raises(ValueError,match='prepare SHA'):m.verify(changed,m.C.digest(p))
    with pytest.raises(ValueError,match='closure drift'):m.verify(changed,m.C.digest(changed))


def test_full_run_known_root_calls_UNCHANGED_run_arm_and_sink(tmp_path,monkeypatch):
    import numpy as np
    from test_p2_composition_predict_frontier_20261027_v1 import base,legacy_inputs
    from experiments.spectral_reachability import p2_matched_composition_20261027 as matched
    old,args,_,_=legacy_inputs();model,_,calls=fake_model(); (tmp_path/'arrays').mkdir()
    class Replay:
        def __init__(self,model,device):self.model=model;self._p2_scope=None
        def __call__(self,c,actions):
            self.model.action_encoder.forward(tensor('action_encoder'));self.model.predict(tensor('predict'))
            c=deepcopy(c);packet=np.asarray(actions).reshape(10).tolist()
            z=c['latent_history'][-1].copy();z[:10]=[a+b for a,b in zip(z[:10],packet)]
            c['latent_history']=c['latent_history'][1:]+[z];c['action_history']=c['action_history'][1:]+[packet]
            return c
    monkeypatch.setattr(old.full20,'RootAnchoredCudaReplay',Replay)
    monkeypatch.setattr(matched,'RootAnchoredCudaReplay',Replay)
    monkeypatch.setattr(m.native,'array_binding',lambda *a:{'scope':'RAM'})
    monkeypatch.setattr(m.native,'provenance',lambda *a:args['provenance'])
    monkeypatch.setattr(old.full20,'metric_for',lambda compiled:args['target_metric'])
    def compile_mock(model,device,ready,compiled,r,journal,save,faults,stages):
        for stage in ('nominal','joint_jacobian'):
            r._p2_scope=dict(arm=old.sra.ARM,phase='compilation',path=[stage])
            if stage=='joint_jacobian':journal(dict(kind='jacobian',completed=False,shape=[192,20],vectorize=True))
            for _ in range(2):r(deepcopy(ready['root']['context']),[[0.,0.]]*5)
            if stage=='joint_jacobian':journal(dict(kind='jacobian',completed=True,shape=[192,20],vectorize=True))
        for k in ('bounded_ridge_svd','mode_svd'):
            e=dict(kind=k,completed=False,svd_calls_requested=1);journal(e);journal(dict(e,completed=True))
        r._p2_scope=None;return base()
    def suffix_mock(model,device,r,context,prefix,compiled,index,journal,save,faults,stages):
        for stage in ('nominal','jacobian'):
            r._p2_scope=dict(arm=m.native.ARMS[1],phase='suffix_compilation',path=[index,stage])
            if stage=='jacobian':journal(dict(kind='suffix_jacobian',index=index,completed=False,shape=[192,10],vectorize=True))
            r(deepcopy(context),[[0.,0.]]*5)
            if stage=='jacobian':journal(dict(kind='suffix_jacobian',index=index,completed=True,shape=[192,10],vectorize=True))
        e=dict(kind='suffix_solve',index=index,completed=False,svd_calls_requested=1)
        journal(e);journal(dict(e,completed=True,svd_calls=1,bounded_solve_calls=1))
        r._p2_scope=None
        return dict(suffix_logits=[0.]*10,input_context_sha256=m.C.digest(context),svd_calls=1,bounded_solve_calls=1)
    monkeypatch.setattr(m.native,'compile_cuda',compile_mock);monkeypatch.setattr(m.native,'suffix_cuda',suffix_mock)
    real_run_arm=m.native.run_arm
    ready=dict(root_index=0,root_sha256=m.native.ROOTS[0],root=args['root'],action_domain=args['action_domain'])
    row=m.run_known_root(tmp_path,{}, {},'a'*64,ready,{},model,NS(type='cuda'),lambda:None)
    assert m.native.run_arm is real_run_arm  # Only compile/suffix metadata callbacks substituted.
    assert row['complete'],row['faults']
    assert len(calls)==664 and len(row['results'])==2
    assert [len(r['attempts']) for r in row['results']]==[80,40]
    assert row['guard']['actual_complete_totals']['predict_calls']==332
    assert len(m.read_journal(tmp_path/'root_0_guard.jsonl'))>2000
