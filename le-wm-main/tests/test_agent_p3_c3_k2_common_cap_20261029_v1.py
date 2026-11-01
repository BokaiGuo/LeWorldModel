"""Real-shaped RAM callbacks only; no CUDA, physics or saved production."""
from copy import deepcopy
import ast
import inspect
import runpy
import pytest
from experiments.spectral_reachability import agent_p3_c3_k2_common_cap_20261029_v1 as m

f=runpy.run_path(str(m.core.ROOT/'tests/test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))
Memory,Native,task=f['Memory'],f['Native'],f['task']


def run(mode, fail=False):
    t=task();store=Memory();sink=type('Sink',(),{'event':lambda self,e:store.event(e)})()
    r=m.run_tree(t,lambda t,s:Native(t,s,fail=fail),sink,mode=mode,K=2,permit_ram=True,kind='RAM_FIXTURE',
                 capsule=lambda cid,call:store.capsule(0,cid,call))
    return t,r,store


def test_fixed_six_roster_and_complete_vector():
    p=m.protocol();assert len(p['roster'])==6 and len(p['common_caps'])==27
    assert p['HOST_dispatch_authorized'] is False and p['equal_FLOPs'] is False
    assert m.caps()['predict_API']==142 and m.caps()['jacobian_API']==22 and m.caps()['replay_calls']==38
    assert m.caps()['predict_context_tokens']==426 and m.caps()['VJP_directions']==4224
    assert all(x['K']==2 for x in p['roster'])


def test_original_atomic_loop_ast_has_only_declared_changes():
    old=ast.parse(inspect.getsource(m.original.run_tree)).body[0]
    new=ast.parse(inspect.getsource(m.run_tree)).body[0]
    new.body=new.body[2:]  # K2/RAM-only guard and frozen source check
    target=next(x for x in new.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='ledger')
    target.value.args[0]=ast.parse('caps(K)',mode='eval').body
    assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)


@pytest.mark.parametrize('mode',m.MODES)
def test_real_callback_queue_and_saved_rebuild(mode):
    t,r,s=run(mode);assert len(r['jobs'])==144 and r['resource_receipt']['caps']==m.caps()
    assert not r['errors'],r['errors']
    assert all(q['status']!='queued' or len(q['requests']) in (4,8) for q in r['queues'])
    assert sum(j['request'] is not None for j in r['jobs'])==sum(len(q['requests']) for q in r['queues'] if q['status']=='queued')
    if r['status']=='resource_cap_before_node_queue':assert r['model_plan'] is None and r['queues'][-1]['status']==r['status']
    assert all(a<=m.caps()[k] for k,a in r['resource_receipt']['attempted'].items())
    assert r['model_plan'] is None or r['status']=='tree_complete_not_admitted'
    summary=m.validate_tree(t,r,s.events,permit_fixture=True,capsule_reader=lambda cid,call:s.reader(0,cid,call))
    assert summary['status']==r['status']
    bad=deepcopy(r);bad['resource_receipt']['caps']['predict_API']+=1
    with pytest.raises(ValueError):m.validate_tree(t,bad,s.events,permit_fixture=True,capsule_reader=lambda cid,call:s.reader(0,cid,call))


def test_no_host_dispatch_and_no_budget_replacement():
    with pytest.raises(ValueError):m.run_tree(task(),None,None,mode='fixedH1',K=2,kind='HOST_CUDA',capsule=lambda *a:None)
    with pytest.raises(ValueError):m.run_tree(task(),None,None,mode='fixedH1',K=1,permit_ram=True,kind='RAM_FIXTURE',capsule=lambda *a:None)


def test_failure_does_not_refund_or_release_plan():
    _,r,_=run('fixedH1',fail=True)
    assert r['status']=='failed_or_budget_exhausted' and r['model_plan'] is None
    assert r['resource_receipt']['attempted']['predict_API']>r['resource_receipt']['completed']['predict_API']


def test_six_slot_failure_retains_full_denominator():
    stores=[Memory() for _ in range(6)]
    def sink_for(i):return type('Sink',(),{'event':lambda self,e:stores[i].event(e)})()
    rows=m.run_panel_ram([task(),task()],lambda t,s:Native(t,s,fail=True),sink_for,
                         lambda i:lambda cid,call:stores[i].capsule(i,cid,call))
    assert len(rows)==6 and rows[0]['status']=='failed_or_budget_exhausted'
    assert all(x['status']=='blocked_after_runtime_failure' and x['result'] is None for x in rows[1:])
