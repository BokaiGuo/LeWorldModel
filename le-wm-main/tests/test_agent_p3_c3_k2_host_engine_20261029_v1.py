"""RAM seam tests only; no model, CUDA, production or simulator."""
import ast
import inspect
import runpy
import pytest
from experiments.spectral_reachability import agent_p3_c3_k2_host_engine_20261029_v1 as m

f=runpy.run_path(str(m.core.ROOT/'tests/test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))
Memory,Native,task=f['Memory'],f['Native'],f['task']


def test_host_gate_requires_explicit_authority_without_dispatch():
    with pytest.raises(ValueError,match='explicit host authorization'):
        m.run_tree(task(),None,None,mode='fixedH1',K=2,capsule=lambda *a:None)


def test_engine_ast_matches_frozen_k2_with_only_gate_changes():
    original=ast.parse(inspect.getsource(m.k2.run_tree)).body[0]
    new=ast.parse(inspect.getsource(m.run_tree)).body[0]
    assert [x.arg for x in new.args.args] == [x.arg for x in original.args.args]
    assert [x.arg for x in new.args.kwonlyargs] == [x.arg for x in original.args.kwonlyargs] + ['authorization']
    # Compare the substantive loop after the two guards.
    assert ast.dump(ast.Module(body=new.body[2:],type_ignores=[]),include_attributes=False)==\
           ast.dump(ast.Module(body=original.body[2:],type_ignores=[]),include_attributes=False)


def test_real_shaped_ram_saved_reconstruction():
    t=task();s=Memory();sink=type('Sink',(),{'event':lambda self,e:s.event(e)})()
    r=m.run_tree(t,lambda t,b:Native(t,b),sink,mode='fixedH1',K=2,permit_ram=True,kind='RAM_FIXTURE',
                 capsule=lambda cid,call:s.capsule(0,cid,call))
    assert len(r['jobs'])==144 and r['resource_receipt']['caps']==m.caps()
    result=m.validate_tree(t,r,s.events,permit_fixture=True,capsule_reader=lambda cid,call:s.reader(0,cid,call))
    assert result['status']==r['status']


def test_k2_axis_caps_are_dominated_by_old_transport_roster():
    # Necessary resource-count premise only: this is NOT a proof that a new
    # six-slot serializer has the old transport's bounded metadata overhead.
    from experiments.spectral_reachability import agent_p3_c3_compact_host_cuda_20261029_v1 as host
    old = host.original
    assert len(m.k2.roster()) == 6 and len(old.roster()) == 12
    for axis in m.core.AXES:
        assert sum(m.k2.caps()[axis] for _ in m.k2.roster()) <= sum(
            old.caps(row['K'])[axis] for row in old.roster())
    proof = host.proof()
    assert proof['live_bytes_upper'] <= proof['cap'] - proof['failure_reserve']
