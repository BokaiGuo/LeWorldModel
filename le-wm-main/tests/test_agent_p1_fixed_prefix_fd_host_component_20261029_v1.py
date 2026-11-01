"""RAM full native-ABI ledger tests; no production host or CUDA."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_fd_host_component_20261029_v1 as h
import test_agent_p1_fixed_prefix_saved_handoff_20261029_v1 as fixture
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as native

inputs=fixture.inputs
complete=fixture.complete
group=fixture.group
saved=fixture.saved
panel=fixture.panel


def test_native_group_saved_replay_and_tamper(saved):
    q,r,_,_=saved; journal=native.Events(); ledger=h.core.Ledger(h.F.caps(80,0),sink=journal.event)
    backend=native.RAM(q['task'],journal,ledger); result=h.fd.evaluate(q,r,backend,journal.event)
    audit=h.audit_group(q,r,result,ledger.receipt(),journal.events)
    assert audit['native_forwards']==80 and not audit['gradient_correctness_established']
    events=deepcopy(journal.events);next(e for e in events if e['kind']=='fd_return')['Q']+=1
    with pytest.raises(ValueError):h.audit_group(q,r,result,ledger.receipt(),events)


def test_group_failure_retains_all64_and_paid_ledger(panel):
    p,r=panel; logs=[]
    def journal(i):
        j=native.Events();logs.append(j);return j
    def fail(task,j,ledger):raise RuntimeError('injected before native')
    out=h.dispatch(p['groups'],r['rows'],fail,journal)
    assert out['status']=='partial_failclosed' and len(out['rows'])==64
    assert out['rows'][0]['status']=='failed_partial' and out['rows'][0]['resources'] is not None
    assert all(x['status']=='not_started' for x in out['rows'][1:])
    assert out['pair_denominator']==2560
