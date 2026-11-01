from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p1_two_packet_native_audit_20261029_v1 as m
import test_agent_p1_two_packet_uphill_generator_20261029_v1 as fixture
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as native_fixture

inputs=fixture.inputs
complete=fixture.complete
requests=fixture.requests


@pytest.fixture(scope='function')
def paid(requests):
    q=requests[0];events=native_fixture.Events();ledger=m.core.Ledger(m.F.caps(36,16),sink=events.event)
    backend=m.LabelAdapter(native_fixture.RAM(q['task'],events,ledger))
    result=m.method.run_arm(q,'stage_ascent',backend,events.event)
    return q,result,ledger.receipt(),events.events


def test_full_native_paid_arm_rebuild(paid):
    q,r,l,e=paid;out=m.audit(q,r,l,e)
    assert out['native_forwards']==36 and out['scalar_reverse']==16
    assert l['attempted']==l['completed']==m.F.caps(36,16)


@pytest.mark.parametrize('bad',['native_gradient','native_call','control','result','ledger'])
def test_corruption_rejected(paid,bad):
    q,r,l,e=deepcopy(paid)
    if bad=='native_gradient':
        event=next(x for x in e if x['kind']=='raw_backward');a=m.F.signal.unblob(event['arrays']['value']).copy();a.flat[0]+=.1
        event['arrays']['value']=m.F.signal.blob(a)
    elif bad=='native_call':e.remove(next(x for x in e if x['kind']=='program_packet'))
    elif bad=='control':next(x for x in e if x['kind']=='score_intent')['iteration']=7
    elif bad=='result':r['iterates'][1]['terminal_Q']+=1
    else:l['completed']['predict_API']-=1
    with pytest.raises((ValueError,KeyError,StopIteration)):m.audit(q,r,l,e)
