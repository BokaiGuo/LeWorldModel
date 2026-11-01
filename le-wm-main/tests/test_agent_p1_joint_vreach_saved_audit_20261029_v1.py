from copy import deepcopy
from pathlib import Path
import runpy

import pytest
import test_agent_p1_two_packet_uphill_generator_20261029_v1 as source
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as scalar_fixture

from experiments.spectral_reachability import agent_p1_joint_vreach_saved_audit_20261029_v1 as m
from experiments.spectral_reachability import agent_p1_joint_vreach_native_callback_20261029_v1 as callback

inputs=source.inputs
complete=source.complete
requests=source.requests


@pytest.fixture
def paid(requests):
    fixture=runpy.run_path(str(Path(__file__).with_name('test_agent_p3_parent_native_cuda_20261029_v1.py')))
    q=requests[0]; events=scalar_fixture.Events()
    ledger=m.core.Ledger(m.F.caps(98,16), sink=events.event)
    scalar=scalar_fixture.RAM(q['task'], events, ledger)
    def factory(task,sink):
        return m.component.Components.ram(root=task, caps=m.component.expected('postcontext_H1'),
            verification_reserve=dict.fromkeys(m.core.AXES,0), factory=lambda t,s:fixture['NumpyNative'](t,s), sink=sink)
    backend=callback.NativeCallbacks(q['task'],scalar,factory,events.event)
    result=m.method.run_arm(q,m.method.ARMS[0],backend,events.event)
    return q,result,ledger.receipt(),events.events


def test_complete_arm_native_and_replay(paid):
    q,result,receipt,events=paid
    actual=m.audit(q,m.method.ARMS[0],result,receipt,events)
    assert actual['candidates']==20 and actual['native_forwards']==138 and actual['jacobian_VJP']==3840


@pytest.mark.parametrize('damage',['baseline','objective','signal'])
def test_mutated_result_and_native_refused(paid, damage):
    q,result,receipt,events=deepcopy(paid)
    if damage=='baseline': result['baseline_terminal_Q']+=1
    elif damage=='objective':
        ev=next(x for x in events if x['kind']=='io_input' and 'grad:loss' in x['arrays'])
        arr=m.F.signal.unblob(ev['arrays']['grad:loss']).copy(); arr.flat[0]+=1
        ev['arrays']['grad:loss']=m.F.signal.blob(arr)
    else:
        ev=next(x for x in events if x['kind']=='joint_signal_return')
        ev['receipt']['result']['jacobian'][0][0]+=1
    with pytest.raises((ValueError,KeyError)):
        m.audit(q,m.method.ARMS[0],result,receipt,events)
