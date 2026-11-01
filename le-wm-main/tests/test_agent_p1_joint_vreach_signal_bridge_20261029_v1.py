from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from experiments.spectral_reachability import agent_p1_joint_vreach_signal_bridge_20261029_v1 as bridge


@pytest.fixture(scope='module')
def call():
    path = Path(__file__).with_name('test_agent_p3_parent_native_cuda_20261029_v1.py')
    fixture = runpy.run_path(str(path))
    task = fixture['material']()['tasks'][0]
    events = []
    store = fixture['Memory']()
    store.events = events
    ledger = bridge.core.Ledger(dict.fromkeys(bridge.core.AXES, 10000), sink=lambda x: events.append(deepcopy(x)))
    ledger.scope = {'fixture': True}
    req = bridge.core.request(task, dict(context=task['root_context']), operation='compile_finite_horizon_J', horizon=1)
    events.append(dict(kind='native_call_attempt', method='signal', task=deepcopy(task), request=deepcopy(req)))
    result = fixture['NumpyNative'](task, store).signal(req, ledger)
    events.append(dict(kind='native_call_return', result=result))
    return task, events, ledger


def test_complete_signal_and_actual_resources(call):
    task, events, ledger = call
    out = bridge.from_transcript(task, task['root_context'], events)
    assert out['nominal_context']['latent_history'][-1] == events[-1]['result']['nominal']
    assert out['jacobian'] == events[-1]['result']['jacobian']
    assert ledger.attempted['predict_API'] == 2
    assert ledger.attempted['VJP_directions'] == ledger.attempted['grad_API'] == 192
    assert ledger.attempted == ledger.completed


@pytest.mark.parametrize('damage', ['parent', 'boundary', 'jacobian', 'failure'])
def test_refuses_mismatched_or_incomplete_native(call, damage):
    task, events, _ = call
    task, events = deepcopy(task), deepcopy(events)
    context = deepcopy(task['root_context'])
    if damage == 'parent':
        context['latent_history'][0][0] += .25
    elif damage == 'boundary':
        e = next(e for e in events if e['kind'] == 'boundary' and e['phase'] == 'nominal')
        a = bridge.native.signal.unblob(e['arrays']['latent']).copy(); a.flat[0] += .25
        e['arrays']['latent'] = bridge.native.signal.blob(a)
    elif damage == 'jacobian':
        events[-1]['result']['jacobian'][0][0] += .25
    else:
        events[-1] = dict(kind='native_call_failure', error={'type': 'RuntimeError', 'message': 'fixture'})
    with pytest.raises(ValueError):
        bridge.from_transcript(task, context, events)
