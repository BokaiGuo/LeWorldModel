from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import runpy

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p1_joint_vreach_native_callback_20261029_v1 as m


@pytest.fixture(scope='module')
def material():
    f = runpy.run_path(str(Path(__file__).with_name('test_agent_p3_parent_native_cuda_20261029_v1.py')))
    return f, f['material']()['tasks'][0]


def test_actual_ram_native_192_and_distinct_parent(material):
    f, task = material; events = []
    context = deepcopy(task['root_context']); context['latent_history'][-1][2] = float(np.float32(1e-8))
    def factory(local, sink):
        return m.components.Components.ram(root=local, caps=m.components.expected('postcontext_H1'),
            verification_reserve=dict.fromkeys(m.core.AXES, 0), factory=lambda t,s:f['NumpyNative'](t,s), sink=sink)
    callback = m.NativeCallbacks(task, None, factory, events.append)
    out = callback.signal(context, 'candidate0')
    receipt = events[-1]['receipt']
    assert receipt['resources']['predict_API'] == 2
    assert receipt['resources']['VJP_directions'] == receipt['resources']['grad_API'] == 192
    assert len(out['jacobian']) == 192 and len(out['jacobian'][0]) == 10
    assert events[0]['root_task_sha256'] == m.core.digest(task)
    assert events[0]['parent_context_sha256'] == m.core.digest(context)
    assert receipt['local_task']['root_context'] == context


def test_native_failure_keeps_partial_receipt(material):
    f, task = material; events = []
    def factory(local, sink):
        return m.components.Components.ram(root=local, caps=m.components.expected('postcontext_H1'),
            verification_reserve=dict.fromkeys(m.core.AXES, 0), factory=lambda t,s:f['NumpyNative'](t,s,fail=True), sink=sink)
    with pytest.raises(m.components.ComponentFailure):
        m.NativeCallbacks(task, None, factory, events.append).signal(task['root_context'], 'bad')
    failure = events[-1]
    assert failure['kind'] == 'joint_signal_failure' and failure['receipt']['status'] == 'failed'
    assert failure['receipt']['resources']['predict_API'] == 1
    assert any(e['kind'] == 'joint_signal_event' for e in events)


def test_scalar_scopes_are_separate_and_gradient_label_auditable(material):
    _, task = material
    class Scalar:
        def __init__(self): self.active = False; self.seen = []
        @contextmanager
        def observe(self):
            assert not self.active
            self.active = True
            try: yield
            finally: self.active = False
        def step(self, c, a, label):
            assert self.active; self.seen.append(label); return c
        def gradient(self, c, a, label):
            assert self.active; self.seen.append(label); return c
    scalar = Scalar(); b = m.NativeCallbacks(task, scalar, lambda *_: None, lambda _: None)
    b.step(task['root_context'], [], 'prefix')
    b.gradient(task['root_context'], [], 's0_i0_stage_gradient')
    assert scalar.seen == ['prefix', 'gradient_s0_i0_stage_gradient']
    assert not scalar.active
