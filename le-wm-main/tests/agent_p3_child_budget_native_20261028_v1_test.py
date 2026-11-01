"""RAM callbacks and saved resources only. No torch, model, or physics."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_child_budget_native_20261028_v1 as m


@pytest.fixture(scope='module')
def prepared():
    p = m.compiler.base.read(m.ROOT/'results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json', 64*1024**2)
    return dict(children=p['children'], tasks=p['tasks'], limits=m.budget.inputs_template())


def forward(ledger, fault=None):
    ledger.call('action_encoder', lambda: None, shape=[1,3,10])
    def predict():
        ledger.call('predictor', lambda: ledger.call('pred_proj', lambda: None, shape=[3,192]), shape=[1,3,192])
        if fault: raise RuntimeError(fault)
    ledger.call('predict', predict, shape=[1,3,192])


class RAMFactories:
    kind = 'RAM_FIXTURE'
    def __init__(self, fault=None): self.fault=fault; self.calls=[]

    def signal(self, task, child, sink):
        self.calls.append(('signal',child['child_id']))
        owner=self
        class Signal:
            kind='RAM_FIXTURE'
            def signal(self, req, ledger):
                h=req['horizon']
                for _ in range(h): forward(ledger, owner.fault)
                def jacobian():
                    for _ in range(h): forward(ledger)
                    for _ in range(192):
                        ledger.call('vjp', lambda: ledger.call('grad', lambda: ledger.call('raw_backward', lambda: None)), directions=1)
                ledger.call('jacobian', jacobian)
                return dict(binding_sha256=req['binding_sha256'], parent_context_sha256=req['parent_context_sha256'],
                    horizon=h, nominal=deepcopy(task['goal']), jacobian=np.zeros((192,10*h)).tolist(),
                    producer_sha256=m.sha(m.paid.__file__))
        return Signal()

    def program(self, task, child, sink):
        self.calls.append(('program',child['child_id']))
        class Program:
            kind='RAM_FIXTURE'
            def replay(self, parent, actions, phase, ledger):
                context=deepcopy(parent); path=[]
                for packet in np.asarray(actions,np.float32).reshape(-1,5,2):
                    forward(ledger)
                    token=((packet-np.asarray(task['action_mean'],np.float32))/np.asarray(task['action_std'],np.float32)).reshape(10)
                    context=dict(context, latent_history=context['latent_history'][1:]+[context['latent_history'][-1]],
                        action_history=[context['action_history'][-1], token.astype(float).tolist()])
                    path.append(deepcopy(context))
                return path
        return Program()


def test_complete_signal_chooser_compiler_hooks_and_independent_payment(prepared):
    events=[]; session=m.Session(prepared, events.append); backend=RAMFactories()
    session.next(backend, permit_ram=True)
    session.next(backend, permit_ram=True)
    assert len(backend.calls)==4  # independent signal AND compiler for BOTH methods
    assert all(r['status']=='budget_complete_native_unverified' for r in session.rows[:2])
    for row in session.rows[:2]:
        resources=row['signal_resources']
        assert resources['attempted']==resources['completed']==m.contract.signal_caps(1)
        m.core.audit_ledger(resources)
        m.core.audit_ledger(row['compiler']['resource_receipt'])
    receipt=session.receipt()
    assert len(receipt['rows'])==48 and receipt['child_denominator']==24
    assert receipt['matched_actual_vector'] is False and receipt['native_verified'] is False
    m.budget.consume_gate(receipt['gate'])


def test_actual_exception_retains_attempts_no_switch_no_new_callback(prepared):
    session=m.Session(prepared, lambda e:None); backend=RAMFactories('RAM predict fault')
    with pytest.raises(RuntimeError, match='RAM predict fault'): session.next(backend, permit_ram=True)
    assert session.gate.actual['fixed']['attempted']['predict_API']==1
    assert session.gate.actual['fixed']['completed']['predict_API']==0
    assert session.gate.pending and session.gate.failed
    with pytest.raises(ValueError,match='fail-stop'): session.next(backend, permit_ram=True)
    assert len(backend.calls)==1 and len(session.receipt()['rows'])==48
    m.budget.consume_gate(session.gate.receipt())


@pytest.mark.parametrize('shape', [[1,3,191],[2,3,192],[1,2,192]])
def test_bad_actual_shape_is_refused_before_callable(prepared, shape):
    session=m.Session(prepared, lambda e:None); c=prepared['children'][0]
    key=session.gate.open(c['child_id'],'fixed',c['context_sha256'],c['ancestry']['task_binding_sha256'])['ticket']
    sink=m.GateSink(session.gate,key,lambda e:None)
    ledger=m.core.Ledger(m.contract.signal_caps(1),sink=sink.event); ledger.scope={'RAM':True}
    called=[]
    with pytest.raises(ValueError): ledger.call('predict',lambda:called.append(1),shape=shape)
    assert called==[] and session.gate.actual['fixed']['attempted']['predict_API']==0


def test_journal_failure_is_predispatch_and_nonrefundable(prepared):
    session=m.Session(prepared,lambda e:None); c=prepared['children'][0]
    key=session.gate.open(c['child_id'],'fixed',c['context_sha256'],c['ancestry']['task_binding_sha256'])['ticket']
    def fail(e): raise OSError('RAM disk fault')
    sink=m.GateSink(session.gate,key,fail)
    ledger=m.core.Ledger(m.contract.signal_caps(1),sink=sink.event); ledger.scope={'RAM':True}
    called=[]
    with pytest.raises(OSError): ledger.call('predict',lambda:called.append(1),shape=[1,3,192])
    assert called==[] and session.gate.pending and session.gate.failed
    assert session.gate.actual['fixed']['attempted']['predict_API']==1
    m.budget.consume_gate(session.gate.receipt())


def replay_ledger(ledger, saved):
    """RAM no-op callbacks reproduce the real saved resource nesting only."""
    entries=saved['events']; cursor=0
    def one():
        nonlocal cursor
        event=entries[cursor]; cursor+=1
        def fn():
            while cursor<len(entries) and entries[cursor]['started_ns']<event['ended_ns']: one()
        ledger.call(event['resource'],fn,shape=event['shape'],directions=event['directions'])
    while cursor<len(entries): one()


def test_all48_saved_resource_shape_chooser_census_without_backend(prepared):
    signals=m.compiler.base.read(m.compiler.SIGNAL_RECEIPT,256*1024**2)
    compiled=m.compiler.base.read(m.budget.STRICT,128*1024**2)
    gate=m.budget.WholeNodeGate(prepared['limits'],prepared['children'],prepared['tasks'])
    census={'finished':0,'refused':0,'not_reached':0}
    for i,row in enumerate(compiled['rows']):
        c=prepared['children'][i//2]; mode=m.contract.MODES[i%2]
        opened=gate.open(c['child_id'],mode,c['context_sha256'],c['ancestry']['task_binding_sha256'],observed=c['status']=='observed_child')
        if not opened['dispatch_allowed']:
            assert opened['reason']=='not_reached'; census['not_reached']+=1; continue
        key=opened['ticket']; acquisition=signals['rows'][i//2]['result']['acquisitions'][i%2]
        sink=m.GateSink(gate,key,lambda e:None)
        ledger=m.core.Ledger(m.contract.signal_caps(1),sink=sink.event); ledger.scope={'RAM_saved_trace':True}
        replay_ledger(ledger,acquisition['resource_receipt'])
        signal=acquisition['signal']['scheduling_signal']
        decision=m.core.choose(prepared['tasks'][c['root_slot']],dict(context=c['context'],depth=0),m.core.FrozenSignals(**signal),mode)
        sealed=gate.seal(key,decision,signal)
        if not sealed['dispatch_allowed']:
            census['refused']+=1; continue
        ledger=m.core.Ledger(row['resource_receipt']['caps'],sink=m.GateSink(gate,key,lambda e:None).event)
        ledger.scope={'RAM_saved_trace':True}; replay_ledger(ledger,row['resource_receipt']); gate.finish(key)
        census['finished']+=1
    assert census==dict(finished=41,refused=5,not_reached=2) and gate.cursor==48
    assert gate.actual['fixed']!=gate.actual['adaptive']
    for mode_index,mode in enumerate(m.contract.MODES):
        for kind in ('attempted','completed'):
            expected=m.budget.v1.zero()
            for i,c in enumerate(prepared['children']):
                if c['status']=='not_reached': continue
                a=signals['rows'][i]['result']['acquisitions'][mode_index]
                expected=m.budget.v1.add(expected,a['resource_receipt'][kind])
                work=compiled['rows'][2*i+mode_index]['resource_receipt']
                if work is not None: expected=m.budget.v1.add(expected,work[kind])
            assert gate.actual[mode][kind]==expected


def test_no_authorization_or_cpu_before_any_native_access(prepared):
    with pytest.raises(ValueError,match='authorization'): m.NativeFactories(None,None,prepared['tasks'])
    with pytest.raises(ValueError,match='CUDA only'):
        m.NativeFactories(None,SimpleNamespace(type='cpu'),prepared['tasks'],authorization='MAIN_APPROVED_BUDGET_CHILD_CUDA')
    backend=RAMFactories(); session=m.Session(prepared,lambda e:None)
    with pytest.raises(ValueError,match='authorization'): session.next(backend)
    assert backend.calls==[] and session.gate.cursor==0


def test_saved_prepare_requires_full_consumer_and_source_pins():
    source=open(m.__file__,encoding='utf-8').read()
    assert 'budget.load_saved()' in source and 'import torch' not in source
    m.contract.check_sources(m.PINS)


def test_material_binds_complete_saved_consumer_not_summary(monkeypatch, prepared):
    calls=[]
    fake=dict(prepared,sources={})
    monkeypatch.setattr(m.budget,'load_saved',lambda: calls.append('full_consume') or
                        (fake,dict(rows=[None]*48),None))
    out=m.material()
    assert calls==['full_consume']
    assert out['fixed_denominator']==48 and len(out['children'])==24
    assert out['runtime_dispatch_allowed'] is False
    assert out['sources'][str(m.budget.STRICT)]==m.budget.STRICT_SHA
    assert out['sources'][str(m.compiler.SIGNAL_RECEIPT)]==m.compiler.SIGNAL_RECEIPT_SHA


def test_source_tamper_precedes_full_consumer(monkeypatch):
    calls=[]
    monkeypatch.setattr(m,'PINS',{str(m.Path(m.__file__).resolve()):'0'*64})
    monkeypatch.setattr(m.budget,'load_saved',lambda: calls.append('BAD'))
    with pytest.raises(ValueError,match='drift'): m.material()
    assert calls==[]
