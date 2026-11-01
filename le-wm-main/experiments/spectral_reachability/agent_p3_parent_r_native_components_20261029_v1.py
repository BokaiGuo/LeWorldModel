"""Bank-independent postcontext H1/suffix components. No production runner."""
from __future__ import annotations

from copy import deepcopy
import argparse
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_parent_native_adapter_20261029_v1 as parent
from experiments.spectral_reachability import agent_p3_parent_r_bank_evaluator_20261029_v1 as kernel

core = parent.core
require, digest = core.require, core.digest
NAME = 'agent_p3_parent_r_native_components_20261029_v1'
PINS = {
    'agent_p3_parent_native_cuda_20261029_v1.py': 'eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b',
    'agent_p3_local_signal_cuda_20261028_v1.py': '6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b',
    'agent_p3_parent_native_adapter_20261029_v1.py': '150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558',
    'agent_p3_parent_r_bank_evaluator_20261029_v1.py': '71e4e178eb180a0d9b740697c98538d6f9afd0ba50d35efb03aa23f91378e04d',
}


def contract():
    return dict(schema=NAME+'.contract', operations=['postcontext_H1', 'bounded_suffix_H1'],
        bank_selected=False, eligible_beam_defined=False, production_runner=False,
        original_R_equivalence=False, P1_admission=False, full_C3_demonstrated=False,
        RAM_HOST_separate=True, extra_nominal_replay=False,
        suffix_rule=kernel.contract()['suffix_chart'], unknowns=dict(FLOPs=None, cuda_ns=None,
        backward_engine_calls=None), source_pins=PINS)


def check_sources():
    for name, expected in PINS.items():
        require(core.sha(Path(__file__).with_name(name)) == expected, 'sealed component source: '+name)
    require(core.sha(kernel.original.__file__) == kernel.SOLVER_SHA256, 'original bounded solver source')


def suffix_raw(params):
    """Return pre-normalization raw only; NativeTree normalizes exactly once."""
    a = core.finite(params, (10,))
    raw = np.tanh(a.astype(np.float64)).astype(np.float32).reshape(5, 2)
    require(np.isfinite(raw).all() and np.max(np.abs(raw)) <= 1, 'bounded float32 raw')
    return raw.tolist()


def expected(operation):
    require(operation in ('postcontext_H1', 'bounded_suffix_H1'), 'known component operation')
    v = parent.progress.predict_axes(2 if operation == 'postcontext_H1' else 1)
    if operation == 'postcontext_H1':
        v.update(signal_compile_calls=1, jacobian_API=1, VJP_directions=192,
            grad_API=192, raw_backward_callbacks=192)
    else:
        v.update(replay_calls=1)
    return v


def post_task(root, actions, path):
    root = core.validate_task(deepcopy(root))
    require(root['root_context']['latent_dim'] == 192 and
        np.array_equal(root['projection'], np.eye(192)), 'identity192 root')
    if actions:
        c = core.checked_path(root, root['root_context'], actions, path)
    else:
        require(path == [], 'empty prefix path')
        c = root['root_context']
    return parent.parent_task(root, c)


class ComponentFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = deepcopy(receipt)


class _Sink:
    def __init__(self, owner):
        self.owner = owner

    def event(self, value):
        self.owner._event('native', value)


class Components:
    """Use ram() OR host(); HOST never accepts an injected fixture factory.

    guard() must enforce MAIN's frozen model/precision/source closure; this
    component does not load a model, authorize a bank, or certify that guard.
    """
    def __init__(self):
        raise TypeError('use explicit ram() or host() constructor')

    @classmethod
    def _new(cls, root, caps, reserve, sink, kind):
        check_sources()
        obj = object.__new__(cls)
        obj.root = core.validate_task(deepcopy(root))
        obj.root_sha256 = digest(obj.root)
        post_task(obj.root, [], [])
        obj.kind, obj.events, obj.calls = kind, [], []
        obj.sink = sink
        require(callable(sink), 'explicit event sink')
        obj.reserve = core.counters(reserve)
        obj.ledger = core.Ledger(caps, sink=lambda e: obj._event('resource', e))
        obj.busy = False
        obj.failed = False
        return obj

    @classmethod
    def ram(cls, *, root, caps, verification_reserve, factory, sink):
        obj = cls._new(root, caps, verification_reserve, sink, 'RAM_FIXTURE')
        require(callable(factory), 'RAM fixture factory')
        obj.factory = factory
        obj.guard = lambda: None
        obj.audit = None
        return obj

    @classmethod
    def host(cls, *, root, caps, verification_reserve, model, device, sink,
             main_authorized=False, guard=None):
        require(main_authorized is True and callable(guard), 'explicit MAIN authorization and guard')
        require(getattr(device, 'type', None) == 'cuda', 'HOST explicit CUDA only')
        # No torch import or model operation occurs until MAIN calls a primitive.
        from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as host
        obj = cls._new(root, caps, verification_reserve, sink, 'HOST_CUDA')
        obj.factory = lambda task, store: host.NativeTree(model, device, task, store)
        obj.guard, obj.audit = guard, host.audit_native
        return obj

    def _event(self, channel, payload):
        event = dict(schema=NAME+'.event', index=len(self.events), call_id=len(self.calls),
            channel=channel, payload=deepcopy(payload))
        self.events.append(event)
        self.sink(deepcopy(event))

    def _run(self, operation, actions, path, params=None):
        require(not self.busy and not self.failed, 'nonrecursive fail-stop components')
        require(digest(self.root) == self.root_sha256, 'immutable original root authority')
        check_sources()
        task = post_task(self.root, actions, path)
        cost = expected(operation)
        receipt = dict(schema=NAME+'.call', call_id=len(self.calls), kind=self.kind, operation=operation,
            root_task_sha256=digest(self.root), root_task=deepcopy(self.root), local_task=task,
            prefix_actions=deepcopy(actions), prefix_path=deepcopy(path), expected_resources=cost,
            verification_reserve=deepcopy(self.reserve), status='not_started', result=None,
            native_closure_verified=False, rule=contract(), events=[], used_before=deepcopy(self.ledger.attempted))
        begin = len(self.events)
        self.busy = True
        old_scope = self.ledger.scope
        self.ledger.scope = dict(component=NAME, call_id=len(self.calls), operation=operation,
            root_task_sha256=digest(self.root), postcontext_sha256=digest(task['root_context']))
        try:
            require(all(self.ledger.attempted[k]+cost[k]+self.reserve[k] <= self.ledger.caps[k]
                for k in core.AXES), 'whole component plus verification reserve before dispatch')
            receipt['guard_before'] = deepcopy(self.guard())
            backend = self.factory(deepcopy(task), _Sink(self))
            require(backend.kind == self.kind, 'RAM/HOST backend kind mismatch')
            if operation == 'postcontext_H1':
                req = core.request(task, dict(context=task['root_context']), operation='compile_finite_horizon_J', horizon=1)
                method, resource = 'signal', 'signal_compile'
            else:
                raw = suffix_raw(params)
                receipt['params_float64'] = np.asarray(params, np.float64).tolist()
                receipt['raw_float32'] = raw
                req = dict(parent_context=deepcopy(task['root_context']), actions=raw, segmented=False)
                method, resource = 'replay', 'replay'

            def invoke():
                start = len(self.events)
                self._event('native', dict(kind='native_call_attempt', method=method, task=task, request=req))
                try:
                    if method == 'signal':
                        out = backend.signal(deepcopy(req), self.ledger)
                    else:
                        out = backend.replay(deepcopy(req['parent_context']), deepcopy(raw), False, self.ledger)
                    self._event('native', dict(kind='native_call_return', result=deepcopy(out)))
                except Exception as exc:
                    self._event('native', dict(kind='native_call_failure', error=dict(type=type(exc).__name__, message=str(exc))))
                    raise
                # The scalar outer signal_compile/replay event is NOT part of
                # the existing inner native audit; all nested ledger IO is.
                transcript = [deepcopy(e['payload']) for e in self.events[start:]]
                if self.audit is not None:
                    require(self.audit(transcript) is True, 'full original native IO/resource closure')
                    receipt['native_closure_verified'] = True
                receipt['native_transcript'] = transcript
                return out

            out = self.ledger.call(resource, invoke)
            receipt['guard_after'] = deepcopy(self.guard())
            check_sources()
            actual = {k:self.ledger.attempted[k]-receipt['used_before'][k] for k in core.AXES}
            require(actual == cost, 'complete component resource callback census')
            if method == 'signal':
                core.keys(out, ('binding_sha256','parent_context_sha256','horizon','nominal','jacobian','producer_sha256'))
                require(out['binding_sha256'] == req['binding_sha256'] and
                    out['parent_context_sha256'] == req['parent_context_sha256'] and
                    type(out['horizon']) is int and out['horizon'] == 1, 'actual postcontext H1 identity')
                core.hash_value(out['producer_sha256'])
                core.finite(out['nominal'], (192,), True)
                core.finite(out['jacobian'], (192,10), True)
                boundaries = [e for e in receipt['native_transcript'] if e['kind']=='boundary' and e.get('phase')=='nominal']
                require(len(boundaries)==1 and boundaries[0]['step']==0, 'one saved nominal H1 boundary')
                from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as signal
                b = boundaries[0]
                core.keys(b['arrays'], ('latent','history'))
                latent, history = (signal.unblob(b['arrays'][k]) for k in ('latent','history'))
                require(latent.shape==(1,3,192) and history.shape==(1,2,10), 'actual full nominal boundary ABI')
                nominal_context = dict(task['root_context'], latent_history=latent[0].tolist(), action_history=history[0].tolist())
                core.checked_path(task, task['root_context'], [[0.,0.]]*5, [nominal_context])
                require(nominal_context['latent_history'][-1] == out['nominal'], 'nominal boundary equals response')
                receipt['result'] = dict(nominal_path=[nominal_context], jacobian=deepcopy(out['jacobian']),
                    input_context_sha256=digest(task['root_context']), jacobian_chart=kernel.contract()['jacobian_chart'])
                receipt['nominal_boundary_pointer'] = dict(call_id=receipt['call_id'],
                    transcript_index=receipt['native_transcript'].index(b), boundary_sha256=digest(b))
                receipt['derived_resources'] = dict(AD_primal_predict_API=1, Jacobian_elements=1920,
                    scalar_VJP_directions=192, native_verified=self.kind=='HOST_CUDA' and receipt['native_closure_verified'])
            else:
                core.checked_path(task, task['root_context'], raw, out)
                receipt['result'] = dict(path=deepcopy(out))
            receipt['status'] = 'complete'
        except Exception as exc:
            self.failed = True
            receipt.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)))
        finally:
            self.ledger.scope = old_scope
            self.busy = False
            receipt['events'] = deepcopy(self.events[begin:])
            receipt['used_after'] = deepcopy(self.ledger.attempted)
            receipt['resources'] = {k:self.ledger.attempted[k]-receipt['used_before'][k] for k in core.AXES}
            self.calls.append(deepcopy(receipt))
        if receipt['status'] == 'failed':
            raise ComponentFailure(receipt['error']['message'], receipt)
        return receipt

    def h1(self, *, prefix_actions, prefix_path):
        return self._run('postcontext_H1', prefix_actions, prefix_path)

    def suffix(self, *, prefix_actions, prefix_path, params):
        # Invalid params fail before any dispatch, including backend construction.
        suffix_raw(params)
        return self._run('bounded_suffix_H1', prefix_actions, prefix_path, params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', action='store_true')
    args = parser.parse_args()
    if args.contract:
        print(core.canonical(contract()).decode())


if __name__ == '__main__':
    main()
