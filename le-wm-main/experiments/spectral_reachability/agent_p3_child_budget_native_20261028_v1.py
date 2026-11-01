"""Budget-gated child native wiring; CLI is saved-only, never a model runner."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p3_child_budget_authorizer_20261028_v2 as budget

NAME = 'agent_p3_child_budget_native_20261028_v1'
ROOT, core, contract = budget.ROOT, budget.core, budget.contract
compiler = budget.producer
paid = compiler.paid
require, sha, digest, canonical = budget.require, budget.sha, budget.digest, budget.canonical
PINS = {
    str(Path(budget.__file__).resolve()): 'bf4c885feaa318347341cf14e6a1ec960d439cf6879d95b9df2c1790a018b072',
    str(Path(budget.v1.__file__).resolve()): '0deb7cb3df6eda5405a995aefb473e83f31de3654bc54c43886bbf181254905b',
    str(Path(compiler.__file__).resolve()): budget.v1.PRODUCER_SHA,
    str(Path(paid.__file__).resolve()): '1a7ea154b09c0caf59b5bc4c4d8b4b333493aaac4b6dc1a1afe1363f9ecc2670',
    str(ROOT/'tests/agent_p3_child_budget_authorizer_20261028_v2_test.py'): '7f238bb499eb77870ad7981891a1eb77b5933f0ec9fb4a2c6b90e750a51351af',
    str(ROOT/'innerdocs/agent_p3_child_budget_authorizer_20261028_v2.md'): 'beb9d31f7b760a85447d0e92552f1362ddfb636acf08f5366f030f01dd582694',
}


class GateSink:
    """Frozen Ledger sink: gate.begin is inside resource_attempt, BEFORE fn().

    Native fullarray events are forwarded unchanged. Nested operations each
    retain their own unit; they are not summed into FLOPs or whole-job wall.
    """
    def __init__(self, gate, ticket, sink):
        require(callable(sink), 'explicit durable sink or RAM collector')
        self.gate, self.ticket, self.sink = gate, ticket, sink
        self.operations = {}

    def event(self, event):
        kind = event['kind']
        entry = event.get('entry')
        ident = None if entry is None else entry['index']
        if kind == 'resource_attempt':
            require(ident not in self.operations, 'no duplicate stage operation')
            op = self.gate.begin(self.ticket, entry['resource'], shape=entry['shape'],
                                 directions=entry['directions'])
            self.operations[ident] = op
            try:
                self.sink(deepcopy(event))
            except BaseException as exc:
                if not self.gate.failed:
                    self.gate.failed_call(op, type(exc).__name__ + ': ' + str(exc))
                raise
        elif kind == 'resource_return':
            require(ident in self.operations, 'return requires actual gated attempt')
            op = self.operations[ident]
            try:
                self.sink(deepcopy(event))
            except BaseException as exc:
                if not self.gate.failed:
                    self.gate.failed_call(op, type(exc).__name__ + ': ' + str(exc))
                raise
            self.gate.returned(op)
        elif kind == 'resource_failure':
            require(ident in self.operations, 'failure requires actual gated attempt')
            if not self.gate.failed:
                self.gate.failed_call(self.operations[ident], entry['error'])
            self.sink(deepcopy(event))
        else:
            self.sink(deepcopy(event))


class NativeFactories:
    """No loader: MAIN supplies an already reviewed frozen CUDA instance.

    Constructor is guarded before native constructors, torch imports or calls.
    CLI intentionally never constructs this object.
    """
    kind = 'HOST_CUDA'

    def __init__(self, model, device, tasks, *, authorization=None):
        require(authorization == 'MAIN_APPROVED_BUDGET_CHILD_CUDA',
                'MAIN authorization before native construction')
        require(getattr(device, 'type', None) == 'cuda', 'CUDA only; no CPU fallback')
        self.model, self.device, self.tasks = model, device, tasks

    def signal(self, task, child, sink):
        return paid.NativeChildSignal(self.model, self.device, task, child, sink, self.tasks)

    def program(self, task, child, sink):
        return compiler.NativeProgram(self.model, self.device, task, child, sink, self.tasks)


class Session:
    """All48 roster exists before any callback; each method pays its own H1/H3.

    This is wiring, not a standalone CUDA lifecycle/strict fullarray consumer.
    No runtime release is granted by saved prepare or a RAM session.
    """
    def __init__(self, prepared, sink):
        self.prepared, self.sink = deepcopy(prepared), sink
        require(callable(sink), 'explicit session sink')
        self.gate = budget.WholeNodeGate(prepared['limits'], prepared['children'],
                                         prepared['tasks'], sink=lambda e: sink(dict(kind='budget_gate', payload=e)))
        self.rows = [dict(child_id=c['child_id'], mode=mode,
                          status='not_reached' if c['status']=='not_reached' else 'not_started',
                          signal=None, signal_resources=None, decision=None,
                          compiler=None, error=None)
                     for c in self.gate.children for mode in contract.MODES]
        self.stopped = False

    def next(self, factories, *, permit_ram=False, authorization=None):
        require(not self.stopped, 'session fail-stop; no strategy switch/retry')
        require((permit_ram is True and factories.kind == 'RAM_FIXTURE') or
                (permit_ram is False and type(factories) is NativeFactories and
                 authorization == 'MAIN_APPROVED_BUDGET_CHILD_CUDA'),
                'explicit RAM fixture or MAIN native authorization before callbacks')
        i = self.gate.cursor
        require(i < 48, 'no reentry after full48')
        child, mode = self.gate.children[i//2], contract.MODES[i%2]
        row = self.rows[i]
        opened = self.gate.open(child['child_id'], mode, child['context_sha256'],
                                child['ancestry']['task_binding_sha256'],
                                observed=child['status']=='observed_child')
        if not opened['dispatch_allowed']:
            row['status'] = opened['reason']
            return deepcopy(row)
        key = opened['ticket']
        task = self.gate.tasks[child['root_slot']]
        node = dict(context=child['context'], depth=0)
        ledger = None
        try:
            row['status'] = 'signal_attempted'
            signal_sink = GateSink(self.gate, key, self.sink)
            backend = factories.signal(task, child, signal_sink)
            require(backend.kind == factories.kind, 'literal backend kind')
            ledger = core.Ledger(contract.signal_caps(1), sink=signal_sink.event)
            ledger.scope = dict(child_id=child['child_id'], group=mode, phase='child_signal',
                                parent_context_sha256=child['context_sha256'])
            signals, receipt = core.local_signal(task, node, backend, ledger, core.policy(task, node))
            require(ledger.attempted == ledger.completed == contract.signal_caps(),
                    'actual complete independent H1/H3 before chooser')
            require(all(r['producer_sha256'] == sha(paid.__file__) for r in receipt['responses']),
                    'native paid source identity; never root J')
            ledger.scope = dict(child_id=child['child_id'], group=mode, mode=mode,
                                phase='schedule', parent_context_sha256=child['context_sha256'])
            decision = ledger.call('schedule', lambda: core.choose(task, node, signals, mode))
            require(ledger.attempted == ledger.completed == contract.signal_caps(1),
                    'full actual independent signal and schedule payment')
            row.update(signal=receipt, decision=decision)
            sealed = self.gate.seal(key, decision, asdict(signals))
            if not sealed['dispatch_allowed']:
                row['status'] = sealed['reason']
                return deepcopy(row)
            h, count = decision['horizon'], decision['candidates']
            response = next(r for r in receipt['responses'] if r['horizon']==h)
            branch = dict(child_id=child['child_id'], root_slot=child['root_slot'], mode=mode,
                          child_sha256=digest(child), parent_context_sha256=child['context_sha256'],
                          status=compiler.READY, decision=deepcopy(decision), response=deepcopy(response))
            row['compiler'] = compiler.initial_rows([branch])[0]
            program_sink = GateSink(self.gate, key, self.sink)
            def factory(actual_child, actual_branch, actual_sink):
                require(actual_child == child and actual_branch == branch and actual_sink is program_sink,
                        'same child/decision/sink at actual native compiler entry')
                return factories.program(task, child, actual_sink)
            compiler.execute_branch(task, child, branch, row['compiler'], factory, program_sink,
                                    permit_ram=permit_ram,
                                    authorization='MAIN_APPROVED_HOST_CHILD_PROGRAM_CUDA' if not permit_ram else None)
            self.gate.finish(key)
            row['status'] = 'budget_complete_native_unverified'
            return deepcopy(row)
        except BaseException as exc:
            self.stopped = True
            row.update(status='engineering_failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            if ledger is not None:
                row['signal_resources'] = ledger.receipt()

    def receipt(self):
        return dict(schema=NAME+'.ram_or_unverified_session', rows=deepcopy(self.rows),
                    gate=self.gate.receipt(), fixed_denominator=48, child_denominator=24,
                    stopped=self.stopped, same_cap=True, matched_actual_vector=False,
                    native_verified=False, **budget.FLAGS)


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def material():
    contract.check_sources(PINS)
    # This is the real full saved compiler consumer, not a summary projection.
    p, receipt, signals = budget.load_saved()
    limits = budget.limits(budget.inputs_template())
    children = [contract.validate_child(c, p['tasks']) for c in p['children']]
    require(len(children)==24 and len(receipt['rows'])==48, 'all24/48 source denominator')
    sources = deepcopy(p['sources'])
    contract.merge(sources, PINS)
    contract.merge(sources, {str(f):sha(f) for f in own_files()})
    contract.merge(sources, {str(budget.REPORT):budget.REPORT_SHA,
        str(budget.STRICT):budget.STRICT_SHA,
        str(compiler.SIGNAL_OUTPUT/'report.json'):compiler.SIGNAL_REPORT_SHA,
        str(compiler.SIGNAL_RECEIPT):compiler.SIGNAL_RECEIPT_SHA})
    contract.check_sources(sources)
    return dict(schema=NAME+'.prepare', tasks=deepcopy(p['tasks']), children=children,
        limits=limits, cap_vector_sha256=budget.CAP_VECTOR_SHA256, sources=sources,
        fixed_denominator=48, child_denominator=24, independent_roots=2,
        historical_census=dict(compiled=41, refused=5, not_reached=2),
        historical_is_not_new_payment=True, signal_sharing='independent',
        runtime_dispatch_allowed=False, missing_before_CUDA=[
            'MAIN reviewed host lifecycle, total output/wall/memory guard and authorization',
            'durable fullarray native strict consumer for NEW gate+signal+compiler interleaved journal'],
        **budget.FLAGS)


def verify(path, expected):
    path = contract.absolute(path)
    require(sha(path)==expected, 'external actual prepare SHA')
    p = compiler.base.read(path, 64*1024**2)
    contract.check_sources(p['sources'])
    require(canonical(p)==canonical(material()) and sha(path)==expected,
            'complete saved source/contract reconstruction')
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--output', required=True)
    a = sub.add_parser('verify'); a.add_argument('--prepared', required=True); a.add_argument('--prepare-sha256', required=True)
    args = parser.parse_args()
    if args.cmd=='prepare':
        output = contract.absolute(args.output)
        require(output.parent == ROOT/'results' and output.name.startswith(NAME+'_') and
                not output.exists(), 'NEW exclusive own results metadata before full consume')
        result = budget.write_new(output, material())
    else:
        verify(args.prepared, args.prepare_sha256)
        result = dict(status='strict_saved_wiring_verified_not_authorized', neural_calls=0, physics_calls=0)
    print(json.dumps(result, sort_keys=True))


if __name__=='__main__': main()
