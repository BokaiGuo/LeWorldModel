"""Whole-node paid-resource preauthorization. No neural/physics runtime here.

Common caps are NOT matched actual costs. Source-bound saved plans never grant
retroactive authorization to the historical experiment or dispatch CUDA.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p3_child_program_cuda_20261028_v1 as producer

NAME = 'agent_p3_child_budget_authorizer_20261028_v1'
ROOT, core, contract = producer.ROOT, producer.core, producer.contract
require, sha, digest, canonical = producer.require, producer.sha, producer.digest, producer.canonical
AXES, MODES = core.AXES, contract.MODES
REPORT = ROOT/'results/agent_p3_child_program_cuda_20261028_v1_cuda_MAIN01/report.json'
REPORT_SHA = '885f69f6de5ce9b78b8f3acdc84f14ef8891ba3ecd79912fc9af4e1920fc7ffb'
STRICT = ROOT/'results/agent_p3_child_program_cuda_20261028_v1_consume_MAIN01.json'
STRICT_SHA = 'd6322717231d72bb18e26eab1acbf6afe2dbc65eaa601af1fface21083a7e91c'
PRODUCER_SHA = 'f3bc4a58fb4dfc1ef640abd2ced50eafe5848a7dd9fc9110901cb98db5293e17'
FLAGS = dict(P1_admission=False, P3_admission=False, full_C3_demonstrated=False,
             equal_FLOPs=False, equal_wall=False, scientific_admission=False, physical_release=False)


def zero(): return dict.fromkeys(AXES, 0)
def add(a, b): return {k: a[k]+b[k] for k in AXES}
def sub(a, b): return {k: a[k]-b[k] for k in AXES}
def le(a, b): return all(a[k] <= b[k] for k in AXES)


def compile_cost(h, n):
    require(type(h) is int and h in (1, 3) and type(n) is int and n in (4, 8), 'unchanged H1/H3 and 4/8 quota')
    return producer.caps(dict(status=producer.READY, decision=dict(horizon=h, candidates=n)))


def quota(h, n):
    """Frozen integer quota; an over-cap node is refused, never rounded."""
    cost = compile_cost(h, n)
    legal = 2*h*n <= 24
    return dict(horizon=h, candidates=n, search_predict=2*h*n, legal=legal,
                cost=cost if legal else None,
                refusal=None if legal else 'original_whole_node_search_budget')


def signal_cost(): return contract.signal_caps(1)


def compile_envelope():
    legal = [quota(h, n)['cost'] for h in (1, 3) for n in (4, 8) if quota(h, n)['legal']]
    return {k: max(c[k] for c in legal) for k in AXES}


def inputs_template():
    node = add(signal_cost(), compile_envelope())
    return dict(schema=NAME+'.limits', common_node_cap=node,
                common_method_cap={k: 24*node[k] for k in AXES},
                allocation='unchanged_policy_H1_H3_candidates4_8_search24_plus_selected1',
                signal_sharing='independent', node_order='original24_children_then_fixed_adaptive',
                scientific_gates=False)


def limits(value):
    core.keys(value, ('schema', 'common_node_cap', 'common_method_cap', 'allocation', 'signal_sharing', 'node_order', 'scientific_gates'))
    template = inputs_template()
    require(all(value[k] == template[k] for k in value if k not in ('common_node_cap', 'common_method_cap')), 'explicit unchanged independent policy/order')
    require(value['common_node_cap'] == template['common_node_cap'] and
            value['common_method_cap'] == template['common_method_cap'],
            'fixed pre-registered vector caps; no post hoc quota or rounding')
    core.counters(value['common_node_cap']); core.counters(value['common_method_cap'])
    return deepcopy(value)


def refusal(reason, **fields): return dict(status='precondition_refused', reason=reason, dispatch_allowed=False, **fields)


def runtime_preconditions(adapter=None):
    # This implementation deliberately exports no unreviewed model loader. A later
    # reviewed CUDA integration must bind both old fullarray validators and these
    # pre-entry hooks; an arbitrary callable/"HOST_CUDA" label is not authority.
    return refusal('missing_reviewed_budget_instrumented_child_native_adapter',
        missing=['native signal+compiler adapter with these predispatch hooks',
                 'durable fullarray/seal/actual-shape closure and native consumer',
                 'MAIN frozen source/prepare and explicit HOST CUDA authorization'],
        supplied_adapter_is_not_authority=adapter is not None, neural_calls=0, physics_calls=0, **FLAGS)


class WholeNodeGate:
    """Callable leaf-entry gate; integer vectors, independent paid methods only.

    Pre-signal envelope conservatively reserves every possible legal original
    decision. After signal, sealing only releases UNATTEMPTED reserve. Attempts
    and returned costs are never reset. sink must persist each event before call.
    This gate cannot certify an adapter that omits its hooks.
    """
    def __init__(self, cap, children, tasks, sink=None):
        self.limits = limits(cap); self.sink = sink or (lambda e: None)
        require(len(tasks) == 2 and len(children) == 24 and
                [(c['episode_id'], c['boundary']) for c in children] ==
                [(e,b) for e in contract.EPISODES for b in contract.STEPS], 'fixed full original24child order')
        self.tasks = [core.validate_task(t) for t in tasks]
        self.children = [contract.validate_child(c, self.tasks) for c in children]
        self.order = [(c['child_id'], m) for c in self.children for m in MODES]
        self.cursor = 0; self.refused = []
        self.actual = {m: dict(attempted=zero(), completed=zero()) for m in MODES}
        self.nodes, self.pending, self.events = {}, {}, []
        self.failed, self.serial, self.stack = False, 0, []

    def emit(self, kind, **fields):
        event = dict(index=len(self.events), previous_sha256=self.events[-1]['event_sha256'] if self.events else None,
                     kind=kind, **deepcopy(fields))
        event['event_sha256'] = digest(event)
        self.events.append(event)
        try: self.sink(deepcopy(event))
        except BaseException:
            self.failed = True; raise

    def exposure(self, mode):
        return {k: self.actual[mode]['attempted'][k]+sum(x['reserve'][k] for x in self.nodes.values() if x['mode'] == mode)
                for k in AXES}

    def open(self, child_id, mode, context_sha256, binding_sha256, *, observed=True):
        require(mode in MODES and type(child_id) is str and child_id and type(observed) is bool, 'literal method/child identity')
        key = child_id+':'+mode
        require(self.cursor < 48 and self.order[self.cursor] == (child_id, mode), 'fixed48slot order; no retry/reentry/subset')
        child = self.children[self.cursor//2]
        require(context_sha256 == child['context_sha256'] and binding_sha256 == child['ancestry']['task_binding_sha256'] and
                observed == (child['status'] == 'observed_child'), 'exact same observed ancestry/context/task; no root or sibling substitute')
        self.cursor += 1
        if not observed:
            result = refusal('not_reached', child_id=child_id, mode=mode)
            self.refused.append(dict(ticket=key, **result))
            self.emit('open_refused', request=dict(child_id=child_id, mode=mode, context_sha256=context_sha256,
                      binding_sha256=binding_sha256, observed=observed), result=result); return result
        core.hash_value(context_sha256); core.hash_value(binding_sha256)
        request = dict(child_id=child_id, mode=mode, context_sha256=context_sha256, binding_sha256=binding_sha256, observed=observed)
        full = add(signal_cost(), compile_envelope())
        if self.failed or not le(full, self.limits['common_node_cap']) or not le(add(self.exposure(mode), full), self.limits['common_method_cap']):
            result = refusal('fail_stop' if self.failed else 'whole_node_signal_plus_compile_envelope_exceeds_cap',
                             child_id=child_id, mode=mode, required=full)
            self.refused.append(dict(ticket=key, **result))
            self.emit('open_refused', request=request, result=result); return result
        self.nodes[key] = dict(mode=mode, context_sha256=context_sha256, binding_sha256=binding_sha256,
                               reserve=full, phase='signal', stage_cap=signal_cost(), stage_attempted=zero(), stage_completed=zero(),
                               attempted=zero(), completed=zero(), decision=None, status='reserved_before_signal')
        result = dict(status='reserved_before_signal', ticket=key, dispatch_allowed=True, reserved=full)
        self.emit('opened', request=request, result=result); return result

    def begin(self, key, resource, *, shape=None, directions=None):
        require(not self.failed and key in self.nodes, 'known fail-stop node')
        node = self.nodes[key]; require(node['status'] in ('reserved_before_signal', 'compile_authorized'), 'no call after refusal/finish')
        change = core.delta(resource, shape, directions)
        native_shapes = dict(predict=[1,3,192], predictor=[1,3,192], action_encoder=[1,3,10], pred_proj=[3,192])
        if resource in native_shapes: require(shape == native_shapes[resource], 'actual native batch1/CLS192/full context3 shape')
        require(le(add(node['stage_attempted'], change), node['stage_cap']) and le(change, node['reserve']), 'whole-stage prepaid resource before dispatch')
        require(not self.stack or self.pending[self.stack[-1]]['ticket'] == key, 'no cross-node nested calls')
        ident = self.serial; self.serial += 1
        entry = dict(operation_id=ident, ticket=key, phase=node['phase'], resource=resource, shape=deepcopy(shape),
                     directions=directions, delta=change, context_sha256=node['context_sha256'], binding_sha256=node['binding_sha256'])
        node['reserve'] = sub(node['reserve'], change)
        node['attempted'] = add(node['attempted'], change); node['stage_attempted'] = add(node['stage_attempted'], change)
        self.actual[node['mode']]['attempted'] = add(self.actual[node['mode']]['attempted'], change)
        self.pending[ident] = entry; self.stack.append(ident)
        self.emit('attempted', entry=entry); return ident

    def returned(self, ident):
        require(not self.failed and self.stack and self.stack[-1] == ident, 'actual nested return, no replayed completion')
        entry = self.pending[ident]; node = self.nodes[entry['ticket']]; change = entry['delta']
        self.emit('returned', operation_id=ident)
        self.stack.pop(); del self.pending[ident]
        node['completed'] = add(node['completed'], change); node['stage_completed'] = add(node['stage_completed'], change)
        self.actual[node['mode']]['completed'] = add(self.actual[node['mode']]['completed'], change)

    def failed_call(self, ident, error):
        require(ident in self.pending, 'actual failed attempt')
        self.nodes[self.pending[ident]['ticket']]['status'] = 'engineering_failed'
        self.failed = True; self.emit('failed', operation_id=ident, error=str(error))

    def call(self, key, resource, fn, *, shape=None, directions=None):
        ident = self.begin(key, resource, shape=shape, directions=directions)
        try:
            value = fn(); self.returned(ident); return value
        except BaseException as exc:
            if not self.failed: self.failed_call(ident, type(exc).__name__+': '+str(exc))
            raise

    def seal(self, key, decision, signals):
        require(not self.failed and key in self.nodes and not self.pending, 'no signal outcome while calls pending/failed')
        node = self.nodes[key]
        require(node['phase'] == 'signal' and node['status'] == 'reserved_before_signal' and
                node['stage_attempted'] == node['stage_completed'] == signal_cost(), 'complete actually paid H1/H3+schedule first')
        child = next(c for c in self.children if key == c['child_id']+':'+node['mode'])
        task = self.tasks[child['root_slot']]
        # Exact inherited chooser. No outcome, selected-Q or later physical state
        # is an input. Native validity of the paid signal still needs its original
        # fullarray consumer, not just this callback's operation counts.
        frozen = core.FrozenSignals(**signals)
        require(canonical(decision) == canonical(core.choose(task, dict(context=child['context'], depth=0), frozen, node['mode'])),
                'unchanged original same-child policy decision; no quota/threshold repair')
        h, n = decision['horizon'], decision['candidates']; q = quota(h, n)
        if not q['legal']: reason = 'original_whole_node_search_budget'
        elif decision['extension_type'] != 'progress' or decision['program_type'] != 'progress': reason = 'unsupported_reconfiguration'
        elif decision['continue_expanding'] is not True: reason = 'original_policy_stopped'
        else: reason = None
        node['decision'] = deepcopy(decision)
        node['reserve'] = zero() if reason else q['cost']
        node.update(phase='compile', stage_cap=zero() if reason else q['cost'], stage_attempted=zero(), stage_completed=zero(),
                    status='refused_after_paid_signal' if reason else 'compile_authorized')
        result = refusal(reason, ticket=key, paid_signal_retained=True) if reason else dict(status='compile_authorized', ticket=key, dispatch_allowed=True)
        self.emit('sealed', ticket=key, decision=decision, signals=signals, result=result); return result

    def finish(self, key):
        require(not self.failed and not self.pending, 'complete actual returns before finish')
        node = self.nodes[key]
        require(node['status'] == 'compile_authorized' and node['stage_attempted'] == node['stage_completed'] == node['stage_cap'] and
                node['reserve'] == zero(), 'whole original compiler quota including independent selected-prefix replay')
        node['status'] = 'budget_complete_native_unverified'; self.emit('finished', ticket=key)
        return dict(status=node['status'], native_verified=False, **FLAGS)

    def receipt(self):
        return deepcopy(dict(schema=NAME+'.gate_receipt', limits=self.limits, children=self.children, tasks=self.tasks,
                             fixed_denominator=48, child_denominator=24, cursor=self.cursor, refused=self.refused,
                             actual=self.actual, nodes=self.nodes,
                             pending=list(self.pending.values()), failed=self.failed, events=self.events,
                             same_cap=True, hook_vectors_equal=self.actual['fixed'] == self.actual['adaptive'],
                             matched_actual_vector=False,
                             FLOPs=None, cuda_ns=None, wall_ns=None, neural_execution_certified=False, **FLAGS))


def consume_gate(receipt):
    """Independent state-machine replay; invokes no backend, not an array validator."""
    gate = WholeNodeGate(receipt['limits'], receipt['children'], receipt['tasks']); previous = None
    for index, e in enumerate(receipt['events']):
        check = dict(e); h = check.pop('event_sha256')
        require(e['index'] == index and e['previous_sha256'] == previous and digest(check) == h, 'gate chained event identity')
        previous = h; kind = e['kind']
        if kind in ('opened', 'open_refused'): gate.open(**e['request'])
        elif kind == 'attempted':
            a = e['entry']; gate.begin(a['ticket'], a['resource'], shape=a['shape'], directions=a['directions'])
        elif kind == 'returned': gate.returned(e['operation_id'])
        elif kind == 'failed': gate.failed_call(e['operation_id'], e['error'])
        elif kind == 'sealed': gate.seal(e['ticket'], e['decision'], e['signals'])
        elif kind == 'finished': gate.finish(e['ticket'])
        else: raise ValueError('unknown gate event')
        require(gate.events[-1] == e, 'replayed exact gate entry/result, no hidden edits')
    require(canonical(gate.receipt()) == canonical(receipt), 'entire strict budget receipt reconstruction')
    return dict(status='budget_hooks_replayed_NOT_native_or_C3_verified', same_cap=True,
                hook_vectors_equal=receipt['hook_vectors_equal'], matched_actual_vector=False, **FLAGS)


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def load_saved():
    for p, h in ((Path(producer.__file__), PRODUCER_SHA), (REPORT, REPORT_SHA), (STRICT, STRICT_SHA),
                 (producer.SIGNAL_RECEIPT, producer.SIGNAL_RECEIPT_SHA)):
        require(sha(p) == h, 'actual authority boundary before consumers: '+str(p))
    receipt = producer.consume(str(REPORT.parent), REPORT_SHA)
    require(canonical(receipt) == canonical(producer.base.read(STRICT, 128*1024**2)), 'full strict compiler receipt, not summary')
    report = producer.base.read(REPORT, 256*1024**2)
    require(sha(report['prepared']) == report['prepare_sha256'], 'actual compiler prepare edge')
    p = producer.base.read(report['prepared'], 64*1024**2); contract.check_sources(p['sources'])
    contract.merge(p['sources'], {str(Path(report['prepared'])): report['prepare_sha256']})
    signals = producer.base.read(producer.SIGNAL_RECEIPT, 256*1024**2)
    return p, receipt, signals


def saved_plan(p, receipt, signals, cap):
    cap = limits(cap)
    require(len(p['children']) == len(signals['rows']) == 24 and len(p['branches']) == len(receipt['rows']) == 48, 'all24child/48methods')
    rows = []; exposure = {m: zero() for m in MODES}; actual = {m: dict(attempted=zero(), completed=zero()) for m in MODES}
    for i, (branch, compiled) in enumerate(zip(p['branches'], receipt['rows'], strict=True)):
        c = contract.validate_child(p['children'][i//2], p['tasks']); mode = MODES[i%2]
        require((branch['child_id'], branch['mode']) == (compiled['child_id'], compiled['mode']) == (c['child_id'], mode) and
                compiled['branch_sha256'] == digest(branch), 'strict same child/mode/branch identity')
        row = dict(child_id=c['child_id'], mode=mode, context_sha256=c['context_sha256'], decision=deepcopy(branch['decision']),
                   status='not_reached', required=None, historical_signal=None, historical_compiler=None, reasons=[],
                   CUDA_dispatch_allowed=False, future_native_precondition=runtime_preconditions())
        if c['status'] != 'not_reached':
            sr = signals['rows'][i//2]; require(sr['child_id'] == c['child_id'] and sr['status'] == 'returned', 'complete same child paid signal')
            a = [x for x in sr['result']['acquisitions'] if x['group'] == branch['signal_group']]
            require(len(a) == 1 and branch['signal_charge'] == dict(numerator=1, denominator=1) and
                    digest(a[0]) == branch['signal_receipt_sha256'], 'exact independent paid acquisition, no free or fractional J')
            paid = a[0]['resource_receipt']; core.audit_ledger(paid)
            require(paid['attempted'] == paid['completed'] == signal_cost(), 'full native 8API/2J/384VJP/schedule charge')
            work = compiled['resource_receipt']; row['historical_signal'] = deepcopy({k: paid[k] for k in ('attempted', 'completed')})
            if branch['status'] == producer.READY:
                require(compiled['status'] == producer.FINAL and work is not None, 'full compiler only, failed remains unknown')
                core.audit_ledger(work)
                require(work['attempted'] == work['completed'] == compile_cost(branch['decision']['horizon'], branch['decision']['candidates']), 'exact actual compiler+replay cost')
                row['historical_compiler'] = deepcopy({k: work[k] for k in ('attempted', 'completed')})
            else: require(work is None and compiled['selected'] is None, 'original refused remains unknown, never zero scientific result')
            for kind in ('attempted', 'completed'):
                actual[mode][kind] = add(actual[mode][kind], paid[kind])
                if work is not None: actual[mode][kind] = add(actual[mode][kind], work[kind])
            original = sr['result']['rows'][i%2]
            require(original['mode'] == mode and original['decision'] == branch['decision'] and
                    original['signal_group'] == branch['signal_group'], 'original full paid chooser, never inferred from outcome')
            # This is a saved feasibility projection, NOT retroactive permission.
            needed = add(signal_cost(), compile_envelope()); row['required'] = needed
            fits = le(needed, cap['common_node_cap']) and le(add(exposure[mode], needed), cap['common_method_cap'])
            if not fits:
                row['reasons'].append('whole_node_signal_plus_compile_envelope_exceeds_cap')
            if branch['status'] != producer.READY: row['reasons'].append(branch['status'])
            row['status'] = 'whole_node_refused' if row['reasons'] else 'budget_feasible_native_adapter_missing'
            if fits:
                # Release only unattempted worst-case surplus after sealed original decision.
                # Even an ORIGINAL H3x8 refusal paid its signal; never refund it.
                paid_work = compile_cost(branch['decision']['horizon'], branch['decision']['candidates']) if branch['status'] == producer.READY else zero()
                exposure[mode] = add(exposure[mode], add(signal_cost(), paid_work))
        else:
            require(branch['status'] == compiled['status'] == 'not_reached' and branch['decision'] is compiled['selected'] is None and
                    compiled['resource_receipt'] is None, 'unreached preserved unknown with no fabricated operations')
        rows.append(row)
    require(sum(r['status'] == 'not_reached' for r in rows) == 2, 'preserve original unreached pair')
    return dict(schema=NAME+'.plan', fixed_denominator=48, child_denominator=24, independent_roots=2,
                rows=rows, common_limits=cap, historical_actual=actual,
                same_cap=True, matched_historical_actual_vector=actual['fixed'] == actual['adaptive'],
                matched_historical_predict=actual['fixed']['attempted']['predict_API'] == actual['adaptive']['attempted']['predict_API'],
                projected_admitted_exposure=exposure, retroactive_authorization=False,
                all_observed_refused=all(r['status'] != 'budget_feasible_native_adapter_missing' for r in rows),
                runtime_precondition=runtime_preconditions(), full_tree=False, no_new_policy=True,
                original_census=dict(pending=41, refused=5, not_reached=2),
                FLOPs=None, wall_ns=None, neural_calls=0, physics_calls=0, **FLAGS)


def material(path, expected):
    path = Path(path); require(path.is_absolute() and sha(path) == expected, 'external limits manifest SHA')
    cap = limits(producer.base.read(path, 1024**2)); p, receipt, signals = load_saved()
    result = saved_plan(p, receipt, signals, cap)
    sources = deepcopy(p['sources']); contract.merge(sources, {str(f): sha(f) for f in own_files()})
    contract.merge(sources, {str(path): expected, str(REPORT): REPORT_SHA, str(STRICT): STRICT_SHA,
                            str(producer.SIGNAL_RECEIPT): producer.SIGNAL_RECEIPT_SHA})
    contract.check_sources(sources)
    require(sha(path) == expected, 'limits unchanged after full saved closure')
    return dict(schema=NAME+'.prepare', limits_path=str(path), limits_sha256=expected, plan=result, sources=sources,
                native_execution_authorized=False, **FLAGS)


def verify(path, expected):
    require(Path(path).is_absolute() and sha(path) == expected, 'external budget prepare SHA')
    p = producer.base.read(path, 64*1024**2); contract.check_sources(p['sources'])
    require(canonical(p) == canonical(material(p['limits_path'], p['limits_sha256'])) and sha(path) == expected,
            'complete strict/source/limits/cost plan reconstruction'); return p


def write_new(path, value):
    path = Path(path)
    require(path.is_absolute() and not path.exists() and not path.is_symlink(), 'NEW absolute exclusive metadata')
    data = canonical(value); require(len(data) <= 64*1024**2, 'bounded saved metadata')
    require(shutil.disk_usage(path.parent).free >= 200*1024**2+len(data), '200MiB output floor')
    with path.open('xb') as f:
        require(f.write(data) == len(data), 'complete metadata'); f.flush(); os.fsync(f.fileno())
    producer.base.fsync_dir(path.parent); return dict(path=str(path), sha256=sha(path), bytes=len(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('inputs'); a.add_argument('--output', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--input', required=True); a.add_argument('--input-sha256', required=True); a.add_argument('--output', required=True)
    for cmd in ('plan', 'consume'):
        a = sub.add_parser(cmd); a.add_argument('--prepared', required=True); a.add_argument('--prepare-sha256', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('preconditions'); a.add_argument('--output', required=True)
    a = sub.add_parser('consume-gate'); a.add_argument('--input', required=True); a.add_argument('--input-sha256', required=True); a.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.cmd == 'inputs': result = write_new(args.output, inputs_template())
    elif args.cmd == 'prepare': result = write_new(args.output, material(args.input, args.input_sha256))
    elif args.cmd == 'preconditions': result = write_new(args.output, runtime_preconditions())
    elif args.cmd == 'consume-gate':
        require(Path(args.input).is_absolute() and sha(args.input) == args.input_sha256, 'external complete gate receipt SHA')
        checked = consume_gate(producer.base.read(args.input, 64*1024**2))
        require(sha(args.input) == args.input_sha256, 'gate receipt unchanged'); result = write_new(args.output, checked)
    else:
        p = verify(args.prepared, args.prepare_sha256)
        result = write_new(args.output, p['plan'] if args.cmd == 'plan' else dict(schema=NAME+'.consume',
            prepared_sha256=args.prepare_sha256, status='strict_saved_budget_plan_verified_not_admitted', plan=p['plan'], **FLAGS))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
