"""Fail-closed v2 of the P3 child budget authorizer.

This module is a saved/RAM contract only.  It adds an immutable cap-vector
identity and a durable failure ticket to the frozen v1 state machine; it does
not dispatch a model, CUDA, or physics operation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p3_child_budget_authorizer_20261028_v1 as v1

NAME = 'agent_p3_child_budget_authorizer_20261028_v2'
ROOT = v1.ROOT
producer, core, contract = v1.producer, v1.core, v1.contract
require, sha, digest, canonical = v1.require, v1.sha, v1.digest, v1.canonical
AXES, MODES = v1.AXES, v1.MODES
REPORT, REPORT_SHA = v1.REPORT, v1.REPORT_SHA
STRICT, STRICT_SHA = v1.STRICT, v1.STRICT_SHA
CAP_VECTOR_SHA256 = '4de9887ccc5b63b3b4ac82c6847877bd3ece96b73f81307e647f08fbf3b7e0eb'
FLAGS = dict(P1_admission=False, P3_admission=False, full_C3_demonstrated=False,
             equal_FLOPs=False, equal_wall=False, scientific_admission=False,
             physical_release=False)


def inputs_template():
    value = deepcopy(v1.inputs_template())
    value['schema'] = NAME + '.limits'
    return value


def _cap_identity(value):
    return digest({'common_node_cap': value['common_node_cap'],
                   'common_method_cap': value['common_method_cap']})


def limits(value):
    """Require the complete pre-registered contract and its cap-vector hash."""
    require(isinstance(value, dict), 'limits object')
    expected = inputs_template()
    require(canonical(value) == canonical(expected),
            'canonical expected fixed/adaptive budget contract mismatch')
    require(_cap_identity(value) == CAP_VECTOR_SHA256,
            'canonical expected cap-vector SHA mismatch')
    core.counters(value['common_node_cap'])
    core.counters(value['common_method_cap'])
    return deepcopy(value)


def runtime_preconditions(adapter=None):
    return v1.runtime_preconditions(adapter)


class WholeNodeGate(v1.WholeNodeGate):
    """v1 gate with immutable caps and a non-reusable failure ticket."""

    def __init__(self, cap, children, tasks, sink=None):
        checked = limits(cap)
        # The frozen v1 constructor performs its own legacy-schema check.  Feed
        # it the identical vector under that internal schema, then retain the
        # v2-validated object as the authoritative receipt field.
        legacy = deepcopy(checked)
        legacy['schema'] = v1.NAME + '.limits'
        super().__init__(legacy, children, tasks, sink=sink)
        self.limits = deepcopy(checked)
        self.failure_ticket = None

    def open(self, child_id, mode, context_sha256, binding_sha256, *, observed=True):
        if not self.failed:
            return super().open(child_id, mode, context_sha256, binding_sha256,
                                observed=observed)
        # Once an attempt failed, validate the next fixed slot and its ancestry
        # but never enter it.  This makes a fixed->adaptive strategy switch a
        # durable refusal rather than a new reservation.
        require(mode in MODES and type(child_id) is str and child_id and
                type(observed) is bool, 'literal method/child identity')
        require(self.cursor < 48 and self.order[self.cursor] == (child_id, mode),
                'fixed48slot order; no retry/reentry/subset')
        child = self.children[self.cursor // 2]
        require(context_sha256 == child['context_sha256'] and
                binding_sha256 == child['ancestry']['task_binding_sha256'] and
                observed == (child['status'] == 'observed_child'),
                'same observed ancestry/context/task; no root or sibling substitute')
        self.cursor += 1
        key = child_id + ':' + mode
        result = v1.refusal('fail_stop', child_id=child_id, mode=mode,
                            failure_ticket=deepcopy(self.failure_ticket))
        self.refused.append(dict(ticket=key, **result))
        request = dict(child_id=child_id, mode=mode,
                       context_sha256=context_sha256,
                       binding_sha256=binding_sha256, observed=observed)
        self.emit('open_refused', request=request, result=result)
        return result

    def failed_call(self, ident, error):
        require(ident in self.pending, 'actual failed attempt')
        entry = self.pending[ident]
        self.failure_ticket = dict(operation_id=ident, ticket=entry['ticket'],
                                   error=str(error))
        return super().failed_call(ident, error)

    def receipt(self):
        result = super().receipt()
        result['schema'] = NAME + '.gate_receipt'
        result['failure_ticket'] = deepcopy(self.failure_ticket)
        result['cap_vector_sha256'] = CAP_VECTOR_SHA256
        return result


def consume_gate(receipt):
    """Replay a v2 gate receipt without any backend or neural import."""
    require(receipt.get('schema') == NAME + '.gate_receipt', 'v2 gate receipt schema')
    require(receipt.get('cap_vector_sha256') == CAP_VECTOR_SHA256,
            'v2 cap-vector identity')
    gate = WholeNodeGate(receipt['limits'], receipt['children'], receipt['tasks'])
    previous = None
    for index, event in enumerate(receipt['events']):
        check = dict(event)
        event_hash = check.pop('event_sha256')
        require(event['index'] == index and event['previous_sha256'] == previous and
                digest(check) == event_hash, 'gate chained event identity')
        previous = event_hash
        kind = event['kind']
        if kind in ('opened', 'open_refused'):
            gate.open(**event['request'])
        elif kind == 'attempted':
            entry = event['entry']
            gate.begin(entry['ticket'], entry['resource'], shape=entry['shape'],
                       directions=entry['directions'])
        elif kind == 'returned':
            gate.returned(event['operation_id'])
        elif kind == 'failed':
            gate.failed_call(event['operation_id'], event['error'])
        elif kind == 'sealed':
            gate.seal(event['ticket'], event['decision'], event['signals'])
        elif kind == 'finished':
            gate.finish(event['ticket'])
        else:
            raise ValueError('unknown gate event')
        require(gate.events[-1] == event, 'replayed exact gate entry/result')
    require(canonical(gate.receipt()) == canonical(receipt),
            'entire strict v2 budget receipt reconstruction')
    return dict(schema=NAME + '.gate_consume', status='budget_hooks_replayed_NOT_native_or_C3_verified',
                cap_vector_sha256=CAP_VECTOR_SHA256, same_cap=True,
                matched_actual_vector=False, **FLAGS)


def load_saved():
    return v1.load_saved()


def saved_plan(p, receipt, signals, cap):
    checked = limits(cap)
    result = deepcopy(v1.saved_plan(p, receipt, signals, checked))
    result['schema'] = NAME + '.plan'
    result['cap_vector_sha256'] = CAP_VECTOR_SHA256
    result['same_cap'] = True
    return result


def own_files():
    return [Path(__file__).resolve(), ROOT / f'tests/{NAME}_test.py',
            ROOT / f'innerdocs/{NAME}.md']


def material(path, expected):
    path = Path(path)
    require(path.is_absolute() and sha(path) == expected, 'external limits manifest SHA')
    cap = limits(producer.base.read(path, 1024**2))
    p, receipt, signals = load_saved()
    result = saved_plan(p, receipt, signals, cap)
    sources = deepcopy(p['sources'])
    contract.merge(sources, {str(f): sha(f) for f in own_files()})
    contract.merge(sources, {str(path): expected, str(REPORT): REPORT_SHA,
                             str(STRICT): STRICT_SHA,
                             str(producer.SIGNAL_RECEIPT): producer.SIGNAL_RECEIPT_SHA})
    contract.check_sources(sources)
    require(sha(path) == expected, 'limits unchanged after full saved closure')
    return dict(schema=NAME + '.prepare', limits_path=str(path),
                limits_sha256=expected, plan=result, sources=sources,
                cap_vector_sha256=CAP_VECTOR_SHA256,
                native_execution_authorized=False, **FLAGS)


def verify(path, expected):
    path = Path(path)
    require(path.is_absolute() and sha(path) == expected, 'external budget prepare SHA')
    prepared = producer.base.read(path, 64 * 1024**2)
    contract.check_sources(prepared['sources'])
    require(prepared.get('cap_vector_sha256') == CAP_VECTOR_SHA256,
            'prepared cap-vector identity')
    require(canonical(prepared) == canonical(material(prepared['limits_path'],
                                                       prepared['limits_sha256'])) and
            sha(path) == expected, 'complete strict/source/limits/cost plan reconstruction')
    return prepared


def write_new(path, value):
    path = Path(path)
    require(path.is_absolute() and not path.exists() and not path.is_symlink(),
            'NEW absolute exclusive metadata')
    data = canonical(value)
    require(len(data) <= 64 * 1024**2, 'bounded saved metadata')
    require(shutil.disk_usage(path.parent).free >= 200 * 1024**2 + len(data),
            '200MiB output floor')
    with path.open('xb') as handle:
        require(handle.write(data) == len(data), 'complete metadata')
        handle.flush()
        os.fsync(handle.fileno())
    producer.base.fsync_dir(path.parent)
    return dict(path=str(path), sha256=sha(path), bytes=len(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('inputs'); a.add_argument('--output', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--input', required=True); a.add_argument('--input-sha256', required=True); a.add_argument('--output', required=True)
    for command in ('plan', 'consume'):
        a = sub.add_parser(command); a.add_argument('--prepared', required=True); a.add_argument('--prepare-sha256', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('preconditions'); a.add_argument('--output', required=True)
    a = sub.add_parser('consume-gate'); a.add_argument('--input', required=True); a.add_argument('--input-sha256', required=True); a.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.cmd == 'inputs':
        result = write_new(args.output, inputs_template())
    elif args.cmd == 'prepare':
        result = write_new(args.output, material(args.input, args.input_sha256))
    elif args.cmd == 'preconditions':
        result = write_new(args.output, runtime_preconditions())
    elif args.cmd == 'consume-gate':
        require(Path(args.input).is_absolute() and sha(args.input) == args.input_sha256,
                'external complete gate receipt SHA')
        result = write_new(args.output,
                           consume_gate(producer.base.read(args.input, 64 * 1024**2)))
    else:
        prepared = verify(args.prepared, args.prepare_sha256)
        value = prepared['plan'] if args.cmd == 'plan' else dict(
            schema=NAME + '.consume', prepared_sha256=args.prepare_sha256,
            status='strict_saved_budget_plan_verified_not_admitted',
            plan=prepared['plan'], cap_vector_sha256=CAP_VECTOR_SHA256, **FLAGS)
        result = write_new(args.output, value)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
