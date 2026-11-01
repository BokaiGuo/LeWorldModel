"""Saved-only five actual cp10 full witnesses. No upstream execution/imports."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import struct

NAME = 'agent_p3_actual_cp10_full_witness_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
A = 'agent_p3_same_time_cp10_rgb_bridge_20261029_v1'
PINS = dict(
    prepare='f2fac71516c36bdb654a9a84bbc0115d6155c04ed10231bc6b87b99ed0bda846',
    report='b5744f311d3f412cfc5016bf37c8854cc7b5272a6c37e62055c7f3aa68f2c657',
    completion='37588f050ed9381b191673e27c0b6fcab3ddf0c726e63a54bb75d9579963facf',
    receipt='af67caf8536a9637d709fcfb11c947d993e566bd4de5914725f4640a75de4d6f')
ORDER = ((0, 'C', 'C', 0), (0, 'Z', 'ZP', 0), (0, 'P', 'ZP', 1),
         (1, 'Z', 'ZP', 2), (1, 'P', 'ZP', 3))
CAP = 128 * 1024**2
NEXT_JOB_BUDGET = dict(jobs=10, PNG_per_job=81, operations_per_job=374, events_per_job=834,
    public_per_job=25, physics_per_job=254, cap_bytes=64*1024**2, reserve_bytes=8*1024**2,
    disk_floor_bytes=200*1024**2, wall_seconds=900,
    resource_caps=dict(constructor=1, close=1, reset=2, set_state=4, goal_render=2, goal_bind=1,
                       goal_setter=3, public_step=25, physics_step=254, explicit_render=81))
FLAGS = dict(new_neural_calls=0, new_physics_calls=0, fullstrict_reexecuted=False,
             scientific_admission=False, physical_release=False, stageB_ready=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def raw(value):
    # Exact original source.canonical, including spaces, no trailing newline.
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(raw(value)).hexdigest()


def same(a, b):
    return raw(a) == raw(b)


def path(value):
    p = Path(value)
    require(p.is_absolute() and str(p) == str(p.absolute()) and '..' not in p.parts,
            'absolute lexical path without parent traversal')
    require(p.is_file() and not p.is_symlink() and p.resolve() == p, 'regular non-symlink source path')
    return p


def sha(value, cap=CAP):
    p = path(value)
    require(p.stat().st_size <= cap, 'bounded source bytes')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def decode(data):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'duplicate JSON key')
            d[k] = v
        return d
    def constant(x):
        raise ValueError('nonfinite JSON ' + x)
    v = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    raw(v)
    return v


def read(p):
    require(path(p).stat().st_size <= CAP, 'bounded JSON')
    return decode(Path(p).read_bytes())


class Sources:
    def __init__(self):
        self.bindings = {}

    def bind(self, p, h):
        p = str(path(p))
        require(type(h) is str and len(h) == 64 and sha(p) == h, 'source SHA: ' + p)
        require(p not in self.bindings or self.bindings[p] == h, 'conflicting source binding')
        self.bindings[p] = h
        return p

    def load(self, ref):
        return read(self.bind(ref['path'], ref['sha256']))

    def finish(self):
        require(all(sha(p) == h for p, h in self.bindings.items()), 'source closure changed during build')


def journal(p, report, sources):
    p = sources.bind(p, report['journal_sha256'])
    tail = None
    events = []
    with Path(p).open('rb') as f:
        for i, line in enumerate(f):
            require(len(line) <= 16 * 1024**2 and line.endswith(b'\n'), 'bounded complete journal line')
            e = decode(line)
            require(line == raw(e) + b'\n' and type(e['index']) is int and e['index'] == i
                    and e['previous_sha256'] == tail
                    and e['event_sha256'] == digest({k: v for k, v in e.items() if k != 'event_sha256'}),
                    'entire original event hashchain')
            tail = e['event_sha256']
            events.append(e)
    require(events and tail == report['journal_chain_tail'], 'journal tail')
    require(events[0]['kind'] == 'run_started' and events[-1]['kind'] == 'run_finished'
            and events[0]['prepare_sha256'] == report['execution_binding']['prepare_sha256']
            and same(events[0]['contract'], report['contract'])
            and all(same(events[-1][k], report[k]) for k in ('status', 'counts', 'errors')),
            'complete journal bookends')
    # Exact nested dispatch/terminal ledger, independently reconstructed.
    intents, terminals, stack = {}, {}, []
    for e in events:
        if e['kind'] not in ('operation_dispatch_intent', 'operation_returned', 'operation_failed'):
            continue
        op = e['receipt']; n = op['index']
        require(type(n) is int, 'operation integer index')
        if e['kind'] == 'operation_dispatch_intent':
            require(n == len(intents) and op['status'] == 'pending', 'contiguous dispatch')
            intents[n] = op; stack.append(n)
        else:
            require(stack and stack.pop() == n and n not in terminals, 'nested terminal order')
            pending = dict(op, status='pending'); pending.pop('error', None)
            require(same(pending, intents[n]) and op['status'] ==
                    ('returned' if e['kind'] == 'operation_returned' else 'failed'), 'dispatch terminal identity')
            terminals[n] = op
    require(not stack and len(intents) == len(terminals) and
            same([terminals[i] for i in range(len(terminals))], report['actual_operation_receipts']),
            'whole operation ledger')
    counts = {k: dict(attempted=0, completed=0) for k in report['counts']}
    for op in terminals.values():
        counts[op['resource']]['attempted'] += 1
        counts[op['resource']]['completed'] += op['status'] == 'returned'
    require(counts == report['counts'], 'whole operation counts')
    require(same([e['receipt'] for e in events if e['kind'] == 'branch_closed'],
                 [r for r in report['rows'] if r['status'] != 'not_started']), 'whole branch closures')
    return events


def f32bits(rows, n):
    require(type(rows) is list and len(rows) == n and all(type(r) is list and len(r) == 2 for r in rows),
            'literal raw Nx2')
    result = []
    for row in rows:
        require(all(type(x) in (int, float) and math.isfinite(x) for x in row), 'finite raw')
        b = struct.pack('<2f', *row)
        require(same(list(struct.unpack('<2f', b)), row), 'raw values already float32 including signed zero')
        result.append(b.hex())
    return result


def physical(kind, authorities, stage_bindings, sources):
    refs = {k: authorities[kind + '_' + k] for k in ('prepare', 'report', 'completion', 'receipt')}
    p, r, c, s = (sources.load(refs[k]) for k in ('prepare', 'report', 'completion', 'receipt'))
    for ref in refs.values():
        require(stage_bindings.get(ref['path']) == ref['sha256'], 'stageA original physical authority join')
    require(s['errors'] == r['errors'] == [] and same(s['rows'], r['rows'])
            and same(s['counts'], r['counts']) and same(s['completion'], c)
            and s['completion_sha256'] == refs['completion']['sha256']
            and s['physical_report_sha256'] == c['report_sha256'] == refs['report']['sha256']
            and c['prepared'] == r['execution_binding']['prepared'] == refs['prepare']['path']
            and c['prepare_sha256'] == r['execution_binding']['prepare_sha256'] == refs['prepare']['sha256'],
            'original complete MAIN receipt/report/completion/prepare exact join')
    jp = str(Path(refs['report']['path']).parent / 'journal.jsonl')
    require(s['bindings'].get(jp) == stage_bindings.get(jp) == r['journal_sha256']
            and c['artifact_sha256']['journal.jsonl'] == r['journal_sha256'], 'journal inherited authentication')
    for key, ref in refs.items():
        if key == 'receipt':
            continue  # A returned receipt cannot contain its own file SHA.
        require(s['bindings'].get(ref['path']) == ref['sha256'], 'original physical receipt binding')
    src = authorities[kind + '_source']; sources.bind(src['path'], src['sha256'])
    return p, r, s, journal(jp, r, sources), refs


def extract(index, order, bridge_row, task, p, r, s, events, refs, sources):
    root, role, kind, slot = order
    row = r['rows'][slot]; plan = p['plans'][slot]; ref = p['prefix_reference'][slot]
    require(row['status'] == 'horizon_completed' and row['root_slot'] == root and row['index'] == slot
            and row['own_returned'] == row['warmup_returned'] == 10 and row['PNG'] == 66
            and same(row['plan'], plan), 'fixed completed endpoint')
    require(all(bridge_row[k] == v for k, v in dict(index=index, root_slot=root, role=role,
            source_kind=kind, source_slot=slot, checkpoint=10).items())
            and bridge_row['source_report'] == refs['report'] and bridge_row['source_receipt'] == refs['receipt']
            and bridge_row['selection_digest'] == row['selection_digest'], 'fixed stageA source identity')
    scoped = [e for e in events if e.get('branch') == slot]
    for e in scoped:
        require(all(same(e[k], row[k]) for k in ('root_slot', 'arm', 'condition', 'selection_digest')),
                'all scoped original event identity')
    obs = [e for e in scoped if e['kind'] == 'fresh_observation']
    expected = [(phase, step, overlay) for phase in ('warmup', 'plan') for step in range(11) for overlay in range(3)]
    require([(e['receipt']['phase'], e['receipt']['step'], e['receipt']['overlay_index']) for e in obs] == expected,
            'exact all66 observation order')
    selected = []
    png_refs = []
    for e in obs:
        f = e['receipt']
        require(same(e['pre'], e['post']) and same(e['post']['snapshot'], f['snapshot'])
                and same(f['state'], f['snapshot']['state']) and len(f['state']) == 7
                and f['condition'] == ('on' if f['overlay_index'] == 1 else 'off'), 'complete inert render snapshot')
        require(struct.pack('<7d', *f['state']).hex() == f['snapshot']['state_float64_bits'], 'state exact float64 bits')
        require(Path(f['png']).name == f['png'] and f['png'] not in ('.', '..'), 'local original PNG name')
        fp = str(Path(refs['report']['path']).parent / f['png'])
        require(s['bindings'].get(fp) == f['file_sha256'], 'original receipt PNG binding')
        sources.bind(fp, f['file_sha256'])
        png_refs.append(dict(event_index=e['index'], path=fp, sha256=f['file_sha256'], pixel_sha256=f['pixel_sha256']))
        if f['overlay_index'] == 0:
            selected.append(f)
    require(same(row['trajectory'], selected), 'whole original off trajectory')
    prefix = [{k: e[k] for k in ('pre', 'post', 'receipt')} for e in obs
              if e['receipt']['phase'] == 'warmup' or e['receipt']['step'] <= 5]
    require(same(row['prefix51'], dict(complete=True, frames=prefix)) and ref['complete'] is True
            and len(ref['frames']) == 51, 'full prefix51 exact journal link')
    normalizations = []
    for i, (actual, old) in enumerate(zip(prefix, ref['frames'], strict=True)):
        require(all(same(actual[k], old[k]) for k in ('pre', 'post')), 'prefix original complete pre/post')
        aa, bb = deepcopy(actual['receipt']), deepcopy(old['receipt'])
        a_png, b_png = aa.pop('png'), bb.pop('png')
        require(same(aa, bb), 'prefix only enumerated receipt.png transportation difference')
        normalizations.append(dict(actual_pointer=f'/observations/{i}/receipt/png',
            reference_pointer=f'/prefix_reference/frames/{i}/receipt/png', actual=a_png, reference=b_png))
    cps = [{k: f[k] for k in ('step', 'png', 'file_sha256', 'pixel_sha256', 'state', 'snapshot')}
           for f in selected if f['phase'] == 'plan' and f['step'] in (0, 5, 10)]
    require(same(row['checkpoints'], cps) and row['missing_checkpoints'] == [], 'exact cp0/5/10')
    require(all(same(cps[1][k], ref['checkpoint'][k]) for k in cps[1] if k != 'png'), 'reference cp5 checkpoint')
    transitions = [e for e in scoped if e['kind'] == 'returned_transition']
    require([(e['phase'], e['step']) for e in transitions] == [(ph, j) for ph in ('warmup', 'plan') for j in range(1, 11)],
            'all20 actual native transitions')
    ops = [op for op in r['actual_operation_receipts'] if op['branch'] == slot and op['resource'] == 'public_step']
    warm = ref['warm_raw10']; commands = plan['commands']
    bits = f32bits(warm, 10) + f32bits(commands, 10)
    require(f32bits(ref['prefix_raw5'], 5) == bits[10:15]
            and f32bits(ref['raw10'], 10) == bits[5:15]
            and f32bits(plan['prefix_commands'], 5) == bits[10:15], 'original cp5 causal raw/history reference')
    require(len(ops) == 20, 'full warm/control public returns')
    for j, (op, tr, b) in enumerate(zip(ops, transitions, bits, strict=True)):
        ph, step = ('warmup' if j < 10 else 'plan'), j % 10 + 1
        require(op['phase'] == ph and op['step'] == step and op['status'] == 'returned'
                and op['raw_bits'] == tr['requested_bits'] == tr['actual_bits'] == b
                and tr['actual_dtype'] == '<f4', 'raw requested/actual public bits')
        frames = [e for e in obs if e['receipt']['phase'] == ph and e['receipt']['step'] == step]
        require(all(same(e['receipt'][k], tr[k]) for e in frames for k in ('state', 'snapshot')),
                'all66 snapshots connected to actual native transitions')
        returned = next(e for e in events if e['kind'] == 'operation_returned' and e['receipt']['index'] == op['index'])
        require(returned['index'] < tr['index'] < frames[0]['index'], 'causal return/render order')
    responses = [e['response'] for e in scoped if e['kind'] == 'own_response']
    require(same(responses, row['responses']) and [x['step'] for x in responses] == list(range(1, 11)), 'all10 actual responses')
    for x, tr in zip(responses, transitions[10:], strict=True):
        require(all(same(x[k], tr[k]) for k in ('step', 'state', 'reward', 'terminated', 'truncated')), 'response native return')
    guards = [e for e in scoped if e['kind'] == 'cp5_guard']
    require(len(guards) == 1 and guards[0]['accepted'] is True and guards[0]['error'] is None,
            'original durable accepted cp5 guard')
    suffix_intent = next(e for e in events if e['kind'] == 'operation_dispatch_intent'
                         and e['receipt']['index'] == ops[15]['index'])
    prefix_indices = [e['index'] for e in obs if e['receipt']['phase'] == 'warmup' or e['receipt']['step'] <= 5]
    require(max(prefix_indices) < guards[0]['index'] < suffix_intent['index'], 'original51 guard before control6')
    frame = bridge_row['source']; cp10 = cps[-1]
    require(all(same(frame[k], cp10[k]) for k in cp10) and frame['path'] == png_refs[-3]['path']
            and same(bridge_row['native_response'], responses[-1]) and same(responses[-1]['state'], cp10['state'])
            and same(bridge_row['causal_raw10'], commands)
            and bridge_row['causal_raw10_bits'] == ''.join(bits[10:])
            and same(task['source'], frame) and same(task['causal_raw10'], commands), 'stageA actual cp10/raw/response/task link')
    require(all(task[k] == v for k, v in dict(index=index, root_slot=root, role=role).items()), 'stageA task fixed identity')
    histories = deepcopy(ref['history_frames'][1:])
    histories.append(dict(absolute_step=20, plan_step=10, png=frame['path'],
        **{k: frame[k] for k in ('file_sha256', 'pixel_sha256', 'state', 'snapshot')}))
    require(same(bridge_row['history_sources'], histories) and same(task['identity']['history_sources'], histories),
            'complete stageA abs10/15/20 history identity')
    placement = [e for e in scoped if e['kind'] == 'placement_recipe']
    goal_events = [e for e in scoped if e['kind'] == 'goal_attached']
    require(placement and goal_events, 'original placement/goal evidence')
    return dict(index=index, root_slot=root, role=role, source_kind=kind, source_slot=slot,
        source_authorities=refs, source_row=deepcopy(row), source_root=deepcopy(p['source_roots'][root]),
        source_slot_metadata=deepcopy(p['source_slots'][root]), prefix_reference=deepcopy(ref),
        observations=obs, scoped_original_events=scoped, public_step_receipts=ops,
        warm_raw10=warm, control_raw10=commands, raw_float32_bits=bits,
        actual_response10=responses[-1], cp10=cp10, cp5_guard=guards[0],
        placement_events=placement, goal_events=goal_events, png_sources=png_refs,
        stageA_row=bridge_row, stageA_task=task, transport_normalizations=normalizations,
        checkpoint_transport_normalization=dict(actual_pointer='/source_row/checkpoints/1/png',
            reference_pointer='/prefix_reference/checkpoint/png', actual=cps[1]['png'], reference=ref['checkpoint']['png']),
        unknown_semantics=dict(rng='unknown: not recorded in original snapshot',
            public_state='unknown: no serialized public_state contract',
            contact_cache='unknown: not recorded; witness does not certify a hydrate restart'),
        guard_scope='all saved fields retained; only listed receipt.png path mapping; no other ignores')


def material(authorities):
    require(set(authorities) == set(PINS), 'exact stageA four authority descriptors')
    sources = Sources()
    ap, ar, ac, ass = (sources.load(authorities[k]) for k in ('prepare', 'report', 'completion', 'receipt'))
    require(ar['errors'] == [] and ar['fixed_denominator'] == ass['fixed_denominator'] == 5
            and ar['prepared'] == ac['prepared'] == authorities['prepare']['path']
            and ar['prepare_sha256'] == ac['prepare_sha256'] == authorities['prepare']['sha256']
            and ass['report_sha256'] == ac['report_sha256'] == authorities['report']['sha256']
            and ass['completion_sha256'] == authorities['completion']['sha256'] and same(ass['completion'], ac)
            and same(ap['authorities'], ass['source_authorities']), 'stageA full four-way exact join')
    require(len(ap['rows']) == len(ar['rows']) == len(ass['tasks']) == 5, 'all five bridge rows/tasks')
    for key, ref in authorities.items():
        if key == 'receipt':
            continue
        require(ass['bindings'].get(ref['path']) == ref['sha256'], 'stageA binding identity')
    for ref in ap['authorities'].values():
        require(ass['bindings'].get(ref['path']) == ref['sha256'], 'all stageA source authority bindings')
        sources.bind(ref['path'], ref['sha256'])
    data = {kind: physical(kind, ap['authorities'], ass['bindings'], sources) for kind in ('C', 'ZP')}
    rows = []
    for i, order in enumerate(ORDER):
        br = ar['rows'][i]
        require(all(same(br[k], v) for k, v in ap['rows'][i].items() if k not in ('latent', 'status')),
                'stageA prepare/report source fields unchanged')
        rows.append(extract(i, order, br, ass['tasks'][i], *data[order[2]][:4], data[order[2]][4], sources))
        for h in br['history_sources']:
            require(ass['bindings'].get(h['png']) == h['file_sha256'], 'stageA full history PNG bindings')
            sources.bind(h['png'], h['file_sha256'])
    own = [Path(__file__).resolve(), ROOT / 'tests' / ('test_' + NAME + '.py'), ROOT / 'innerdocs' / (NAME + '.md')]
    for p in own:
        sources.bind(str(p), sha(p))
    sources.finish()
    return dict(schema=NAME + '.witness', authorities=authorities, bindings=sources.bindings,
        actual_MAIN_authority=all(authorities[k]['sha256'] == PINS[k] for k in PINS),
        authentication='inherited MAIN complete consumes; local saved hashchain/joins only, NOT fullstrict',
        fixed_denominator=5, observations_per_endpoint=66, prefix_per_endpoint=51, rows=rows,
        next_adapter_budget=NEXT_JOB_BUDGET, stageB_authority=None, production_adapter_ready=False, **FLAGS)


def write_new(output, value):
    p = Path(output)
    require(p.is_absolute() and p.parent.is_dir() and not p.exists() and not p.is_symlink(), 'exclusive absolute output')
    body = raw(value) + b'\n'
    require(len(body) <= CAP, 'bounded full witness, no lossy truncation')
    with p.open('xb') as f:
        require(f.write(body) == len(body), 'complete durable output')
        f.flush(); os.fsync(f.fileno())
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(p), sha256=sha(p), bytes=len(body))


def build(output, authorities):
    require(not Path(output).exists(), 'do not overwrite existing witness')
    return write_new(output, material(authorities))


def verify(output, expected):
    require(sha(output) == expected, 'whole witness SHA')
    saved = read(output)
    require(same(saved, material(saved['authorities'])) and sha(output) == expected, 'whole saved witness reconstruction')
    return dict(status='all5_full66_witnesses_reconstructed', sha256=expected,
                actual_MAIN_authority=saved['actual_MAIN_authority'], **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--output', required=True)
    for k in PINS:
        b.add_argument('--' + k, required=True)
    v = sub.add_parser('verify'); v.add_argument('--witness', required=True); v.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        refs = {k: dict(path=str(path(getattr(args, k))), sha256=PINS[k]) for k in PINS}
        result = build(args.output, refs)
    else:
        saved = read(args.witness)
        require(all(saved['authorities'][k]['sha256'] == PINS[k] for k in PINS), 'CLI requires actual MAIN pins')
        result = verify(args.witness, args.sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
