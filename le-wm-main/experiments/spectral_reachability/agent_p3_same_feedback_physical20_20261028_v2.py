"""Saved-only v2 hardening of frozen MPC20 v1; no producer/model/physics API."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import io
import json
import os
from pathlib import Path
import sysconfig

import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as base

NAME = 'agent_p3_same_feedback_physical20_20261028_v2'
require, sha, digest = base.require, base.sha, base.digest
COMMON = {'index', 'kind', 'previous_sha256', 'event_sha256'}
FIELDS = {
    'run_started': {'contract'}, 'run_finished': {'status', 'counts', 'nodes_sha256'},
    'operation_attempted': {'operation_id', 'episode_id', 'resource', 'units', 'region'},
    'operation_completed': {'operation_id', 'episode_id', 'resource', 'units', 'region'},
    'fresh_observation': {'episode_id', 'receipt'},
    'guard_passed': {'episode_id', 'phase', 'step', 'canonical_agent_pixels'},
    'warmup_return': {'episode_id', 'step', 'command', 'terminated', 'truncated'},
    'authorized_dispatch': {'episode_id', 'step', 'packet_boundary', 'legacy_gate_ordinal', 'command'},
    'control_return': {'episode_id', 'step', 'command', 'terminated', 'truncated', 'reward'},
    'handler_event': {'namespace', 'event'},
    'boundary_encoded': {'episode_id', 'actual_step', 'mapping_sha256', 'encoding_sha256'},
    'boundary_sealed': {'episode_id', 'boundary', 'seal_sha256'},
    'episode_closed': {'receipt'},
    'invalid_render': {'episode_id', 'shape', 'dtype', 'raw_render', 'state', 'snapshot'},
}
REGIONS = {
    'constructor': {'setup'}, 'reset': {'setup'}, 'set_state': {'setup'},
    'goal_render': {'setup'}, 'goal_setter': {'setup', 'goal_bind'},
    'goal_bind': {'goal_bind'}, 'public_step': {'warmup', 'prefix', 'suffix'},
    'physics_step': {'setup', 'warmup', 'prefix', 'suffix'},
    'explicit_render': {'setup', 'warmup', 'goal_bind', 'prefix', 'suffix'},
    'encoder': {'encoding'}, 'projector': {'encoding'}, 'close': {'closing'},
}


def regular(path, cap):
    p = Path(path).absolute()
    require('..' not in p.parts and p.resolve() == p and p.is_file() and
            not any(q.is_symlink() for q in (p, *p.parents)) and p.stat().st_size <= cap,
            'bounded canonical regular saved file')
    return p


def read_json(path, cap):
    value = json.loads(regular(path, cap).read_bytes(), object_pairs_hook=base.signal.unique_pairs)
    base.raw(value)  # finite JSON, including unused fields
    return value


def load_events(output, report):
    path = regular(Path(output) / 'journal.jsonl', base.contract()['output_bytes'])
    require(sha(path) == report['journal_sha256'], 'physical journal external binding')
    payload = path.read_bytes()
    require(payload.endswith(b'\n'), 'complete physical journal line')
    events = [json.loads(line, object_pairs_hook=base.signal.unique_pairs) for line in payload.splitlines()]
    previous = None
    for index, event in enumerate(events):
        base.raw(event)
        require(type(event) is dict and event.get('kind') in FIELDS, 'unknown physical event kind')
        require(set(event) == COMMON | FIELDS[event['kind']], 'exact physical event schema')
        require(type(event['index']) is int and event['index'] == index and event['previous_sha256'] == previous
                and event['event_sha256'] == base.old.source.digest(
                    {k: v for k, v in event.items() if k != 'event_sha256'}), 'physical event chain')
        previous = event['event_sha256']
    require(events and previous == report['journal_chain_tail'] and events[0]['kind'] == 'run_started'
            and events[-1]['kind'] == 'run_finished', 'physical terminal chain')
    return events


def census(events, report):
    """Account for EVERY event, every resource, every episode, including prefixes.

    Clean native-stop is complete engineering evidence, not completed20. Partial
    prefixes retain attempts/pending and are never granted complete census.
    """
    rows = report['rows']; nodes = report['nodes']; ids = [r['episode_id'] for r in base.roster()]
    require([r['episode_id'] for r in rows] == ids, 'fixed four episode denominator')
    require([(n['episode_id'], n['boundary']) for n in nodes] ==
            [(n['episode_id'], n['boundary']) for n in base.node_roster()], 'fixed sixteen node denominator')
    clean = report['status'] == 'known_same_feedback_recorded' and not report['errors'] and not report['pending']
    require(events[0]['contract'] == base.contract() and events[-1]['status'] == report['status']
            and events[-1]['counts'] == report['counts'] and events[-1]['nodes_sha256'] == digest(nodes),
            'run bookends bind original contract/counts/nodes')
    require(Counter(e['kind'] for e in events)['run_started'] == 1 and
            Counter(e['kind'] for e in events)['run_finished'] == 1, 'unique run bookends')
    resources = base.contract()['physical_caps'] | base.contract()['visual_caps']
    totals = {k: dict(attempted=0, completed=0) for k in resources}
    per = {eid: {k: dict(attempted=0, completed=0) for k in resources} for eid in ids}
    grouped = {eid: [] for eid in ids}; pending = {}; serial = -1
    active = 0; stack = []; region_counts = {eid: Counter() for eid in ids}
    for e in events[1:-1]:
        kind = e['kind']
        require(kind in FIELDS and set(e) == COMMON | FIELDS[kind], 'known exact physical event schema')
        if kind == 'episode_closed':
            require(active < 4 and e['receipt'] == rows[active], 'exact ordered episode close, no duplicate')
            grouped[ids[active]].append(e); active += 1; continue
        require(active < 4, 'no event after four episode closes')
        eid = ids[active]
        if kind == 'handler_event':
            require(type(e['namespace']) is str and e['namespace'] in
                    [eid + f'_b{b:02}_encode' for b in (0, 5, 10, 15, 20)], 'own current encoder namespace')
            native_event = e['event']
            require(type(native_event) is dict and set(native_event) == {'kind', 'native'} and
                    native_event['kind'] == 'native_event', 'known visual handler event')
            value = native_event['native']
            require(type(value) is dict and set(value) == {'kind', 'entry'} and
                    value['kind'] in ('module_attempt', 'module_return'), 'known visual native event')
            entry = value['entry']
            require(type(entry) is dict and set(entry) == {'index', 'image', 'resource', 'completed'} and
                    type(entry['completed']) is bool and entry['completed'] == (value['kind'] == 'module_return') and
                    type(entry['image']) is int and entry['image'] == 0 and type(entry['index']) is int and
                    (entry['index'], entry['resource']) in ((0, 'encoder'), (1, 'projector')),
                    'exact visual native entry')
        else:
            require(e.get('episode_id') == eid, 'event belongs to active ordered episode')
        grouped[eid].append(e)
        if kind not in ('operation_attempted', 'operation_completed'):
            continue
        k = e['resource']; units = e['units']; ident = e['operation_id']
        require(k in resources and type(units) is int and units == 1 and type(ident) is int and
                e['region'] in REGIONS[k], 'native single-call units and legal resource region')
        operation = {k: e[k] for k in FIELDS[kind]}
        if kind == 'operation_attempted':
            require(ident == serial + 1, 'consecutive unique physical operation IDs')
            serial = ident; pending[ident] = operation; stack.append(ident); axis = 'attempted'
        else:
            require(stack and stack[-1] == ident and pending.pop(ident, None) == operation,
                    'nested native operation matching return')
            stack.pop(); axis = 'completed'
        totals[k][axis] += units; per[eid][k][axis] += units
        region_counts[eid][k, e['region'], axis] += units
    require(totals == report['counts'] and {str(k): v for k, v in pending.items()} ==
            {str(k): v for k, v in report['pending'].items()}, 'all-resource physical census and pending')
    for k, c in totals.items():
        require(0 <= c['completed'] <= c['attempted'] <= resources[k], 'all physical/visual caps')
    if not clean:
        return dict(complete=False, event_counts=dict(Counter(e['kind'] for e in events)),
                    counts=totals, per_episode=per, reason='partial prefix; full census not certified')
    require(active == 4 and not stack and not pending, 'all four episodes closed with no pending operations')
    for row in rows:
        eid = row['episode_id']; es = grouped[eid]; by = defaultdict(list)
        for e in es: by[e['kind']].append(e)
        require(not by['invalid_render'], 'clean run has no invalid RGB event')
        warm, control, enc = row['warmup_returned'], row['control_returned'], row['encodings']
        require(type(warm) is int and 0 <= warm <= 10 and type(control) is int and 0 <= control <= 20,
                'bounded actual returned steps')
        expected_observations = [('warmup', b) for b in range(warm + 1)]
        if row['initial_exact']: expected_observations += [('plan', b) for b in range(control + 1)]
        observations = by['fresh_observation']
        require([(e['receipt']['phase'], e['receipt']['step']) for e in observations] == expected_observations
                and row['PNG'] == len(observations), 'all actual observations and PNG census')
        require([(e['phase'], e['step']) for e in by['guard_passed']] == expected_observations,
                'one guard for every actual observation')
        for observation, guard in zip(observations, by['guard_passed'], strict=True):
            require(observation['index'] < guard['index'] and observation['receipt']['observed_event_index'] ==
                    observation['index'] and type(guard['canonical_agent_pixels']) is int and
                    guard['canonical_agent_pixels'] > 0, 'observation/guard order and real positive guard')
        for kind, n in (('warmup_return', warm), ('control_return', control), ('authorized_dispatch', control)):
            require([e['step'] for e in by[kind]] == list(range(1, n + 1)), 'all returned/authorized step census')
        require([e['actual_step'] for e in by['boundary_encoded']] == [e['step'] for e in enc],
                'all and only current encoding events')
        sealed = [n for n in nodes if n['episode_id'] == eid and n['status'] == 'sealed']
        require([e['boundary'] for e in by['boundary_sealed']] == [n['boundary'] for n in sealed],
                'all and only sealed node events')
        require([c['step'] for c in row['checkpoints']] ==
                [b for phase, b in expected_observations if phase == 'plan' and b in (0, 5, 10, 15, 20)],
                'complete checkpoint census')
        require(row['checkpoints'] == [e['receipt'] for e in observations if e['receipt']['phase'] == 'plan'
                and e['receipt']['step'] in (0, 5, 10, 15, 20)], 'checkpoint receipts equal actual observations')
        for event, node in zip(by['boundary_sealed'], sealed, strict=True):
            require(event['seal_sha256'] == node['seal']['seal_sha256'], 'all seal events bind actual sealed node')
        # Pinned native constructor has no implicit reset. Each explicit reset
        # invokes two setters, one goal render and two integration substeps.
        want = dict(constructor=1, reset=2, set_state=4, goal_render=2,
                    goal_setter=2 + int(row['initial_exact']), goal_bind=int(row['initial_exact']),
                    public_step=warm + control, physics_step=4 + 10 * (warm + control),
                    explicit_render=len(observations), close=1, encoder=len(enc), projector=len(enc))
        require(per[eid] == {k: dict(attempted=v, completed=v) for k, v in want.items()},
                'independent per-episode ALL resource census')
        regions = {('constructor', 'setup'): 1, ('reset', 'setup'): 2, ('set_state', 'setup'): 4,
            ('goal_render', 'setup'): 2, ('goal_setter', 'setup'): 2,
            ('goal_setter', 'goal_bind'): int(row['initial_exact']), ('goal_bind', 'goal_bind'): int(row['initial_exact']),
            ('public_step', 'warmup'): warm, ('physics_step', 'setup'): 4, ('physics_step', 'warmup'): 10 * warm,
            ('explicit_render', 'setup'): 1, ('explicit_render', 'warmup'): warm,
            ('explicit_render', 'goal_bind'): int(row['initial_exact']), ('close', 'closing'): 1,
            ('encoder', 'encoding'): len(enc), ('projector', 'encoding'): len(enc)}
        for region, steps in (('prefix', min(control, 5)), ('suffix', max(control - 5, 0))):
            regions.update({('public_step', region): steps, ('physics_step', region): 10 * steps,
                            ('explicit_render', region): steps})
        require(region_counts[eid] == Counter({(k, region, axis): n for (k, region), n in regions.items()
                for axis in ('attempted', 'completed') if n}), 'all-resource per-region native census')
        expected_handlers = []
        for encoded, event in zip(enc, by['boundary_encoded'], strict=True):
            b = encoded['step']; ns = eid + f'_b{b:02}_encode'
            require(event['encoding_sha256'] == encoded['receipt_sha256'], 'encoding event binds exact receipt')
            local = [h for h in by['handler_event'] if h['namespace'] == ns]
            require(len(local) == 4, 'exact four visual handler events per encoding')
            for i, (resource, action) in enumerate((('encoder', 'attempt'), ('encoder', 'return'),
                                                   ('projector', 'attempt'), ('projector', 'return'))):
                h = local[i]; native = dict(kind='module_' + action, entry=dict(
                    completed=action == 'return', image=0, index=i // 2, resource=resource))
                require(h['event'] == dict(kind='native_event', native=native) and h['index'] < event['index'],
                        'exact visual handler native lifecycle')
                expected_handlers.append(h['index'])
            ops = [op for op in es if op['kind'] in ('operation_attempted', 'operation_completed') and
                   op['resource'] in ('encoder', 'projector') and local[0]['index'] < op['index'] < event['index']]
            require([(op['kind'], op['resource']) for op in ops] ==
                    [('operation_attempted', 'encoder'), ('operation_completed', 'encoder'),
                     ('operation_attempted', 'projector'), ('operation_completed', 'projector')],
                    'each visual handler has exact metered native calls')
            require(all(local[i]['index'] < ops[i]['index'] <
                    (local[i + 1]['index'] if i < 3 else event['index']) for i in range(4)),
                    'visual handler and metering exact interleave')
        require(expected_handlers == [h['index'] for h in by['handler_event']], 'no stray visual handler events')
        # Every kind has an exact expected census; operations include ALL axes.
        wanted_kinds = dict(operation_attempted=sum(want.values()), operation_completed=sum(want.values()),
            fresh_observation=len(observations), guard_passed=len(observations), warmup_return=warm,
            authorized_dispatch=control, control_return=control, handler_event=4 * len(enc),
            boundary_encoded=len(enc), boundary_sealed=len(sealed), episode_closed=1)
        require(Counter(e['kind'] for e in es) == Counter({k: v for k, v in wanted_kinds.items() if v}),
                'full physical event-kind census')
    return dict(complete=True, event_counts=dict(Counter(e['kind'] for e in events)), counts=totals, per_episode=per,
                per_region={eid: [dict(resource=k, region=region, axis=axis, units=n)
                    for (k, region, axis), n in sorted(counts.items())] for eid, counts in region_counts.items()})


def image_contract(prepared):
    """AST/literal reads only; original source SHA closure, no foreign import."""
    pure = Path(sysconfig.get_paths()['purelib'])
    paths = [base.ROOT / 'experiments/run_pusht_counterfactual.py',
             pure / 'stable_pretraining/data/dataset_stats.py',
             pure / 'torchvision/transforms/v2/_misc.py',
             pure / 'torchvision/transforms/v2/functional/_misc.py',
             pure / 'torchvision/transforms/v2/functional/_geometry.py']
    bindings = {}
    for p in paths:
        regular(p, 512 * 1024)
        require(str(p) in prepared['sources'] and sha(p) == prepared['sources'][str(p)],
                'original transform/statistics implementation source SHA')
        bindings[str(p)] = sha(p)
    tree = ast.parse(paths[0].read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'image_transform')
    expected = ast.parse('''def image_transform():
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])
''').body[0]
    require(ast.dump(fn) == ast.dump(expected), 'exact original ToDtype/Normalize/Resize224 ABI')
    nodes = [n.value for n in ast.parse(paths[1].read_text()).body if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == 'ImageNet' for t in n.targets)]
    require(len(nodes) == 1 and isinstance(nodes[0], ast.Call) and isinstance(nodes[0].func, ast.Name) and
            nodes[0].func.id == 'dict' and not nodes[0].args, 'explicit original ImageNet literal')
    stats = {kw.arg: ast.literal_eval(kw.value) for kw in nodes[0].keywords}
    require(set(stats) == {'mean', 'std'} and all(np.asarray(stats[k]).shape == (3,) and
            np.isfinite(stats[k]).all() for k in stats) and np.all(np.asarray(stats['std']) > 0),
            'finite original RGB normalization statistics')
    return dict(statistics=stats, sources=bindings, resize='native224 identity', dtype='float32',
                layout=[1, 1, 3, 224, 224], arithmetic='cast; mul(float32(1/255)); sub(mean); div(std)')


def preprocess(rgb, stats):
    require(rgb.dtype == np.uint8 and rgb.shape == (224, 224, 3), 'native224 RGB; no implicit resize/convert')
    mean = np.asarray(stats['mean'], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(stats['std'], dtype=np.float32).reshape(3, 1, 1)
    value = rgb.transpose(2, 0, 1).astype(np.float32)
    np.multiply(value, np.float32(1.0 / 255), out=value)
    np.subtract(value, mean, out=value)
    np.divide(value, std, out=value)
    return value[None, None]


def audit_rgb(output, report, prepared):
    """Decode own saved PNG and recompute every saved float32 input bit."""
    output = Path(output).absolute(); contract = image_contract(prepared); audited = []
    for row in report['rows']:
        eid = row['episode_id']
        for encoded in row['encodings']:
            b = encoded['step']; ns = eid + f'_b{b:02}'
            er = read_json(output / (ns + '_encoding.json'), 1024**2)
            require(base.old.source.digest(er) == encoded['receipt_sha256'] and er['actual_step'] == b,
                    'original encoding receipt digest')
            entry = er['mapping']['original_png']; path = regular(entry['path'], 1024**2)
            alias = regular(output / (ns + '_plan_05.png'), 1024**2)
            require(path == output / (eid + f'_plan_{b:02}.png') and sha(path) == sha(alias) == entry['file_sha256']
                    and er['mapping']['actual_step'] == b and er['mapping']['alias_png'] == str(alias)
                    and er['request']['png'] == dict(path=str(alias), file_sha256=entry['file_sha256'],
                                                   pixel_sha256=entry['pixel_sha256']), 'actual frame/alias/request binding')
            with Image.open(path) as image:
                require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224, 224), 'exact native RGB PNG')
                rgb = np.asarray(image).copy()
            require(base.old.array_sha(rgb) == entry['pixel_sha256'], 'actual decoded RGB pixels')
            npy = regular(output / (ns + '_encode') / 'preprocessed_pixels.npy', 1024**2)
            stream = io.BytesIO(npy.read_bytes()); value = np.load(stream, allow_pickle=False)
            expected = preprocess(rgb, contract['statistics'])
            require(stream.tell() == stream.getbuffer().nbytes and value.dtype == np.dtype('float32') and
                    value.shape == expected.shape and np.isfinite(value).all(), 'full preprocessing native array ABI')
            require(value.tobytes(order='C') == expected.tobytes(order='C'), 'every preprocessing float32 value/bit equals saved RGB')
            require(Path(er['response']['latent']['path']) == output / (ns + '_encode') / 'native192.npy',
                    'input and output share exact original encoder namespace')
            require(read_json(output / (ns + '_encode') / 'receipt.json', 1024**2) == er['response'],
                    'durable visual response binding')
            audited.append(dict(episode_id=eid, step=b, png=str(path), png_sha256=sha(path),
                pixel_sha256=entry['pixel_sha256'], encoding_receipt_sha256=encoded['receipt_sha256'],
                array=str(npy), array_file_sha256=sha(npy), array_sha256=base.old.array_sha(value),
                shape=list(value.shape), dtype=str(value.dtype), values_checked=int(value.size),
                exact_float32_bits=True, latent=er['response']['latent']))
    return dict(contract=contract, frames=len(audited), arrays=audited, independent_neural_replay=False,
                original_preprocessed_file_SHA_in_v1_receipt=False,
                binding='v2 recomputation binds saved input values to original PNG/encoding receipt; not retroactive producer attestation')


def consume(output, report_sha, prepared, prepare_sha):
    output = Path(output).absolute(); p = base.verify(prepared, prepare_sha)
    require(sha(regular(output / 'report.json', base.contract()['final_reserve_bytes'])) == report_sha,
            'external original report byte SHA')
    r = read_json(output / 'report.json', base.contract()['final_reserve_bytes'])
    require(r['schema'] == base.NAME + '.report', 'original v1 production report')
    events = load_events(output, r)
    full = census(events, r)  # unknown events fail even when v1 would preserve partial
    rgb = audit_rgb(output, r, p)
    inherited = base.consume(output, report_sha, prepared, prepare_sha)
    complete = inherited.get('complete') is True and full['complete']
    require(sha(output / 'report.json') == report_sha and sha(output / 'journal.jsonl') == r['journal_sha256'],
            'saved inputs unchanged across consumption')
    return dict(schema=NAME + '.consume', status='saved_v2_hardened_NOT_admitted' if complete else 'saved_partial_preserved',
        complete=complete, source_report_sha256=report_sha, prepare_sha256=prepare_sha,
        consumer_sources={str(Path(__file__).resolve()): sha(__file__), str(Path(base.__file__).resolve()): sha(base.__file__)},
        original_consume=inherited, physical_census=full, RGB_preprocessing=rgb,
        P1_admission=False, P3_admission=False, freshness=False, scientific_admission=False,
        independent_neural_replay=False, equal_FLOPs=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('consume', 'verify'))
    for name in ('output', 'report-sha', 'prepared', 'prepare-sha'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--receipt', help='NEW exclusive receipt JSON; never writes original output')
    args = parser.parse_args(argv)
    result = consume(args.output, args.report_sha, args.prepared, args.prepare_sha)
    if args.receipt:
        target = Path(args.receipt).absolute()
        require(target.parent == base.ROOT / 'results' and target.name.startswith(NAME) and
                not target.exists() and target.parent.resolve() == target.parent, 'new exclusive v2 receipt path')
        payload = base.raw(result)
        with target.open('xb') as f:
            require(f.write(payload) == len(payload), 'complete receipt write'); f.flush(); os.fsync(f.fileno())
        base.signal.fsync_dir(target.parent)
        print(base.raw(dict(receipt=str(target), sha256=sha(target), status=result['status'])).decode())
    else:
        print(base.raw(result).decode())


if __name__ == '__main__':
    main()
