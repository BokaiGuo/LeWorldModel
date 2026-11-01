"""Saved-only actual-cp5 task join. No producer import or upstream fullstrict."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

NAME = 'agent_p3_actual_cp5_task_builder_20261029_v1'
PLANNER = 'agent_p3_shared_panel_20261029_v1'
LIVE = 'agent_p3_fullcommit_live_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
ARMS = ('fixedH1_endpoints', 'fixedH1_midpoints', 'adaptive_endpoints', 'adaptive_midpoints')
CAPS = dict(prepared=16*1024**2, report=256*1024**2, receipt=32*1024**2)
EXPORT_CAP = 8*1024**2
OUTPUT_CAP = 16*1024**2
FLAGS = dict(P1_admission=False, P3_admission=False, C3=False, physical_release=False,
             freshness=False, independent_confirmation=False)
BINDINGS = ('root_sha256', 'task_id', 'task_sha256', 'task_contract_sha256',
            'blind_sha256', 'model_sha256', 'goal_sha256', 'projection_sha256',
            'normalizer_sha256', 'context_sha256', 'action_domain_sha256')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def hash_value(value):
    require(type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'literal SHA256')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON field')
            out[k] = v
        return out
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda x: require(False, 'nonfinite JSON'))
    canonical(value)
    return value


def read_pinned(path, expected, cap, *, compressed=False):
    hash_value(expected)
    p = Path(path).absolute()
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap,
            'bounded regular saved file')
    require(sha(p) == expected, 'saved file SHA mismatch')
    if compressed:
        with gzip.open(p, 'rb') as f:
            raw = f.read(cap+1)
    else:
        raw = p.read_bytes()
    require(len(raw) <= cap, 'decoded file cap')
    value = decode(raw)
    require(sha(p) == expected, 'saved file changed during read')
    return value


def array(value, shape, *, f32=True):
    def numeric(x):
        return all(numeric(y) for y in x) if type(x) is list else type(x) in (int, float)
    require(type(value) is list and numeric(value), 'literal numeric list, no bool')
    a = np.asarray(value, np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite exact array shape')
    if f32:
        require(np.array_equal(a, a.astype(np.float32).astype(np.float64)), 'exact float32; no repair')
        a = a.astype(np.float32)
    return a


def same(a, b, why):
    require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), why)


def context(c):
    require(type(c) is dict and set(c) == {'layout', 'context_length', 'latent_dim',
            'action_dim', 'latent_history', 'action_history'}, 'complete native context')
    require(c['layout'] == 'states_with_between_state_actions_v1' and
            all(type(c[k]) is int and c[k] == n for k, n in
                (('context_length', 3), ('latent_dim', 192), ('action_dim', 10))), 'native context dimensions')
    return array(c['latent_history'], (3, 192)), array(c['action_history'], (2, 10))


def normalize(task, raw):
    mean, std = array(task['action_mean'], (2,)), array(task['action_std'], (2,))
    require((std > 0).all() and (np.abs(raw) <= 1).all(), 'positive scale and legal rawbox')
    return ((raw-mean)/std).reshape(2, 10)


def task_check(t):
    require(type(t) is dict and set(t) == {'binding', 'root_context', 'goal', 'projection',
            'action_mean', 'action_std'}, 'complete task ABI')
    b = t['binding']
    require(type(b) is dict and set(b) == set(BINDINGS), 'complete task binding')
    for value in b.values():
        hash_value(value)
    context(t['root_context'])
    array(t['goal'], (192,), f32=False)
    same(array(t['projection'], (192, 192), f32=False), np.eye(192), 'original identity Q')
    array(t['action_mean'], (2,))
    require((array(t['action_std'], (2,)) > 0).all(), 'positive normalizer')
    domain = dict(raw_bounds=[-1, 1], packet_steps=5, action_dim=2,
                  normalization='original saved float32 subtract/divide; no /100 or clipping')
    require(b['context_sha256'] == digest(t['root_context']) and b['goal_sha256'] == digest(t['goal']) and
            b['projection_sha256'] == digest(t['projection']) and
            b['normalizer_sha256'] == digest([t['action_mean'], t['action_std']]) and
            b['action_domain_sha256'] == digest(domain), 'task content bindings')


def local_task(base, c, identity, namespace):
    t = deepcopy(base)
    t['root_context'] = deepcopy(c)
    t['binding'].update(context_sha256=digest(c), task_id=digest([base['binding']['task_id'], identity]),
                        task_sha256=digest([base, c, identity]), task_contract_sha256=digest([namespace, identity]))
    task_check(t)
    return t


def join(prepared, report, receipt, exports, authorities):
    """Pure fixture seam; production callers use resolve_actual_cp5_tasks for file authentication."""
    require(prepared['schema'] == PLANNER+'.prepare' and report['schema'] == PLANNER+'.report' and
            receipt['schema'] == PLANNER+'.consume', 'exact source schemas')
    require(receipt['complete'] is True and receipt['status'] == 'saved_shared_panel_verified_NOT_admitted',
            'completed MAIN strict receipt required')
    require(receipt['report_sha256'] == authorities['report']['sha256'] and
            receipt['prepare_sha256'] == report['prepare_sha256'] == authorities['prepared']['sha256'] and
            receipt['prepared'] == report['prepared'] == authorities['prepared']['path'], 'authority join')
    require(len(prepared['tasks']) == len(prepared['layers']) == len(exports) == 2, 'complete known2 roots')
    require(type(report['rows']) is list and len(report['rows']) == 8, 'all eight original branches')
    tasks, provenance = [], []
    selected = [n for n in report['nodes'] if type(n.get('boundary')) is int and n['boundary'] == 5]
    require(len(selected) == 8, 'all eight b5 nodes; no subset')
    for root in range(2):
        base, layer, export = prepared['tasks'][root], prepared['layers'][root], exports[root]
        task_check(base)
        require(type(layer['source_slot']) is int and layer['source_slot'] == root and
                layer['task_id'] == base['binding']['task_id'] and
                layer['task_sha256'] == base['binding']['task_sha256'], 'original layer/task identity')
        require(export['schema'] == 'p1_constructive_moving_history_encode_cuda_20261027_v1.task_export' and
                type(export['source_slot']) is int and export['source_slot'] == root and
                export['task_id'] == base['binding']['task_id'] and
                export['task_contract']['observed_steps'] == [0, 5, 10] and
                all(type(x) is int for x in export['task_contract']['observed_steps']) and
                export['history_semantics'] == 'actual_moving_raw_between_observed0_5_10', 'original warmup timing authority')
        original = export['task']
        require(original['task_contract_sha256'] == base['binding']['task_contract_sha256'], 'original task contract identity')
        for key in ('root_context', 'goal', 'projection', 'action_mean', 'action_std'):
            require(canonical(original[key]) == canonical(base[key]), 'original export/task content')
        warm = array(original['root_raw_action_history'], (10, 2))
        prefix = array(layer['commands'][:5], (5, 2))
        expected_raw = np.concatenate((warm[-5:], prefix))
        old_z, old_a = context(base['root_context'])
        same(old_a, normalize(base, warm), 'original warmup raw10 tokens')
        branch_refs, common = [], None
        for arm in ARMS:
            eid = f's{root}_{arm}'
            nodes = [n for n in selected if n['episode_id'] == eid]
            rows = [r for r in report['rows'] if r['episode_id'] == eid]
            audits = [a for a in receipt['audited'] if a['episode_id'] == eid and a['boundary'] == 5]
            require(len(nodes) == len(rows) == len(audits) == 1 and audits[0]['node_verified'] is True,
                    'unique branch/node/strict audit')
            n, row = nodes[0], rows[0]
            require(type(n['source_slot']) is int and n['source_slot'] == root and n['mode'] == arm and
                    n['status'] == 'sealed' and type(row['source_slot']) is int and row['source_slot'] == root and
                    row['mode'] == arm and row['prefix_exact'] is True, 'sealed branch cp5 identity')
            same(array(row['prefix']['commands'], (5, 2)), prefix, 'fixed executed prefix bits')
            require(row['prefix']['raw_float32_sha256'] == hashlib.sha256(prefix.tobytes()).hexdigest() and
                    row['prefix']['source_selection_sha256'] == layer['selection_sha256'], 'prefix authority')
            req = n['request']
            require(set(req) == {'schema', 'base_task_sha256', 'task', 'episode_id', 'source_slot',
                    'mode', 'boundary', 'raw10', 'observation'} and req['schema'] == PLANNER+'.node' and
                    type(req['source_slot']) is int and req['source_slot'] == root and req['episode_id'] == eid and
                    req['mode'] == arm and type(req['boundary']) is int and req['boundary'] == 5 and
                    req['base_task_sha256'] == digest(base), 'saved b5 request identity')
            t, ob = req['task'], req['observation']
            task_check(t)
            identity = {k: deepcopy(req[k]) for k in ('episode_id', 'source_slot', 'mode', 'boundary', 'raw10', 'observation')}
            require(canonical(t) == canonical(local_task(base, t['root_context'], identity, LIVE)), 'original saved local binding')
            z, a = context(t['root_context'])
            same(z[:2], old_z[1:], 'actual history shifts original abs5/10 exactly')
            raw = array(req['raw10'], (10, 2))
            same(raw, expected_raw, 'causal raw10 warmup tail5 plus prefix5')
            same(a, normalize(base, raw), 'once-normalized actual cp5 raw10')
            require(type(ob) is dict and set(ob) == {'boundary', 'png_sha256', 'pixel_sha256', 'encoding_sha256',
                    'observed_index', 'encoded_index', 'observed_latent_sha256'} and
                    type(ob['boundary']) is int and ob['boundary'] == 5 and
                    type(ob['observed_index']) is int and type(ob['encoded_index']) is int and
                    0 <= ob['observed_index'] < ob['encoded_index'], 'actual RGB observation chronology')
            for k in ('png_sha256', 'pixel_sha256', 'encoding_sha256', 'observed_latent_sha256'):
                hash_value(ob[k])
            require(ob['observed_latent_sha256'] == hashlib.sha256(canonical(t['root_context']['latent_history'][-1])+b'\n').hexdigest(),
                    'original LF packet digest of actual cp5 latent')
            payload = {k: t[k] for k in ('root_context', 'goal', 'projection', 'action_mean', 'action_std')}
            if common is None:
                common = deepcopy(payload)
            require(canonical(payload) == canonical(common), 'all four branches share exact cp5 context/goal/normalizer/Q')
            branch_refs.append(dict(episode_id=eid, request_sha256=digest(req), saved_task_binding=deepcopy(t['binding']),
                                    observation=deepcopy(ob)))
        identity = dict(schema=NAME+'.local_identity', source_slot=root, control_boundary=5,
                        authorities=deepcopy(authorities), branch_requests=[b['request_sha256'] for b in branch_refs])
        tasks.append(local_task(base, common['root_context'], identity, NAME))
        provenance.append(dict(source_slot=root, original_task=deepcopy(base), original_task_sha256=digest(base),
            original_observed_absolute_steps=[0, 5, 10], original_control_steps=[-10, -5, 0],
            actual_cp5_absolute_steps=[5, 10, 15], actual_cp5_control_steps=[-5, 0, 5],
            raw10=expected_raw.tolist(), prefix5=prefix.tolist(), branch_witnesses=branch_refs,
            task_export=layer['task_export'], local_identity=identity, local_task_sha256=digest(tasks[-1])))
    return dict(schema=NAME+'.tasks', status='known2_actual_cp5_saved_join_ready_NOT_admitted',
        tasks=tasks, provenance=provenance, authorities=deepcopy(authorities),
        evidence=dict(upstream_fullstrict_rebuilt_here=False, inherited_MAIN_complete_receipt=True,
                      RGB_files_redecoded_here=False, latent_history_reencoded_here=False,
                      original_R_tasks_or_results_modified=False),
        new_calls=dict(encode=0, projector=0, predict=0, AD=0, physics=0), **FLAGS)


def resolve_actual_cp5_tasks(authorities):
    """Return tasks[2]+provenance from explicit path/SHA descriptors; never call a producer."""
    require(type(authorities) is dict and set(authorities) == set(CAPS), 'three explicit authorities')
    a = deepcopy(authorities)
    for desc in a.values():
        require(type(desc) is dict and set(desc) == {'path', 'sha256'} and type(desc['path']) is str,
                'exact path/SHA descriptor')
        desc['path'] = str(Path(desc['path']).absolute())
    p, r, c = [read_pinned(a[k]['path'], a[k]['sha256'], CAPS[k]) for k in CAPS]
    exports, bindings = [], {d['path']: d['sha256'] for d in a.values()}
    require(len(p['layers']) == 2, 'two original layers')
    for layer in p['layers']:
        path = layer['task_export']
        require(path in p['sources'] and p['sources'][path] == layer['task_sha256'], 'prepare-bound original export')
        exports.append(read_pinned(path, p['sources'][path], EXPORT_CAP, compressed=True))
        bindings[path] = p['sources'][path]
    result = join(p, r, c, exports, a)
    result['bindings'] = bindings
    for path in own_files():
        require(path.is_file() and not path.is_symlink(), 'own complete source/tests/doc')
        result['bindings'][str(path)] = sha(path)
    for path, expected in result['bindings'].items():
        require(sha(path) == expected, 'after-join immutable saved inputs and own files')
    return result


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def write_new(path, value):
    raw = canonical(value)+b'\n'
    require(len(raw) <= OUTPUT_CAP, 'own output cap')
    p = Path(path).absolute()
    require(p.parent.is_dir() and not p.is_symlink(), 'existing output parent; no symlink')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete exclusive output')
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(p), sha256=sha(p), bytes=len(raw))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for cmd in ('prepare', 'verify'):
        q = sub.add_parser(cmd)
        for key in CAPS:
            q.add_argument('--'+key, required=True)
            q.add_argument('--'+key+'-sha', required=True)
        q.add_argument('--output', required=True)
        if cmd == 'verify':
            q.add_argument('--output-sha', required=True)
    args = parser.parse_args(argv)
    authorities = {k: dict(path=getattr(args, k), sha256=getattr(args, k+'_sha')) for k in CAPS}
    result = resolve_actual_cp5_tasks(authorities)
    if args.cmd == 'prepare':
        print(json.dumps(write_new(args.output, result), sort_keys=True))
    else:
        actual = read_pinned(args.output, args.output_sha, OUTPUT_CAP)
        require(canonical(actual) == canonical(result), 'entire own task join reconstruction')
        print(json.dumps(dict(verified=True, tasks=2, upstream_fullstrict_rebuilt_here=False)))


if __name__ == '__main__':
    main()
