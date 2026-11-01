"""Offline linker of authenticated saved programs, not a planner or fullstrict."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import numpy as np

NAME = 'agent_p2_saved_program_linker_20261029_v1'
PRODUCER = 'agent_p2_three_program_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
ARMS = ('joint_H3_no_composition', 'three_program_imagined_recompile')
SOURCE_SHA = '3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5'
ACTUAL = dict(
    prepared=dict(path=str(ROOT/'results'/f'{PRODUCER}_prepare_MAIN01.json'),
                  sha256='86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461'),
    report=dict(path=str(ROOT/'results'/f'{PRODUCER}_cuda_MAIN01/report.json'),
                sha256='56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef'),
    receipt=dict(path=str(ROOT/'results'/f'{PRODUCER}_consume_MAIN01.json'),
                 sha256='c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f'))
CAPS = dict(prepared=128*1024**2, report=128*1024**2, receipt=32*1024**2)
ROOT_CAP, OUTPUT_CAP, JOB_CAP = 256*1024**2, 32*1024**2, 1024**2
FLOOR = 200*1024**2
LOGIT_CAP = float(.75*np.sqrt(15))
FLAGS = dict(P1_admission=False, P2_mechanism=False, physical_release=False,
             freshness=False, independent_confirmation=False, equal_FLOPs=False)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    """Producer's exact compact JSON + one LF ABI."""
    return hashlib.sha256(canonical(value)+b'\n').hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def hash_value(h):
    require(type(h) is str and len(h) == 64 and all(c in '0123456789abcdef' for c in h), 'literal SHA256')


def decode(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON field')
            out[k] = v
        return out
    out = json.loads(raw, object_pairs_hook=pairs,
                     parse_constant=lambda x: require(False, 'nonfinite JSON'))
    canonical(out)
    return out


def read_pin(path, h, cap, *, gz=False):
    hash_value(h)
    p = Path(path).absolute()
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap, 'bounded regular saved input')
    require(sha(p) == h, 'saved input SHA')
    if gz:
        with gzip.open(p, 'rb') as f:
            raw = f.read(cap+1)
    else:
        raw = p.read_bytes()
    require(len(raw) <= cap, 'decoded input cap')
    value = decode(raw)
    require(sha(p) == h, 'input changed during read')
    return value


def numeric(value, shape, *, f32=True):
    def valid(v):
        return all(valid(x) for x in v) if type(v) is list else type(v) in (int, float)
    require(type(value) is list and valid(value), 'literal numbers, no bool')
    a = np.asarray(value, np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite exact shape')
    if f32:
        require(np.array_equal(a, a.astype(np.float32).astype(np.float64)), 'literal f32, no rounding repair')
        a = a.astype('<f4')
    return a


def eq(a, b, why):
    require(canonical(a) == canonical(b), why)


def bits_eq(a, b, why):
    require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), why)


def context(c):
    require(type(c) is dict and set(c) == {'layout', 'context_length', 'latent_dim', 'action_dim',
            'latent_history', 'action_history'}, 'complete context ABI')
    require(c['layout'] == 'states_with_between_state_actions_v1' and
            all(type(c[k]) is int and c[k] == v for k, v in
                (('context_length', 3), ('latent_dim', 192), ('action_dim', 10))), 'native context dimensions')
    return numeric(c['latent_history'], (3, 192)), numeric(c['action_history'], (2, 10))


def chart(logits, packets):
    a = numeric(logits, (packets*10,), f32=False)
    return np.tanh(a).astype('<f4').reshape(packets, 5, 2)


def unblob(b):
    require(type(b) is dict and set(b) == {'shape', 'dtype', 'raw_hex', 'sha256'} and
            b['dtype'] == 'float32' and type(b['shape']) is list and 1 <= len(b['shape']) <= 3 and
            all(type(x) is int and 0 < x <= 192 for x in b['shape']), 'native blob ABI')
    raw = bytes.fromhex(b['raw_hex'])
    require(len(raw) <= 192*30*4 and len(raw) == int(np.prod(b['shape']))*4 and
            hashlib.sha256(raw).hexdigest() == b['sha256'], 'native blob bytes/hash')
    a = np.frombuffer(raw, '<f4').reshape(b['shape'])
    require(np.isfinite(a).all(), 'finite native blob')
    return a


class MissingTransition(Exception):
    pass


def saved_path(forwards, start, raw, expected, tag, normalization):
    """Validate only referenced saved IO, never synthesize an absent transition."""
    require(len(expected) == len(raw), 'one complete context per packet')
    current, refs = deepcopy(start), []
    mean = numeric(normalization['mean'], (2,))
    std = numeric(normalization['std'], (2,))
    require((std > 0).all(), 'positive original normalizer')
    for k, actions in enumerate(raw):
        key = tag+'/'+str(k)
        if key not in forwards:
            raise MissingTransition(key)
        f = forwards[key]
        context(current)
        require(f['completed'] is True and type(f['index']) is int and f['index'] >= 0, 'complete saved forward')
        eq(f['input'], current, 'exact forward input full context')
        bits_eq(unblob(f['raw']), actions, 'literal saved actuator bits')
        eq(f['request'], dict(index=f['index'], tag=key, input_context_sha256=digest(current),
            raw_sha256=digest(f['raw']), completed=False), 'exact saved request binding')
        require(f['tag'] == key and (np.abs(actions) <= 1).all(), 'saved tag/raw legality')
        modules = f['modules']
        require(len(modules) == 3 and [e['resource'] for e in modules] ==
                ['action_encoder', 'predictor', 'pred_proj'] and
                all(e['completed'] is True and type(e['index']) is int and e['index'] == 3*f['index']+i
                    for i, e in enumerate(modules)), 'native three-module IO identity')
        ins = [[unblob(b) for b in e['inputs']] for e in modules]
        outs = [unblob(e['output']) for e in modules]
        tokens = np.concatenate((numeric(current['action_history'], (2, 10)), ((actions-mean)/std).reshape(1, 10)))
        require([len(a) for a in ins] == [1, 2, 1], 'native module input arities')
        bits_eq(ins[0][0], tokens[None], 'once-normalized causal native tokens')
        bits_eq(ins[1][0], numeric(current['latent_history'], (3, 192))[None], 'native full latent input')
        require(outs[0].shape == outs[1].shape == (1, 3, 192) and outs[2].shape == (3, 192), 'native output shapes')
        bits_eq(ins[1][1], outs[0], 'native action-encoder connection')
        bits_eq(ins[2][0], outs[1].reshape(3, 192), 'native predictor/projector connection')
        rolled = deepcopy(current)
        rolled.update(latent_history=np.concatenate((numeric(current['latent_history'], (3, 192))[1:], outs[2][-1:])).tolist(),
                      action_history=tokens[1:].tolist())
        eq(f['output'], rolled, 'saved full causal output roll')
        eq(rolled, expected[k], 'candidate context has actual saved transition')
        current = rolled
        refs.append(dict(tag=key, forward_index=f['index'], record_sha256=digest(f),
                         input_context_sha256=digest(f['input']), output_context_sha256=digest(rolled)))
    return refs


def compiler_evidence(row, forwards, tag, start, logits, load_array, normalization):
    matches = [e for e in row['AD'] if e['request']['tag'] == tag]
    if not matches:
        raise MissingTransition(tag)
    require(len(matches) == 1, 'unique compiler anchor')
    entry = matches[0]
    req = entry['request']
    packets = len(logits)//10
    expected_raw = chart(logits, packets)
    require(entry['completed'] is True and type(req['index']) is int and
            req['index'] == row['AD'].index(entry) and req['completed'] is False and
            req['coordinates'] == ('logit_zero' if tag == 'root_J' else 'raw_incumbent') and
            req['shape'] == [192, len(logits)] and type(req['mathematical_VJP']) is int and
            req['mathematical_VJP'] == 192 and req['vectorize'] is True and
            req['input_context_sha256'] == digest(start), 'exact completed compiler input anchor')
    eq(req['at_logits'], logits, 'literal compiler incumbent, not chosen action')
    bits_eq(numeric(req['raw'], (packets, 5, 2)), expected_raw, 'compiler raw/logit chart')
    d = entry['array']
    require(set(d) == {'file', 'sha256', 'shape', 'dtype'} and
            d['file'] == f'root{row["index"]}_J{req["index"]:03}.npy' and
            d['shape'] == [192, len(logits)] and d['dtype'] == 'float32', 'compiler array identity')
    a = load_array(d)
    require(a.dtype == np.float32 and a.shape == (192, len(logits)) and np.isfinite(a).all(), 'saved literal J shape/finite')
    refs = saved_path(forwards, start, expected_raw, entry['contexts'], tag+'/forward', normalization)
    return dict(tag=tag, input_context_sha256=digest(start), request_sha256=digest(req),
                array=deepcopy(d), coordinates=req['coordinates'], planned_steps=packets*5,
                forward_refs=refs, relinearized_here=False)


def program_id(root, arm, candidate, index):
    return f'r{root}/{arm}/c{candidate:02}/p{index}'


def make_program(*, root, arm, candidate, index, start_step, start, path, logits, committed,
                 scope, forwards, tag, compiler, normalization):
    raw = chart(logits, len(logits)//10)[:committed//5]
    p = dict(id=program_id(root, arm, candidate, index), root_slot=root, arm=arm,
             candidate_index=candidate, program_index=index, start_step=start_step,
             end_step=start_step+committed, planned_steps=len(logits)//2, committed_steps=committed,
             scope=deepcopy(scope), input_context=deepcopy(start), contexts=deepcopy(path),
             logits=deepcopy(logits[:committed*2]), planned_logits_sha256=digest(logits),
             commands=raw.reshape(committed, 2).tolist(), raw_float32_bits=raw.tobytes().hex(),
             compiler=deepcopy(compiler), transition_refs=[], status='linked_saved_program', missing=[])
    try:
        p['transition_refs'] = saved_path(forwards, start, raw, path, tag, normalization)
        if compiler is None:
            raise MissingTransition('compiler anchor')
    except MissingTransition as exc:
        p.update(status='blocked_missing_saved_transition', missing=[str(exc)])
        p['raw_float32_bits'] = None  # no executable export without the required proof
    p['program_sha256'] = digest(p)
    return p


def link_programs(catalog, refs, *, expected_steps=15):
    """Connect fixed saved programs only; no callback, nearest-context or label lookup."""
    require(type(refs) is list and 1 <= len(refs) <= 3 and all(type(x) is str for x in refs) and
            type(expected_steps) is int and expected_steps in (5, 10, 15), 'bounded explicit program references')
    by_id = {p['id']: p for p in catalog}
    require(len(by_id) == len(catalog), 'unique program identities')
    out = dict(refs=deepcopy(refs), expected_steps=expected_steps, status='blocked', reason=None,
               program_count=len(refs), packet_count=None, commands=None, raw_float32_bits=None,
               step_provenance=None, terminal_context=None, **FLAGS)
    selected = []
    for ref in refs:
        if ref not in by_id:
            out['reason'] = 'missing_program_reference'
            return out
        p = by_id[ref]
        require(p['program_sha256'] == digest({k: v for k, v in p.items() if k != 'program_sha256'}), 'immutable program capsule')
        if p['status'] != 'linked_saved_program':
            out['reason'] = 'missing_saved_transition_or_compiler'
            return out
        selected.append(p)
    elapsed, first, current, actions, logs, steps = 0, selected[0], None, [], [], []
    for p in selected:
        if canonical(p['scope']) != canonical(first['scope']):
            out['reason'] = 'different_task_goal_model_normalizer_domain_or_root'
            return out
        if p['start_step'] != elapsed:
            out['reason'] = 'overlap_gap_or_wrong_absolute_program_interval'
            return out
        if current is not None and canonical(current) != canonical(p['input_context']):
            out['reason'] = 'disconnected_full_context_not_terminal_only'
            return out
        if elapsed == 0 and digest(p['input_context']) != p['scope']['root_context_sha256']:
            out['reason'] = 'initial_context_not_original_root'
            return out
        require(p['compiler']['input_context_sha256'] == digest(p['input_context']), 'compiler must belong to this program input')
        context(p['input_context'])
        raw = numeric(p['commands'], (p['committed_steps'], 2))
        require(raw.tobytes().hex() == p['raw_float32_bits'] and len(p['transition_refs']) == p['committed_steps']//5,
                'literal program bytes and all packet witnesses')
        bits_eq(chart(p['logits'], p['committed_steps']//5).reshape(p['committed_steps'], 2), raw,
                'committed logit/raw chart exact, no projection')
        current = p['contexts'][-1]
        context(current)
        actions.extend(p['commands'])
        logs.extend(p['logits'])
        for i in range(p['committed_steps']):
            steps.append(dict(step=elapsed+i+1, program_id=p['id'], program_local_step=i+1,
                              transition=p['transition_refs'][i//5]))
        elapsed = p['end_step']
    if elapsed != expected_steps:
        out['reason'] = 'incomplete_requested_horizon'
        return out
    if np.linalg.norm(numeric(logs, (2*elapsed,), f32=False)) > LOGIT_CAP:
        out['reason'] = 'outside_original_joint30_logit_domain'
        return out
    raw = numeric(actions, (elapsed, 2))
    require((np.abs(raw) <= 1).all(), 'literal rawbox')
    out.update(status='linked_saved_actions_NOT_admitted', reason=None, packet_count=elapsed//5,
               commands=actions, raw_float32_bits=raw.tobytes().hex(), step_provenance=steps,
               terminal_context=deepcopy(current), root_slot=first['root_slot'])
    return out


def extract_programs(prepared, report, receipt, rows, authorities, load_array):
    """Pure RAM seam. File-pinned production path is analyze(), never a producer consumer."""
    require(prepared['schema'] == PRODUCER+'.prepare' and report['schema'] == PRODUCER+'.report' and
            receipt['schema'] == PRODUCER+'.consume', 'exact original schema')
    require(receipt['status'] == 'full15_engineering_verified_science_blocked' and receipt['errors'] == [] and
            receipt['planner_report_sha256'] == authorities['report']['sha256'] and
            report['prepared'] == authorities['prepared']['path'] and
            report['prepare_sha256'] == authorities['prepared']['sha256'], 'inherited actual MAIN complete receipt join')
    require(report['status'] == 'all4_full15_plans_sealed_not_admitted' and report['errors'] == [] and
            report['first_error'] is None and report['model_unchanged'] is report['source_verified'] is True and
            report['model_before'] == report['model_after'] == prepared['expected_model_hash'], 'complete saved run authority')
    require(len(prepared['slots']) == len(rows) == len(report['rows']) == 2 and len(receipt['plans']) == 4,
            'full known2/four original winner identities')
    require([(p['root_slot'], p['arm']) for p in receipt['plans']] == [(r, a) for r in (0, 1) for a in ARMS],
            'fixed original four plans, not a candidate subset')
    catalog, candidates = [], []
    for root, (slot, row, rr) in enumerate(zip(prepared['slots'], rows, report['rows'], strict=True)):
        require(type(slot['root_slot']) is int and slot['root_slot'] == root and slot['condition'] == 'off' and
                type(row['index']) is int and row['index'] == root and row['status'] == 'sealed' and
                rr['root_slot'] == root and rr['file'] == f'root{root}.json.gz' and rr['candidate_denominator'] == [42, 42] and
                len(row['forwards']) == 1068 and len(row['AD']) == 85, 'all original candidate/call denominators')
        base = slot['packet']['root_context']
        context(base)
        numeric(slot['packet']['goal'], (192,), f32=False)
        bits_eq(numeric(slot['q'], (192, 192), f32=False), np.eye(192), 'original identity Q')
        plans = receipt['plans'][2*root:2*root+2]
        for index, plan in enumerate(plans):
            eq(plan['root_context'], base, 'receipt/root full context')
            eq(plan['goal'], slot['packet']['goal'], 'receipt/root goal')
            require(plan['task_id'] == slot['task_id'] and plan['task_sha256'] == slot['task_sha256'] and
                    plan['model_hash'] == prepared['expected_model_hash'] and plan['program_count'] == (1 if index == 0 else 3) and
                    plan['planner_report_sha256'] == authorities['report']['sha256'] and
                    plan['prepare_sha256'] == authorities['prepared']['sha256'], 'original task/model/program/authority identity')
        eq(plans[0]['normalization'], plans[1]['normalization'], 'same frozen original normalizer')
        normalization = plans[0]['normalization']
        numeric(normalization['mean'], (2,))
        require((numeric(normalization['std'], (2,)) > 0).all(), 'positive frozen normalizer')
        warm = numeric(slot['warm_raw10'], (10, 2))
        require((np.abs(warm) <= 1).all(), 'original legal warmup raw10')
        for plan in plans:
            bits_eq(numeric(plan['root_warm_raw10'], (10, 2)), warm, 'original causal warmup bits')
        bits_eq(numeric(base['action_history'], (2, 10)),
                ((warm-numeric(normalization['mean'], (2,)))/numeric(normalization['std'], (2,))).reshape(2, 10),
                'original root causal tokens from warm raw10')
        scope = dict(root_slot=root, condition='off', task_id=slot['task_id'], task_sha256=slot['task_sha256'],
            blind_sha256=slot['blind_sha256'], model_hash=prepared['expected_model_hash'],
            goal_sha256=digest(slot['packet']['goal']), Q_sha256=digest(slot['q']),
            normalization=deepcopy(normalization), root_context_sha256=digest(base),
            domain=dict(raw_bounds=[-1, 1], joint_logit_dim=30, logit_cap=LOGIT_CAP,
                        chart='NumPy tanh(float64) -> float32; no repair'), authorities=deepcopy(authorities))
        fw = {f['tag']: f for f in row['forwards']}
        require(len(fw) == len(row['forwards']), 'unique saved forward tags')
        compilers = {}
        def compile_at(tag, start, logits):
            try:
                if tag not in compilers:
                    compilers[tag] = compiler_evidence(row, fw, tag, start, logits, load_array, normalization)
                return compilers[tag]
            except MissingTransition:
                return None
        root_compiler = compile_at('root_J', base, [0.]*30)
        result = row['result']
        require(set(result['candidates']) == set(ARMS) and len(result['bank']['params']) == 42 and
                all(len(result['candidates'][a]) == 42 for a in ARMS), 'all42 each arm, no pruning')
        for i in range(42):
            joint, composed = [result['candidates'][a][i] for a in ARMS]
            for candidate in (joint, composed):
                require(type(candidate['index']) is int and candidate['index'] == i and len(candidate['contexts']) == 3,
                        'literal candidate/time identity')
                require(np.linalg.norm(numeric(candidate['logits'], (30,), f32=False)) <= LOGIT_CAP,
                        'original full30 domain, no projection')
            eq(joint['logits'], result['bank']['params'][i], 'fixed original seed')
            require(composed['root_seed_sha256'] == digest(joint['logits']), 'same original root-bank identity')
            bits_eq(chart(joint['logits'], 3)[0], chart(composed['logits'], 3)[0], 'fixed raw5 prefix')
            eq(joint['contexts'][0], composed['contexts'][0], 'same first context, not just terminal')
            e2, e3 = composed['edge2'], composed['edge3']
            eq(e2['input_context'], composed['contexts'][0], 'cp5 compiler context')
            eq(e3['input_context'], composed['contexts'][1], 'cp10 compiler context')
            eq(e2['fixed_prefix_logits'], joint['logits'][:10], 'fixed prefix compiler identity')
            eq(e2['incumbent_logits'], joint['logits'][10:], 'H2 original incumbent identity')
            eq(e3['fixed_prefix_logits'], composed['logits'][:20], 'H1 full fixed prefix')
            eq(e3['incumbent_logits'], e2['selected_logits'][10:], 'H1 incumbent from H2, not another program')
            eq(composed['logits'], joint['logits'][:10]+e2['selected_logits'][:10]+e3['selected_logits'], 'exact composed logits')
            eq(e2['selected_contexts'][0], composed['contexts'][1], 'H2 committed first packet')
            eq(e3['selected_contexts'], composed['contexts'][2:], 'H1 selected endpoint')
            specs = [(ARMS[0], 0, 0, base, joint['contexts'], joint['logits'], 15, f'joint/{i}', root_compiler),
                (ARMS[1], 0, 0, base, composed['contexts'][:1], joint['logits'], 5,
                 f'compose/{i}/first', root_compiler),
                (ARMS[1], 1, 5, composed['contexts'][0], composed['contexts'][1:2], e2['selected_logits'], 5,
                 f'compose/{i}/edge2/selected_replay', compile_at(f'compose/{i}/edge2/J', e2['input_context'], e2['incumbent_logits'])),
                (ARMS[1], 2, 10, composed['contexts'][1], composed['contexts'][2:], e3['selected_logits'], 5,
                 f'compose/{i}/edge3/selected_replay', compile_at(f'compose/{i}/edge3/J', e3['input_context'], e3['incumbent_logits']))]
            for arm, pi, step, start, path, logits, committed, tag, compiler in specs:
                # H2 selection planned two packets, but only its first packet is committed before recompile.
                catalog.append(make_program(root=root, arm=arm, candidate=i, index=pi, start_step=step,
                    start=start, path=path, logits=logits, committed=committed, scope=scope, forwards=fw,
                    tag=tag, compiler=compiler, normalization=normalization))
            for arm in ARMS:
                refs = [program_id(root, arm, i, k) for k in range(1 if arm == ARMS[0] else 3)]
                candidates.append(dict(root_slot=root, arm=arm, candidate_index=i, refs=refs))
    links = [dict(**c, result=link_programs(catalog, c['refs'])) for c in candidates]
    return dict(schema=NAME+'.catalog', catalog=catalog, candidates=links,
                denominator=dict(known_roots=2, candidates_per_root_per_arm=42, candidate_rows=168,
                                 programs=336, default_link_attempts=168),
                evidence=dict(inherited_complete_MAIN_receipt=True, upstream_fullstrict_rebuilt_here=False,
                    selected_saved_native_IO_checked_here=True, new_model_calls=0, new_AD_calls=0, new_physics_calls=0,
                    cross_candidate_unseen_transition_inferred=False), **FLAGS)


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def analyze(authorities, requests=None):
    require(type(authorities) is dict and set(authorities) == set(CAPS), 'explicit prepare/report/strict authority')
    a = deepcopy(authorities)
    for d in a.values():
        require(type(d) is dict and set(d) == {'path', 'sha256'} and type(d['path']) is str, 'path/SHA descriptor')
        d['path'] = str(Path(d['path']).absolute())
    p, r, c = [read_pin(a[k]['path'], a[k]['sha256'], CAPS[k]) for k in CAPS]
    bindings = {d['path']: d['sha256'] for d in a.values()}
    source = Path(__file__).with_name(PRODUCER+'.py')
    require(sha(source) == SOURCE_SHA, 'frozen producer source ABI')
    bindings[str(source)] = SOURCE_SHA
    output = Path(a['report']['path']).parent
    rows = []
    for root in range(2):
        name = f'root{root}.json.gz'
        h = r['artifact_sha256'][name]
        rows.append(read_pin(output/name, h, ROOT_CAP, gz=True))
        bindings[str(output/name)] = h
    def load_array(d):
        name = d['file']
        require(type(name) is str and Path(name).name == name and name in r['artifact_sha256'] and
                r['artifact_sha256'][name] == d['sha256'], 'report-bound J array')
        path = output/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2 and
                sha(path) == d['sha256'], 'bounded saved J bytes')
        raw = path.read_bytes()
        stream = io.BytesIO(raw)
        value = np.load(stream, allow_pickle=False)
        require(stream.tell() == len(raw), 'J exact NPY EOF')
        bindings[str(path)] = d['sha256']
        return value
    result = extract_programs(p, r, c, rows, a, load_array)
    result['arguments'] = dict(authorities=a, requests=deepcopy(requests))
    result['requested_links'] = []
    if requests is not None:
        require(type(requests) is dict and set(requests) == {'path', 'sha256'}, 'explicit request file authority')
        jobs = read_pin(requests['path'], requests['sha256'], JOB_CAP)
        require(type(jobs) is list and len(jobs) <= 256, 'bounded saved request list')
        for j in jobs:
            require(type(j) is dict and set(j) == {'id', 'refs', 'expected_steps'} and type(j['id']) is str, 'exact link request ABI')
            result['requested_links'].append(dict(id=j['id'], result=link_programs(result['catalog'], j['refs'], expected_steps=j['expected_steps'])))
        require(len({j['id'] for j in jobs}) == len(jobs), 'unique request ids')
        bindings[str(Path(requests['path']).absolute())] = requests['sha256']
    for path in own_files():
        require(path.is_file() and not path.is_symlink(), 'own frozen triple')
        bindings[str(path)] = sha(path)
    for path, h in bindings.items():
        require(sha(path) == h, 'final saved input/source seals')
    result['bindings'] = bindings
    result['summary'] = dict(linked=sum(x['result']['status'] == 'linked_saved_actions_NOT_admitted' for x in result['candidates']),
                             blocked=sum(x['result']['status'] == 'blocked' for x in result['candidates']))
    return result


def write_new(path, value):
    raw = canonical(value)+b'\n'
    require(len(raw) <= OUTPUT_CAP, '32MiB output cap')
    p = Path(path).absolute()
    require(p.parent.is_dir() and not p.is_symlink() and shutil.disk_usage(p.parent).free >= FLOOR+len(raw), 'parent/floor')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete exclusive write')
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
    q = sub.add_parser('analyze')
    for key in CAPS:
        q.add_argument('--'+key, required=True)
        q.add_argument('--'+key+'-sha', required=True)
    q.add_argument('--requests')
    q.add_argument('--requests-sha')
    q.add_argument('--output', required=True)
    q = sub.add_parser('verify')
    q.add_argument('--analysis', required=True)
    q.add_argument('--analysis-sha', required=True)
    args = parser.parse_args(argv)
    if args.cmd == 'analyze':
        require(bool(args.requests) == bool(args.requests_sha), 'request path/SHA pair')
        authorities = {k: dict(path=getattr(args, k), sha256=getattr(args, k+'_sha')) for k in CAPS}
        requests = dict(path=args.requests, sha256=args.requests_sha) if args.requests else None
        out = analyze(authorities, requests)
        print(json.dumps(dict(**write_new(args.output, out), **out['summary'])))
    else:
        prior = read_pin(args.analysis, args.analysis_sha, OUTPUT_CAP)
        eq(prior, analyze(**prior['arguments']), 'entire own saved linker reconstruction')
        print(json.dumps(dict(verified=True, upstream_fullstrict_rebuilt_here=False)))


if __name__ == '__main__':
    main()
