"""Pure saved pre/post physical-probe contract. No simulator/model executor.

An executor must implement this NEW evidence ABI; the old post-only probe
report is not renamed or accepted as a before/after experiment.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

NAME = 'agent_mixedmode_capability_probe_20261028_v2'
ROOT = Path(__file__).resolve().parents[2]
PREPARE_SHA = '8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b'
PRODUCER_SHA = 'ec912669c678960aac0d004a2e764b4bf036194673da5d1c63d0b535bc199396'
ORDER = [(s, c) for s in (0, 1) for c in ('off', 'on')]
PROBES = [('zero', (0., 0.)), ('+x', (.1, 0.)), ('-x', (-.1, 0.)),
          ('+y', (0., .1)), ('-y', (0., -.1))]
CAP, FLOOR = 64 * 1024**2, 200 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_value(value):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'literal SHA256')
    return value


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= CAP,
            'bounded regular metadata file')
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    obj = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    canonical(obj)
    return obj


def raw(value, n):
    original = np.asarray(value, dtype=np.float64)
    if n == 0 and original.size == 0:
        original = original.reshape(0, 2)
    require(original.shape == (n, 2) and np.isfinite(original).all(), 'raw shape/finite')
    a = original.astype('<f4')
    require(np.array_equal(a.astype(np.float64), original), 'literal float32 raw, no recast rounding')
    return a


def vector(value):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == (7,) and np.isfinite(a).all(), 'native7 state')
    return a


def contract():
    return dict(schema=NAME+'.contract', fixed_tasks=4, episodes=40,
        roles=['before', 'after'], probes=[dict(probe_id=k, commands=np.tile(
            np.asarray(d, dtype='<f4'), (5, 1)).tolist()) for k, d in PROBES],
        before='fresh original reset recipe + original raw warmup10',
        after='fresh original reset recipe + original raw warmup10 + selected literal prefix5',
        units='original native raw float32; no /100, no tanh or clipping',
        response='native state7 endpoint minus same-role zero endpoint; wrapped angle separately',
        comparison='same fixed probes at different causal states, descriptive finite response, not derivative proof',
        coverage='constant XY five-step controls only, not all ten packet coordinates',
        confounding='time/velocity/contact/cache differ after prefix; no matched-time causal isolation',
        public_steps_cap=700, reset_cap=80, probe_steps_cap=200, prefix_steps_cap=100,
        warmup_steps_cap=400, internal_reset_physics='must meter attempts/returns; not hidden in public700',
        renderer='existing off/on/off invariant triplet and fresh RGB guard required at every observed state',
        physical_executor='pending new adapter; old post-only runner is not this ABI',
        physics_render_cap=None, physics_internal_cap=None,
        execution_ready=False, retries=0, new_neural_calls=0, P1_admission=False,
        independent_confirmation=False, equal_total_compute=False)


def assemble(compiled, consumed, metadata):
    """Pure seam; production build first invokes frozen producer full consume."""
    require(compiled['schema'] == 'agent_mixedmode_prefix_20261028_v1.prepare' and
            consumed['status'] == 'all4_mixedmode_commands_verified_not_admitted' and
            consumed['errors'] == [] and len(compiled['slots']) == len(consumed['plans']) == 4,
            'complete four mixedmode tasks only')
    require(len(metadata) == 2, 'both source roots retained')
    tasks, jobs = [], []
    for i, (slot, plan) in enumerate(zip(compiled['slots'], consumed['plans'], strict=True)):
        key = ORDER[i]
        require((slot['root_slot'], slot['condition']) == (plan['root_slot'], plan['condition']) == key,
                'fixed four identity/order')
        require(all(slot[k] == plan[k] for k in ('task_id', 'task_sha256', 'blind_sha256')),
                'full encoded task and blind identity')
        body = {k: v for k, v in plan.items() if k not in ('selection_digest', 'whole_run_report_sha256')}
        # Producer canonicalization is independently checked by its consume.
        require(isinstance(plan['selection_digest'], str) and body['horizon'] == 2,
                'original selected plan authority')
        commands = raw(plan['commands'], 10)
        require(commands.tobytes() == raw(plan['prefix_commands']+plan['suffix_commands'], 10).tobytes(),
                'literal prefix/suffix split')
        require(0 <= plan['selected_index'] < 222, 'full candidate denominator')
        task = dict(index=i, slot=deepcopy(slot), plan=deepcopy(plan),
                    native=deepcopy(metadata[key[0]]))
        task['context_digest'] = digest(slot)  # Includes complete packet/history/Q, not just last latent.
        tasks.append(task)
        for role in ('before', 'after'):
            for probe in contract()['probes']:
                job = dict(index=len(jobs), task=i, role=role, probe_id=probe['probe_id'],
                    context_digest=task['context_digest'], selection_digest=plan['selection_digest'],
                    prefix_commands=[] if role == 'before' else commands[:5].tolist(),
                    probe_commands=probe['commands'])
                job['job_digest'] = digest(job)
                jobs.append(job)
    return dict(schema=NAME+'.prepare', contract=contract(), tasks=tasks, jobs=jobs,
                bindings=deepcopy(compiled['bindings']), status='prepared_contract_executor_pending',
                P1_admission=False)


def build(output, report_sha, prepared, prepare_sha, source_sha):
    """Read saved data only. Missing actual mixedmode terminal report blocks here."""
    from experiments.spectral_reachability import agent_mixedmode_prefix_20261028_v1 as producer
    from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as adapter
    require(hash_value(source_sha) == PRODUCER_SHA == sha(producer.__file__), 'frozen producer source')
    require(hash_value(prepare_sha) == PREPARE_SHA, 'MAIN actual fixed mixedmode prepare')
    compiled = producer.verify(prepared, prepare_sha)
    consumed = producer.consume(output, hash_value(report_sha))
    r = producer.overlay.read(Path(output)/'report.json', producer.CAP)
    require(Path(r['prepared']).resolve() == Path(prepared).resolve() and
            r['prepare_sha256'] == prepare_sha, 'report/prepare execution binding')
    data = adapter.source_material()
    p = assemble(compiled, consumed, adapter.root_metadata(data))
    p['source_slots'] = [deepcopy(r['slot']) for r in data['roots']]
    p['warm_traces'] = [deepcopy(r['warm_trace']) for r in data['roots']]
    for f, h in data['bindings'].items():
        require(f not in p['bindings'] or p['bindings'][f] == h, 'transitive closure conflict')
        p['bindings'][f] = h
    for f in [Path(producer.__file__), Path(adapter.__file__), Path(prepared),
              Path(output)/'report.json', Path(__file__),
              ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]:
        p['bindings'][str(f.resolve())] = sha(f)
    # Bind ALL actual producer payloads, without copying its large arrays.
    for name, h in r['artifact_sha256'].items():
        p['bindings'][str((Path(output)/name).resolve())] = h
    p['arguments'] = dict(output=str(Path(output).resolve()), report_sha=report_sha,
        prepared=str(Path(prepared).resolve()), prepare_sha=prepare_sha, source_sha=source_sha)
    verify_bindings(p)
    require(len(canonical(p)) <= CAP, 'complete prepare64MiB')
    return p


def verify_bindings(p):
    for f, h in p['bindings'].items():
        require(Path(f).is_absolute() and not Path(f).is_symlink() and sha(f) == hash_value(h),
                'complete sealed input closure')


def verify(path, expected):
    require(sha(path) == hash_value(expected), 'external prepare SHA')
    p = read(path)
    require(p == build(**p['arguments']), 'full prepare regeneration')
    return p


def reuse_check(new, old):
    """Eligibility ONLY, never manufacture inherited evidence or fresh status.

    old is a caller-normalized record from a complete old own-consumer, not
    arbitrary old42 candidate index. Both original records must be retained.
    """
    fields = ('root_slot', 'condition', 'task_id', 'task_sha256', 'blind_sha256',
              'source_report_sha256', 'model_sha256', 'context_digest',
              'goal_digest', 'renderer_digest', 'reset_recipe_digest', 'warmup_digest')
    reasons = [k for k in fields if k not in new or k not in old or new[k] != old[k]]
    for k in ('old_consume_report_sha256', 'old_physical_report_sha256'):
        if not isinstance(old.get(k), str) or len(old[k]) != 64:
            reasons.append(k)
    if old.get('consumer_verified') is not True:
        reasons.append('old_complete_consumer_required')
    if old.get('role') != new.get('role'):
        reasons.append('causal_role')
    for k in ('causal_raw_commands', 'probe_commands'):
        try:
            a, b = raw(new[k], len(new[k])), raw(old[k], len(old[k]))
            if a.shape != b.shape or a.tobytes() != b.tobytes():
                reasons.append(k)
        except (KeyError, ValueError, TypeError):
            reasons.append(k)
    return dict(eligible=not reasons, reasons=reasons, evidence=None,
                status='eligible_requires_old_complete_evidence' if not reasons else 'unknown_no_reuse',
                fresh=False, P1_admission=False)


def summarize(p, rows):
    """Validated endpoints only. No metric, rank or success-based selection."""
    require(len(rows) == len(p['jobs']) == 40, 'all40 including failures')
    endpoints = {}
    for job, row in zip(p['jobs'], rows, strict=True):
        require(row['job_digest'] == job['job_digest'], 'fixed job order/identity')
        endpoints[(job['task'], job['role'], job['probe_id'])] = (
            vector(row['end_state']) if row['status'] == 'completed' else None)
    contrasts = []
    for t in range(4):
        for probe_id, _ in PROBES:
            responses = {}
            for role in ('before', 'after'):
                a = endpoints[(t, role, probe_id)]
                z = endpoints[(t, role, 'zero')]
                if a is None or z is None:
                    responses[role] = None
                else:
                    delta = a-z
                    delta[4] = np.arctan2(np.sin(delta[4]), np.cos(delta[4]))
                    responses[role] = delta.tolist()
            before, after = responses.values()
            diff = None if before is None or after is None else (np.asarray(after)-before)
            if diff is not None:
                diff[4] = np.arctan2(np.sin(diff[4]), np.cos(diff[4]))
            contrasts.append(dict(task=t, probe_id=probe_id, **responses,
                after_minus_before=None if diff is None else diff.tolist()))
    return dict(schema=NAME+'.consume', status='saved_probe_responses_not_admitted',
        fixed_denominator=40, rows=deepcopy(rows), contrasts=contrasts,
        capability_causal_change='unknown; finite response contrast confounded by time/state',
        endpoint_improvement_is_capability=False, P1_admission=False,
        independent_confirmation=False, new_neural_calls=0, new_physics_calls=0)


def consume(prepared, prepare_sha, evidence, evidence_sha, executor_source, executor_sha):
    """Future executor ABI. Strict chain + dispatch/return + full state/frame files.

    A source hash authenticates implementation identity, NOT its scientific
    correctness. MAIN must review that executor before execution.
    """
    p = verify(prepared, prepare_sha)
    require(sha(executor_source) == hash_value(executor_sha), 'external actual executor source')
    require(sha(evidence) == hash_value(evidence_sha), 'external actual evidence SHA')
    r = read(evidence)
    require(r['schema'] == NAME+'.evidence' and r['prepare_sha256'] == prepare_sha and
            r['executor_source_sha256'] == executor_sha and r['status'] in ('complete', 'partial') and
            len(r['rows']) == 40, 'NEW evidence ABI and full denominator')
    require(r['new_neural_calls'] == 0, 'no neural evidence here')
    parent = Path(evidence).resolve().parent
    def artifact(receipt):
        name = receipt['file']
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat evidence artifact')
        path = parent/name
        require(not path.is_symlink() and sha(path) == hash_value(receipt['sha256']), 'actual artifact SHA')
        return path
    jp = artifact(r['journal'])
    require(jp.stat().st_size <= CAP, 'metadata journal cap')
    lines = jp.read_bytes()
    require(lines.endswith(b'\n'), 'complete journal line')
    events = [json.loads(v) for v in lines.splitlines()]
    tail = None
    for i, e in enumerate(events):
        require(e['index'] == i and e['previous_sha256'] == tail and
                e['event_sha256'] == digest({k:v for k,v in e.items() if k != 'event_sha256'}),
                'hashchain')
        tail = e['event_sha256']
    require(tail == r['journal_tail'] and events[-1]['kind'] == 'run_finished', 'terminal chain')
    require(all(e.get('job_digest') in {j['job_digest'] for j in p['jobs']}
                for e in events if 'job_digest' in e), 'no extra job/replacement')
    # Generic meter covers reset/internal hooks/render/close as well as public
    # steps. It is distinct from probe_attempted/probe_returned annotations.
    counts = {}
    outstanding = {}
    for e in events:
        if e['kind'] not in ('operation_attempted', 'operation_returned'):
            continue
        key = (e['job_digest'], e['operation_index'])
        resource = e['resource']
        require(isinstance(resource, str), 'meter resource')
        c = counts.setdefault(resource, dict(attempted=0, completed=0))
        if e['kind'] == 'operation_attempted':
            require(key not in outstanding, 'operation index no retry/replacement')
            outstanding[key] = dict(resource=resource, returned=False)
            c['attempted'] += 1
        else:
            require(key in outstanding and not outstanding[key]['returned'] and
                    outstanding[key]['resource'] == resource, 'operation return after unique attempt')
            outstanding[key]['returned'] = True
            c['completed'] += 1
    require(r['counts'] == counts, 'all resources actual attempted/returned census')
    require(set(('constructor', 'close', 'reset', 'public_step', 'internal_physics_step',
                 'explicit_render')).issubset(counts), 'all physical resource axes including reset hooks')
    if r['status'] == 'complete':
        for resource, n in [('constructor', 40), ('close', 40), ('reset', 80), ('public_step', 700)]:
            require(counts[resource] == dict(attempted=n, completed=n), 'complete full public/reset denominator')
        require(counts['internal_physics_step']['completed'] >= 700 and
                counts['explicit_render']['completed'] >= 720, 'internal work/render not zeroed')
    for job, row in zip(p['jobs'], r['rows'], strict=True):
        require(row['job_digest'] == job['job_digest'] and row['status'] in
                ('completed', 'failed', 'native_stopped', 'unstarted'), 'fixed ordered complete denominator')
        scoped = [e for e in events if e.get('job_digest') == job['job_digest']]
        returns = [e for e in scoped if e['kind'] == 'probe_returned']
        attempts = [e for e in scoped if e['kind'] == 'probe_attempted']
        require([e['step'] for e in attempts] == list(range(1, len(attempts)+1)) and
                0 <= len(returns) <= len(attempts) <= 5, 'no retry; attempt lower bounds')
        for i, e in enumerate(attempts):
            require(e['raw_bits'] == raw(job['probe_commands'], 5)[i].tobytes().hex(),
                    'failed attempted action also bound')
        for i, e in enumerate(returns):
            require(e['step'] == i+1 and e['raw_bits'] == attempts[i]['raw_bits'] ==
                    raw(job['probe_commands'], 5)[i].tobytes().hex(), 'literal dispatch/return')
            vector(e['state'])
        require(row['attempted'] == len(attempts) and row['completed'] == len(returns), 'actual attempt/return ledger')
        if row['status'] == 'completed':
            require(len(returns) == len(attempts) == 5 and row['error'] is None and
                    row['start_verified'] is True and row['context_digest'] == job['context_digest'] and
                    row['causal_prefix_commands'] == job['prefix_commands'] and
                    row['end_state'] == returns[-1]['state'], 'complete actual response and original context')
            native = p['tasks'][job['task']]['native']
            require(all(row[k] == native[k] for k in ('root_state', 'goal', 'root_sha256', 'goal_pose')) and
                    row['goal_pixel_sha256'] == native['goal_pixels'][p['tasks'][job['task']]['plan']['condition']],
                    'original full native root/goal and condition image')
            replay = [e for e in scoped if e['kind'] == 'causal_replay_returned']
            warm = np.repeat(np.asarray([[.1,0], [0,.1]], dtype='<f4'), 5, axis=0)
            expected_replay = np.concatenate([warm, raw(job['prefix_commands'], len(job['prefix_commands']))])
            require(len(replay) == len(expected_replay), 'complete original warmup and selected prefix')
            for i, e in enumerate(replay):
                require(e['step'] == i+1 and e['raw_bits'] == expected_replay[i].tobytes().hex() and
                        e['terminated'] is e['truncated'] is False, 'causal raw replay, no post-terminal dispatch')
                vector(e['state'])
            require(replay[9]['state'] == native['root_state'] and
                    row['start_state'] == replay[-1]['state'], 'root exact then actual probe-start state')
            require(row['causal_replay_verified'] is True and row['rgb_guard_accepted'] is True and
                    len(row['frames']) == 6, 'fresh initial+5 frames and full causal reconstruction')
            for step, frame in enumerate(row['frames']):
                require(frame['step'] == step and frame['pre'] == frame['post'] and
                        frame['state'] == frame['post']['snapshot']['state'] and
                        frame['state'] == (row['start_state'] if step == 0 else returns[step-1]['state']),
                        'actual render snapshot invariant and response alignment')
                require(frame['pre']['goal_pose'] == native['goal_pose'] and
                        frame['pre']['renderer'] == native['renderer'], 'same goal pose/renderer throughout')
                # Full image file hash; pixel hash additionally verified by PNG decoder.
                from PIL import Image
                path = artifact(frame)
                with Image.open(path) as im:
                    pixels = np.asarray(im)
                require(pixels.shape == (224,224,3) and pixels.dtype == np.uint8 and
                        hashlib.sha256(pixels.tobytes()).hexdigest() == frame['pixel_sha256'], 'official RGB pixels')
        else:
            require(row['end_state'] is None, 'failed/partial endpoint remains unknown')
    require(r['status'] != 'complete' or all(x['status'] == 'completed' for x in r['rows']), 'no false complete')
    require(r['status'] != 'complete' or r['errors'] == [], 'complete cannot hide execution errors')
    verify_bindings(p)
    require(sha(evidence) == evidence_sha, 'evidence unchanged during read')
    result = summarize(p, r['rows'])
    result['evidence_sha256'] = evidence_sha
    result['counts'] = deepcopy(counts)
    result['errors'] = deepcopy(r['errors'])
    result['journal_sha256'] = r['journal']['sha256']
    result['executor_source_sha256'] = executor_sha
    result['prepare_sha256'] = prepare_sha
    return result


def write_new(path, obj):
    path = Path(path).absolute()
    data = canonical(obj)
    require(path.name.startswith(NAME+'_') and path.suffix == '.json' and len(data) <= CAP,
            'own new bounded metadata output')
    require(shutil.disk_usage(path.parent).free >= FLOOR+len(data), 'disk floor200MiB')
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return sha(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('prepare')
    for name in ('output', 'report-sha', 'prepared', 'prepare-sha', 'source-sha', 'destination'):
        b.add_argument('--'+name, required=True)
    v = sub.add_parser('verify')
    v.add_argument('--prepared', required=True); v.add_argument('--prepare-sha', required=True)
    c = sub.add_parser('consume')
    for name in ('prepared', 'prepare-sha', 'evidence', 'evidence-sha', 'executor-source', 'executor-sha', 'destination'):
        c.add_argument('--'+name, required=True)
    a = vars(parser.parse_args(argv)); command = a.pop('command')
    destination = a.pop('destination', None)
    result = verify(a['prepared'], a['prepare_sha']) if command == 'verify' else build(**a) if command == 'prepare' else consume(**a)
    print(write_new(destination, result) if destination else digest(result))


if __name__ == '__main__':
    main()
