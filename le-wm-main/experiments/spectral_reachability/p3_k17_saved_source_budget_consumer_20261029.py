"""Independent saved K17 journal replay, anchored by caller-held hashes.

This cannot authenticate a native runtime or establish physical/scientific success.
"""
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p3_c3_k17_candidate_join_20261029_v1 as join
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as meter
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport

FILES = ('p3_k17_live_source_budget_seam_20261029.py',
         'agent_p3_c3_k17_native_outer_20261029_v1.py',
         'agent_p3_c3_k17_native_episode_20261029_v1.py',
         'agent_p3_c3_k17_native_meter_20261029_v1.py',
         'agent_p3_c3_k17_native_store_20261029_v1.py',
         'agent_p3_c3_k17_physical_gate_20261029_v1.py',
         'agent_p3_c3_k17_physical_transport_20261029_v1.py')
OTHER = {'placement_recipe', 'goal_attached', 'invalid_observation', 'fresh_observation', 'own_response'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_bytes(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'missing or symlink file')
    return path.read_bytes()


def read_events(path, expected):
    raw = file_bytes(path)
    need(isinstance(expected, str) and len(expected) == 64 and sha(raw) == expected, 'journal external SHA')
    need(raw.endswith(b'\n'), 'incomplete journal')
    events, previous = [], None
    for line in raw.splitlines():
        event = json.loads(line)
        need(isinstance(event, dict) and set(event) == {'index', 'previous', 'kind', 'payload', 'sha256'}, 'journal schema')
        need(transport.canonical(event) == line and type(event['index']) is int and event['index'] == len(events)
             and event['previous'] == previous, 'journal order or canonical encoding')
        body = {k: event[k] for k in ('index', 'previous', 'kind', 'payload')}
        need(sha(transport.canonical(body)) == event['sha256'], 'journal chain')
        need(isinstance(event['payload'], dict), 'journal payload')
        previous = event['sha256']
        events.append(event)
    return events


def sources():
    directory = Path(__file__).resolve().parent
    return {name: sha(file_bytes(directory/name)) for name in FILES}


def consume(directory, journal_sha256, source_files_sha256, *, output=None):
    """Rebuild saved source material and audit a snapshot, not live provenance.

    Caller must obtain both hashes from an independent trusted channel. Output,
    when requested, is exclusive and fsynced. No producer completion exists yet.
    """
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink(), 'journal directory')
    need(isinstance(source_files_sha256, dict) and set(source_files_sha256) == set(FILES), 'external source roster')
    need(sources() == source_files_sha256, 'external source SHA')
    events = read_events(directory/'journal.jsonl', journal_sha256)
    need(len(events) >= 2 and events[0]['kind'] == 'source_budget_binding' and events[1]['kind'] == 'run_started', 'missing binding/start')
    data, rows = outer.material()
    binding = events[0]['payload']
    want_source = sha(transport.canonical(join.snapshot(data, rows)))
    row_hashes = [sha(transport.canonical(join.normalize(r))) for r in rows]
    need(binding == {'source_sha256': want_source, 'row_sha256': row_hashes,
         'order': [list(x) for x in outer.ORDER], 'resource_caps': meter.CAPS,
         'wall_seconds': meter.WALL, 'source_files_sha256': source_files_sha256,
         'production_authorized': False, 'native_provenance_verified': False}, 'source binding mismatch')
    need(events[1]['payload'] == {'denominator': 6, 'order': [list(x) for x in outer.ORDER]}, 'run start contract')
    actions = []
    for (item, _, _), identity in zip(rows, outer.ORDER, strict=True):
        need((item['slot'], item['root'], item['mode'], item['K']) == (*identity, 17), 'source identity')
        raw = bytes.fromhex(item['first5_raw_f32_hex'])
        need(len(raw) == 40, 'source action bytes')
        actions.append([raw[i:i+8].hex() for i in range(0, 40, 8)])
    counts = {k: {'attempted': 0, 'completed': 0, 'failed': 0} for k in meter.CAPS}
    pending, completed_public, png_names, branches = {}, {}, set(), []
    branch, active, guarded, plan, transition, stopped, failed = -1, False, False, 0, 0, False, False
    warm, captures = 0, 0
    for event in events[2:]:
        kind, p = event['kind'], event['payload']
        if kind == 'operation_dispatch_intent':
            r = p.get('receipt')
            need(isinstance(r, dict) and r.get('status') == 'pending', 'intent receipt')
            resource, idx = r.get('resource'), r.get('index')
            need(resource in counts and type(idx) is int and idx == sum(c['attempted'] for c in counts.values()), 'intent index/resource')
            need(not failed or resource == 'close', 'dispatch after failure')
            if resource == 'constructor':
                need(not active and not pending and branch < 5 and r.get('branch') == branch+1, 'constructor order')
                branch += 1
                active, guarded, plan, transition, stopped, warm, captures = True, False, 0, 0, False, 0, 0
                completed_public = {}
            need(active and (r.get('branch'), r.get('slot'), r.get('root_slot'), r.get('mode')) == (branch, *outer.ORDER[branch]), 'operation identity')
            need(counts[resource]['attempted'] < meter.CAPS[resource], 'resource cap exceeded')
            if resource == 'public_step':
                phase, step = r.get('phase'), r.get('step')
                need(phase in ('warmup', 'plan') and type(step) is int, 'public step phase')
                if phase == 'plan':
                    need(guarded and not stopped and transition == plan and step == plan+1 and step <= 5, 'plan ordering/guard')
                    need(r.get('raw_bits') == actions[branch][step-1], 'paid replay action')
                    plan = step
                else:
                    need(not guarded and step == warm+1 and step <= 10, 'warmup ordering')
                    warm = step
            counts[resource]['attempted'] += 1
            pending[idx] = dict(r)
        elif kind in ('operation_returned', 'operation_failed'):
            r = p.get('receipt')
            need(isinstance(r, dict) and type(r.get('index')) is int, 'return receipt')
            original = pending.get(r['index'])
            need(original is not None, 'return without intent')
            want = dict(original, status='failed' if kind == 'operation_failed' else 'returned')
            if kind == 'operation_failed':
                need(isinstance(r.get('error'), dict), 'missing error')
                want['error'] = r['error']
            need(r == want, 'receipt changed')
            resource = r['resource']
            counts[resource]['failed' if kind == 'operation_failed' else 'completed'] += 1
            del pending[r['index']]
            if kind == 'operation_failed':
                failed = True
            elif resource == 'public_step':
                key = (r['phase'], r['step'])
                need(key not in completed_public, 'duplicate public return')
                completed_public[key] = r
        elif kind == 'k17_guard':
            proof = p.get('proof', {})
            need(active and not failed and not guarded and warm == 10 and captures == 36
                 and not pending and not completed_public and p.get('accepted') is True and isinstance(proof, dict)
                 and proof.get('physical_dispatch_ready') is False and proof.get('history', {}).get('accepted') is True
                 and proof.get('action', {}).get('actual_replay_tokens_exact') is True, 'guard contract')
            guarded = True
        elif kind == 'returned_transition':
            need(active and (p.get('branch'), p.get('slot'), p.get('root_slot'), p.get('mode')) == (branch, *outer.ORDER[branch]), 'transition identity')
            phase, step = p.get('phase'), p.get('step')
            need(phase in ('warmup', 'plan') and type(step) is int, 'transition phase')
            rec = completed_public.pop((phase, step), None)
            need(rec is not None and p.get('requested_bits') == rec.get('raw_bits') and p.get('actual_bits') == rec.get('raw_bits')
                 and p.get('actual_dtype') == '<f4' and type(p.get('terminated')) is bool and type(p.get('truncated')) is bool, 'transition/action mismatch')
            if phase == 'plan':
                need(step == transition+1 and step == plan, 'plan transition order')
                transition = step
            if p['terminated'] or p['truncated']:
                stopped = True
        elif kind == 'branch_closed':
            row = p.get('row')
            need(active and isinstance(row, dict) and (row.get('index'), row.get('slot'), row.get('root_slot'), row.get('mode')) == (branch, *outer.ORDER[branch]), 'branch close identity')
            status = row.get('status')
            need(status in ('horizon_completed', 'native_stopped', 'engineering_failed'), 'branch status')
            if status == 'horizon_completed':
                need(not failed and not stopped and not pending and not completed_public and guarded and
                     warm == 10 and captures == 51 and plan == transition == 5, 'false completed branch')
            if failed:
                need(status == 'engineering_failed', 'failure concealed')
            branches.append({'index': branch, 'status': status, 'warmup_intents': warm,
                             'plan_intents': plan, 'plan_transitions': transition, 'observations': captures})
            active = False
        elif kind in OTHER:
            need(active, 'event outside branch')
            if kind == 'fresh_observation':
                receipt = p.get('receipt')
                need(isinstance(receipt, dict), 'observation receipt')
                name = receipt.get('png')
                need(isinstance(name, str) and Path(name).name == name and name.endswith('.png') and name not in png_names, 'PNG name')
                need(sha(file_bytes(directory/name)) == receipt.get('file_sha256'), 'PNG digest')
                png_names.add(name)
                captures += 1
            elif kind == 'invalid_observation':
                failed = True
        else:
            raise ValueError('unknown or repeated event: '+str(kind))
    coverage = (len(branches) == 6 and not active and not pending and not failed and
                all(b['status'] == 'horizon_completed' and b['warmup_intents'] == 10 and
                    b['plan_transitions'] == 5 and b['observations'] == 51 for b in branches))
    report = {'schema': 'p3_k17_saved_source_budget_consumer_20261029.v1',
              'journal_sha256': journal_sha256, 'source_sha256': want_source,
              'source_files_sha256': source_files_sha256, 'events': len(events),
              'counts': counts, 'pending': list(pending.values()), 'branches': branches,
              'active_branch': branch if active else None, 'fault_recorded': failed,
              'six_slot_event_coverage': coverage, 'complete_receipt_verified': False,
              'native_provenance_verified': False, 'production_authorized': False, 'scientific_pass': False,
              'limitations': 'No producer completion/manifest, trusted native provenance, full capture/guard, wall or compute certification.'}
    if output is not None:
        path = Path(output)
        raw = transport.canonical(report) + b'\n'
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return report
