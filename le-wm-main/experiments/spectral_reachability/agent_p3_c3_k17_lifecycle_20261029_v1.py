"""Durable fixed-six lifecycle and saved audit, not a physical provenance receipt."""
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p3_c3_k17_bounded_store_20261029_v1 as bounded
from experiments.spectral_reachability.agent_p3_c3_k17_physical_transport_20261029_v1 import read_journal

ORDER = ((3, 0, 'fixedH1'), (4, 0, 'staticH3_progress'), (5, 0, 'adaptive'),
         (9, 1, 'fixedH1'), (10, 1, 'staticH3_progress'), (11, 1, 'adaptive'))
SCHEMA = 'agent_p3_c3_k17_lifecycle_20261029_v1'


def identities():
    return [dict(index=i, slot=s, root=r, mode=m) for i, (s, r, m) in enumerate(ORDER)]


def _error(stage, exc):
    return {'stage': stage, 'type': type(exc).__name__, 'message': str(exc)}


def run(directory, startup, execute):
    """Persist startup, sequential callbacks and six-slot outcome on real disk.

    Only fixture callbacks are authorized here. Startup must perform all real
    source checks before any future production adapter can call this function.
    Store failure is raised, not claimed to have been durably recorded.
    """
    store = bounded.Store(directory)
    rows = [dict(**identity, status='not_started', result=None, error=None) for identity in identities()]
    store.emit('lifecycle_start', {'schema': SCHEMA, 'rows': rows})
    startup_error = None
    try:
        context = startup()
    except Exception as exc:
        startup_error = _error('startup', exc)
        store.emit('startup_failed', startup_error)
    else:
        store.emit('startup_returned', {})
        for row in rows:
            identity = {k: row[k] for k in ('index', 'slot', 'root', 'mode')}
            store.emit('slot_intent', identity)
            try:
                result = execute(identity.copy(), context)
                if type(result) is not dict or result.get('status') not in ('horizon_completed', 'native_stopped'):
                    raise ValueError('invalid terminal slot result')
                bounded.canonical(result)
            except Exception as exc:
                row.update(status='engineering_failed', error=_error('execute', exc))
                store.emit('slot_failed', {'identity': identity, 'error': row['error']})
                break
            row.update(status=result['status'], result=result)
            store.emit('slot_returned', {'identity': identity, 'result': result})
    report = {'schema': SCHEMA + '.report', 'denominator': 6, 'rows': rows,
              'startup_error': startup_error, 'physical_provenance_verified': False,
              'physical_release': False, 'equal_compute': False}
    store.put('report.json', report)
    report_bytes = (store.path / 'report.json').read_bytes()
    journal_bytes = (store.path / 'journal.jsonl').read_bytes()
    completion = {'schema': SCHEMA + '.completion',
                  'report_sha256': hashlib.sha256(report_bytes).hexdigest(),
                  'journal_sha256': hashlib.sha256(journal_bytes).hexdigest(),
                  'physical_release': False}
    store.put('completion.json', completion)
    return report


def consume(directory, report_sha256, completion_sha256):
    """Independently reconstruct lifecycle from canonical hash-chained events.

    This checks lifecycle only: no source, PNG, native action, or budget audit.
    """
    path = Path(directory)
    def read(name, expected):
        file = path / name
        if not file.is_file() or file.is_symlink():
            raise ValueError('missing or symlink artifact')
        raw = file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('external artifact SHA')
        value = json.loads(raw)
        if raw != bounded.canonical(value) + b'\n':
            raise ValueError('noncanonical artifact')
        return value
    report = read('report.json', report_sha256)
    completion = read('completion.json', completion_sha256)
    if (completion != {'schema': SCHEMA + '.completion', 'report_sha256': report_sha256,
                       'journal_sha256': hashlib.sha256((path / 'journal.jsonl').read_bytes()).hexdigest(),
                       'physical_release': False}):
        raise ValueError('completion binding')
    events = read_journal(path / 'journal.jsonl')
    rows = [dict(**identity, status='not_started', result=None, error=None) for identity in identities()]
    if not events or events[0]['kind'] != 'lifecycle_start' or events[0]['payload'] != {'schema': SCHEMA, 'rows': rows}:
        raise ValueError('initial denominator')
    if len(events) < 2:
        raise ValueError('startup status missing')
    startup_event = events[1]
    startup_error = None
    if startup_event['kind'] == 'startup_failed':
        startup_error = startup_event['payload']
        if startup_error.get('stage') != 'startup' or len(events) != 2:
            raise ValueError('startup failure sequence')
    elif startup_event['kind'] != 'startup_returned' or startup_event['payload'] != {}:
        raise ValueError('startup sequence')
    else:
        remaining = events[2:]
        if len(remaining) % 2 or len(remaining) > 12:
            raise ValueError('slot event count')
        failed = False
        for i in range(len(remaining) // 2):
            intent, terminal = remaining[2*i:2*i+2]
            identity = identities()[i]
            if failed or intent['kind'] != 'slot_intent' or intent['payload'] != identity:
                raise ValueError('slot order or post-failure dispatch')
            p = terminal['payload']
            if p.get('identity') != identity:
                raise ValueError('terminal slot identity')
            if terminal['kind'] == 'slot_failed':
                if set(p) != {'identity', 'error'} or p['error'].get('stage') != 'execute':
                    raise ValueError('slot failure')
                rows[i].update(status='engineering_failed', error=p['error'])
                failed = True
            elif terminal['kind'] == 'slot_returned':
                result = p.get('result')
                if set(p) != {'identity', 'result'} or type(result) is not dict or result.get('status') not in ('horizon_completed', 'native_stopped'):
                    raise ValueError('slot return')
                rows[i].update(status=result['status'], result=result)
            else:
                raise ValueError('terminal event missing')
        if not failed and len(remaining) != 12:
            raise ValueError('missing slots without failure')
    expected = {'schema': SCHEMA + '.report', 'denominator': 6, 'rows': rows,
                'startup_error': startup_error, 'physical_provenance_verified': False,
                'physical_release': False, 'equal_compute': False}
    if report != expected:
        raise ValueError('report disagrees with lifecycle')
    return {'schema': SCHEMA + '.consume', 'denominator': 6, 'rows': rows,
            'lifecycle_verified': True, 'physical_provenance_verified': False,
            'physical_release': False, 'equal_compute': False}
