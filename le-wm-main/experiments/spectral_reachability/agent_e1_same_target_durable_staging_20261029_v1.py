"""Durable four-slot staging bridge. Fixture-only; no CUDA/production CLI."""
from __future__ import annotations
from pathlib import Path
from experiments.spectral_reachability import agent_e1_same_target_four_slot_execution_20261029_v1 as four

old = four.old


def execute(store, tasks, targets, prefixes, backend_for, guard_for, sync, check):
    """Append one hash-chain and persist partial rows on all ordinary failures.

    Caller owns the final report and completion witness; no inference guard or
    authenticated source/task/model authority is provided by this function.
    """
    rows = [dict(root=r, arm=a, status='not_started') for r,a in four.SLOTS]
    errors = []
    result = None
    try:
        def sink_for(i, phase):
            return lambda event: store.event(dict(type='event',slot=i,phase=phase,event=event))
        def barrier(name, value):
            store.event(dict(type='barrier',name=name,value=value))
        def seal(i, phase, value):
            name = f'slot_{i}.{phase}.z'
            store.put(name, value)
            return old.reload_packed(store, name)
        result = four.execute(tasks, targets, prefixes, backend_for, guard_for, sink_for,
                              seal, barrier, sync, check, rows)
        store.put('result.z', result)
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)[:2048]))
        try:
            store.event(dict(type='failure',error=errors[0]))
        except BaseException as secondary:
            errors.append(dict(type=type(secondary).__name__, message='failure journal: '+str(secondary)[:2048]))
    return dict(status='all_four_sealed' if not errors and all(r['status']=='sealed' for r in rows) else 'partial_or_failed',
                rows=rows, errors=errors, result_sealed=result is not None,
                authority_authenticated=False, scientific_admission=False)


def verify_payload(tasks, targets, prefixes, report, payload, events):
    """Independent saved mathematical/IO reconstruction of successful staging."""
    old.require(report['status'] == 'all_four_sealed' and report['errors'] == [] and
                len(report['rows']) == 4 and report['result_sealed'] is True, 'successful four-slot report')
    expected = {'result.z'} | {f'slot_{i}.{p}.z' for i in range(4) for p in ('controller','choice','replay')}
    old.require(set(payload) == expected, 'complete slot payload set')
    result = payload['result.z']
    for i,(root,arm) in enumerate(four.SLOTS):
        row = report['rows'][i]
        old.require((row['root'],row['arm'],row['status']) == (root,arm,'sealed'), 'fixed row identity')
        for phase, count in (('controller',144),('extra',2),('replay',4)):
            old.require(row[phase+'_resources'] == four.metering.native.budget(count)
                        and row[phase+'_samples'] == dict(attempted=count,completed=count), 'module/resource census')
        for kind, field in (('controller','results'),('choice','choices'),('replay','replays')):
            old.require(old.canonical(payload[f'slot_{i}.{kind}.z']) == old.canonical(result[field][i]), 'per-slot payload identity')
    return four.verify_timeline(tasks, targets, prefixes, result, events)


def inspect_partial(report):
    """A failed report is diagnostic only and is never a successful receipt."""
    old.require(report['status'] == 'partial_or_failed' and len(report['rows']) == 4 and report['errors'], 'failed denominator')
    return dict(status='partial_not_consumable', rows=report['rows'], errors=report['errors'], scientific_admission=False)


def finish(store, staged):
    """Durably publish a terminal diagnostic; disk failure can still prevent it."""
    report = dict(staged)
    report['errors'] = list(staged['errors'])
    try:
        store.close()
    except BaseException as exc:
        report['errors'].append(dict(type=type(exc).__name__, message='journal close: '+str(exc)[:2048]))
        report['status'] = 'partial_or_failed'
    report.update(artifacts={p.name:old.sha(p.read_bytes()) for p in store.path.iterdir()
                             if p.is_file() and not p.is_symlink()},
                  journal_events=store.index, journal_sha256=store.previous,
                  array_bytes=store.archive.size, metadata_bytes=store.meta)
    store.write('report.json',old.canonical(report),final=True)
    completion, timing = old.publish_completion(store,old.sha((store.path/'report.json').read_bytes()))
    return dict(report=report, report_sha256=completion['report_sha256'],
                completion_sha256=timing['completion_sha256'],
                timing_sha256=old.sha((store.path/'timing.json').read_bytes()))
