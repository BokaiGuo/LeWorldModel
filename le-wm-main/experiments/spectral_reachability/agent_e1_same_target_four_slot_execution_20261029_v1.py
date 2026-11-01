"""Four-slot staged native Dispatch and saved-IO reconstruction; not a CLI."""
from __future__ import annotations
from copy import deepcopy
from experiments.spectral_reachability import agent_e1_same_target_backend_pair_20261029_v1 as pair

old = pair.old
beam = pair.beam
spec = pair.spec
metering = pair.metering
SLOTS = spec.SLOTS


def execute(tasks, targets, prefixes, backend_for, guard_for, sink_for, seal_reload, barrier, sync, check, rows):
    """Four fixed slots, all controllers -> bank barrier -> choices -> replays.

    Callers supply durable sinks/barriers and retain `rows` even on exceptions.
    A return from this function is not a CUDA or source authorization.
    """
    old.require(len(tasks) == len(targets) == len(prefixes) == 2 and len(rows) == 4, 'fixed roots and slots')
    results, choices, replays = [], [], []
    for i, (root, arm) in enumerate(SLOTS):
        row = rows[i]; old.require(row['root'] == root and row['arm'] == arm and row['status'] == 'not_started', 'slot identity')
        row['status'] = 'controller_attempted'
        sink = sink_for(i, 'controller'); meter = metering.HookMeter('controller', sink, sync, check)
        dispatch = beam.Dispatch(tasks[root], backend_for(i), meter, sink)
        try:
            with guard_for(i, meter):
                value = old.core.optimize(tasks[root], root, pair.ARM, targets[root], dispatch, sink)
            old.require(dispatch.index == 72, 'controller edge count')
            metering.phase_complete(meter)
            saved = seal_reload(i, 'controller', value)
            old.require(old.canonical(saved) == old.canonical(value), 'controller durable identity')
            barrier('slot_payload_sealed', dict(slot=i, phase='controller', sha256=old.sha(old.canonical(saved))))
            results.append(saved); row['status'] = 'controller_sealed'
        finally:
            row['controller_resources'] = meter.counts(); row['controller_samples'] = meter.predictor_samples()
    banks = [metering.compare_banks(results[2*r]['candidates'], results[2*r+1]['candidates']) for r in (0, 1)]
    barrier('all_banks_sealed', banks)
    for i, (root, arm) in enumerate(SLOTS):
        row = rows[i]; row['status'] = 'extra_attempted'
        sink = sink_for(i, 'extra'); meter = metering.HookMeter('extra', sink, sync, check)
        dispatch = beam.Dispatch(tasks[root], backend_for(i), meter, sink)
        raw_extra = spec.extras(prefixes[root])[arm]
        try:
            with guard_for(i, meter):
                context = dispatch(beam.checked_context(tasks[root]['root_context']), raw_extra, 'extra', [0,1])
            metering.phase_complete(meter)
        finally:
            row['extra_resources'] = meter.counts(); row['extra_samples'] = meter.predictor_samples()
        cost = pair.q(tasks[root], targets[root], context)
        candidates = results[i]['candidates']
        winner = min(range(73), key=lambda j: (candidates[j]['cost'] if j < 72 else cost, j))
        raw = spec.raw_actions(candidates[winner]['commands'] if winner < 72 else raw_extra, 10)
        choice = dict(arm=arm, extra=raw_extra, extra_contexts=context, extra_cost=cost, winner_index=winner, selected_raw=raw)
        saved = seal_reload(i, 'choice', choice)
        old.require(old.canonical(saved) == old.canonical(choice), 'choice durable identity')
        barrier('slot_payload_sealed', dict(slot=i, phase='choice', sha256=old.sha(old.canonical(saved))))
        choices.append(saved); row['status'] = 'choice_sealed'
    barrier('all_choices_sealed', [old.sha(old.canonical(x)) for x in choices])
    for i, (root, arm) in enumerate(SLOTS):
        row = rows[i]; row['status'] = 'replay_attempted'
        sink = sink_for(i, 'replay'); meter = metering.HookMeter('replay', sink, sync, check)
        dispatch = beam.Dispatch(tasks[root], backend_for(i), meter, sink)
        raw = choices[i]['selected_raw']; initial = beam.checked_context(tasks[root]['root_context'])
        try:
            with guard_for(i, meter):
                whole = dispatch(deepcopy(initial), deepcopy(raw), 'replay_whole', [0,1])
                first = dispatch(deepcopy(initial), deepcopy(raw[:5]), 'replay_segment', [0])
                second = dispatch(deepcopy(first[0]), deepcopy(raw[5:]), 'replay_segment', [1])
            metering.phase_complete(meter)
        finally:
            row['replay_resources'] = meter.counts(); row['replay_samples'] = meter.predictor_samples()
        replay = dict(raw=raw, whole=whole, segmented=first+second)
        saved = seal_reload(i, 'replay', replay)
        old.require(old.canonical(saved) == old.canonical(replay), 'replay durable identity')
        barrier('slot_payload_sealed', dict(slot=i, phase='replay', sha256=old.sha(old.canonical(saved))))
        replays.append(saved); row['status'] = 'sealed'
    barrier('all_replays_sealed', [old.sha(old.canonical(x)) for x in replays])
    return dict(results=results, choices=choices, replays=replays, banks=banks)


def verify_saved(tasks, targets, prefixes, result, events, barriers):
    old.require(len(events) == 4 and len(result['results']) == len(result['choices']) == len(result['replays']) == 4, 'four slot denominator')
    verified = []
    for root in (0,1):
        s = slice(2*root,2*root+2)
        verified.append(pair.verify_saved(tasks[root],root,targets[root],prefixes[root],
                                         result['results'][s],result['choices'][s],result['replays'][s],events[s]))
    banks = [x['bank'] for x in verified]
    expected = [('all_banks_sealed', banks),
                ('all_choices_sealed', [old.sha(old.canonical(x)) for x in result['choices']]),
                ('all_replays_sealed', [old.sha(old.canonical(x)) for x in result['replays']])]
    old.require(result['banks'] == banks and barriers == expected, 'global barriers exact')
    return dict(verified=True, slots=4, banks=banks, scientific_admission=False)


def verify_timeline(tasks, targets, prefixes, result, records):
    """Split one complete ordered journal; reject late/missing/extra events."""
    old.require(isinstance(records, list), 'timeline list')
    events = [dict(controller=[], extra=[], replay=[]) for _ in SLOTS]
    barriers = []; at = 0
    for phase, marker in (('controller','all_banks_sealed'), ('extra','all_choices_sealed'),
                          ('replay','all_replays_sealed')):
        for i in range(4):
            while at < len(records) and records[at].get('type') == 'event' and records[at].get('slot') == i and records[at].get('phase') == phase:
                r = records[at]
                old.require(set(r) == {'type','slot','phase','event'}, 'event envelope')
                events[i][phase].append(r['event']); at += 1
            old.require(events[i][phase], 'missing slot phase')
            payload_phase = 'choice' if phase == 'extra' else phase
            values = result['results' if phase == 'controller' else 'choices' if phase == 'extra' else 'replays']
            seal = dict(type='barrier', name='slot_payload_sealed', value=dict(
                slot=i, phase=payload_phase, sha256=old.sha(old.canonical(values[i]))))
            old.require(at < len(records) and records[at] == seal, 'slot payload seal chronology')
            at += 1
        old.require(at < len(records) and set(records[at]) == {'type','name','value'}
                    and records[at]['type'] == 'barrier' and records[at]['name'] == marker, 'barrier chronology')
        barriers.append((marker, records[at]['value'])); at += 1
    old.require(at == len(records), 'trailing timeline event')
    return verify_saved(tasks, targets, prefixes, result, events, barriers)
