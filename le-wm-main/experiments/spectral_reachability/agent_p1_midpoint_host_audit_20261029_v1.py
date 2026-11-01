"""Saved-row audit for midpoint compiler; not a production runner/receipt."""
from copy import deepcopy
from experiments.spectral_reachability import agent_p1_midpoint_monotone_compiler_20261029_v1 as method

original = method.original
F = method.frozen
require, raw = original.require, original.raw


def audit(spec, result, receipt, events):
    """Validate the complete 26-forward native transcript and selection.

    Caller must independently authenticate the prepare, result, journal, model
    and sources. This function does not create or authenticate a receipt.
    """
    q = spec['request']
    require(set(result) == {'schema','original','midpoint_rows','selected_iteration','selected_commands',
                            'status','physical_release','P1_admission','production_ready'}, 'exact result fields')
    require(result['schema'] == method.NAME+'.result' and all(result[k] is False for k in
            ('physical_release','P1_admission','production_ready')), 'science/release gates')
    base = result['original']; original.strict_numeric(q, base['numeric'], base['numeric_events'])
    its = base['numeric']['iterates']; rows = result['midpoint_rows']
    require(len(its) == 9 and len(rows) == 8, 'fixed candidate and midpoint census')
    schedule = original.schedule(q, base['numeric'])
    control = {'compile_intent','numeric_event','winner_replay_intent','compile_return',
               'midpoint_intent','midpoint_return','selection'}
    mids = []
    for i, row in enumerate(rows, 1):
        require(set(row) == {'iteration','midpoint_logits','midpoint_commands','midpoint_raw_bits',
                             'midpoint_context','midpoint_Q','candidate_Q'} and type(row['iteration']) is int
                and row['iteration'] == i, 'exact ordered midpoint rows')
        x = method.midpoint_logits(its[0]['logits'], its[i]['logits'], F.CAP)
        a = F.chart(x)
        require(raw(row['midpoint_logits']) == raw(x.tolist()) and raw(row['midpoint_commands']) == raw(a.tolist())
                and row['midpoint_raw_bits'] == original.F.bits(a) and row['candidate_Q'] == its[i]['Q'],
                'original chart midpoint and candidate binding')
        original.core.checked_path(q['task'], q['observed_context'], a.tolist(), [row['midpoint_context']])
        require(row['midpoint_Q'] == F.cost(q['task']['goal'], row['midpoint_context']), 'independent original Q')
        schedule.append((f'midpoint_{i}',q['observed_context'],a.tolist()))
        mids.append((x,a,row))
    native = [e for e in events if e['kind'] not in control]
    starts, ends, _ = F.audit_native(q['task'], receipt, native, schedule)
    require(len(starts) == len(ends) == 26, 'all26 native calls')
    for i, (_, _, row) in enumerate(mids):
        require(ends[18+i]['context'] == row['midpoint_context'], 'saved native returned midpoint context')
    chosen = method.select(its[0]['Q'], rows)
    commands = None if chosen is None else q['incumbent']['prefix'] + its[chosen]['commands']
    require(result['selected_iteration'] == chosen and raw(result['selected_commands']) == raw(commands)
            and result['status'] == ('abstain' if chosen is None else 'model_candidate'), 'exact selection/abstention')
    wanted = [dict(kind='compile_intent',request_sha256=original.digest(q),
                   midpoint_row_binding_sha256=spec['midpoint_row_binding_sha256'])]
    wanted += [dict(kind='numeric_event',event=e) for e in base['numeric_events']]
    wanted += [dict(kind='winner_replay_intent',iteration=base['numeric']['winner_iteration'],
                    commands=base['numeric']['commands'][5:]),
               dict(kind='compile_return',numeric_sha256=original.digest(base['numeric']),
                    plan_sha256=original.digest(base['plan']))]
    for i, (x,a,row) in enumerate(mids,1):
        wanted += [dict(kind='midpoint_intent',iteration=i,context_sha256=original.digest(q['observed_context']),
                        logits=x.tolist(),commands=a.tolist(),raw_bits=original.F.bits(a)),
                   dict(kind='midpoint_return',row=deepcopy(row))]
    wanted += [dict(kind='selection',selected_iteration=chosen,status=result['status'])]
    require(raw([e for e in events if e['kind'] in control]) == raw(wanted), 'exact control transcript')
    positions = {id(e):i for i,e in enumerate(events)}
    controls = [e for e in events if e['kind'] in control]
    require(events[0] == wanted[0] and events[-1] == wanted[-1], 'control bookends')
    offset = len(wanted)-17
    for i in range(8):
        require(positions[id(controls[offset+2*i])] < positions[id(starts[18+i])]
                < positions[id(ends[18+i])] < positions[id(controls[offset+2*i+1])], 'midpoint forward bracket')
    return dict(selected_iteration=chosen, selected_commands=deepcopy(commands), midpoint_count=8)
