"""Two-arm native Dispatch integration and saved reconstruction, no production CLI."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_e1_six_slot_common_cuda_20261029_v2 as old
from experiments.spectral_reachability import agent_e1_same_target_source_meter_20261029_v1 as metering
from experiments.spectral_reachability import agent_e1_same_target_candidate_intervention_20261029_v1 as spec

ARM = 'dynamics_compiled_witness'
beam = old.phase.beam


def q(task, target, contexts):
    old.require(len(contexts) == 2, 'two H2 contexts')
    target = old.core.old.vec(target)
    goal = old.core.old.vec(task['goal'])
    prefix = float(np.mean((old.core.old.vec(contexts[0]['latent_history'][-1]) - target)**2))
    terminal = float(np.mean((old.core.old.vec(contexts[1]['latent_history'][-1]) - goal)**2))
    return .5 * (prefix + terminal)


def run_pair(task, root, target, witness_raw5, backend_for, guard_for, sink_for, seal_reload, sync, check, rows):
    """Real native Dispatch and backend guard; sink/seal are caller-owned durable IO.

    Both original controllers are independently evaluated and sealed before
    comparing complete bank bytes. No model loading or CUDA authority here.
    The caller MUST persist `rows` in finally, including on exceptions.
    """
    old.require(len(rows) == 2 and all(r['status'] == 'not_started' for r in rows), 'two fresh rows')
    results = []
    for i in range(2):
        row = rows[i]; row['status'] = 'controller_attempted'
        sink = sink_for(i, 'controller')
        meter = metering.HookMeter('controller', sink, sync, check)
        dispatch = beam.Dispatch(task, backend_for(i), meter, sink)
        try:
            with guard_for(i, meter):
                result = old.core.optimize(task, root, ARM, target, dispatch, sink)
            old.require(dispatch.index == 72, '72 independent bank dispatches')
            metering.phase_complete(meter)
            results.append(seal_reload(i, 'controller', result))
            old.require(results[-1] == result, 'controller sealed/reloaded')
            row['status'] = 'controller_sealed'
        finally:
            row['controller_resources'] = meter.counts(); row['controller_samples'] = meter.predictor_samples()
    bank = metering.compare_banks(results[0]['candidates'], results[1]['candidates'])
    extras = spec.extras(witness_raw5)
    for i, arm in enumerate(spec.ARMS):
        row = rows[i]; row['bank'] = bank; row['status'] = 'extra_attempted'
        sink = sink_for(i, 'extra')
        meter = metering.HookMeter('extra', sink, sync, check)
        dispatch = beam.Dispatch(task, backend_for(i), meter, sink)
        try:
            with guard_for(i, meter):
                extra = dispatch(beam.checked_context(task['root_context']), extras[arm], 'extra', [0, 1])
            metering.phase_complete(meter)
        finally:
            row['extra_resources'] = meter.counts(); row['extra_samples'] = meter.predictor_samples()
        extra_cost = q(task, target, extra)
        candidates = results[i]['candidates']
        winner = min(range(73), key=lambda j: (candidates[j]['cost'] if j < 72 else extra_cost, j))
        raw = spec.raw_actions(candidates[winner]['commands'] if winner < 72 else extras[arm], 10)
        choice = dict(arm=arm, extra=extras[arm], extra_contexts=extra, extra_cost=extra_cost,
                      winner_index=winner, selected_raw=raw)
        old.require(seal_reload(i, 'choice', choice) == choice, 'choice sealed/reloaded')
        row['status'] = 'replay_attempted'
        sink = sink_for(i, 'replay')
        meter = metering.HookMeter('replay', sink, sync, check)
        # Existing replay checks the phase name, so dispatch its three edges here.
        dispatch = beam.Dispatch(task, backend_for(i), meter, sink)
        try:
            with guard_for(i, meter):
                initial = beam.checked_context(task['root_context'])
                whole = dispatch(deepcopy(initial), deepcopy(raw), 'replay_whole', [0, 1])
                first = dispatch(deepcopy(initial), deepcopy(raw[:5]), 'replay_segment', [0])
                second = dispatch(deepcopy(first[0]), deepcopy(raw[5:]), 'replay_segment', [1])
            metering.phase_complete(meter)
        finally:
            row['replay_resources'] = meter.counts(); row['replay_samples'] = meter.predictor_samples()
        replay = dict(raw=raw, whole=whole, segmented=first+second)
        old.require(seal_reload(i, 'replay', replay) == replay, 'replay sealed/reloaded')
        row.update(status='sealed', choice=choice, replay=replay)
    return dict(bank=bank, rows=rows, scientific_admission=False)


def verify_saved(task, root, target, witness_raw5, controller_results, choices, replays, events):
    """Rebuild both controllers, extra H2 and all replay edges from saved IO."""
    old.require(all(len(x) == 2 for x in (controller_results, choices, replays, events)), 'two arms')
    for i in range(2):
        old.verify_controller(task, root, ARM, target, controller_results[i], events[i]['controller'])
    bank = metering.compare_banks(controller_results[0]['candidates'], controller_results[1]['candidates'])
    for i, arm in enumerate(spec.ARMS):
        raw_extra = spec.extras(witness_raw5)[arm]
        cursor = beam.Cursor(events[i]['extra']); dispatch = beam.SavedDispatch(task, cursor)
        contexts = dispatch(beam.checked_context(task['root_context']), raw_extra, 'extra', [0, 1])
        old.require(cursor.index == len(events[i]['extra']) and dispatch.index == 1 and len(dispatch.operations) == 8,
                    'complete extra IO')
        costs = controller_results[i]['candidates']; value = q(task, target, contexts)
        winner = min(range(73), key=lambda j: (costs[j]['cost'] if j < 72 else value, j))
        raw = spec.raw_actions(costs[winner]['commands'] if winner < 72 else raw_extra, 10)
        want = dict(arm=arm, extra=raw_extra, extra_contexts=contexts, extra_cost=value,
                    winner_index=winner, selected_raw=raw)
        old.require(choices[i] == want, 'extra score and selection exact')
        old.require(old.phase.verify_saved_replay(task, raw, events[i]['replay']) == replays[i], 'complete replay exact')
    return dict(bank=bank, verified=True, scientific_admission=False)
