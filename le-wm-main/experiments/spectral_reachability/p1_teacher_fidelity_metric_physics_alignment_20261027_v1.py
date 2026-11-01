"""Saved-only metric/physics sign descriptions. No model or simulator execution."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Both imports are saved-only arithmetic modules, not experimental executors.
from experiments.spectral_reachability import p1_teacher_fidelity_error_decomposition_20261027_v1 as dec
from experiments.spectral_reachability import p1_moving_live_feedback_analysis_20261027_v2 as live

NAME = 'p1_teacher_fidelity_metric_physics_alignment_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
DECOMPOSITION = ROOT / 'results/p1_teacher_fidelity_error_decomposition_20261027_v1_saved_main01.json'
DECOMPOSITION_SHA = '483c40c33eb856c45b380e7d0892e555784236d8e7584c198b3881098c16a0e7'
ANALYSIS = ROOT / 'results/p1_moving_live_feedback_analysis_20261027_v2_main01.json'
ANALYSIS_SHA = '5df7dd90b3da010d1bafbb66198334272ec6470b7b0f7bc62991e4b0dae0b5d8'
PAIRS = (('I', 'O'), ('R', 'I'), ('R', 'O'))
# Direction is for improvement, not a cross-metric weighted scalar.
DIRECTIONS = dict(native_distance7=-1, allxy_position_error4=-1, agent_position_error2=-1,
                  object_position_error=-1, wrapped_object_angle_error=-1, native_angle_term=-1,
                  native_success_recomputed=1, native_success_receipt=1,
                  wrapped_allxy_success=1, object_success=1)
require = dec.require


def scalar(x):
    if x is None:
        return None
    if type(x) is bool:
        return int(x)
    require(type(x) in (int, float) and np.isfinite(x), 'finite scalar or explicit null')
    return x


def delta(lhs, rhs):
    lhs, rhs = scalar(lhs), scalar(rhs)
    return None if lhs is None or rhs is None else lhs-rhs


def compare(latent_delta, physical_delta, direction):
    """Positive improvement favors lhs; exact zero ties never count as reversals."""
    latent_delta, physical_delta = scalar(latent_delta), scalar(physical_delta)
    if latent_delta is None or physical_delta is None or direction is None:
        return dict(status='unknown', latent_improvement_sign=None, physical_improvement_sign=None,
                    same_sign=None, strict_reversal=None, nonzero_agreement=None)
    ls, ps = dec.sign(-latent_delta), dec.sign(direction * physical_delta)
    tie = ls == 0 or ps == 0
    return dict(status='tie' if tie else 'same_direction' if ls == ps else 'opposite_direction',
                latent_improvement_sign=ls, physical_improvement_sign=ps,
                same_sign=ls == ps, strict_reversal=ls*ps == -1,
                nonzero_agreement=None if tie else ls == ps)


def index_rows(rows, identity=False):
    out = {}
    for row in rows:
        i = row['identity'] if identity else row
        require(type(i['source_slot']) is int, 'literal source slot, not bool')
        key = i['source_slot'], i['prefix_arm'], i['mode']
        require(key in dec.KEYS and key not in out and i['episode_id'] == f'e{dec.LAYERS.index(key[:2]):02}_{key[2]}',
                'unique fixed own episode')
        out[key] = row
    require(set(out) == set(dec.KEYS) and len(rows) == 12, 'all12 fixed episodes; no favorable-root selection')
    return out


def make_alignment(decomposition, analysis, report, prepared, teacher_prepare):
    ds = index_rows(decomposition['rows'], True)
    ars, rs = index_rows(analysis['rows']), index_rows(report['rows'])
    require(len(prepared['layers']) == 4 and
            {(l['source_slot'], l['arm']) for l in prepared['layers']} == set(dec.LAYERS), 'all four live layers')
    layers = {(l['source_slot'], l['arm']): l for l in prepared['layers']}
    evaluators = index_rows(teacher_prepare['evaluators'], True)
    old_contrasts = {(c['source_slot'], c['prefix_arm'], c['contrast']): c for c in analysis['contrasts']}
    require(len(old_contrasts) == len(analysis['contrasts']) == 12, 'complete unique original pair contrasts')
    rows, rowmap = [], {}
    for index, key in enumerate(dec.KEYS):
        d, a, r, ev = ds[key], ars[key], rs[key], evaluators[key]
        layer = layers[key[:2]]
        i = d['identity']
        require(type(i['index']) is int and i['index'] == index and i == ev['identity'] and
                i['parent_task_id'] == layer['task_id'] and
                i['parent_selection_sha256'] == layer['selection_sha256'], 'teacher/live task+selection identity')
        require(r['status'] == a['status'] == 'completed10' and r['full_horizon'] is True and
                a['issues'] == [] and r['first_error'] is None and r['secondary_errors'] == [], 'complete own physical episode')
        endpoints = {}
        for step in (5, 10):
            cp = live.checkpoint(r, step)
            require(cp is not None, 'required original native checkpoint')
            # Exact native state7; never truncate velocity to fit a five-coordinate interface.
            state = live.array(cp['state'], (7,)).astype(np.float64)
            goal = live.array(cp['goal'], (7,)).astype(np.float64)
            native = live.native_metrics(state, goal)
            old = a['endpoints'][str(step)]
            require(old['checkpoint_present'] is True and old['native_recomputed'] == native,
                    'independent native7 metrics match original analysis')
            receipt = ev['prefix' if step == 5 else 'label']
            require(r[f'encoding{step}']['latent'] == receipt, 'same own teacher/live latent receipt')
            difference = state-goal
            values = dict(native_distance7=native['native_distance'],
                          allxy_position_error4=float(np.linalg.norm(difference[:4])),
                          agent_position_error2=float(np.linalg.norm(difference[:2])),
                          object_position_error=native['object_position_error'],
                          wrapped_object_angle_error=native['wrapped_object_angle_error'],
                          native_angle_term=native['native_angle_term'],
                          native_success_recomputed=native['native_success'],
                          native_success_receipt=r['native_success'] if step == 10 else None,
                          wrapped_allxy_success=native['wrapped_allxy_success'], object_success=native['object_success'])
            if values['native_success_receipt'] is not None:
                require(type(values['native_success_receipt']) is bool and
                        values['native_success_receipt'] == native['native_success'], 'native success receipt agrees or is null')
            q = old['observed_Q']
            require(q is not None and scalar(q) is not None, 'observed latent Q required, never zero-imputed')
            endpoints[str(step)] = dict(state7=state.tolist(), goal7=goal.tolist(),
                                        signed_native_residual7=difference.tolist(),
                                        state7_array_sha256=dec.sha(state.tobytes()), goal7_array_sha256=dec.sha(goal.tobytes()),
                                        observed_latent_Q=q, metrics=values,
                                        negative_native_angle=native['negative_native_angle'],
                                        native_metric_origin='saved_state_recomputation_not_new_physics',
                                        native_success_origin='original_driver_receipt' if step == 10 else 'unknown_no_step5_receipt')
        require(endpoints['10']['observed_latent_Q'] == d['endpoint']['actual_Q'] and
                dec.sha(live.array(ev['goal'], (192,)).astype(np.float32).tobytes()) == d['goal_array_sha256'] and
                ev['label']['array_sha256'] == d['actual_receipt']['array_sha256'] and
                ev['prefix']['array_sha256'] == d['prefix_array_sha256'], 'exact decomposition Q/goal/label/prefix closure')
        item = dict(identity=i, endpoints=endpoints, predicted_endpoint_latent_Q=d['endpoint']['predicted_Q'],
                    actual_endpoint_latent_Q=d['endpoint']['actual_Q'])
        rows.append(item)
        rowmap[key] = item
    contrasts = []
    for layer_index, (slot, arm) in enumerate(dec.LAYERS):
        for lhs, rhs in PAIRS:
            l, r = rowmap[slot, arm, lhs], rowmap[slot, arm, rhs]
            lr, rr = rs[slot, arm, lhs], rs[slot, arm, rhs]
            old = old_contrasts[slot, arm, lhs+'-'+rhs]
            require(old['prefix_pairing_exact'] is True and old['reason'] is None and
                    lr['initial_exact'] is True and rr['initial_exact'] is True and
                    lr['prefix_pairing'] is not None and lr['prefix_pairing'] == rr['prefix_pairing'],
                    'original exact paired prefix closure; no unpaired fallback')
            predicted = delta(l['predicted_endpoint_latent_Q'], r['predicted_endpoint_latent_Q'])
            endpoints = {}
            for step in ('5', '10'):
                le, re = l['endpoints'][step], r['endpoints'][step]
                q = delta(le['observed_latent_Q'], re['observed_latent_Q'])
                require(q == old['endpoints'][step]['Q'], 'original paired observed Q delta')
                metrics = {}
                for name, direction in DIRECTIONS.items():
                    lv, rv = le['metrics'][name], re['metrics'][name]
                    raw_delta = delta(lv, rv)
                    effective_direction = None if name == 'native_angle_term' and (
                        le['negative_native_angle'] or re['negative_native_angle']) else direction
                    metrics[name] = dict(lhs=lv, rhs=rv, delta_lhs_minus_rhs=raw_delta,
                                         improvement_direction=effective_direction,
                                         actual_latent_Q_alignment=compare(q, raw_delta, effective_direction),
                                         predicted_latent_Q_alignment=compare(predicted, raw_delta, effective_direction) if step == '10' else None)
                    original_name = {'native_distance7': 'native_distance', 'native_success_recomputed': 'native_success'}.get(name, name)
                    if original_name in old['endpoints'][step]:
                        require(raw_delta == old['endpoints'][step][original_name], 'exact original native contrast delta')
                endpoints[step] = dict(observed_latent_Q_delta=q, physics_metrics=metrics)
            contrasts.append(dict(layer_index=layer_index, source_slot=slot, prefix_arm=arm, contrast=lhs+'-'+rhs,
                                  lhs_episode=l['identity']['episode_id'], rhs_episode=r['identity']['episode_id'],
                                  prefix_pairing_exact=True, predicted_endpoint_latent_Q_delta=predicted,
                                  predicted_vs_actual_latent_Q=compare(predicted, endpoints['10']['observed_latent_Q_delta'], -1),
                                  endpoints=endpoints))
    return rows, contrasts


def build(decomposition_path=DECOMPOSITION, decomposition_sha=DECOMPOSITION_SHA,
          analysis_path=ANALYSIS, analysis_sha=ANALYSIS_SHA, saved=None):
    s = saved or dec.Saved()
    d = s.json(decomposition_path, decomposition_sha)
    a = s.json(analysis_path, analysis_sha)
    require(d['schema'] == dec.NAME+'.report' and d['status'] == 'saved_array_diagnostic_not_admitted' and
            a['schema'] == live.NAME+'.report' and a['status'] == 'saved_descriptive_analysis_not_admitted' and
            d['independent_tasks'] == a['independent_tasks'] == 2, 'exact saved authorities')
    # Close every original file binding, then independently reconstruct both saved reports.
    for path, receipt in d['inputs'].items():
        s.read(path, receipt['sha256'])
    s.read(dec.__file__, d['implementation_sha256'])
    reconstructed = dec.analyze(d['input_report'], d['input_report_sha256'],
                              dec.Saved(reader=s.reader, listing=s.listing))
    require({k: v for k, v in d.items() if k != 'implementation_sha256'} == reconstructed,
            'complete saved decomposition reconstruction')
    for path, h in a['input_sha256'].items():
        driver = str(Path(live.__file__).with_name(live.DRIVER+'.py').absolute())
        require((path in d['inputs'] and d['inputs'][path]['sha256'] == h) or
                (path == driver and h == live.DRIVER_SHA), 'shared inputs or original frozen driver authority')
        s.read(path, h)
    # Original analysis source/tests/doc are separately sealed by the pinned analysis.
    # Its tests/doc need not also belong to the teacher's runtime input closure.
    for path, h in a['analysis_source_sha256'].items():
        s.read(path, h)
    ai = a['inputs']
    require(live.build(ai['report'], ai['report_sha256'], ai['prepared'], ai['prepare_sha256']) == a,
            'complete original live analysis v2 reconstruction')
    teacher = s.json(d['input_report'], d['input_report_sha256'])
    tp = dec.decode_prepare(s.read(teacher['prepared'], teacher['prepare_sha256']))
    arguments = tp['arguments']
    require(str(Path(arguments['analysis_path']).absolute()) == str(Path(analysis_path).absolute()) and
            arguments['analysis_sha'] == analysis_sha and arguments['report_path'] == ai['report'] and
            arguments['report_sha'] == ai['report_sha256'] and arguments['live_prepared'] == ai['prepared'] and
            arguments['live_prepare_sha'] == ai['prepare_sha256'], 'teacher original live/analysis arguments closure')
    report = s.json(ai['report'], ai['report_sha256'])
    prepared = s.json(ai['prepared'], ai['prepare_sha256'])
    rows, contrasts = make_alignment(d, a, report, prepared, tp)
    for path, receipt in list(s.files.items()):
        s.read(path, receipt['sha256'])
    return dict(schema=NAME+'.report', status='complete_saved_descriptive_alignment_not_admitted',
                fixed_episode_denominator=12, fixed_layer_denominator=4, independent_tasks=2,
                independent_samples=False, significance=None, rows=rows, contrasts=contrasts,
                decomposition=dict(path=str(Path(decomposition_path).absolute()), sha256=decomposition_sha),
                analysis=dict(path=str(Path(analysis_path).absolute()), sha256=analysis_sha),
                inputs=s.files, original_latent_array_receipts=d['arrays'],
                metric_improvement_directions=DIRECTIONS,
                delta_definition='lhs minus rhs, unrounded; distances lower and successes higher are better',
                tie_policy='exact zero is tie, never reversal; null stays unknown; no epsilon threshold',
                new_neural=0, new_physics=0, training=False, Qfit=False, P1_admission=False, P4_admission=False,
                limitations=['Known two roots, four paired layers, not 12 independent samples.',
                             'No aggregate weighted score, favorable metric/root selection or causal efficacy claim.',
                             'Step5 native success receipt and object-only success are unknown, never zero-filled.',
                             'Native full7 distance includes velocity; allxy4 and objectXY2 differ in meaning.',
                             'Negative legacy native angle disables its improvement-direction comparison.',
                             'Latent/physical sign agreement is descriptive, not proof of physical benefit.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive NEW output')
    result = build()
    result['implementation_sha256'] = dec.sha(Path(__file__).read_bytes())
    raw = dec.canonical(result)
    require(len(raw) <= 4*1024**2, 'bounded descriptive output')
    with args.output.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'complete output write')
        stream.flush()
        os.fsync(stream.fileno())
    print(dec.canonical(dict(output=str(args.output.absolute()), sha256=dec.sha(raw), episodes=12,
                             paired_layers=4, pair_contrasts=12, independent_tasks=2)).decode(), end='')


if __name__ == '__main__':
    main()
