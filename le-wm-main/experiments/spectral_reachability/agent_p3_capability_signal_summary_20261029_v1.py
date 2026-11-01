"""Saved-only seven full J / five U arithmetic, never a physical capability gate.

Only two externally SHA-bound JSON inputs and sealed source bytes are read.
The original complete MAIN receipt is inherited, not re-consumed or re-executed.
No producer imports, model loading, encoder, AD, physics or action selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

NAME = 'agent_p3_capability_signal_summary_20261029_v1'
STAGE = 'agent_p3_same_time_capability_probe_cuda_20261029_v1'
ENDPOINT = 'agent_p3_secant_endpoint_summary_20261029_v1'
BRIDGE = 'agent_p3_same_time_cp10_rgb_bridge_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
ORDER = ((0, 'C'), (0, 'Z'), (0, 'P'), (1, 'Z'), (1, 'P'))
LOGICAL = ((0, 'H1'), (0, 'H3'), (1, 'H1'), (1, 'H3'))
DAMPING = .002
RADIUS = float(np.sqrt(5)*.75)
META, OUTPUT, FLOOR, WALL = 128*1024**2, 16*1024**2, 200*1024**2, 180
ENDPOINT_SHA = '6a6513ebfa68f331bbbecac139af84b0614446e3df739b3d00959a3119f743a0'
SOURCE_PINS = {
    'experiments/spectral_reachability/'+STAGE+'.py': 'dac75c998e1a5ba5b1e8ab562cbaf02c04662844d884c3ae637ccbed3dbf25b4',
    'tests/test_'+STAGE+'.py': '50af7c27fdf473049f79a097b2634ddd3742d89adc23bd4df7211b15c53b46f7',
    'innerdocs/'+STAGE+'.md': '8e16358220327f1696e272493bf632b29dab21c93e5c4df8685ebfc36fb98113',
    'experiments/spectral_reachability/'+ENDPOINT+'.py': '8215c1c0b85f5ccc082c7fa34cb4fb67ac2af6118e9f5183ab077c7ea799dc0e',
    'experiments/spectral_reachability/pusht_full_packet_compiler_20261027.py': 'f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a',
    'innerdocs/agent_p3_secant_same_time_capability_protocol_20261029_v1.md': 'b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf',
}
FALSE_FLAGS = ('P1_admission', 'P2_admission', 'P3_admission', 'freshness',
    'full_C3_demonstrated', 'independent_confirmation', 'physical_release',
    'compute_advantage', 'walltime_advantage', 'scientific_admission',
    'equal_FLOPs', 'fresh', 'C3', 'P1', 'P3', 'training', 'execution_authorized')
AXES = ('predict_API', 'predict_samples', 'predict_context_tokens', 'action_encoder_calls',
    'action_encoder_samples', 'action_encoder_context_tokens', 'predictor_calls',
    'predictor_samples', 'predictor_context_tokens', 'pred_proj_calls', 'pred_proj_rows',
    'encoder_calls', 'encoder_samples', 'projector_calls', 'projector_samples',
    'jacobian_API', 'VJP_directions', 'grad_API', 'raw_backward_callbacks',
    'model_SVD_calls', 'array_SVD_calls', 'solve_calls', 'signal_compile_calls',
    'program_compile_calls', 'replay_calls', 'metric_calls', 'schedule_decisions')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    """Producer compact JSON hash, explicitly no newline."""
    return hashlib.sha256(encode(value)).hexdigest()


def same(a, b):
    return encode(a) == encode(b)


def hash_value(h):
    need(type(h) is str and len(h) == 64 and all(c in '0123456789abcdef' for c in h), 'literal SHA256')


def absolute(path):
    p = Path(path)
    need(p.is_absolute() and p.resolve() == p and not p.is_symlink(), 'canonical absolute path')
    return p


def sha(path):
    p = absolute(path)
    need(p.is_file() and p.stat().st_size <= META, 'bounded regular input/source')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(path, expected):
    hash_value(expected)
    need(sha(path) == expected, 'external byte SHA')
    body = absolute(path).read_bytes()
    need(hashlib.sha256(body).hexdigest() == expected, 'stable read SHA')
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    value = json.loads(body, object_pairs_hook=pairs,
                       parse_constant=lambda s: need(False, 'nonfinite JSON'))
    encode(value)
    return value


def array(v, shape, f32=False):
    def numeric(x):
        if type(x) is list:
            for item in x:
                numeric(item)
        else:
            need(type(x) in (int, float) and math.isfinite(x), 'finite literal number, not bool')
    numeric(v)
    a = np.asarray(v, np.float64)
    need(a.shape == shape and np.isfinite(a).all(), 'complete exact array shape')
    if f32:
        need(np.array_equal(a, a.astype(np.float32).astype(np.float64)), 'exact original float32 values')
    return a


def fsum_sq(v):
    return math.fsum(float(x)*float(x) for x in v)


def q(v, goal):
    return fsum_sq(array(v, (192,))-array(goal, (192,)))/192


def cmp(a, b):
    if a is None or b is None:
        return dict(a_minus_b=None, relation='null')
    need(math.isfinite(a) and math.isfinite(b), 'finite comparison')
    return dict(a_minus_b=a-b, relation='less' if a < b else 'greater' if a > b else 'tie')


def expected_resources():
    v = dict.fromkeys(AXES, 0)
    for k in ('predict_API', 'predict_samples', 'action_encoder_calls', 'action_encoder_samples',
              'predictor_calls', 'predictor_samples', 'pred_proj_calls'):
        v[k] = 24
    for k in ('predict_context_tokens', 'action_encoder_context_tokens', 'predictor_context_tokens', 'pred_proj_rows'):
        v[k] = 72
    v.update(jacobian_API=7, VJP_directions=1344, grad_API=1344, raw_backward_callbacks=1344,
             array_SVD_calls=7, solve_calls=7, signal_compile_calls=7,
             program_compile_calls=5, replay_calls=10, metric_calls=22)
    return v


def context(c):
    need(c['latent_dim'] == 192 and c['context_length'] == 3 and c['action_dim'] == 10, 'full native context')
    array(c['latent_history'], (3, 192), True)
    array(c['action_history'], (2, 10), True)


def check_child(t, after, raw5):
    before = t['root_context']
    context(before)
    context(after)
    raw5 = array(raw5, (5, 2), True).astype('<f4')
    mean = array(t['action_mean'], (2,), True).astype(np.float32)
    std = array(t['action_std'], (2,), True).astype(np.float32)
    need(np.all(std > 0) and np.all(np.abs(raw5) <= 1), 'unchanged legal raw/positive normalizer')
    token = ((raw5-mean)/std).reshape(10)
    need(same(after['latent_history'][:-1], before['latent_history'][1:]) and
         np.asarray(after['action_history'], '<f4').tobytes() ==
         np.asarray([before['action_history'][-1], token.tolist()], '<f4').tobytes(),
         'all shifted native history and once-normalized outgoing token bits')


def native_metrics(state, goal):
    d = array(state, (7,))-array(goal, (7,))
    total, xy, vel = fsum_sq(d), fsum_sq(d[:4]), fsum_sq(d[5:])
    a = abs(float(d[4]))
    native_angle = min(a, 2*math.pi-a)
    return dict(native7_distance=math.sqrt(total), native7_squared=total,
                joint_xy_error=math.sqrt(xy), joint_xy_squared=xy,
                agent_velocity_error=math.sqrt(vel), agent_velocity_squared=vel,
                raw_angle_error=a, raw_angle_squared=float(d[4])**2,
                wrapped_angle_error=abs(math.atan2(math.sin(d[4]), math.cos(d[4]))),
                native_angle_term=native_angle,
                native_success=bool(math.sqrt(xy) < 20 and native_angle < math.pi/9))


def validate_endpoint(e):
    need(e['schema'] == ENDPOINT+'.summary' and e['denominator'] ==
         dict(C_logical_slots=4, C_no_witness=3, ZP_references=4, comparisons=8), 'endpoint whole fixed denominator')
    need(e['actual_time'] == dict(control=10, absolute=20) and e['known_roots'] == 2 and
         e['fresh'] is False and all(e[k] is False for k in
         ('scientific_admission', 'P1_admission', 'P3_admission', 'full_C3_demonstrated')), 'endpoint original scope/time')
    need(len(e['C_slots']) == len(e['ZP_root_references']) == 4 and len(e['comparisons']) == 8,
         'all logical physical endpoints/comparisons')
    for rows, order in ((e['C_slots'], LOGICAL), (e['ZP_root_references'], ((0,'Z'),(0,'P'),(1,'Z'),(1,'P')))):
        for i, (row, key) in enumerate(zip(rows, order, strict=True)):
            need((row['index'], row['root_slot'], row['arm']) == (i, *key) and
                 row['control_step'] == 10 and row['absolute_step'] == 20, 'endpoint original fixed roles/time')
            need(row['available'] is (row['endpoint'] is not None) and
                 (row['metrics'] is None) is (row['endpoint'] is None), 'null not zero/tie')
            if row['endpoint'] is None:
                need(row['status'] == 'no_witness' and row['missing_reason'] == 'no_witness', 'fixed missing C authority')
            else:
                m = row['metrics']
                need(same(m['state7'], row['endpoint']['state']) and row['endpoint']['step'] == 10, 'real same-time state')
                independently = native_metrics(m['state7'], m['goal7'])
                need(all(same(m[k], v) for k, v in independently.items()), 'independent endpoint native arithmetic')
    need(e['C_slots'][0]['available'] is True and all(x['endpoint'] is None for x in e['C_slots'][1:]),
         'one physical C / three no_witness, never borrowed states')


def validate_receipt(r):
    need(r['schema'] == STAGE+'.consume' and r['status'] == 'saved_complete_NOT_admitted',
         'actual complete MAIN receipt, no RAM/partial')
    need(r['fixed_signals'] == len(r['signals']) == 7 and
         r['fixed_probes'] == len(r['probes']) == len(r['plans']) == 5 and len(r['metrics']) == 22,
         'all seven signals / five probes / twenty-two Q')
    need(all(r[k] is False for k in FALSE_FLAGS), 'all literal scientific flags false')
    v = r['verification']
    need(v['full_native_closure'] is True and v['native_calls'] == 17 and v['saved_SVD_calls'] == 7 and
         v['saved_new_neural_calls'] == v['saved_new_physics'] == 0, 'inherit full native closure, not status alone')
    resource = r['resource_receipt']
    need(resource['caps'] == resource['attempted'] == resource['completed'] == expected_resources(), 'full paid resource vector')
    need(resource['FLOPs'] is resource['cuda_ns'] is resource['backward_engine_calls'] is None and
         r['unknowns'] == dict(FLOPs=None, backward_engine_calls=None, cuda_ns=None), 'unknown compute is not zero')
    need(r['upstream_fullstrict_rebuilt_here'] is False, 'explicit inherited upstream scope')
    need([(m['kind'], m['index']) for m in r['metrics']] ==
         [('actual', i) for i in range(5)]+[('nominal', i) for i in range(7)]+
         [(field, i) for i in range(5) for field in ('whole', 'segmented')], 'all22 fixed ordered metrics')
    for i, s in enumerate(r['signals']):
        need(s['index'] == i and s['status'] == 'complete', 'complete ordered signal')
        resp = s['response']
        need(resp['horizon'] == 1 and resp['producer_sha256'] ==
             '6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b', 'same original H1 producer')
        array(resp['nominal'], (192,), True)
        array(resp['jacobian'], (192, 10), True)
        array(s['solve']['params'], (10,))
        array(s['solve']['singular_values'], (10,))
        need(s['solve']['radius'] == RADIUS and s['solve']['ball_multiplier'] >= 0, 'unchanged original radius/multiplier')
        context(s['nominal_context'])
        need(same(s['nominal_context']['latent_history'][-1], resp['nominal']), 'full nominal endpoint')
    return r


def signal_math(response, solve, goal):
    """Independent full 192x10 SVD and fsum math; not the producer solver."""
    j = array(response['jacobian'], (192, 10), True)
    residual = array(goal, (192,))-array(response['nominal'], (192,), True)
    u, singular, vh = np.linalg.svd(j, full_matrices=False)
    target = np.asarray([math.fsum(float(u[k, i])*float(residual[k]) for k in range(192)) for i in range(10)])
    mu = float(solve['ball_multiplier'])
    coord = singular*target/(singular*singular+DAMPING+mu)
    unbounded = singular*target/(singular*singular+DAMPING)
    perp = residual-u@target
    perp_energy = fsum_sq(perp)
    modes = []
    for i in range(10):
        s, t, c, un = map(float, (singular[i], target[i], coord[i], unbounded[i]))
        rem = t-s*c
        penalty = DAMPING*c*c/192
        modes.append(dict(index=i, singular_value=s, target_coordinate=t, target_energy=t*t/192,
            bounded_coordinate=c, unconstrained_ridge_coordinate=un,
            predicted_response_coordinate=s*c, remaining_target_coordinate=rem,
            linear_terminal_contribution=rem*rem/192, ridge_penalty_contribution=penalty,
            ridge_objective_contribution=rem*rem/192+penalty,
            linear_gain_contribution=(t*t-rem*rem)/192,
            cap_objective_change=(rem*rem+DAMPING*c*c-(t-s*un)**2-DAMPING*un*un)/192))
    prefixes = []
    for rank in range(11):
        linear = math.fsum([perp_energy/192]+[x['linear_terminal_contribution'] for x in modes[:rank]]+
                           [x['target_energy'] for x in modes[rank:]])
        ridge = math.fsum(x['ridge_penalty_contribution'] for x in modes[:rank])
        prefixes.append(dict(retained_rank=rank, linear_terminal=linear, ridge_penalty=ridge,
            ridge_objective=linear+ridge, omitted_target_energy=math.fsum(x['target_energy'] for x in modes[rank:]),
            omitted_full_solution_gain=math.fsum(x['linear_gain_contribution'] for x in modes[rank:])))
    p = array(solve['params'], (10,))
    fitted = np.asarray([math.fsum(float(j[k,i])*float(p[i]) for i in range(10)) for k in range(192)])
    remaining = residual-fitted
    direct = dict(nominal_Q=fsum_sq(residual)/192, linear_terminal_cost=fsum_sq(remaining)/192,
                  ridge_penalty=DAMPING*fsum_sq(p)/192)
    direct['own_reach_value'] = direct['linear_terminal_cost']+direct['ridge_penalty']
    independent_p = vh.T@coord
    # No numerical rank threshold or tolerance is a new scientific acceptance gate.
    return dict(jacobian=j.tolist(), nominal=response['nominal'], goal=goal, residual=residual.tolist(),
        singular_values=singular.tolist(), saved_singular_values=solve['singular_values'],
        singular_value_roundoff=(singular-np.asarray(solve['singular_values'])).tolist(),
        modes=modes, rank_prefixes=prefixes, orthogonal_residual=perp.tolist(),
        orthogonal_target_energy=perp_energy/192,
        target_energy_roundoff=fsum_sq(residual)/192-math.fsum([perp_energy/192]+[m['target_energy'] for m in modes]),
        direct_saved_params=direct, saved_original_solve=deepcopy(solve),
        independent_params_from_saved_multiplier=independent_p.tolist(),
        params_roundoff=(independent_p-p).tolist(),
        direct_minus_saved={k:direct[k]-solve[k] for k in ('linear_terminal_cost','ridge_penalty','own_reach_value')},
        spectral_minus_direct=dict(linear_terminal=prefixes[-1]['linear_terminal']-direct['linear_terminal_cost'],
                                   ridge_objective=prefixes[-1]['ridge_objective']-direct['own_reach_value']),
        truncation='all ranks 0..10; retain full-solution multiplier, NOT reoptimized or selected; original has no hard truncation',
        physical_gain_guaranteed=False)


def check_plan(r, i):
    plan, probe, sig = r['plans'][i], r['probes'][i], r['signals'][i+2]
    bare = deepcopy(plan)
    h = bare.pop('witness_sha256')
    need(digest(bare) == h and plan['schema'] == STAGE+'.probe_plan' and
         (plan['index'], plan['root_slot'], plan['role']) == (i, *ORDER[i]) and
         plan['checkpoint'] == 10 and plan['probe_checkpoint'] == 15, 'fixed whole plan/witness hash identity')
    need(all(plan[k] is False for k in FALSE_FLAGS), 'plan all false flags')
    a = plan['authority']
    need(a['prepared'] == r['prepared'] and all(a[k] == r[k] for k in
         ('prepare_sha256', 'report_sha256', 'completion_sha256')), 'actual receipt authority domains')
    need(plan['bridge_authority'] == r['bridge_authority'] and plan['signal'] == sig['response'] and
         plan['solve'] == sig['solve'] and plan['nominal_context'] == sig['nominal_context'], 'whole signal/solve/bridge exact join')
    need(probe['index'] == i and probe['status'] == 'complete' and
         same(plan['raw5'], probe['raw']) and same(plan['whole'], probe['whole']) and
         same(plan['segmented'], probe['segmented']) and same(plan['whole'], plan['segmented']) and
         len(plan['whole']) == 1, 'same U independent dual full-context H1 replay')
    need(plan['native_call_ids'] == dict(signal=i+2, whole=7+2*i, segmented=8+2*i), 'all original native call IDs')
    t = plan['task']
    need(digest(t) == plan['task_sha256'] and np.array_equal(array(t['projection'], (192,192)), np.eye(192)), 'full original task/identity Q')
    for key, value in (('goal_sha256',t['goal']), ('projection_sha256',t['projection']),
                       ('normalizer_sha256',[t['action_mean'],t['action_std']]), ('context_sha256',t['root_context'])):
        need(t['binding'][key] == digest(value), 'bound original '+key)
    need(plan['signal']['binding_sha256'] == digest(t['binding']) and
         plan['signal']['parent_context_sha256'] == digest(t['root_context']), 'actual cp10 signal task/context')
    raw5 = array(plan['raw5'], (5,2), True).astype('<f4')
    params = array(plan['params_float64'], (10,))
    need(same(plan['params_float64'], plan['solve']['params']) and
         np.tanh(params).astype('<f4').tobytes() == raw5.tobytes() and
         raw5.tobytes().hex() == plan['raw_float32_bits'] and
         hashlib.sha256(raw5.tobytes()).hexdigest() == plan['raw_float32_sha256'], 'original one-cast tanh U bits')
    incoming = array(plan['causal_raw10'], (10,2), True).astype('<f4')
    need(incoming.tobytes().hex() == plan['causal_raw10_bits'], 'causal raw10 bits')
    token = ((incoming-np.asarray(t['action_mean'],np.float32))/np.asarray(t['action_std'],np.float32)).reshape(2,10)
    need(token.tobytes() == np.asarray(t['root_context']['action_history'],'<f4').tobytes(), 'both chronological incoming tokens regenerated once')
    ident = plan['endpoint_identity']
    need(ident['namespace'] == BRIDGE and (ident['index'],ident['root_slot'],ident['role']) == (i,*ORDER[i]) and
         ident['absolute_steps'] == [10,15,20] and [x['absolute_step'] for x in ident['history_sources']] == [10,15,20] and
         ident['raw10_float32_sha256'] == hashlib.sha256(incoming.tobytes()).hexdigest(), 'actual endpoint three-history namespace')
    check_child(t, plan['whole'][0], plan['raw5'])
    check_child(t, plan['nominal_context'], [[0.,0.]]*5)
    return plan


def summarize(r, e):
    validate_receipt(r)
    validate_endpoint(e)
    plans = [check_plan(r, i) for i in range(5)]
    goals = {root:plans[next(i for i,x in enumerate(ORDER) if x[0] == root)]['task']['goal'] for root in (0,1)}
    for root in (0,1):
        group = [p for p in plans if p['root_slot'] == root]
        for p in group:
            need(all(same(p['task'][k],group[0]['task'][k]) for k in
                 ('goal','projection','action_mean','action_std')) and
                 p['task']['binding']['model_sha256'] == group[0]['task']['binding']['model_sha256'], 'same-root original goal/Q/normalizer/model')
    rows = []
    for i, s in enumerate(r['signals']):
        root, role, cp = (i,'reference',5) if i < 2 else (*ORDER[i-2],10)
        m = signal_math(s['response'], s['solve'], goals[root])
        rows.append(dict(index=i,root_slot=root,role=role,control_step=cp,absolute_step=cp+10,math=m,
                         source_binding_sha256=s['response']['binding_sha256'],
                         source_parent_context_sha256=s['response']['parent_context_sha256']))
    metric = {(x['kind'],x['index']):x['Q'] for x in r['metrics']}
    probes = []
    for i, p in enumerate(plans):
        # No future cp15 actual data is available here. These are model predictions only.
        t = p['task']; goal = t['goal']; root,role = ORDER[i]
        quantities = dict(actual_cp10=q(t['root_context']['latent_history'][-1],goal),
                          nominal=q(p['signal']['nominal'],goal),
                          whole=q(p['whole'][0]['latent_history'][-1],goal),
                          segmented=q(p['segmented'][0]['latent_history'][-1],goal))
        originals = dict(actual_cp10=metric['actual',i], nominal=metric['nominal',i+2],
                         whole=metric['whole',i], segmented=metric['segmented',i])
        need(same(p['Q'],originals), 'all original plan/metric values joined')
        for key, v in quantities.items():
            vector = t['root_context']['latent_history'][-1] if key == 'actual_cp10' else p['signal']['nominal'] if key == 'nominal' else p[key][0]['latent_history'][-1]
            # Original exact NumPy route is a separate control, never substitutes for fsum.
            native_q = float(np.mean((np.asarray(t['projection'],np.float64) @
                                     (np.asarray(vector,np.float64)-np.asarray(goal,np.float64)))**2))
            need(native_q == originals[key], 'original exact NumPy Q route')
        physical = e['C_slots'][0] if role == 'C' else e['ZP_root_references'][2*root+(role == 'P')]
        source = p['source']; f = physical['endpoint']
        need(physical['available'] is True and physical['task_id'] == p['base_task_id'] and
             source['step'] == f['step'] == 10 and all(same(source[k],f[k]) for k in
             ('state','snapshot','pixel_sha256','file_sha256')), 'same original physical endpoint/task/full snapshot/RGB')
        need(same(source['state'],source['snapshot']['state']), 'actual native state bookend')
        parent_key = 'C' if role == 'C' else 'ZP'
        need(p['endpoint_identity']['source_receipt'] == e['authorities'][parent_key] and
             p['endpoint_identity']['source_report']['sha256'] == e['inherited_accounting'][parent_key]['physical_report_sha256'] and
             same(p['endpoint_identity']['history_sources'][-1]['snapshot'],source['snapshot']) and
             p['endpoint_identity']['history_sources'][-1]['pixel_sha256'] == source['pixel_sha256'],
             'same physical full authority and actual last history frame')
        ref = rows[root]['math']; now = rows[i+2]['math']
        probes.append(dict(index=i,root_slot=root,role=role,control_step=10,probe_control_step=15,
            task_sha256=p['task_sha256'],endpoint_identity=p['endpoint_identity'],source=source,
            raw5=p['raw5'],raw_float32_sha256=p['raw_float32_sha256'],witness_sha256=p['witness_sha256'],
            Q_fsum=quantities,Q_original_numpy=originals,
            fsum_minus_original={k:quantities[k]-originals[k] for k in quantities},
            predicted_improvement_vs_same_context_zero=quantities['nominal']-quantities['whole'],
            predicted_improvement_vs_actual_cp10=quantities['actual_cp10']-quantities['whole'],
            same_U_whole_vs_segmented=cmp(quantities['whole'],quantities['segmented']),
            linear_vs_nonlinear=cmp(now['direct_saved_params']['linear_terminal_cost'],quantities['whole']),
            cp10_minus_same_job_cp5=dict(
                own_reach=now['direct_saved_params']['own_reach_value']-ref['direct_saved_params']['own_reach_value'],
                nominal_Q=now['direct_saved_params']['nominal_Q']-ref['direct_saved_params']['nominal_Q'],
                singular_values=[a-b for a,b in zip(now['singular_values'],ref['singular_values'],strict=True)]),
            native_cp10=deepcopy(physical['metrics']),actual_cp15_U=None,actual_cp15_Z2=None,
            actual_probe_gain=None,physical_gain_guaranteed=False))
    # Include all seven nominal Q native-route checks, including the two cp5 references.
    for i,s in enumerate(r['signals']):
        root = i if i < 2 else ORDER[i-2][0]
        value = float(np.mean((np.asarray(s['response']['nominal'],np.float64)-np.asarray(goals[root],np.float64))**2))
        need(value == metric['nominal',i], 'all seven original nominal Q controls')
    pairs = []
    for ci,(root,arm) in enumerate(LOGICAL):
        c = probes[0] if ci == 0 else None
        for role in ('Z','P'):
            b = probes[next(i for i,x in enumerate(ORDER) if x == (root,role))]
            cm = None if c is None else c['native_cp10']
            bm = b['native_cp10']
            aidx, bidx = (2, b['index']+2)
            R = cmp(None if c is None else rows[aidx]['math']['direct_saved_params']['own_reach_value'],
                    rows[bidx]['math']['direct_saved_params']['own_reach_value'])
            original_R = cmp(None if c is None else r['signals'][aidx]['solve']['own_reach_value'],
                             r['signals'][bidx]['solve']['own_reach_value'])
            D = cmp(None if cm is None else cm['native7_distance'],bm['native7_distance'])
            pairs.append(dict(index=len(pairs),root_slot=root,C_arm=arm,control_role=role,
                available=c is not None,missing_reason=None if c is not None else 'original_C_no_witness',
                D10_C_vs_control=D,immediate_cost_condition=None if c is None else D['relation']=='greater',
                native_axes_C_vs_control={k:cmp(None if cm is None else cm[k],bm[k]) for k in
                    ('joint_xy_error','raw_angle_error','wrapped_angle_error','agent_velocity_error')},
                native_success_C=None if cm is None else cm['native_success'],native_success_control=bm['native_success'],
                R10_C_vs_control=R,R10_original_solver_C_vs_control=original_R,
                singular_values_C_minus_control=None if c is None else [a-b for a,b in zip(
                    rows[aidx]['math']['singular_values'],rows[bidx]['math']['singular_values'],strict=True)],
                predicted_zero_relative_improvement_C_vs_control=cmp(None if c is None else
                    c['predicted_improvement_vs_same_context_zero'],b['predicted_improvement_vs_same_context_zero']),
                actual_G_C_vs_control=None,actual_D15_CU_vs_controlU=None,full_protocol_condition=None))
    conjunction = [dict(root_slot=root,C_arm=arm,
        C_immediate_cost_both_controls=None if ci else all(p['immediate_cost_condition'] for p in pairs[:2]),
        full_delayed_benefit_chain=None,reason='stageC actual U/Z2 absent; local SVD and model gain are not physical evidence')
        for ci,(root,arm) in enumerate(LOGICAL)]
    root_pairs = []
    for ai,bi in ((0,1),(0,2),(1,2),(3,4)):
        a,b = probes[ai],probes[bi]
        am,bm = rows[ai+2]['math'],rows[bi+2]['math']
        root_pairs.append(dict(index=len(root_pairs),root_slot=a['root_slot'],A_role=a['role'],B_role=b['role'],
            control_step=10,same_time=True,
            own_reach_A_vs_B=cmp(am['direct_saved_params']['own_reach_value'],bm['direct_saved_params']['own_reach_value']),
            nominal_Q_A_vs_B=cmp(am['direct_saved_params']['nominal_Q'],bm['direct_saved_params']['nominal_Q']),
            singular_values_A_minus_B=[x-y for x,y in zip(am['singular_values'],bm['singular_values'],strict=True)],
            J_difference_squared_frobenius=fsum_sq((np.asarray(am['jacobian'])-np.asarray(bm['jacobian'])).reshape(-1)),
            predicted_zero_relative_improvement_A_vs_B=cmp(a['predicted_improvement_vs_same_context_zero'],b['predicted_improvement_vs_same_context_zero']),
            physical_probe_gain_A_vs_B=None,causal_identification=False))
    return dict(signals=rows,probes=probes,comparisons=pairs,C_dual_control_conditions=conjunction,
        same_time_role_pairs=root_pairs,
        denominator=dict(signals=7,cp5_references=2,cp10_endpoints=5,full_J_elements=13440,
                         singular_values=70,probes=5,dual_model_replays=10,logical_C_slots=4,
                         C_no_witness=3,C_control_comparisons=8,same_time_role_pairs=4),
        known_roots=2,independent_new_roots=0,physical_probe_slots_pending=10,
        inherited_stageB_resource_receipt=deepcopy(r['resource_receipt']),
        inherited_encoder_accounting=deepcopy(r['inherited_counts']),
        inherited_physical_accounting=deepcopy(r['inherited_physical_counts']),
        unknowns=deepcopy(r['unknowns']),new_neural_calls=0,new_AD_calls=0,new_physics_calls=0,
        local_array_SVD_calls=7,independent_fullstrict=False,winner=None,**dict.fromkeys(FALSE_FLAGS,False))


def source_closure():
    sources = {}
    for rel, h in SOURCE_PINS.items():
        path = ROOT/rel
        need(sha(path) == h, 'frozen source byte identity '+rel)
        sources[str(path)] = h
    for path in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        sources[str(path)] = sha(path)
    return sources


def build(refs, clock=time.monotonic):
    start = clock()
    need(set(refs) == {'stageB_receipt','endpoint_summary'}, 'exact two saved authorities')
    for ref in refs.values():
        need(set(ref) == {'path','sha256'}, 'explicit path/SHA authority')
        absolute(ref['path']); hash_value(ref['sha256'])
    need(refs['endpoint_summary']['sha256'] == ENDPOINT_SHA, 'fixed already completed endpoint summary SHA')
    r = read(refs['stageB_receipt']['path'],refs['stageB_receipt']['sha256'])
    e = read(refs['endpoint_summary']['path'],refs['endpoint_summary']['sha256'])
    sources = source_closure()
    result = summarize(r,e)
    sources.update({v['path']:v['sha256'] for v in refs.values()})
    for path,h in sources.items():
        need(sha(path) == h, 'stable source/authority after analysis')
    need(clock()-start <= WALL, 'bounded saved build wall')
    return dict(schema=NAME+'.summary',authorities=deepcopy(refs),sources=sources,
        authentication='MAIN complete SHA-bound receipt inherited; no upstream fullstrict rerun; no ancestor IO',
        arithmetic='independent NumPy thin SVD7; scalar math.fsum192/full10; original NumPy Q control separate',
        SVD_basis_warning='signs/repeated-subspace basis not scientifically identifiable; full-rank aggregate and roundoff reported',
        native_angle_warning='original raw native min-angle preserved; wrapped diagnostic separate',**result)


def write(path,value):
    p = absolute(path); body = encode(value)+b'\n'
    need(p.name.startswith(NAME+'_') and len(body) <= OUTPUT and
         shutil.disk_usage(p.parent).free >= FLOOR+len(body), 'own bounded new output and free disk floor')
    with p.open('xb') as f:
        need(f.write(body) == len(body), 'complete saved output write')
        f.flush(); os.fsync(f.fileno())
    fd = os.open(p.parent,os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(p),sha256=sha(p))


def verify(path, expected):
    summary = read(path,expected)
    need(same(summary,build(summary['authorities'])), 'whole saved summary rebuilt, no ignored fields')
    return dict(status='saved_arithmetic_verified_NOT_admitted',new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd',required=True)
    p = sub.add_parser('build')
    for stem in ('receipt','endpoint-summary'):
        p.add_argument('--'+stem,required=True)
        p.add_argument('--'+stem+'-sha256',required=True)
    p.add_argument('--output',required=True)
    p = sub.add_parser('verify')
    p.add_argument('--summary',required=True);p.add_argument('--summary-sha256',required=True)
    args = ap.parse_args(argv); start = time.monotonic()
    if args.cmd == 'build':
        refs = dict(stageB_receipt=dict(path=args.receipt,sha256=args.receipt_sha256),
                    endpoint_summary=dict(path=args.endpoint_summary,sha256=args.endpoint_summary_sha256))
        value = write(args.output,build(refs))
    else:
        value = verify(args.summary,args.summary_sha256)
    need(time.monotonic()-start <= WALL, 'inclusive read/math/fsync wall; failure artifact retained, not accepted')
    print(encode(value).decode(),flush=True)


if __name__ == '__main__':
    main()
