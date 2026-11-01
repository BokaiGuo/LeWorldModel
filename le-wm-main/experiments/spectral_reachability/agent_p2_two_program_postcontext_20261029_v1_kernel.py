"""Approved full30 exploratory program transfer; NumPy only, no neural dispatch."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np

NAME = 'agent_p2_two_program_postcontext_20261029_v1'
ARMS = ('two_program_transfer', 'joint_two_round')
C15, C30 = float(.75*np.sqrt(15)), float(.75*np.sqrt(30))
ALPHAS = (1., .5, .25, .125)
COUNTS = {ARMS[0]: (18, 0), ARMS[1]: (96, 2)}
FLAGS = dict(P1_admission=False, P2_mechanism=False, physical_release=False,
             fresh=False, scientific_admission=False, equal_FLOPs=False)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'


def digest(value):
    return hashlib.sha256(raw(value)).hexdigest()


def equal(a, b, why):
    require(raw(a) == raw(b), why)


def finite(x, shape):
    a = np.asarray(x)
    require(a.dtype.kind in 'fiu' and a.shape == shape and np.isfinite(a).all(), 'literal finite numeric shape')
    # Mixed bool/number Python containers must not be promoted into valid numbers.
    def literals(v):
        return all(literals(z) for z in v) if isinstance(v, (list, tuple)) else not isinstance(v, (bool, np.bool_))
    require(literals(x), 'no bool numeric fields')
    return a.astype(np.float64)


def context(c):
    require(type(c) is dict and set(c) == {'layout', 'context_length', 'latent_dim', 'action_dim',
            'latent_history', 'action_history'}, 'full context exact fields')
    require(c['layout'] == 'states_with_between_state_actions_v1' and
            all(type(c[k]) is int and c[k] == v for k, v in
                (('context_length', 3), ('latent_dim', 192), ('action_dim', 10))), 'native context contract')
    for k, shape in (('latent_history', (3, 192)), ('action_history', (2, 10))):
        a = finite(c[k], shape)
        require(np.array_equal(a, a.astype(np.float32).astype(float)), 'literal float32 context, no repair')
    return c


def chart(u):
    a = np.asarray(u)
    require(a.ndim == 1 and a.size in (30, 60), 'only full15/full30 chart')
    return np.tanh(finite(u, (a.size,))).astype('<f4').reshape(-1, 5, 2)


def legal(u, cap=C30):
    a = finite(u, (60,) if cap == C30 else (30,))
    require(float(np.linalg.norm(a)) <= cap, 'strict float64 logit sphere, no repair')
    return a


def objective(c, goal):
    context(c)
    value = float(np.mean((finite(c['latent_history'][-1], (192,))-finite(goal, (192,)))**2))
    require(np.isfinite(value), 'finite original model objective')
    return value


def proposals(u, z, j, goal):
    u = legal(u)
    j, z, goal = finite(j, (192, 60)), finite(z, (192,)), finite(goal, (192,))
    grad = (2/192)*(j*(1-np.tanh(u)**2)[None, :]).T@(z-goal)
    n = float(np.linalg.norm(grad))
    require(np.isfinite(grad).all() and np.isfinite(n), 'finite continuous gradient and norm')
    target = u.copy() if n == 0 else -.5*C30*grad/n
    d = target-u
    slope = float(grad@d)
    if slope >= 0:
        d = np.zeros_like(u)
    values = [u]+[u+a*d for a in ALPHAS]
    for v in values:
        legal(v)
    return dict(logits=[v.tolist() for v in values], gradient=grad.tolist(),
                direction=d.tolist(), slope=slope)


def replay(c, u, backend, tag):
    """New whole six packets plus two independently initiated program segments."""
    actions = chart(u)
    whole = backend.rollout(c, actions, tag+'/whole30')
    first = backend.rollout(c, actions[:3], tag+'/segment_A')
    second = backend.rollout(first[-1], actions[3:], tag+'/segment_B')
    equal(whole, first+second, 'whole30 / two full15 segment exact full-context replay')
    return whole, first+second


def execute(slot, arm, backend):
    require(arm in ARMS, 'fixed arm')
    c = deepcopy(context(slot['root_context']))
    goal = finite(slot['goal'], (192,))
    a, b = slot['programs']
    for program, index in ((a, 1), (b, 2)):
        require(program['id'] == f'r{slot["root_slot"]}/joint_H3_no_composition/c{index:02}/p0' and
                program['planned_steps'] == program['committed_steps'] == 15, 'fixed full programs, not packets')
        legal(program['logits'], C15)
        equal(program['input_context'], c, 'original programs both rooted at original task')
    u = legal(a['logits']+b['logits'])
    inherited = dict(B_old_input=deepcopy(b['input_context']), B_old_contexts=deepcopy(b['contexts']))
    rounds = []
    if arm == ARMS[0]:
        ca = backend.rollout(c, chart(a['logits']), 'transfer/A')
        equal(ca, a['contexts'], 'fresh A exact inherited complete contexts')
        cb = backend.rollout(ca[-1], chart(b['logits']), 'transfer/B_at_A_postcontext')
        initial = ca+cb
        winner = initial
        backend.event(dict(kind='transfer_frozen', A=a['program_sha256'], B=b['program_sha256'],
                           new_B_context_sha256=digest(ca[-1]), contexts_sha256=digest(initial)))
    else:
        initial = None
        for iteration in range(2):
            tag = f'joint/round{iteration}'
            paths = backend.rollout(c, chart(u), tag+'/incumbent')
            if initial is None:
                initial = deepcopy(paths)
            j = backend.jacobian(c, u, tag+'/raw_J')
            equal(backend.last_ad_contexts, paths, 'literal raw J primal full six-context replay')
            prop = proposals(u, paths[-1]['latent_history'][-1], j, goal)
            es = [dict(index=0, logits=u.tolist(), contexts=paths, Q=objective(paths[-1], goal))]
            for index, v in enumerate(prop['logits'][1:], 1):
                path = backend.rollout(c, chart(v), tag+f'/candidate{index}')
                es.append(dict(index=index, logits=v, contexts=path, Q=objective(path[-1], goal)))
            w = min(es, key=lambda e: (e['Q'], e['index']))
            backend.event(dict(kind='joint_selection', iteration=iteration, selected=w['index'],
                               evaluations_sha256=digest(es), proposals_sha256=digest(prop)))
            winner = backend.rollout(c, chart(w['logits']), tag+'/selected_replay')
            equal(winner, w['contexts'], 'guarded selected whole replay')
            require(w['Q'] <= es[0]['Q'], 'modeled nonworsening only')
            rounds.append(dict(iteration=iteration, input_logits=u.tolist(), proposals=prop,
                               evaluations=es, selected=w['index'], selected_contexts=winner))
            u = legal(w['logits'])
    whole, segmented = replay(c, u, backend, arm)
    equal(whole, winner, 'fresh final whole contexts equal selected witness')
    result = dict(arm=arm, root_slot=slot['root_slot'], initial_logits=a['logits']+b['logits'],
                  logits=u.tolist(), commands=chart(u).reshape(30, 2).tolist(),
                  raw_float32_bits=chart(u).tobytes().hex(), initial_contexts=initial,
                  contexts=whole, segmented_contexts=segmented, rounds=rounds,
                  inherited=inherited, initial_Q=objective(initial[-1], goal),
                  final_Q=objective(whole[-1], goal), **FLAGS)
    if arm == ARMS[0]:
        result['new_B_input_context'] = deepcopy(initial[2])
    else:
        require(result['final_Q'] <= result['initial_Q'], 'two-round incumbent guard')
    return result


def resources(forwards, ads):
    attempted = len(forwards)
    done = sum(x.get('completed') is True for x in forwards)
    return dict(predict_attempted=attempted, predict_completed=done,
                modules_completed={k: sum(len(f.get('modules', [])) > i and f['modules'][i].get('completed') is True for f in forwards)
                                   for i, k in enumerate(('action_encoder', 'predictor', 'pred_proj'))},
                jacobian_attempted=len(ads), jacobian_completed=sum(x.get('completed') is True for x in ads),
                requested_VJP=192*len(ads), requested_J_elements=192*60*len(ads),
                requested_AD_primal=6*len(ads), encoder=0, projector=0, SVD=0, solve=0, physics=0,
                engine_backward=None, FLOPs=None)


def contract():
    return dict(schema=NAME+'.contract', roots=[0, 1], arms=list(ARMS), slots=4,
                programs=['c01/p0', 'c02/p0'], raw_steps=30, packets=6, program_steps=15,
                original_logit_cap=C15, joint_logit_cap=C30, new_horizon_not_original_full15=True,
                joint_stronger_support_and_more_compute=True, alphas=list(ALPHAS), rounds=2,
                chart='numpy tanh(float64) -> float32 raw; no repair',
                derivative='literal float32 raw leaf 192x60 J; continuous float64 tanh chain, not cast derivative',
                per_root={a: dict(predict=n, J=j) for a, (n, j) in COUNTS.items()},
                total=dict(predict=228, J=4, VJP=768, J_elements=46080, AD_primal=24),
                wall_seconds=1800, slot_wall_seconds=450, output_cap_bytes=128*1024**2,
                wall_scope='before own verify through durable report; post-publication check, failure marker on overrun',
                report_elapsed_boundary='before final report publication; completion guarded after its fsync',
                final_reserve_bytes=2*1024**2, disk_floor_bytes=200*1024**2,
                allocated_cap_bytes=4*1024**3, reserved_cap_bytes=6*1024**3,
                new_encoder=0, new_projector=0, new_physics=0, training=False,
                CUDA_only=True, TF32=False, autocast=False, MATH=True, **FLAGS)
