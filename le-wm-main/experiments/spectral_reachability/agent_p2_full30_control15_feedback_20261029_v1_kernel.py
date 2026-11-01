"""Fixed one-round H3 observed-vs-imagined input intervention; NumPy only."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p2_two_program_postcontext_20261029_v1_kernel as parent

NAME = 'agent_p2_full30_control15_feedback_20261029_v1'
require, equal, digest, raw, finite, context = (parent.require, parent.equal, parent.digest,
                                              parent.raw, parent.finite, parent.context)
FLAGS = dict(parent.FLAGS, P2_admission=False, C3=False)
POLICIES = ('O', 'I', 'F')
ALPHAS = (1., .5, .25, .125)
JOB_FIELDS = {'base_index', 'root_slot', 'arm', 'policy', 'input_context', 'goal', 'parent_logits',
              'incumbent_logits', 'suffix_cap', 'prefix_raw', 'parent_raw_bits', 'task_id',
              'task_sha256', 'parent_witness_sha256'}


def validate(job):
    require(type(job) is dict and set(job) == JOB_FIELDS, 'generator exact whitelist; no future labels')
    require(type(job['base_index']) is int and job['base_index'] in range(4), 'fixed four base identities')
    require(job['root_slot'] == job['base_index']//2 and type(job['root_slot']) is int and
            job['arm'] == parent.ARMS[job['base_index']%2] and job['policy'] in ('I', 'F'), 'fixed job scope')
    context(job['input_context'])
    finite(job['goal'], (192,))
    full = parent.legal(job['parent_logits'])
    parent_raw = parent.chart(full).reshape(30, 2)
    require(parent_raw.tobytes().hex() == job['parent_raw_bits'], 'literal original raw30 chart witness')
    u = finite(job['incumbent_logits'], (30,))
    equal(u.tolist(), full[30:].tolist(), 'incumbent original float64 suffix, no repair')
    equal(job['prefix_raw'], parent_raw[:15].tolist(), 'original prefix15 bits')
    if job['arm'] == parent.ARMS[0]:
        parent.legal(full[:30], parent.C15)
        parent.legal(u, parent.C15)
        cap = parent.C15
    else:
        remaining = parent.C30**2-float(full[:30]@full[:30])
        require(np.isfinite(remaining) and remaining >= 0, 'nonnegative original remaining joint square')
        cap = float(np.sqrt(remaining))
    require(type(job['suffix_cap']) is float and job['suffix_cap'] == cap, 'exact original residual domain')
    # Original joint norm is the iterate0 authority; do not substitute a new rounded sqrt gate.
    return u, cap


def legal_new(v, job):
    v = finite(v, (30,))
    cap = job['suffix_cap']
    if job['arm'] == parent.ARMS[0]:
        require(np.linalg.norm(v) <= cap, 'new transfer suffix sphere')
    else:
        parent.legal(np.concatenate((finite(job['parent_logits'], (60,))[:30], v)))
    return v


def proposals(job, z, jac):
    u, cap = validate(job)
    j = finite(jac, (192, 30))
    g = (2/192)*(j*(1-np.tanh(u)**2)[None, :]).T@(finite(z, (192,))-finite(job['goal'], (192,)))
    n = float(np.linalg.norm(g))
    require(np.isfinite(g).all() and np.isfinite(n), 'finite gradient')
    target = u.copy() if n == 0 else -.5*cap*g/n
    d = target-u
    slope = float(g@d)
    if slope >= 0:
        d = np.zeros_like(u)
    values = [u]+[u+a*d for a in ALPHAS]
    # Zero direction reuses the certified original witness, not a newly tightened cap.
    for v in values[1:]:
        if not np.array_equal(v, u):
            legal_new(v, job)
    return dict(logits=[v.tolist() for v in values], gradient=g.tolist(), direction=d.tolist(), slope=slope)


def execute(job, backend):
    u, _ = validate(job)
    c = deepcopy(job['input_context'])
    incumbent = backend.rollout(c, parent.chart(u), 'incumbent')
    j = backend.jacobian(c, u, 'raw_J')
    equal(incumbent, backend.last_ad_contexts, 'AD primal full context parity')
    prop = proposals(job, incumbent[-1]['latent_history'][-1], j)
    es = [dict(index=0, logits=u.tolist(), contexts=incumbent,
               Q=parent.objective(incumbent[-1], job['goal']))]
    for index, v in enumerate(prop['logits'][1:], 1):
        path = backend.rollout(c, parent.chart(v), f'candidate{index}')
        es.append(dict(index=index, logits=v, contexts=path, Q=parent.objective(path[-1], job['goal'])))
    w = min(es, key=lambda e: (e['Q'], e['index']))
    backend.event(dict(kind='suffix_selection', selected=w['index'], evaluations_sha256=digest(es),
                       proposals_sha256=digest(prop), input_context_sha256=digest(c)))
    selected = backend.rollout(c, parent.chart(w['logits']), 'selected_replay')
    whole = backend.rollout(c, parent.chart(w['logits']), 'whole15')
    first = backend.rollout(c, parent.chart(w['logits'])[:1], 'segment5')
    rest = backend.rollout(first[-1], parent.chart(w['logits'])[1:], 'segment10')
    for path in (selected, whole, first+rest):
        equal(path, w['contexts'], 'independent suffix15 full-context replay')
    require(w['Q'] <= es[0]['Q'], 'modeled nonworsening on same input only')
    suffix = parent.chart(w['logits']).reshape(15, 2)
    full = np.concatenate((np.asarray(job['prefix_raw'], '<f4'), suffix))
    return dict(selected=w['index'], proposals=prop, evaluations=es, contexts=whole,
                initial_Q=es[0]['Q'], new_suffix_Q=w['Q'], suffix_logits=w['logits'],
                commands=full.tolist(), raw_float32_bits=full.tobytes().hex(),
                input_context_sha256=digest(c), segmented_contexts=first+rest, **FLAGS)


def resources(forwards, ads):
    out = parent.resources(forwards, ads)
    out.update(requested_J_elements=192*30*len(ads), requested_AD_primal=3*len(ads))
    return out


def contract():
    return dict(schema=NAME+'.contract', base_slots=4, policy_slots=12, new_jobs=8,
                policies=list(POLICIES), alphas=list(ALPHAS), rounds=1, suffix_steps=15,
                total=dict(predict=216, J=8, VJP=1536, J_elements=46080, AD_primal=24),
                per_job=dict(predict=27, J=1), encoder=0, projector=0, physics=0, SVD=0, solve=0,
                wall_seconds=1800, slot_wall_seconds=450, output_cap_bytes=128*1024**2,
                reserve_bytes=2*1024**2, floor_bytes=200*1024**2,
                allocated_cap_bytes=4*1024**3, reserved_cap_bytes=6*1024**3,
                TF32=False, autocast=False, MATH=True, CUDA_only=True, training=False,
                labels_visible_through_control=15, total_FLOPs=None, engine_backward=None,
                chart='numpy tanh(float64) -> float32 raw; original iterate0 witness retained',
                derivative='raw float32 leaf192x30 J plus continuous float64 tanh chain; not cast derivative', **FLAGS)
