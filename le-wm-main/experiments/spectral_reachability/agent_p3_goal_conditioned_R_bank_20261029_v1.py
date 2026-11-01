"""Result-blind goal-projected/tangent R8 RAM kernel. No model or producer imports."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

NAME = 'agent_p3_goal_conditioned_R_bank_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
LAMBDA = .002
RADII = (float(np.sqrt(5)*.25), float(np.sqrt(5)*.75))
EPS = float(np.finfo(np.float64).eps)
COST = dict(program_compile_calls=1, array_SVD_calls=1, solve_calls=1)
AXES = ('predict_API', 'predict_samples', 'predict_context_tokens', 'action_encoder_calls',
        'action_encoder_samples', 'action_encoder_context_tokens', 'predictor_calls', 'predictor_samples',
        'predictor_context_tokens', 'pred_proj_calls', 'pred_proj_rows', 'encoder_calls', 'encoder_samples',
        'projector_calls', 'projector_samples', 'jacobian_API', 'VJP_directions', 'grad_API',
        'raw_backward_callbacks', 'model_SVD_calls', 'array_SVD_calls', 'solve_calls',
        'signal_compile_calls', 'program_compile_calls', 'replay_calls', 'metric_calls', 'schedule_decisions')
BINDINGS = ('root_sha256', 'task_id', 'task_sha256', 'task_contract_sha256', 'blind_sha256',
            'model_sha256', 'goal_sha256', 'projection_sha256', 'normalizer_sha256',
            'context_sha256', 'action_domain_sha256')
FLAGS = dict(P1_admission=False, P3_admission=False, C3=False, freshness=False, physical_release=False,
             original_R_equivalence=False, bank_authenticated=False, postcontext_improvement_guaranteed=False)
CAP = 16*1024**2


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    # Same compact/no-LF task and signal digest ABI as original P3 core.
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024**2), b''):
            h.update(data)
    return h.hexdigest()


def keys(value, names):
    require(type(value) is dict and set(value) == set(names), 'exact outcome-free fields: '+','.join(names))


def hash_value(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'literal SHA256')


def array(value, shape, *, f32=False):
    def numeric(v):
        return all(numeric(x) for x in v) if type(v) is list else type(v) in (int, float)
    require(type(value) is list and numeric(value), 'literal numeric list, no bool')
    a = np.asarray(value, np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite exact array shape')
    if f32:
        require(np.array_equal(a, a.astype(np.float32).astype(np.float64)), 'literal f32 source, no repair')
    return a


def domain():
    return dict(raw_bounds=[-1, 1], packet_steps=5, action_dim=2,
                normalization='original saved float32 subtract/divide; no /100 or clipping')


def contract():
    return dict(schema=NAME+'.contract', horizons=[1, 3], candidates=8, lambda_value=LAMBDA,
        radii=list(RADII), order='radius,d0/d1,sign(-1,+1)', Q_semantics='projection factor; identity192 only',
        SVD='float64(Q@raw_J), full_matrices=False; all 10h right directions, including zero singular values',
        solve='one np.linalg.solve(diag(s*s+.002), s*(U.T@r))',
        directions='p=Vt.T@w; d0=p/normp; t=p-b*((b@p)/(b@b)); d1=t/normt',
        residual='r=Q@(goal-nominal); b=(Q@raw_J).T@r',
        degeneracy='b/p zero => both null; normt<=64*eps64*normp => d1 null; blocked slots, no replacement',
        map='float64 tanh -> once float32 cast -> time-major(5h,2); no rescale/clip/projection',
        tangent_scope='raw-zero first order only; b_dot_param and b_dot_cast32_action never eligibility filters',
        separate_zero_evaluation_paid=False, no_dedup=True, expected_bank_resources=COST.copy(),
        source_artifact_authenticated_here=False, neural_calls=0, physics_calls=0, FLOPs=None, **FLAGS)


def validate_inputs(task, signal, provenance):
    keys(task, ('binding', 'root_context', 'goal', 'projection', 'action_mean', 'action_std'))
    keys(task['binding'], BINDINGS)
    for v in task['binding'].values():
        hash_value(v)
    c = task['root_context']
    keys(c, ('layout', 'context_length', 'latent_dim', 'action_dim', 'latent_history', 'action_history'))
    require(c['layout'] == 'states_with_between_state_actions_v1' and all(type(c[k]) is int and c[k] == n
            for k, n in (('context_length', 3), ('latent_dim', 192), ('action_dim', 10))), 'native full history layout')
    array(c['latent_history'], (3, 192), f32=True)
    array(c['action_history'], (2, 10), f32=True)
    goal = array(task['goal'], (192,))
    q = array(task['projection'], (192, 192))
    require(np.array_equal(q, np.eye(192)), 'original identity192 projection factor only, not arbitrary metric square root')
    array(task['action_mean'], (2,), f32=True)
    require((array(task['action_std'], (2,), f32=True) > 0).all(), 'positive frozen normalizer')
    b = task['binding']
    require(b['context_sha256'] == digest(c) and b['goal_sha256'] == digest(task['goal']) and
            b['projection_sha256'] == digest(task['projection']) and
            b['normalizer_sha256'] == digest([task['action_mean'], task['action_std']]) and
            b['action_domain_sha256'] == digest(domain()), 'original task/goal/Q/normalizer/domain/context bindings')
    keys(signal, ('binding_sha256', 'parent_context_sha256', 'horizon', 'nominal', 'jacobian', 'producer_sha256'))
    require(type(signal['horizon']) is int and signal['horizon'] in (1, 3), 'H1/H3 literal integer')
    hash_value(signal['producer_sha256'])
    require(signal['binding_sha256'] == digest(b) and signal['parent_context_sha256'] == digest(c),
            'same local task and complete parent context; no root-J relabel')
    h = signal['horizon']
    z0 = array(signal['nominal'], (192,), f32=True)
    j = array(signal['jacobian'], (192, 10*h), f32=True)
    keys(provenance, ('kind', 'artifact_path', 'artifact_sha256', 'call_id', 'origin_index', 'nominal_index',
                      'jacobian_index', 'task_sha256', 'signal_sha256', 'paid_source_resources'))
    require(provenance['kind'] in ('RAM_FIXTURE', 'SAVED_REFERENCE') and type(provenance['artifact_path']) is str and
            Path(provenance['artifact_path']).is_absolute(), 'explicit source pointer, not automatic HOST certification')
    hash_value(provenance['artifact_sha256'])
    for k in ('call_id', 'origin_index', 'nominal_index', 'jacobian_index'):
        require(type(provenance[k]) is int and provenance[k] >= 0, 'literal source index')
    require(provenance['task_sha256'] == digest(task) and provenance['signal_sha256'] == digest(signal),
            'source pointer bound to exact supplied task and raw signal')
    resources = provenance['paid_source_resources']
    if resources is not None:
        keys(resources, AXES)
        require(all(type(v) is int and v >= 0 for v in resources.values()), 'known source vector or null, no bool')
    return q, goal, z0, j, h


class Ledger:
    """Actual local dispatches, nested failures retained; no durability claim."""
    def __init__(self, caps, clock, start):
        keys(caps, COST)
        require(all(type(v) is int and v >= 0 for v in caps.values()), 'literal bank caps')
        self.caps, self.clock, self.start = deepcopy(caps), clock, start
        self.attempted = dict.fromkeys(COST, 0)
        self.returned = dict.fromkeys(COST, 0)
        self.entries, self.events, self.stack = [], [], []
        self.failed = False

    def now(self):
        value = self.clock()-self.start
        require(type(value) is int and value >= 0, 'nonnegative local clock')
        return value

    def emit(self, kind, entry):
        self.events.append(dict(sequence=len(self.events), kind=kind, entry=deepcopy(entry)))

    def call(self, resource, fn):
        require(not self.failed and resource in COST and self.attempted[resource] < self.caps[resource], 'pre-dispatch bank budget')
        e = dict(index=len(self.entries), parent=self.stack[-1] if self.stack else None, resource=resource,
                 status='attempted', started_ns=self.now(), ended_ns=None, error=None)
        self.entries.append(e)
        self.stack.append(e['index'])
        self.attempted[resource] += 1
        self.emit('attempt', e)
        try:
            value = fn()
            e.update(status='returned', ended_ns=self.now())
            self.returned[resource] += 1
            self.emit('return', e)
            return value
        except BaseException as exc:
            e.update(status='failed', ended_ns=self.now(), error=dict(type=type(exc).__name__, message=str(exc)))
            self.failed = True
            self.emit('failure', e)
            raise
        finally:
            require(self.stack.pop() == e['index'], 'balanced nested math dispatch')

    def receipt(self):
        inclusive = dict.fromkeys(COST, 0)
        for e in self.entries:
            inclusive[e['resource']] += e['ended_ns']-e['started_ns']
        return deepcopy(dict(caps=self.caps, attempted=self.attempted, returned=self.returned,
                             entries=self.entries, events=self.events, inclusive_ns=inclusive, failed=self.failed))


def slots():
    return [dict(index=i, radius=radius, direction=direction, sign=sign, status='blocked_not_generated',
                 reason='not_dispatched', params=None, actions=None, raw_float32_bits=None,
                 param_norm=None, radius_roundoff_delta=None, b_dot_param=None, b_dot_cast32_action=None)
            for i, (radius, direction, sign) in enumerate((r, d, s) for r in RADII for d in ('d0', 'd1') for s in (-1, 1))]


def direction_data(p, b):
    require(p.dtype == b.dtype == np.float64 and p.ndim == b.ndim == 1 and p.shape == b.shape and
            np.isfinite(p).all() and np.isfinite(b).all(), 'finite float64 direction inputs')
    pnorm, bnorm = float(np.linalg.norm(p)), float(np.linalg.norm(b))
    require(np.isfinite([pnorm, bnorm]).all(), 'finite projection norms')
    data = dict(pnorm=pnorm, bnorm=bnorm, t=None, tnorm=None, b_dot_t=None,
                tangent_cutoff=64*EPS*pnorm, projection_coefficient=None)
    directions, reasons = dict(d0=None, d1=None), dict(d0=None, d1=None)
    if (b == 0).all() or (p == 0).all():
        reason = 'b_zero' if (b == 0).all() else 'p_zero'
        return directions, dict(d0=reason, d1=reason), data
    require(pnorm > 0 and bnorm > 0, 'nonzero direction must have representable positive norm')
    coefficient = float((b@p)/(b@b))
    t = p-b*coefficient
    tnorm = float(np.linalg.norm(t))
    require(np.isfinite(t).all() and np.isfinite([coefficient, tnorm]).all(), 'finite tangent diagnostic')
    data.update(t=t.tolist(), tnorm=tnorm, b_dot_t=float(b@t), projection_coefficient=coefficient)
    directions['d0'] = (p/pnorm).tolist()
    if tnorm <= data['tangent_cutoff']:
        reasons['d1'] = 'tangent_at_or_below_64eps_pnorm'
    else:
        directions['d1'] = (t/tnorm).tolist()
    return directions, reasons, data


def build_bank(task, signal, provenance, *, caps=None, clock=time.monotonic_ns):
    """Pure RAM math. Source pointers bound, NOT read or certified as complete artifacts."""
    start = clock()
    require(type(start) is int, 'integer clock')
    q, goal, nominal, j, h = validate_inputs(task, signal, provenance)
    ledger = Ledger(COST if caps is None else caps, clock, start)
    out = dict(schema=NAME+'.bank', status='not_started', rule=contract(), horizon=h, candidates=slots(),
        task_binding=deepcopy(task['binding']), task_sha256=digest(task), signal_sha256=digest(signal),
        provenance=deepcopy(provenance), source_artifact_authenticated_here=False,
        source_artifact_bytes_read_here=False, inherited_source_resources=deepcopy(provenance['paid_source_resources']),
        intermediates={}, directions=dict(d0=None, d1=None), direction_reasons=dict(d0=None, d1=None),
        zero=dict(actions=np.zeros((5*h, 2), np.float32).tolist(), evaluation_paid=False,
                  nominal_not_reused_as_postcontext_qualification=True),
        error=None, numeric_environment=dict(numpy_version=np.__version__, float64_eps=EPS,
                float64_dtype=np.dtype(np.float64).str, float32_dtype=np.dtype(np.float32).str), **FLAGS)

    def generate():
        v = out['intermediates']
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            a, r = q@j, q@(goal-nominal)
            v.update(A=a.tolist(), r=r.tolist())
            b = a.T@r
            require(np.isfinite(b).all(), 'finite projected raw-zero gradient')
            v['b'] = b.tolist()
            left, singular, right = ledger.call('array_SVD_calls', lambda: np.linalg.svd(a, full_matrices=False))
            n = 10*h
            require(left.shape == (192, n) and singular.shape == (n,) and right.shape == (n, n) and
                    all(x.dtype == np.float64 and np.isfinite(x).all() for x in (left, singular, right)), 'full economy float64 SVD')
            v.update(U=left.tolist(), singular_values=singular.tolist(), Vt=right.tolist())
            matrix = np.diag(singular*singular+LAMBDA)
            rhs = singular*(left.T@r)
            v.update(solve_matrix=matrix.tolist(), solve_rhs=rhs.tolist())
            w = ledger.call('solve_calls', lambda: np.linalg.solve(matrix, rhs))
            require(w.shape == (n,) and w.dtype == np.float64 and np.isfinite(w).all(), 'finite all-basis solve')
            p = right.T@w
            v.update(w=w.tolist(), p=p.tolist())
            directions, reasons, info = direction_data(p, b)
            out['directions'], out['direction_reasons'] = directions, reasons
            v.update(info)
            for row in out['candidates']:
                direction = out['directions'][row['direction']]
                if direction is None:
                    row.update(status='blocked_degenerate_direction', reason=out['direction_reasons'][row['direction']])
                    continue
                param = row['sign']*row['radius']*np.asarray(direction, np.float64)
                cast = np.tanh(param).astype(np.float32)
                require(np.isfinite(param).all() and np.isfinite(cast).all() and (np.abs(cast) <= 1).all(), 'finite literal actuator transport')
                norm = float(np.linalg.norm(param))
                row.update(status='generated_unexecuted', reason=None, params=param.tolist(), actions=cast.reshape(5*h, 2).tolist(),
                           raw_float32_bits=np.ascontiguousarray(cast, dtype='<f4').tobytes().hex(),
                           param_norm=norm, radius_roundoff_delta=norm-row['radius'],
                           b_dot_param=float(b@param), b_dot_cast32_action=float(b@cast.astype(np.float64)))

    if any(ledger.caps[k] < v for k, v in COST.items()):
        out['status'] = 'blocked_before_bank_budget'
    else:
        try:
            ledger.call('program_compile_calls', generate)
            out['status'] = 'complete' if all(r['status'] == 'generated_unexecuted' for r in out['candidates']) else 'complete_with_blocked_directions'
        except Exception as exc:
            out.update(status='partial_failure', error=dict(type=type(exc).__name__, message=str(exc)))
            # Retain any partial math/rows, but no executable bank is released.
    out['ledger'] = ledger.receipt()
    out['new_resource_vector'] = {k: ledger.attempted.get(k, 0) for k in AXES}
    out['new_returned_vector'] = {k: ledger.returned.get(k, 0) for k in AXES}
    out['whole_kernel_ns'] = ledger.now()
    out['FLOPs'] = None
    return out


def audit_ledger(value):
    """Attempt-order entries and return-order events differ legitimately for nested failures."""
    keys(value, ('caps', 'attempted', 'returned', 'entries', 'events', 'inclusive_ns', 'failed'))
    for key in ('caps', 'attempted', 'returned', 'inclusive_ns'):
        keys(value[key], COST)
        require(all(type(v) is int and v >= 0 for v in value[key].values()), 'literal local ledger counters')
    attempted, returned, inclusive = [dict.fromkeys(COST, 0) for _ in range(3)]
    stack, opened, closed = [], {}, {}
    last_time, failed = 0, False
    for seq, event in enumerate(value['events']):
        keys(event, ('sequence', 'kind', 'entry'))
        require(type(event['sequence']) is int and event['sequence'] == seq, 'all events in chronological order')
        e = event['entry']
        keys(e, ('index', 'parent', 'resource', 'status', 'started_ns', 'ended_ns', 'error'))
        require(type(e['index']) is int and (e['parent'] is None or type(e['parent']) is int) and
                e['resource'] in COST and type(e['started_ns']) is int and e['started_ns'] >= 0,
                'literal operation identity/clock')
        if event['kind'] == 'attempt':
            require(not failed and e['index'] == len(opened) and e['parent'] == (stack[-1] if stack else None) and
                    e['status'] == 'attempted' and e['ended_ns'] is None and e['error'] is None and
                    e['started_ns'] >= last_time, 'nested attempt lifecycle')
            opened[e['index']] = e
            stack.append(e['index'])
            attempted[e['resource']] += 1
            last_time = e['started_ns']
        else:
            require(event['kind'] in ('return', 'failure') and stack and stack.pop() == e['index'], 'no unknown/unpaired return event')
            require(not failed or event['kind'] == 'failure', 'after child failure only ancestor failure unwind is legal')
            initial = opened[e['index']]
            require(all(e[k] == initial[k] for k in ('parent', 'resource', 'started_ns')) and
                    type(e['ended_ns']) is int and e['ended_ns'] >= last_time and
                    e['status'] == ('returned' if event['kind'] == 'return' else 'failed') and
                    ((e['error'] is None) if event['kind'] == 'return' else type(e['error']) is dict), 'exact returned or failed operation')
            closed[e['index']] = e
            returned[e['resource']] += int(event['kind'] == 'return')
            inclusive[e['resource']] += e['ended_ns']-e['started_ns']
            last_time = e['ended_ns']
            failed = failed or event['kind'] == 'failure'
    require(not stack and value['entries'] == [closed[i] for i in range(len(closed))] and
            value['attempted'] == attempted and value['returned'] == returned and value['inclusive_ns'] == inclusive and
            type(value['failed']) is bool and value['failed'] == any(e['status'] == 'failed' for e in closed.values()) and
            all(attempted[k] <= value['caps'][k] for k in COST), 'entire local ledger reconstruction')
    return True


def decode(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    out = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda x: require(False, 'nonfinite JSON'))
    canonical(out)
    return out


def read_pin(path, h):
    hash_value(h)
    p = Path(path).absolute()
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= CAP and sha(p) == h, 'bounded explicit input SHA')
    value = decode(p.read_bytes())
    require(sha(p) == h, 'saved input changed')
    return value


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def write_new(path, value):
    raw = canonical(value)+b'\n'
    p = Path(path).absolute()
    require(len(raw) <= CAP and p.parent.is_dir() and not p.is_symlink() and
            shutil.disk_usage(p.parent).free >= 200*1024**2+len(raw), 'bounded exclusive output/floor')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete output')
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(p), sha256=sha(p), bytes=len(raw))


def build_file(path, input_sha):
    value = read_pin(path, input_sha)
    keys(value, ('schema', 'task', 'signal', 'provenance'))
    require(value['schema'] == NAME+'.input', 'own explicit input envelope')
    sources = {str(p): sha(p) for p in own_files()}
    out = build_bank(value['task'], value['signal'], value['provenance'])
    out['input'] = dict(path=str(Path(path).absolute()), sha256=input_sha)
    out['sources'] = sources
    require(sha(path) == input_sha and all(sha(p) == h for p, h in sources.items()), 'final input/source unchanged')
    return out


def verify_file(path, expected):
    old = read_pin(path, expected)
    require(old['schema'] == NAME+'.bank' and old['status'] in ('complete', 'complete_with_blocked_directions'), 'complete numeric result, not scientific authentication')
    audit_ledger(old['ledger'])
    require(type(old['whole_kernel_ns']) is int and old['whole_kernel_ns'] >= 0 and
            all(e['ended_ns'] <= old['whole_kernel_ns'] for e in old['ledger']['entries']), 'whole numerical clock bounds all entries')
    for p in own_files():
        require(old['sources'][str(p)] == sha(p), 'frozen numerical source/tests/protocol')
    rebuilt = build_file(old['input']['path'], old['input']['sha256'])
    audit_ledger(rebuilt['ledger'])
    keys(old, rebuilt)
    for k in old:
        if k not in ('ledger', 'whole_kernel_ns'):
            require(canonical(old[k]) == canonical(rebuilt[k]), 'numerical payload reconstruction: '+k)
    require(old['ledger']['attempted'] == rebuilt['ledger']['attempted'] == COST and
            old['ledger']['returned'] == rebuilt['ledger']['returned'] == COST and
            old['ledger']['caps'] == COST, 'fixed complete actual operations')
    require(sha(path) == expected and sha(old['input']['path']) == old['input']['sha256'], 'final saved seals')
    return dict(verified_numeric_only=True, source_artifact_authenticated_here=False,
                verification_additional_CPU_resources=rebuilt['ledger']['attempted'], **FLAGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('contract')
    q = sub.add_parser('build')
    q.add_argument('--input', required=True)
    q.add_argument('--input-sha', required=True)
    q.add_argument('--output', required=True)
    q = sub.add_parser('verify')
    q.add_argument('--report', required=True)
    q.add_argument('--report-sha', required=True)
    a = parser.parse_args(argv)
    if a.cmd == 'contract':
        print(canonical(contract()).decode())
    elif a.cmd == 'build':
        out = build_file(a.input, a.input_sha)
        print(json.dumps(dict(**write_new(a.output, out), status=out['status'])))
    else:
        print(canonical(verify_file(a.report, a.report_sha)).decode())


if __name__ == '__main__':
    main()
