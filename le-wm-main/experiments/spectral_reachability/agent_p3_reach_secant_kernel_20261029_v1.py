"""Pure two-dimensional I/R central-secant math; no source authentication or NN."""
from copy import deepcopy
import math

import numpy as np

NAME = 'agent_p3_reach_secant_kernel_20261029_v1'
R0 = float(np.sqrt(5)*.25)
RADII = tuple(R0/2**k for k in range(1, 5))
EPS = float(np.finfo(np.float64).eps)


def require(ok, why):
    if not ok: raise ValueError(why)


def array(value, shape):
    def numbers(v):
        return all(numbers(x) for x in v) if type(v) is list else type(v) in (int, float)
    require(type(value) is list and numbers(value), 'literal finite numeric array')
    out = np.asarray(value, np.float64)
    require(out.shape == shape and np.isfinite(out).all(), 'exact finite array shape')
    return out


def score(value):
    require(type(value) in (int, float) and math.isfinite(value), 'literal finite I/R score')
    return float(value)


def slots():
    return [dict(index=i, radius=r, status='blocked_not_generated', reason=None,
                 params=None, actions=None, raw_float32_bits=None,
                 surrogate_delta_immediate=None, surrogate_delta_reach=None) for i, r in enumerate(RADII)]


def validate(h, directions, small):
    require(type(h) is int and h in (1, 3), 'literal H1/H3')
    require(type(directions) is dict and set(directions) == {'d0', 'd1'}, 'two original named directions')
    require(type(small) is list and len(small) == 4, 'all four small-radius sources')
    missing = False
    for i, row in enumerate(small):
        direction, sign = 'd'+str(i//2), (-1, 1)[i % 2]
        require(type(row['index']) is int and row['index'] == i and row['direction'] == direction
            and row['sign'] == sign and row['radius'] == R0, 'original small-radius index/direction/sign/radius')
        d = directions[direction]
        if d is None:
            require(row['status'] == 'blocked_degenerate_direction' and row['params'] is None
                and row['actions'] is None and row['immediate'] is None and row['reach'] is None,
                'source null is explicit, not missing or fabricated zero')
            missing = True; continue
        d = array(d, (10*h,))
        require(row['status'] == 'complete', 'complete scored small-radius source')
        expected = sign*R0*d; params = array(row['params'], (10*h,))
        require(params.tobytes() == expected.tobytes(), 'original chart params exact bits')
        raw = np.tanh(expected).astype(np.float32).reshape(5*h, 2)
        actual = array(row['actions'], (5*h, 2))
        require(np.array_equal(actual, actual.astype(np.float32).astype(np.float64))
            and actual.astype(np.float32).tobytes() == raw.tobytes(), 'original raw exact single cast')
        score(row['immediate']); score(row['reach'])
    return missing


def build(h, directions, small, dispatch, *, on_start=lambda value: None):
    """dispatch(resource, callable): original ledger or explicit RAM test ledger.

    Unknown/bad inputs raise. Declared null/rank/zero/cone degeneracy returns
    all four null slots. Numerical failures propagate without refund; the
    caller keeps the partially mutated result via on_start.
    """
    missing = validate(h, directions, small)
    require(callable(dispatch), 'explicit paid math dispatch')
    out = dict(schema=NAME+'.surrogate', status='running', reason=None, horizon=h,
        source_artifact_authenticated_here=False, exact_gradient=False,
        directions=deepcopy(directions), small_rows=deepcopy(small),
        intermediates={}, direction=None, candidates=slots())
    on_start(out)
    def null(reason):
        out.update(status='surrogate_unavailable', reason=reason)
        for row in out['candidates']: row.update(status='blocked_surrogate', reason=reason)
    if missing:
        null('source_direction_null'); return out

    def generate():
        v = out['intermediates']
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            d = np.column_stack([np.asarray(directions[k], np.float64) for k in ('d0', 'd1')])
            si = np.array([(score(small[j+1]['immediate'])-score(small[j]['immediate']))/(2*R0) for j in (0, 2)], np.float64)
            sr = np.array([(score(small[j+1]['reach'])-score(small[j]['reach']))/(2*R0) for j in (0, 2)], np.float64)
            g = d.T@d; require(np.isfinite(g).all() and np.isfinite(si).all() and np.isfinite(sr).all(), 'finite Gram/secants')
            v.update(D=d.tolist(), G=g.tolist(), sI=si.tolist(), sR=sr.tolist())
            u, singular, vt = dispatch('array_svd', lambda: np.linalg.svd(g, full_matrices=False))
            require(all(np.isfinite(x).all() for x in (u, singular, vt)), 'finite Gram decomposition')
            v.update(Gram_U=u.tolist(), Gram_singular=singular.tolist(), Gram_Vt=vt.tolist())
            if singular[0] == 0 or singular[-1] <= 64*EPS*singular[0]:
                null('gram_rank_unresolved'); return
            m = dispatch('solve', lambda: np.linalg.solve(g, np.eye(2, dtype=np.float64)))
            require(np.isfinite(m).all(), 'finite Gram inverse'); v['M'] = m.tolist()
            qi, qr = float(si@m@si), float(sr@m@sr)
            require(math.isfinite(qi) and math.isfinite(qr) and qi >= 0 and qr >= 0, 'numerical_failure_negative_or_nonfinite_dual_norm')
            ni, nr = float(np.sqrt(qi)), float(np.sqrt(qr)); v.update(normI=ni, normR=nr)
            if ni == 0 or nr == 0:
                null('secant_zero_or_unresolved'); return
            a, b = si/ni, -sr/nr; kappa = float(a@m@b)
            v.update(a=a.tolist(), b=b.tolist(), kappa=kappa)
            if not math.isfinite(kappa) or not -1 <= kappa <= 1 or 1+kappa <= 64*EPS:
                null('secant_cone_unresolved'); return
            c = m@(a+b)/(1+kappa)
            if not np.isfinite(c).all():
                null('secant_cone_unresolved'); return
            norm2 = float(c@g@c); delta_i, delta_r = float(si@c), float(sr@c)
            v.update(c_star=c.tolist(), constraint_residuals=[float(a@c)-1, float(b@c)-1],
                     sI_dot_c=delta_i, sR_dot_c=delta_r, chart_norm_squared=norm2)
            require(math.isfinite(norm2) and norm2 > 0 and delta_i > 0 and delta_r < 0,
                    'numerical_failure_common_direction_sign_or_norm')
            norm = float(np.sqrt(norm2)); direction = d@c/norm
            require(np.isfinite(direction).all(), 'finite chart direction')
            out['direction'] = direction.tolist(); v['direction_norm'] = float(np.linalg.norm(direction))
            for row in out['candidates']:
                param = row['radius']*direction; raw = np.tanh(param).astype(np.float32).reshape(5*h, 2)
                require(np.isfinite(raw).all() and (np.abs(raw) <= 1).all(), 'finite raw domain')
                coeff = row['radius']*c/norm
                row.update(status='generated_unexecuted', params=param.tolist(), actions=raw.tolist(),
                    raw_float32_bits=np.ascontiguousarray(raw, dtype='<f4').tobytes().hex(),
                    surrogate_delta_immediate=float(si@coeff), surrogate_delta_reach=float(sr@coeff))
            out.update(status='complete', reason=None)
    dispatch('program_compile', generate)
    return out
