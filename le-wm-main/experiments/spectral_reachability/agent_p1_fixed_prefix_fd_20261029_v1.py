"""Fixed 64-group, two-scale raw/chart finite differences; callback kernel only."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p1_fixed_prefix_saved_handoff_20261029_v1 as source

F, core = source.F, source.core
require, raw = source.require, source.raw
STEPS = (2.0**-8, 2.0**-10)


def contract():
    return dict(groups=64, group_point='suffix iterate 0; saved gradient 1', coordinates=10,
        spaces=['raw','chart_logit'], scales=list(STEPS), pairs_per_group=40, pair_denominator=2560,
        max_predict_per_group=80, max_new_predict=5120, new_reverse=0,
        inherited_saved_gradient='one previously paid raw reverse per group, original native/scalar audit required',
        raw_rule='float32 +/- requested step; actual representable denominator; no clip; invalid pair null',
        chart_rule='float32 logits +/- step through original float32 chart; no projection; outside ball pair null',
        objective='float64 saved endpoint identity MSE; compare float32 AD loss derivative with reduction caveat',
        projection='boundary and proposed norms reported only; finite differences do not differentiate projection',
        outcome_selection=False, training=False, physical_release=False, P1_admission=False, freshness=False)


def pairs(q, result):
    rows = result['iterates']; g = result['gradients'][0]
    require(len(rows)==9 and len(result['gradients'])==8 and raw(g['context'])==raw(rows[0]['context']), 'fixed first saved gradient')
    x = core.finite(rows[0]['logits'],(10,)).astype(np.float32)
    a = F.chart(x).reshape(10)
    require(F.bits(a)==rows[0]['raw_bits'] and F.bits(a)==F.bits(F.chart(q['suffix_logits'])), 'exact base raw')
    grad = core.finite(g['raw_gradient'],(5,2),True).astype(np.float32).reshape(10)
    out=[]
    for space in ('raw','chart_logit'):
        for h in STEPS:
            for j in range(10):
                base = a if space=='raw' else x
                hi=base.copy(); lo=base.copy()
                hi[j]=np.float32(float(base[j])+h); lo[j]=np.float32(float(base[j])-h)
                dp=float(hi[j])-float(base[j]); dm=float(base[j])-float(lo[j]); denom=float(hi[j])-float(lo[j])
                plusnorm = float(np.linalg.norm(hi.astype(float))) if space=='chart_logit' else None
                minusnorm = float(np.linalg.norm(lo.astype(float))) if space=='chart_logit' else None
                valid=dp>0 and dm>0 and (space!='raw' or (float(lo[j])>=-1 and float(hi[j])<=1))
                valid=bool(valid and (space!='chart_logit' or (plusnorm<=F.CAP and minusnorm<=F.CAP)))
                ap=hi.reshape(5,2) if space=='raw' else F.chart(hi)
                am=lo.reshape(5,2) if space=='raw' else F.chart(lo)
                if F.bits(ap)==F.bits(am):valid=False
                chain=float(grad[j]) if space=='raw' else float(grad[j])*(1-float(a[j])**2)
                out.append(dict(space=space,requested_step=h,coordinate=j,valid=valid,
                    invalid_reason=None if valid else 'float32_collapse_or_domain_or_projection_boundary',
                    plus_step=dp,minus_step=dm,denominator=denom,plus_norm=plusnorm,minus_norm=minusnorm,
                    base_logit_norm=float(np.linalg.norm(x.astype(float))),projection_radius=F.CAP,
                    plus_commands=ap.tolist() if valid else None,minus_commands=am.tolist() if valid else None,
                    plus_raw_bits=F.bits(ap) if valid else None,minus_raw_bits=F.bits(am) if valid else None,
                    saved_AD_direction=chain))
    require(len(out)==40,'fixed 40 pairs');return out


def evaluate(q,result,backend,sink):
    """Run predeclared valid pairs, preserve invalid and no physical fallback."""
    ps=pairs(q,result); goal=q['task']['goal']; mid=q['expected_mid']
    for i,p in enumerate(ps):
        if not p['valid']:
            p.update(plus_Q=None,minus_Q=None,finite_difference=None,error_vs_saved_AD=None)
            sink(dict(kind='pair_null',pair=i,reason=p['invalid_reason']));continue
        values=[]
        for sign in ('plus','minus'):
            label=f'fd_{i:02}_{sign}'; commands=np.asarray(p[sign+'_commands'],np.float32)
            sink(dict(kind='fd_intent',pair=i,sign=sign,raw_bits=F.bits(commands)))
            c=backend.step(deepcopy(mid),commands,label); core.checked_path(q['task'],mid,commands.tolist(),[c])
            value=F.cost(goal,c); values.append(value)
            sink(dict(kind='fd_return',pair=i,sign=sign,context=deepcopy(c),Q=value))
        p['plus_Q'],p['minus_Q']=values
        p['finite_difference']=(values[0]-values[1])/p['denominator']
        p['error_vs_saved_AD']=p['finite_difference']-p['saved_AD_direction']
    return dict(index=q['index'],pairs=ps,pair_denominator=40,
                valid_pairs=sum(p['valid'] for p in ps),predict_calls=2*sum(p['valid'] for p in ps),
                native_gradient_correctness='diagnostic_only',P1_admission=False,physical_release=False)
