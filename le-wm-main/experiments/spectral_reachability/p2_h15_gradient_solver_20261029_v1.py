"""Fixed H15 projected objective-gradient shooting, using full native Jacobians."""
import numpy as np
from experiments.spectral_reachability import p2_h15_projected_gn_20261029_v1 as domain

STEPS=(1.,4.,16.,64.)
ITERATIONS=4
raw=domain.raw
project=domain.project
RADIUS=domain.RADIUS


def solve(goal,evaluate,linearize):
    goal=np.asarray(goal,np.float64)
    if goal.shape!=(192,) or not np.isfinite(goal).all():raise ValueError('invalid goal')
    def terminal(value):
        a=np.asarray(value,np.float64)
        if a.shape!=(192,) or not np.isfinite(a).all():raise ValueError('invalid terminal')
        return a
    z=np.zeros(30,np.float64);value=terminal(evaluate(raw(z)));score=float(np.mean((value-goal)**2));history=[]
    for iteration in range(ITERATIONS):
        nominal,jac=linearize(z.copy());nominal=terminal(nominal);jac=np.asarray(jac,np.float64)
        if jac.shape!=(192,30) or not np.isfinite(jac).all() or not np.array_equal(nominal,value):
            raise ValueError('invalid current Jacobian or incumbent')
        gradient=(2./192.)*(jac.T@(nominal-goal))
        if not np.isfinite(gradient).all():raise ValueError('nonfinite objective gradient')
        candidates=[]
        for step in STEPS:
            candidate=project(z-step*gradient);out=terminal(evaluate(raw(candidate)))
            q=float(np.mean((out-goal)**2));candidates.append((candidate,out,q))
        chosen=-1
        for i,(candidate,out,q) in enumerate(candidates):
            if q<score:z,value,score,chosen=candidate,out,q,i
        history.append(dict(iteration=iteration,scores=[x[2] for x in candidates],chosen=chosen,score=score))
    return dict(logits=z.tolist(),actions=raw(z).tolist(),score=score,history=history,
                evaluation_calls=17,linearization_calls=4)
