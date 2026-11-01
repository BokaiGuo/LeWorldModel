"""Fixed-residual ridge response diagnostic, not a causal efficacy estimator."""
import numpy as np


def response(jacobian, residual, projection, damping=.01):
    values=[]
    for x in (jacobian,residual,projection):
        x=np.asarray(x)
        if x.dtype.kind not in 'fi' or not np.isfinite(x).all():
            raise ValueError('finite real inputs required')
        values.append(x.astype(float))
    j,r,q=values
    if (j.ndim!=2 or j.shape[1]!=2 or r.shape!=(j.shape[0],) or q.shape!=(len(r),len(r))
            or type(damping) not in (int,float) or not np.isfinite(damping) or damping<=0):
        raise ValueError('aligned two-coordinate Jacobian, fixed raw residual and Q required')
    a=q@j; b=q@r
    delta=np.linalg.lstsq(np.vstack([a,np.sqrt(damping)*np.eye(2)]),np.r_[b,[0.,0.]],rcond=None)[0]
    nominal=float(b@b/len(r))
    value=float((np.sum((b-a@delta)**2)+damping*(delta@delta))/len(r))
    return dict(nominal=nominal,regularized_value=value,action_improvement=nominal-value,
        delta_params=delta.tolist(),jacobian_frobenius=float(np.linalg.norm(a)),
        singular_values=np.linalg.svd(a,compute_uv=False).tolist(),P1_admission=False)


def central_difference(probes, epsilon):
    if type(epsilon) not in (int,float) or not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('positive fixed parameter epsilon required')
    columns=[]
    for axis in range(2):
        pair=[]
        for sign in (-1,1):
            matches=[p for p in probes if p['axis']==axis and p['sign']==sign]
            if len(matches)!=1 or matches[0]['status']!='completed':
                raise ValueError('all four complete unique probes required; no filtering failures')
            x=np.asarray(matches[0]['vector'])
            if x.ndim!=1 or x.dtype.kind not in 'fi' or not np.isfinite(x).all():
                raise ValueError('finite probe vector required')
            pair.append(x)
        columns.append((pair[1]-pair[0])/(2*float(np.float32(epsilon))))
    return np.stack(columns,axis=1)
