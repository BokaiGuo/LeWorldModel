"""Fixed two-direction current-iterate finite-difference guard; no dispatch."""
import numpy as np
from experiments.spectral_reachability import p2_h15_gn_native_ad_20261029_v1 as ad


def directions():
    return (np.eye(30,dtype=np.float64)[0], np.ones(30,dtype=np.float64)/np.sqrt(30))


def verify(result, evaluate):
    """Pay four model evaluations; caller meters and journals evaluate calls.

    Probes are validation-only, never eligible for selection. Failure aborts the
    slot. A replay must use the exact same task, model and raw mapping as AD.
    """
    z=np.asarray(result['logits'],np.float64)
    jac=np.asarray(result['logit_jacobian'],np.float64)
    if not np.array_equal(np.asarray(result['raw'],np.float32),ad.gn.raw(z)):
        raise ValueError('AD raw mapping')
    records=[]
    for i,d in enumerate(directions()):
        plus,minus=ad.probes(z,d)
        yp=np.asarray(evaluate(ad.gn.raw(plus)),np.float64)
        ym=np.asarray(evaluate(ad.gn.raw(minus)),np.float64)
        checked=ad.check_difference(plus,minus,yp,ym,jac)
        records.append(dict(direction=i,plus=plus.tolist(),minus=minus.tolist(),
                            yplus=yp.tolist(),yminus=ym.tolist(),check=checked))
    return records
