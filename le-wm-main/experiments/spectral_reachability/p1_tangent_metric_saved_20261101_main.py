"""Independent numerical replay of the goal-only geometric metric receipt."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_rgb_tangent_metric_20261101_main as tangent
from . import p1_goal_cometric_cuda_20261101_main as base


def check(path):
    value=json.loads(Path(path).read_bytes())
    if (value["status"]!="measured_not_control_validated"
            or value["measurement"]["encoder_forwards"]!=21
            or value["measurement"]["input_vjp_rows"]!=0
            or value["training"] or value["physical_future_labels_read"]
            or value["model_before"]!=value["model_after"]):
        raise ValueError("measurement contract")
    vectors={(r["axis"],r["scale"],r["sign"]):np.asarray(r["latent"],float) for r in value["vectors"]}
    expected={(a,s,t) for a in tangent.AXES for s in (1.,.5) for t in (-1,1)}
    if set(vectors)!=expected or len(value["vectors"])!=20:
        raise ValueError("geometric perturbation roster")
    jac=np.stack([vectors[a,.5,1]-vectors[a,.5,-1] for a in tangent.AXES],axis=1)
    if not np.array_equal(jac,np.asarray(value["jacobian"])):
        raise ValueError("finite-difference Jacobian")
    sqrt,diagnostics=tangent.construct(jac)
    if not np.allclose(sqrt,np.asarray(value["sqrt_matrix"]),rtol=1e-10,atol=1e-10):
        raise ValueError("metric square root")
    if any(base.source.base.sha(p)!=h for p,h in value["sources"].items()):
        raise ValueError("source drift")
    return dict(status="saved_metric_verified",source_sha256=base.source.base.sha(path),
                trace=diagnostics["trace"],physical_calibration=False,P1_admission=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=check(args.input)
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)
