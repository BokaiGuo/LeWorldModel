"""Measure a fixed target-only metric on host CUDA, not train a readout."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from . import p1_goal_cometric_cuda_20261101_main as prior
from . import p1_rgb_tangent_metric_20261101_main as tangent
from . import p1_rgb_visibility_20261101_main as visibility

NAME = "p1_rgb_tangent_cuda_20261101_main"
DOC = prior.source.ROOT.parent / "docs/D2P_P1_RGB_TANGENT_PROTOCOL_20261101_MAIN.md"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base = prior.source.base
    output = Path(output)
    if not output.is_absolute() or output.parent != prior.source.ROOT/"results" or output.exists():
        raise ValueError("exclusive output")
    if base.sha(prior.TARGET) != prior.TARGET_SHA:
        raise ValueError("fixed task")
    row = json.loads(prior.TARGET.read_bytes())["rows"][0]
    pixels = np.frombuffer(base64.b64decode(row["pixels_uint8_base64"]),np.uint8).reshape(224,224,3).copy()
    screen = visibility.inspect(pixels)
    if not screen["eligible"]:
        raise ValueError("target visibility")
    images = [(axis, scale, sign, tangent.perturb(pixels,axis,scale*sign))
              for axis in tangent.AXES for scale in (1.0,.5) for sign in (-1,1)]
    sources = base.bindings()
    for p in (prior.TARGET, DOC, Path(__file__), Path(tangent.__file__), Path(visibility.__file__)):
        p = p.resolve()
        sources[str(p)] = base.sha(p)
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL),device=torch.device("cuda:0")).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda",enabled=False):
            def encode(image):
                return model.encode({"pixels":p0._image_tensor(image,transform=p0.image_transform(),
                                        device=torch.device("cuda:0"))})["emb"][0,0].cpu().numpy()
            goal = encode(pixels)
            if goal.tolist() != row["latent"]:
                raise ValueError("unaltered goal encoding")
            vectors = {(axis,scale,sign):encode(image) for axis,scale,sign,image in images}
        columns, checks = [], []
        for axis in tangent.AXES:
            full = (vectors[axis,1.,1].astype(float)-vectors[axis,1.,-1])/2
            half = vectors[axis,.5,1].astype(float)-vectors[axis,.5,-1]
            columns.append(half)
            checks.append(dict(axis=axis, relative_difference=float(np.linalg.norm(full-half)/
                max(np.linalg.norm(full),np.linalg.norm(half),1e-12)),
                cosine=float(full@half/max(np.linalg.norm(full)*np.linalg.norm(half),1e-12))))
        jacobian = np.stack(columns,axis=1)
        sqrt, diagnostics = tangent.construct(jacobian)
        torch.cuda.synchronize()
        if base.model_hash(model) != before or any(base.sha(p)!=h for p,h in sources.items()):
            raise ValueError("source/model drift")
        value = dict(schema=NAME,status="measured_not_control_validated",target_sha256=prior.TARGET_SHA,
                     goal_latent=goal.tolist(),sqrt_matrix=sqrt.tolist(),jacobian=jacobian.tolist(),
                     vectors=[dict(axis=a,scale=s,sign=t,latent=v.tolist()) for (a,s,t),v in vectors.items()],
                     measurement=dict(encoder_forwards=21,input_vjp_rows=0,consistency=checks,**diagnostics),
                     visibility=screen,sources=sources,host=host,model_before=before,model_after=before,
                     physical_future_labels_read=False,training=False,P1_admission=False)
        digest = base.write(output,value)
    print(json.dumps(dict(status=value["status"],sha256=digest,consistency=checks)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--authorize-host-cuda",action="store_true",required=True)
    args=parser.parse_args()
    run(args.output)
