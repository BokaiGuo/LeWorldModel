"""cp5 RGB encoding and fresh replay bridge, not an experiment launcher.

Model module wrappers count actual encoder/projector invocations independently.
The existing durable leaf driver meters action encoder, predict and backward.
Neither interface establishes process isolation or complete dependency closure.
"""
import base64
from contextlib import contextmanager
import io

import numpy as np
from PIL import Image
import torch

from . import codex_p1_cp5_common_suffix_20261030_v1 as cp5
from . import codex_four_arm_final_replay_20261030 as replay


@contextmanager
def encode_modules(model, budget, sync):
    """Patch true module forward boundaries; fail-stop after a failed call."""
    originals = []
    failed = False
    counts = {"encoder": 0, "projector": 0}
    def wrap(axis, fn):
        def call(*args, **kwargs):
            nonlocal failed
            if failed:
                raise ValueError("encoding guard is failed")
            counts[axis] += 1
            try:
                def invoke():
                    result = fn(*args, **kwargs)
                    sync()
                    return result
                return budget.call(axis, 1, invoke)
            except BaseException:
                failed = True
                raise
        return call
    try:
        for axis in ("encoder", "projector"):
            module = getattr(model, axis)
            original = module.forward
            originals.append((module, "forward", original))
            module.forward = wrap(axis, original)
        def deny(*args, **kwargs):
            raise ValueError("non-encoding model call during cp5 encoding")
        for obj, name in ((model, "predict"), (model.predictor, "forward"),
                          (model.action_encoder, "forward"), (model.pred_proj, "forward")):
            original = getattr(obj, name)
            originals.append((obj, name, original))
            setattr(obj, name, deny)
        yield counts
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def encode_cp5(row, model, budget, transform, device):
    """Decode verified cp5 PNG and use frozen eval model on CUDA only."""
    cp5.blind.validate_row(row)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("host CUDA required")
    if (not isinstance(model, torch.nn.Module) or any(m.training for m in model.modules())
            or any(p.requires_grad or p.device != device for p in model.parameters())
            or any(b.device != device for b in model.buffers())):
        raise ValueError("frozen eval CUDA model required")
    from . import preflight_pusht_d2p_p0 as p0
    with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
        rgb = np.asarray(image).copy()
    pixels = p0._image_tensor(rgb, transform=transform, device=device)
    if pixels.shape != (1, 1, 3, 224, 224) or pixels.device != device:
        raise ValueError("CUDA image ABI")
    with torch.inference_mode(), encode_modules(model, budget, lambda: torch.cuda.synchronize(device)) as counts:
        encoded = model.encode({"pixels": pixels})
    if counts != {"encoder": 1, "projector": 1}:
        raise ValueError("unexpected encoder/projector call count")
    z = encoded["emb"]
    if (not isinstance(z, torch.Tensor) or z.shape != (1, 1, 192)
            or z.device != device or z.dtype != torch.float32 or not bool(torch.isfinite(z).all())):
        raise ValueError("CUDA latent ABI")
    return z[0, 0].detach()


def plan_and_replay(row, latent, leaf, config, policy, contract):
    """Plan once and independently replay its actual post-update 10 commands."""
    states, history, goal, mean, std = cp5.context(row, latent, device=latent.device)
    raw = cp5.plan(row, latent, leaf, config)
    result = replay.replay_and_select(leaf, states, history, raw, goal, mean, std, policy, contract)
    if result["selected_index"] != 0 or not torch.equal(result["actions"], raw):
        raise ValueError("post-update replay identity")
    return result
