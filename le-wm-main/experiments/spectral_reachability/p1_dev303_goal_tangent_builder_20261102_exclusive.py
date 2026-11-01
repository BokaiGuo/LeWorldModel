"""RGB-only goal tangent metric builder; no simulator or outcome inputs."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from . import p1_rgb_tangent_metric_20261101_main as tangent
from . import p1_rgb_visibility_20261101_main as visibility


def _pixel_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _encode_rgb(model, rgb, device, transform, preflight):
    pixels = preflight._image_tensor(rgb, transform=transform, device=device)
    if pixels.device.type != "cuda" or pixels.device != device:
        raise ValueError("image tensor must be on requested CUDA device")
    output = model.encode({"pixels": pixels})["emb"]
    if (not isinstance(output, torch.Tensor) or output.device != device
            or output.dtype != torch.float32 or tuple(output.shape) != (1, 1, 192)):
        raise ValueError("expected CUDA float32 projected embedding [1,1,192]")
    latent = output[0, 0].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(latent).all():
        raise ValueError("nonfinite embedding")
    return latent


def build_goal_tangent_metric(model, goal_rgb, device):
    """Return (float64 192x192 sqrt(Q), JSON-compatible audit metadata).

    Encodes the original once, then each AXES perturbation at +0.5 and -0.5
    separately. The central difference denominator is 1.0, matching the
    sealed saved-metric construction. No training, state or outcome is used.
    """
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("an explicit CUDA device such as cuda:0 is required")
    if getattr(model, "training", None) is not False:
        raise ValueError("model must be in evaluation mode")
    if not isinstance(goal_rgb, np.ndarray) or goal_rgb.dtype != np.uint8 or goal_rgb.shape != (224, 224, 3):
        raise ValueError("goal must be uint8 RGB 224x224")
    goal = np.ascontiguousarray(goal_rgb).copy()
    screen = visibility.inspect(goal)
    if not screen["eligible"]:
        raise ValueError("goal boundary visibility")

    # Import only after cheap rejection checks; no model is loaded here.
    from . import preflight_pusht_d2p_p0 as preflight

    transform = preflight.image_transform()
    records = []
    vectors = {}
    with torch.inference_mode():
        goal_latent = _encode_rgb(model, goal, device, transform, preflight)
        for axis in tangent.AXES:
            for amount in (0.5, -0.5):
                pixels = tangent.perturb(goal, axis, amount)
                latent = _encode_rgb(model, pixels, device, transform, preflight)
                vectors[(axis, amount)] = latent
                records.append({"axis": axis, "amount": amount,
                                "pixel_sha256": _pixel_sha(pixels)})
    jac = np.stack([vectors[(axis, 0.5)] - vectors[(axis, -0.5)]
                    for axis in tangent.AXES], axis=1)
    sqrt_q, diagnostics = tangent.construct(jac)
    return sqrt_q, {
        "goal_pixel_sha256": _pixel_sha(goal),
        "pixel_hash_definition": "SHA256 of contiguous uint8 HWC RGB bytes",
        "visibility": screen,
        "perturbations": records,
        "encoder_calls": 1 + len(records),
        "goal_latent": goal_latent.tolist(),
        "singular_values": diagnostics["singular_values"],
        "metric_diagnostics": diagnostics,
        "difference": "latent(+0.5) - latent(-0.5)",
        "neural_device": str(device),
        "training": False,
        "simulator_state_or_outcome_used": False,
        "P1_admission": False,
    }


def build(model, goal_rgb, device):
    """Runner-facing alias for the goal tangent metric builder."""
    return build_goal_tangent_metric(model, goal_rgb, device)
