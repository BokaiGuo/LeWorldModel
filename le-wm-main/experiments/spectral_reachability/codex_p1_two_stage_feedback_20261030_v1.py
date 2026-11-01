"""Causal two-stage planner bridge; not a physical launcher or source attestation.

The physical caller must independently authenticate the five receipts and the
fresh final image against its durable journal. In-process input whitelisting is
not OS isolation. No GPU or physics is started by importing this module.
"""
import hashlib
import io
import numpy as np
from PIL import Image
import torch

from . import codex_p1_cp5_common_suffix_20261030_v1 as common
from . import codex_p1_cp5_cuda_replay_20261030_v1 as encoding
from . import codex_four_arm_final_replay_20261030 as replay
from . import codex_four_arm_raw_suffix_20261030_main as optimizer

CONFIG = {"steps": 20, "learning_rate": 0.05, "history_size": 3}
RECEIPT_KEYS = {"row", "step", "action_f32_hex", "terminated", "truncated",
                "pixel_sha256"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def packet_bytes(packet):
    need(isinstance(packet, np.ndarray) and packet.dtype == np.dtype("<f4")
         and packet.shape == (5, 2) and np.isfinite(packet).all()
         and (np.abs(packet) <= 1).all(), "raw packet ABI")
    return packet.tobytes(order="C")


def rgb_pixels(png):
    need(type(png) is bytes, "PNG bytes")
    with Image.open(io.BytesIO(png)) as image:
        need(image.format == "PNG" and image.mode == "RGB" and image.size == (224, 224), "RGB ABI")
        return np.asarray(image).copy()


def verify_feedback(row_index, planned_packet, receipts, fresh_png):
    """Check declared transitions; caller must authenticate against live journal.

    Stop anywhere forbids phase two. The last image is compared by decoded
    pixels, not by an exception name or an unverified predicted latent.
    """
    need(type(row_index) is int and 0 <= row_index < 24, "row identity")
    raw = packet_bytes(planned_packet)
    need(type(receipts) is list and len(receipts) == 5, "five completed transitions required")
    for i, item in enumerate(receipts):
        need(type(item) is dict and set(item) == RECEIPT_KEYS, "receipt fields")
        need(type(item["row"]) is int and item["row"] == row_index
             and type(item["step"]) is int and item["step"] == i + 1, "receipt identity/order")
        need(item["action_f32_hex"] == raw[i * 8:(i + 1) * 8].hex(), "executed action mismatch")
        need(type(item["terminated"]) is bool and type(item["truncated"]) is bool
             and not item["terminated"] and not item["truncated"], "native stop forbids replan")
        need(type(item["pixel_sha256"]) is str and len(item["pixel_sha256"]) == 64
             and all(c in "0123456789abcdef" for c in item["pixel_sha256"]), "pixel digest")
    pixels = rgb_pixels(fresh_png)
    need(hashlib.sha256(pixels.tobytes()).hexdigest() == receipts[-1]["pixel_sha256"], "fresh RGB mismatch")
    return pixels


def second_context(row, cp5_latent, cp10_latent, executed_packet):
    """Advance with actual cp10 latent and actually executed raw packet only."""
    states, actions, goal, mean, std = common.context(row, cp5_latent, device=cp5_latent.device)
    need(isinstance(cp10_latent, torch.Tensor) and cp10_latent.shape == (192,)
         and cp10_latent.dtype == torch.float32 and cp10_latent.device == states.device
         and bool(torch.isfinite(cp10_latent).all().item()), "cp10 latent ABI")
    packet_bytes(executed_packet)
    packet = torch.as_tensor(executed_packet.copy(), dtype=states.dtype, device=states.device)
    token = ((packet - mean) / std).reshape(1, 1, 10)
    return (torch.cat((states[:, 1:], cp10_latent.detach().reshape(1, 1, 192)), dim=1),
            torch.cat((actions[:, 1:], token), dim=1), goal, mean, std)


def encode_fresh(pixels, model, budget, transform, device):
    """Encode caller-authenticated RGB on frozen CUDA with actual module metering."""
    device = torch.device(device)
    need(device.type == "cuda" and torch.cuda.is_available(), "host CUDA required")
    need(isinstance(pixels, np.ndarray) and pixels.shape == (224, 224, 3)
         and pixels.dtype == np.uint8, "RGB pixels ABI")
    need(isinstance(model, torch.nn.Module) and all(not m.training for m in model.modules())
         and all(not p.requires_grad and p.device == device for p in model.parameters())
         and all(b.device == device for b in model.buffers()), "frozen eval CUDA model")
    from . import preflight_pusht_d2p_p0 as p0
    x = p0._image_tensor(pixels, transform=transform, device=device)
    need(x.shape == (1, 1, 3, 224, 224) and x.device == device, "image tensor ABI")
    with torch.inference_mode(), encoding.encode_modules(
            model, budget, lambda: torch.cuda.synchronize(device)) as counts:
        encoded = model.encode({"pixels": x})
    need(counts == {"encoder": 1, "projector": 1}, "encoder counts")
    z = encoded["emb"]
    need(isinstance(z, torch.Tensor) and z.shape == (1, 1, 192)
         and z.dtype == torch.float32 and z.device == device
         and bool(torch.isfinite(z).all().item()), "latent ABI")
    return z[0, 0].detach()


def plan_context(context, leaf, policy, contract):
    """Same fixed optimizer and post-update full-horizon replay for both stages."""
    states, history, goal, mean, std = context
    initial = torch.zeros((1, 2, 5, 2), dtype=states.dtype, device=states.device)
    raw = optimizer.optimize(leaf, states, history, initial, goal, mean, std, CONFIG.copy())
    result = replay.replay_and_select(leaf, states, history, raw, goal, mean, std, policy, contract)
    need(result["selected_index"] == 0 and torch.equal(result["actions"], raw), "replay identity")
    return raw.detach().clone(), result
