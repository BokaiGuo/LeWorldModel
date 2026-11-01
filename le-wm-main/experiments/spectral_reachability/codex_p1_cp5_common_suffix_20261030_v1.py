"""Planner-only cp5 context and common raw suffix bridge; not a launcher.

The caller must enforce OS-level isolation, frozen CUDA, source binding and
encoder/projector accounting. This module alone authorizes no execution.
"""
import numpy as np
import torch

from . import codex_p1_three_prefix_blind_inputs_20261030_v1 as blind
from . import codex_four_arm_raw_suffix_20261030_main as optimizer


def context(row, cp5_latent, *, device):
    blind.validate_row(row)
    if not isinstance(cp5_latent, torch.Tensor) or cp5_latent.shape != (192,):
        raise ValueError("cp5 latent shape")
    if cp5_latent.device != torch.device(device) or cp5_latent.dtype != torch.float32:
        raise ValueError("cp5 latent device/dtype")
    if not bool(torch.isfinite(cp5_latent).all().item()):
        raise ValueError("cp5 latent finite")
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    past_states = tensor(row["root_latent_history"])
    history = tensor(row["root_action_history"])
    mean, std = tensor(row["action_mean"]), tensor(row["action_std"])
    prefix = np.frombuffer(bytes.fromhex(row["prefix_raw_f32_hex"]), dtype="<f4").copy().reshape(5, 2)
    token = ((tensor(prefix) - mean) / std).reshape(10)
    states = torch.cat((past_states[1:], cp5_latent.detach()[None]), dim=0)[None]
    actions = torch.stack((history[-1], token))[None]
    return states, actions, tensor(row["goal_latent"])[None], mean, std


def plan(row, cp5_latent, leaf, config):
    """Optimize two packets from actual cp5; returned actions are not replayed."""
    states, actions, goal, mean, std = context(row, cp5_latent, device=cp5_latent.device)
    initial = torch.zeros((1, 2, 5, 2), dtype=torch.float32, device=states.device)
    result = optimizer.optimize(leaf, states, actions, initial, goal, mean, std, config)
    return result
