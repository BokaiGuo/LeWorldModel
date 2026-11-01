"""Projected common-suffix optimizer in the physical raw-action chart.

Model calls are dispatched through the existing leaf meter. This module does
not generate prefixes, select methods, execute physics, or establish fairness.
"""
import math

import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leaf_module


def optimize(leaf, states, normalized_history, raw_initial, goal, mean, std, config):
    """Optimize [batch, packets, 5, 2] raw commands, each constrained to [-1,1]."""
    spec.require(type(leaf) is leaf_module.LeafAdapter, "leaf adapter required")
    try:
        spec.require(type(config) is dict and set(config) == {"steps", "learning_rate", "history_size"},
                     "raw suffix config")
        spec.require(spec.digest(config) == leaf.config_sha, "suffix config hash mismatch")
        spec.require(type(config["steps"]) is int and config["steps"] > 0 and
                     type(config["history_size"]) is int and config["history_size"] > 0 and
                     type(config["learning_rate"]) in (int, float) and
                     math.isfinite(config["learning_rate"]) and config["learning_rate"] > 0,
                     "invalid optimizer configuration")
        tensors = (states, normalized_history, raw_initial, goal, mean, std)
        for value in tensors:
            leaf._tensor(value)
            spec.require(value.is_floating_point() and leaf_module._finite(value), "finite float tensor")
        spec.require(all(x.device == states.device and x.dtype == states.dtype for x in tensors),
                     "device or dtype mismatch")
        spec.require(states.ndim == 3 and raw_initial.ndim == 4 and goal.ndim == 2
                     and normalized_history.ndim == 3, "tensor ranks")
        batch, history, dim = states.shape
        spec.require(history == config["history_size"] and goal.shape == (batch, dim)
                     and normalized_history.shape == (batch, history - 1, 10)
                     and raw_initial.shape[0] == batch and raw_initial.shape[1] > 0
                     and raw_initial.shape[2:] == (5, 2)
                     and mean.shape == std.shape == (2,) and bool((std > 0).all().item())
                     and bool((raw_initial.abs() <= 1).all().item()), "action/history dimensions or bounds")
        initial = raw_initial.detach().clone()
        state0, history0, target = states.detach(), normalized_history.detach(), goal.detach()
        for _ in range(config["steps"]):
            raw = initial.detach().requires_grad_(True)
            z, past = state0, history0
            for step in range(raw.shape[1]):
                normalized = ((raw[:, step] - mean) / std).reshape(batch, 1, 10)
                context = torch.cat((past, normalized), dim=1)
                predicted = leaf.predict(z, leaf.action_encode(context, phase="common_suffix"),
                                         phase="common_suffix")
                spec.require(isinstance(predicted, torch.Tensor) and predicted.shape == z.shape
                             and leaf_module._finite(predicted), "invalid prediction")
                z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -history:]
                past = context[:, -(history - 1):] if history > 1 else context[:, :0]
            gradient = leaf.gradient((z[:, -1] - target).square().mean(), raw, phase="common_suffix")
            spec.require(gradient.shape == raw.shape and leaf_module._finite(gradient), "invalid gradient")
            initial = (raw - config["learning_rate"] * gradient).clamp(-1, 1).detach()
            spec.require(leaf_module._finite(initial), "invalid update")
        return initial
    except BaseException:
        leaf._abort()
        raise
