"""Packet-constant H4 initializers, independent of runner and neural model."""
import numpy as np
import torch

from . import p1_h4_parfit_20261101 as h4

SEEDS = {"native": 20261030, "random": 20261031, "immediate_progress": 20261101}


def aggregate_jacobian(jac, horizon):
    """Chain rule for a shared two-component action across each five-step packet."""
    if type(horizon) is not int or horizon not in (1, 2, 3, 4):
        raise ValueError("invalid horizon")
    jac = np.asarray(jac, np.float64)
    if (jac.shape != (192, 40) or not np.isfinite(jac).all()
            or np.any(jac[:, 10 * horizon:] != 0)):
        raise ValueError("invalid or nonzero inactive Jacobian")
    return jac.reshape(192, 4, 5, 2)[:, :horizon].sum(axis=2).reshape(192, 2 * horizon)


def response_bank(jac, projection, goal, nominal, horizon):
    """Original Q-DLS/top-3 policy, restricted only at initialization."""
    jac = aggregate_jacobian(jac, horizon)
    projection = np.asarray(projection, np.float64)
    goal, nominal = np.asarray(goal, np.float64), np.asarray(nominal, np.float64)
    if (projection.shape != (192, 192) or goal.shape != (192,) or nominal.shape != (192,)
            or not all(np.isfinite(x).all() for x in (projection, goal, nominal))):
        raise ValueError("invalid Q inputs")
    matrix, residual = projection @ jac, projection @ (goal - nominal)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    directions = np.zeros((8, 4, 2), np.float64)
    if singular[0] != 0:
        damping = .01 * singular[0] ** 2
        dls = right.T @ ((singular / (singular ** 2 + damping)) * (left.T @ residual))
        directions[1, :horizon] = (dls / max(1., np.max(np.abs(dls)))).reshape(horizon, 2)
        # H1 has only two action coordinates; the unavailable third pair stays zero.
        for i in range(min(3, right.shape[0])):
            vector = right[i].copy()
            if vector[np.argmax(abs(vector))] < 0:
                vector *= -1
            vector *= .5 / max(np.max(abs(vector)), 1e-30)
            directions[2 + 2*i, :horizon] = vector.reshape(horizon, 2)
            directions[3 + 2*i, :horizon] = -vector.reshape(horizon, 2)
    return np.repeat(directions[:, :, None, :], 5, axis=2).astype(np.float32), singular


def gaussian_bank(arm, dtype):
    if arm not in SEEDS or dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("invalid Gaussian arm/dtype")
    count = 16 if arm == "immediate_progress" else 8
    generator = torch.Generator(device="cpu").manual_seed(SEEDS[arm])
    bank = (.4 * torch.randn((count, 4, 1, 2), generator=generator, dtype=dtype)).clamp(-1, 1)
    bank = bank.repeat(1, 1, 5, 1)
    if arm == "native":
        bank[0] = 0
    return bank


def feedback_mix(bank, boundary):
    value = np.asarray(bank)
    h4.dimensions(boundary)
    if (value.shape != (8, 4, 5, 2) or value.dtype != np.float32
            or not np.isfinite(value).all() or np.any(np.abs(value) > 1)
            or not np.all(value == value[:, :, :1, :])):
        raise ValueError("invalid coherent feedback bank")
    if boundary == 0:
        return value.copy()
    generator = np.random.default_rng(20261031 + boundary)
    random = np.clip(.4 * generator.standard_normal((4, 4, 1, 2)), -1, 1).astype(np.float32)
    return np.concatenate((value[[3, 5, 6, 7]], np.repeat(random, 5, axis=2)))
