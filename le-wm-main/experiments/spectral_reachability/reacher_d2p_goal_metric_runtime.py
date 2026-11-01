"""Goal-only frozen encoder co-metric for prospective score interventions.

Neural Jacobians are strictly CUDA; score arithmetic permits CPU numeric tests.
No physical future, joint state, fitted readout, or learned parameter is input.
"""
from __future__ import annotations

import math
import json
from pathlib import Path

import torch

from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric, RIDGE
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler

MODES = ("gain_only_original_costs", "consistent_Q_costs")


def declared_encoder_source(policy):
    """Resolve checkpoint-declared code without instantiating a neural model."""
    from hydra.utils import get_class
    config = json.loads((Path(policy)/"config.json").read_text())
    declared = get_class(config["_target_"])
    return Path(declared.encode.__code__.co_filename)


def measure_goal(model, processed_goal, expected_vector, *, seed):
    """Exactly the previously frozen full-Jacobian/FD recipe, no retries."""
    if processed_goal.device.type != "cuda" or processed_goal.dtype != torch.float32:
        raise ValueError("host CUDA float32 neural goal input required; CPU fallback forbidden")
    goal = processed_goal.detach().clone().requires_grad_(True)
    forwards = 0

    def encode(image):
        nonlocal forwards
        if image.device.type != "cuda":
            raise RuntimeError("neural CPU fallback forbidden")
        forwards += 1
        return model.encode({"pixels": image})["emb"][0, -1]

    value = encode(goal)
    if value.shape != (192,) or value.detach().tolist() != expected_vector:
        raise RuntimeError("unaltered full original goal encoding required")
    jacobian = torch.stack([torch.autograd.grad(value[i], goal, retain_graph=i < 191)[0].detach().flatten()
        for i in range(192)])
    if not torch.isfinite(jacobian).all():
        raise RuntimeError("nonfinite full encoder input Jacobian")
    checks = []
    generator = torch.Generator(device="cuda").manual_seed(seed ^ 0xA61E)
    for coordinate in range(3):
        direction = torch.randn(goal.shape, device="cuda", generator=generator)
        direction = direction / direction.square().mean().sqrt()
        analytic = jacobian.double() @ direction.flatten().double()
        for step in (.001, .003):
            with torch.no_grad():
                finite = (encode(goal.detach()+step*direction).double()-encode(goal.detach()-step*direction).double())/(2*step)
            denominator = torch.maximum(torch.linalg.vector_norm(finite), torch.linalg.vector_norm(analytic)).clamp_min(1e-12)
            relative = float(torch.linalg.vector_norm(finite-analytic)/denominator)
            cosine = float((finite @ analytic)/(torch.linalg.vector_norm(finite)*torch.linalg.vector_norm(analytic)).clamp_min(1e-24))
            checks.append({"direction": coordinate, "step": step, "relative_error": relative, "cosine": cosine})
    if any(not math.isfinite(c["relative_error"]) or not math.isfinite(c["cosine"])
            or c["relative_error"] > .15 or c["cosine"] < .98 for c in checks):
        raise RuntimeError("predeclared full encoder Jacobian FD check failed: " + str(checks))
    gram = jacobian.double() @ jacobian.double().T
    metric = GoalEncoderMetric(gram)
    return metric, {"gram": gram.tolist(), "gradient_checks": checks,
        "encoder_forwards": forwards, "input_vjp_rows": 192,
        "same_metric_all_tasks": True, "physical_future_labels_read": False}


def score_bank(metric, plan, target):
    """Preserve all17 original actions/suffixes, score both fixed modes."""
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=metric.device)
    target = tensor(target)
    residual = metric.apply(tensor(plan["common_residual"]))
    if [b["index"] for b in plan["bank"]] != list(range(17)):
        raise ValueError("complete ordered original17 bank required")
    compiler.select(plan["bank"])
    banks = {mode: [] for mode in MODES}
    for b in plan["bank"]:
        gain = qp.certified_response_gain(metric.apply(tensor(b["response"])), residual, damping=RIDGE)
        for mode, bank in banks.items():
            weighted = mode == "consistent_Q_costs"
            bank.append({"index": b["index"], "common_response_gain": gain,
                "prefix_cost": float(metric.apply(tensor(b["prefix_vector"])-target).square().mean()) if weighted else b["prefix_cost"],
                "terminal_cost": float(metric.apply(tensor(b["terminal_vector"])-target).square().mean()) if weighted else b["terminal_cost"]})
    return {mode: {"bank": bank, "decision": compiler.select(bank)} for mode, bank in banks.items()}
