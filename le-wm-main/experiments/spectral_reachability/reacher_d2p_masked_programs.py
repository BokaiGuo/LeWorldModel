"""Reduced raw-torque programs with exact masked-action/history contracts.

Small exploratory compiler: task masks never enter the observation history;
the same frozen recurrence scores full actions and split action witnesses.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability.bounded_response_qp import solve_box_quadratic, certified_response_gain


def after_prefix(model, emb, history, prefix, *, task, mean, std):
    if prefix.shape[1] != 1:
        raise ValueError("one-packet raw prefix required")
    raw = chart.expand_commands(prefix, task)
    normalized = chart.normalize_raw(raw, mean, std)
    with torch.no_grad():
        value = chart.terminal_normalized(model, emb, history, normalized)
    return torch.cat((emb[:, 1:], value[:, None]), dim=1), torch.cat((history[:, 1:], normalized), dim=1), value


def fit(terminal, initial, target, *, iterations=4):
    """Fixed-budget box GN, keeping the starting witness as incumbent."""
    if initial.ndim != 1 or not torch.isfinite(initial).all() or bool((initial.abs() > 1).any()):
        raise ValueError("finite raw-action initialization within literal torque bounds required")
    current = initial.detach().clone()
    trace = []
    for step in range(iterations):
        with torch.no_grad():
            predicted = terminal(current)
        jacobian = torch.autograd.functional.jacobian(terminal, current.requires_grad_(True), vectorize=True).detach()
        response = jacobian.to(torch.float64)
        residual = (target-predicted).to(torch.float64)
        now = current.detach().to(torch.float64)
        gram = response.T @ response + .01*torch.eye(len(current), device=current.device, dtype=torch.float64)
        endpoint, certificate = solve_box_quadratic(gram, response.T@residual + gram@now)
        if not certificate["active_set_terminated"] or certificate["box_feasibility_violation"] != 0 or certificate["objective_duality_gap_upper_bound"] > 1e-8*max(1., float(residual.square().sum()), float(now@gram@now)):
            raise RuntimeError("uncertified masked-action local solve")
        before = float((predicted-target).square().mean())
        selected, best, scale = current.detach().clone(), before, 0.
        with torch.no_grad():
            for alpha in (1., .5, .25, .125, .0625, .03125):
                trial = ((1-alpha)*current.detach()+alpha*endpoint.to(current.dtype)).clamp(-1, 1)
                cost = float((terminal(trial)-target).square().mean())
                if cost < best:
                    selected, best, scale = trial, cost, alpha
        trace.append({"iteration": step+1, "cost_before": before, "cost_after": best,
            "accepted_scale": scale, "qp_certificate": certificate})
        current = selected.detach()
    return current, {"iterations": iterations, "trace": trace, "physical_labels_used": False}


def spectral_bank(response):
    if response.ndim != 2 or response.shape[1] < 4 or not torch.isfinite(response).all():
        raise ValueError("finite full response with at least four raw coordinates required")
    _, _, vh = torch.linalg.svd(response, full_matrices=False)
    rows = [torch.zeros_like(vh[0])]
    for amplitude in (.25, .75):
        for i in range(4):
            direction = vh[i] / vh[i].abs().max()
            rows.extend((amplitude*direction, -amplitude*direction))
    return torch.stack(rows).detach()


def select(rows):
    if len(rows) != 17 or [r["index"] for r in rows] != list(range(17)):
        raise ValueError("complete fixed spectral prefix bank required")
    for row in rows:
        gain = row["common_response_gain"]
        if (any(not math.isfinite(float(row[k])) or row[k] < 0 for k in ("prefix_cost", "terminal_cost"))
                or gain.get("objective_certificate_passed") is not True or gain.get("physical_labels_used") is not False
                or any(not math.isfinite(float(gain[k])) for k in ("bounded_quadratic_gain", "bounded_quadratic_gain_upper_bound"))
                or gain["bounded_quadratic_gain_upper_bound"] < gain["bounded_quadratic_gain"]):
            raise ValueError("finite model-only costs and certified gain intervals required")
    greedy = min(range(17), key=lambda i: (rows[i]["prefix_cost"], i))
    threshold = max(rows[i]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] for i in (0, greedy))
    qualified = [i for i in range(1, 17) if rows[i]["prefix_cost"] > rows[0]["prefix_cost"] + 1e-6
        and rows[i]["terminal_cost"] < rows[greedy]["terminal_cost"] - 1e-6
        and rows[i]["common_response_gain"]["bounded_quadratic_gain"] > threshold]
    native = min(qualified, key=lambda i: (rows[i]["terminal_cost"], i)) if qualified else greedy
    return {"native_index": native, "greedy_index": greedy, "qualified_indices": qualified,
        "reconfiguration_witness_found": bool(qualified), "physical_labels_used": False,
        "scope": "Fixed masked-task diagnostic selector, not calibrated P1 or efficacy admission."}


def flat_cem_control(batch_terminal, initial, initial_prediction, target, *, seed):
    """Historical zero-centered CEM plus singleton-scored GN incumbent guard."""
    if (initial.ndim!=1 or initial_prediction.shape!=target.shape or target.ndim!=1
            or not torch.isfinite(initial).all() or bool((initial.abs()>1).any())
            or not torch.isfinite(initial_prediction).all() or not torch.isfinite(target).all()):
        raise ValueError("bounded flat GN incumbent and complete finite target/prediction required")
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    center,sigma = torch.zeros_like(initial),torch.ones_like(initial)
    incumbent = initial.detach().clone()
    initial_cost = float((initial_prediction-target).square().mean())
    if not math.isfinite(initial_cost): raise ValueError("finite flat GN singleton cost required")
    cost = initial_cost
    with torch.no_grad():
        for _ in range(3):
            sample = (center+sigma*torch.randn((128,len(initial)),generator=generator,device=initial.device)).clamp(-1,1)
            sample[0],sample[1] = incumbent,center
            values = batch_terminal(sample)
            if values.shape!=(128,len(target)) or not torch.isfinite(values).all():
                raise ValueError("complete finite flat CEM terminal vectors required")
            costs = (values-target).square().mean(dim=1)
            if not torch.isfinite(costs).all(): raise ValueError("finite flat CEM proposal costs required")
            elite = torch.argsort(costs,stable=True)[:16]
            if float(costs[elite[0]]) < cost:
                incumbent,cost = sample[elite[0]].clone(),float(costs[elite[0]])
            center,sigma = sample[elite].mean(0),sample[elite].std(0).clamp_min(.05)
        values = batch_terminal(incumbent[None])
        if values.shape!=(1,len(target)) or not torch.isfinite(values).all():
            raise ValueError("complete finite flat CEM singleton vector required")
        prediction = values[0]
        singleton_cost = float((prediction-target).square().mean())
        if not math.isfinite(singleton_cost): raise ValueError("finite flat CEM singleton cost required")
        fallback = singleton_cost >= initial_cost
        if fallback: incumbent,prediction = initial.detach().clone(),initial_prediction.detach().clone()
    return incumbent,prediction,{"samples":128,"steps":3,"elite":16,"flat_gn_incumbent":True,
        "physical_labels_used":False,"singleton_incumbent_guard":True,"singleton_fallback":fallback}


def compile_programs(model, emb, history, target, *, task, mean, std, seed):
    width = len(chart.active_columns(task))
    def terminal(state, actions, horizon):
        return lambda flat: chart.terminal(model, state, actions, flat.reshape(1, horizon, width), task=task, mean=mean, std=std)[0]
    prefix_terminal = terminal(emb, history, 1)
    zero = torch.zeros(width, device=emb.device)
    root_j = torch.autograd.functional.jacobian(prefix_terminal, zero.requires_grad_(True), vectorize=True).detach()
    bank = spectral_bank(root_j)
    root_future = terminal(emb, history, 3)
    with torch.no_grad():
        common_residual = target - root_future(torch.zeros(3*width, device=emb.device))
    rows = []
    for index, prefix in enumerate(bank):
        after, actions, prefix_prediction = after_prefix(model, emb, history, prefix.reshape(1, 1, width), task=task, mean=mean, std=std)
        suffix_terminal = terminal(after, actions, 2)
        suffix_zero = torch.zeros(2*width, device=emb.device)
        response = torch.autograd.functional.jacobian(suffix_terminal, suffix_zero.requires_grad_(True), vectorize=True).detach()
        gain = certified_response_gain(response.to(torch.float64), common_residual.to(torch.float64), damping=.01)
        suffix, receipt = fit(suffix_terminal, suffix_zero.detach(), target)
        full = torch.cat((prefix, suffix)).reshape(1, 3, width)
        with torch.no_grad():
            split, direct = suffix_terminal(suffix), root_future(full.ravel())
        if not torch.equal(split, direct):
            raise RuntimeError("split predicted prefix/suffix does not equal full same-action recurrence")
        rows.append({"index": index, "raw_actions": chart.expand_commands(full, task)[0].reshape(15, 2).tolist(),
            "prefix_vector": prefix_prediction[0].tolist(), "terminal_vector": direct.tolist(), "full_split_exact": True,
            "prefix_cost": float((prefix_prediction[0]-target).square().mean()),
            "terminal_cost": float((direct-target).square().mean()), "common_response_gain": gain, "suffix_receipt": receipt})
    decision = select(rows)
    flat, receipt = fit(root_future, torch.zeros(3*width, device=emb.device), target)
    with torch.no_grad():
        prediction = root_future(flat)
    controls = {"flat_gn": {"raw_actions": chart.expand_commands(flat.reshape(1, 3, width), task)[0].reshape(15, 2).tolist(),
        "terminal_vector": prediction.tolist(), "terminal_cost": float((prediction-target).square().mean()), "receipt": receipt}}
    def batch_terminal(flat_batch):
        n = len(flat_batch)
        return chart.terminal(model,emb.expand(n,-1,-1),history.expand(n,-1,-1),flat_batch.reshape(n,3,width),task=task,mean=mean,std=std)
    incumbent,prediction,cem_receipt = flat_cem_control(batch_terminal,flat,prediction,target,seed=seed)
    controls["flat_cem_gn_incumbent"] = {"raw_actions": chart.expand_commands(incumbent.reshape(1, 3, width), task)[0].reshape(15, 2).tolist(),
        "terminal_vector": prediction.tolist(), "terminal_cost": float((prediction-target).square().mean()),
        "receipt": cem_receipt}
    return {"task": task, "bank": rows, "decision": decision, "controls": controls,
        "common_model_residual": common_residual.tolist(), "trained_modules": False, "compute_matched": False}
