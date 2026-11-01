"""Consistent fixed goal co-metric in spectral generation and action search.

Prospective implementation, not a claim of demonstrated physical efficacy.
The live compiler requires CUDA; terminal wrappers support numeric fixtures.
Original historical selectors, reports, and P1 gates are not overwritten.
"""
from __future__ import annotations

import torch

from experiments.spectral_reachability import reacher_d2p_masked_programs as original
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import reacher_d2p_masked_suffix_search as search
from experiments.spectral_reachability import reacher_d2p_proposal_native_selection as native
from experiments.spectral_reachability import bounded_response_qp as qp

VARIANT = "goal_metric_cem_then_gn4"


def weighted_terminal(terminal, metric):
    """Change only output geometry, preserving action dtype and gradients."""
    def wrapped(action):
        value = terminal(action)
        if (value.ndim not in (1,2) or value.shape[-1] != metric.rows
                or value.device != metric.device or not torch.isfinite(value).all()):
            raise ValueError("complete finite terminal output aligned with goal metric required")
        if value.ndim == 1:
            return metric.apply(value.double())
        return metric.apply(value.double().T).T
    return wrapped


def flat_controls(weighted_single, weighted_batch, initial, target, *, seed):
    """Numeric-tested adapter to the existing three-result CEM contract."""
    flat, receipt = original.fit(weighted_single,initial,target)
    with torch.no_grad():
        initial_prediction = weighted_single(flat)
    refined, _, cem_receipt = original.flat_cem_control(weighted_batch,
        flat,initial_prediction,target,seed=seed)
    return {"flat_gn":(flat,receipt),"flat_cem_gn_incumbent":(refined,cem_receipt)}


def compile_programs(model, emb, history, target, *, metric, task, mean, std, seed):
    if (emb.device.type != "cuda" or history.device != emb.device or target.device != emb.device
            or metric.device != emb.device or metric.rows != 192):
        raise ValueError("host CUDA frozen neural compiler required; CPU fallback forbidden")
    width = len(chart.active_columns(task))
    target_weighted = metric.apply(target.double())

    def terminal(state, actions, horizon):
        return lambda flat: chart.terminal(model,state,actions,flat.reshape(1,horizon,width),
            task=task,mean=mean,std=std)[0]

    prefix_terminal = terminal(emb,history,1)
    root_terminal = terminal(emb,history,3)
    zero = torch.zeros(width,device=emb.device,dtype=emb.dtype)
    root_response = torch.autograd.functional.jacobian(prefix_terminal,zero.requires_grad_(True),vectorize=True).detach()
    weighted_response = metric.apply(root_response.double())
    # Action encoder keeps its original float32 contract. Cast only the SVD
    # matrix, not the neural model or the full fixed float64 co-metric.
    prefixes = original.spectral_bank(weighted_response.to(root_response.dtype))
    with torch.no_grad():
        common_residual = target-root_terminal(torch.zeros(3*width,device=emb.device,dtype=emb.dtype))
    bank = []
    for i,prefix in enumerate(prefixes):
        after, actions, prefix_prediction = original.after_prefix(model,emb,history,
            prefix.reshape(1,1,width),task=task,mean=mean,std=std)
        suffix_terminal = terminal(after,actions,2)
        def batch_suffix(flat):
            n = len(flat)
            return chart.terminal(model,after.expand(n,-1,-1),actions.expand(n,-1,-1),
                flat.reshape(n,2,width),task=task,mean=mean,std=std)
        initial = torch.zeros(2*width,device=emb.device,dtype=emb.dtype)
        response = torch.autograd.functional.jacobian(suffix_terminal,initial.requires_grad_(True),vectorize=True).detach()
        with torch.no_grad():
            own_zero = suffix_terminal(initial.detach())
        transformed = metric.apply(response.double())
        own_gain = qp.certified_response_gain(transformed,metric.apply((target-own_zero).double()),damping=.01)
        common_gain = qp.certified_response_gain(transformed,metric.apply(common_residual.double()),damping=.01)
        weighted_single = weighted_terminal(suffix_terminal,metric)
        weighted_batch = weighted_terminal(batch_suffix,metric)
        gn, initial_receipt = original.fit(weighted_single,initial.detach(),target_weighted,iterations=4)
        cem, cem_receipt = search.search(weighted_batch,gn,target_weighted,seed=seed+100000+i)
        refined, gn_receipt = original.fit(weighted_single,cem,target_weighted,iterations=4)
        full = torch.cat((prefix,refined)).reshape(1,3,width)
        with torch.no_grad():
            split, direct = suffix_terminal(refined),root_terminal(full.ravel())
        if not torch.equal(split,direct):
            raise RuntimeError("weighted compiler split/full same-action prediction mismatch")
        prefix_cost = float(metric.apply((prefix_prediction[0]-target).double()).square().mean())
        terminal_cost = float(metric.apply((direct-target).double()).square().mean())
        bank.append({"index":i,"prefix_cost":prefix_cost,"terminal_cost":terminal_cost,
            "prefix_vector":prefix_prediction[0].tolist(),"terminal_vector":direct.tolist(),
            "own_zero_suffix_vector":own_zero.tolist(),"response":response.tolist(),
            "common_response_gain":common_gain,"own_response_gain":own_gain,
            "raw_actions":chart.expand_commands(full,task)[0].reshape(15,2).tolist(),
            "full_split_exact":True,"variants":{VARIANT:{"terminal_cost":terminal_cost}},
            "initial_gn_receipt":initial_receipt,"cem_receipt":cem_receipt,"gn_receipt":gn_receipt})
    weighted_root = weighted_terminal(root_terminal,metric)
    def batch_root(flat_batch):
        n = len(flat_batch)
        return chart.terminal(model,emb.expand(n,-1,-1),history.expand(n,-1,-1),
            flat_batch.reshape(n,3,width),task=task,mean=mean,std=std)
    flat_methods = flat_controls(weighted_root,weighted_terminal(batch_root,metric),
        torch.zeros(3*width,device=emb.device,dtype=emb.dtype),target_weighted,seed=seed)
    controls = {}
    for name,(action,receipt) in flat_methods.items():
        with torch.no_grad():
            predicted = root_terminal(action)
        controls[name] = {"raw_actions":chart.expand_commands(action.reshape(1,3,width),task)[0].reshape(15,2).tolist(),
            "terminal_vector":predicted.tolist(),"terminal_cost":float(metric.apply((predicted-target).double()).square().mean()),
            "receipt":receipt}
    return {"task":task,"bank":bank,"controls":controls,"common_residual":common_residual.tolist(),
        "decision":native.select(bank,VARIANT),"historical_common_rule_decision":original.select(bank),
        "root_response":root_response.tolist(),
        "root_weighted_singular_values":torch.linalg.svdvals(weighted_response).tolist(),
        "same_metric_in_spectrum_stage_reach_and_search":True,"trained_modules":False,
        "compute_matched":False,"P1_admission":False}
