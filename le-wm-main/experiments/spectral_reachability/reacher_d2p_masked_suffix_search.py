"""Incumbent-preserving suffix search; no task, prefix or threshold changes."""
from __future__ import annotations

import math
import torch


def search(batch_terminal, initial, target, *, seed, samples=128, steps=3, elite=16):
    if (initial.ndim != 1 or target.ndim != 1 or not torch.isfinite(initial).all()
            or bool((initial.abs() > 1).any()) or not torch.isfinite(target).all()
            or initial.device != target.device or not 2 <= elite < samples or steps < 1):
        raise ValueError("bounded incumbent, matching target/device and valid CEM budget required")
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    best, center, sigma = initial.detach().clone(), initial.detach().clone(), torch.ones_like(initial)
    with torch.no_grad():
        output = batch_terminal(best[None])
        if output.shape != (1,len(target)) or not torch.isfinite(output).all():
            raise ValueError("complete finite initial singleton terminal vector required")
        prediction = output[0]
        best_cost = float((prediction-target).square().mean())
        if not math.isfinite(best_cost):
            raise ValueError("finite initial singleton deployment cost required")
        initial_cost = best_cost
        trace = []
        for step in range(steps):
            candidates = (center+sigma*torch.randn((samples,len(initial)),generator=generator,device=initial.device)).clamp(-1,1)
            candidates[0],candidates[1] = best,center
            predictions = batch_terminal(candidates)
            if predictions.shape != (samples,len(target)) or not torch.isfinite(predictions).all():
                raise ValueError("complete finite batched terminal vectors required")
            costs = (predictions-target).square().mean(dim=1)
            selected = torch.argsort(costs,stable=True)[:elite]
            if float(costs[selected[0]]) < best_cost:
                best,best_cost = candidates[selected[0]].clone(),float(costs[selected[0]])
            center = candidates[selected].mean(dim=0)
            sigma = candidates[selected].std(dim=0).clamp_min(.05)
            trace.append({"step":step+1,"best_cost":best_cost})
        output = batch_terminal(best[None])
        if output.shape != (1,len(target)) or not torch.isfinite(output).all():
            raise ValueError("complete finite final singleton terminal vector required")
        prediction = output[0]
        # Batch kernels can round differently. Preserve the incumbent under
        # singleton deployment scoring, not just batched proposal costs.
        final_cost = float((prediction-target).square().mean())
        if not math.isfinite(final_cost):
            raise ValueError("finite singleton deployment score required")
        if final_cost >= initial_cost:
            best,best_cost = initial.detach().clone(),initial_cost
        else:
            best_cost = final_cost
    if best_cost > initial_cost or not torch.isfinite(best).all() or bool((best.abs()>1).any()):
        raise RuntimeError("CEM lost its bounded incumbent")
    return best,{"initial_cost":initial_cost,"best_cost":best_cost,"trace":trace,"seed":seed,
        "samples":samples,"steps":steps,"elite":elite,"model_terminal_sample_evaluations":samples*steps+2,
        "physical_labels_used":False,"trained_modules":False}
