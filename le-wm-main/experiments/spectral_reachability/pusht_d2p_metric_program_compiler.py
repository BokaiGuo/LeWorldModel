"""CUDA-only fixed-bank caller, NOT spectral prefix generation/full D2P/P1.

No environment, physical labels, outcome selection, training or file writes.
Compiler coordinates are atanh live commands, not raw or normalized actions.
"""
from __future__ import annotations

import numpy as np

from experiments.spectral_reachability import pusht_d2p_nuisance_program_scoring as scoring
from experiments.spectral_reachability.pusht_d2p_goal_nuisance_metric import compile_metric


def _require_real_input(value):
    """Validate before float conversion can erase imaginary components."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError("real numeric goal/bank values required")
    elif np.asarray(value).dtype.kind not in "fi":
        raise ValueError("real numeric goal/bank values required")


def compile_programs(model, emb, actions, goal_vectors, prefix_params, suffix_horizon,
                     *, damping=.01, beta=1.):
    """Compile all explicit prefixes with own-state zero-nominal delta ridge.

    Inputs: CUDA float32 emb [1,T,D], already encoded incoming actions
    [1,T,10], goal vectors [G,D] (G>=2), parameters [N,P,2]. Goals/bank may
    be arrays or CUDA tensors. The first literal-zero prefix is the zero ID.
    Caller must supply an outcome-independent bank; provenance is not inferred.
    Returned nonlinear forecasts are diagnostics, never replacement scores.
    """
    # Reject CPU before importing neural helpers or touching the model.
    if (getattr(getattr(emb, "device", None), "type", None) != "cuda"
            or getattr(getattr(actions, "device", None), "type", None) != "cuda"):
        raise ValueError("CUDA-only encoded context; CPU neural fallback forbidden")
    import torch
    from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1

    if (not isinstance(suffix_horizon, int) or isinstance(suffix_horizon, bool)
            or suffix_horizon < 1 or not np.isfinite(damping) or damping <= 0
            or not np.isfinite(beta) or beta < 0):
        raise ValueError("positive suffix horizon/damping and nonnegative beta required")
    if (emb.ndim != 3 or emb.shape[0] != 1 or emb.shape[1] < 2
            or actions.shape != (*emb.shape[:2], 10) or actions.device != emb.device
            or emb.dtype != torch.float32 or actions.dtype != emb.dtype
            or not torch.isfinite(emb).all() or not torch.isfinite(actions).all()
            or emb.requires_grad or actions.requires_grad):
        raise ValueError("detached finite float32 aligned RGB/action context required")
    if (any(m.training for m in model.modules())
            or any(p.requires_grad or p.device != emb.device for p in model.parameters())
            or any(b.device != emb.device for b in model.buffers())):
        raise ValueError("frozen eval model entirely on context CUDA device required")
    for value in (goal_vectors, prefix_params):
        _require_real_input(value)
        if isinstance(value, torch.Tensor) and value.device != emb.device:
            raise ValueError("tensor goal/bank must already be on context CUDA device")
    goals = torch.as_tensor(goal_vectors, dtype=emb.dtype, device=emb.device).detach().clone()
    prefixes = torch.as_tensor(prefix_params, dtype=emb.dtype, device=emb.device).detach().clone()
    if (goals.ndim != 2 or goals.shape[0] < 2 or goals.shape[1] != emb.shape[-1]
            or not torch.isfinite(goals).all() or prefixes.ndim != 3
            or prefixes.shape[0] < 1 or prefixes.shape[1] < 1 or prefixes.shape[2] != 2
            or not torch.isfinite(prefixes).all()):
        raise ValueError("finite goal set [G,D] and explicit prefix bank [N,P,2] required")
    arrays = lambda value: value.detach().cpu().numpy().astype(np.float64)
    goals_np, prefix_np = arrays(goals), arrays(prefixes)
    zeros = np.flatnonzero(np.all(prefix_np == 0, axis=(1, 2)))
    if not len(zeros):
        raise ValueError("literal zero prefix required in explicit bank")
    projection, _ = compile_metric(goals_np)
    ids = [f"prefix_{i}" for i in range(len(prefixes))]
    own, contexts = [], []
    # No inference_mode/no_grad around differentiation: only prefix context is
    # detached. The raw Jacobian must include tanh and released normalization.
    with torch.enable_grad():
        for prefix in prefixes:
            with torch.no_grad():
                after, history = p1._advance(model, emb, actions, prefix[None])
            after, history = after.detach(), history.detach()
            nominal = emb.new_zeros((suffix_horizon, 2))

            def terminal(flat):
                return p1._suffix_terminal(model, after, history,
                                           flat.reshape(1, suffix_horizon, 2))[0]

            with torch.no_grad():
                origin = terminal(nominal.reshape(-1))
            jacobian = torch.autograd.functional.jacobian(
                terminal, nominal.reshape(-1).requires_grad_(True), vectorize=True)
            if (origin.shape != goals.shape[1:] or jacobian.shape != (goals.shape[1], 2*suffix_horizon)
                    or not torch.isfinite(origin).all() or not torch.isfinite(jacobian).all()
                    or not torch.isfinite(after).all() or not torch.isfinite(history).all()):
                raise ValueError("finite own-state nominal terminal/raw Jacobian required")
            own.append({"prefix_vector": arrays(after[0, -1]),
                        "nominal": arrays(nominal), "nominal_terminal": arrays(origin),
                        "jacobian": arrays(jacobian), "context_emb": arrays(after),
                        "context_actions": arrays(history)})
            contexts.append((after, history))
    compiled = scoring.compile_candidates(
        goal_vectors=goals_np, candidate_ids=ids, prefix_params=prefix_np,
        prefix_vectors=np.stack([r["prefix_vector"] for r in own]),
        suffix_nominal_params=np.stack([r["nominal"] for r in own]),
        suffix_nominal_terminal_vectors=np.stack([r["nominal_terminal"] for r in own]),
        suffix_jacobians=np.stack([r["jacobian"] for r in own]),
        zero_candidate_id=ids[int(zeros[0])], damping=damping, beta=beta,
        metric_by_role={role: projection for role in ("stage", "reach", "terminal")})
    predictions, receipts = [], []
    with torch.no_grad():
        for row, context, measured, prefix in zip(compiled["rows"], contexts, own, prefixes, strict=True):
            suffix = torch.as_tensor(row["suffix_params"], dtype=emb.dtype, device=emb.device)
            predicted = p1._suffix_terminal(model, *context, suffix[None])[0]
            if predicted.shape != goals.shape[1:] or not torch.isfinite(predicted).all():
                raise ValueError("finite nonlinear terminal required")
            predictions.append(arrays(predicted))
            full = torch.cat((prefix, suffix), dim=0)
            receipts.append({"candidate_id": row["candidate_id"],
                **{k: v.tolist() for k, v in measured.items()},
                "prefix_params": arrays(prefix).tolist(),
                "compiled_program_params_float64": row["program_params"],
                "executed_suffix_params": arrays(suffix).tolist(),
                "executed_program_params": arrays(full).tolist(),
                "program_live_commands": arrays(torch.tanh(full)).tolist(),
                "predicted_terminal_vector": predictions[-1].tolist(),
                "metric_sha256": compiled["metric"]["sha256"]})
    forecast = scoring.score_terminal_predictions(goal_vectors=goals_np,
        terminal_vectors=np.stack(predictions), projection=projection)
    if forecast["metric_sha256"] != compiled["metric"]["sha256"]:
        raise RuntimeError("terminal metric identity mismatch")
    for receipt, cost in zip(receipts, forecast["predicted_terminal_costs"], strict=True):
        receipt["nonlinear_terminal_cost_diagnostic_only"] = cost
    return {"schema": "pusht_d2p_metric_program_compiler.v1", "status": "compiled",
        "scope": "fixed-bank caller, not spectral prefix generation/full D2P/P1",
        "selection_rule": "stage_cost + beta * own_reach_value; stable bank-order ties",
        "selected_candidate_id": compiled["selected_candidate_id"],
        "scoring": compiled, "receipts": receipts,
        "input_context": {"emb": arrays(emb).tolist(), "actions": arrays(actions).tolist()},
        "goal_vectors": goals_np.tolist(), "suffix_horizon": suffix_horizon,
        "regularization": "delta parameters at zero nominal; unchanged existing array scorer",
        "jacobian_coordinates": "raw latent terminal by atanh suffix params; unprojected",
        "nonlinear_predictions_used_for_selection": False,
        "bank_provenance": "caller supplied; outcome independence not independently certified",
        "cuda_only": True, "model_inference_executed": True,
        "physical_labels_used": False, "additional_training": False,
        "spectral_prefix_generation": False, "P1_admission": False}
