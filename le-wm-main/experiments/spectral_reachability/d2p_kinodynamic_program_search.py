"""Deterministic model-native beam search over explicit raw PushT programs.

No physics oracle, neural implementation, padding, clipping or latent waypoints.
The trusted replay callback must use host CUDA (e.g. CudaReplay); tests may use
non-neural graph fixtures. Attach PredictionBudget externally to ALL predict
calls: an attempted edge is NOT a prediction sample. Target metric is minimized
and must close over the frozen saved goal, never physical evaluator outcomes.
"""
from copy import deepcopy
import math

import numpy as np

from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudgetExceeded
from experiments.spectral_reachability.d2p_composition_witness_v2 import _context


_CONTEXT_KEYS = {"layout", "context_length", "latent_dim", "action_dim",
                 "latent_history", "action_history"}


def _validated_context(value, layout=None):
    if not isinstance(value, dict) or set(value) != _CONTEXT_KEYS:
        raise ValueError("only explicit model latent/action context fields allowed")
    _context(value)
    if layout is not None and any(value[k] != layout[k] for k in (
            "layout", "context_length", "latent_dim", "action_dim")):
        raise ValueError("replay changed the context layout")
    return deepcopy(value)


def _bank(value):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) == 0:
        raise ValueError("explicit nonempty outcome-independent primitive bank required")
    bank = []
    for primitive in value:
        raw = np.asarray(primitive)
        if (raw.dtype.kind not in "fi" or raw.ndim != 2 or raw.shape[1] != 2
                or raw.shape[0] == 0 or raw.shape[0] % 5
                or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1)):
            raise ValueError("each raw primitive must contain complete bounded five-step [T,2] packets")
        # Preserve values/dtype; do not pad, clip or cast away witness precision.
        bank.append(raw.copy())
    return bank


def _cost(value):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fi" or not np.isfinite(raw):
        raise ValueError("metric must return a finite real scalar")
    cost = float(raw)
    if not math.isfinite(cost):
        raise ValueError("metric overflows finite scalar")
    return cost


def search_programs(*, primitive_bank, root_context, replay, target_metric,
                    max_depth, beam_width, max_attempted_expansions):
    """Return best finite NONEMPTY visited plan and an honest coverage receipt.

    Replay gets a copy of the parent context and only the next primitive, not
    the full prefix again. Nodes preserve concatenated exact raw actions and
    their full successor context. Metric gets a separate copy. Callbacks may
    mutate their arguments but must otherwise be deterministic and side-effect
    isolated. Bank ordering is frozen by caller before opening outcomes.

    Rank by (cost, depth, lexicographic bank-index sequence); expand each sorted
    frontier in bank order. Keep best across all depths, not just the last beam.
    Ordinary callback errors mark status failed but permit remaining branches;
    PredictionBudgetExceeded stops immediately with budget_exhausted. Invalid
    inputs raise before replay. Neither status nor cost grants P1 admission.
    """
    for name, value in (("max_depth", max_depth), ("beam_width", beam_width),
                        ("max_attempted_expansions", max_attempted_expansions)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive Python integer")
    if not callable(replay) or not callable(target_metric):
        raise ValueError("trusted replay and saved-goal metric callbacks required")
    bank = _bank(primitive_bank)
    root = _validated_context(root_context)
    frontier = [{"context": root, "actions": [], "primitive_indices": (), "depth": 0}]
    best = None
    attempts = []
    layers = []
    pruned = 0
    halted = False
    status = "completed"

    def rank(node):
        return (node["cost"], node["depth"], node["primitive_indices"])

    for depth in range(1, max_depth + 1):
        candidates = []
        expected = len(frontier) * len(bank)
        layer_attempts = 0
        for parent in frontier:
            for index, primitive in enumerate(bank):
                if len(attempts) >= max_attempted_expansions:
                    status, halted = "budget_exhausted", True
                    break
                sequence = parent["primitive_indices"] + (index,)
                event = {"primitive_indices": list(sequence), "depth": depth,
                         "stage": "replay", "succeeded": False}
                # Count BEFORE callback, including exceptions and budget blocks.
                attempts.append(event)
                layer_attempts += 1
                try:
                    child_context = _validated_context(
                        replay(deepcopy(parent["context"]), primitive.copy()), root)
                    event["replay_completed"] = True
                    event["stage"] = "metric"
                    cost = _cost(target_metric(deepcopy(child_context)))
                except PredictionBudgetExceeded as exc:
                    event.update(error_type=type(exc).__name__, error=str(exc))
                    status, halted = "budget_exhausted", True
                    break
                except Exception as exc:
                    event.update(error_type=type(exc).__name__, error=str(exc))
                    status = "failed"
                    continue
                event.update(succeeded=True, cost=cost)
                node = {"context": child_context,
                        "actions": parent["actions"] + primitive.tolist(),
                        "primitive_indices": sequence, "depth": depth, "cost": cost}
                candidates.append(node)
                if best is None or rank(node) < rank(best):
                    best = node
            if halted:
                break
        candidates.sort(key=rank)
        dropped = max(0, len(candidates) - beam_width)
        pruned += dropped
        frontier = candidates[:beam_width]
        layers.append({"depth": depth, "scheduled_expansions": expected,
                       "attempted_expansions": layer_attempts,
                       "finite_children": len(candidates), "retained": len(frontier),
                       "beam_pruned": dropped,
                       "layer_fully_attempted": layer_attempts == expected})
        if halted or not frontier:
            break

    failures = [e for e in attempts if not e["succeeded"]]
    # Even after a prior ordinary error, a subsequent cap exhaustion has the
    # terminal status budget_exhausted; failures remain explicit in the receipt.
    return deepcopy({
        "schema": "d2p_kinodynamic_program_search.v1",
        "algorithm": "model_native_raw_packet_beam_search",
        "status": status, "best_plan": best,
        "max_depth": max_depth, "beam_width": beam_width,
        "max_attempted_expansions": max_attempted_expansions,
        "attempted_expansions": len(attempts),
        "failed_attempts": len(failures),
        "replay_failed_attempts": sum(e["stage"] == "replay" for e in failures),
        "metric_failed_attempts": sum(e["stage"] == "metric" for e in failures),
        "attempts": attempts, "layers": layers, "beam_pruned": pruned,
        "partial_coverage": bool(pruned or failures or halted),
        "optimality_claim": False, "official_paper_faithfulness": False,
        "prediction_budget_scope": "external_PredictionBudget_not_expansion_counts",
        "p1_admitted": False,
    })
