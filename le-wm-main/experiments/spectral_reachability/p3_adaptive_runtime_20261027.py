"""Causal finite-response scheduling over the existing bounded search kernel.

No neural backend here. Production replay must be CudaReplay under one external
PredictionBudget; graph fixtures are logic tests only. Signals are calculated
from already attempted model transitions, never supplied labels or outcomes.
"""
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from experiments.spectral_reachability.d2p_adaptive_budget_v2 import FrozenSchedule, FrozenSignals
from experiments.spectral_reachability.d2p_kinodynamic_program_search import (
    _bank, _cost, _validated_context, search_programs,
)


@dataclass(frozen=True)
class RuntimePolicy:
    source_seal: str
    max_depth: int = 3
    base_beam: int = 4
    expansion_factor: int = 2
    max_expansions: int = 144
    chi_threshold: float = 0.5
    gain_threshold: float = 0.05
    svd_relative_cutoff: float = 1e-6

    def __post_init__(self):
        self.schedule()  # Reuse existing strict threshold/count validation.
        if type(self.max_expansions) is not int or self.max_expansions < 1:
            raise ValueError('positive expansion cap required')
        if (type(self.svd_relative_cutoff) not in (int, float)
                or not np.isfinite(self.svd_relative_cutoff)
                or not 0 < self.svd_relative_cutoff < 1):
            raise ValueError('finite relative SVD cutoff in (0,1) required')

    def schedule(self):
        return FrozenSchedule(self.source_seal, (1, self.max_depth), self.base_beam,
                              self.expansion_factor, self.chi_threshold, self.gain_threshold)


def response_signal(parent, children, goal, projection, policy):
    """Finite action-response span, not learned readout/Jacobian or theorem.

    Uncentered child-minus-parent displacements include autonomous drift; this
    is local model-response geometry, NOT identified action controllability.
    horizon_gain is best signed first-order goal progress / squared residual.
    No cost-improvement/stagnation history enters scheduling.
    """
    x = np.asarray(parent['latent_history'][-1], dtype=float)
    g, q = np.asarray(goal, dtype=float), np.asarray(projection, dtype=float)
    ys = np.asarray([c['latent_history'][-1] for c in children], dtype=float)
    if (g.shape != x.shape or q.shape != (len(x), len(x)) or ys.ndim != 2
            or ys.shape[1:] != x.shape or not len(ys)
            or not all(np.isfinite(v).all() for v in (x, g, q, ys))):
        raise ValueError('finite aligned goal, projection and response children required')
    residual = (g - x) @ q.T
    response = (ys - x) @ q.T
    if not np.isfinite(residual).all() or not np.isfinite(response).all():
        raise ValueError('nonfinite projected response')
    _, singular, vt = np.linalg.svd(response, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * policy.svd_relative_cutoff)) if singular[0] else 0
    norm2 = float(residual @ residual)
    if not np.isfinite(norm2) or not np.isfinite(response @ residual).all():
        raise ValueError('response arithmetic overflow')
    if norm2 == 0:
        chi, gain = 0., 0.
    else:
        parallel = (vt[:rank] @ residual) @ vt[:rank]
        chi = float(np.clip(np.sum((residual - parallel) ** 2) / norm2, 0, 1))
        gain = float(np.clip(np.max(response @ residual) / norm2, 0, 1))
    return FrozenSignals(policy.source_seal, chi, gain), {
        'response_rank': rank, 'singular_values': singular.tolist(),
        'unreachable_fraction': chi, 'horizon_gain': gain,
        'residual_squared_norm': norm2, 'children': len(children),
    }


def run_search(*, primitive_bank, root_context, replay, goal, projection, policy,
               mode='adaptive'):
    """Expand one layer using search_programs(max_depth=1), then causally decide.

    Both arms pay identical signal computation on their own visited edges. A
    failed/incomplete layer cannot produce a schedule signal. Preserve every
    kernel receipt, including failed attempts; best is over all nonempty nodes.
    max_expansions is NOT a prediction budget. External guard includes verify.
    """
    if not isinstance(policy, RuntimePolicy) or mode not in ('fixed', 'adaptive'):
        raise ValueError('explicit RuntimePolicy and fixed/adaptive mode required')
    root = _validated_context(root_context)
    bank = _bank(primitive_bank)
    if any(len(a) != 5 for a in bank):
        raise ValueError('this runtime requires exactly one token per primitive')
    goal, projection = np.array(goal, dtype=float, copy=True), np.array(projection, dtype=float, copy=True)
    # Validate the metric even if all replay calls fail.
    response_signal(root, [root], goal, projection, policy)

    def metric(c):
        delta = (np.asarray(c['latent_history'][-1]) - goal) @ projection.T
        return _cost(np.mean(delta * delta))

    def rank(n):
        return n['cost'], n['depth'], n['primitive_indices']

    frontier = [dict(context=root, actions=[], primitive_indices=(), depth=0)]
    best, attempts, width = None, 0, policy.base_beam
    layers, status, metric_calls, signal_seconds = [], 'completed', 0, 0.
    for depth in range(1, policy.max_depth + 1):
        children, kernels, signals = [], [], []
        expected = len(frontier) * len(bank)
        layer_start = attempts
        for parent in frontier:
            remaining = policy.max_expansions - attempts
            if remaining == 0:
                status = 'budget_exhausted'
                break
            observed = []

            def measured_metric(c):
                nonlocal metric_calls
                metric_calls += 1
                cost = metric(c)
                observed.append(deepcopy(c))
                return cost

            kernel = search_programs(primitive_bank=bank, root_context=parent['context'],
                replay=replay, target_metric=measured_metric, max_depth=1,
                beam_width=len(bank), max_attempted_expansions=remaining)
            kernels.append(dict(parent_indices=list(parent['primitive_indices']), receipt=kernel))
            attempts += kernel['attempted_expansions']
            successes = [a for a in kernel['attempts'] if a['succeeded']]
            for event, context in zip(successes, observed, strict=True):
                index = event['primitive_indices'][0]
                node = dict(context=context, cost=event['cost'], depth=depth,
                    primitive_indices=parent['primitive_indices'] + (index,),
                    actions=parent['actions'] + bank[index].tolist())
                children.append(node)
                if best is None or rank(node) < rank(best):
                    best = node
            if kernel['status'] != 'completed':
                status = kernel['status']
                break
            started = perf_counter()
            signal, details = response_signal(parent['context'], observed, goal, projection, policy)
            signal_seconds += perf_counter() - started
            signals.append((signal, details))
        complete = attempts - layer_start == expected and status == 'completed'
        decision = None
        if complete:
            # Worst unreachable fraction widens breadth; best local gain permits
            # depth continuation. Aggregate only this fully observed frontier.
            combined = FrozenSignals(policy.source_seal,
                max(s.unreachable_fraction for s, _ in signals),
                max(s.horizon_gain for s, _ in signals))
            decision = policy.schedule().choose(combined)
            if mode == 'adaptive':
                width = decision['expanded_candidates']
        children.sort(key=rank)
        frontier = children[:width]
        layers.append(dict(depth=depth, expected_expansions=expected,
            attempted_expansions=attempts-layer_start, layer_complete=complete,
            retained=len(frontier), beam_width=width, kernels=kernels,
            signals=[d for _, d in signals], decision=decision))
        if not complete or not frontier:
            break
        if mode == 'adaptive' and decision['horizon_tokens'] == 1:
            status = 'dynamics_horizon_stop'
            break
    return deepcopy(dict(schema='p3_adaptive_runtime_20261027.v1', mode=mode, status=status,
        best_plan=best, layers=layers, attempted_expansions=attempts,
        metric_calls=metric_calls, signal_seconds=signal_seconds,
        partial_coverage=True, optimality_claim=False, p1_dependency='not_passed',
        p3_admitted=False, compute_advantage=False, efficacy_claim=False))
