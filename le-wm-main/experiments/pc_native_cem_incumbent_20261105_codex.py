"""Native CEM with an optional best-scored candidate; no changed sampling.

The best candidate is selected within the current solve only. In particular,
this does not reuse a score after an environment observation has changed.
The final native distribution mean is not scored or included in this pool.
"""
from __future__ import annotations

import torch
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.callbacks.common import Callback


class _BestScored(Callback):
    def reset(self):
        super().reset()
        self.batches = []
        self.best_cost = None
        self.best_action = None

    def start_batch(self):
        if self.best_action is not None:
            self.batches.append((self.best_cost, self.best_action))
        self.best_cost = None
        self.best_action = None

    def compute(self, **state):
        costs = state["costs"]
        candidates = state["candidates"]
        if not torch.isfinite(costs).all():
            raise ValueError("non-finite candidate cost; refusing incumbent selection")
        vals, idx = costs.min(dim=1)
        actions = candidates[torch.arange(costs.shape[0], device=costs.device), idx]
        if self.best_cost is None:
            self.best_cost = vals.clone()
            self.best_action = actions.clone()
        else:
            improved = vals < self.best_cost
            self.best_cost = torch.where(improved, vals, self.best_cost)
            self.best_action = torch.where(improved[:, None, None], actions, self.best_action)
        return None

    def end_solve(self):
        if self.best_action is not None:
            self.batches.append((self.best_cost, self.best_action))
        self.best_cost = None
        self.best_action = None


class NativeCEMIncumbent(CEMSolver):
    def __init__(self, *args, return_best=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_best = return_best
        self._best_recorder = _BestScored()
        if return_best:
            self.callbacks.append(self._best_recorder)

    def solve(self, *args, **kwargs):
        result = super().solve(*args, **kwargs)
        if self.return_best:
            batches = self._best_recorder.batches
            if not batches:
                raise RuntimeError("no scored candidates available")
            result["native_final_mean"] = result["actions"].clone()
            result["actions"] = torch.cat([action.cpu() for _, action in batches], dim=0)
            result["best_scored_cost"] = torch.cat([cost.cpu() for cost, _ in batches], dim=0)
        return result
