"""Cost-matched finite-response versus random combinations on native CEM."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_native_cem_incumbent_20261105_codex import NativeCEMIncumbent


class NativeFiniteResponse(NativeCEMIncumbent):
    def __init__(self, *args, mode="response", n_directions=4, amplitude=0.25,
                 audit_path=None, **kwargs):
        if mode not in ("response", "random") or n_directions < 1 or amplitude <= 0:
            raise ValueError("invalid finite-response configuration")
        super().__init__(*args, return_best=True, **kwargs)
        self.mode = mode
        self.n_directions = n_directions
        self.amplitude = amplitude
        self.audit_path = Path(audit_path) if audit_path else None
        self.solve_index = 0
        if self.audit_path is not None:
            self.audit_path.open("x", encoding="utf-8").close()

    def _infos(self, info, start, end, n):
        expanded = {}
        for key, value in info.items():
            v = value[start:end]
            if torch.is_tensor(v):
                dtype = self.dtype if v.is_floating_point() else None
                v = v.to(device=self.device, dtype=dtype).unsqueeze(1)
                expanded[key] = v.expand(end - start, n, *v.shape[2:])
            elif isinstance(v, np.ndarray):
                expanded[key] = np.repeat(v[:, None, ...], n, axis=1)
            # Match the native CEM evaluator: other metadata is not passed.
        return expanded

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        result = super().solve(info_dict, init_action)
        parent = result["actions"].to(self.device)
        parent_cost = result["best_scored_cost"].to(self.device)
        count = parent.shape[0]
        output = []
        output_cost = []
        audit_batches = []
        for start in range(0, count, self.batch_size):
            end = min(start + self.batch_size, count)
            base = parent[start:end]
            bs = end - start
            dirs = torch.randn(bs, self.n_directions, self.horizon, self.action_dim,
                               generator=self.torch_gen, device=self.device, dtype=self.dtype)
            norms = dirs.flatten(2).norm(dim=-1).clamp_min(1e-12)
            dirs = dirs / norms[:, :, None, None]
            probes = torch.cat([base[:, None] + self.amplitude * dirs,
                                base[:, None] - self.amplitude * dirs], dim=1)
            costs = self.model.get_cost(self._infos(info_dict, start, end, 2 * self.n_directions), probes)
            if costs.shape != (bs, 2 * self.n_directions) or not torch.isfinite(costs).all():
                raise ValueError("invalid probe costs")
            if self.mode == "response":
                weights = costs[:, self.n_directions:] - costs[:, :self.n_directions]
            else:
                weights = torch.ones(bs, self.n_directions, device=self.device, dtype=self.dtype)
                weights[:, 1::2] = -1
            combined = (weights[:, :, None, None] * dirs).sum(dim=1)
            combined_norm = combined.flatten(1).norm(dim=-1).clamp_min(1e-12)
            proposal = base + self.amplitude * combined / combined_norm[:, None, None]
            proposal_cost = self.model.get_cost(self._infos(info_dict, start, end, 1), proposal[:, None])
            if proposal_cost.shape != (bs, 1) or not torch.isfinite(proposal_cost).all():
                raise ValueError("invalid combined proposal cost")
            candidates = torch.cat([base[:, None], probes, proposal[:, None]], dim=1)
            all_cost = torch.cat([parent_cost[start:end, None], costs, proposal_cost], dim=1)
            best_cost, idx = all_cost.min(dim=1)
            output.append(candidates[torch.arange(bs, device=self.device), idx].cpu())
            output_cost.append(best_cost.cpu())
            if self.audit_path is not None:
                audit_batches.append({
                    "start": start, "end": end,
                    "actions": candidates.cpu().tolist(),
                    "costs": all_cost.cpu().tolist(),
                    "selected": idx.cpu().tolist(),
                })
        result["actions"] = torch.cat(output)
        result["best_scored_cost"] = torch.cat(output_cost)
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "mode": self.mode,
                                         "batches": audit_batches}, allow_nan=False) + "\n")
        self.solve_index += 1
        return result
