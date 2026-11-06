"""Evaluate the native final mean and best sampled CEM chain in one context."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_native_cem_incumbent_20261105_codex import NativeCEMIncumbent


class NativeMeanSelection(NativeCEMIncumbent):
    def __init__(self, *args, mode="mean", audit_path=None, **kwargs):
        if mode not in {"mean", "sample", "joint"}:
            raise ValueError(f"invalid selection mode {mode}")
        super().__init__(*args, return_best=True, **kwargs)
        self.mode = mode
        self.audit_path = Path(audit_path) if audit_path else None
        self.solve_index = 0
        if self.audit_path is not None:
            self.audit_path.open("x", encoding="utf-8").close()

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        result = super().solve(info_dict, init_action)
        means = result["native_final_mean"].to(self.device)
        samples = result["actions"].to(self.device)
        sample_costs = result["best_scored_cost"].to(self.device)
        mean_costs = []
        for start in range(0, means.shape[0], self.batch_size):
            end = min(start + self.batch_size, means.shape[0])
            expanded = {}
            for key, value in info_dict.items():
                sliced = value[start:end]
                if torch.is_tensor(sliced):
                    dtype = self.dtype if sliced.is_floating_point() else None
                    expanded[key] = sliced.to(device=self.device, dtype=dtype).unsqueeze(1)
                elif isinstance(sliced, np.ndarray):
                    expanded[key] = sliced[:, None, ...]
            costs = self.model.get_cost(expanded, means[start:end, None])
            if costs.shape != (end - start, 1) or not torch.isfinite(costs).all():
                raise ValueError("invalid native mean score")
            mean_costs.append(costs[:, 0])
        mean_costs = torch.cat(mean_costs)
        if self.mode == "mean":
            choose_sample = torch.zeros_like(mean_costs, dtype=torch.bool)
        elif self.mode == "sample":
            choose_sample = torch.ones_like(mean_costs, dtype=torch.bool)
        else:
            choose_sample = sample_costs < mean_costs
        chosen = torch.where(choose_sample[:, None, None], samples, means)
        result["actions"] = chosen.cpu()
        result["native_mean_cost"] = mean_costs.cpu()
        if self.audit_path is not None:
            record = {
                "solve": self.solve_index, "mode": self.mode,
                "mean_actions": means.cpu().tolist(), "sample_actions": samples.cpu().tolist(),
                "mean_costs": mean_costs.cpu().tolist(), "sample_costs": sample_costs.cpu().tolist(),
                "choose_sample": choose_sample.cpu().tolist(),
            }
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.solve_index += 1
        return result
