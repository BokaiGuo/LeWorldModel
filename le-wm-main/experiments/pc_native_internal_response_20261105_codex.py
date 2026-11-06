"""Fixed-budget finite-response replacement inside native CEM."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.utils import prepare_init_action


class NativeInternalResponse(CEMSolver):
    def __init__(self, *args, mode="response", amplitude=0.25, audit_path=None, **kwargs):
        if mode not in ("native", "response", "random") or amplitude <= 0:
            raise ValueError("invalid mode or amplitude")
        super().__init__(*args, **kwargs)
        if self.num_samples < 13 or self.topk > self.num_samples:
            raise ValueError("insufficient candidate budget")
        self.mode, self.amplitude = mode, amplitude
        self.audit_path = Path(audit_path) if audit_path else None
        self.solve_index = 0
        if self.audit_path is not None:
            self.audit_path.open("x", encoding="utf-8").close()
        self.direction_gen = torch.Generator(device=self.device).manual_seed(
            int(kwargs.get("seed", 1234)) + 1000003
        )

    def _infos(self, info, start, end, count):
        expanded = {}
        for key, value in info.items():
            value = value[start:end]
            if torch.is_tensor(value):
                dtype = self.dtype if value.is_floating_point() else None
                value = value.to(device=self.device, dtype=dtype).unsqueeze(1)
                value = value.expand(end - start, count, *value.shape[2:])
            elif isinstance(value, np.ndarray):
                value = np.repeat(value[:, None, ...], count, axis=1)
            expanded[key] = value
        return expanded

    def _cost(self, info, actions):
        result = self.model.get_cost(info, actions)
        if result.shape != actions.shape[:2] or not torch.isfinite(result).all():
            raise ValueError("invalid candidate costs")
        return result

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if self.mode == "native":
            return super().solve(info_dict, init_action)
        if self.callbacks:
            raise ValueError("callbacks are unsupported by this development solver")
        total = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(self.model, info_dict, init_action, self.horizon,
                                          n_envs=total, action_dim=self.action_dim)
        mean, var = self.init_action_distrib(total, init_action)
        mean, var = mean.to(self.device), var.to(self.device)
        final_costs = []
        audit = []
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            size = end - start
            batch_mean, batch_var = mean[start:end], var[start:end]
            for step in range(self.n_steps):
                # Draw the full native-sized tensor to keep its RNG progression.
                noise = torch.randn(size, self.num_samples, self.horizon, self.action_dim,
                                    generator=self.torch_gen, device=self.device, dtype=self.dtype)
                candidates = noise * batch_var[:, None] + batch_mean[:, None]
                candidates[:, 0] = batch_mean
                if step == 0:
                    costs = self._cost(self._infos(info_dict, start, end, self.num_samples), candidates)
                else:
                    dirs = torch.randn(size, 4, self.horizon, self.action_dim,
                                       generator=self.direction_gen, device=self.device, dtype=self.dtype)
                    dirs = dirs / dirs.flatten(2).norm(dim=-1).clamp_min(1e-12)[:, :, None, None]
                    probes = torch.cat((batch_mean[:, None] + self.amplitude * dirs,
                                        batch_mean[:, None] - self.amplitude * dirs), dim=1)
                    partial = torch.cat((candidates[:, :self.num_samples - 12], probes), dim=1)
                    partial_cost = self._cost(self._infos(info_dict, start, end, self.num_samples - 4),
                                              partial)
                    probe_cost = partial_cost[:, -8:]
                    if self.mode == "response":
                        signs = torch.where(probe_cost[:, 4:] < probe_cost[:, :4], -1.0, 1.0)
                    else:
                        signs = torch.ones_like(probe_cost[:, :4])
                    signed = signs[:, :, None, None] * dirs
                    combined = signed + torch.roll(signed, shifts=-1, dims=1)
                    combined = combined / combined.flatten(2).norm(dim=-1).clamp_min(1e-12)[:, :, None, None]
                    proposal = batch_mean[:, None] + self.amplitude * combined
                    proposal_cost = self._cost(self._infos(info_dict, start, end, 4), proposal)
                    candidates = torch.cat((partial, proposal), dim=1)
                    costs = torch.cat((partial_cost, proposal_cost), dim=1)
                values, indices = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(size, device=self.device)[:, None].expand(-1, self.topk)
                elite = candidates[batch_indices, indices]
                if self.audit_path is not None:
                    audit.append({
                        "batch_start": start, "batch_end": end, "step": step,
                        "scored_per_environment": int(costs.shape[1]),
                        "mean_candidate_cost": costs[:, 0].cpu().tolist(),
                        "best_cost": values[:, 0].cpu().tolist(),
                        "probe_elites": ((indices >= self.num_samples - 12) &
                                         (indices < self.num_samples - 4)).sum(1).cpu().tolist()
                            if step else [0] * size,
                        "proposal_elites": (indices >= self.num_samples - 4).sum(1).cpu().tolist()
                            if step else [0] * size,
                    })
                batch_mean, batch_var = elite.mean(dim=1), elite.std(dim=1)
            mean[start:end], var[start:end] = batch_mean, batch_var
            final_costs.extend(values.mean(dim=1).cpu().tolist())
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "mode": self.mode,
                                         "iterations": audit}, allow_nan=False) + "\n")
        self.solve_index += 1
        return {"actions": mean.cpu(), "mean": [mean.cpu()], "var": [var.cpu()],
                "costs": final_costs}
