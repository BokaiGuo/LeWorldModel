"""Explicit-prefix CEM retaining the best already-scored executable chain."""
import json
import time

import torch
from stable_worldmodel.solver.utils import prepare_init_action

from experiments.pc_bounded_cem_20261105_codex import normalized_bounds
from experiments.pc_explicit_prefix_program_20261106_codex import ExplicitPrefixProgram


class ExplicitPrefixScoredIncumbent(ExplicitPrefixProgram):
    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if self.callbacks or self.prefix_steps >= self.horizon:
            raise ValueError("unsupported callbacks or horizon")
        started = time.perf_counter()
        total = len(next(iter(info_dict.values())))
        initial = prepare_init_action(self.model, info_dict, init_action, self.horizon,
                                      n_envs=total, action_dim=self.action_dim)
        mean, _ = self.init_action_distrib(total, initial)
        low, high = normalized_bounds(self.action_mean, self.action_scale, self.action_dim,
                                      self.margin, self.device, self.dtype)
        output, variances, selected_costs, audit = [], [], [], []
        for i in range(total):
            center = mean[i].to(self.device).clamp(min=low, max=high)
            population = (center + self.var_scale * torch.randn(
                30, self.horizon, self.action_dim, generator=self.torch_gen,
                device=self.device, dtype=self.dtype)).clamp(min=low, max=high)
            population[0] = center
            initial_costs = self._cost(self._infos(info_dict, i, i + 1, 30), population[None])[0]
            if not torch.isfinite(initial_costs).all():
                raise ValueError("nonfinite initial costs")
            selected = (torch.topk(initial_costs, 2, largest=False).indices if self.prefix_mode == "lookahead"
                        else torch.tensor([0, 1], device=self.device))
            prefixes = population[selected, :self.prefix_steps].clone()
            branch_means = population[selected].clone()
            branch_stds = torch.full_like(branch_means, self.var_scale)
            initial_winner = int(initial_costs.argmin())
            best = population[initial_winner].clone()
            best_cost = float(initial_costs[initial_winner])
            best_source = {"kind": "initial", "slot": initial_winner}
            for round_index in range(self.n_steps):
                random = torch.randn(2, self.num_samples, self.horizon, self.action_dim,
                                     generator=self.torch_gen, device=self.device, dtype=self.dtype)
                candidates = (branch_means[:, None] + random * branch_stds[:, None]).clamp(min=low, max=high)
                candidates[:, 0] = branch_means
                candidates[:, :, :self.prefix_steps] = prefixes[:, None]
                costs = self._cost(self._infos(info_dict, i, i + 1, 2 * self.num_samples),
                                   candidates.reshape(1, -1, self.horizon, self.action_dim)).reshape(2, -1)
                if not torch.isfinite(costs).all():
                    raise ValueError("nonfinite candidate costs")
                slot = int(costs.reshape(-1).argmin())
                if float(costs.reshape(-1)[slot]) < best_cost:
                    best_cost = float(costs.reshape(-1)[slot])
                    best = candidates.reshape(-1, self.horizon, self.action_dim)[slot].clone()
                    best_source = {"kind": "sample", "round": round_index,
                                   "branch": slot // self.num_samples, "slot": slot % self.num_samples}
                elite_idx = torch.topk(costs, self.topk, largest=False, dim=1).indices
                elites = torch.gather(candidates, 1, elite_idx[..., None, None].expand(
                    -1, -1, self.horizon, self.action_dim))
                branch_means, branch_stds = elites.mean(1), elites.std(1)
                branch_means[:, :self.prefix_steps] = prefixes
                if not torch.isfinite(branch_means).all() or not torch.isfinite(branch_stds).all():
                    raise ValueError("nonfinite distribution")
            final_costs = self._cost(self._infos(info_dict, i, i + 1, 2), branch_means[None])[0]
            if not torch.isfinite(final_costs).all():
                raise ValueError("nonfinite final costs")
            mean_winner = int(final_costs.argmin())
            if float(final_costs[mean_winner]) < best_cost:
                best_cost = float(final_costs[mean_winner])
                best = branch_means[mean_winner].clone()
                best_source = {"kind": "mean", "branch": mean_winner}
            output.append(best.cpu())
            variances.append(branch_stds[mean_winner].cpu())
            selected_costs.append(best_cost)
            audit.append({"index": i, "prefix_mode": self.prefix_mode, "initial_indices": selected.cpu().tolist(),
                          "initial_costs": initial_costs.cpu().tolist(), "prefix_steps": self.prefix_steps,
                          "prefixes": prefixes.cpu().tolist(), "branch_final_costs": final_costs.cpu().tolist(),
                          "winner": mean_winner, "best_source": best_source, "selected_cost": best_cost,
                          "selected_actions": best.cpu().tolist(), "scored_model_candidates": 632})
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "elapsed_seconds": time.perf_counter() - started,
                                         "environments": audit}, allow_nan=False) + "\n")
        self.solve_index += 1
        selected = torch.stack(output)
        return {"actions": selected, "mean": [selected], "var": [torch.stack(variances)], "costs": selected_costs}
