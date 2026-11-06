"""Fixed-prefix, independently optimized tails under a 632-score development budget."""
import json
import time

import torch
from stable_worldmodel.solver.utils import prepare_init_action

from experiments.pc_bounded_cem_20261105_codex import BoundedCEM, normalized_bounds


class ExplicitPrefixProgram(BoundedCEM):
    def __init__(self, *args, prefix_mode="uniform", prefix_steps=2, **kwargs):
        super().__init__(*args, **kwargs)
        if prefix_mode not in ("uniform", "lookahead") or prefix_steps < 1:
            raise ValueError("invalid prefix configuration")
        self.prefix_mode, self.prefix_steps = prefix_mode, prefix_steps
        if (self.num_samples, self.n_steps, self.topk) != (100, 3, 10):
            raise ValueError("fixed development budget requires 100 x 3, topk 10")

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
            if self.prefix_mode == "lookahead":
                selected = torch.topk(initial_costs, 2, largest=False).indices
            else:
                selected = torch.tensor([0, 1], device=self.device)
            prefixes = population[selected, :self.prefix_steps].clone()
            branch_means = population[selected].clone()
            branch_stds = torch.full_like(branch_means, self.var_scale)
            for _ in range(self.n_steps):
                random = torch.randn(2, self.num_samples, self.horizon, self.action_dim,
                                     generator=self.torch_gen, device=self.device, dtype=self.dtype)
                candidates = (branch_means[:, None] + random * branch_stds[:, None]).clamp(min=low, max=high)
                candidates[:, 0] = branch_means
                candidates[:, :, :self.prefix_steps] = prefixes[:, None]
                costs = self._cost(self._infos(info_dict, i, i + 1, 2 * self.num_samples),
                                   candidates.reshape(1, -1, self.horizon, self.action_dim)).reshape(2, -1)
                elite_idx = torch.topk(costs, self.topk, largest=False, dim=1).indices
                elites = torch.gather(candidates, 1, elite_idx[..., None, None].expand(
                    -1, -1, self.horizon, self.action_dim))
                branch_means, branch_stds = elites.mean(1), elites.std(1)
                # Reassign the exact original values; averaging can introduce float drift.
                branch_means[:, :self.prefix_steps] = prefixes
                if not torch.isfinite(branch_means).all() or not torch.isfinite(branch_stds).all():
                    raise ValueError("nonfinite distribution")
            final_costs = self._cost(self._infos(info_dict, i, i + 1, 2), branch_means[None])[0]
            winner = int(final_costs.argmin())
            output.append(branch_means[winner].cpu())
            variances.append(branch_stds[winner].cpu())
            selected_costs.append(float(final_costs[winner]))
            audit.append({"index": i, "prefix_mode": self.prefix_mode, "initial_indices": selected.cpu().tolist(),
                          "initial_costs": initial_costs.cpu().tolist(), "prefix_steps": self.prefix_steps,
                          "prefixes": prefixes.cpu().tolist(), "branch_final_costs": final_costs.cpu().tolist(),
                          "winner": winner, "selected_actions": branch_means[winner].cpu().tolist(),
                          "scored_model_candidates": 632})
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "elapsed_seconds": time.perf_counter() - started,
                                         "environments": audit}, allow_nan=False) + "\n")
        self.solve_index += 1
        selected = torch.stack(output)
        return {"actions": selected, "mean": [selected], "var": [torch.stack(variances)], "costs": selected_costs}
