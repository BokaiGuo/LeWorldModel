"""Two independently maintained full-chain CEM branches, no response probes."""
import json

import torch
from stable_worldmodel.solver.utils import prepare_init_action

from experiments.pc_bounded_cem_20261105_codex import BoundedCEM, normalized_bounds


class BoundedBranchCEM(BoundedCEM):
    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if (self.batch_size, self.num_samples, self.n_steps, self.topk) != (1, 300, 10, 30):
            raise ValueError("development contract requires batch=1, 300x10, topk=30")
        if self.callbacks:
            raise ValueError("callbacks unsupported")
        total = len(next(iter(info_dict.values())))
        initial = prepare_init_action(self.model, info_dict, init_action, self.horizon,
                                      n_envs=total, action_dim=self.action_dim)
        mean, _ = self.init_action_distrib(total, initial)
        mean = mean.to(self.device)
        low, high = normalized_bounds(self.action_mean, self.action_scale, self.action_dim,
                                      self.margin, self.device, self.dtype)
        means, variances, costs_out, audit = [], [], [], []
        for index in range(total):
            center = mean[index].clamp(min=low, max=high)
            init = torch.randn(300, self.horizon, self.action_dim, generator=self.torch_gen,
                               device=self.device, dtype=self.dtype) * self.var_scale + center
            init[0] = center
            candidates = init.clamp(min=low, max=high)
            projected = int((init != candidates).flatten(1).any(1).sum())
            cost = self._cost(self._infos(info_dict, index, index + 1, 300),
                              candidates.unsqueeze(0))[0]
            elites = candidates[torch.topk(cost, 30, largest=False).indices]
            flat = elites.flatten(1)
            farthest = torch.linalg.vector_norm(flat - flat[0], dim=1).argmax()
            distance0 = torch.linalg.vector_norm(flat - flat[0], dim=1)
            distance1 = torch.linalg.vector_norm(flat - flat[farthest], dim=1)
            labels = distance1 < distance0
            fallback = bool(labels.sum() < 2 or (~labels).sum() < 2)
            if fallback:
                labels = torch.arange(30, device=self.device) % 2 == 1
            branches = [elites[~labels], elites[labels]]
            branch_mean = torch.stack([x.mean(0) for x in branches])
            branch_std = torch.stack([x.std(0) for x in branches])
            for step in range(1, 10):
                count = 150 if step < 9 else 149
                noise = torch.randn(2, count, self.horizon, self.action_dim,
                                    generator=self.torch_gen, device=self.device, dtype=self.dtype)
                raw = noise * branch_std[:, None] + branch_mean[:, None]
                raw[:, 0] = branch_mean
                candidate = raw.clamp(min=low, max=high)
                projected += int((raw != candidate).flatten(2).any(-1).sum())
                branch_cost = self._cost(self._infos(info_dict, index, index + 1, 2 * count),
                                         candidate.reshape(1, 2 * count, self.horizon, self.action_dim))
                branch_cost = branch_cost.reshape(2, count)
                choices = torch.topk(branch_cost, 15, dim=1, largest=False).indices
                selected = torch.gather(candidate, 1, choices[..., None, None].expand(
                    -1, -1, self.horizon, self.action_dim))
                branch_mean, branch_std = selected.mean(1), selected.std(1)
                if not torch.isfinite(branch_mean).all() or not torch.isfinite(branch_std).all():
                    raise ValueError("nonfinite branch distribution")
            final_cost = self._cost(self._infos(info_dict, index, index + 1, 2), branch_mean[None])[0]
            winner = int(final_cost.argmin())
            means.append(branch_mean[winner])
            variances.append(branch_std[winner])
            costs_out.append(float(final_cost[winner]))
            audit.append({"index": index, "scored": 3000, "projected": projected,
                          "fallback_split": fallback, "branch_sizes": [len(x) for x in branches],
                          "selected_branch": winner, "branch_costs": final_cost.cpu().tolist()})
        output_mean, output_var = torch.stack(means).cpu(), torch.stack(variances).cpu()
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "environments": audit},
                                        allow_nan=False) + "\n")
        self.solve_index += 1
        return {"actions": output_mean, "mean": [output_mean], "var": [output_var], "costs": costs_out}
