"""Matched bounded branch CEM: finite-response combinations versus lookahead."""
import json

import torch
from stable_worldmodel.solver.utils import prepare_init_action

from experiments.pc_bounded_cem_20261105_codex import BoundedCEM, normalized_bounds


class BranchEffectLookahead(BoundedCEM):
    def __init__(self, *args, comparison="lookahead", prefix_steps=2, **kwargs):
        super().__init__(*args, **kwargs)
        if comparison not in ("lookahead", "effect") or prefix_steps < 1:
            raise ValueError("invalid comparison or prefix")
        self.comparison, self.prefix_steps = comparison, prefix_steps

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if (self.batch_size, self.num_samples, self.n_steps, self.topk) != (1, 300, 10, 30):
            raise ValueError("contract requires batch=1, 300x10, topk=30")
        if self.callbacks or self.prefix_steps >= self.horizon:
            raise ValueError("unsupported callbacks or prefix")
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
            cost = self._cost(self._infos(info_dict, index, index + 1, 300), candidates[None])[0]
            elites = candidates[torch.topk(cost, 30, largest=False).indices]
            flat = elites.flatten(1)
            farthest = torch.linalg.vector_norm(flat - flat[0], dim=1).argmax()
            labels = torch.linalg.vector_norm(flat - flat[farthest], dim=1) < torch.linalg.vector_norm(flat - flat[0], dim=1)
            fallback = bool(labels.sum() < 2 or (~labels).sum() < 2)
            if fallback:
                labels = torch.arange(30, device=self.device) % 2 == 1
            branches = [elites[~labels], elites[labels]]
            branch_mean = torch.stack([x.mean(0) for x in branches])
            branch_std = torch.stack([x.std(0) for x in branches])
            rounds = []
            for step in range(1, 10):
                count = 150 if step < 9 else 149
                noise = torch.randn(2, count, self.horizon, self.action_dim,
                                    generator=self.torch_gen, device=self.device, dtype=self.dtype)
                raw = noise * branch_std[:, None] + branch_mean[:, None]
                raw[:, 0] = branch_mean
                random = raw.clamp(min=low, max=high)
                dirs = torch.randn(2, self.horizon, self.action_dim, generator=self.direction_gen,
                                   device=self.device, dtype=self.dtype)
                dirs[:, :self.prefix_steps] = 0
                dirs = dirs / dirs.flatten(1).norm(dim=-1).clamp_min(1e-12)[:, None, None]
                offsets = torch.cat((dirs, -dirs), dim=0)
                probes = (branch_mean[:, None] + self.amplitude * offsets[None]).clamp(min=low, max=high)
                partial = torch.cat((random[:, :count - 6], probes), dim=1)
                partial_cost = self._cost(self._infos(info_dict, index, index + 1, 2 * (count - 2)),
                                          partial.reshape(1, -1, self.horizon, self.action_dim)).reshape(2, count - 2)
                pcost = partial_cost[:, -4:]
                signs = torch.where(pcost[:, 2:] < pcost[:, :2], -1.0, 1.0)
                signed = signs[:, :, None, None] * dirs[None]
                vectors = torch.stack((signed[:, 0] + signed[:, 1], signed[:, 0] - signed[:, 1]), dim=1)
                vectors = vectors / vectors.flatten(2).norm(dim=-1).clamp_min(1e-12)[..., None, None]
                effect = (branch_mean[:, None] + self.amplitude * vectors).clamp(min=low, max=high)
                proposal = effect if self.comparison == "effect" else random[:, -2:]
                proposal_cost = self._cost(self._infos(info_dict, index, index + 1, 4),
                                           proposal.reshape(1, 4, self.horizon, self.action_dim)).reshape(2, 2)
                candidate = torch.cat((partial, proposal), dim=1)
                branch_cost = torch.cat((partial_cost, proposal_cost), dim=1)
                choices = torch.topk(branch_cost, 15, dim=1, largest=False).indices
                selected = torch.gather(candidate, 1, choices[..., None, None].expand(-1, -1, self.horizon, self.action_dim))
                rounds.append({"step": step, "scored": 2 * count, "probe_costs": pcost.cpu().tolist(),
                               "proposal_costs": proposal_cost.cpu().tolist(),
                               "proposal_elites": (choices >= count - 2).sum(1).cpu().tolist(),
                               "proposal_diff_from_random": (effect - random[:, -2:]).flatten(2).norm(dim=-1).cpu().tolist()})
                branch_mean, branch_std = selected.mean(1), selected.std(1)
                if not torch.isfinite(branch_mean).all() or not torch.isfinite(branch_std).all():
                    raise ValueError("nonfinite branch distribution")
            final_cost = self._cost(self._infos(info_dict, index, index + 1, 2), branch_mean[None])[0]
            winner = int(final_cost.argmin())
            means.append(branch_mean[winner])
            variances.append(branch_std[winner])
            costs_out.append(float(final_cost[winner]))
            audit.append({"index": index, "scored": 3000, "fallback_split": fallback,
                          "selected_branch": winner, "branch_costs": final_cost.cpu().tolist(),
                          "selected_normalized_chain": branch_mean[winner].cpu().tolist(),
                          "rounds": rounds})
        output_mean, output_var = torch.stack(means).cpu(), torch.stack(variances).cpu()
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "comparison": self.comparison,
                                         "environments": audit}, allow_nan=False) + "\n")
        self.solve_index += 1
        return {"actions": output_mean, "mean": [output_mean], "var": [output_var], "costs": costs_out}
