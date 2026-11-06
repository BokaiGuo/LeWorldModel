"""Budgeted bounded CEM with explicitly rescored completed warm-tail competition."""
import json
import time

import torch
from stable_worldmodel.solver.utils import prepare_init_action

from experiments.pc_bounded_cem_20261105_codex import BoundedCEM, normalized_bounds


class FeedbackCEM(BoundedCEM):
    """Experimental solver; samples, final mean and warm candidate are all accounted."""

    def __init__(self, *args, selection="mean", **kwargs):
        super().__init__(*args, **kwargs)
        if selection not in ("mean", "scored_pool"):
            raise ValueError("unknown selection rule")
        if self.num_samples < 2 or self.n_steps < 1 or not 2 <= self.topk <= self.num_samples:
            raise ValueError("invalid CEM budget")
        self.selection = selection

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if self.callbacks:
            raise ValueError("callbacks unsupported")
        started = time.perf_counter()
        total = len(next(iter(info_dict.values())))
        warm_length = 0 if init_action is None else init_action.shape[1]
        if warm_length > self.horizon:
            raise ValueError("warm tail exceeds horizon")
        initial = prepare_init_action(self.model, info_dict, init_action, self.horizon,
                                      n_envs=total, action_dim=self.action_dim)
        mean, variance = self.init_action_distrib(total, initial)
        low, high = normalized_bounds(self.action_mean, self.action_scale, self.action_dim,
                                      self.margin, self.device, self.dtype)
        mean, variance = mean.to(self.device).clamp(min=low, max=high), variance.to(self.device)
        completed_warm = mean.clone() if warm_length else None
        means, stds, output_costs, audits = [], [], [], []
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            size = end - start
            center, std = mean[start:end], variance[start:end]
            info = self._infos(info_dict, start, end, self.num_samples)
            best_cost = torch.full((size,), float("inf"), device=self.device)
            best_action = torch.zeros_like(center)
            best_round = torch.full((size,), -1, dtype=torch.long, device=self.device)
            for step in range(self.n_steps):
                noise = torch.randn(size, self.num_samples, self.horizon, self.action_dim,
                                    generator=self.torch_gen, device=self.device, dtype=self.dtype)
                candidate = (noise * std[:, None] + center[:, None]).clamp(min=low, max=high)
                candidate[:, 0] = center
                costs = self._cost(info, candidate)
                values, indices = torch.topk(costs, k=self.topk, largest=False, dim=1)
                rows = torch.arange(size, device=self.device)
                improve = values[:, 0] < best_cost
                best_action[improve] = candidate[rows[improve], indices[improve, 0]]
                best_cost = torch.minimum(best_cost, values[:, 0])
                best_round[improve] = step
                elite = candidate[rows[:, None], indices]
                center, std = elite.mean(1), elite.std(1)
                if not torch.isfinite(center).all() or not torch.isfinite(std).all():
                    raise ValueError("nonfinite CEM distribution")
            # Always score exactly two entries: the final mean and the completed
            # warm tail (or a duplicate mean when no warm tail is available).
            warm = completed_warm[start:end] if completed_warm is not None else center
            scored = torch.stack((center, warm), dim=1)
            scored_cost = self._cost(self._infos(info_dict, start, end, 2), scored)
            options = torch.cat((scored, best_action[:, None]), dim=1)
            option_costs = torch.cat((scored_cost, best_cost[:, None]), dim=1)
            winner = option_costs.argmin(1) if self.selection == "scored_pool" else torch.zeros(
                size, dtype=torch.long, device=self.device)
            rows = torch.arange(size, device=self.device)
            chosen = options[rows, winner]
            means.append(chosen.cpu())
            stds.append(std.cpu())
            output_costs.extend(option_costs[rows, winner].cpu().tolist())
            for i in range(size):
                audits.append({"index": start + i, "warm_tail_length": warm_length,
                               "warm_completion": "prepare_init_action" if warm_length else None,
                               "scored_model_candidates": self.num_samples * self.n_steps + 2,
                               "option_names": ["final_mean", "completed_warm", "best_sample"],
                               "option_costs": option_costs[i].cpu().tolist(),
                               "best_sample_round": int(best_round[i]),
                               "selected": int(winner[i]),
                               "completed_warm_actions": warm[i].cpu().tolist() if warm_length else None,
                               "selected_actions": chosen[i].cpu().tolist()})
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "selection": self.selection,
                                         "elapsed_seconds": time.perf_counter() - started,
                                         "environments": audits}, allow_nan=False) + "\n")
        self.solve_index += 1
        selected = torch.cat(means)
        return {"actions": selected, "mean": [selected], "var": [torch.cat(stds)],
                "costs": output_costs}
