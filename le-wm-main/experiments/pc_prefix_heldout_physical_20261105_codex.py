"""Exploratory held-out tail evaluation, separate from candidate selection."""
from __future__ import annotations

import json

import torch
from stable_worldmodel.solver.utils import prepare_init_action

from experiments.pc_native_internal_response_20261105_codex import NativeInternalResponse


def candidate_set_difference(left, right, atol=1e-6):
    """Count candidates without a matching action in the opposite unordered set."""
    distances = (left[:, :, None] - right[:, None, :]).abs().flatten(3).amax(-1)
    return (distances > atol).all(dim=2), (distances > atol).all(dim=1)


class NativePrefixHeldoutPhysical(NativeInternalResponse):
    def __init__(self, *args, mode="conditional", prefix_steps=2, **kwargs):
        if mode not in ("conditional", "shared"):
            raise ValueError("invalid conditional mode")
        super().__init__(*args, mode="response", **kwargs)
        self.mode = mode
        self.prefix_steps = prefix_steps
        self.heldout_gen = torch.Generator(device=self.device).manual_seed(
            int(kwargs.get("seed", 1234)) + 2000003
        )
        if prefix_steps < 1:
            raise ValueError("prefix must not be empty")

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if self.prefix_steps >= self.horizon or self.callbacks:
            raise ValueError("invalid prefix horizon or unsupported callbacks")
        total = len(next(iter(info_dict.values())))
        initial = prepare_init_action(self.model, info_dict, init_action, self.horizon,
                                      n_envs=total, action_dim=self.action_dim)
        mean, var = self.init_action_distrib(total, initial)
        mean, var = mean.to(self.device), var.to(self.device)
        final_costs, audit = [], []
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            size = end - start
            batch_mean, batch_var = mean[start:end], var[start:end]
            for step in range(self.n_steps):
                noise = torch.randn(size, self.num_samples, self.horizon, self.action_dim,
                                    generator=self.torch_gen, device=self.device, dtype=self.dtype)
                candidates = noise * batch_var[:, None] + batch_mean[:, None]
                candidates[:, 0] = batch_mean
                sign_disagreements = [0] * size
                set_audit = None
                if step == 0:
                    costs = self._cost(self._infos(info_dict, start, end, self.num_samples), candidates)
                else:
                    parents = batch_mean[:, None].repeat(1, 2, 1, 1)
                    parents[:, 1, :self.prefix_steps] = candidates[:, 1, :self.prefix_steps]
                    dirs = torch.randn(size, 2, self.horizon, self.action_dim,
                                       generator=self.direction_gen, device=self.device, dtype=self.dtype)
                    dirs[:, :, :self.prefix_steps] = 0
                    dirs = dirs / dirs.flatten(2).norm(dim=-1).clamp_min(1e-12)[:, :, None, None]
                    offsets = torch.cat((dirs, -dirs), dim=1)
                    probes = (parents[:, :, None] + self.amplitude * offsets[:, None]).reshape(
                        size, 8, self.horizon, self.action_dim)
                    partial = torch.cat((candidates[:, :self.num_samples - 12], probes), dim=1)
                    partial_cost = self._cost(
                        self._infos(info_dict, start, end, self.num_samples - 4), partial)
                    probe_cost = partial_cost[:, -8:].reshape(size, 2, 4)
                    raw_signs = torch.where(probe_cost[:, :, 2:] < probe_cost[:, :, :2], -1.0, 1.0)
                    sign_disagreements = (raw_signs[:, 0] != raw_signs[:, 1]).sum(1).cpu().tolist()
                    signs = raw_signs.clone()
                    if self.mode == "shared":
                        signs[:, 1] = signs[:, 0]
                    signed = signs[:, :, :, None, None] * dirs[:, None]
                    vectors = torch.stack((signed[:, :, 0] + signed[:, :, 1],
                                           signed[:, :, 0] - signed[:, :, 1]), dim=2)
                    vectors = vectors / vectors.flatten(3).norm(dim=-1).clamp_min(1e-12)[..., None, None]
                    proposals = (parents[:, :, None] + self.amplitude * vectors).reshape(
                        size, 4, self.horizon, self.action_dim)
                    # Construct both counterfactual sets from exactly the same parents
                    # and directions. They are not scored unless selected by `mode`.
                    conditional_signed = raw_signs[:, :, :, None, None] * dirs[:, None]
                    shared_signs = raw_signs.clone()
                    shared_signs[:, 1] = shared_signs[:, 0]
                    shared_signed = shared_signs[:, :, :, None, None] * dirs[:, None]

                    def make_set(signed_dirs):
                        vectors = torch.stack((signed_dirs[:, :, 0] + signed_dirs[:, :, 1],
                                               signed_dirs[:, :, 0] - signed_dirs[:, :, 1]), dim=2)
                        vectors = vectors / vectors.flatten(3).norm(dim=-1).clamp_min(1e-12)[..., None, None]
                        return (parents[:, :, None] + self.amplitude * vectors).reshape(
                            size, 4, self.horizon, self.action_dim)

                    conditional_set, shared_set = make_set(conditional_signed), make_set(shared_signed)
                    conditional_new, shared_new = candidate_set_difference(conditional_set, shared_set)
                    set_audit = {
                        "conditional_only": conditional_new.sum(1).cpu().tolist(),
                        "shared_only": shared_new.sum(1).cpu().tolist(),
                        "conditional_only_mask": conditional_new.cpu().tolist(),
                        "shared_only_mask": shared_new.cpu().tolist(),
                    }
                    proposal_cost = self._cost(self._infos(info_dict, start, end, 4), proposals)
                    candidates = torch.cat((partial, proposals), dim=1)
                    costs = torch.cat((partial_cost, proposal_cost), dim=1)
                values, indices = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(size, device=self.device)[:, None].expand(-1, self.topk)
                elite = candidates[batch_indices, indices]
                heldout = None
                if step == 1 and self.solve_index == 0:
                    # Four common B tails are never submitted to top-k. Their RNG
                    # is independent of solver and construction-probe RNGs.
                    b = torch.randn(size, 4, self.horizon, self.action_dim,
                                    generator=self.heldout_gen, device=self.device, dtype=self.dtype)
                    b[:, :, :self.prefix_steps] = 0
                    b = b / b.flatten(2).norm(dim=-1).clamp_min(1e-12)[:, :, None, None]
                    b_chains = (parents[:, :, None] + self.amplitude * b[:, None]).reshape(
                        size, 8, self.horizon, self.action_dim)
                    evaluation_chains = torch.cat((parents, b_chains), dim=1)
                    evaluation_cost = self._cost(self._infos(info_dict, start, end, 10),
                                                 evaluation_chains)
                    heldout = {
                        "parent_cost": evaluation_cost[:, :2].cpu().tolist(),
                        "construction_probe_cost": probe_cost.cpu().tolist(),
                        "heldout_b_cost": evaluation_cost[:, 2:].reshape(size, 2, 4).cpu().tolist(),
                        "parent_actions": parents.cpu().tolist(),
                        "heldout_b_actions": b_chains.reshape(size, 2, 4, self.horizon,
                                                              self.action_dim).cpu().tolist(),
                        "additional_full_chain_scores_per_environment": 10,
                    }
                if self.audit_path is not None:
                    audit.append({
                        "batch_start": start, "batch_end": end, "step": step,
                        "scored_per_environment": int(costs.shape[1]),
                        "sign_disagreements": sign_disagreements,
                        "paired_set_comparison": set_audit,
                        "heldout_effect": heldout,
                        "probe_elites": ((indices >= self.num_samples - 12) &
                                         (indices < self.num_samples - 4)).sum(1).cpu().tolist()
                            if step else [0] * size,
                        "proposal_elites": (indices >= self.num_samples - 4).sum(1).cpu().tolist()
                            if step else [0] * size,
                    })
                batch_mean, batch_var = elite.mean(dim=1), elite.std(dim=1)
            mean[start:end], var[start:end] = batch_mean, batch_var
            final_costs.extend(values.mean(1).cpu().tolist())
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"solve": self.solve_index, "mode": self.mode,
                                         "iterations": audit, "returned_actions": mean.cpu().tolist()},
                                        allow_nan=False) + "\n")
        self.solve_index += 1
        return {"actions": mean.cpu(), "mean": [mean.cpu()], "var": [var.cpu()],
                "costs": final_costs}
