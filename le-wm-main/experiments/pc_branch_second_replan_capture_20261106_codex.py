"""Capture selected actual second-plan pools without changing solver outputs."""
from pathlib import Path

import torch

from experiments.pc_branch_panel_capture_20261105_codex import PanelCapture
from experiments.pc_branch_effect_lookahead_20261105_codex import BranchEffectLookahead


class SecondReplanCapture(PanelCapture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_indices=[15, 25, 35], **kwargs)
        self.first_ids = None
        self.replan_tasks = None
        self.second_call = 0

    def solve(self, info_dict, init_action=None):
        ids = info_dict["id"].reshape(-1).detach().cpu().tolist()
        if self.solve_index == 0:
            if len(ids) != 50 or len(set(ids)) != 50:
                raise ValueError("unexpected initial environment identity")
            self.first_ids = {value: index for index, value in enumerate(ids)}
        elif self.solve_index == 1:
            if self.first_ids is None or any(value not in self.first_ids for value in ids):
                raise ValueError("unbound replan identity")
            self.replan_tasks = [self.first_ids[value] for value in ids]
        return super().solve(info_dict, init_action)

    def _cost(self, info, actions):
        if self.solve_index != 1:
            return super()._cost(info, actions)
        cost = BranchEffectLookahead._cost(self, info, actions)
        index, call = divmod(self.second_call, 20)
        if self.replan_tasks is None or index >= len(self.replan_tasks):
            raise ValueError("replan call attribution failed")
        task = self.replan_tasks[index]
        if task in (15, 25, 35):
            if "predicted_emb" not in info or "goal_emb" not in info:
                raise ValueError("same-query embeddings unavailable")
            record = {"task": task, "call": call, "actions": actions.detach().cpu().clone(),
                      "cost": cost.detach().cpu().clone(),
                      "prediction": info["predicted_emb"].detach().cpu().clone(),
                      "goal": info["goal_emb"].detach().cpu().clone(),
                      "context": {k: v[:, :1].detach().cpu().clone() for k, v in info.items()
                                  if torch.is_tensor(v) and k not in ("predicted_emb", "goal_emb", "action")}}
            with (Path(self.capture_dir) / f"task_{task:02d}_replan_call_{call:02d}.pt").open("xb") as stream:
                torch.save(record, stream)
        self.second_call += 1
        return cost
