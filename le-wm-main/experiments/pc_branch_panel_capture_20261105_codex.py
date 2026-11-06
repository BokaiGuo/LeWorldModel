"""Non-intervening first-plan capture for fixed panel and warm-start audit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_branch_effect_lookahead_20261105_codex import BranchEffectLookahead


class PanelCapture(BranchEffectLookahead):
    def __init__(self, *args, capture_dir, task_indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=False)
        self.batch_in_first_solve = 0
        self.panel = set((0, 10, 20, 30, 40) if task_indices is None else task_indices)
        if not self.panel or any(not isinstance(i, int) or not 0 <= i < 50 for i in self.panel):
            raise ValueError("invalid fixed task roster")

    def _cost(self, info, actions):
        cost = super()._cost(info, actions)
        if self.solve_index == 0:
            task, call = divmod(self.batch_in_first_solve, 20)
            if task in self.panel:
                if "predicted_emb" not in info or "goal_emb" not in info:
                    raise ValueError("model rollout unavailable")
                result = {
                    "task": task, "call": call, "actions": actions.detach().cpu().clone(),
                    "cost": cost.detach().cpu().clone(),
                    "prediction": info["predicted_emb"].detach().cpu().clone(),
                    "goal": info["goal_emb"].detach().cpu().clone(),
                    "context": {k: v[:, :1].detach().cpu().clone() for k, v in info.items()
                                if torch.is_tensor(v) and k not in ("predicted_emb", "goal_emb", "action")},
                }
                result["action_sha256"] = hashlib.sha256(result["actions"].numpy().tobytes()).hexdigest()
                with (self.capture_dir / f"task_{task:02d}_call_{call:02d}.pt").open("xb") as stream:
                    torch.save(result, stream)
            self.batch_in_first_solve += 1
        return cost

    def solve(self, info_dict, init_action=None):
        def compact(value):
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value
        before = {"solve": self.solve_index,
                  "identity": {k: compact(info_dict[k]) for k in ("id", "step_idx", "episode_idx")
                               if k in info_dict},
                  "warm_start": compact(init_action)}
        result = super().solve(info_dict, init_action)
        before["selected_actions"] = compact(result["actions"])
        with (self.capture_dir / "replanning.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(before, allow_nan=False) + "\n")
        return result
