"""Record, without extra inference, every scored chain in one fixed plan."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.pc_branch_effect_lookahead_20261105_codex import BranchEffectLookahead


class FullPoolCapture(BranchEffectLookahead):
    def __init__(self, *args, capture_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=False)
        self.capture_call = 0
        self.capture_finished = False

    def _cost(self, info, actions):
        cost = super()._cost(info, actions)
        if self.solve_index == 0 and not self.capture_finished:
            if "predicted_emb" not in info or "goal_emb" not in info:
                raise ValueError("model did not expose its same-query rollout and goal")
            prediction, goal = info["predicted_emb"], info["goal_emb"]
            if prediction.shape[:2] != actions.shape[:2]:
                raise ValueError("prediction does not align with candidate batch")
            tensors = {k: v[:, :1].detach().cpu().clone() for k, v in info.items()
                       if torch.is_tensor(v) and k not in ("predicted_emb", "goal_emb", "action")}
            arrays = {k: np.array(v[:, :1], copy=True) for k, v in info.items()
                      if isinstance(v, np.ndarray)}
            result = {"call": self.capture_call, "actions": actions.detach().cpu().clone(),
                      "cost": cost.detach().cpu().clone(), "prediction": prediction.detach().cpu().clone(),
                      "goal": goal.detach().cpu().clone(), "context": tensors, "context_numpy": arrays}
            result["action_sha256"] = hashlib.sha256(result["actions"].numpy().tobytes()).hexdigest()
            file = self.capture_dir / f"call_{self.capture_call:02d}.pt"
            with file.open("xb") as stream:
                torch.save(result, stream)
            self.capture_call += 1
            if self.capture_call == 20:
                self.capture_finished = True
        return cost
