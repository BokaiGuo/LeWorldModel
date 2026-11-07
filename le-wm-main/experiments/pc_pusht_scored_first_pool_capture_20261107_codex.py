"""Passive first-solve capture of the selected-action-only prefix solver."""
import hashlib
from pathlib import Path

import torch

from experiments.pc_explicit_prefix_output_contract_20261107_codex import ExplicitPrefixSelectedActionOnly


class ScoredFirstPoolCapture(ExplicitPrefixSelectedActionOnly):
    def __init__(self, *args, capture_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=False)
        self.capture_call = 0

    def _cost(self, info, actions):
        cost = super()._cost(info, actions)
        if self.solve_index == 0:
            prediction, goal = info["predicted_emb"], info["goal_emb"]
            if prediction.shape[:2] != actions.shape[:2]:
                raise ValueError("predictions and candidates differ in batch shape")
            action = actions.detach().cpu().clone()
            artifact = {"call": self.capture_call, "actions": action, "cost": cost.detach().cpu().clone(),
                        "prediction": prediction.detach().cpu().clone(), "goal": goal.detach().cpu().clone(),
                        "action_sha256": hashlib.sha256(action.numpy().tobytes()).hexdigest()}
            with (self.capture_dir / f"call_{self.capture_call:03}.pt").open("xb") as output:
                torch.save(artifact, output)
            self.capture_call += 1
        return cost
