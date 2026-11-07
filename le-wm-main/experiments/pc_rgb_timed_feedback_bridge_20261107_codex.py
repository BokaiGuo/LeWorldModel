"""Opt-in bridge from the existing feedback planner to the timed RGB boundary.

Bindings are caller declarations and do not independently prove dispatch.
"""
import torch

from experiments.pc_input_permissions import RGBLeWM
from experiments.pc_rgb_timed_history_contract_20261107_codex import TimedHistoryRGB


class TimedFeedbackBridge(RGBLeWM):
    def __init__(self, timed):
        if not isinstance(timed, TimedHistoryRGB):
            raise ValueError("timed RGB boundary required")
        super().__init__(timed.model, timed.audit)
        self.timed = timed
        self._binding = None

    def bind(self, *, frame_steps, current_step, history):
        if not isinstance(history, torch.Tensor):
            raise ValueError("history tensor required")
        self._binding = (tuple(frame_steps), current_step, history.detach().clone())

    def score(self, inputs, actions):
        if self._binding is None:
            raise ValueError("bind observed clock and history before scoring")
        if set(inputs) != {"pixels", "goal", "action"} or not isinstance(actions, torch.Tensor):
            raise ValueError("RGB, goal, history, and candidate tensors required")
        frame_steps, current_step, bound_history = self._binding
        history = inputs["action"]
        if not isinstance(history, torch.Tensor) or not torch.equal(history, bound_history):
            raise ValueError("history differs from bound observed history")
        if actions.ndim != 4 or history.ndim != 4 or actions.shape[2] <= history.shape[2]:
            raise ValueError("nonempty future required")
        prefix = actions[:, :, :history.shape[2]]
        if (actions.shape[0] != history.shape[0] or actions.shape[3] != history.shape[3]
                or not torch.equal(prefix, history.expand(-1, actions.shape[1], -1, -1))):
            raise ValueError("candidate history differs from observed history")
        return self.timed.score_timed(pixels=inputs["pixels"], goal=inputs["goal"],
                                      history=history, future=actions[:, :, history.shape[2]:],
                                      frame_steps=frame_steps, current_step=current_step)
