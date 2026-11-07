"""Explicitly bind observed action history to future-only LeWM candidates.

The upstream rollout consumes history from the first H-1 candidate packets;
its ``info['action']`` is overwritten and is not a source of history.
"""
import torch

from experiments.pc_input_permissions import RGBLeWM


def score_future(wrapper, *, pixels, goal, history, future):
    """Score future packets after prepending shared, observed historical packets.

    Shapes: pixels [B,1,H,C,Y,X], goal [B,1,G,C,Y,X],
    history [B,1,H-1,A], future [B,S,F,A].  All packets are in the
    same model action coordinates, and the first future packet starts at
    the latest observed frame.
    """
    if not isinstance(wrapper, RGBLeWM):
        raise ValueError("audited RGBLeWM wrapper required")
    if not all(isinstance(x, torch.Tensor) for x in (pixels, goal, history, future)):
        raise ValueError("tensor inputs required")
    if pixels.ndim != 6 or goal.ndim != 6 or history.ndim != 4 or future.ndim != 4:
        raise ValueError("invalid observation, history, or future rank")
    batch, shared, observed = pixels.shape[:3]
    if shared != 1 or observed < 1 or goal.shape[:2] != (batch, 1):
        raise ValueError("observations must have one shared sample axis")
    if future.shape[0] != batch or future.shape[1] < 1 or future.shape[2] < 1 or future.shape[3] < 1:
        raise ValueError("nonempty future candidates required")
    if history.shape != (batch, 1, observed - 1, future.shape[3]):
        raise ValueError("history must contain exactly H-1 observed packets")
    if len({x.device for x in (pixels, goal, history, future)}) != 1:
        raise ValueError("inputs must share a device")
    if history.dtype != future.dtype or not torch.is_floating_point(future):
        raise ValueError("history and future must use the same floating dtype")
    if not all(torch.isfinite(x).all().item() for x in (pixels, goal, history, future)):
        raise ValueError("nonfinite inputs")
    candidates = torch.cat((history.expand(-1, future.shape[1], -1, -1), future), dim=2)
    inputs = {"pixels": pixels, "goal": goal, "action": history}
    return wrapper.score(inputs, candidates)
