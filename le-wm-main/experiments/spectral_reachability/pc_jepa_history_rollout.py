"""JEPA rollout with explicit, causally aligned observed action history.

Inputs are already in the model's normalized packet coordinates.  This module
does not obtain observations, construct goals, normalize actions, or decide
whether proprioception is permitted for a particular benchmark.
"""
import torch


def unroll_history(model, visual, proprio, action_history, action_suffix, context_window):
    """Predict a suffix and return time-major visual and proprio sequences.

    ``visual`` and ``proprio`` are batch-major encoded context. ``action_history``
    holds one packet for each transition between the context frames.
    ``action_suffix`` is [steps, batch, packet_dim], starting at the last
    observed frame. History/context batch one can be broadcast to suffix batch.
    """
    if not isinstance(visual, torch.Tensor) or not isinstance(proprio, torch.Tensor):
        raise ValueError("visual and proprio tensors are required")
    if not isinstance(action_history, torch.Tensor) or not isinstance(action_suffix, torch.Tensor):
        raise ValueError("action history and suffix tensors are required")
    if visual.ndim < 3 or proprio.ndim < 3 or action_history.ndim != 3 or action_suffix.ndim != 3:
        raise ValueError("invalid context or action dimensions")
    steps, batch, width = action_suffix.shape
    context = visual.shape[1]
    if steps < 1 or batch < 1 or width < 1 or context < 1 or not isinstance(context_window, int) or context_window < 1:
        raise ValueError("positive lengths required")
    if proprio.shape[1] != context or action_history.shape[1:] != (context - 1, width):
        raise ValueError("context and action history are not aligned")
    if any(t.shape[0] not in (1, batch) for t in (visual, proprio, action_history)):
        raise ValueError("context batch must be one or equal to suffix batch")
    if any(t.device != action_suffix.device for t in (visual, proprio, action_history)):
        raise ValueError("all tensors must share a device")
    if any(not torch.isfinite(t).all().item() for t in (visual, proprio, action_history, action_suffix)):
        raise ValueError("non-finite inputs")
    video = visual.expand(batch, *visual.shape[1:])
    prop = proprio.expand(batch, *proprio.shape[1:])
    past = action_history.expand(batch, *action_history.shape[1:])
    actions = torch.cat((past, action_suffix.transpose(0, 1)), dim=1)
    action_features = model.encode_act(actions)
    if action_features.shape[:2] != actions.shape[:2]:
        raise ValueError("encoded action time alignment changed")
    for step in range(steps):
        end = context + step
        start = max(0, end - context_window)
        predicted, _, predicted_prop = model.forward_pred(
            video[:, start:end], action_features[:, start:end], prop[:, start:end]
        )
        if predicted is None or predicted_prop is None or predicted.shape[1] != end - start or predicted_prop.shape[1] != end - start:
            raise ValueError("predictor output time alignment changed")
        video = torch.cat((video, predicted[:, -1:]), dim=1)
        prop = torch.cat((prop, predicted_prop[:, -1:]), dim=1)
    return {"visual": video.transpose(0, 1), "proprio": prop.transpose(0, 1)}
