"""Prospective chunk gate for the existing JEPA predict/action_encoder ABI.

The caller supplies an already planned, normalized token sequence and current
latent context. One token is one model transition and contains token_steps
physical actions. No outcome is inspected until the decision is complete.
This adapter meters only calls it dispatches; upstream planning/encoding and
model-internal operations are outside its accounting scope.
"""
import torch

from . import codex_p3_chunk_extension_20261101 as kernel


def run_chunk(*, model, env, states, history_actions, action_tokens,
              physical_actions, goal, projection, token_steps, config, arm, meter):
    """Choose then execute one chunk via ``env.step``; return decision and ledger.

    ``states`` is [1, history, latent], ``history_actions`` is
    [1, history-1, action_features], and ``action_tokens`` is
    [1, tokens, action_features]. The raw physical actions must correspond to
    those tokens; that binding and all upstream costs are the caller's duty.
    """
    if type(token_steps) is not int or token_steps <= 0:
        raise ValueError("token_steps must be positive")
    if not isinstance(config, kernel.Config) or not isinstance(meter, kernel.Meter):
        raise ValueError("invalid config or meter")
    if arm not in ("fixed", "dynamic") or config.base_steps % token_steps or config.extension_steps % token_steps:
        raise ValueError("invalid arm or unaligned chunk")
    if (states.ndim != 3 or states.shape[0] != 1 or states.shape[1] < 1
            or history_actions.ndim != 3 or history_actions.shape[:2] != (1, states.shape[1] - 1)
            or action_tokens.ndim != 3 or action_tokens.shape[0] != 1
            or history_actions.shape[2] != action_tokens.shape[2]
            or len(physical_actions) != action_tokens.shape[1] * token_steps
            or goal.shape != states.shape[2:]
            or projection.ndim != 2 or projection.shape[1] != states.shape[2]
            or projection.shape[0] == 0):
        raise ValueError("invalid context, actions, goal, or projection")
    if any(x.device != states.device or not bool(torch.isfinite(x).all())
           for x in (states, history_actions, action_tokens, goal, projection)):
        raise ValueError("non-finite or mixed-device input")
    max_steps = config.base_steps + config.extension_steps * config.max_extensions
    if len(physical_actions) < (max_steps if arm == "dynamic" else config.base_steps):
        raise ValueError("planned actions do not cover requested horizon")

    def charge(**kwargs):
        return {axis: kwargs.get(axis, 0) for axis in kernel.AXES}

    def predict_prefix(steps):
        latent = states
        actions = history_actions
        with torch.no_grad():
            for index in range(steps // token_steps):
                actions = torch.cat((actions, action_tokens[:, index:index + 1]), dim=1)
                context = latent[:, -states.shape[1]:]
                action_context = actions[:, -states.shape[1]:]
                # Count one predictor sample for this single-trajectory token.
                pred = meter.call("extension_predict", charge(predict=1, predictor_samples=1),
                                  lambda: model.predict(context, model.action_encoder(action_context)))
                if pred.shape != context.shape or not bool(torch.isfinite(pred).all()):
                    raise ValueError("invalid model prediction")
                latent = torch.cat((latent, pred[:, -1:]), dim=1)
        return latent[0, -1]

    def cost(latent):
        return float((projection @ (goal - latent)).square().mean().item())

    decisions = []
    length = config.base_steps
    if arm == "dynamic":
        for index in range(config.max_extensions):
            current = predict_prefix(length)
            extended = predict_prefix(length + config.extension_steps)
            # Proxy disagreement: projected extended prediction vs persistence
            # at the current predicted state; not ensemble uncertainty.
            disagreement = float((projection @ (extended - current)).square().mean().item())
            signal = kernel.Signal(cost(current), cost(extended), disagreement)
            accepted, reason = kernel.choose(config, signal)
            decisions.append({"index": index, "from_steps": length,
                              "to_steps": length + config.extension_steps,
                              "signal": vars(signal), "accepted": accepted, "reason": reason})
            if not accepted:
                break
            length += config.extension_steps

    # Count actual dispatch attempts, including an env.step that raises.
    physical = {"attempted": 0, "returned": 0}
    terminated = truncated = False
    try:
        for action in physical_actions[:length]:
            physical["attempted"] += 1
            _, _, terminated, truncated, _ = env.step(action)
            physical["returned"] += 1
            if terminated or truncated:
                break
    except BaseException as exc:
        # Preserve partial accounting for the caller even on execution failure.
        exc.p3_physical_counts = physical.copy()
        raise
    return {"arm": arm, "chunk_steps": length, "decisions": decisions,
            "resources": {"attempted": meter.attempted.copy(), "returned": meter.returned.copy(),
                          "caps": meter.caps.copy(), "events": list(meter.events),
                          "physical_env_steps": physical},
            "terminated": bool(terminated), "truncated": bool(truncated),
            "accounting_scope": "adapter predictor calls and env.step only; upstream costs excluded",
            "P3_admission": False, "equal_total_compute": False,
            "physical_safety_certified": False}
