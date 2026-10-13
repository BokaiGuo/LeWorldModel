"""Protocol checks shared by the visual online-planning experiments."""

from __future__ import annotations


def expected_action_block(model, action_dim: int) -> int:
    """Return the frameskip encoded by a released LeWM checkpoint.

    The checkpoint action encoder is trained on a flattened action block of
    ``frameskip * action_dim`` channels.  This is part of the learned model's
    input protocol and is not an inference-time tuning parameter.
    """
    encoded_dim = getattr(getattr(model, "action_encoder", None), "input_dim", None)
    if encoded_dim is None:
        raise ValueError("checkpoint action encoder does not expose input_dim")
    encoded_dim = int(encoded_dim)
    action_dim = int(action_dim)
    if action_dim <= 0 or encoded_dim <= 0 or encoded_dim % action_dim:
        raise ValueError(
            "checkpoint/action-space dimensions are incompatible: "
            f"encoder input_dim={encoded_dim}, action_dim={action_dim}"
        )
    return encoded_dim // action_dim


def validate_action_protocol(model, *, action_dim: int, action_block: int) -> int:
    """Reject a window length that is not matched to checkpoint training."""
    expected = expected_action_block(model, action_dim)
    if int(action_block) != expected:
        raise ValueError(
            "invalid LeWM action protocol: this checkpoint encodes "
            f"{expected} environment frames per action token, but "
            f"--action-block={int(action_block)} was requested. "
            "Use the checkpoint-matched action block; padding or repeating "
            "a shorter block would not be a semantically matched evaluation."
        )
    return expected
