import pytest

from experiments.online_planning_protocol import (
    expected_action_block,
    validate_action_protocol,
)


class _ActionEncoder:
    def __init__(self, input_dim):
        self.input_dim = input_dim


class _Model:
    def __init__(self, input_dim):
        self.action_encoder = _ActionEncoder(input_dim)


def test_checkpoint_matched_action_block_is_inferred_from_encoder():
    model = _Model(input_dim=10)
    assert expected_action_block(model, action_dim=2) == 5
    assert validate_action_protocol(model, action_dim=2, action_block=5) == 5


def test_short_action_block_is_rejected_instead_of_padded():
    model = _Model(input_dim=10)
    with pytest.raises(ValueError, match="invalid LeWM action protocol"):
        validate_action_protocol(model, action_dim=2, action_block=1)


def test_incompatible_action_dimensions_are_rejected():
    model = _Model(input_dim=10)
    with pytest.raises(ValueError, match="dimensions are incompatible"):
        expected_action_block(model, action_dim=3)
