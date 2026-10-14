import json

import numpy as np
import torch

from experiments.run_causal_visual_pusht_world_model import jsonable
from experiments.run_causal_visual_pusht_world_model import VisualCausalWorldModel, model_loss
from experiments.run_causal_visual_pusht_world_model import goal_pose_vector


def test_causal_visual_report_jsonable_converts_nested_numpy_trace_values():
    payload = {
        "trace": [
            {"action": np.asarray([0.1, -0.2], dtype=np.float32)},
            {"score": np.float64(0.5)},
        ],
        "array": np.asarray([[1, 2]], dtype=np.int64),
    }
    encoded = json.dumps(jsonable(payload), sort_keys=True)
    assert json.loads(encoded)["trace"][0]["action"] == [0.10000000149011612, -0.20000000298023224]


def test_causal_visual_report_jsonable_preserves_plain_values():
    assert jsonable({"status": "completed", "count": 3}) == {"status": "completed", "count": 3}


def test_orientation_supervision_ignores_noncontact_rows():
    """Free-space no-motion rows must not dominate contact orientation learning."""
    torch.manual_seed(7)
    model = VisualCausalWorldModel(8, "mask_moments", False)
    image = torch.zeros(2, 3, 64, 64)
    image[0, 2, 20:24, 20:24] = 1.0
    image[1, 2, 30:34, 30:34] = 1.0
    action = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    state = torch.zeros(2, 6)
    state[:, 5] = 1.0
    next_state = state.clone()
    next_state[1, 4:6] = torch.tensor([0.6, 0.8])
    goal_image = image.clone()
    goal_pose = torch.zeros(2, 4)
    next_state_raw = next_state.clone()
    goal_pose_raw = goal_pose.clone()
    state_raw = state.clone()
    weights = {
        "readout": 0.0, "transition": 0.0, "effect": 0.0, "block_effect": 0.0,
        "orientation_transition": 1.0, "orientation_effect": 0.0,
        "angle_effect": 0.0, "angle_readout": 0.0, "goal": 0.0,
        "goal_angle": 0.0, "utility": 0.0, "rank": 0.0,
        "successor_rank": 0.0, "uncertainty": 0.0, "mode": 0.0,
    }
    first_loss, first_terms = model_loss(
        model, image, action, state, next_state, goal_image, goal_pose,
        next_state_raw, goal_pose_raw, state_raw,
        contact_mode=torch.tensor([0, 2]), loss_weights=weights,
    )
    changed = next_state.clone()
    changed[0, 4:6] = torch.tensor([-0.8, 0.6])
    second_loss, second_terms = model_loss(
        model, image, action, state, changed, goal_image, goal_pose,
        changed, goal_pose_raw, state_raw,
        contact_mode=torch.tensor([0, 2]), loss_weights=weights,
    )
    assert torch.allclose(first_loss, second_loss)
    assert torch.allclose(first_terms["orientation_transition"], second_terms["orientation_transition"])
    assert first_terms["contact_supported_fraction"].item() == 0.5


def test_goal_angle_readout_is_decoupled_from_current_state_readout():
    """Goal-only angle calibration must not inject an untrained current angle head."""
    model = VisualCausalWorldModel(
        8, "mask_moments", False, "scalar", "legacy", False, False, False,
        False, False, False, False, False, False, True,
    )
    assert not model.explicit_angle_readout
    assert model.explicit_goal_angle_readout
    assert not hasattr(model, "angle_readout_head")
    assert hasattr(model, "goal_angle_head")


def test_goal_pose_vector_uses_rendered_goal_pose_not_reset_goal_state():
    target = goal_pose_vector(np.asarray([256.0, 256.0, np.pi / 4], dtype=np.float32))
    assert np.allclose(target, [256.0, 256.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
