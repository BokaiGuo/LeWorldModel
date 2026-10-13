from pathlib import Path

import torch

from experiments.evaluate_causal_visual_checkpoint import (
    checkpoint_adapter_architecture,
    checkpoint_utility_mode,
    parse_args as parse_checkpoint_evaluation_args,
)
from experiments.run_causal_visual_pusht_world_model import (
    PUSHT_RENDER_SIZE,
    VisualCausalWorldModel,
    mask_moment_task_distance,
    physical_contact_mode,
)
from experiments.audit_causal_visual_checkpoint import load_checkpoint


def _payload(mode: str) -> dict:
    model = VisualCausalWorldModel(
        latent_dim=16, vision_encoder="mask_moments", temporal_context=True, utility_mode=mode,
    )
    return {"args": {}, "model": model.state_dict()}


def test_legacy_scalar_checkpoint_contract_is_inferred_from_saved_weight_width():
    payload = _payload("scalar")
    assert checkpoint_utility_mode(payload) == "scalar"
    restored = VisualCausalWorldModel(16, "mask_moments", True, checkpoint_utility_mode(payload))
    restored.load_state_dict(payload["model"])


def test_phase_checkpoint_contract_is_inferred_from_saved_weight_width():
    payload = _payload("phase")
    assert checkpoint_utility_mode(payload) == "phase"
    restored = VisualCausalWorldModel(16, "mask_moments", True, checkpoint_utility_mode(payload))
    restored.load_state_dict(payload["model"])


def test_mode_uncertain_checkpoint_has_explicit_and_loadable_architecture_contract():
    model = VisualCausalWorldModel(16, "mask_moments", True, "scalar", "mode_uncertain")
    payload = {"args": {"adapter_architecture": "mode_uncertain"}, "model": model.state_dict()}
    assert checkpoint_adapter_architecture(payload) == "mode_uncertain"
    restored = VisualCausalWorldModel(16, "mask_moments", True, "scalar", "mode_uncertain")
    restored.load_state_dict(payload["model"])


def test_offline_contact_mode_labels_distinguish_no_contact_sliding_and_separation():
    source = torch.tensor([0.0, 0.0, 100.0, 0.0]).numpy()
    assert physical_contact_mode(source, source.copy()) == 0
    source = torch.tensor([0.0, 0.0, 40.0, 0.0]).numpy()
    sliding = source.copy()
    sliding[2] += 3.0
    assert physical_contact_mode(source, sliding) == 2
    separated = source.copy()
    separated[0] = -100.0
    assert physical_contact_mode(source, separated) == 3


def test_saved_formal_checkpoint_can_be_loaded_without_architecture_guessing():
    checkpoint = Path(
        "results/causal_visual_pusht_rankutility_goalmlp_formaltrain30_20261009_seed20261009/"
        "causal_visual_world_model.pt"
    )
    if not checkpoint.exists():
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_utility_mode(payload) == "scalar"
    args = payload["args"]
    restored = VisualCausalWorldModel(
        int(args["latent_dim"]), str(args["vision_encoder"]), bool(args["temporal_context"]), "scalar",
    )
    restored.load_state_dict(payload["model"])


def test_new_checkpoint_with_explicit_architecture_fields_loads_without_duplicate_kwargs(tmp_path):
    model = VisualCausalWorldModel(16, "mask_moments", True, "scalar", "mode_uncertain")
    checkpoint = tmp_path / "new.pt"
    torch.save({
        "model": model.state_dict(), "normalization": {"mean": [0.0] * 6, "std": [1.0] * 6},
        "args": {"latent_dim": 16, "vision_encoder": "mask_moments", "temporal_context": True,
                 "utility_mode": "scalar", "adapter_architecture": "mode_uncertain"},
    }, checkpoint)
    restored, _normalizer, runtime = load_checkpoint(checkpoint, "cpu")
    assert restored.adapter_architecture == "mode_uncertain"
    assert runtime.utility_mode == "scalar"


def test_structured_readout_variant_has_linear_rgb_geometry_heads():
    model = VisualCausalWorldModel(16, "mask_moments", True, "scalar", "mode_uncertain", True)
    assert isinstance(model.state_head, torch.nn.Linear)
    assert isinstance(model.goal_head, torch.nn.Linear)


def test_contact_frame_dynamics_uses_calibrated_relative_action_chart():
    model = VisualCausalWorldModel(16, "mask_moments", True, "scalar", "mode_uncertain", False, True, True)
    mean = torch.zeros(6)
    std = torch.ones(6)
    model.set_state_normalization(mean, std)
    state = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 1.0]])
    action = torch.tensor([[1.0, 0.0]])
    features = model._dynamics_input(state, action)
    assert features.shape == (1, 15)
    assert torch.isfinite(features).all()
    predicted = model.predict_next(state, action)
    assert predicted.shape == state.shape


def test_mode_conditioned_dynamics_accepts_offline_mode_labels_and_soft_runtime_mix():
    model = VisualCausalWorldModel(16, "mask_moments", False, "scalar", "mode_uncertain", False, True, True)
    model.set_state_normalization(torch.zeros(6), torch.ones(6))
    state = torch.zeros(4, 6)
    action = torch.zeros(4, 2)
    modes = torch.tensor([0, 1, 2, 3])
    hard = model.predict_next(state, action, modes)
    soft = model.predict_next(state, action)
    assert hard.shape == soft.shape == (4, 6)
    assert torch.isfinite(hard).all() and torch.isfinite(soft).all()


def test_temporal_dynamics_keeps_pose_readout_current_only_and_conditions_effects_on_frame_motion():
    model = VisualCausalWorldModel(
        16, "mask_moments", True, "phase", "mode_uncertain", False, True, True, True, False, True,
    )
    model.set_state_normalization(torch.zeros(6), torch.ones(6))
    image = torch.zeros(2, 3, 64, 64)
    previous = torch.zeros_like(image)
    image[:, 2, 20:24, 20:24] = 1.0
    image[:, 0:3, 30:38, 30:38] = 0.5
    previous[:, 2, 19:23, 20:24] = 1.0
    previous[:, 0:3, 30:38, 29:37] = 0.5
    state = model.encode_state(image, previous)
    context = model.encode_dynamics_context(image, previous)
    action = torch.zeros(2, 2)
    assert state.shape == (2, 6)
    assert context is not None and context.shape == (2, 4)
    assert torch.isfinite(model.predict_next(state, action, temporal_context=context)).all()
    assert torch.isfinite(model.score_action_components(state, torch.zeros(2, 4), action, context)).all()


def test_evaluator_accepts_a_predeclared_common_episode_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_causal_visual_checkpoint.py",
            "--checkpoint", "model.pt",
            "--output", str(tmp_path),
            "--eval-seed", "9411",
        ],
    )
    args = parse_checkpoint_evaluation_args()
    assert args.eval_seed == 9411
    assert not args.direct_utility
    assert not args.reversible_orientation_phase
    assert not args.phase_template_only


def test_mask_moment_geometry_uses_the_declared_task_chart_scale():
    left = torch.tensor([[0.0, 0.0]])
    right = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(mask_moment_task_distance(left, right), torch.tensor([PUSHT_RENDER_SIZE]))


def test_previous_action_contact_memory_is_explicit_and_finite():
    model = VisualCausalWorldModel(
        16, "mask_moments", True, "phase", "mode_uncertain", False, True, True, True, False, True, True,
    )
    model.set_state_normalization(torch.zeros(6), torch.ones(6))
    state = torch.zeros(3, 6)
    action = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    previous_action = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    context = torch.zeros(3, 4)
    next_state = model.predict_next(state, action, temporal_context=context, previous_action=previous_action)
    logits = model.predict_contact_logits(state, action, context, previous_action)
    assert next_state.shape == state.shape
    assert logits.shape == (3, 4)
    assert torch.isfinite(next_state).all() and torch.isfinite(logits).all()


def test_explicit_angle_effect_head_preserves_unit_orientation_chart():
    model = VisualCausalWorldModel(
        16, "mask_moments", True, "phase", "mode_uncertain", False, True, True, True, False, True, False, True,
    )
    model.set_state_normalization(torch.zeros(6), torch.ones(6))
    state = torch.zeros(4, 6)
    state[:, 5] = 1.0
    action = torch.randn(4, 2)
    predicted = model.predict_next(state, action, temporal_context=torch.zeros(4, 4))
    norm = torch.linalg.vector_norm(predicted[:, 4:6], dim=-1)
    assert torch.isfinite(predicted).all()
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_explicit_angle_readout_is_rgb_only_and_preserves_unit_orientation_chart():
    model = VisualCausalWorldModel(
        16, "mask_moments", True, "phase", "mode_uncertain", False, True, True, True, False, True, False, False, True,
    )
    model.set_state_normalization(torch.zeros(6), torch.ones(6))
    image = torch.zeros(4, 3, 64, 64)
    image[:, 2, 20:26, 20:26] = 1.0
    image[:, 0:3, 30:40, 30:40] = 0.5
    state = model.encode_state(image, image)
    assert torch.isfinite(state).all()
    assert torch.allclose(torch.linalg.vector_norm(state[:, 4:6], dim=-1), torch.ones(4), atol=1e-5)
