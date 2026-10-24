import numpy as np
import torch
from pathlib import Path

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity import _first_successor_prediction
from experiments.spectral_reachability.train_reacher_d2p_counterfactual_repair import (
    _normalize,
    _predict_successor,
    _preprocess_pixels,
    counterfactual_loss,
)


class _Encoder(torch.nn.Module):
    def forward(self, pixels, interpolate_pos_encoding=True):
        class Result:
            last_hidden_state = pixels.mean(dim=(1, 2, 3), keepdim=False).reshape(-1, 1, 1)
        return Result()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder(); self.projector = torch.nn.Identity(); self.action_encoder = torch.nn.Identity()
        self.predictor = torch.nn.Identity(); self.pred_proj = torch.nn.Identity()

    def encode(self, info):
        pixels = info["pixels"].float()
        return {"emb": pixels.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)}

    def predict(self, emb, actions):
        return emb + actions.mean(dim=-1, keepdim=True)


def test_counterfactual_loss_detaches_rgb_targets_and_tracks_same_root_order():
    model = _Model()
    batch = {
        "context": torch.zeros(2, 3, 3, 224, 224), "root": torch.zeros(2, 3, 224, 224), "goal": torch.zeros(2, 3, 224, 224),
        "candidate_successor": torch.ones(2, 3, 224, 224), "zero_successor": torch.zeros(2, 3, 224, 224),
        "history": torch.zeros(2, 3, 10), "candidate": torch.ones(2, 10),
    }
    loss, metrics = counterfactual_loss(model, batch, mean=torch.zeros(10), std=torch.ones(10), temperature=0.1)
    assert torch.isfinite(loss)
    assert metrics["target_requires_grad"].item() == 0.0
    assert metrics["counterfactual_order_accuracy"].item() == 1.0


def test_counterfactual_packet_normalization_preserves_width_and_rejects_mismatch():
    values = torch.zeros(3, 10)
    assert _normalize(values, torch.zeros(10), torch.ones(10)).shape == values.shape


def test_counterfactual_trainer_and_p1_runtime_share_the_exact_root_anchored_successor():
    """A repair target is useful only if its deployment recurrence is identical."""

    model = _Model()
    preprocessor = _Preprocessor(
        ActionNormalizer(np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32)),
    )
    frames = [np.full((224, 224, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    root = np.full((224, 224, 3), 40, dtype=np.uint8)
    goal = np.zeros((224, 224, 3), dtype=np.uint8)
    raw_history = np.arange(30, dtype=np.float32).reshape(3, 10)
    raw_action = np.arange(10, dtype=np.float32)
    runtime_pixels, _, runtime_history = p0._prepare(
        preprocessor, frames, root, goal, raw_history, torch.device("cpu"),
    )
    trainer_context = _preprocess_pixels(
        torch.from_numpy(np.stack([frames[1], frames[2], root])[None]).permute(0, 1, 4, 2, 3),
    )
    trainer_history = _normalize(torch.from_numpy(raw_history[None]), torch.zeros(10), torch.ones(10))
    trainer_action = _normalize(torch.from_numpy(raw_action[None]), torch.zeros(10), torch.ones(10))

    training_prediction = _predict_successor(model, trainer_context, trainer_history, trainer_action)
    runtime_prediction = _first_successor_prediction(model, runtime_pixels, runtime_history, trainer_action)

    assert torch.allclose(training_prediction[0], runtime_prediction, rtol=0.0, atol=1.0e-6)


def test_counterfactual_launcher_requires_strict_replay_order_before_p1_outcome():
    launcher = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/run_reacher_d2p_counterfactual_repair_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "model_to_observation_goal_order_agreement_count == .summary.eligible_roots_audited" in text
    assert "counterfactual calibration did not preserve" in text
