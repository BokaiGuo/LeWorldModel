"""Safe, self-contained runtime for the public DINO-WM checkpoint.

The released checkpoint predates the current JEPA-WM package and serializes
the predictor modules directly.  This module reconstructs only the temporal
path needed by the spectral audit: DINO patch encoding, proprio/action
embedding, and predictor rollout.  It never loads the decoder or optimizer
and it never uses ``weights_only=False``.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from models.dino import DinoV2Encoder
from models.proprio import ProprioceptiveEmbedding
from models.vit import Attention, FeedForward, Transformer, ViTPredictor
from models.vqvae import Decoder, Quantize, ResBlock, VQVAE


class _SafeDefaultDict:
    """Weights-only compatible replacement for a serialized defaultdict.

    The public checkpoint uses ``defaultdict(dict)`` only for optimizer state.
    The temporal evaluator never needs the default-factory behavior.  PyTorch
    2.12 deliberately rejects ``SETITEMS`` into a real defaultdict under its
    weights-only unpickler, so this allowlisted constructor returns a plain
    dict while preserving every serialized key/value.  It is intentionally
    private and cannot execute checkpoint-provided callables.
    """

    def __new__(cls, *args: Any) -> dict[Any, Any]:
        del args
        return {}


def checkpoint_safe_globals() -> list[Any]:
    """Return the explicit allowlist required by the public checkpoint."""

    return [
        (ViTPredictor, "models.vit.ViTPredictor"),
        (Transformer, "models.vit.Transformer"),
        (FeedForward, "models.vit.FeedForward"),
        (Attention, "models.vit.Attention"),
        (ProprioceptiveEmbedding, "models.proprio.ProprioceptiveEmbedding"),
        (DinoV2Encoder, "models.dino.DinoV2Encoder"),
        (VQVAE, "models.vqvae.VQVAE"),
        (Quantize, "models.vqvae.Quantize"),
        (Decoder, "models.vqvae.Decoder"),
        (ResBlock, "models.vqvae.ResBlock"),
        torch.optim.Adam,
        torch.optim.AdamW,
        collections.OrderedDict,
        (_SafeDefaultDict, "collections.defaultdict"),
        dict,
        set,
        torch.device,
        nn.Dropout,
        nn.Sequential,
        nn.LayerNorm,
        nn.Linear,
        nn.ModuleList,
        nn.Softmax,
        nn.GELU,
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose2d,
        nn.ReLU,
    ]


def load_temporal_components(checkpoint: Path) -> dict[str, Any]:
    """Load only model components from a trusted public DINO-WM checkpoint."""

    from accelerate.optimizer import AcceleratedOptimizer
    from accelerate.state import AcceleratorState, GradientState
    from accelerate.utils.dataclasses import DynamoBackend, DistributedType, TorchDynamoPlugin

    safe = checkpoint_safe_globals() + [
        AcceleratedOptimizer,
        AcceleratorState,
        GradientState,
        DistributedType,
        TorchDynamoPlugin,
        DynamoBackend,
    ]
    with torch.serialization.safe_globals(safe):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required = ("predictor", "action_encoder", "proprio_encoder")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"checkpoint lacks temporal components: {missing}")
    return {name: payload[name] for name in required}


class DinoTemporalModel(nn.Module):
    """DINO-WM temporal path with the checkpoint's 196-token interface."""

    def __init__(self, components: dict[str, Any], device: torch.device, encoder: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder or DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
        self.predictor = components["predictor"]
        self.action_encoder = components["action_encoder"]
        self.proprio_encoder = components["proprio_encoder"]
        self.device = device
        self.context_len = int(self.predictor.pos_embedding.shape[1] // 196)
        if self.predictor.pos_embedding.shape[1] != self.context_len * 196:
            raise ValueError("unexpected DINO-WM predictor positional capacity")
        self.to(device).eval()
        # Older serialized Attention objects may carry ``bias`` as a plain
        # tensor attribute rather than a registered buffer.  Move it
        # explicitly so CPU-loaded checkpoints work on the host GPU too.
        for block in self.predictor.transformer.layers:
            attention = block[0]
            if hasattr(attention, "bias"):
                attention.bias = attention.bias.to(device)

    @staticmethod
    def _visual_transform(visual: torch.Tensor) -> torch.Tensor:
        # The released predictor has 3 * 196 positional slots, hence the
        # public DINOv2 backbone must be evaluated on a 196x196 crop.
        visual = visual.float()
        if visual.max() > 1.5:
            visual = visual / 255.0
        visual = resize(visual, [196, 196], interpolation=InterpolationMode.BICUBIC, antialias=True)
        return normalize(visual, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @staticmethod
    def _embed_proprio(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        """Apply the serialized 1-D embedding without relying on new attrs.

        PyTorch module pickles restore parameters and buffers but do not run a
        newer class' ``__init__``.  The public checkpoint's old embedding
        class therefore lacks newer convenience attributes such as
        ``use_mlp``; its serialized contract is exactly the Conv1d patch
        projection, which is all the DINO-WM temporal path uses.
        """

        projected = module.patch_embed(values.permute(0, 2, 1))
        return projected.permute(0, 2, 1)

    def _encode_features(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        visual = obs["visual"].to(self.device)
        proprio = obs["proprio"].to(self.device).float()
        actions = actions.to(self.device).float()
        b, t = visual.shape[:2]
        frames = self._visual_transform(visual.reshape(b * t, *visual.shape[2:]))
        patches = self.encoder(frames).reshape(b, t, 196, 384)
        action_tokens = self._embed_proprio(self.action_encoder, actions)
        proprio_tokens = self._embed_proprio(self.proprio_encoder, proprio)
        return torch.cat(
            [
                patches,
                proprio_tokens.unsqueeze(2).expand(-1, -1, 196, -1),
                action_tokens.unsqueeze(2).expand(-1, -1, 196, -1),
            ],
            dim=-1,
        )

    def encode(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        """Return visual patch tokens, matching the historical API."""

        return self._encode_features(obs, actions)[..., :384]

    def _predict_next(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t, n, d = tokens.shape
        flat = tokens.reshape(b, t * n, d)
        pred = self.predictor(flat)
        return pred.reshape(b, t, n, d)

    @torch.no_grad()
    def rollout(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], None]:
        actions = actions.to(self.device)
        b, t = actions.shape[:2]
        context = min(self.context_len, t - 1)
        context_obs = {key: value[:, :context] for key, value in obs.items()}
        encoded = self._encode_features(context_obs, actions[:, :context])
        visual = encoded[..., :384].unsqueeze(2)
        history = encoded[:, :context].clone()
        predicted = []
        for index in range(context, t):
            window = history[:, -context:]
            pred_full = self._predict_next(window)[:, -1]
            predicted.append(pred_full[..., :384])
            # The predictor supplies visual/proprio features; the action for
            # the current state remains the observed action block.  This is
            # the action-conditioned autoregressive interface used by the
            # released DINO-WM checkpoint.
            next_action = self._embed_proprio(self.action_encoder, actions[:, index - 1 : index])
            next_action = next_action.expand(-1, pred_full.shape[1], -1)
            next_proprio = pred_full[..., 384:394]
            history = torch.cat(
                [history, torch.cat([pred_full[..., :384], next_proprio, next_action], dim=-1).unsqueeze(1)],
                dim=1,
            )
        # Align with the evaluator's convention: history frames are observed,
        # later frames are predicted.  At horizon h, index context-1+h is the
        # h-step prediction.
        out = torch.cat([visual[:, :context], torch.stack(predicted, dim=1).unsqueeze(2)], dim=1)
        return {"visual": out}, None


class VWorldModel(nn.Module):
    """Drop-in temporal API used by the historical audit scripts."""

    def __init__(
        self,
        image_size: int,
        num_hist: int,
        num_pred: int,
        encoder: nn.Module,
        proprio_encoder: nn.Module,
        action_encoder: nn.Module,
        decoder: nn.Module | None,
        predictor: nn.Module,
        proprio_dim: int,
        action_dim: int,
        concat_dim: int,
        num_action_repeat: int,
        num_proprio_repeat: int,
        train_encoder: bool,
        train_predictor: bool,
        train_decoder: bool,
    ) -> None:
        del image_size, num_hist, num_pred, decoder, proprio_dim, action_dim, concat_dim
        del num_action_repeat, num_proprio_repeat, train_encoder, train_predictor, train_decoder
        super().__init__()
        components = {
            "predictor": predictor,
            "action_encoder": action_encoder,
            "proprio_encoder": proprio_encoder,
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime = DinoTemporalModel(components, device=device, encoder=encoder)
        self.device = device

    def encode(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        return self.runtime.encode(obs, actions)

    def rollout(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        return self.runtime.rollout(obs, actions)

    def eval(self):
        self.runtime.eval()
        return super().eval()

