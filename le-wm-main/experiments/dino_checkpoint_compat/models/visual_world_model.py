"""Minimal local runtime for the released DINO-WM checkpoint interface.

The public checkpoint serializes predictor and action/proprioception modules,
but the original ``visual_world_model.py`` source is not shipped in this
repository.  This implementation reconstructs the serialized interface:
patch-token DINO features are concatenated with one action and one proprio
token per patch, passed through the released causal predictor, and the first
384 channels are retained as visual predictions.  It is used only to produce
an enriched export; all metadata records the compatibility runtime.
"""

from __future__ import annotations

import torch
from torch import nn

from app.plan_common.models.dino import DinoEncoder
from app.plan_common.models.vit import ViTPredictor


class VWorldModel(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        num_hist: int,
        num_pred: int,
        encoder: nn.Module,
        proprio_encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        proprio_dim: int,
        action_dim: int,
        **_: object,
    ) -> None:
        super().__init__()
        del image_size, num_hist, num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.latent_dim = int(getattr(encoder, "emb_dim", 384))
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # Older serialized predictor blocks store the causal attention mask as
        # a plain tensor attribute rather than a registered buffer.  ``to``
        # therefore cannot move it; do so explicitly after device placement.
        for block in getattr(self.predictor.transformer, "layers", []):
            attention = block[0]
            if hasattr(attention, "bias"):
                attention.bias = attention.bias.to(self.device)
        self.eval()

    def _visual(self, visual: torch.Tensor) -> torch.Tensor:
        batch, frames = visual.shape[:2]
        features = self.encoder(visual.reshape(batch * frames, *visual.shape[2:]))
        return features.reshape(batch, frames, features.shape[1], features.shape[2])

    @staticmethod
    def _embed(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        tokens = module(values)
        if tokens.ndim == 4:
            tokens = tokens.mean(dim=2)
        return tokens

    @classmethod
    def _tokenize(
        cls,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        proprio_encoder: nn.Module,
        action_encoder: nn.Module,
    ) -> torch.Tensor:
        batch, frames, patches, dimension = visual.shape
        del dimension
        proprio_tokens = cls._embed(proprio_encoder, proprio)
        action_tokens = cls._embed(action_encoder, actions)
        proprio_tokens = proprio_tokens.unsqueeze(2).expand(-1, -1, patches, -1)
        action_tokens = action_tokens.unsqueeze(2).expand(-1, -1, patches, -1)
        return torch.cat([visual, proprio_tokens, action_tokens], dim=-1).reshape(batch, frames * patches, -1)

    def encode(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        del actions
        # ``encode`` is the target-side feature extractor in the released
        # evaluator.  Applying the predictor here would exceed its fixed
        # three-frame positional capacity when the evaluator asks for future
        # target frames.
        return self._visual(obs["visual"].to(self.device))

    def rollout(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], None]:
        visual_input = obs["visual"].to(self.device)
        proprio_input = obs["proprio"].to(self.device)
        actions = actions.to(self.device)
        context_visual = self._visual(visual_input)
        context_len = context_visual.shape[1]
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B,T,A], got {tuple(actions.shape)}")
        future = max(int(actions.shape[1]) - context_len, 0)
        patch_count = context_visual.shape[2]
        # Keep the complete predictor token, rather than converting the
        # predicted proprio embedding back into raw proprio coordinates.
        history = self._tokenize(
            context_visual,
            proprio_input[:, :context_len],
            actions[:, :context_len],
            self.proprio_encoder,
            self.action_encoder,
        ).reshape(context_visual.shape[0], context_len, patch_count, -1)
        predicted_visual = []
        for index in range(context_len, context_len + future):
            window = history[:, -context_len:]
            prediction = self.predictor(window.reshape(window.shape[0], -1, window.shape[-1]))
            if prediction.ndim != 3:
                raise ValueError(f"predictor output must be [B, tokens, D], got {tuple(prediction.shape)}")
            next_token = prediction[:, -patch_count:]
            predicted_visual.append(next_token[..., : self.latent_dim])
            if next_token.shape[-1] < self.latent_dim + self.proprio_dim:
                raise ValueError(
                    "checkpoint predictor does not emit the proprio successor "
                    f"required by token-space rollout: got {next_token.shape[-1]} channels"
                )
            next_action = self._embed(self.action_encoder, actions[:, index - 1 : index])
            next_action = next_action.expand(-1, patch_count, -1)
            history = torch.cat(
                [
                    history,
                    torch.cat(
                        [
                            next_token[..., : self.latent_dim],
                            next_token[..., self.latent_dim : self.latent_dim + self.proprio_dim],
                            next_action,
                        ],
                        dim=-1,
                    ).unsqueeze(1),
                ],
                dim=1,
            )
        if predicted_visual:
            future_visual = torch.stack(predicted_visual, dim=1)
            visual = torch.cat([context_visual, future_visual], dim=1)
        else:
            visual = context_visual
        return {"visual": visual}, None


__all__ = ["VWorldModel"]
