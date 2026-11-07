"""Decision-local frozen RGB encoding reuse, preserving the single-frame boundary."""
import torch
from torch import nn


class CachedJEPA(nn.Module):
    """Proxy for the one-frame JEPA inference contract, not a history-aware cache."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._key = None
        self._encoded = None
        self.encoder_calls = 0
        self.eval()

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def get_cost(self, info, action_candidates):
        if not torch.is_inference_mode_enabled():
            raise ValueError("cache is valid only for inference")
        pixels, goal = info["pixels"], info["goal"]
        if pixels.shape[:3] != (1, 1, 1) or goal.shape != pixels.shape:
            raise ValueError("only matching single-frame RGB inputs supported")
        key = (pixels, goal)
        if (self._key is None or any(old.device != new.device or old.dtype != new.dtype
                or old.shape != new.shape or not torch.equal(old, new)
                for old, new in zip(self._key, key, strict=True))):
            self._key = tuple(x.detach().clone() for x in key)
            self._encoded = (
                self.model.encode({"pixels": pixels[:, 0]})["emb"].detach().clone(),
                self.model.encode({"pixels": goal[:, 0]})["emb"].detach().clone(),
            )
            self.encoder_calls += 2
        initial, target = self._encoded
        batch, samples, horizon, _ = action_candidates.shape
        if batch != 1 or horizon < 1:
            raise ValueError("unsupported action shape")
        emb = initial[:, None].expand(batch, samples, -1, -1).reshape(batch * samples, 1, -1).clone()
        actions = action_candidates.reshape(batch * samples, horizon, -1)
        for step in range(horizon):
            act_emb = self.model.action_encoder(actions[:, :step + 1])
            prediction = self.model.predict(emb[:, -3:], act_emb[:, -3:])[:, -1:]
            emb = torch.cat((emb, prediction), dim=1)
        info["emb"] = initial[:, None].expand(batch, samples, -1, -1)
        info["goal_emb"] = target
        info["predicted_emb"] = emb.reshape(batch, samples, horizon + 1, -1)
        return self.model.criterion(info)
