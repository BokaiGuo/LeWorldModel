"""CUDA adapter for exact bounded PushT witness replay, not physical efficacy.

Context contains three latent boundaries and two between-boundary normalized
10-coordinate action packets. Append a real five-step raw packet before predict;
no invented incoming action before the oldest boundary is needed.
"""
import numpy as np
from experiments.spectral_reachability.d2p_composition_witness_v2 import _context


def packets(actions):
    source = np.asarray(actions)
    if (source.dtype.kind not in 'fi' or source.ndim != 2 or source.shape[1] != 2
            or source.shape[0] < 5 or source.shape[0] % 5
            or not np.isfinite(source).all() or np.any(np.abs(source) > 1)):
        raise ValueError('finite bounded raw PushT five-step packets required')
    return source.astype(np.float32).reshape(-1, 5, 2)


class CudaReplay:
    def __init__(self, model, device):
        if getattr(device, 'type', None) != 'cuda':
            raise ValueError('host CUDA only; CPU neural fallback forbidden')
        self.model, self.device = model, device
        self._check_model()

    def _check_model(self):
        if (any(m.training for m in self.model.modules())
                or any(p.requires_grad or p.device != self.device for p in self.model.parameters())
                or any(b.device != self.device for b in self.model.buffers())):
            raise ValueError('entire model must be frozen eval on explicit CUDA device')

    def __call__(self, context, actions):
        _context(context)
        if (context['context_length'] != 3 or context['action_dim'] != 10
                or context['latent_dim'] != 192):
            raise ValueError('official PushT context layout required')
        raw = packets(actions)
        self._check_model()
        import torch
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        latent = torch.tensor(context['latent_history'], dtype=torch.float32, device=self.device)[None]
        history = torch.tensor(context['action_history'], dtype=torch.float32, device=self.device)[None]
        live = torch.tensor(raw, dtype=torch.float32, device=self.device)
        tokens = p0.packets.normalize_live_packet(live, mean=p0.PUBLIC_ACTION_MEAN, std=p0.PUBLIC_ACTION_STD)
        with torch.no_grad():
            for token in tokens:
                outgoing = torch.cat((history, token[None, None]), dim=1)
                next_latent = self.model.predict(latent, self.model.action_encoder(outgoing))[:, -1]
                if not torch.isfinite(next_latent).all():
                    raise ValueError('nonfinite CUDA transition')
                latent = torch.cat((latent[:, 1:], next_latent[:, None]), dim=1)
                history = outgoing[:, 1:]
        result = dict(context, latent_history=latent[0].cpu().tolist(), action_history=history[0].cpu().tolist())
        _context(result)
        return result
