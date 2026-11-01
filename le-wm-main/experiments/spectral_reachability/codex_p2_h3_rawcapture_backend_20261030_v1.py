"""Future H3 native backend: capture actual raw CUDA packet before transition."""
import numpy as np
from experiments.spectral_reachability import codex_p2_symmetric_native_bridge_20261030_v1 as prior
from experiments.spectral_reachability import codex_p2_h3_physical_preflight_20261030_v1 as capture


class RawCaptureBackend(prior.NativeCudaBackend):
    """The caller supplies a durable sink and must bind the enclosing operation."""
    def _terminal(self, context, u, capture_context):
        t = self.t
        state = t.tensor(context["latent_history"], dtype=t.float32, device=self.device)[None]
        history = t.tensor(context["action_history"], dtype=t.float32, device=self.device)[None]
        actions = t.tanh(u).reshape(3, 5, 2)
        contexts = []
        for i in range(3):
            self.emit(dict(kind="packet_attempt", index=self.counts["packet"], packet=i))
            self.counts["packet"] += 1
            capture.save_cuda_raw_packet(actions[i], index=i, sink=self.emit)
            state, history = self.transition(state, history, actions[i])
            self.clock()
            self.emit(dict(kind="packet_return", index=self.counts["packet"]-1, packet=i))
            if capture_context:
                contexts.append(dict(context, latent_history=state[0].detach().cpu().tolist(),
                                     action_history=history[0].detach().cpu().tolist()))
        return state[0, -1], contexts

