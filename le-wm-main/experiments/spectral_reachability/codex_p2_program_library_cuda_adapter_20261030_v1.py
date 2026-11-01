"""HOST CUDA transition and H2 root Jacobian adapter; no production launcher."""
import numpy as np


class CudaAdapter:
    """Caller owns lock, model/source identity, meter, lifetime, and journal."""

    def __init__(self, model, device, meter):
        if getattr(device, "type", None) != "cuda" or not callable(getattr(meter, "call", None)):
            raise ValueError("CUDA device and durable resource meter required")
        import torch
        from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
        self.torch=torch
        self.device=device
        self.meter=meter
        self.transition=compiler.make_transition(model,device)

    def _inputs(self, context):
        t=self.torch
        return (t.tensor(context["latent_history"],dtype=t.float32,device=self.device)[None],
                t.tensor(context["action_history"],dtype=t.float32,device=self.device)[None])

    def _packet(self, state, history, packet, phase):
        def call():
            result=self.transition(state,history,packet)
            self.torch.cuda.synchronize(self.device)
            return result
        return self.meter.call("packet",call,dict(phase=phase))

    def root_jacobian(self, context):
        t=self.torch
        def work():
            def terminal(logits):
                state,history=self._inputs(context)
                raw=t.tanh(logits).reshape(2,5,2)
                for packet in raw:
                    state,history=self._packet(state,history,packet,"root_jacobian")
                return state[0,-1]
            x=t.zeros(20,dtype=t.float32,device=self.device,requires_grad=True)
            j=t.autograd.functional.jacobian(terminal,x,vectorize=True)
            t.cuda.synchronize(self.device)
            return j.detach().cpu().numpy().astype(np.float64)
        j=self.meter.call("jacobian",work,dict(shape=[192,20],mathematical_vjp_rows=192))
        if j.shape!=(192,20) or not np.isfinite(j).all():
            raise ValueError("actual finite root Jacobian")
        return j

    def step(self, context, raw):
        a=np.asarray(raw)
        if a.dtype!=np.dtype("<f4") or a.shape!=(5,2) or not np.isfinite(a).all() or np.any(np.abs(a)>1):
            raise ValueError("exact float32 raw packet")
        t=self.torch
        with t.no_grad():
            state,history=self._inputs(context)
            packet=t.tensor(a,dtype=t.float32,device=self.device)
            # Reject any unexpected device transfer conversion before dispatch.
            if packet.detach().cpu().numpy().tobytes()!=a.tobytes():
                raise ValueError("raw CUDA packet bytes")
            state,history=self._packet(state,history,packet,"candidate_or_replay")
            result=dict(context,latent_history=state[0].detach().cpu().tolist(),
                        action_history=history[0].detach().cpu().tolist())
        return result
