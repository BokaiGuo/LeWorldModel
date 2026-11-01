"""CUDA-only full-horizon Jacobian at fixed nonzero 20-logit anchors."""
import numpy as np
from experiments.spectral_reachability.codex_p2_prefix_recompile_cuda_20261030_v1 import PrefixCudaAdapter


class MultistartCudaAdapter(PrefixCudaAdapter):
    def full_jacobian_at(self,context,anchor):
        t=self.torch
        anchor=np.asarray(anchor,dtype=np.float64)
        if anchor.shape!=(20,) or not np.isfinite(anchor).all():
            raise ValueError('finite full20 anchor')
        def work():
            def terminal(logits):
                state,history=self._inputs(context)
                for packet in t.tanh(logits).reshape(2,5,2):
                    state,history=self._packet(state,history,packet,'joint_anchor_jacobian')
                return state[0,-1]
            x=t.tensor(anchor,dtype=t.float32,device=self.device,requires_grad=True)
            j=t.autograd.functional.jacobian(terminal,x,vectorize=True)
            t.cuda.synchronize(self.device)
            return j.detach().cpu().numpy().astype(np.float64)
        j=self.meter.call('jacobian',work,dict(shape=[192,20],mathematical_vjp_rows=192))
        if j.shape!=(192,20) or not np.isfinite(j).all():
            raise ValueError('actual finite anchored full20 Jacobian')
        return j
