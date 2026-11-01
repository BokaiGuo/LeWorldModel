"""CUDA-only one-packet Jacobian extension; caller owns lock and model lifetime."""
import numpy as np

from experiments.spectral_reachability.codex_p2_program_library_cuda_adapter_20261030_v1 import CudaAdapter


class PrefixCudaAdapter(CudaAdapter):
    def packet_jacobian(self,context):
        t=self.torch
        def work():
            def terminal(logits):
                state,history=self._inputs(context)
                state,history=self._packet(state,history,t.tanh(logits).reshape(5,2),'prefix_jacobian')
                return state[0,-1]
            x=t.zeros(10,dtype=t.float32,device=self.device,requires_grad=True)
            j=t.autograd.functional.jacobian(terminal,x,vectorize=True)
            t.cuda.synchronize(self.device)
            return j.detach().cpu().numpy().astype(np.float64)
        j=self.meter.call('jacobian',work,dict(shape=[192,10],mathematical_vjp_rows=192))
        if j.shape!=(192,10) or not np.isfinite(j).all():
            raise ValueError('actual finite prefix Jacobian')
        return j
