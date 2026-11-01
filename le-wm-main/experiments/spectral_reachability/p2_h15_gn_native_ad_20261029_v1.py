"""H15 current-incumbent native raw AD primitive; not a HOST lifecycle.

No loader, CUDA dispatch, output selection, or physical execution is provided.
"""
import numpy as np
from experiments.spectral_reachability import p2_h15_projected_gn_20261029_v1 as gn
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native

EPSILON = .001
ATOL = .003
RTOL = .05


def probes(z, direction):
    """Two feasible probes; compare J against the actual projected displacement."""
    z = gn.project(z)
    d = np.asarray(direction, dtype=np.float64)
    if d.shape != (30,) or not np.isfinite(d).all() or np.linalg.norm(d) == 0:
        raise ValueError('invalid direction')
    d = d / np.linalg.norm(d)
    return gn.project(z + EPSILON*d), gn.project(z - EPSILON*d)


def check_difference(zplus, zminus, yplus, yminus, jac):
    a, b = np.asarray(zplus, np.float64), np.asarray(zminus, np.float64)
    yp, ym = np.asarray(yplus, np.float64), np.asarray(yminus, np.float64)
    j = np.asarray(jac, np.float64)
    if a.shape != (30,) or b.shape != (30,) or yp.shape != (192,) or ym.shape != (192,) or j.shape != (192,30):
        raise ValueError('difference shape')
    if not all(np.isfinite(x).all() for x in (a,b,yp,ym,j)):
        raise ValueError('nonfinite difference')
    delta = a-b
    if np.linalg.norm(delta) < EPSILON/10:
        raise ValueError('degenerate projected direction')
    actual = (yp-ym)/(2*EPSILON)
    expected = (j@delta)/(2*EPSILON)
    error = float(np.linalg.norm(actual-expected))
    limit = float(ATOL + RTOL*max(np.linalg.norm(actual), np.linalg.norm(expected)))
    if error > limit:
        raise ValueError('directional difference failed')
    return dict(error=error, limit=limit, actual_norm=float(np.linalg.norm(actual)),
                expected_norm=float(np.linalg.norm(expected)))


class NativeCurrentAD(native.NativeTree):
    """Frozen model AD at float32 raw incumbent; J_logit via tanh chain rule."""
    def linearize(self, logits, ledger):
        import torch
        z = np.asarray(logits, np.float64)
        actions = gn.raw(z)
        self.frozen(); self.ledger = ledger; self.step = None; self.phase = 'current_AD'
        context = self.task['root_context']
        state0 = torch.tensor(context['latent_history'], dtype=torch.float32, device=self.device)[None]
        hist0 = torch.tensor(context['action_history'], dtype=torch.float32, device=self.device)[None]
        mean = torch.tensor(self.task['action_mean'], dtype=torch.float32, device=self.device)
        std = torch.tensor(self.task['action_std'], dtype=torch.float32, device=self.device)

        def calculate():
            with torch.enable_grad():
                raw = torch.tensor(actions.reshape(30), dtype=torch.float32, device=self.device, requires_grad=True)
                def on_raw(value):
                    return ledger.call('raw_backward', lambda: self._raw_return(value))
                hook = raw.register_hook(on_raw)
                try:
                    state, history = state0, hist0
                    self.arrays('origin', latent=state, history=history, raw_flat=raw, mean=mean, std=std)
                    for i, packet in enumerate(raw.reshape(3,5,2)):
                        self.step = i
                        token = ((packet-mean)/std).reshape(10)
                        outgoing = torch.cat((history, token[None,None]), dim=1)
                        self.arrays('packet', raw=packet, token=token, outgoing=outgoing)
                        encoded = self.model.action_encoder(outgoing)
                        predicted = self.call('predict', lambda: self.model.predict(state,encoded),
                                              dict(latent=state,act_emb=encoded),shape=list(state.shape))
                        native.require(tuple(predicted.shape)==(1,3,192), 'native prediction ABI')
                        state = torch.cat((state[:,1:],predicted[:,-1:]),dim=1);history=outgoing[:,1:]
                        self.arrays('boundary',latent=state,history=history)
                    out = state[0,-1];self.step=None;self.arrays('AD_terminal',value=out)
                    rows=[]
                    for i in range(192):
                        basis=torch.zeros_like(out);basis[i]=1
                        self.store.event(dict(kind='direction',scope=ledger.scope,direction=i))
                        def grad():
                            return self.call('grad',lambda:torch.autograd.grad(out,raw,grad_outputs=basis,
                                retain_graph=i<191,create_graph=False,allow_unused=False)[0],
                                dict(output=out,raw=raw,basis=basis))
                        rows.append(ledger.call('vjp',grad,directions=1))
                    j=torch.stack(rows);self.arrays('jacobian',value=j)
                    return out.detach().cpu().numpy(),j.detach().cpu().numpy()
                finally:
                    hook.remove()
        with self.observe():
            value,jraw=ledger.call('jacobian',calculate)
        self.frozen()
        jlogit=jraw.astype(np.float64)*(1-np.tanh(z)**2)[None,:]
        if not np.isfinite(jlogit).all():raise ValueError('nonfinite Jacobian')
        return dict(logits=z.tolist(),raw=actions.tolist(),nominal=value.tolist(),
                    raw_jacobian=jraw.tolist(),logit_jacobian=jlogit.tolist())
