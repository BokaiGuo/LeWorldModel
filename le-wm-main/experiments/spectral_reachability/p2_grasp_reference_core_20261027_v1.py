"""Independent GRASP core mapped to the official LeWM/JEPA-WMs H1 adapters.

Algorithm references (not imported): michael-psenka/grasp at 64f988a,
le-wm-grasp at 94be3c4, jepa-wms-grasp at 2d81ea8. Original MIT notice is
retained in the source audit's sources/LICENSE.source.txt. This is not the
large DINO-WM planner, paper-result reproduction, or a real-H3 adapter.
No model loading, encoding, downloads, execution CLI, or neural CPU fallback.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    # Official LeWM config/eval/solver/grasp.yaml, except local horizon/seed
    # explicitly declared here; these are not outcome-selected hyperparameters.
    horizon: int = 3
    steps: int = 750
    lr_s: float = .1
    lr_a: float = .005
    state_noise: float = .005
    sync_interval: int = 50
    sync_steps: int = 10
    sync_lr: float = .05
    seed: int = 20261027

    def __post_init__(self):
        for name in ('horizon', 'steps', 'sync_steps'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('positive integer ' + name + ' required')
        for name in ('sync_interval', 'seed'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError('nonnegative integer ' + name + ' required')
        for name in ('lr_s', 'lr_a', 'sync_lr', 'state_noise'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (name != 'state_noise' and value == 0):
                raise ValueError('finite positive rates/nonnegative noise required')


def protocol():
    return dict(algorithm='GRASP mechanism / official H1 adapter subset, independent implementation',
        virtual_states='T-1 linear-interpolated flat visual states; fixed initial and goal',
        soft_loss='sum_t mean((F(stopgrad(s_t),a_t)-s_(t+1))^2) + sum_t mean((F(stopgrad(s_t),a_t)-goal)^2)',
        noise='after each joint Adam step: iid Gaussian on virtual states only, constant std',
        sync='(k+1)%interval==0 and k>0; fresh action-only Adam on serial terminal MSE; full state chain gradients; reset joint Adam; do not refresh virtual states',
        input_context='last_state_only_H1; generated history during sync',
        same_as_existing_real_H3_contract=False, official_equivalence=False,
        limitations=['no DINO-WM cylinder/lagrangian/accept-reject path',
                     'no official schedule_decay or CEM sync extension',
                     'local generator replaces global noise RNG',
                     'float32/64 explicit input dtype; no mixed precision',
                     'independent terminal verification added and separately counted',
                     'no action clipping/tanh or raw normalization inside core',
                     'no raw actions or physical execution authorized'],
        performance_evidence=False, local_reproduction=False, P0_admission=False, P2_admission=False)


def budget(cfg):
    syncs = [k+1 for k in range(cfg.steps) if cfg.sync_interval > 0 and (k+1)%cfg.sync_interval == 0 and k > 0]
    return dict(soft_transition_samples_per_root=cfg.steps*cfg.horizon,
        sync_transition_samples_per_root=len(syncs)*cfg.sync_steps*cfg.horizon,
        verification_transition_samples_per_root=cfg.horizon,
        backward_api_calls_per_root=cfg.steps+len(syncs)*cfg.sync_steps,
        sync_after_iterations_1based=syncs, model_encoding_cost='outside core; unknown until runtime receipt',
        notes='batch1 estimates; transition != wall-clock compute; predictor/action encoder hooks still required')


def soft_loss(wm, start, goal, states, actions):
    import torch
    full = torch.cat((start[:,None], states, goal[:,None]), dim=1)
    loss = actions.new_zeros(())
    for t in range(actions.shape[1]):
        prediction = wm.edge(full[:,t].detach(), actions[:,t])
        loss = loss + (prediction-full[:,t+1]).square().mean() + (prediction-goal.detach()).square().mean()
    return loss


def serial(wm, start, actions):
    import torch
    context = wm.begin(start)
    states = [start]
    for t in range(actions.shape[1]):
        context, prediction = wm.advance(context, actions[:,t])
        states.append(prediction)
    return torch.stack(states, dim=1)


class PlanningFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def plan(wm, start, goal, cfg, *, analytic_fixture=False):
    """Only encoded tensors; production must explicitly be reviewed host CUDA.

    CPU mode is reserved for non-Module analytic test doubles, not real models.
    Returned ledgers count transition boundary attempts, not hidden module calls.
    """
    import torch
    if not isinstance(cfg, Config):
        raise ValueError('validated Config required')
    if not analytic_fixture and (start.device.type != 'cuda' or not isinstance(wm, (LeWMH1, NativeH1)) or wm.analytic_fixture):
        raise ValueError('reviewed explicit CUDA adapter required; no neural CPU fallback')
    if analytic_fixture and isinstance(wm, torch.nn.Module):
        raise ValueError('analytic fixtures cannot be neural Modules')
    if (start.ndim != 2 or start.shape != goal.shape or start.device != goal.device or
            start.dtype != goal.dtype or start.dtype not in (torch.float32, torch.float64) or
            not start.numel() or not torch.isfinite(start).all() or not torch.isfinite(goal).all()):
        raise ValueError('aligned finite encoded start/goal [B,D] required')
    start, goal = start.detach(), goal.detach()
    b, d = start.shape
    if type(wm.action_dim) is not int or wm.action_dim < 1:
        raise ValueError('positive action dimension required')
    weights = torch.arange(1,cfg.horizon,device=start.device,dtype=start.dtype)/cfg.horizon
    states = (start[:,None]+weights[None,:,None]*(goal-start)[:,None]).clone().requires_grad_(True)
    actions = start.new_zeros((b,cfg.horizon,wm.action_dim),requires_grad=True)
    generator = torch.Generator(device=start.device).manual_seed(cfg.seed)
    receipt = dict(config=asdict(cfg), official_equivalence=False, analytic_fixture=analytic_fixture,
        status='started', transitions_requested={k:0 for k in ('soft','sync','verification')},
        transitions_completed={k:0 for k in ('soft','sync','verification')}, backward_requested=0,
        backward_completed=0, iterations=[], events=[], sync_after_iterations_1based=[],
        accounting_scope='transition attempts/completions and batch samples; not module hooks or encoding cost')
    phase = 'soft'

    class Counted:
        def call(self, function, *args):
            event = dict(phase=phase, status='requested', batch_samples=b)
            receipt['events'].append(event)
            receipt['transitions_requested'][phase] += b
            try:
                out = function(*args)
                prediction = out[1] if isinstance(out, tuple) else out
                if prediction.shape != start.shape or not torch.isfinite(prediction).all():
                    raise ValueError('transition shape/finite contract rejected')
                receipt['transitions_completed'][phase] += b
                event['status'] = 'completed'
                return out
            except Exception as exc:
                event.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
                raise
        def edge(self, s, a): return self.call(wm.edge, s, a)
        def begin(self, s): return wm.begin(s)
        def advance(self, state, a): return self.call(wm.advance, state, a)

    counted = Counted()
    def optimizer():
        return torch.optim.Adam([{'params':states,'lr':cfg.lr_s},{'params':actions,'lr':cfg.lr_a}])
    def backward(loss):
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        receipt['backward_requested'] += 1
        loss.backward()
        receipt['backward_completed'] += 1
    try:
        opt = optimizer()
        for k in range(cfg.steps):
            phase = 'soft'
            opt.zero_grad()
            loss = soft_loss(counted,start,goal,states,actions)
            backward(loss)
            opt.step()
            with torch.no_grad():
                if states.numel() and cfg.state_noise > 0:
                    states.add_(cfg.state_noise*torch.randn(states.shape,device=states.device,dtype=states.dtype,generator=generator))
            receipt['iterations'].append(dict(iteration_1based=k+1, pre_update_soft_loss=float(loss.detach())))
            if cfg.sync_interval > 0 and (k+1)%cfg.sync_interval == 0 and k > 0:
                phase = 'sync'
                receipt['sync_after_iterations_1based'].append(k+1)
                synced = actions.detach().clone().requires_grad_(True)
                sync_opt = torch.optim.Adam([synced],lr=cfg.sync_lr)
                for _ in range(cfg.sync_steps):
                    sync_opt.zero_grad()
                    sync_loss = (serial(counted,start,synced)[:,-1]-goal).square().mean()
                    backward(sync_loss)
                    sync_opt.step()
                actions = synced.detach().clone().requires_grad_(True)
                opt = optimizer()
        phase = 'verification'
        with torch.no_grad():
            verification = serial(counted,start,actions)
        receipt['status'] = 'analytic_fixture_completed' if analytic_fixture else 'encoded_core_completed_not_physical'
        return dict(actions=actions.detach(),states=states.detach(),verification=verification,receipt=receipt)
    except Exception as exc:
        receipt.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise PlanningFailure(str(exc),receipt) from exc


def _guard(module, analytic_fixture):
    import torch
    if analytic_fixture:
        if isinstance(module, torch.nn.Module):
            raise ValueError('non-neural analytic test doubles only')
        return
    if not isinstance(module, torch.nn.Module) or any(m.training for m in module.modules()):
        raise ValueError('frozen eval CUDA Module required')
    if any(p.requires_grad or p.device.type != 'cuda' for p in module.parameters()) or any(b.device.type != 'cuda' for b in module.buffers()):
        raise ValueError('frozen CUDA parameters/buffers required')


class LeWMH1:
    """Official LeWM adapter's H1 edge; does NOT ingest real H3 history."""
    def __init__(self, model, *, history_size, action_dim=1, analytic_fixture=False):
        _guard(model,analytic_fixture)
        if type(history_size) is not int or history_size < 1 or type(action_dim) is not int or action_dim < 1:
            raise ValueError('explicit positive history/action dimensions required')
        self.model,self.history_size,self.action_dim,self.analytic_fixture = model,history_size,action_dim,analytic_fixture
    def edge(self,s,a):
        return self.model.predict(s[:,None],self.model.action_encoder(a[:,None]))[:,-1]
    def begin(self,s):
        return s[:,None],s.new_empty((len(s),0,self.action_dim))
    def advance(self,state,a):
        import torch
        z,history = state
        acts = torch.cat((history,a[:,None]),dim=1)
        pred = self.model.predict(z,self.model.action_encoder(acts))[:,-1]
        z = torch.cat((z,pred[:,None]),dim=1)[:,-self.history_size:]
        return (z,acts[:,-(self.history_size-1):] if self.history_size>1 else acts[:,:0]),pred


class NativeH1:
    """Direct native AdaLN ABI, full patches and encoded proprio, not CLS LeWM.

    Pairwise calls reuse the last real proprio. Sync propagates predicted
    proprio and generated visual/action history. This is the official adapter's
    H1 reset policy, not the existing p5 real three-frame factual contract.
    No encoder/normalizer calls: actions must already be native normalized.
    """
    def __init__(self,predictor,*,last_proprio,visual_shape,action_dim,history_size,analytic_fixture=False):
        _guard(predictor,analytic_fixture)
        if len(visual_shape)!=4 or any(type(d) is not int or d<1 for d in visual_shape) or last_proprio.ndim!=4 or last_proprio.shape[1]!=1:
            raise ValueError('explicit native visual/proprio shape contract required')
        if type(history_size) is not int or history_size<1 or type(action_dim) is not int or action_dim<1:
            raise ValueError('explicit positive history/action dimensions required')
        if not analytic_fixture and (last_proprio.device.type!='cuda' or visual_shape!=(1,16,16,384) or
                                    tuple(last_proprio.shape[2:])!=(256,16) or action_dim!=10):
            raise ValueError('reviewed native PushT CUDA ABI required')
        self.predictor,self.last_proprio = predictor,last_proprio.detach()
        self.visual_shape,self.action_dim,self.history_size,self.analytic_fixture=visual_shape,action_dim,history_size,analytic_fixture
    def _predict(self,z,a,p):
        result=self.predictor(z,a,p)
        if len(result)!=3:
            raise ValueError('native predictor three-output ABI required')
        y,_,prop=result
        b,t=z.shape[:2]
        if y.shape[:2]!=(b,t) or prop.shape[:2]!=(b,t):
            raise ValueError('native predictor time/batch alignment rejected')
        return y.reshape(b,t,*self.visual_shape),prop
    def edge(self,s,a):
        y,_=self._predict(s.reshape(len(s),1,*self.visual_shape),a[:,None],self.last_proprio)
        return y[:,-1].reshape(len(s),-1)
    def begin(self,s):
        return (s.reshape(len(s),1,*self.visual_shape),self.last_proprio,
                s.new_empty((len(s),0,self.action_dim)))
    def advance(self,state,a):
        import torch
        z,p,history=state
        acts=torch.cat((history,a[:,None]),dim=1)
        y,prop=self._predict(z,acts,p)
        next_visual=y[:,-1:]
        z=torch.cat((z,next_visual),dim=1)[:,-self.history_size:]
        p=torch.cat((p,prop[:,-1:]),dim=1)[:,-self.history_size:]
        history=acts[:,-(self.history_size-1):] if self.history_size>1 else acts[:,:0]
        return (z,p,history),next_visual[:,0].reshape(len(a),-1)
