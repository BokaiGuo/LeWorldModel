"""Normalized history3 target tracker core. RAM fixtures are never HOST evidence.

No loader, proposals, raw inverse, physics, or implicit search configuration.
MAIN owns authenticated construction, input lineage and production scheduling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from experiments.spectral_reachability import agent_hilewm_normalized_transition_20261029_v1 as low
from experiments.spectral_reachability import agent_hilewm_differentiable_low_20261029_v2 as diff

TAG = 'agent_hilewm_history3_target_tracker_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
CEM = ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/solver/cem.py'
UTILS = CEM.parent/'utils.py'
PINS = {
    CEM: 'a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b',
    UTILS: 'd30c6225a9c4c83782132180a94c94188b07e0381d2c938c658fe5a49aed48e1',
    Path(low.__file__): '56c88a29048833546654a47257e2a34b681d103f66bfd94ccdedb0bd4763c1b2',
    Path(diff.__file__): '7b10f644320a9c55d4805a3414901cbbc7472e3505baa03c5b2bb360820f713a',
}
KEYS = ('candidates', 'costs', 'topk_vals', 'topk_inds', 'topk_candidates',
        'mean', 'var', 'prev_mean', 'prev_var')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(b): return hashlib.sha256(b).hexdigest()


def source_closure():
    paths = dict(PINS)
    for p in (Path(__file__), ROOT/'tests'/('test_'+TAG+'.py'), ROOT/'innerdocs'/(TAG+'.md')):
        paths[p] = None
    site=ROOT/'.venv/lib/python3.12/site-packages'
    for rel in ('stable_worldmodel/protocols.py','stable_worldmodel/solver/solver.py',
                'torch/nn/functional.py','torch/nn/attention/__init__.py','torch/autograd/grad_mode.py',
                'torch/nn/modules/module.py','torch/_tensor.py'):
        paths[site/rel]=None
    out = {}
    for p, expected in paths.items():
        need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 4*1024**2, 'bounded regular source')
        h = digest(p.read_bytes())
        need(expected is None or expected == h, 'frozen source: '+str(p))
        out[str(p.resolve())] = h
    out.update(diff.source_closure())
    low.verify_sources()
    return out


@dataclass(frozen=True)
class Spec:
    horizon: int
    samples: int
    iterations: int
    topk: int
    var_scale: float
    seed: int
    segments: tuple
    initialization: str
    action_domain: str
    objective: str
    predict_api_cap: int
    predict_sample_cap: int
    array_bytes_cap: int
    wall_seconds: float

    def checked(self):
        for name, lo, hi in (('horizon', 1, 16), ('samples', 2, 64), ('iterations', 1, 128),
                            ('predict_api_cap', 1, 100000), ('predict_sample_cap', 1, 10000000),
                            ('array_bytes_cap', 1024, 256*1024**2)):
            x = getattr(self, name); need(type(x) is int and lo <= x <= hi, name)
        need(type(self.topk) is int and 2 <= self.topk <= self.samples, 'sample std requires topk >=2')
        need(type(self.seed) is int and 0 <= self.seed < 2**63, 'explicit seed')
        for name in ('var_scale', 'wall_seconds'):
            x = getattr(self, name)
            need(type(x) in (int, float) and math.isfinite(x) and x > 0, name)
        need(self.wall_seconds <= 900, 'bounded wall <=900s')
        need(type(self.segments) in (tuple, list) and self.segments and
             all(type(v) is int and v > 0 for v in self.segments) and sum(self.segments) == self.horizon,
             'explicit positive full partition')
        need(self.initialization == 'full_horizon_zero_no_witness', 'no witness or actor warmstart')
        need(self.action_domain == 'unclipped_normalized_gaussian', 'no implied Box clamp/raw domain')
        need(self.objective == 'identity192_terminal_meanMSE_float64', 'fixed metric, no Q fit')
        return self

    def planned(self):
        self.checked(); h, n, it = self.horizon, self.samples, self.iterations
        return dict(predict_API=h*(it+2), predictor_samples=h*(n*it+2),
                    candidate_cost_calls=it, candidate_trajectories=n*it,
                    mean_paid_replays=2, replay_trajectories=2,
                    action_encoder_calls=h*(it+2), low_projection_calls=h*(it+2),
                    action_encoder_tokens=(n*it+2)*sum(3+i for i in range(h)),
                    low_predictor_context_tokens=3*h*(n*it+2),
                    low_projection_rows=3*h*(n*it+2),
                    AD_primal=0, J=0, VJP=0, backward=0,
                    encoder=0, physics=0, FLOPs=None)


def spec_from_dict(d):
    need(type(d) is dict and set(d) == set(Spec.__dataclass_fields__), 'all spec fields explicit; no extras/defaults')
    return Spec(**d).checked()


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C')


class Trace:
    """Full arrays plus failed attempts; array payload cap, not a file-output cap."""
    def __init__(self, spec, backend, capture):
        self.spec, self.backend, self.capture = spec.checked(), backend, capture
        self.started = time.monotonic(); self.arrays = {}; self.used = 0
        self.capture_wall = 0.; self.events = []; self.api = self.samples = 0
        self.report = dict(schema=TAG+'.trace', backend=backend, spec=asdict(spec), completed=False,
                           errors=[], events=self.events, searches=[], replays=[],
                           raw_inverse=False, scientific_admission=False, real_meter_authenticated=False)

    def guard(self):
        need(time.monotonic()-self.started <= self.spec.wall_seconds, 'cooperative whole core wall')

    def save(self, value):
        self.guard(); start = time.monotonic()
        a = self.capture(value)
        need(type(a) is np.ndarray and not a.dtype.hasobject and np.isfinite(a).all(), 'finite snapshot')
        need(self.used+a.nbytes <= self.spec.array_bytes_cap, 'array payload cap before persist')
        name = 'array_'+str(len(self.arrays)); self.arrays[name] = a.copy(order='K')
        self.used += a.nbytes; self.capture_wall += time.monotonic()-start
        return name

    def call(self, kind, args, fn):
        e = dict(index=len(self.events), kind=kind, returned=False, dispatched=False,
                 module_returned=False, budget_debited=False,
                 shapes=[list(a.shape) for a in args])
        self.events.append(e); start = time.monotonic()
        try:
            self.guard()
            if kind == 'low_predictor':
                rows = int(args[0].shape[0])
                need(self.api+1 <= self.spec.predict_api_cap and self.samples+rows <= self.spec.predict_sample_cap,
                     'predispatch predict API/sample cap')
                self.api += 1; self.samples += rows; e['budget_debited']=True
            e['inputs'] = [self.save(a) for a in args]
            e['dispatched'] = True
            value = fn(); e['module_returned'] = True; e['output'] = self.save(value)
            e['returned'] = True
            return value
        except BaseException as ex:
            e['error'] = type(ex).__name__+': '+str(ex)
            raise
        finally:
            e['wall_including_capture_sync'] = time.monotonic()-start

    def finish(self):
        modules={}
        for kind in ('action_encoder','low_predictor','low_pred_proj'):
            ee=[e for e in self.events if e['kind']==kind]
            dispatched=[e for e in ee if e['dispatched']]
            modules[kind]=dict(attempts=len(ee),dispatched=len(dispatched),
                module_returns=sum(e['module_returned'] for e in ee),
                captured_returns=sum(e['returned'] for e in ee),
                rows_dispatched=sum(e['shapes'][0][0] for e in dispatched),
                tokens_dispatched=sum(e['shapes'][0][0]*e['shapes'][0][1] for e in dispatched
                                      if len(e['shapes'][0])==3))
        self.report.update(predict_API_debited=self.api, predictor_samples_debited=self.samples,
            array_payload_bytes=self.used, capture_sync_wall_seconds=self.capture_wall,
            whole_core_wall_seconds=time.monotonic()-self.started,
            arrays={k: dict(shape=list(a.shape), dtype=a.dtype.str, sha256=digest(a.tobytes(order='C')))
                    for k,a in self.arrays.items()},
            modules=modules,
            resources=dict(J=0, VJP=0, backward=0, physics=0, encoder=0, FLOPs=None,
                           AD_primal=0))
        return self.report, self.arrays


def traced_kernels(k, trace):
    def wrap(kind, fn): return lambda *xs: trace.call(kind, xs, lambda: fn(*xs))
    return low.Kernels(k.cat, k.stack, k.check, wrap('action_encoder', k.action_encoder),
                       wrap('low_predictor', k.low_predictor), wrap('low_pred_proj', k.low_pred_proj), k.kind)


def save_roll(trace, r):
    return dict(prediction=trace.save(r.prediction),
                checkpoints=[dict(z=trace.save(c.z), a=trace.save(c.a)) for c in r.contexts],
                final=dict(z=trace.save(r.final.z), a=trace.save(r.final.a)))


def cost_mean64(prediction, target, ops):
    # No learned metric; conversion happens before subtraction/squaring.
    return ops.mean64(prediction[:, :, -1, :], target)


class Engine:
    def __init__(self, z, a, target, spec, kernels, ops, trace):
        self.z, self.a, self.target, self.spec = z, a, target, spec
        self.k, self.ops, self.trace = traced_kernels(kernels, trace), ops, trace

    def evaluate(self, future, phase, segmented=False):
        tr, ops, s = self.trace, self.ops, self.spec
        row = dict(phase=phase, returned=False, future=tr.save(future)); tr.report['searches' if phase=='candidate' else 'replays'].append(row)
        shape = tuple(future.shape)
        need(shape[0] == 1 and shape[2:] == (s.horizon, 10) and
             shape[1] == (s.samples if phase == 'candidate' else 1), 'explicit B1/S/H/10')
        z, a = ops.expand(self.z, shape[1]), ops.expand(self.a, shape[1])
        goal = ops.expand(self.target, shape[1])
        row.update(context_z=tr.save(z), context_a=tr.save(a), target=tr.save(goal), first_event=len(tr.events))
        r = (low.segmented_rollout(z,a,future,segments=s.segments,kernels=self.k) if segmented else
             low.rollout(z,a,future,kernels=self.k))
        cost = cost_mean64(r.prediction, goal, ops)
        row.update(rollout=save_roll(tr,r), cost=tr.save(cost), end_event=len(tr.events), returned=True)
        return r, cost


def check_inputs(z, a, target, ops):
    for x, shape in ((z,(1,3,192)), (a,(1,2,10)), (target,(1,192))):
        ops.check(x); need(tuple(x.shape)==shape, 'original history3/a2/target192; no fabricated history')


def execute(z, a, target, spec, kernels, ops, solve, backend):
    spec.checked(); check_inputs(z,a,target,ops)
    tr = Trace(spec, backend, ops.capture)
    try:
        tr.report['inputs'] = dict(z=tr.save(z), a=tr.save(a), target=tr.save(target))
        engine = Engine(z,a,target,spec,kernels,ops,tr)
        mean, var, elite_cost = solve(engine, tr)
        tr.report['selected'] = dict(mean=tr.save(mean), var=tr.save(var), elite_average_not_mean_cost=tr.save(elite_cost))
        future = ops.sample_axis(mean)
        w,cw = engine.evaluate(future, 'mean_whole')
        seg,cs = engine.evaluate(future, 'mean_segmented', segmented=True)
        low.check_replay(w,seg,exact=ops.exact)
        need(ops.exact(cw,cs), 'mean replay costs bitwise')
        need(all(exact(tr.arrays[tr.report['inputs'][k]], ops.capture(v))
                 for k,v in (('z',z),('a',a),('target',target))), 'input mutation')
        tr.guard()
        tr.report.update(completed=True, selected_mean_replay_exact=True)
    except Exception as ex:
        tr.report['errors'].append(type(ex).__name__+': '+str(ex))
    return tr.finish()


class RAMOps:
    def check(self,x):
        need(type(x) is np.ndarray and x.dtype==np.float32 and np.isfinite(x).all(), 'RAM float32 array')
    def capture(self,x): return np.asarray(x).copy()
    def expand(self,x,s): return np.broadcast_to(x[:,None],(1,s,*x.shape[1:]))
    def sample_axis(self,x): return x[:,None]
    def mean64(self,x,y): return np.mean((x.astype(np.float64)-y.astype(np.float64))**2,axis=-1)
    exact = staticmethod(exact)


def track_fixture(z,a,target,*,spec,kernels):
    """Literal CEM arithmetic fixture, NOT installed torch execution/model evidence."""
    need(kernels.kind=='ram_fixture', 'only RAM callbacks')
    def solve(engine,tr):
        rng = np.random.default_rng(spec.seed)
        mean=np.zeros((1,spec.horizon,10),np.float32); var=np.full_like(mean,spec.var_scale)
        for step in range(spec.iterations):
            # Same frozen math/order; NumPy RNG/reductions are NOT CUDA parity.
            candidates=rng.standard_normal((1,spec.samples,spec.horizon,10),dtype=np.float32)*var[:,None]+mean[:,None]
            candidates[:,0]=mean
            _,cost=engine.evaluate(candidates,'candidate')
            inds=np.argsort(cost,axis=1,kind='stable')[:,:spec.topk]
            elite=candidates[np.arange(1)[:,None],inds]
            nxt=elite.mean(axis=1,dtype=np.float32); std=elite.std(axis=1,ddof=1,dtype=np.float32)
            state=dict(candidates=candidates,costs=cost,topk_vals=np.take_along_axis(cost,inds,axis=1),
                       topk_inds=inds,topk_candidates=elite,mean=nxt,var=std,prev_mean=mean,prev_var=var)
            tr.report.setdefault('cem',[]).append(dict(step=step, arrays={k:tr.save(v) for k,v in state.items()}))
            mean,var=nxt,std
        return mean,var,state['topk_vals'].mean(axis=1)
    return execute(z,a,target,spec,kernels,RAMOps(),solve,'ram_fixture_not_production')


def raw_inverse(*args, **kwargs):
    raise ValueError('raw inverse unavailable: no authenticated original frozen scaler; no fit or borrowed constants')


def validate_saved(report, arrays):
    """Full local saved wiring/score verification, no model/CEM import or rerun.

    The HOST's same-device exact CEM update checks are source-bound witnesses,
    not an independent saved CUDA reduction or RNG/gradient reconstruction.
    """
    need(report['schema']==TAG+'.trace', 'trace schema')
    s=spec_from_dict(report['spec'])
    need(report['completed'] is True and report['errors']==[], 'partial remains partial')
    need(report['backend'] in ('ram_fixture_not_production','host_cuda_installed_CEM'), 'explicit backend')
    need(set(arrays)==set(report['arrays']), 'full array inventory')
    for name,a in arrays.items():
        need(type(a) is np.ndarray and not a.dtype.hasobject and np.isfinite(a).all(), 'finite saved array')
        need(report['arrays'][name]==dict(shape=list(a.shape),dtype=a.dtype.str,sha256=digest(a.tobytes(order='C'))), 'saved array hash/header')
    need(sum(x.nbytes for x in arrays.values())==report['array_payload_bytes']<=s.array_bytes_cap, 'actual payload count')
    refs=set()
    def scan(x):
        if type(x) is str and x.startswith('array_'): refs.add(x)
        elif type(x) is list:
            for v in x: scan(v)
        elif type(x) is dict:
            for v in x.values(): scan(v)
    scan({k:v for k,v in report.items() if k!='arrays'})
    need(refs==set(arrays), 'all snapshots referenced; no missing arrays')
    def A(name): return arrays[name]
    def E(a,b,msg): need(exact(a,b),msg)
    z,a,target=(A(report['inputs'][k]) for k in ('z','a','target'))
    check_inputs(z,a,target,RAMOps())
    events=report['events']; cursor=0
    calls=report['searches']+report['replays']
    need(len(report['searches'])==s.iterations and len(report['replays'])==2, 'all candidates and two paid replays')
    for i,row in enumerate(calls):
        expected_phase='candidate' if i<s.iterations else ('mean_whole' if i==s.iterations else 'mean_segmented')
        need(row['returned'] is True and row['phase']==expected_phase and row['first_event']==cursor, 'phase order/attempt return')
        n=s.samples if i<s.iterations else 1
        f=A(row['future']); need(f.shape==(1,n,s.horizon,10) and f.dtype==np.float32,'full normalized future')
        zz=np.broadcast_to(z[:,None],(1,n,3,192)); aa=np.broadcast_to(a[:,None],(1,n,2,10))
        goal=np.broadcast_to(target[:,None],(1,n,192))
        E(A(row['context_z']),zz,'same history'); E(A(row['context_a']),aa,'same action history')
        E(A(row['target']),goal,'actual delivered target'); zz=zz.reshape(n,3,192); aa=aa.reshape(n,2,10)
        r=row['rollout']; need(len(r['checkpoints'])==s.horizon,'complete checkpoint denominator'); ys=[]
        for j,c in enumerate(r['checkpoints']):
            need(cursor+3<=len(events),'all module events'); ee=events[cursor:cursor+3]
            need([e['kind'] for e in ee]==['action_encoder','low_predictor','low_pred_proj'], 'source order')
            for k,e in enumerate(ee):
                need(e['index']==cursor+k and e['dispatched'] is True and e['module_returned'] is True and e['returned'] is True,
                     'complete actual module return')
                need(e['shapes']==[list(A(v).shape) for v in e['inputs']], 'event shape witness')
            aa=np.concatenate((aa,f[:,:,j,:].reshape(n,1,10)),axis=1)
            need(len(ee[0]['inputs'])==1 and len(ee[1]['inputs'])==2 and len(ee[2]['inputs'])==1, 'module arity')
            E(A(ee[0]['inputs'][0]),aa,'append outgoing before AE'); emb=A(ee[0]['output'])
            need(emb.shape==(n,j+3,192) and emb.dtype==np.float32,'full action embedding')
            E(A(ee[1]['inputs'][0]),zz[:,-3:],'last3 latent')
            E(A(ee[1]['inputs'][1]),emb[:,-3:],'last3 AE delivery')
            hidden=A(ee[1]['output']); need(hidden.shape==(n,3,192) and hidden.dtype==np.float32,'predictor3slots')
            E(A(ee[2]['inputs'][0]),hidden.reshape(-1,192),'flatten projection delivery')
            proj=A(ee[2]['output']); need(proj.shape==(n*3,192) and proj.dtype==np.float32,'projection3rows')
            y=proj.reshape(n,3,192)[:,-1:]; zz=np.concatenate((zz,y),axis=1); ys.append(y[:,0])
            E(A(c['z']),zz,'full z history append'); E(A(c['a']),aa,'full a history append')
            cursor+=3
        E(A(r['prediction']),np.stack(ys,axis=1).reshape(1,n,s.horizon,192),'all H predictions')
        E(A(r['final']['z']),zz,'final z'); E(A(r['final']['a']),aa,'final a')
        need(row['end_event']==cursor, 'exact event slice')
        actual=A(row['cost']); need(actual.shape==(1,n) and actual.dtype==np.float64,'float64 identity meanMSE')
        pred=A(r['prediction'])[0,:,-1,:].astype(np.float64)
        for sample in range(n):
            delta=pred[sample]-target[0].astype(np.float64)
            ref=math.fsum(float(v)*float(v) for v in delta)/192
            need(abs(float(actual[0,sample])-ref)<=8*192*np.finfo(np.float64).eps*max(abs(ref),np.finfo(np.float64).tiny),
                 'independent float64 goal meanMSE arithmetic')
    need(cursor==len(events), 'no extra events')
    expected=s.planned()
    need(report['predict_API_debited']==expected['predict_API'] and
         report['predictor_samples_debited']==expected['predictor_samples'], 'full paid budget incl replays')
    need(expected['predict_API']<=s.predict_api_cap and expected['predictor_samples']<=s.predict_sample_cap,'cap')
    for kind in ('action_encoder','low_predictor','low_pred_proj'):
        ee=[e for e in events if e['kind']==kind]
        mm=dict(attempts=len(ee),dispatched=len(ee),module_returns=len(ee),captured_returns=len(ee),
                rows_dispatched=sum(e['shapes'][0][0] for e in ee),
                tokens_dispatched=sum(e['shapes'][0][0]*e['shapes'][0][1] for e in ee if len(e['shapes'][0])==3))
        need(report['modules'][kind]==mm,'actual module axes separate from prediction debit')
    need(all(e['budget_debited'] is (e['kind']=='low_predictor') for e in events),'predict-only cap debit')
    mean=A(report['selected']['mean']); var=A(report['selected']['var'])
    need(mean.shape==var.shape==(1,s.horizon,10) and mean.dtype==var.dtype==np.float32,'selected full mean/var')
    for row in report['replays']: E(A(row['future']),mean[:,None],'actual returned mean replay, not best elite')
    w,seg=report['replays']
    for key in ('prediction',): E(A(w['rollout'][key]),A(seg['rollout'][key]),'whole/seg exact predictions')
    for c,d in zip(w['rollout']['checkpoints'],seg['rollout']['checkpoints'],strict=True):
        for key in ('z','a'): E(A(c[key]),A(d[key]),'complete exact replay seam')
    E(A(w['cost']),A(seg['cost']),'exact replay costs')
    need(len(report['cem'])==s.iterations,'full CEM iterations')
    prevmean=np.zeros_like(mean); prevvar=np.full_like(var,s.var_scale)
    for i,e in enumerate(report['cem']):
        need(e['step']==i and set(e['arrays'])==set(KEYS),'CEM callback ABI/order')
        v={k:A(h) for k,h in e['arrays'].items()}
        E(v['candidates'],A(report['searches'][i]['future']),'actual CEM candidates delivered')
        E(v['costs'],A(report['searches'][i]['cost']),'actual CEM costs')
        E(v['prev_mean'],prevmean,'mean recurrence'); E(v['prev_var'],prevvar,'std recurrence')
        E(v['candidates'][:,0],prevmean,'forced mean candidate')
        inds=v['topk_inds']; need(inds.shape==(1,s.topk) and np.issubdtype(inds.dtype,np.integer) and
            len(set(inds[0].tolist()))==s.topk and np.all((inds>=0)&(inds<s.samples)),'valid unique elites')
        E(v['topk_candidates'],v['candidates'][np.arange(1)[:,None],inds],'elite gather')
        E(v['topk_vals'],np.take_along_axis(v['costs'],inds,axis=1),'elite costs')
        need(np.all(np.diff(v['topk_vals'],axis=1)>=0) and
             np.all(v['topk_vals']<=np.sort(v['costs'],axis=1)[:,:s.topk]),'topk values, ties not reselected')
        if report['backend']=='ram_fixture_not_production':
            E(v['mean'],v['topk_candidates'].mean(axis=1,dtype=np.float32),'RAM native mean')
            E(v['var'],v['topk_candidates'].std(axis=1,ddof=1,dtype=np.float32),'RAM unbiased std')
        else:
            need(e['cuda_exact_checks']==dict(topk=True,gather=True,mean=True,std=True,first_is_mean=True), 'actual CUDA update witnesses')
        prevmean,prevvar=v['mean'],v['var']
    E(mean,prevmean,'final actual mean'); E(var,prevvar,'final actual std')
    elite=A(report['selected']['elite_average_not_mean_cost'])
    need(elite.shape==(1,) and np.allclose(elite,v['topk_vals'].mean(axis=1),rtol=8*192*np.finfo(np.float64).eps,atol=0), 'native elite average distinct from replay score')
    if report['backend']=='host_cuda_installed_CEM':
        need(report['model_before']==report['model_after'] and report['sources_before']==report['sources_after']==source_closure(), 'HOST model/source guards')
        need(len(report['inference_delivery'])==s.iterations,'all inference lifts')
        for i,e in enumerate(report['inference_delivery']):
            E(A(e['source']),A(e['delivered']),'inference lift exact bytes')
            E(A(e['delivered']),A(report['searches'][i]['future']),'lift delivered to kernel')
        need([report[k] for k in ('cem_reset','cem_start','cem_end')]==[1,1,1], 'one installed solve lifecycle')
        native=report['native_return']
        E(A(native['actions']),mean,'native CPU returned actions -> selected CUDA mean')
        E(A(native['mean']),mean,'native CPU returned mean')
        E(A(native['var']),var,'native CPU returned std')
    need(report['raw_inverse'] is False and report['scientific_admission'] is False,'no scientific/raw release')
    need(report['real_meter_authenticated'] is False and report['selected_mean_replay_exact'] is True,
         'local wiring witness is not upstream authority')
    need(report['resources']==dict(J=0,VJP=0,backward=0,physics=0,encoder=0,FLOPs=None,AD_primal=0),
         'no hidden AD/physics/encoder axes')
    return dict(schema=TAG+'.saved_check',backend=report['backend'],
                status='complete_local_trace_verified',predict_API=expected['predict_API'],
                predictor_samples=expected['predictor_samples'],all_mean_replays_verified=True,
                independent_CUDA_reduction_or_gradient_recompute=False,
                authority_certified=False,raw_inverse=False,scientific_admission=False)


def track_cuda(graph,z,a,target,*,spec,expected_state,main_authorized=False):
    """HOST integration function, not a loader or authority certificate.

    Installed CEM is unchanged. At each get_cost boundary, inference candidates
    are cloned under ordinary enabled-grad mode, with source/delivery witnesses.
    Facade never runs inside inference mode. No witness/init_action argument.
    """
    need(main_authorized is True, 'MAIN authorization before torch import/root access')
    entered=time.monotonic()
    sources=source_closure()
    import torch as t
    need(t.cuda.is_available() and not t.is_inference_mode_enabled() and t.is_grad_enabled(),
         'actual HOST CUDA, ordinary enabled grad; no CPU fallback')
    from stable_worldmodel.solver.cem import CEMSolver
    from gymnasium.spaces import Box
    need(Path(inspect.getfile(CEMSolver)).resolve()==CEM.resolve(), 'installed pinned CEM origin')
    spec.checked()
    facade=diff.wrap(graph,t,main_authorized=True)
    params=list(facade.parameters()); device=params[0].device
    need(not t.backends.cuda.matmul.allow_tf32 and not t.backends.cudnn.allow_tf32 and
         t.are_deterministic_algorithms_enabled() and
         not t.is_autocast_enabled('cuda'), 'MAIN configured deterministic/TF32off/autocastoff')
    def state_hashes():
        return {k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=digest(v.detach().contiguous().cpu().numpy().tobytes()))
                for k,v in facade.state_dict().items()}
    before=state_hashes(); need(before==expected_state and before, 'all frozen actual state values prebound')
    class Ops:
        def check(self,x):
            need(t.is_tensor(x) and x.device==device and x.dtype==t.float32 and not t.is_inference(x)
                 and not x.requires_grad and bool(t.isfinite(x).all()), 'actual finite ordinary frozen-input CUDA float32')
        def capture(self,x):
            need(t.is_tensor(x) and x.device.type in ('cuda','cpu'), 'real tensor snapshot')
            t.cuda.synchronize(device); return x.detach().cpu().numpy().copy()
        def expand(self,x,s): return x[:,None].expand(1,s,*x.shape[1:])
        def sample_axis(self,x): return x[:,None]
        def mean64(self,x,y): return (x.to(t.float64)-y.to(t.float64)).square().mean(dim=-1)
        def exact(self,x,y): return exact(self.capture(x),self.capture(y))
    ops=Ops(); k=low.cuda_kernels(t,facade,main_authorized=True)
    def solve(engine,tr):
        class Cost:
            def parameters(self): return facade.parameters()
            def get_cost(self,info,candidates):
                need(set(info)=={'z','a','target'}, 'only tensor solver metadata')
                with t.inference_mode(False), t.enable_grad():
                    row=dict(source=tr.save(candidates), source_inference=t.is_inference(candidates))
                    tr.report.setdefault('inference_delivery',[]).append(row)
                    # Clone actual sampled values, not reconstructed/re-normalized actions.
                    delivered=candidates.clone()
                    row['delivered']=tr.save(delivered)
                    need(ops.exact(candidates,delivered) and not t.is_inference(delivered), 'candidate value-preserving lift')
                    for key,original in (('z',z),('a',a),('target',target)):
                        need(ops.exact(info[key],ops.expand(original,spec.samples)), 'solver expanded context/target lineage')
                    _,cost=engine.evaluate(delivered,'candidate')
                    return cost
        class Recorder:
            output_key=TAG
            def __init__(self): self.history=[]
            def reset(self): tr.report['cem_reset']=tr.report.get('cem_reset',0)+1
            def start_batch(self): tr.report['cem_start']=tr.report.get('cem_start',0)+1
            def end_solve(self): tr.report['cem_end']=tr.report.get('cem_end',0)+1
            def __call__(self,**s):
                vals,inds=t.topk(s['costs'],spec.topk,dim=1,largest=False)
                elite=s['candidates'][t.arange(1,device=device)[:,None],inds]
                checks=dict(topk=t.equal(vals,s['topk_vals']) and t.equal(inds,s['topk_inds']),
                            gather=t.equal(elite,s['topk_candidates']),
                            mean=t.equal(elite.mean(dim=1),s['mean']),
                            std=t.equal(elite.std(dim=1),s['var']),
                            first_is_mean=t.equal(s['candidates'][:,0],s['prev_mean']))
                event=dict(step=s['step'],arrays={k:tr.save(s[k]) for k in KEYS},cuda_exact_checks=checks)
                tr.report.setdefault('cem',[]).append(event); self.history.append(dict(step=s['step']))
                need(all(checks.values()), 'unchanged installed CUDA CEM updates')
        callback=Recorder()
        solver=CEMSolver(model=Cost(),batch_size=1,num_samples=spec.samples,var_scale=spec.var_scale,
                         n_steps=spec.iterations,topk=spec.topk,device=device,seed=spec.seed,callbacks=[callback])
        solver.configure(action_space=Box(-np.inf,np.inf,shape=(1,2),dtype=np.float32),n_envs=1,
                         config=SimpleNamespace(horizon=spec.horizon,action_block=5))
        initial=t.zeros((1,spec.horizon,10),device=device,dtype=t.float32)
        tr.report['initial']=tr.save(initial)
        out=solver.solve(dict(z=z,a=a,target=target),init_action=initial)
        need(t.equal(out['actions'],out['mean'][0]), 'actual actions are final mean')
        tr.report['native_callbacks']=out['callbacks'][TAG]
        tr.report['native_return']=dict(actions=tr.save(out['actions']),mean=tr.save(out['mean'][0]),var=tr.save(out['var'][0]))
        # Explicit ordinary tensors; never depend on inference .to alias semantics.
        selected=out['actions'].to(device).clone(); std=out['var'][0].to(device).clone()
        ops.check(selected); ops.check(std)
        need(ops.exact(selected,out['actions']) and ops.exact(std,out['var'][0]), 'actual mean/std value-preserving delivery')
        return (selected,std,
                t.tensor(out['costs'],dtype=t.float64,device=device))
    # Temporary official SDPA context, restored by PyTorch; no method/global patch.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        report,arrays=execute(z,a,target,spec,k,ops,solve,'host_cuda_installed_CEM')
    try:
        report['model_before']=before; report['model_after']=state_hashes()
        report['sources_before']=sources; report['sources_after']=source_closure()
        need(report['model_before']==report['model_after'] and sources==report['sources_after'], 'frozen model/source drift')
    except Exception as ex:
        report['completed']=False; report['errors'].append(type(ex).__name__+': '+str(ex))
    # Local tensor/byte/source checks only; not original loader or upstream certification.
    report['real_meter_authenticated']=False
    report['host_tensor_seam_checked']=True
    report['host_entry_through_postguard_wall_seconds']=time.monotonic()-entered
    report['wall_scope']='core includes CEM/capture/sync/2 replays; excludes initial/final source+model hash; outer host wall separately recorded'
    return report,arrays


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify-sources','validate-spec')); p.add_argument('--spec')
    args=p.parse_args(argv)
    if args.command=='verify-sources': out=dict(sources=source_closure(),production_run=False)
    else:
        need(args.spec is not None,'explicit spec file')
        path=Path(args.spec); need(path.stat().st_size<=65536,'small spec cap')
        s=spec_from_dict(json.loads(path.read_bytes())); out=dict(spec=asdict(s),planned=s.planned(),production_run=False)
    print(json.dumps(out,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
