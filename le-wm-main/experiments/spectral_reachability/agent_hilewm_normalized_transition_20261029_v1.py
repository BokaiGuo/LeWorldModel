"""Differentiable normalized Hi low seam. No runtime imports or model runner.

Explicit sample axes only; callbacks own tensor operations. RAM callback checks
are not CUDA numerical equivalence, original-class parity, or P2 pairing.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

TAG = 'agent_hilewm_normalized_transition_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'experiments/spectral_reachability/agent_hilewm_low_cem_20261029_v1.py':
        '09b33db464c4eb0f2181359bd9cc450ee9ff569d8e05007a37e9fc42b7f6b2bf',
    'results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__models__jepa.py.source.txt':
        'bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2',
}
WIDTH, LATENT, LOW_CTX, MAX_T = 10, 192, 3, 16


def need(ok, message):
    if not ok:
        raise ValueError(message)


def protocol():
    return dict(schema=TAG+'.protocol', input='normalized grouped actions only',
        z_hist='explicit [B,S,Lz,192]', a_hist='explicit [B,S,La,10]',
        future='explicit [B,S,T,10]', flat_step='[B*S,Lz,192], [B*S,La,10], outgoing [B*S,1,10]',
        bounds=dict(B=[1,5], S=[1,64], T=[1,MAX_T], initial_history=[1,MAX_T]),
        low_context=LOW_CTX, order=['append_outgoing', 'encode_full_action_history',
            'last_min_context', 'low_predictor', 'flatten_projection', 'last_append'],
        history='retain full z_seq/a_seq across all segments; never bootstrap or crop stored history',
        assembly='slice future[:,:,k,:].reshape(B*S,1,10); no action-list stack or contiguous conversion',
        autograd='caller-owned; no inference/no_grad/detach or input tensor reconstruction',
        resources='module attempts/returns/rows/tokens; whole and segmented charged independently',
        CUDA_numerical_equivalence=False, gradient_verified=False, P2_pairing=False,
        raw_inverse=False, physical_release=False, scientific_admission=False,
        original_class_parity=False, training=False)


def verify_sources():
    """Bounded source bytes only, not any saved model/full consumer."""
    rows=[]
    for name, expected in PINS.items():
        path=ROOT/name
        need(not path.is_symlink() and path.is_file() and path.stat().st_size<=256*1024,
             'regular bounded source: '+name)
        body=path.read_bytes()
        need(len(body)<=256*1024 and hashlib.sha256(body).hexdigest()==expected,
             'source SHA: '+name)
        rows.append(dict(path=str(path), bytes=len(body), sha256=expected))
    return dict(schema=TAG+'.source_check', sources=rows,
                source_bytes_verified=True, CUDA_numerical_equivalence=False)


@dataclass(frozen=True)
class FlatContext:
    """Live tensors, not serialized values. No copies or graph severing."""
    z: Any
    a: Any


@dataclass(frozen=True)
class Kernels:
    cat: Callable
    stack: Callable
    check: Callable
    action_encoder: Callable
    low_predictor: Callable
    low_pred_proj: Callable
    kind: str


@dataclass
class Ledger:
    """Callback-boundary counts, NOT authenticated HOST/CUDA evidence."""
    events: list = field(default_factory=list)

    def call(self, kind, inputs, callback, check):
        shapes=[list(x.shape) for x in inputs]
        e=dict(index=len(self.events), kind=kind, inputs=shapes,
               attempted=True, returned=False, error=None, output=None)
        self.events.append(e)
        try:
            value=callback(*inputs)
            check(value)
            e['output']=list(value.shape)
            e['returned']=True
            return value
        except Exception as exc:
            e['error']=type(exc).__name__+': '+str(exc)
            raise

    def summary(self):
        rows={}
        for kind in ('action_encoder','low_predictor','low_pred_proj'):
            es=[e for e in self.events if e['kind']==kind]
            rows[kind]=dict(attempts=len(es), returns=sum(e['returned'] for e in es),
                input_rows=sum(e['inputs'][0][0] for e in es),
                input_tokens=sum(e['inputs'][0][0]*e['inputs'][0][1]
                    for e in es if len(e['inputs'][0])==3))
        return dict(modules=rows, real_meter_authenticated=False, FLOPs=None,
                    backward=None, J=None, VJP=None, AD_primal=None)


@dataclass(frozen=True)
class Rollout:
    prediction: Any
    final: FlatContext
    contexts: tuple


def _shape(x, expected, check, message):
    check(x)
    need(tuple(x.shape)==tuple(expected), message)


def _context(c, k):
    need(isinstance(c,FlatContext), 'explicit flat context')
    k.check(c.z); k.check(c.a)
    need(len(c.z.shape)==len(c.a.shape)==3, 'flat rank3 context')
    n,lz,d=c.z.shape; na,la,w=c.a.shape
    need(n==na and 1<=n<=320 and d==LATENT and w==WIDTH and
         1<=lz<=2*MAX_T and 1<=la<=2*MAX_T, 'bounded full normalized context')
    return n,lz,la


def transition(context, outgoing, *, kernels, ledger=None):
    """One source-ordered step; outgoing is already a normalized grouped token.

    Callbacks must not mutate inputs/root. No clipping, inverse, padding,
    sample expansion, history normalization, CPU copy or gradient mode changes.
    """
    k=kernels
    need(k.kind in ('ram_fixture','host_cuda'), 'explicit kernel kind')
    n,lz,la=_context(context,k)
    need(lz<2*MAX_T and la<2*MAX_T, 'append fits bounded full history')
    _shape(outgoing,(n,1,WIDTH),k.check,'literal outgoing grouped token')
    log=ledger if ledger is not None else Ledger()
    a=k.cat((context.a,outgoing),dim=1)
    _shape(a,(n,la+1,WIDTH),k.check,'append outgoing before embedding')
    emb=log.call('action_encoder',(a,),k.action_encoder,k.check)
    _shape(emb,(n,la+1,LATENT),k.check,'kernel1 encoder preserves complete action time')
    ctx=min(LOW_CTX,lz,emb.shape[1])
    zi,ai=context.z[:,-ctx:],emb[:,-ctx:]
    h=log.call('low_predictor',(zi,ai),k.low_predictor,k.check)
    _shape(h,(n,ctx,LATENT),k.check,'literal low predictor shape')
    flat=h.reshape(-1,LATENT)
    p=log.call('low_pred_proj',(flat,),k.low_pred_proj,k.check)
    _shape(p,(n*ctx,LATENT),k.check,'flattened low projection shape')
    nxt=p.reshape(n,ctx,LATENT)[:,-1:]
    z=k.cat((context.z,nxt),dim=1)
    _shape(z,(n,lz+1,LATENT),k.check,'append last projected latent')
    return FlatContext(z,a),nxt[:,0]


def _start(z_hist,a_hist,future,k):
    for x in (z_hist,a_hist,future):k.check(x)
    need(len(z_hist.shape)==len(a_hist.shape)==len(future.shape)==4,
         'only explicit [B,S,time,width], no ambiguous rank2/3 or inferred zero history')
    b,s,t,w=future.shape; bz,sz,lz,d=z_hist.shape; ba,sa,la,aw=a_hist.shape
    need((b,s)==(bz,sz)==(ba,sa) and 1<=b<=5 and 1<=s<=64 and
         1<=t<=MAX_T and 1<=lz<=MAX_T and 1<=la<=MAX_T and
         w==aw==WIDTH and d==LATENT, 'bounded explicit sample/time/latent ABI')
    return FlatContext(z_hist.reshape(b*s,lz,LATENT),a_hist.reshape(b*s,la,WIDTH)),b,s,t


def _segment(c,future,b,s,k,log):
    predictions=[];contexts=[]
    for step in range(future.shape[2]):
        # Match official assembly; do not replace with stack(action_list).
        outgoing=future[:,:,step,:].reshape(b*s,1,WIDTH)
        c,y=transition(c,outgoing,kernels=k,ledger=log)
        contexts.append(c);predictions.append(y)
    return c,predictions,contexts


def rollout(z_hist,a_hist,future_actions,*,kernels,ledger=None):
    """Whole model-only rollout; returned histories keep their autograd graph."""
    c,b,s,t=_start(z_hist,a_hist,future_actions,kernels)
    c,p,contexts=_segment(c,future_actions,b,s,kernels,ledger if ledger is not None else Ledger())
    return Rollout(kernels.stack(p,dim=1).reshape(b,s,t,LATENT),c,tuple(contexts))


def segmented_rollout(z_hist,a_hist,future_actions,*,segments,kernels,ledger=None):
    """Independent replay with exact full-context seams; no reset between parts."""
    c,b,s,t=_start(z_hist,a_hist,future_actions,kernels)
    need(type(segments) in (tuple,list) and segments and
         all(type(x) is int and x>0 for x in segments) and sum(segments)==t,
         'positive exact partition; no truncation/padding')
    p=[];contexts=[];start=0
    log=ledger if ledger is not None else Ledger()
    for length in segments:
        c,ys,cs=_segment(c,future_actions[:,:,start:start+length,:],b,s,kernels,log)
        p.extend(ys);contexts.extend(cs);start+=length
    return Rollout(kernels.stack(p,dim=1).reshape(b,s,t,LATENT),c,tuple(contexts))


def check_replay(whole,segmented,*,exact):
    """Caller-supplied strict equality; no numerical tolerance or auto-detach.

    MAIN must choose/persist its CUDA value/layout/gradient checks separately.
    A fixture return here certifies only those fixture values.
    """
    need(len(whole.contexts)==len(segmented.contexts)>0, 'all checkpoint denominator')
    need(exact(whole.prediction,segmented.prediction), 'whole/segmented predictions')
    for a,b in zip(whole.contexts,segmented.contexts,strict=True):
        need(exact(a.z,b.z) and exact(a.a,b.a), 'every full context seam')
    need(exact(whole.final.z,segmented.final.z) and exact(whole.final.a,segmented.final.a),
         'final full context')
    return dict(all_saved_contexts_equal=True, comparison='caller-provided exact callback',
                CUDA_numerical_equivalence=False, gradient_verified=False)


def cuda_kernels(torch,root,*,main_authorized=False,trace=None):
    """Bind existing frozen CUDA nodes without importing torch or patching root.

    Not a loader/runner. MAIN supplies authenticated tensor-only owned root and
    runtime/source/array persistence. Authorizing this binding is not a cert.
    """
    need(main_authorized is True, 'MAIN authorization required before root access')
    verify_sources()
    need(not torch.is_inference_mode_enabled(), 'inference tensors forbidden for differentiable seam')
    need(all(not node.training for node in root.modules()), 'all existing nodes eval')
    params=list(root.parameters())
    need(params and all(not p.requires_grad and p.device.type=='cuda' for p in params),
         'frozen existing CUDA model only, no CPU fallback')
    device=params[0].device
    need(all(p.device==device for p in params), 'one explicit CUDA device')
    buffers=list(root.buffers())
    need(all(p.device==device for p in buffers), 'buffers on same CUDA')
    need(tuple(root.low_predictor.pos_embedding.shape)==(1,LOW_CTX,LATENT),
         'literal existing low context3')
    n=root.action_encoder.patch_embed
    need(tuple(n.weight.shape)==(WIDTH,WIDTH,1) and n.stride==(1,) and
         n.padding==(0,) and n.dilation==(1,) and n.groups==1, 'literal action Conv1d1')
    def check(x):
        need(torch.is_tensor(x) and x.device==device and x.dtype==torch.float32 and
             not torch.is_inference(x), 'original CUDA float32, non-inference tensor')
        need(bool(torch.isfinite(x).all()), 'finite CUDA values')
    def call(kind,args,fn):
        return fn() if trace is None else trace.call(kind,args,fn)
    def action(a):
        return call('action_encoder',[a],lambda:root.action_encoder.embed(
            torch.nn.functional.conv1d(a.transpose(1,2),n.weight,n.bias).transpose(1,2)))
    def predict(z,a):
        return call('low_predictor',[z,a],lambda:root.low_predictor(z,a))
    def project(h):
        return call('low_pred_proj',[h],lambda:root.low_pred_proj(h))
    return Kernels(torch.cat,torch.stack,check,action,predict,project,'host_cuda')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('protocol','verify-sources'))
    args=p.parse_args()
    value=protocol() if args.command=='protocol' else verify_sources()
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
