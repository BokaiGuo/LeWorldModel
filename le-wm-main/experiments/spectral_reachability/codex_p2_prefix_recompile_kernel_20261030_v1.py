"""Fixed per-prefix recompiled chain vs joint search seam, not a launcher.

Callbacks must supply the real metered CUDA adapter, source identity and a
durable event sink. No result in this module is physical or a scientific gate.
"""
from copy import deepcopy
import math
import time

import numpy as np

from experiments.spectral_reachability import codex_p2_program_library_pair_20261030_v1 as old

NAME='codex_p2_prefix_recompile_kernel_20261030_v1'
CAPS=dict(packet=40,action_encoder=40,predict=40,predictor=40,pred_proj=40,
          jacobian=5,svd=5,score=17)
ARMS=('joint','prefix_chain')


def contract():
    return dict(schema=NAME+'.contract',arms=list(ARMS),horizon=10,packets=2,
        candidates_per_arm=16,first_prefixes=4,independent_suffixes_per_prefix=4,
        caps=deepcopy(CAPS),wall_seconds=450.0,
        joint='existing first eight full20 right singular directions, both signs',
        prefix_chain='four root packet choices; each actual prefix has its own one-packet Jacobian and SVD, two local directions with both signs',
        selection='terminal latent mean squared error, lowest candidate index on ties',
        replay='whole selected two-packet winner once from original context',
        budget='common upper bounds per arm, not required equal utilization or equal FLOPs',
        order=[[base,arm] for base in range(4) for arm in ARMS],
        known_roots=True,preregistered=False,after_h25_results_known=True,
        dynamic_source_closure_proven=False,equal_total_compute=False,
        total_flops=None,energy=None,engine_backward_calls=None,
        P1_admission=False,P2_admission=False,physical=False)


class Budget:
    def __init__(self,emit,clock=time.monotonic):
        old.require(callable(emit) and callable(clock),'callbacks')
        self.emit=emit;self.clock=clock;self.start=self.last=clock()
        self.attempted=dict.fromkeys(CAPS,0);self.returned=dict.fromkeys(CAPS,0)
        self.failed=False;self.depth=0

    def tick(self):
        now=self.clock()
        old.require(math.isfinite(now) and now>=self.last,'monotonic clock')
        self.last=now
        return now-self.start

    def call(self,kind,fn,details=None):
        elapsed=self.tick()
        old.require(not self.failed and kind in CAPS and self.attempted[kind]<CAPS[kind]
                    and elapsed<contract()['wall_seconds'] and callable(fn),'budget predispatch')
        n=self.attempted[kind]+1
        self.emit(dict(kind='resource_attempt',resource=kind,ordinal=n,depth=self.depth,
                       elapsed_seconds=elapsed,details=details))
        self.attempted[kind]=n;self.depth+=1
        try:
            result=fn()
        except BaseException as exc:
            self.failed=True
            self.emit(dict(kind='resource_failure',resource=kind,ordinal=n,depth=self.depth-1,
                           error_type=type(exc).__name__,message=str(exc)[:1000]))
            raise
        finally:
            self.depth-=1
        elapsed=self.tick();self.returned[kind]+=1
        self.emit(dict(kind='resource_return',resource=kind,ordinal=n,depth=self.depth,
                       elapsed_seconds=elapsed))
        if elapsed>=contract()['wall_seconds']:
            self.failed=True
            raise RuntimeError('in-flight wall limit')
        return result

    def call_extra(self,kind,fn):
        return self.call(kind,fn)

    def snapshot(self):
        return dict(attempted=deepcopy(self.attempted),returned=deepcopy(self.returned),
                    wall_seconds=self.tick(),failed=self.failed)


def local_basis(jac):
    j=np.asarray(jac,dtype=np.float64)
    old.require(j.shape==(192,10) and np.isfinite(j).all(),'local one-packet Jacobian')
    _,s,v=np.linalg.svd(j,full_matrices=False)
    for row in v:
        if row[np.argmax(np.abs(row))]<0:
            row*=-1
    return s,v


def run_branch(case,arm,adapter_factory,emit,model_scope,clock=time.monotonic):
    old.require(arm in ARMS and type(case['base']) is int and case['base'] in range(4)
        and callable(adapter_factory) and callable(emit) and callable(model_scope),'fixed branch and callbacks')
    initial=deepcopy(case['context']);goal=np.asarray(case['goal'],dtype=np.float64)
    old.require(goal.shape==(192,) and np.isfinite(goal).all(),'goal shape')
    def event(e):
        emit(dict(base=case['base'],arm=arm,event=e))
    budget=Budget(event,clock)
    event(dict(kind='branch_start',context_sha256=old.digest(initial),goal_sha256=old.digest(goal.tolist()),caps=CAPS))
    with model_scope(budget):
        adapter=adapter_factory(budget)
        root=adapter.root_jacobian(initial)
        root_singular,basis=budget.call('svd',lambda:old.canonical_basis(root),dict(scope='root',shape=[192,20]))
        library,banks=old.candidates(basis)
        if arm=='joint':
            result=old.execute_arm(initial,goal,'joint',banks['joint'],adapter.step,event,
                                   score_call=lambda fn:budget.call('score',fn))
            candidates=result['candidates'];winner=result['winner'];replay=result['replay']
            contexts=[]
        else:
            candidates=[];contexts=[]
            for p,first in enumerate(library[0]):
                first_raw=np.frombuffer(bytes.fromhex(first['raw_float32_hex']),dtype='<f4').reshape(5,2).copy()
                mid=adapter.step(deepcopy(initial),first_raw)
                mid_hash=old.digest(mid)
                event(dict(kind='prefix_return',prefix_index=p,program_id=first['program_id'],
                    raw_float32_hex=first['raw_float32_hex'],context_sha256=mid_hash))
                jac=adapter.packet_jacobian(deepcopy(mid))
                singular,local=budget.call('svd',lambda:local_basis(jac),dict(scope='prefix',prefix_index=p,shape=[192,10]))
                contexts.append(dict(prefix_index=p,context=deepcopy(mid),context_sha256=mid_hash,
                    jacobian=np.asarray(jac,dtype=np.float64).tolist(),
                    singular_values=singular.tolist(),basis=local.tolist()))
                for mode in range(2):
                    for sign in (-1,1):
                        logits=sign*(old.CAP/math.sqrt(2))*local[mode]
                        second=np.tanh(logits).astype('<f4').reshape(5,2)
                        final=adapter.step(deepcopy(mid),second)
                        z=np.asarray(final['latent_history'][-1],dtype=np.float64)
                        old.require(z.shape==(192,) and np.isfinite(z).all(),'terminal latent')
                        cost=budget.call('score',lambda:float(np.mean((z-goal)**2)))
                        old.require(math.isfinite(cost),'finite score')
                        item=dict(index=len(candidates),prefix_index=p,mode=mode,sign=sign,
                            first_raw_float32_hex=first_raw.tobytes().hex(),
                            second_raw_float32_hex=second.tobytes().hex(),prefix_context_sha256=mid_hash,
                            terminal_context_sha256=old.digest(final),terminal_context=deepcopy(final),cost=cost)
                        event(dict(kind='candidate_return',candidate=deepcopy(item)))
                        candidates.append(item)
            winner=min(candidates,key=lambda x:(x['cost'],x['index']))
            a=np.frombuffer(bytes.fromhex(winner['first_raw_float32_hex']),dtype='<f4').reshape(5,2).copy()
            b=np.frombuffer(bytes.fromhex(winner['second_raw_float32_hex']),dtype='<f4').reshape(5,2).copy()
            mid=adapter.step(deepcopy(initial),a);end=adapter.step(deepcopy(mid),b)
            z=np.asarray(end['latent_history'][-1],dtype=np.float64)
            cost=budget.call('score',lambda:float(np.mean((z-goal)**2)))
            old.require(old.digest(mid)==winner['prefix_context_sha256'] and
                old.digest(end)==winner['terminal_context_sha256'] and cost==winner['cost'],
                'whole winner replay exact')
            replay=dict(prefix_context_sha256=old.digest(mid),terminal_context_sha256=old.digest(end),cost=cost)
            event(dict(kind='winner_replayed',index=winner['index'],replay=replay))
    costs=budget.snapshot()
    old.require(not costs['failed'] and costs['attempted']==costs['returned'],'all dispatched calls returned')
    result=dict(schema=NAME+'.branch',base=case['base'],arm=arm,
        context_sha256=old.digest(initial),goal_sha256=old.digest(goal.tolist()),
        root_jacobian=np.asarray(root,dtype=np.float64).tolist(),
        root_singular_values=root_singular.tolist(),root_basis=basis.tolist(),
        candidates=candidates,winner=winner,replay=replay,prefix_contexts=contexts,
        resource_counts=costs,P2_admission=False,physical=False)
    event(dict(kind='branch_complete',result_sha256=old.digest(result),resource_counts=costs))
    return result
