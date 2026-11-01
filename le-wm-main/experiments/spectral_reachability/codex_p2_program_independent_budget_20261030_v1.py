"""Per-prefix independent library vs joint optimization with common ceilings.

This orchestration seam does not instantiate CUDA, acquire a lock, or execute
physics. The caller supplies a real CUDA adapter and a durable event sink.
"""
from copy import deepcopy
import math
import time

import numpy as np

from experiments.spectral_reachability import codex_p2_program_library_pair_20261030_v1 as kernel

NAME="codex_p2_program_independent_budget_20261030_v1"
CAPS=dict(packet=36,action_encoder=36,predict=36,predictor=36,pred_proj=36,
          jacobian=1,svd=1,score=17)
ARMS=kernel.ARMS


def contract():
    return dict(schema=NAME+".contract",order=[[base,arm] for base in range(4) for arm in ARMS],
        per_prefix_per_arm_caps=CAPS,wall_seconds=450.0,
        compilation="each prefix and arm independently recomputes root Jacobian and SVD",
        candidates="each arm separately constructs and evaluates its own fixed search bank",
        quota="upper bounds only; actual attempted and returned reported, no padding or shared credit",
        feedback="same authenticated per-prefix context and goal for both arms",
        failure="durable failure; no retry, no substitution, never interpret as success",
        known_roots=True,preregistered=False,equal_total_compute=False,
        total_flops=None,energy=None,engine_backward_calls=None,
        P1_admission=False,P2_admission=False,physical=False)


class Budget:
    def __init__(self,emit,clock=time.monotonic):
        kernel.require(callable(emit) and callable(clock),"required durable budget callbacks")
        self.emit,self.clock=emit,clock
        self.start=self.last=clock();self.attempted=dict.fromkeys(CAPS,0)
        self.returned=dict.fromkeys(CAPS,0);self.failed=False;self.depth=0

    def tick(self):
        now=self.clock()
        kernel.require(math.isfinite(now) and now>=self.last,"monotonic budget clock")
        self.last=now
        return now-self.start

    def call(self,kind,fn,details=None):
        elapsed=self.tick()
        kernel.require(not self.failed and kind in CAPS and self.attempted[kind]<CAPS[kind]
                       and elapsed<contract()["wall_seconds"] and callable(fn),"budget predispatch")
        ordinal=self.attempted[kind]+1
        self.emit(dict(kind="resource_attempt",resource=kind,ordinal=ordinal,
                       depth=self.depth,elapsed_seconds=elapsed,details=details))
        self.attempted[kind]=ordinal;self.depth+=1
        try:
            value=fn()
        except BaseException as exc:
            self.failed=True
            self.emit(dict(kind="resource_failure",resource=kind,ordinal=ordinal,
                           depth=self.depth-1,error_type=type(exc).__name__,message=str(exc)[:1000]))
            raise
        finally:
            self.depth-=1
        elapsed=self.tick();self.returned[kind]+=1
        self.emit(dict(kind="resource_return",resource=kind,ordinal=ordinal,
                       depth=self.depth,elapsed_seconds=elapsed))
        if elapsed>contract()["wall_seconds"]:
            self.failed=True
            raise RuntimeError("in-flight wall limit")
        return value

    def call_extra(self,kind,fn):
        return self.call(kind,fn)

    def snapshot(self):
        return dict(attempted=dict(self.attempted),returned=dict(self.returned),
                    wall_seconds=self.tick(),failed=self.failed)


def run_branch(case,arm,adapter_factory,emit,clock=time.monotonic,model_scope=None):
    """Recompile and optimize one arm; no other arm can share its accounting."""
    kernel.require(arm in ARMS and type(case["base"]) is int and case["base"] in range(4)
                   and callable(adapter_factory) and callable(emit),"fixed case and arm")
    initial=deepcopy(case["context"]);goal=deepcopy(case["goal"])
    def scoped(event):
        emit(dict(base=case["base"],arm=arm,event=event))
    budget=Budget(scoped,clock)
    scoped(dict(kind="branch_start",context_sha256=kernel.digest(initial),goal_sha256=kernel.digest(goal),
                caps=CAPS))
    # model_scope supplies wrappers for the actual action_encoder/predict/
    # predictor/pred_proj calls and must clean up even after failure.
    kernel.require(callable(model_scope),"real model entry instrumentation required")
    with model_scope(budget):
        adapter=adapter_factory(budget)
        jac=adapter.root_jacobian(initial)
        singular,basis=budget.call("svd",lambda:kernel.canonical_basis(jac),dict(shape=[192,20]))
        library,banks=kernel.candidates(basis)
        result=kernel.execute_arm(initial,goal,arm,banks[arm],adapter.step,scoped,
                                  score_call=lambda fn:budget.call("score",fn))
    costs=budget.snapshot()
    kernel.require(not costs["failed"] and costs["attempted"]==costs["returned"],
                   "completed attempts only; underuse is allowed")
    unique=len({entry["raw_float32_hex"] for entry in banks[arm]})
    value=dict(schema=NAME+".branch",base=case["base"],arm=arm,
        context_sha256=kernel.digest(initial),goal_sha256=kernel.digest(goal),
        jacobian=np.asarray(jac,dtype=np.float64).tolist(),singular_values=singular.tolist(),
        basis=basis.tolist(),library=library,result=result,unique_action_sequences=unique,
        resource_counts=costs,physical=False,P2_admission=False)
    scoped(dict(kind="branch_complete",result_sha256=kernel.digest(value),
                unique_action_sequences=unique,resource_counts=costs))
    return value
