"""Fixed reusable two-program library versus joint spectral candidates.

Pure search kernel. The caller must supply metered HOST CUDA transitions and
durable events; this module is not a production launcher or scientific gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import numpy as np

NAME = "codex_p2_program_library_pair_20261030_v1"
ARMS = ("joint", "library_chain")
CAP = .75 * math.sqrt(10)
PACKET = 5
COUNT = 16


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def contract():
    return dict(schema=NAME+".contract", horizon=10, packets=2, candidates_per_arm=COUNT,
                order=list(ARMS), primitive_library_per_position=4,
                library_selection="ordered Cartesian product; no feedback-dependent recompilation",
                joint_selection="first eight joint right singular directions, negative then positive",
                selector="stable first minimum terminal mean squared error",
                shared_root_jacobian=1, shared_root_svd=1, mathematical_vjp_rows=192,
                exclusive_packet_per_arm=34, exclusive_candidate_scores_per_arm=17,
                shared_compile_predict_calls="backend-dependent; separately measured",
                action_domain="tanh of joint20 logits, float32, joint L2 cap",
                replay="winner once from same initial feedback, exact raw and contexts",
                no_retries=True, known_roots=True, preregistered=False,
                equal_total_compute=False, total_flops=None, energy=None,
                P1_admission=False, P2_admission=False, physical_release=False)


def canonical_basis(jacobian):
    j=np.asarray(jacobian,dtype=np.float64)
    require(j.shape==(192,20) and np.isfinite(j).all(), "finite 192x20 root Jacobian")
    _,s,v=np.linalg.svd(j,full_matrices=False)
    # Fix sign only; degenerate subspaces are still implementation-dependent.
    for row in v:
        k=int(np.argmax(np.abs(row)))
        if row[k]<0: row *= -1
    return s,v


def library_from_basis(basis):
    """Eight independent reusable packet programs, frozen before evaluation."""
    v=np.asarray(basis,dtype=np.float64)
    require(v.shape==(20,20) and np.isfinite(v).all(), "basis shape")
    require(np.allclose(v @ v.T,np.eye(20),rtol=1e-10,atol=1e-10), "orthonormal basis")
    limit=CAP/math.sqrt(2)
    libraries=[]
    for position in range(2):
        items=[]
        for n in range(2):
            part=v[n,position*10:(position+1)*10]
            norm=float(np.linalg.norm(part))
            for sign in (-1,1):
                u=np.zeros(10) if norm==0 else part*(sign*limit/norm)
                raw=np.tanh(u).astype("<f4").reshape(5,2)
                item=dict(program_id=f"p{position}_m{n}_{sign:+d}", position=position,
                          logits=u.tolist(), raw_float32_hex=raw.tobytes().hex())
                item["program_sha256"]=digest(item)
                items.append(item)
        libraries.append(items)
    return libraries


def candidates(basis):
    v=np.asarray(basis,dtype=np.float64)
    libraries=library_from_basis(v)
    joint=[]
    for n in range(8):
        for sign in (-1,1):
            u=sign*CAP*v[n]
            raw=np.tanh(u).astype("<f4").reshape(10,2)
            joint.append(dict(index=len(joint),arm="joint",logits=u.tolist(),
                              raw_float32_hex=raw.tobytes().hex(),program_ids=["joint_"+str(len(joint))]))
    chains=[]
    for a in libraries[0]:
        for b in libraries[1]:
            u=np.asarray(a["logits"]+b["logits"],dtype=np.float64)
            require(np.linalg.norm(u)<=CAP+1e-12,"chain joint-ball bound")
            chains.append(dict(index=len(chains),arm="library_chain",logits=u.tolist(),
                raw_float32_hex=a["raw_float32_hex"]+b["raw_float32_hex"],
                program_ids=[a["program_id"],b["program_id"]],
                program_sha256=[a["program_sha256"],b["program_sha256"]]))
    return libraries,{"joint":joint,"library_chain":chains}


def execute_arm(context, goal, arm, bank, transition, emit, score_call=None):
    """All candidate and replay calls are explicit; exceptions are never retried."""
    require(arm in ARMS and callable(transition) and callable(emit), "fixed callbacks")
    require(len(bank)==COUNT and all(x["index"]==i and x["arm"]==arm
        for i,x in enumerate(bank)), "fixed candidate denominator/order")
    target=np.asarray(goal,dtype=np.float64)
    require(target.shape==(192,) and np.isfinite(target).all(), "goal shape")
    rows=[]
    def evaluate(item, replay):
        raw=bytes.fromhex(item["raw_float32_hex"])
        require(len(raw)==80,"literal full10 raw bytes")
        a=np.frombuffer(raw,dtype="<f4").reshape(10,2)
        require(np.isfinite(a).all() and np.all(np.abs(a)<=1),"raw action domain")
        current=deepcopy(context);edges=[]
        for step in range(2):
            packet=a[step*PACKET:(step+1)*PACKET]
            start=digest(current)
            emit(dict(kind="packet_intent",arm=arm,index=item["index"],replay=replay,packet=step,
                      raw_float32_hex=packet.tobytes().hex(),input_context_sha256=start))
            try:
                current=transition(deepcopy(current),packet.copy())
            except BaseException as exc:
                emit(dict(kind="packet_failure",arm=arm,index=item["index"],replay=replay,
                          packet=step,error_type=type(exc).__name__,message=str(exc)[:1000]))
                raise
            end=digest(current)
            edge=dict(index=step,program_id=(item["program_ids"][step] if arm=="library_chain"
                       else item["program_ids"][0]),raw_float32_hex=packet.tobytes().hex(),
                       input_context_sha256=start,output_context_sha256=end,
                       output_context=deepcopy(current))
            emit(dict(kind="packet_return",arm=arm,index=item["index"],replay=replay,edge=edge))
            edges.append(edge)
        z=np.asarray(current["latent_history"][-1],dtype=np.float64)
        require(z.shape==(192,) and np.isfinite(z).all(),"terminal latent")
        score=lambda: float(np.mean((z-target)**2))
        cost=score() if score_call is None else score_call(score)
        require(math.isfinite(cost),"finite score")
        result=dict(candidate=deepcopy(item),edges=edges,terminal_context=current,cost=cost)
        emit(dict(kind="candidate_return",arm=arm,index=item["index"],replay=replay,
                  raw_sha256=hashlib.sha256(raw).hexdigest(),cost=cost,edges=edges,
                  terminal_context_sha256=digest(current)))
        return result
    for item in bank:
        rows.append(evaluate(item,False))
    best=min(rows,key=lambda r:(r["cost"],r["candidate"]["index"]))
    replay=evaluate(best["candidate"],True)
    require(replay==best,"winner full-context deterministic replay")
    return dict(schema=NAME+".arm_result",arm=arm,candidates=rows,winner_index=best["candidate"]["index"],
                winner=best,replay=replay,packet_returns=34,score_returns=17,
                independent_neural_reexecution=False,physical=False,P2_admission=False)
