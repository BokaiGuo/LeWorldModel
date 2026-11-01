"""Six-slot CPU/fixture planning and replay core; NOT a CUDA runner or receipt."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import numpy as np
from experiments.spectral_reachability import agent_experiment1_common_controller_20261028_v1 as old

NAME='agent_e1_six_slot_controller_core_20261029_v1'
ARMS=('arbitrary_latent_waypoint','existing_learned_prior','dynamics_compiled_witness')
SLOTS=tuple((r,a) for r in (0,1) for a in ARMS)


def require(v,message):
    if not v:raise ValueError(message)


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def optimize(task,root,arm,target,dispatch,sink):
    """Original 3x24 H2 arithmetic; learned arm is permitted explicitly."""
    old.core.validate_task(task)
    require((root,arm) in SLOTS,'fixed six-slot identity')
    w=old.vec(target);goal=old.vec(task['goal'])
    require(task['packet_logit_cap']==old.beam.CAP,'original projected ball')
    rng=np.random.Generator(np.random.PCG64(old.SEEDS[root]))
    mean=np.zeros((2,10));std=np.ones((2,10));bank=[];updates=[]
    initial=old.beam.checked_context(task['root_context'])
    for it in range(3):
        params=rng.normal(mean,std,size=(24,2,10))
        logits=[old.sampling.gn.screen.projected_logits(x,old.beam.CAP) for x in params]
        sink(dict(kind='bank_frozen',iteration=it,mean=mean.tolist(),std=std.tolist(),parameters=params.tolist(),logits=[x.tolist() for x in logits]))
        chunk=[]
        for j,x in enumerate(logits):
            raw=np.tanh(x.astype(np.float32)).reshape(10,2).astype(float).tolist()
            states=dispatch(initial,raw,'whole_H2',[it,j]);require(len(states)==2,'complete two-packet result')
            prefix=float(np.mean((old.vec(states[0]['latent_history'][-1])-w)**2))
            terminal=float(np.mean((old.vec(states[1]['latent_history'][-1])-goal)**2))
            c=dict(index=len(bank),iteration=it,parameters=params[j].tolist(),logits=x.tolist(),commands=raw,
                   contexts=states,prefix_target_cost=prefix,model_terminal_cost=terminal,cost=.5*(prefix+terminal))
            bank.append(c);chunk.append(c);sink(dict(kind='candidate_saved',candidate=c))
        elite=sorted(chunk,key=lambda c:(c['cost'],c['index']))[:4]
        xx=np.asarray([c['logits'] for c in elite]);mean=xx.mean(axis=0);std=np.clip(xx.std(axis=0,ddof=0),.05,1.)
        update=dict(iteration=it,mean=mean.tolist(),std=std.tolist(),elite_indices=[c['index'] for c in elite])
        updates.append(update);sink(dict(kind='distribution_update',update=update))
    winner=min(range(72),key=lambda i:(bank[i]['cost'],i))
    sink(dict(kind='winner_sealed_no_replay',index=winner,commands=bank[winner]['commands'],cost=bank[winner]['cost']))
    return dict(task_id=old.core.task_id(task),source_slot=root,arm=arm,seed=old.SEEDS[root],target=list(target),
                candidates=bank,updates=updates,winner_index=winner)


class Attempts:
    """Fixture-level logical packet cap; not actual CUDA module metering."""
    def __init__(self):self.attempted=0;self.returned=0;self.poisoned=False
    def invoke(self,packet_count,callback):
        require(not self.poisoned and packet_count in (1,2) and self.attempted+packet_count<=148,'pre-dispatch logical packet cap')
        self.attempted+=packet_count
        try:
            result=callback();require(len(result)==packet_count,'complete packet return')
            self.returned+=packet_count;return result
        except BaseException:
            self.poisoned=True;raise


def plan(task,root,arm,target,transition,sink,meter):
    def dispatch(before,raw,phase,path):
        return meter.invoke(2,lambda:transition(deepcopy(before),deepcopy(raw),phase,path))
    result=optimize(task,root,arm,target,dispatch,sink)
    require(meter.attempted==meter.returned==144,'144 planned packet attempts/returns')
    return result


def replay_slot(task,result,transition,sink,meter):
    """Reload saved raw literal, no regenerate from logits; no scores here."""
    raw=deepcopy(result['candidates'][result['winner_index']]['commands'])
    old.beam.commands(raw)
    root=old.beam.checked_context(task['root_context'])
    sink(dict(kind='replay_raw_sealed',raw=raw,raw_sha256=digest(raw)))
    whole=meter.invoke(2,lambda:transition(deepcopy(root),deepcopy(raw),'replay_whole',[0,1]))
    first=meter.invoke(1,lambda:transition(deepcopy(root),deepcopy(raw[:5]),'replay_segment',[0]))
    second=meter.invoke(1,lambda:transition(deepcopy(first[0]),deepcopy(raw[5:]),'replay_segment',[1]))
    row=dict(root=result['source_slot'],arm=result['arm'],raw=raw,whole=whole,segmented=first+second)
    sink(dict(kind='replay_predictions_sealed',root=row['root'],arm=row['arm'],sha256=digest(row)))
    require(meter.attempted==meter.returned==148,'full 144+4 logical attempts/returns')
    return row


def score_after_barrier(rows,seal):
    require(len(rows)==6 and tuple((r['root'],r['arm']) for r in rows)==SLOTS,'six complete replay slots')
    require(seal==digest(rows),'all replay values sealed before scoring')
    require(all(len(r['whole'])==len(r['segmented'])==2 for r in rows),'two replay checkpoints per slot')
    return [dict(root=r['root'],arm=r['arm'],context_json_equal=all(
        digest(a)==digest(b) for a,b in zip(r['whole'],r['segmented'],strict=True))) for r in rows]
