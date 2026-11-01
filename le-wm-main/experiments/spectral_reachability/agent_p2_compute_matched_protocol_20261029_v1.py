"""Saved-only compute-matching preregistration; no model/physics execution CLI.

Contains an executable backend ABI for a fixed joint challenger. Its measured
vector must not be called matched merely because Jacobian counts coincide.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p2_three_program_20261028_v1 as old

NAME='agent_p2_compute_matched_protocol_20261029_v1'
REPORT_SHA='56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef'
RECEIPT_SHA='c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f'
SOURCE_SHA='3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5'
ROOT=old.ROOT
require=old.require
UNKNOWN=('total_FLOPs','engine_internal_backward_calls','peak_allocated_bytes','peak_reserved_bytes','wall_seconds')


def budget(arm):
    """Standalone per-root, includes root bank and winner dual replay.

    Predict includes AD primal calls. VJP is mathematical rows, NOT measured
    engine backward calls. Root SVD/ridge is shared identically in both arms.
    """
    require(arm in ('composition','joint_two_pass'),'fixed arms only')
    comp=arm=='composition'
    h2,h1,h3=(42,42,1) if comp else (0,0,85)
    n=936 if comp else 1902
    return dict(predict=n,action_encoder=n,predictor=n,pred_proj=n,
        jacobian=85,mathematical_VJP=85*192,
        jacobian_elements=192*(h1*10+h2*20+h3*30),
        AD_rollout_steps=h1+2*h2+3*h3,
        AD_H1_raw10=h1,AD_H2_raw20=h2,AD_H3_raw30=0 if comp else 84,
        AD_H3_logit30=1,root_SVD=2,root_ridge=1,
        encoder=0,projector=0,physics=0,training=0,padding=0,retries=0,
        **{k:None for k in UNKNOWN})


def compare(a,b):
    require(set(a)==set(b)==set(budget('composition')),'complete vector schema, not predict-only')
    unknown=[k for k in a if a[k] is None or b[k] is None]
    different={k:dict(composition=a[k],joint=b[k]) for k in a if a[k] is not None and b[k] is not None and a[k]!=b[k]}
    return dict(matched=not unknown and not different,different=different,unknown=unknown,
        count_equality_is_not_FLOPs_equality=True)


def validate_ledger(actual,expected):
    require(set(actual)==set(expected),'all resource fields required, no unknown fields')
    for k,v in actual.items():
        require(v is None or (type(v) in (int,float) and np.isfinite(v) and v>=0),'finite nonnegative ledger')
        if k not in UNKNOWN:require(type(v) is int and v==expected[k],'exact fixed schedule resource '+k)
    return compare(actual,expected)


def schedule():
    return [dict(root=r,seed=i,**{'pass':p},dimensions=30,horizon=3,alphas=list(old.ALPHAS))
        for r in (0,1) for i in range(42) for p in (0,1)]


def local_joint(context,incumbent,goal,backend,tag):
    """Full30 joint raw-J local step, not old suffix-only proposals.

    Backend is the frozen CUDA rollout/jacobian ABI; worker tests use RAM only.
    No gradients of Q or new AD route, no physics/state/labels accepted.
    """
    u=np.asarray(incumbent,np.float64)
    require(u.shape==(30,) and old.legal([],u),'legal full30 exact incumbent; no projection')
    path=backend.rollout(context,old.chart(u),tag+'/incumbent')
    j=old.finite(backend.jacobian(context,u,tag+'/J',False),(192,30))
    old.need_equal(backend.last_ad_contexts,path,'AD full context equals literal incumbent')
    z=old.finite(path[-1]['latent_history'][-1],(192,));g=old.finite(goal,(192,))
    grad=(2/192)*(j*(1-np.tanh(u)**2)[None,:]).T@(z-g)
    norm=float(np.linalg.norm(grad))
    target=u.copy() if norm==0 else -.5*old.CAP_LOGIT*grad/norm
    direction=target-u
    if float(grad@direction)>=0:direction=np.zeros_like(u)
    logits=[u]+[u+a*direction for a in old.ALPHAS]
    require(all(old.legal([],v) for v in logits),'all fixed proposals legal, never clip')
    evaluations=[dict(index=0,logits=u.tolist(),contexts=path,cost=old.objective(path[-1],goal))]
    for i,v in enumerate(logits[1:],1):
        p=backend.rollout(context,old.chart(v),tag+'/candidate'+str(i))
        evaluations.append(dict(index=i,logits=v.tolist(),contexts=p,cost=old.objective(p[-1],goal)))
    winner=min(evaluations,key=lambda x:(x['cost'],x['index']))
    replay=backend.rollout(context,old.chart(winner['logits']),tag+'/selected_replay')
    old.need_equal(replay,winner['contexts'],'literal selected whole15 local replay')
    return dict(logits=winner['logits'],contexts=replay,cost=winner['cost'],selected_index=winner['index'],evaluations=evaluations)


def joint_candidates(slot,bank,backend):
    """Fixed42 seeds, TWO full30 passes each; complete stable selection.

    Includes initial seed full15 evaluation and final whole/segmented replay.
    Shared root nominal/J/bank paid by caller exactly as frozen old.plan.
    This ABI is not authorization to run an unmatched experiment.
    """
    require(len(bank['params'])==42,'all original bank seeds including null modes')
    c=slot['packet']['root_context'];g=slot['packet']['goal'];candidates=[]
    for i,seed in enumerate(bank['params']):
        path=backend.rollout(c,old.chart(seed),'joint_matched/'+str(i)+'/seed')
        u=seed;passes=[]
        for p in (0,1):
            result=local_joint(c,u,g,backend,f'joint_matched/{i}/pass{p}')
            passes.append(result);u=result['logits']
        require(result['cost']<=old.objective(path[-1],g),'incumbent safeguarded model cost only')
        candidates.append(dict(index=i,logits=u,contexts=result['contexts'],cost=result['cost'],passes=passes))
    w=deepcopy(min(candidates,key=lambda x:(x['cost'],x['index'])))
    whole=backend.rollout(c,old.chart(w['logits']),'joint_matched/winner_whole')
    current=deepcopy(c);segmented=[]
    for k,a in enumerate(old.chart(w['logits'])):
        current=backend.rollout(current,a[None],'joint_matched/winner_segment'+str(k))[0];segmented.append(current)
    old.need_equal(whole,w['contexts'],'whole full15 winner');old.need_equal(segmented,whole,'all3 segmented boundary contexts')
    return dict(candidates=candidates,winner=w)


def protocol(authority):
    a,b=budget('composition'),budget('joint_two_pass')
    return dict(schema=NAME+'.prepare',authority=authority,
        status='blocked_no_vector_compute_match',run_authorized=False,
        roots=[0,1],condition='off',known_roots=2,fixed_candidates_per_root=42,
        schedule=schedule(),per_root=dict(composition=a,joint_two_pass=b),comparison=compare(a,b),
        original_combined_total=dict(predict=2136,jacobian=170,mathematical_VJP=32640),
        shared_cost_rule='standalone arms each pay root nominal/J/bank; original combined1068/root shares6 predict and1 root J; never sum standalone as original total',
        rule='same42 original root bank, all30 joint twice at original context, original mean192 Q; four fixed alphas; stable cost/index; incumbent kept on ties; winner whole+segmented replay',
        domain='original full30 logit L2 .75sqrt15; NumPy tanh float64 to rawfloat32 once; no inverse/clipping/new seed/normalizer/Q',
        runtime_requirements=['complete per-call intent/return/module/AD shape and contexts',
            'actual attempted/completed counts, no refunds or padding',
            'same installed classes/frozen weights, beforeafter model/source seals',
            'TF32/autocast off, MATH SDP, original native float32 transition',
            'wall/peak allocated/reserved measured, no inference from counters',
            'unknown engine backward/FLOPs remain null, cannot certify matching'],
        gate_rule='no execution release under compute-matched label unless every resource field is known and equal; revise algorithm via NEW pre-run protocol, never outcome-tune this seal',
        selection_inputs='original root latents/history/goal, model-only predictions; no physical/RGB outcomes',
        outcome_access=False,new_neural_calls=0,new_physics_calls=0,training=False,Qfit=False,
        equal_FLOPs=False,equal_total_compute=False,P1_admission=False,P2_admission=False,
        P2_mechanism=False,fresh=False,physical_release=False)


def files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(output,receipt):
    """MAIN saved-only preparation, replays original strict consumer, never NN."""
    output=Path(output).resolve();receipt=Path(receipt).resolve()
    require(old.sha(Path(old.__file__))==SOURCE_SHA,'frozen original planner source')
    require(old.sha(output/'report.json')==REPORT_SHA and old.sha(receipt)==RECEIPT_SHA,'fixed original report/strict SHAs')
    consumed=old.consume(output,REPORT_SHA)
    old.need_equal(consumed,json.loads(receipt.read_bytes()),'complete actual saved strict receipt rebuilt')
    report=json.loads((output/'report.json').read_bytes())
    compiled=old.verify(report['prepared'],report['prepare_sha256'])
    authority=dict(report=str(output/'report.json'),report_sha256=REPORT_SHA,
        receipt=str(receipt),receipt_sha256=RECEIPT_SHA,source_sha256=SOURCE_SHA,
        prepare=report['prepared'],prepare_sha256=report['prepare_sha256'],
        tasks=[{k:s[k] for k in ('root_slot','condition','task_id','task_sha256')} for s in compiled['slots']])
    p=protocol(authority);bindings=deepcopy(compiled['bindings_sha256'])
    bindings[report['prepared']]=report['prepare_sha256']
    for name,h in report['artifact_sha256'].items():bindings[str(output/name)]=h
    for f in files():bindings[str(f)]=old.sha(f)
    bindings[str(output/'report.json')]=REPORT_SHA;bindings[str(receipt)]=RECEIPT_SHA
    p['bindings_sha256']=bindings
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify'))
    ap.add_argument('--planner-output',required=True);ap.add_argument('--planner-receipt',required=True)
    ap.add_argument('--output');ap.add_argument('--prepared');ap.add_argument('--prepare-sha');a=ap.parse_args()
    p=build(a.planner_output,a.planner_receipt)
    if a.command=='prepare':
        path=Path(a.output)
        require(path.is_absolute() and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(),'new own absolute output')
        body=old.raw(p);require(len(body)<=4*1024**2,'bounded4MiB metadata')
        with path.open('xb') as f:
            require(f.write(body)==len(body),'complete own metadata');f.flush();os.fsync(f.fileno())
        old.fsync_dir(path.parent)
        result=dict(path=str(path),sha256=old.sha(path),status=p['status'],run_authorized=False)
    else:
        require(old.sha(a.prepared)==a.prepare_sha,'actual own prepare SHA')
        old.need_equal(json.loads(Path(a.prepared).read_bytes()),p,'full saved preregistration rebuilt')
        result=dict(verified=True,status=p['status'],run_authorized=False)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
