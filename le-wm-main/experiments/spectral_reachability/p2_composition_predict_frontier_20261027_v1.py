"""Frozen candidate/budget bridge, NOT an executable CUDA planner or P2 pass.

Preserve legacy deterministic40; append seeded spectral mixtures for joint80.
Only model-free CLI/source/candidate work. No neural/physics API is exposed.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

NAME='p2_composition_predict_frontier_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
LEGACY='experiments/spectral_reachability/p2_composition_paired_20261027_v2.py'
SRA='experiments/spectral_reachability/p2_no_composition_sra_20261027_v2.py'
LEDGER='experiments/spectral_reachability/p2_matched_composition_20261027.py'
LEGACY_SHA='8d3f924c2bc38730e3d3eeedb24149cfb3a420857d6f8132bc95eeb01fe87379'
SRA_SHA='6e1402ac8abe3aabfffefd24aa016c341e6eb4f40d3f60e233152460ca70e45a'
LEDGER_SHA='1b15e2c3f85e7459bef595655815c2a924bc2f5bd287a4d75b808b3af659ee9e'
ROOTS=('9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333',
    '23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050',
    '4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca',
    'cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2')
ARMS=('joint_h2_no_composition','conditional_suffix_composition')
ROOT_BANK_DIGESTS=('3370168bc4f3c84e766543734717cf4caedd79e98b0213b074fe0af158ce2475',
    '59edb962b640641b077d61a4b596a61339c031b5fbd84f26bff8573205afe11b',
    'ac05419d79fe41b0355a3d833c758844ff10a9c3ef4e60350e9bdf19d43bce06',
    'c65daafcf9310562142680fad1ce53cb7e2d90109d57b076a3fd3baa3d7e73b6')
CAP=float(np.sqrt(10)*.75)
RADII=(float(np.sqrt(10)*.25),CAP)
SEED_BASE=20261027


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def array(value,shape):
    a=np.asarray(value);require(a.shape==shape and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite numeric exact shape');return a.astype(np.float64)


def contract():
    return dict(status='bridge_only_awaiting_new_runner',authority='main-directed frontier budgets; implementation-thread predeclared seeded tail, pending main review',
        roots=list(ROOTS),horizon=2,raw_commands=10,context='original three latents/two between-state action packets',
        goal_Q='original goal0/original Q, no goal-patch intervention',feedback='none within H2: open-loop predicted-prefix conditional compile',
        budget_points=[dict(joint_candidates=40,composition_candidates=40,standalone_predict=[88,168]),
            dict(joint_candidates=80,composition_candidates=40,standalone_predict=[168,168])],
        purpose='quality versus model-call frontier; not total-FLOP/resource matching',
        root_compile='nominal H2=2 + vectorized-Jacobian forward H2=2; shared4',
        base40='EXACT inherited zero+DLS3+top9 modes x radii2 x signs2; no seed in legacy40',
        tail40='PCG64(20261027+root_index),10 top9 Gaussian mixtures x original radii2 x signs(-,+); retain all',
        candidate_prefix='joint40 logits/actions/labels exact prefix of joint80; composition uses only original40 prefixes',
        support='joint pre-tanh20D L2 cap sqrt(10)*.75, float32 tanh then original raw[-1,1]; NOT per-packet balls',
        zero_rank='same inherited LAPACK basis including null modes; no redraw/truncation',
        selection='stable first nonlinear original-Q terminal mean-square minimum over ALL evaluated slots',
        replay='winner segmented H2 plus independent whole H2 =4 predicts; no padding/refund',
        stale_legacy_export_for80_forbidden=True,new_neural_calls=0,new_physical_steps=0,
        P1_admission=False,P2_admission=False,total_resource_matched=False,actual_cuda_calibrated=False)


def validate_base(bank):
    u=array(bank['logits'],(40,20));raw=array(bank['commands'],(40,2,5,2));right=array(bank['right_vectors'],(20,20))
    require(np.allclose(right@right.T,np.eye(20),rtol=0,atol=1e-10),'inherited orthonormal spectral basis')
    require(all(v[np.argmax(np.abs(v))]>=0 for v in right),'canonical inherited mode signs')
    dls=array(bank['dls']['params'],(20,))
    expected=np.stack([np.zeros(20)]+[s*dls for s in (.25,.5,1.)]+[sign*r*right[m] for m in range(9) for r in RADII for sign in (-1,1)])
    labels=['zero','dls_scale_.25','dls_scale_.5','dls_scale_1']+[f'mode_{m}_radius_{r}_sign_{s}' for m in range(9) for r in range(2) for s in (-1,1)]
    require(np.array_equal(u,expected) and bank['labels']==labels,'EXACT original40 logits/order')
    require(np.array_equal(raw,np.tanh(u).astype(np.float32).reshape(40,2,5,2)) and np.max(np.linalg.norm(u,axis=1))<=CAP*(1+1e-12),'original bounded action mapping')
    require(all(isinstance(bank.get(k),str) and len(bank[k])==64 for k in ('goal_sha256','projection_sha256')),'bound original goal/Q required')
    return u,right


def candidate_bank(bank,root_sha256,count=80):
    require(root_sha256 in ROOTS and type(count) is int and count in (40,80),'fixed four roots and frozen40/80 budget points')
    base,right=validate_base(bank);i=ROOTS.index(root_sha256);rng=np.random.Generator(np.random.PCG64(SEED_BASE+i))
    tail=[];labels=[];coefficients=[]
    for k in range(10):
        c=rng.normal(size=9);norm=float(np.linalg.norm(c));require(norm>0,'degenerate draw; strict refusal not redraw')
        c=c/norm;direction=c@right[:9];coefficients.append(c.tolist())
        for radius_index,r in enumerate(RADII):
            for sign in (-1,1):
                tail.append(sign*r*direction);labels.append(f'mixture_{k}_radius_{radius_index}_sign_{sign}')
    all_logits=np.concatenate((base,np.asarray(tail)),axis=0)
    require(np.max(np.linalg.norm(all_logits,axis=1))<=CAP*(1+1e-12),'whole joint ball cap')
    commands=np.tanh(all_logits).astype(np.float32).reshape(80,2,5,2)
    require(np.array_equal(commands[:40],np.asarray(bank['commands'])),'unchanged prefix40 raw packets')
    body=dict(schema=NAME+'.candidate_bank',root_sha256=root_sha256,root_index=i,seed=SEED_BASE+i,
        count=count,logits=all_logits[:count].tolist(),commands=commands[:count].tolist(),labels=(bank['labels']+labels)[:count],
        original_bank_sha256=digest(bank),original40_sha256=digest(dict(logits=bank['logits'],commands=bank['commands'],labels=bank['labels'])),
        goal_sha256=bank['goal_sha256'],projection_sha256=bank['projection_sha256'],tail_coefficients=coefficients if count==80 else [],
        new_svd_calls=0,new_jacobian_calls=0,new_predict_calls=0,neural_ready=False,source_verified=False,
        source_scope='pure generation recipe; use bound_candidate_bank for exact saved root/goal/Q/basis linkage')
    body['candidate_bank_sha256']=digest(body);return body


def bound_candidate_bank(bank,root_sha256,count=80):
    require(root_sha256 in ROOTS,'fixed root')
    require(digest(bank)==ROOT_BANK_DIGESTS[ROOTS.index(root_sha256)],'exact root-bound original bank/goal/Q/derivative arrays')
    out=candidate_bank(bank,root_sha256,count);out.pop('candidate_bank_sha256')
    out.update(source_verified=True,source_scope='exact archived root bank verified; no fresh root compile or model execution credited')
    out['candidate_bank_sha256']=digest(out);return out


def expected_entries(joint_candidates):
    """Individual forward entry specification, not formula-only budget totals.

    Mirrors old compile terminal two-packet loop, actual run_arm dispatch, and
    suffix_cuda nominal/AD terminal forwards. Every packet has both entries.
    """
    require(type(joint_candidates) is int and joint_candidates in (40,80),'fixed frontier point')
    entries=[]
    def packet(scope,index,role,packet_index=0):
        for kind in ('action_encoder','predict'):
            entries.append(dict(kind=kind,scope=scope,index=index,role=role,packet=packet_index,samples=1,context_tokens=3,completed=True))
    for role in ('root_nominal','root_Jacforward'):
        for p in range(2):packet('shared',None,role,p)
    for arm,n in zip(ARMS,(joint_candidates,40)):
        for i in range(n):
            packet(arm,i,'candidate_prefix')
            if arm==ARMS[1]:
                packet(arm,i,'suffix_nominal');packet(arm,i,'suffix_Jacforward')
            packet(arm,i,'candidate_suffix')
        packet(arm,None,'winner_segmented',0);packet(arm,None,'winner_segmented',1)
        packet(arm,None,'winner_whole',0);packet(arm,None,'winner_whole',1)
    return entries


def validate_entries(entries,joint_candidates):
    require(entries==expected_entries(joint_candidates),'exact ordered attempted/completed forward census; no padding/refund or batch change')
    scopes=('shared',*ARMS);exclusive={s:sum(e['kind']=='predict' and e['scope']==s for e in entries) for s in scopes}
    shared=exclusive.pop('shared');standalone=[shared+exclusive[a] for a in ARMS]
    return dict(shared_predict=shared,exclusive_predict=exclusive,standalone_predict=standalone,
        paired_unique_predict=shared+sum(exclusive.values()),action_encoder_calls=sum(e['kind']=='action_encoder' for e in entries),
        context_tokens_per_entry_type=3*(shared+sum(exclusive.values())),predict_budget_matched=standalone[0]==standalone[1],
        completed_candidates={ARMS[0]:joint_candidates,ARMS[1]:40},selection_evaluations=joint_candidates+40,
        standalone_Jacobian_calls=[1,41],standalone_mathematical_VJP_directions=[192,7872],
        paired_unique_Jacobian_calls=41,paired_unique_mathematical_VJP_directions=7872,
        paired_unique_SVD='2 shared + positive-radius suffix count (0..40)',
        paired_unique_ridge_solves='1 shared + positive-radius suffix count (0..40)',
        backward_engine_API_calls=None,total_FLOPs=None,total_resource_matched=False,
        cached_RGB_goal_context_Q_cost='inherited unknown; not zero',
        CPU_tail_generation='10 Gaussian9 draws +10 normalizations +10 9x20 combinations +40 scalings; no extra SVD',
        quality_measurements=None,actual_CUDA_execution=False,P1_admission=False,P2_admission=False)


def legacy_source_contract(root=PROJECT):
    """Bind exact files, inspect actual fixed-loop compile/forward entry sites."""
    root=Path(root).resolve();paths={root/LEGACY:LEGACY_SHA,root/SRA:SRA_SHA,root/LEDGER:LEDGER_SHA}
    texts={}
    for path,h in paths.items():
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==h,'legacy source drift: '+str(path));texts[path.name]=raw.decode()
    paired=ast.parse(texts[Path(LEGACY).name]);sra=ast.parse(texts[Path(SRA).name])
    funcs={f.name:f for f in paired.body if isinstance(f,ast.FunctionDef)};sfuncs={f.name:f for f in sra.body if isinstance(f,ast.FunctionDef)}
    arm=ast.get_source_segment(texts[Path(LEGACY).name],funcs['run_arm'])
    suffix=ast.get_source_segment(texts[Path(LEGACY).name],funcs['suffix_cuda'])
    compilation=ast.get_source_segment(texts[Path(SRA).name],sfuncs['compile_cuda'])
    require('for index in range(40)' in arm and arm.count('ledger.call(')==5,'legacy actual candidate2 and winner3 dispatch sites')
    require("actions[:5]" in arm and "actions[5:]" in arm and "actions,'replay',[42]" in arm,'legacy winner packet1+packet1+whole2')
    require('for packet in range(2)' in compilation and 'nominal = terminal(zero)' in compilation and 'torch.autograd.functional.jacobian(terminal,' in compilation,'root nominal2 + ADforward2 source')
    require('nominal = terminal(zero)' in suffix and 'torch.autograd.functional.jacobian(terminal,' in suffix and 'transition(emb,history,torch.tanh(u).reshape(5,2))' in suffix,'own-context suffix nominal1+ADforward1')
    ledger_tree=ast.parse(texts[Path(LEDGER).name]);ledger_class=next(f for f in ledger_tree.body if isinstance(f,ast.ClassDef) and f.name=='_Ledger')
    call=next(f for f in ledger_class.body if isinstance(f,ast.FunctionDef) and f.name=='call')
    ledger_source=ast.get_source_segment(texts[Path(LEDGER).name],call)
    require('for start in range(0, len(actions), 5)' in ledger_source and 'predict_calls=1, predict_samples=1' in ledger_source and 'actions[start:start + 5]' in ledger_source,'ledger splits whole H2 into TWO separate replay dispatches/events')
    return dict(source_bindings={str(p):h for p,h in paths.items()},verified_call_sites=True,
        old_run_arm_predict_exclusive=[84,164],old_shared_predict=4,
        old_physical_forward_used=False,AD_forward_count_scope='source-level known terminal invocation; not backward engine calls',
        old40_contract_accepts80=False,legacy_schedule_or_exporter_may_be_reused_for80=False,
        remaining_runner_interfaces=remaining_interfaces())


def remaining_interfaces():
    return ['NEW source-bound prepare: preserve four root/context/goal0/Q/normalizer/model/action-domain bindings; no outcome reads',
        'NEW guarded CUDA runner: one shared root compile, joint80 x2 real forwards, conditional40 x4 real forwards, each winner4; per-arm164 exclusive cap, global332/root',
        'Instrument predict AND action encoder at entry including root/suffix AD forwards, durable nonrefundable attempts/errors, operation Jac/VJP/SVD/solve and CPU selection/generation costs',
        'NEW all-slot witness writer/exporter:80 joint single-edge +40 conditional dual-edge, exact full/segmented winner replay and full-bank digest; cannot use old40 selected_actions',
        'NEW report schema/frontier auditor: retain joint40 minimum diagnostic from nested80 without extra replay; stored historical88 point separately calibrated, never mix old/new runs as same execution',
        'NEW independent physical consumer executes fresh winners only after main review/CUDA; no old endpoint substitution; no P1/P2 admission']


def summary():
    return dict(schema=NAME+'.bridge',contract=contract(),legacy=legacy_source_contract(),
        frontier=[validate_entries(expected_entries(n),n) for n in (40,80)],
        four_root_new_point=dict(paired_unique_predict=1328,paired_unique_action_encoder=1328,context_tokens_each=3984,
            standalone_predict_each_arm=672,Jacobian_calls=164,mathematical_VJP_directions=31488),
        status='source_checked_bridge_not_CUDA_ready',new_neural_calls=0,new_physical_steps=0,P2_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bank-json',type=Path);parser.add_argument('--root-sha256');parser.add_argument('--count',type=int,default=80)
    a=parser.parse_args()
    if a.bank_json:
        require(a.root_sha256 is not None,'fixed root identity');require(a.bank_json.stat().st_size<=4*1024**2,'small supplied source bank only')
        r=bound_candidate_bank(json.loads(a.bank_json.read_text()),a.root_sha256,a.count)
    else:r=summary()
    print(json.dumps(r,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
