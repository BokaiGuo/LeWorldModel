"""All42 paired full15 model-mechanism table; saved only, no deployment choice."""
from __future__ import annotations
import argparse
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import agent_p2_three_program_20261028_v1 as planner

NAME='agent_p2_three_program_model_mechanism_20261029_v1'
ROOT,require,sha,raw,digest=planner.ROOT,planner.require,planner.sha,planner.raw,planner.digest
REPORT_SHA='56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef'
PREP_SHA='86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461'
PINS={
 'experiments/spectral_reachability/agent_p2_three_program_20261028_v1.py':'3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5',
 'tests/test_agent_p2_three_program_20261028_v1.py':'f5e678b822778b3f411e7aa90d8700d38e05f9ef0a61f809fb8eb50b716ded18',
 'innerdocs/agent_p2_three_program_20261028_v1.md':'ba1f40977f029541d400c6b5b112b1b01454ad15301c464bd6463557ae1b6c20'}
CAP=16*1024**2


def protocol():
    return dict(schema=NAME+'.protocol',known_roots=[0,1],condition='off',candidates_per_root=42,
        arm_rows=168,paired_candidate_rows=84,unordered_pairs_per_root=861,paired_rank_rows=1722,
        no_new_selection=True,no_physical_inputs=True,all_ties_retained=True,outcome_filter=False,
        score='original identity192 terminal MSE, float64 saved arithmetic',
        prefix_retreat='strict model Q at cp5 > Q at original root10; not physical retreat or ability change',
        attribution='matched seed jointQ -> cp5 recompile chosen H2 Q -> cp10 recompile chosen H1 Q',
        factorial_causality=False,non_greedy_physical_capability=None,native_success=None,physical_metrics=None,
        actual_latent_Q=None,new_model_calls=0,new_physics_calls=0,distinct_known_roots=2,independent_confirmation_samples=0,
        fresh=False,P1_admission=False,P2_mechanism=False,physical_release=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def sign(x):
    require(type(x) in (int,float) and np.isfinite(x),'finite real sign, no bool/epsilon')
    return -1 if x<0 else 1 if x>0 else 0


def q(context,goal):
    planner.kernel.validate_context(context)
    z=planner.finite(context['latent_history'][-1],(192,));g=planner.finite(goal,(192,))
    return float(np.mean((z-g)**2))


def exact_delta(a,b):
    """Orientation a-b, no epsilon or float subtraction cancellation."""
    require(type(a) is float and type(b) is float and np.isfinite([a,b]).all(),'original finite float64 cost fields')
    v=Fraction(a)-Fraction(b)
    return dict(value=float(v),sign=-1 if v<0 else 1 if v>0 else 0,numerator=str(v.numerator),denominator=str(v.denominator))


def local(edge,goal):
    evaluations=edge['evaluations'];selected=edge['selected_index']
    require(len(evaluations)==5 and type(selected) is int and selected in range(5),'complete incumbent+four candidate denominator')
    costs=[]
    for i,e in enumerate(evaluations):
        cost=q(e['contexts'][-1],goal)
        require(type(e['index']) is int and e['index']==i and type(e['cost']) is float and e['cost']==cost,'all original nonlinear Q values')
        costs.append(cost)
    # Check the *sealed* selector, do not create or alter a deployment selection.
    require(selected==min(range(5),key=lambda i:(costs[i],i)) and edge['incumbent_cost']==costs[0]
        and edge['selected_cost']==costs[selected] and costs[selected]<=costs[0],'sealed stable incumbent guard')
    planner.need_equal(edge['selected_contexts'],evaluations[selected]['contexts'],'exact selected local context')
    return dict(selected_index=selected,incumbent_retained=selected==0,selected_alpha=None if selected==0 else planner.ALPHAS[selected-1],
        all5_Q=costs,all5_delta_vs_incumbent=[exact_delta(v,costs[0]) for v in costs],
        incumbent_Q=costs[0],selected_Q=costs[selected],selected_delta=exact_delta(costs[selected],costs[0]),
        context_sha256=digest(edge['input_context']),fixed_prefix_sha256=digest(edge['fixed_prefix_logits']),
        selected_logits_sha256=digest(edge['selected_logits']),proposal_logit_gradient_norm=float(np.linalg.norm(edge['proposals']['gradient'])),
        proposal_slope=float(edge['proposals']['slope']),remaining_radius=float(edge['proposals']['remaining_radius']))


def compute(slots,rows):
    """Pure complete-table arithmetic. Expects separately authenticated native rows."""
    require(len(slots)==len(rows)==2 and [(s['root_slot'],s['condition']) for s in slots]==[(0,'off'),(1,'off')],
        'fixed two known off roots, no subset or extra roots')
    roots=[];all_rows=[];all_pairs=[]
    for s,r in zip(slots,rows,strict=True):
        require(r['index']==s['root_slot'] and r['status']=='sealed','source root row identity')
        result=r['result'];goal=s['packet']['goal'];root_Q=q(s['packet']['root_context'],goal)
        require(set(result['candidates'])==set(planner.ARMS) and len(result['bank']['params'])==42,'fixed original bank/arms')
        joint,composed=(result['candidates'][a] for a in planner.ARMS)
        require(len(joint)==len(composed)==42,'no missing/pruned candidates')
        paired=[]
        for i,(j,c) in enumerate(zip(joint,composed,strict=True)):
            require(type(j['index']) is int and type(c['index']) is int and j['index']==c['index']==i
                and len(j['contexts'])==len(c['contexts'])==3,'all42 original chronological candidate identities')
            require(j['logits']==result['bank']['params'][i] and c['root_seed_sha256']==digest(j['logits']), 'same root proposal provenance')
            planner.need_equal(j['contexts'][0],c['contexts'][0],'fixed same first imagined context')
            require(planner.chart(j['logits'])[0].tobytes()==planner.chart(c['logits'])[0].tobytes(),'fixed exact raw5 prefix')
            jQ,cQ=q(j['contexts'][-1],goal),q(c['contexts'][-1],goal)
            require(j['cost']==jQ and c['cost']==cQ,'original complete endpoint costs')
            edge2,edge3=c['edge2'],c['edge3']
            planner.need_equal(edge2['input_context'],c['contexts'][0],'cp5 complete conditional context')
            planner.need_equal(edge3['input_context'],c['contexts'][1],'cp10 complete conditional context')
            planner.need_equal(edge2['evaluations'][0]['contexts'],j['contexts'][1:],'same original H2 incumbent path')
            planner.need_equal(edge3['evaluations'][0]['contexts'][0],edge2['selected_contexts'][-1],'same original H1 incumbent path')
            two,three=local(edge2,goal),local(edge3,goal)
            require(two['incumbent_Q']==jQ and three['incumbent_Q']==two['selected_Q'] and three['selected_Q']==cQ,'terminal cost stage joins')
            d2=Fraction(two['selected_Q'])-Fraction(jQ);d3=Fraction(cQ)-Fraction(two['selected_Q'])
            total=Fraction(cQ)-Fraction(jQ);require(d2+d3==total,'exact rational telescoping, no float residual tolerance')
            stage={arm:[q(ctx,goal) for ctx in candidate['contexts']] for arm,candidate in zip(planner.ARMS,(j,c),strict=True)}
            entry=dict(root_slot=s['root_slot'],condition='off',task_id=s['task_id'],index=i,bank_kind=deepcopy(result['bank']['kinds'][i]),
                root_Q=root_Q,prefix_Q=stage[planner.ARMS[0]][0],prefix_delta=exact_delta(stage[planner.ARMS[0]][0],root_Q),
                strict_model_prefix_retreat=stage[planner.ARMS[0]][0]>root_Q,
                stage_Q=stage,joint_terminal_Q=jQ,composed_terminal_Q=cQ,
                joint_delta_vs_root=exact_delta(jQ,root_Q),composed_delta_vs_root=exact_delta(cQ,root_Q),
                cp5_recompile=two,cp10_recompile=three,composed_minus_joint=exact_delta(cQ,jQ),telescoping_exact=True,
                root_seed_sha256=digest(j['logits']),raw_prefix_sha256=digest(planner.blob(planner.chart(j['logits'])[0])),
                action_changed=dict(middle=planner.chart(j['logits'])[1].tobytes()!=planner.chart(c['logits'])[1].tobytes(),
                    terminal=planner.chart(j['logits'])[2].tobytes()!=planner.chart(c['logits'])[2].tobytes()),
                complete_literal_actions={a:planner.chart(candidate['logits']).reshape(15,2).tolist()
                    for a,candidate in zip(planner.ARMS,(j,c),strict=True)},
                native_success=None,physical_metrics=None,actual_latent_Q=None,physical_non_greedy_capability=None)
            paired.append(entry);all_rows.append(entry)
        ranks=[]
        for i,k in combinations(range(42),2):
            dJ=exact_delta(paired[i]['joint_terminal_Q'],paired[k]['joint_terminal_Q'])
            dC=exact_delta(paired[i]['composed_terminal_Q'],paired[k]['composed_terminal_Q'])
            kind='strict_reversal' if dJ['sign']*dC['sign']==-1 else 'both_tie' if dJ['sign']==dC['sign']==0 else (
                'joint_tie_only' if dJ['sign']==0 else 'composed_tie_only' if dC['sign']==0 else 'same_strict_direction')
            row=dict(root_slot=s['root_slot'],condition='off',left_index=i,right_index=k,
                joint_left_minus_right=dJ,composed_left_minus_right=dC,classification=kind)
            ranks.append(row);all_pairs.append(row)
        selected=[]
        for arm,candidates in zip(planner.ARMS,(joint,composed),strict=True):
            w=result['winners'][arm];index=w['index'];require(type(index) is int and index in range(42),'source winner index')
            require(index==min(range(42),key=lambda i:(candidates[i]['cost'],i)) and w['cost']==candidates[index]['cost'],'source-sealed full42 selector')
            planner.need_equal(w['contexts'],candidates[index]['contexts'],'source winner contexts')
            selected.append(dict(arm=arm,index=index,model_Q=w['cost'],source_selection_digest=r['plans'][planner.ARMS.index(arm)]['selection_digest'],
                source_commands_sha256=digest(w['commands']),new_selection=False,
                model_prefix_retreat=paired[index]['strict_model_prefix_retreat'],native_success=None))
        roots.append(dict(root_slot=s['root_slot'],condition='off',root_Q=root_Q,task_id=s['task_id'],
            candidates=42,paired_ranks=len(ranks),source_sealed_winners=selected,
            counts=dict(model_prefix_retreat=sum(e['strict_model_prefix_retreat'] for e in paired),
                cp5_incumbent_retained=sum(e['cp5_recompile']['incumbent_retained'] for e in paired),
                cp10_incumbent_retained=sum(e['cp10_recompile']['incumbent_retained'] for e in paired),
                terminal_improve=sum(e['composed_minus_joint']['sign']<0 for e in paired),
                terminal_tie=sum(e['composed_minus_joint']['sign']==0 for e in paired),
                terminal_worse=sum(e['composed_minus_joint']['sign']>0 for e in paired),
                rank_categories={k:sum(e['classification']==k for e in ranks) for k in
                    ('strict_reversal','both_tie','joint_tie_only','composed_tie_only','same_strict_direction')})))
    require(len(all_rows)==84 and len(all_pairs)==1722,'all42 paired rows and 861 unordered pairs/root')
    return dict(roots=roots,paired_candidates=all_rows,all_unordered_model_rank_pairs=all_pairs,
        arm_candidate_denominator=168,paired_candidate_denominator=84,model_rank_pair_denominator=1722,
        physical_capability_identifiable=False,physical_non_greedy_effect=None,
        causal_identification='sequential model-only changes, NOT independently randomized edge factorial effects')


def material(planner_output,strict_receipt,strict_sha):
    output=Path(planner_output).resolve();receipt=Path(strict_receipt).resolve()
    for rel,h in PINS.items():require(sha(ROOT/rel)==h,'frozen planner triple')
    require(sha(output/'report.json')==REPORT_SHA and sha(receipt)==strict_sha,'actual report/strict receipt external SHA')
    source=planner.parent.overlay.read(output/'report.json',planner.PREP_CAP)
    require(source['prepare_sha256']==PREP_SHA and source['status']=='all4_full15_plans_sealed_not_admitted','fixed actual preparation and terminal only')
    accepted=planner.consume(output,REPORT_SHA)  # Full arrays/events/Q/J/replays; NOT status-only.
    expected=planner.parent.overlay.read(receipt,CAP);planner.need_equal(accepted,expected,'whole MAIN strict receipt reconstructed')
    require(accepted['P2_mechanism'] is False and len(accepted['plans'])==4,'four sealed plans/no science')
    p=planner.parent.overlay.read(source['prepared'],planner.PREP_CAP)
    rows=[planner.parent.overlay.read(output/f'root{i}.json.gz',planner.CAP) for i in (0,1)]
    bindings=dict(p['bindings_sha256']);bindings.update({str(output/n):h for n,h in source['artifact_sha256'].items()})
    bindings.update({str(output/'report.json'):REPORT_SHA,str(receipt):strict_sha,str(Path(source['prepared']).resolve()):PREP_SHA})
    bindings.update({str(f):sha(f) for f in own_files()})
    for path,h in bindings.items():require(not Path(path).is_symlink() and sha(path)==h,'full source/input binding')
    return p,rows,bindings


def build(planner_output,strict_receipt,strict_sha):
    p,rows,bindings=material(planner_output,strict_receipt,strict_sha)
    report=dict(schema=NAME+'.report',status='all42_model_mechanism_saved_not_science',protocol=protocol(),
        arguments=dict(planner_output=str(Path(planner_output).resolve()),strict_receipt=str(Path(strict_receipt).resolve()),strict_sha=strict_sha),
        planner_report_sha256=REPORT_SHA,planner_prepare_sha256=PREP_SHA,full_upstream_strict_rebuilt_here=True,
        bindings_sha256=bindings,model_analysis=compute(p['slots'],rows),errors=[],new_model_calls=0,new_physics_calls=0,
        P1_admission=False,P2_mechanism=False,physical_release=False,fresh=False)
    require(len(raw(report))<=CAP,'bounded16MiB all42 saved table');return report


def write(path,value):
    p=Path(path).absolute();data=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink()
        and len(data)<=CAP and shutil.disk_usage(p.parent).free>=planner.FLOOR+len(data),'exclusive bounded own report')
    with p.open('xb') as f:require(f.write(data)==len(data),'short report');f.flush();os.fsync(f.fileno())
    planner.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p),status=value['status'])


def verify(path,expected):
    require(sha(path)==expected,'own report SHA');r=planner.parent.overlay.read(path,CAP)
    planner.need_equal(r,build(**r['arguments']),'entire all42 analysis and upstream strict rebuilt')
    require(sha(path)==expected,'final report unchanged');return dict(status='all42_saved_analysis_verified_not_science',sha256=expected,
        paired_candidates=84,all_model_rank_pairs=1722,P2_mechanism=False)


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('analyze');a.add_argument('--planner-output',required=True);a.add_argument('--strict-receipt',required=True)
    a.add_argument('--strict-sha',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('verify');a.add_argument('--report',required=True);a.add_argument('--report-sha',required=True)
    args=p.parse_args()
    if args.cmd=='analyze':value=write(args.output,build(args.planner_output,args.strict_receipt,args.strict_sha))
    else:value=verify(args.report,args.report_sha)
    print(raw(value).decode(),end='')


if __name__=='__main__':main()
