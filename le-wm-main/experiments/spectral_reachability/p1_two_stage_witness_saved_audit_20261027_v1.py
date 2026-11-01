"""Independent saved-array arithmetic audit. NO neural/physics execution.

Wait for a terminal report and externally retained actual SHA. Verify full
source/task bindings; reconcile saved operation pairs, local J rows, nonlinear
paths and independent H2 replays. Numerical differences are descriptive, not
AD correctness or P1 evidence. No optimizer implementation is called here.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import numpy as np

from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core
from experiments.spectral_reachability import p1_two_stage_witness_20261027_v1 as producer

NAME='p1_two_stage_witness_saved_audit_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
require=core.require
ALGORITHM_SHA='e4d35e014e793db65276474fa90fbc3a6877c01f39ab0839259e746586c34719'
EXPECTED=dict(producer=dict(predict=120,action_encoder=120,jacobian=17,svd=17,
    vjp_directions=3264,linear_solve=17,prefix_generation=1,suffix_proposal=64,selection=64),
    witness=dict(predict=6,action_encoder=6,jacobian=0,svd=0,vjp_directions=0,
    linear_solve=0,prefix_generation=0,suffix_proposal=0,selection=0))


def one(events,kind,stage=None):
    rows=[e for e in events if e['kind']==kind and (stage is None or e.get('stage')==stage)]
    require(len(rows)==1,'exact one '+kind+' / '+str(stage))
    return rows[0]


def near(actual,expected,label):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and a.dtype.kind in 'fi' and np.isfinite(a).all() and
        np.allclose(a,b,rtol=1e-9,atol=1e-11),'arithmetic receipt '+label)


def packet(task,before,logits,after):
    core.context(before);core.context(after);x=core.array(logits,(10,))
    require(np.linalg.norm(x)<=task['packet_logit_cap']+1e-12,'per-packet action ball')
    raw=np.tanh(x.astype(np.float32)).reshape(5,2)
    require(after['latent_history'][:2]==before['latent_history'][1:],'packet latent shift')
    expected=np.concatenate((np.asarray(before['action_history']),core.normalized_tokens(task,raw)))[-2:]
    require(np.array_equal(after['action_history'],expected),'exact normalized history')
    return raw.tolist()


def feasible(x,cap):
    x=core.array(x,(10,)).copy();n=np.linalg.norm(x)
    if n>cap:x*=cap/n
    x=x.astype(np.float32)
    if np.linalg.norm(x.astype(float))>cap:x*=np.float32(1-4*np.finfo(np.float32).eps)
    require(np.linalg.norm(x.astype(float))<=cap,'feasible float32 packet')
    return x.astype(float)


def check_compile(task,j,z,c):
    """Independent ridge/KKT arithmetic, not a call to producer compiler.

    Check saved singular system against A.T A and solve against damped normal
    equation. Do not require arbitrary null/equal-SVD bases to be unique.
    """
    a=(np.asarray(task['projection'])@core.array(j,(192,10)))/np.sqrt(192)
    r=np.asarray(task['projection'])@(np.asarray(task['goal'])-core.array(z,(192,)))/np.sqrt(192)
    v=core.array(c['right_vectors'],(10,10));s=core.array(c['singular_values'],(10,))
    require((s>=0).all() and (np.diff(s)<=1e-12).all(),'ordered nonnegative singular values')
    near(v@v.T,np.eye(10),'orthogonal saved singular vectors')
    near((a.T@a)@v.T,v.T*(s*s)[None,:],'singular system for THIS context response')
    require(all(row[np.argmax(np.abs(row))]>=0 for row in v),'canonical spectral signs')
    cap=task['packet_logit_cap'];lam=c['ball_multiplier']
    require(type(lam) in (float,int) and np.isfinite(lam) and lam>=0 and c['radius']==cap,'ridge multiplier/radius')
    exact=np.linalg.solve(a.T@a+(.002+lam)*np.eye(10),a.T@r)
    # Saved step has deterministic float32 feasibility rounding.
    near(c['step'],feasible(exact,cap),'bounded ridge proposal')
    require(c['objective']=='mean_square_linear_residual_plus_damping_times_step_norm_squared','same damping units')
    arithmetic=c['arithmetic']
    require(set(arithmetic)=={'coefficient_evaluations','bracket_doublings','bisection_iterations'},'CPU solve counters')
    for n in arithmetic.values():core.integer(n)
    if lam==0:
        require(arithmetic==dict(coefficient_evaluations=1,bracket_doublings=0,bisection_iterations=0),'unconstrained solve work')
        require(np.linalg.norm(exact)<=cap+1e-10,'unconstrained ball')
    else:
        require(arithmetic['bisection_iterations']==70 and arithmetic['bracket_doublings']<=1024 and
            arithmetic['coefficient_evaluations']==73+arithmetic['bracket_doublings'],'bounded solve work')
        require(abs(np.linalg.norm(exact)-cap)<=1e-8,'active ball KKT')


def ledger(events,reported,core_events,complete):
    starts=[];pending={};finished=set();stages={}
    for outer in events:
        if outer['kind'] not in ('operation_attempted','operation_completed'):continue
        e=outer['event'];core.fields(e,'index phase evaluation_id resource samples completed','operation receipt')
        phase=e['phase'];resource=e['resource'];i=core.integer(e['index'])
        require(phase in EXPECTED and resource in EXPECTED[phase] and e['evaluation_id']==phase and
                e['samples']==1 and type(e['samples']) is int and type(e['completed']) is bool,'scoped operation')
        if outer['kind']=='operation_attempted':
            require(i==len(starts) and e['completed'] is False,'ordered nonrefundable attempt')
            starts.append(deepcopy(e));pending[i]=deepcopy(e)
            stages[i]=outer['stage']
        else:
            require(i in pending and i not in finished and e['completed'] is True and outer['stage']==stages[i],
                'unique completion after actual attempt in same stage')
            expected=dict(pending.pop(i),completed=True)
            require(e==expected,'completion identity')
            starts[i]['completed']=True;finished.add(i)
    require(starts==reported,'durable events / report extended ledger exact')
    narrow=[dict(e,index=i) for i,e in enumerate(e for e in starts if e['resource'] in core.RESOURCES)]
    require(narrow==core_events,'core ledger is exact narrow projection')
    work={p:{r:dict(attempted=sum(e['samples'] for e in starts if e['phase']==p and e['resource']==r),
        completed=sum(e['samples'] for e in starts if e['phase']==p and e['resource']==r and e['completed']))
        for r in caps} for p,caps in EXPECTED.items()}
    for p,caps in EXPECTED.items():
        for r,n in caps.items():
            require(work[p][r]['attempted']<=n,'nonrefundable phase budget exceeded')
            if complete:require(work[p][r]==dict(attempted=n,completed=n),'complete exact resource budget')
    by_stage={}
    for e in starts:
        k=stages[e['index']];d=by_stage.setdefault(k,{r:0 for r in EXPECTED['producer']})
        d[e['resource']]+=1
    if complete:
        expected={
            'root_nominal':dict(predict=6,action_encoder=6),
            'root_fullH2_J':dict(predict=2,action_encoder=2,jacobian=1,vjp_directions=192,svd=1,linear_solve=1,prefix_generation=1),
            'independent_winner_H2_replays':dict(predict=6,action_encoder=6)}
        for i in range(16):expected[f'prefix_{i}']=dict(predict=7,action_encoder=7,jacobian=1,vjp_directions=192,
            svd=1,linear_solve=1,suffix_proposal=4,selection=4)
        require(set(by_stage)==set(expected),'all17 compile stages and independent verification')
        for stage,values in expected.items():
            require(by_stage[stage]=={r:values.get(r,0) for r in EXPECTED['producer']},'exact stage budget '+stage)
    return work


def check_root(task,r,events):
    core.validate_task(task);require(r['task_id']==core.task_id(task),'stable task identity')
    require(r['protocol_sha256']==ALGORITHM_SHA and r['budget']==EXPECTED,'fixed producer protocol/budget')
    require(r['P1_admission'] is False and r['model_task_eligible'] is False and r['physical_future_labels_read'] is False and
        r['direct_plateau_established_by_this_producer'] is False,'no admission/stall inference')
    complete=r['status']=='witness_completed_not_admitted'
    require(complete or r['status']=='failed','recognized terminal root status')
    if complete:require(r['error'] is None and not r['secondary_errors'],'complete error-free root')
    else:require(r['error'] is not None,'failed root primary evidence')
    work=ledger(events,r['extended_journal'],r['journal'],complete)
    require(work==r['resources'],'reported actual resource accounting')
    completed_candidates=[e['candidate'] for e in events if e['kind']=='candidate_complete']
    finished=[c for c in r['candidates'] if 'cost' in c]
    require(completed_candidates==finished,'all nonlinear candidates bound to durable journal')
    require(len(r['candidates'])<=64 and len(finished)<=64,'finite candidate census')
    best=None
    for k,c in enumerate(finished):
        require(c['index']==k and c['prefix_index']==k//4 and c['suffix_index']==k%4,'stable all-slot identity')
        require(len(c['contexts'])==2,'full H2 only, not partial suffix')
        mid,end=c['contexts']
        raw=packet(task,task['root_context'],c['logits'][0],mid)+packet(task,mid,c['logits'][1],end)
        require(c['commands']==raw and c['cost']==core.model_cost(task,end['latent_history'][-1]),'same-Q nonlinear terminal / raw actions')
        if best is None or c['cost']<best['cost']:best=c
        ev=[e for e in events if e['kind']=='candidate_complete'][k]
        require(ev['incumbent_index']==best['index'],'stable nonlinear incumbent')
    if not complete:
        return dict(status='partial_saved_evidence',complete_H2=False,completed_candidates=len(finished),
            resources=work,root_error=r['error'],P1_admission=False,model_task_eligible=False,
            limitations=['partial arrays retained; no complete recompilation/winner conclusion'])
    require(len(finished)==64 and len(r['prefix_compilations'])==16 and r['winner']==best,'complete stable64 selection')
    started=one(events,'producer_started')
    require(started['task_id']==core.task_id(task) and core.digest(started['protocol'])==ALGORITHM_SHA and started['budgets']==EXPECTED,'runtime predeclaration')
    root_stage='root_fullH2_J';rootJ=one(events,'root_fullH2_J_complete',root_stage)
    j=core.array(rootJ['jacobian'],(192,20));z=core.array(rootJ['terminal'],(192,))
    jrows=[e for e in events if e['kind']=='jacobian_row' and e.get('stage')==root_stage]
    require([e['row'] for e in jrows]==list(range(192)),'all root fullH2 reverse rows')
    near([e['raw_control_derivative'] for e in jrows],j,'root zero-logit tanh chain / full J rows')
    rootpred=one(events,'prediction_complete','root_nominal')['prediction'];core.check_prediction(task,rootpred)
    require(rootpred['logits']==np.zeros((2,10)).tolist() and np.array_equal(z,rootpred['contexts'][-1]['latent_history'][-1]),'root fullH2 nominal binding')
    forward=one(events,'jacobian_forward_complete',root_stage);joint_event=one(events,'joint_logit_jacobian',root_stage)
    require(forward['terminal']==joint_event['terminal']==rootJ['terminal'] and forward['logits']==rootpred['logits'] and
        joint_event['jacobian']==rootJ['jacobian'],'root AD forward / row chain / full J aliases')
    rootcomp=one(events,'compiled_response',root_stage)['receipt'];check_compile(task,j[:,:10],z,rootcomp)
    prefix_event=one(events,'prefix_slots_frozen',root_stage);prefixes=prefix_event['logits']
    expected=[np.zeros(10).tolist()]+[feasible(s*np.asarray(rootcomp['step']),task['packet_logit_cap']).tolist() for s in (.25,.5,1.)]
    for radius in (1/3,1.):
        for mode in range(3):
            for sign in (-1,1):expected.append(feasible(sign*radius*task['packet_logit_cap']*np.asarray(rootcomp['right_vectors'][mode]),task['packet_logit_cap']).tolist())
    require(prefixes==expected,'all16 predeclared prefix slots, no posthoc bank')
    matrices=[];contexts=[];root_suffix_differences=[]
    for i,d in enumerate(r['prefix_compilations']):
        stage=f'prefix_{i}';actual=one(events,'actual_prefix',stage);mid=finished[4*i]['contexts'][0]
        require(actual['index']==d['index']==i and actual['logits']==prefixes[i] and actual['context']==mid,'actual prefix identity')
        h=core.digest(mid);contexts.append(h);derivative=d['derivative']
        require(d['input_context_sha256']==derivative['input_context_sha256']==h and derivative['task_id']==r['task_id'] and
            derivative['at_logits']==[0.]*10,'suffix J bound to THIS actual full prefix context')
        packets=[e for e in events if e['kind']=='packet_completed' and e.get('stage')==stage]
        require([e['label'] for e in packets]==['actual_prefix','suffix_nominal']+[f'candidate_{i}_{k}' for k in range(4)],'all paid prefix/nominal/nonlinear packet receipts')
        require(packets[0]['context']==mid and packets[0]['input_context_sha256']==core.digest(task['root_context']) and packets[0]['logits']==prefixes[i] and
            packets[0]['commands']==packet(task,task['root_context'],prefixes[i],mid),'prefix native packet receipt')
        for e in packets[1:]:
            require(e['input_context_sha256']==h and e['commands']==packet(task,mid,e['logits'],e['context']),'own-context native packet receipt')
        nominal=packets[1]['context'];ad=one(events,'suffix_AD_forward',stage)
        require(ad['input_context_sha256']==h and ad['context']==nominal and derivative['terminal']==nominal['latent_history'][-1],'paid zero nominal / AD forward exact')
        require(packets[1]['logits']==[0.]*10,'zero suffix nominal')
        jevent=one(events,'suffix_J_complete',stage)
        require(jevent['derivative']==derivative,'local J complete receipt')
        rows=[e for e in events if e['kind']=='suffix_J_row' and e.get('stage')==stage]
        require([e['row'] for e in rows]==list(range(192)),'all192 local reverse directions')
        local=core.array(derivative['jacobian'],(192,10));near([e['derivative'] for e in rows],local,'local zero tanh-chain J rows')
        require(one(events,'compiled_response',stage)['receipt']==d['compilation'],'local saved solve')
        check_compile(task,local,derivative['terminal'],d['compilation']);matrices.append(local)
        root_suffix_differences.append(float(np.linalg.norm(local-j[:,10:])))
        for k,scale in enumerate((0.,.25,.5,1.)):
            c=finished[4*i+k];e=packets[2+k]
            require(c['logits'][0]==prefixes[i] and c['contexts'][0]==mid and c['contexts'][1]==e['context'] and
                c['logits'][1]==e['logits']==feasible(scale*np.asarray(d['compilation']['step']),task['packet_logit_cap']).tolist(),'all four fixed own-context suffix proposals')
    w=r['witness'];core.check_prediction(task,w)
    require(w['logits']==best['logits'] and w['commands']==best['commands'] and w['contexts']==best['contexts'] and w['cost']==best['cost'],'independent exact winner H2 replay')
    wevent=one(events,'prediction_complete','independent_winner_H2_replays')
    require(wevent['prediction']==w,'winner bound to durable independent evaluation')
    for stage,pred in [('root_nominal',rootpred),('independent_winner_H2_replays',w)]:
        paths=[e for e in events if e['kind']=='prediction_path' and e.get('stage')==stage]
        require([e['mode'] for e in paths]==['nominal','whole_replay','segmented_replay'] and
            all(e['contexts']==pred['contexts'] for e in paths),'three independently paid complete H2 path receipts')
        native=[e for e in events if e['kind']=='native_packet' and e.get('stage')==stage]
        require([(e['mode'],e['packet']) for e in native]==[(mode,k) for mode in ('nominal','whole_replay','segmented_replay') for k in range(2)] and
            all(e['context']==pred['contexts'][e['packet']] for e in native),'six independent native packet receipts')
    final=one(events,'producer_finished')
    require(final['status']==r['status'] and final['error'] is None and final['resources']==work and final['witness']==w,'durable root terminal receipt')
    pairwise=[float(np.linalg.norm(a-b)) for k,a in enumerate(matrices) for b in matrices[k+1:]]
    return dict(status='complete_saved_arrays_consistent',complete_H2=True,completed_candidates=64,
        resources=work,unique_prefix_contexts=len(set(contexts)),unique_suffix_J_arrays=len({core.digest(a.tolist()) for a in matrices}),
        suffix_J_pairwise_frobenius=dict(min=min(pairwise),max=max(pairwise),exact_different_pairs=sum(x>0 for x in pairwise)),
        suffix_J_vs_initial_H2_suffix_frobenius=root_suffix_differences,
        winner_index=best['index'],winner_model_Q_cost=best['cost'],P1_admission=False,model_task_eligible=False,
        direct_plateau_established=False,limitations=['saved arrays/journal consistency, not independent AD finite-difference verification',
        'model prediction only; no actual pose/official score/physical Q or physical consequence',
        'no direct stall / independent threshold authority / fresh confirmation'])


def audit_saved(prepared,prepare_sha,actual,actual_sha):
    p=producer.verify(prepared,prepare_sha)
    require(core.digest(p['algorithm'])==ALGORITHM_SHA and p['budget_per_root']==EXPECTED,'independent fixed protocol identity')
    directory=Path(actual).absolute();report_path=directory/'report.json'
    require(producer.io.sha(report_path)==actual_sha,'external terminal report SHA')
    r=producer.io.read(report_path)
    require(r['schema']==producer.NAME+'.report' and r['prepare_sha256']==prepare_sha and r['algorithm_sha256']==ALGORITHM_SHA,'terminal prepared/source identity')
    require(r['status'] in ('cuda_completed_not_admitted','partial_or_failed') and r['P1_admission'] is False and
        r['model_task_eligible'] is False and r['physical_future_labels_read'] is False and r['direct_plateau_established_by_this_producer'] is False,
        'terminal non-admission scope')
    complete=r['status']=='cuda_completed_not_admitted'
    require(len(r['roots'])<=2 and (not complete or len(r['roots'])==2),'all original roots or explicit partial')
    if complete:require(not r['errors'] and r['model_unchanged'] is True and r['source_verified'] is True and
        r['model_hash_before']==r['model_hash_after'] and r['model_hash_before'] is not None,'complete actual provenance')
    require(r['inherited']==p['inputs']['inherited'] and r['source_purpose']==p['inputs']['source_purpose'] and
        r['history_semantics']==p['inputs']['history_semantics'] and r['new_image_encoder_calls']==r['new_physical_steps']==0,'inherited acquisition and no new physics')
    rows=[];bindings={str(report_path):actual_sha}
    for i,summary in enumerate(r['roots']):
        item=p['inputs']['tasks'][i]
        require(summary['root_sha256']==item['root_sha256'] and summary['task_id']==item['task_id'],'stable source-root order/identity')
        root_path=directory/f'root_{i}.json';journal_path=directory/f'root_{i}.events.jsonl'
        require(summary['report_path']==str(root_path),'exact local root report path')
        if not complete and ('report_sha256' not in summary or 'journal_sha256' not in summary):
            rows.append(dict(root_sha256=item['root_sha256'],status='partial_unfinalized_artifacts',complete_H2=False));continue
        require(summary['journal_path']==str(journal_path) and producer.io.sha(root_path)==summary['report_sha256'] and
                producer.io.sha(journal_path)==summary['journal_sha256'],'bound root report/journal hashes')
        require(root_path.stat().st_size<=8*1024**2 and journal_path.stat().st_size<=16*1024**2,'bounded saved evidence')
        root=producer.io.read(root_path);events=[]
        with journal_path.open('rb') as stream:
            for line in stream:
                require(len(line)<=8*1024**2,'bounded durable event');events.append(producer.io.io.decode(line))
        result=check_root(item['task'],root,events)
        require(summary['status']==root['status'] and summary['error']==root['error'] and summary['resources']==root['resources'],'top/root report agreement')
        rows.append(dict(root_sha256=item['root_sha256'],task_id=item['task_id'],**result))
        bindings.update({str(root_path):summary['report_sha256'],str(journal_path):summary['journal_sha256']})
    full=complete and len(rows)==2 and all(row['complete_H2'] for row in rows)
    total={p0:{res:dict(attempted=sum(row.get('resources',{}).get(p0,{}).get(res,{}).get('attempted',0) for row in rows),
        completed=sum(row.get('resources',{}).get(p0,{}).get(res,{}).get('completed',0) for row in rows)) for res in caps}
        for p0,caps in EXPECTED.items()}
    if full:
        require(sum(total[phase]['predict']['completed'] for phase in EXPECTED)==252,'all252 actual predict calls')
    for f,h in bindings.items():require(producer.io.sha(f)==h,'saved evidence drift during arithmetic audit')
    producer.verify(prepared,prepare_sha)
    own={str(path):producer.io.sha(path) for path in [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]}
    return dict(schema=NAME+'.report',status='saved_arrays_audited_not_admitted' if full else 'partial_saved_evidence_not_admitted',
        prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha,actual_report_sha256=actual_sha,
        input_bindings=bindings,auditor_sources_sha256=own,roots=rows,actual_new_resources=total,
        resource_totals_complete=full,
        inherited=r['inherited'],P1_admission=False,model_task_eligible=False,direct_plateau_established=False,
        no_neural_or_physics_execution=True,errors_from_actual=r['errors'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--prepare-sha',required=True)
    parser.add_argument('--actual',type=Path,required=True);parser.add_argument('--actual-sha',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(not args.output.exists(),'NEW arithmetic report only')
    result=audit_saved(args.prepared,args.prepare_sha,args.actual,args.actual_sha)
    raw=(core.canonical(result)+'\n').encode();require(len(raw)<1024**2,'small arithmetic report')
    require(shutil.disk_usage(args.output.parent).free>=200*1024**2+len(raw),'200MiB preserved')
    with args.output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(core.canonical(dict(status=result['status'],output=str(args.output.absolute()),sha256=producer.io.sha(args.output))))


if __name__=='__main__':main()
