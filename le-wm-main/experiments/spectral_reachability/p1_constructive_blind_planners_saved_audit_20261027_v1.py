"""Independent exported-array/planner journal audit. No planner/model imports.

Reads explicit latent exports, source hashes and saved planner artifacts only.
Never reads blind NPZ, images, sourcepack, certificate or physical outcomes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import zlib
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_encode_saved_audit_20261027_v1 as io

NAME='p1_constructive_blind_planners_saved_audit_20261027_v1'
PLANNER='p1_constructive_blind_planners_cuda_20261027_v1'
PLANNER_SHA='e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27'
ARMS=('line_search_GN','full10_conditional_D2P')
SCALES=(1.,.5,.25,.125,.0625)
CAP=float(.75*np.sqrt(5))
BUDGETS={ARMS[0]:dict(predict=78,action_encoder=78,jacobian=4,vjp_directions=768,svd=4,linear_solve=4,proposal=4,selection=4),
         ARMS[1]:dict(predict=66,action_encoder=66,jacobian=11,svd=12)}
API={ARMS[0]:dict(grad=768,jacobian=0),ARMS[1]:dict(grad=11,jacobian=11)}
need=io.need


def load(path, expected=None, limit=32*1024**2):
    p=Path(path); need(p.is_file() and not p.is_symlink() and p.stat().st_size<=8*1024**2,'bounded regular artifact')
    raw=p.read_bytes()
    if expected is not None: need(io.hash_bytes(raw)==expected,'external artifact SHA '+p.name)
    if p.suffix=='.gz':
        d=zlib.decompressobj(31); raw=d.decompress(raw,limit+1)
        need(len(raw)<=limit and d.eof and not d.unused_data and not d.unconsumed_tail,'single complete bounded gzip')
    need(len(raw)<=limit,'expanded cap')
    return raw


def read(path,expected=None): return io.parse(load(path,expected))
def arr(v,shape):
    a=np.asarray(v)
    need(a.dtype.kind in 'fi' and a.shape==shape and np.isfinite(a).all(),'finite array '+str(shape))
    return a.astype(float)


def close(a,b,label):
    # Numerical arithmetic reconstruction tolerance only, never an efficacy gate.
    need(np.allclose(a,b,rtol=1e-8,atol=1e-10),label)


def counts(values): return {k:dict(attempted=v,completed=v) for k,v in values.items()}
def tokens(t,commands):
    raw=arr(commands,(10,2)).astype(np.float32).reshape(2,10)
    need((np.abs(raw)<=1).all(),'bounded commands')
    return (raw-np.tile(np.asarray(t['action_mean'],np.float32),5))/np.tile(np.asarray(t['action_std'],np.float32),5)


def context(c):
    need(set(c)==set('layout context_length latent_dim action_dim latent_history action_history'.split()),'context exact keys')
    need(c['layout']=='states_with_between_state_actions_v1' and c['context_length']==3 and c['latent_dim']==192 and c['action_dim']==10,'context layout')
    arr(c['latent_history'],(3,192)); arr(c['action_history'],(2,10))


def path_check(t,commands,path):
    need(len(path)==2,'complete H2 path')
    h=tokens(t,commands); before=t['root_context']; context(before)
    for k,c in enumerate(path):
        context(c)
        need(c['latent_history'][:2]==before['latent_history'][1:],'causal latent shift')
        need(np.array_equal(c['action_history'],np.concatenate((before['action_history'],h[k:k+1]))[-2:]),'normalized action content chronology')
        before=c


def cost(t,z):
    residual=arr(t['projection'],(192,192))@(arr(z,(192,))-arr(t['goal'],(192,)))
    return float(np.mean(residual**2))


def prediction(t,p,replays=False):
    need(p['task_id']==io.task_hash(t),'prediction taskID')
    x=arr(p['logits'],(2,10)); need((np.linalg.norm(x,axis=1)<=CAP).all(),'feasible logit packets')
    need(np.array_equal(p['commands'],np.tanh(x.astype(np.float32)).reshape(10,2)),'exact NumPy float32 tanh mapping')
    path_check(t,p['commands'],p['contexts'])
    need(p['cost']==cost(t,p['contexts'][-1]['latent_history'][-1]),'own-Q terminal cost')
    if replays: need(p['whole_replay']==p['contexts'][-1] and p['segmented_replay']==p['contexts'],'whole/segmented identity')


def export_task(path,expected):
    e=read(path,expected); t=e['task']; tc=e['task_contract']
    need(e['schema']==io.PRODUCER+'.task_export' and e['task_id']==io.task_hash(t) and t['task_contract_sha256']==io.task_hash(tc),'export task/contract ID')
    need(e['native_root_sha256'] is None and e['with_target'] is False and tc['with_target'] is False and
         e['blind_sha256']==e['blind_source_sha256']==tc['blind_source_sha256'],'no invented native root')
    need(t['environment']=='swm/PushT-v1' and t['representation']=='native_projected_cls' and t['feedback']=='open_loop_model_only' and
         (t['horizon'],t['packet_steps'],t['control_dim'],t['latent_dim'])==(2,5,2,192),'native H2 task contract')
    need(t['model_sha256']==io.MODELS[str(io.POLICY/'weights.pt')] and t['model_config_sha256']==io.MODELS[str(io.POLICY/'config.json')],'frozen model identity')
    need(np.array_equal(t['projection'],np.eye(192)) and t['packet_logit_cap']==CAP,'fixed identity Q/cap')
    need(np.array_equal(np.asarray(t['action_mean'],np.float32),np.asarray([-.0087,.0068],np.float32)) and
         np.array_equal(np.asarray(t['action_std'],np.float32),np.asarray([.2019,.2002],np.float32)),'original normalizer')
    need(not arr(t['root_raw_action_history'],(10,2)).any(),'actual zero history')
    context(t['root_context']); need(np.array_equal(t['root_context']['action_history'],tokens(t,t['root_raw_action_history'])),'observed history token content')
    for x in (arr(t['goal'],(192,)),arr(t['root_context']['latent_history'],(3,192))):
        need(np.array_equal(x,x.astype(np.float32)),'native float32 latent')
    return t


def project(x):
    a=arr(x,(2,10)).copy()
    for row in a:
        n=np.linalg.norm(row)
        if n>CAP: row*=CAP/n
    a=a.astype(np.float32)
    for row in a:
        if np.linalg.norm(row.astype(float))>CAP: row*=np.float32(1-4*np.finfo(np.float32).eps)
    return a.astype(float)


def lane(events,attempt,returned,key,expected):
    pending={}; done=[]; sequence=[]; stack=[]
    for pos,e in enumerate(events):
        if e['kind'] not in (attempt,returned): continue
        v=e[key]; index=v['index']
        if e['kind']==attempt:
            need(type(index) is int and index==len(sequence) and v['completed'] is False,'ordered unique attempt')
            pending[index]=(v,pos); sequence.append(v); stack.append(index)
        else:
            need(index in pending and v['completed'] is True and stack and stack.pop()==index,'return must have unique nested attempt')
            old,start=pending.pop(index); want=dict(old,completed=True)
            need(v==want and pos>start,'same entry completed')
            done.append(v)
    need(not pending,'unreturned work/partial')
    actual={k:dict(attempted=sum(v.get('resource',v.get('kind'))==k for v in sequence),
                        completed=sum(v.get('resource',v.get('kind'))==k for v in done)) for k in expected}
    need(len(sequence)==sum(expected.values()) and actual==counts(expected),'exact full ledger, unknown work not ignored')
    return sequence


def gn_audit(t,result,events):
    need(result['task_id']==io.task_hash(t) and result['resources']==counts(BUDGETS[ARMS[0]]),'GN result task/resources')
    evaluations=[e for e in events if e['kind']=='evaluation_complete']
    requests=[e for e in events if e['kind']=='evaluation_request']
    groups=['initial']*3+[g for i in range(1,5) for g in [*[f'iter_{i}_scale_{j}' for j in range(5)],*[f'iter_{i}_selected_replay']*3]]
    modes=['nominal','whole_replay','segmented_replay']+[m for _ in range(4) for m in ['whole_replay']*5+['nominal','whole_replay','segmented_replay']]
    need(len(evaluations)==len(requests)==35,'all 35 fresh H2 evaluations')
    packets=[e for e in events if e['kind']=='native_packet']
    need(len(packets)==70,'all70 saved evaluation packet contexts')
    for n,(e,r,g,m) in enumerate(zip(evaluations,requests,groups,modes)):
        p=e['prediction']; prediction(t,p)
        need(e['evaluation_id']==r['evaluation_id']==g and e['mode']==r['mode']==m and
             r['logits']==p['logits'] and r['commands']==p['commands'],'request/result input binding')
        need([dict(packet=k,context=p['contexts'][k],mode=m,evaluation_id=g,kind='native_packet') for k in (0,1)]==packets[2*n:2*n+2],'native full-path event binding')
    # Check the request/path/complete chronology, not just three list censuses.
    stream=[e for e in events if e['kind'] in ('evaluation_request','native_packet','evaluation_complete')]
    need([e['kind'] for e in stream]==['evaluation_request','native_packet','native_packet','evaluation_complete']*35,'causal evaluation chronology')
    values=[e['prediction'] for e in evaluations]
    need(values[0]==values[1]==values[2] and not arr(values[0]['logits'],(2,10)).any(),'zero initial triple')
    incumbent=values[0]; steps=[]
    derivatives=[e for e in events if e['kind']=='derivative_complete']
    proposals=[e for e in events if e['kind']=='line_proposals_frozen']
    selections=[e for e in events if e['kind']=='selection_complete']
    updates=[e for e in events if e['kind']=='update_verified']
    need(len(derivatives)==len(proposals)==len(selections)==len(updates)==4,'four GN updates')
    lifecycle=[e['kind'] for e in events if e['kind'] in ('initial_verified','derivative_complete','line_proposals_frozen','selection_complete','update_verified')]
    need(lifecycle==['initial_verified']+['derivative_complete','line_proposals_frozen','selection_complete','update_verified']*4,'GN causal freeze/select/update order')
    need(next(e for e in events if e['kind']=='initial_verified')['incumbent']['contexts']==incumbent['contexts'],'initial saved incumbent')
    for it,(d,p,s,u) in enumerate(zip(derivatives,proposals,selections,updates),1):
        d=d['derivative']; j=arr(d['raw_terminal_jacobian'],(192,20))
        need(d['task_id']==io.task_hash(t) and d['at_logits']==incumbent['logits'] and d['at_terminal']==incumbent['contexts'][-1]['latent_history'][-1],'fresh derivative point')
        r=arr(t['projection'],(192,192))@(arr(d['at_terminal'],(192,))-np.asarray(t['goal'])); a=np.asarray(t['projection'])@j
        close(d['objective_gradient'],2*a.T@r/192,'GN gradient')
        group=f'iter_{it}_AD'
        rows=[e for e in events if e['kind']=='jacobian_row' and e['evaluation_id']==group]
        need([e['row'] for e in rows]==list(range(192)),'all192 raw VJP rows per iteration')
        rawj=arr([e['raw_control_derivative'] for e in rows],(192,20))
        need(np.array_equal(j,rawj*(1-np.asarray(incumbent['commands']).reshape(20)**2)[None]),'single tanh chain, all J rows')
        joint=[e for e in events if e['kind']=='joint_logit_jacobian' and e['evaluation_id']==group]
        forward=[e for e in events if e['kind']=='jacobian_forward_complete' and e['evaluation_id']==group]
        need(len(joint)==len(forward)==1 and joint[0]['jacobian']==j.tolist() and joint[0]['terminal']==d['at_terminal'] and
             forward[0]['terminal']==d['at_terminal'] and forward[0]['logits']==incumbent['logits'],'saved AD forward/J binding')
        adpath=[e['context'] for e in events if e['kind']=='jacobian_forward_packet' and e['evaluation_id']==group]
        path_check(t,incumbent['commands'],adpath); need(adpath==incumbent['contexts'],'fresh AD fullH2 path')
        left,sv,right=np.linalg.svd(a/np.sqrt(192),full_matrices=False)
        delta=-right.T@(sv*(left.T@(r/np.sqrt(192)))/(sv*sv+.002))
        close(p['delta'],delta,'ridge GN direction')
        close(p['endpoint'],project(np.asarray(incumbent['logits'])+delta.reshape(2,10)),'projected GN endpoint')
        need(p['iteration']==it and p['scales']==list(SCALES) and p['incumbent_logits']==incumbent['logits'],'fixed all scales')
        endpoint=arr(p['endpoint'],(2,10)); x=arr(incumbent['logits'],(2,10))
        trials=[(x+scale*(endpoint-x)).astype(np.float32).astype(float).tolist() for scale in SCALES]
        need(p['logits']==trials,'fixed float32 feasible segment')
        start=3+(it-1)*8; candidates=values[start:start+5]
        need([c['logits'] for c in candidates]==trials,'every scale fresh evaluated')
        pool=[incumbent]+candidates; winner=min(range(6),key=lambda k:(pool[k]['cost'],k))
        row=dict(iteration=it,incumbent_cost=incumbent['cost'],trial_costs=[c['cost'] for c in candidates],selected_index=winner,
                 selected_scale=None if winner==0 else SCALES[winner-1],selected_cost=pool[winner]['cost'])
        need(all(s[k]==v for k,v in row.items()),'stable all-scale model selection')
        replay=values[start+5:start+8]; need(replay==[pool[winner]]*3 and u['prediction']['contexts']==pool[winner]['contexts'],'selected fresh triple')
        prediction(t,u['prediction'],True); incumbent=pool[winner]; steps.append(row)
    need(result['steps']==steps,'GN full step result')
    prediction(t,result['final_prediction'],True)
    need(all(result['final_prediction'][k]==v for k,v in incumbent.items()),'GN final incumbent')
    return dict(task_id=io.task_hash(t),logits=incumbent['logits'],commands=incumbent['commands'],prediction=incumbent['contexts'][-1],model_terminal_cost=incumbent['cost'])


def ridge(k,r):
    left,s,right=np.linalg.svd(k,full_matrices=False); target=left.T@r
    def solve(lam): return right.T@(s*target/(s*s+.002+lam))
    lam=0.; v=solve(lam)
    if np.linalg.norm(v)>CAP:
        low,high=0.,1.
        while np.linalg.norm(solve(high))>CAP: high*=2
        for _ in range(70):
            mid=(low+high)/2
            if np.linalg.norm(solve(mid))>CAP: low=mid
            else: high=mid
        lam=high; v=solve(lam)
    remaining=r-k@v
    return v,float((remaining@remaining+.002*(v@v))/192),s,lam


def d2p_audit(t,result,events):
    need(result['status']=='completed' and result['q_sha256']==io.task_hash(t['projection']),'D2P completed own Q hash')
    # D2P digest traces to quality_cost_audit.canonical: sorted JSON default spaces.
    q=np.asarray(t['projection']); goal=np.asarray(t['goal']); initial=t['root_context']
    rootj=arr(result['root_jacobian'],(192,10)); rootn=arr(result['root_nominal'],(192,))
    bank_events=[e for e in events if e['kind']=='bank']; need(len(bank_events)==1,'single root spectral bank')
    need(bank_events[0]['root_jacobian']==result['root_jacobian'] and bank_events[0]['root_nominal']==result['root_nominal'] and bank_events[0]['bank']==result['bank'],'root bank/J saved binding')
    nominal=[e for e in events if e['kind']=='nominal']; derivative=[e for e in events if e['kind']=='derivative']
    need([e['phase'] for e in nominal]==[e['phase'] for e in derivative]==['root']+[f'suffix_{i}' for i in range(10)],'fresh root+ten suffix derivatives')
    need(nominal[0]['nominal']==rootn.tolist() and derivative[0]['jacobian']==rootj.tolist(),'root nominal/J')
    _,singular,right=np.linalg.svd(q@rootj,full_matrices=False)
    for vector in right:
        if vector[np.argmax(np.abs(vector))]<0: vector*=-1
    progress,_,_,_=ridge(q@rootj,q@(goal-rootn))
    params=[np.zeros(10),progress]; kinds=['zero','root_dls_progress']
    for radius in (float(np.sqrt(5)*.25),CAP):
        for mode in range(2):
            for sign in (-1,1):
                params.append(sign*radius*right[mode]); kinds.append(f'spectral_mode_{mode}_radius_{radius:.8f}_sign_{sign}')
    bank=result['bank']; close(bank['params'],params,'root spectral/DLS bank'); close(bank['singular_values'],singular,'bank singular values')
    close(bank['right_vectors'],right,'canonical spectral directions'); need(bank['kinds']==kinds,'fixed bank order')
    candidates=result['candidates']; need(len(candidates)==10,'all ten candidates, repeats retained')
    prefix_events=[e for e in events if e['kind']=='prefix']
    completed=[e for e in events if e['kind']=='candidate' and e['completed']]
    need(completed==candidates and [e['index'] for e in prefix_events]==list(range(10)),'all candidate completions/prefixes')
    for index,c in enumerate(candidates):
        need(c['index']==index and c['proposal_kind']==kinds[index] and c['prefix_params']==bank['params'][index],'bank candidate coverage')
        need(c['nominal_terminal']==nominal[index+1]['nominal'] and c['jacobian']==derivative[index+1]['jacobian'],'fresh conditional J binding')
        prefix=prefix_events[index]['context']; path_check_prefix=prefix['latent_history'][:2]==initial['latent_history'][1:]
        need(path_check_prefix and prefix['latent_history'][-1]==c['predicted_prefix'],'prefix latent chronology')
        j=arr(c['jacobian'],(192,10)); n=arr(c['nominal_terminal'],(192,)); residual=q@(goal-n)
        v,reach,s,lam=ridge(q@j,residual); close(c['suffix_params'],v,'conditional bounded ridge action')
        solve=c['suffix_solve']; close(solve['singular_values'],s,'suffix singulars'); close(solve['ball_multiplier'],lam,'ridge ball multiplier')
        close(c['own_reach_value'],reach,'ridge own reach'); need(c['suffix_params']==solve['params'] and c['own_reach_value']==solve['own_reach_value'],'ridge score/action identity')
        v_saved=arr(c['suffix_params'],(10,)); rem=residual-q@j@v_saved
        need(solve['linear_terminal_cost']==float(rem@rem/192) and solve['ridge_penalty']==float(.002*(v_saved@v_saved)/192) and
             solve['own_reach_value']==float((rem@rem+.002*(v_saved@v_saved))/192),'exact saved ridge costs')
        logits=[c['prefix_params'],c['suffix_params']]; commands=np.tanh(np.asarray(logits,np.float32)).reshape(10,2)
        need(np.array_equal(c['commands'],commands) and np.array_equal(c['prefix_commands'],commands[:5]) and np.array_equal(c['suffix_commands'],commands[5:]),'full ten action mapping')
        path_check(t,c['commands'],[prefix,c['context']])
        need(c['predicted_terminal']==c['context']['latent_history'][-1] and c['stage_cost']==cost(t,c['predicted_prefix']) and
             c['nonlinear_terminal_cost']==cost(t,c['predicted_terminal']) and c['objective']==c['stage_cost']+c['own_reach_value'],'all ten nonlinear/selection costs')
        need(c['completed'] is True and c['independent_model_replay_exact'] is True,'candidate full replay')
        replay=[e for e in events if e['kind']=='replay' and e['phase']==f'candidate_{index}']
        need(len(replay)==2 and replay[0]['completed'] is False and replay[1]['completed'] is True and
             replay[0]['commands']==replay[1]['commands']==c['commands'] and replay[1]['context']==c['context'],'candidate independent replay event')
    w=min(candidates,key=lambda c:(c['objective'],c['index']))
    need(result['winner']==w and result['whole_and_segmented_exact'] is True and
         result['whole_replay_context']==result['segmented_replay_context']==w['context'],'stable objective winner and final replay')
    winner_replays=[e for e in events if e['kind']=='replay' and e['phase'].startswith('winner_') and e['completed']]
    need([e['phase'] for e in winner_replays]==['winner_whole','winner_segment_0','winner_segment_1'],'all final replay branches')
    wp=prefix_events[w['index']]['context']
    need([e['context'] for e in winner_replays]==[w['context'],wp,w['context']] and
         [e['commands'] for e in winner_replays]==[w['commands'],w['prefix_commands'],w['suffix_commands']],'winner replay content')
    operations=result['operations']; math=[e for e in events if e['kind'] in ('jacobian','svd') and e['completed']]
    kinds=['jacobian','svd','svd']+['jacobian','svd']*10
    need(operations==math and [e['kind'] for e in operations]==kinds and [e['index'] for e in operations]==list(range(23)),'all23 math operations')
    for e in operations:
        need(e['q_sha256']==result['q_sha256'],'ownQ operations')
        if e['kind']=='jacobian': need(e['shape']==[192,10] and e['mathematical_vjp_directions']==192,'192 output directions per J')
    return dict(task_id=io.task_hash(t),logits=[w['prefix_params'],w['suffix_params']],commands=w['commands'],prediction=w['context'],model_terminal_cost=w['nonlinear_terminal_cost'])


def ledger_audit(events,arm):
    need(not any('error' in e or 'forbidden' in e['kind'] for e in events),'no failure hidden in completed ledger')
    if arm==ARMS[0]: lane(events,'operation_attempted','operation_completed','event',BUDGETS[arm])
    else:
        lane(events,'native_module_attempt','native_module_completed','entry',dict(predict=66,action_encoder=66))
        math=[dict(kind='math_return' if e['completed'] else 'math_attempt',entry=dict(e,resource=e['kind'])) for e in events if e['kind'] in ('jacobian','svd')]
        lane(math,'math_attempt','math_return','entry',dict(jacobian=11,svd=12))
    api=lane(events,'AD_API_attempt','AD_API_completed','entry',API[arm])
    for e in api:
        need(e['is_grads_batched'] is (arm==ARMS[1] and e['resource']=='grad'),'actual batched AD API distinction')


def audit(report_path,report_sha,prepare_sha):
    report_path=Path(report_path).absolute(); r=read(report_path,report_sha); output=report_path.parent
    need(r['schema']==PLANNER+'.report' and r['prepare_sha256']==prepare_sha,'external report/prepare binding')
    need(r['status']=='model_plans_sealed_not_admitted' and r['errors']==[] and r['first_error'] is None and r['commands_released'] is True,'partial/failure cannot release plans')
    p=read(r['prepared'],prepare_sha); slots=p['slots']; n=len(slots)
    need(p['schema']==PLANNER+'.prepare' and 1<=n<=4,'planner prepare schema')
    source=io.ROOT/'experiments/spectral_reachability'/(PLANNER+'.py')
    need(p['bindings_sha256'].get(str(source))==PLANNER_SHA and io.sha(source)==PLANNER_SHA,'frozen e56bee61 planner')
    need(r['contract']==p['contract'] and p['contract']['per_root_budget']==BUDGETS and
         p['contract']['equal_budget_claim'] is False and p['expected_model_hash']==io.TENSOR_SHA,'original unequal budget contract')
    task_paths=[s['task_export'] for s in slots]; need(len(set(task_paths))==n,'no duplicate tasks')
    for path,h in p['bindings_sha256'].items():
        f=Path(path)
        need(path in task_paths or path in io.MODELS or (f.suffix in ('.py','.md') and 'results' not in f.parts),'never follow privileged sourcepack/NPZ/certificate bindings')
        need(io.sha(f)==h,'source/export/model file hash '+f.name)
    names={'binding.json','events.jsonl.gz'}|{f'root_{i}_{arm}.{suffix}' for i in range(n) for arm in ARMS for suffix in ('result.json.gz','plan.json')}
    need(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'all arms exact inventory')
    for name,h in r['artifact_sha256'].items(): need(io.sha(output/name)==h,'artifact hash '+name)
    need(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'binding file')
    raw=load(output/'events.jsonl.gz'); need(raw.endswith(b'\n'),'complete durable journal line')
    wrapped=[io.parse(line) for line in raw.splitlines()]
    expected=[(i,arm) for i in range(n) for arm in ARMS]
    need([(x['order'],x['arm']) for x in r['rows']]==expected,'complete ordered census')
    groups=[]
    for e in wrapped:
        key=(e['root_order'],e['arm'])
        if not groups or groups[-1][0]!=key: groups.append((key,[]))
        groups[-1][1].append(e)
    need([key for key,_ in groups]==expected,'causal root/arm order, no resumed earlier arm')
    summaries=[]
    for row,((order,arm),group) in zip(r['rows'],groups):
        slot=slots[order]; t=export_task(slot['task_export'],slot['task_sha256']); ident=io.task_hash(t)
        need(slot['order']==order and slot['task_id']==ident and row['task_id']==ident and all(e['task_id']==ident for e in group),'opaque task join')
        events=[e['event'] for e in group]
        need(events[0]==dict(kind='arm_started',task_sha256=slot['task_sha256']) and events[-1]['kind']=='model_selection_sealed','arm lifecycle')
        ledger_audit(events,arm)
        stem=f'root_{order}_{arm}'
        need(row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json' and
             row['resources']==counts(BUDGETS[arm]) and row['AD_API_resources']==counts(API[arm]),'per-arm report receipt')
        result=read(output/row['result_file']); choice=gn_audit(t,result,events) if arm==ARMS[0] else d2p_audit(t,result,events)
        plan=read(output/row['plan_file']); identity=plan.pop('selection_sha256')
        want=dict(schema=PLANNER+'.model_plan',status='selected_pending_global_closure',arm=arm,order=order,
                  task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=t['task_contract_sha256'],
                  with_target=False,physical_execution=False,P1_admission=False,selection_uses_physics=False,**choice)
        need(plan==want and identity==io.task_hash(plan)==row['selection_sha256'] and
             events[-1]==dict(kind='model_selection_sealed',selection_sha256=identity,commands=choice['commands']),'independent complete selection/commands seal')
        summaries.append(dict(order=order,arm=arm,task_id=ident,terminal_cost=choice['model_terminal_cost'],
                              selection_sha256=identity,resources=counts(BUDGETS[arm]),AD_API_resources=counts(API[arm])))
    totals={a:counts({k:n*v for k,v in b.items()}) for a,b in BUDGETS.items()}
    need(r['resources']==totals,'whole-run resource totals')
    need(r['source_verified'] is True and r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==io.TENSOR_SHA and
         r['new_images']==r['new_physics']==0 and r['training'] is False and r['P1_admission'] is False and r['equal_budget_claim'] is False,'execution claims/limits')
    used=sum(f.stat().st_size for f in output.iterdir()); need(used<=8*1024**2,'8MiB actual saved output cap')
    need(io.sha(report_path)==report_sha and io.sha(r['prepared'])==prepare_sha,'final receipt TOCTOU')
    for path,h in p['bindings_sha256'].items(): need(io.sha(path)==h,'final source/export TOCTOU')
    for name,h in r['artifact_sha256'].items(): need(io.sha(output/name)==h,'final artifact TOCTOU')
    return dict(schema=NAME+'.report',status='saved_model_plans_consistent_not_admitted',report_sha256=report_sha,prepare_sha256=prepare_sha,
                planner_sha256=PLANNER_SHA,rows=summaries,resources=totals,event_count=len(wrapped),output_bytes=used,
                mathematical_vjp_per_root={ARMS[0]:768,ARMS[1]:2112},new_neural=0,new_physics=0,equal_budget_claim=False,P1_admission=False,
                limitations=['Arithmetic saved consistency, not independent neural or derivative truth.',
                             'No physical outcomes, sourcepack/certificate or hidden root metadata read.',
                             'Inherited image cost separate; no full pipeline/FLOPs equivalence.',
                             'Partial/error runs rejected for release; original receipts retained, never repaired.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('report','report-sha','prepare-sha','output'): parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    try: result=audit(args.report,args.report_sha,args.prepare_sha); code=0
    except Exception as exc:
        result=dict(schema=NAME+'.report',status='saved_audit_rejected',error=dict(type=type(exc).__name__,message=str(exc)),report_sha256=args.report_sha,P1_admission=False); code=1
    result['auditor_sha256']=io.sha(__file__); raw=io.canonical(result); out=Path(args.output)
    need(len(raw)<128*1024 and shutil.disk_usage(out.parent).free>=200*1024**2+len(raw),'small exclusive output/reserve')
    with out.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    print(raw.decode().strip()); return code


if __name__=='__main__': raise SystemExit(main())
