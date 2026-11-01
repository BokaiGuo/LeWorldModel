"""Independent saved sampling audit: NumPy/JSON only; no producer imports."""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_saved_audit_20261027_v1 as pure

io=pure.io
need=io.need
NAME='p1_constructive_sampling_saved_audit_20261027_v1'
PRODUCER='p1_constructive_sampling_cuda_20261027_v1'
SOURCE_SHA='5fe2af6256071d4a056a9f40908ee616c284e7b72e61a5c1fc5a79e359b86862'
REPORT_SHA='0c2db96dc506a7a2d71e4dcd43f5990fe0cc8ad0ad27f8483388cecdf47b9698'
PREPARE_SHA='4c209a1a2fb1b6c9a9e3067b21b5bc3dd6e3364234cada2278b6071d6668b4ce'
REPORT=io.ROOT/f'results/{PRODUCER}_cuda_main01/report.json'
EXPORT_SHAS=('714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a',
             '70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2')
ARMS=('random_shooting','CEM')
BUDGET=dict(predict=134,action_encoder=134)
read=pure.read


def protocol(c):
    """Independent fixed parameter assertions, not producer contract()."""
    need(c['arms']==list(ARMS) and c['roots']==2 and c['packet_logit_cap']==pure.CAP,'same fixed task/arm domain')
    need(c['random']==dict(candidates=64,distribution='iid standard Gaussian logits, packet-ball projection'),'random64')
    need(c['CEM']==dict(iterations=4,samples=16,topk=4,initial_mean=0.,initial_std=1.,
        update='elite projected logits mean/std ddof0, no smoothing, std clip[.05,1]',incumbent_injection=False),'frozen CEM algorithm')
    need(c['per_arm_root_budget']==BUDGET and c['total_budget']==dict(predict=536,action_encoder=536) and
         c['candidate_predict']==128 and c['winner_predict']==6,'128+6 not128 alone')
    need(all(c[k]==0 for k in ('AD_API','mathematical_VJP','jacobian','svd','solve','image_encoder','image_projector','physical_steps')),'declared resource vector')
    need(c['numerical']==dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False) and
         all(c[k] is False for k in ('P1_admission','training','new_Q','new_independent_confirmation','equal_D2P_compute_claim')),'no scientific/compute admission')
    need(c['logits_shape']==[2,10] and c['raw_commands_shape']==[10,2] and c['output_cap_bytes']==8*1024**2 and c['reserve_bytes']==200*1024**2,'action/disk contract')


def stream(group):
    pos=0
    def take(kind):
        nonlocal pos
        need(pos<len(group) and group[pos]['kind']==kind,f'causal event {pos}: expected {kind}')
        value=group[pos];pos+=1;return value
    def finish():need(pos==len(group),'no unaccounted events or hidden extra work')
    return take,finish


def audit_branch(task,slot,arm,result,row,plan,events):
    s=slot['source_slot'];ident=io.task_hash(task);seed=20261027+101*s+(arm=='CEM')
    need(s in (0,1) and arm in ARMS and result['source_slot']==s and result['arm']==arm and result['seed']==seed and
         result['task_id']==ident==slot['task_id'],'branch seed and stable task identity')
    need(result['resources']==row['resources']==pure.counts(BUDGET),'per-branch complete actual resources')
    bank=result['candidates'];need(len(bank)==64,'full64 denominator')
    need(row['source_slot']==s and row['arm']==arm and row['blind_sha256']==slot['blind_sha256'] and row['task_id']==ident and row['status']=='model_plan_sealed','row identity')
    take,finish=stream(events)
    need(take('arm_started')==dict(kind='arm_started',seed=seed,task_sha256=slot['task_sha256']),'initial lifecycle input SHA')
    op_index=0;eval_count=0;packet_count=0
    def evaluation(logits,commands,gid,mode):
        nonlocal op_index,eval_count,packet_count
        want=dict(evaluation_id=gid,mode=mode,logits=logits,commands=commands)
        need(take('evaluation_request')==dict(kind='evaluation_request',**want),'request exact action binding')
        path=[]
        for k in (0,1):
            for resource in ('action_encoder','predict'):
                entry=dict(index=op_index,phase='direct',evaluation_id=gid,resource=resource,samples=1,completed=False)
                need(take('operation_attempted')==dict(kind='operation_attempted',evaluation_id=gid,event=entry),'predispatch exact actual entry')
                need(take('operation_completed')==dict(kind='operation_completed',evaluation_id=gid,event=dict(entry,completed=True)),'same actual return, no retries')
                op_index+=1
            e=take('native_packet')
            need(set(e)=={'kind','evaluation_id','mode','packet','context'} and e['evaluation_id']==gid and e['mode']==mode and e['packet']==k,'packet index/mode scope')
            path.append(e['context']);packet_count+=1
        pure.path_check(task,commands,path)
        pred=dict(logits=logits,commands=commands,cost=pure.cost(task,path[-1]['latent_history'][-1]),terminal=path[-1]['latent_history'][-1])
        need(take('evaluation_complete')==dict(kind='evaluation_complete',evaluation_id=gid,mode=mode,prediction=pred),'own-Q and terminal recomputed from actual packet arrays')
        eval_count+=1;return pred,path

    # Explicit PCG64 stream; generate standard variates then affine-transform,
    # independently of producer proposals/rng.normal/elite_update.
    rng=np.random.Generator(np.random.PCG64(seed));mu=np.zeros((2,10));sigma=np.ones((2,10));updates=[]
    for it in range(4 if arm=='CEM' else 1):
        n=16 if arm=='CEM' else 64
        normals=rng.standard_normal((n,20)).reshape(n,2,10)
        xs=[pure.project(x) for x in normals*sigma+mu]
        previous=None if it==0 else f'candidate_{it*16-1}'
        need(take('bank_frozen')==dict(kind='bank_frozen',evaluation_id=previous,iteration=it,mean=mu.tolist(),std=sigma.tolist(),logits=[x.tolist() for x in xs]),'RNG bank frozen before any candidate cost')
        for j,x in enumerate(xs):
            i=it*n+j;c=bank[i];gid=f'candidate_{i}'
            raw=np.tanh(x.astype(np.float32)).reshape(10,2).astype(float).tolist()
            pred,path=evaluation(x.tolist(),raw,gid,'whole_replay')
            expected=dict(index=i,iteration=it,**pred)
            need(c==expected,'ALL64 independent proposal/commands/cost/terminal rows')
            need(take('candidate_saved')==dict(kind='candidate_saved',evaluation_id=gid,candidate=expected),'candidate immediately durable before next dispatch')
        if arm=='CEM':
            ids=sorted(range(it*16,(it+1)*16),key=lambda k:(bank[k]['cost'],k))[:4]
            elite=np.stack([bank[k]['logits'] for k in ids])
            mu=np.mean(elite,axis=0);sigma=np.maximum(.05,np.minimum(1.,np.std(elite,axis=0,ddof=0)))
            update=dict(iteration=it,elite_indices=ids,mean=mu.tolist(),std=sigma.tolist());updates.append(update)
            need(take('CEM_update')==dict(kind='CEM_update',evaluation_id=f'candidate_{(it+1)*16-1}',**update),'top4 causal update, no lookahead or smoothing')
    need(result['updates']==updates,'result full CEM update chain')
    w=min(range(64),key=lambda i:(bank[i]['cost'],i));winner=bank[w]
    need(result['winner_index']==w,'global64 stable argmin, not final CEM iteration only')
    need(take('winner_frozen_before_replay')==dict(kind='winner_frozen_before_replay',evaluation_id='candidate_63',index=w,cost=winner['cost'],commands=winner['commands']),'selection after64 before replay')
    paths=[]
    for mode in ('nominal','whole_replay','segmented_replay'):
        pred,path=evaluation(winner['logits'],winner['commands'],'winner_triple_replay',mode)
        need(pred=={k:winner[k] for k in pred},'search/replay exact MATH terminal/action/cost')
        paths.append(path)
    need(paths[0]==paths[1]==paths[2],'all three actual replay contexts equal')
    final=result['final_prediction'];pure.prediction(task,final,replays=True)
    need(final==dict(task_id=ident,logits=winner['logits'],commands=winner['commands'],contexts=paths[0],cost=winner['cost'],whole_replay=paths[1][-1],segmented_replay=paths[2]),'complete final independent replay content')
    need(op_index==268 and eval_count==67 and packet_count==134,'exact actual denominator including triple replay')
    pure.lane(events,'operation_attempted','operation_completed','event',BUDGET)
    selected=dict(task_id=ident,logits=winner['logits'],commands=winner['commands'],prediction=paths[0][-1],model_terminal_cost=winner['cost'])
    want=dict(schema=PRODUCER+'.model_plan',source_slot=s,arm=arm,blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
        task_contract_sha256=task['task_contract_sha256'],with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**selected)
    h=io.task_hash(want)
    need(plan==dict(want,selection_sha256=h) and row['selection_sha256']==h and row['commands']==winner['commands'],'plan/row selection seal and commands')
    need(take('model_selection_sealed')==dict(kind='model_selection_sealed',selection_sha256=h,commands=winner['commands']),'durable final selection')
    finish()
    return dict(source_slot=s,arm=arm,blind_sha256=slot['blind_sha256'],task_id=ident,seed=int(seed),
        candidates=64,evaluations=67,packets=134,predict=134,action_encoder=134,winner_index=w,
        model_cost=winner['cost'],selection_sha256=h,CEM_elite_indices=[u['elite_indices'] for u in updates],
        max_candidate_packet_logit_norm=float(np.linalg.norm(np.asarray([c['logits'] for c in bank]),axis=2).max()))


def audit(report_path=REPORT,report_sha=REPORT_SHA,prepare_sha=PREPARE_SHA):
    audit_sources={str(Path(mod.__file__).resolve()):io.sha(mod.__file__) for mod in (pure,io)}
    audit_sources[str(Path(__file__).resolve())]=io.sha(__file__)
    path=Path(report_path).absolute();r=read(path,report_sha);directory=path.parent
    need(r['schema']==PRODUCER+'.report' and r['prepare_sha256']==prepare_sha and r['status']=='sampling_model_plans_sealed_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['commands_released'] is True and r['source_verified'] is True,'only complete actual report')
    need(r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==io.TENSOR_SHA and
        r['P1_admission'] is False and r['equal_D2P_compute_claim'] is False and r['new_physics']==r['new_images']==0 and r['training'] is False,'frozen model and scope')
    p=read(r['prepared'],prepare_sha);need(p['schema']==PRODUCER+'.prepare' and p['contract']==r['contract'],'prepare/report protocol equality');protocol(p['contract'])
    need(p['expected_model_hash']==io.TENSOR_SHA and len(p['slots'])==2,'original checkpoint/two tasks')
    source=io.ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    need(p['bindings_sha256'].get(str(source))==SOURCE_SHA and io.sha(source)==SOURCE_SHA,'exact frozen source, hash only never imported')
    slots=p['slots'];allowed_exports={s['task_export'] for s in slots}
    for filename,h in p['bindings_sha256'].items():
        f=Path(filename)
        need(filename in allowed_exports or filename in io.MODELS or (f.suffix in ('.py','.md') and 'results' not in f.parts),'no privileged input bindings followed')
        need(io.sha(f)==h,'source/input binding '+f.name)
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{a}.{suffix}' for s in (0,1) for a in ARMS for suffix in ('result.json.gz','plan.json')}
    need(set(r['artifact_sha256'])==names and {f.name for f in directory.iterdir()}==names|{'report.json'},'complete exact saved inventory')
    need(sum(f.stat().st_size for f in directory.iterdir())<=8*1024**2,'actual producer8MiB cap')
    for n,h in r['artifact_sha256'].items():need(io.sha(directory/n)==h,'artifact SHA '+n)
    need(read(directory/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'binding file')
    plain=pure.load(directory/'events.jsonl.gz');need(plain.endswith(b'\n'),'complete journal final line')
    wrapped=[io.parse(x) for x in plain.splitlines()];expected=[(s,a) for s in (0,1) for a in ARMS];groups=[]
    for e in wrapped:
        need(set(e)=={'source_slot','arm','task_id','event'},'only scoped actual events, no global errors')
        key=(e['source_slot'],e['arm'])
        if not groups or groups[-1][0]!=key:groups.append((key,[]))
        groups[-1][1].append(e)
    need([key for key,_ in groups]==expected and [(r['source_slot'],r['arm']) for r in r['rows']]==expected,'fixed full four-branch census')
    rows=[]
    for row,((s,a),group) in zip(r['rows'],groups,strict=True):
        slot=slots[s];need(slot['source_slot']==slot['order']==s and slot['task_sha256']==EXPORT_SHAS[s] and slot['blind_sha256']==io.BLIND_SHA[s],'approved stable slot export/blind identity')
        task=pure.export_task(slot['task_export'],slot['task_sha256']);e=read(slot['task_export'])
        need(e['source_slot']==s and e['blind_sha256']==slot['blind_sha256'] and slot['task_id']==io.task_hash(task) and all(x['task_id']==slot['task_id'] for x in group),'task ID joins not arbitrary array order')
        stem=f'root_{s}_{a}';need(row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','flat fixed filenames')
        rows.append(audit_branch(task,slot,a,read(directory/row['result_file']),row,read(directory/row['plan_file']),[x['event'] for x in group]))
    need(r['resources']==pure.counts(dict(predict=536,action_encoder=536)),'actual total536 each')
    # Recheck all external bytes after arithmetic, without following new inputs.
    need(io.sha(path)==report_sha and io.sha(r['prepared'])==prepare_sha,'report/prepare final TOCTOU')
    for n,h in r['artifact_sha256'].items():need(io.sha(directory/n)==h,'final artifact TOCTOU')
    for filename,h in p['bindings_sha256'].items():need(io.sha(filename)==h,'final source/input TOCTOU')
    for filename,h in audit_sources.items():need(io.sha(filename)==h,'independent auditor source unchanged')
    return dict(schema=NAME+'.report',status='saved_arithmetic_verified_not_scientific_admission',
        actual_report_sha256=report_sha,prepare_sha256=prepare_sha,producer_source_sha256=SOURCE_SHA,branches=rows,
        auditor_sources_sha256=audit_sources,
        totals=dict(branches=4,candidates=256,fullH2_evaluations=268,predict=536,action_encoder=536),
        checks=dict(external_bindings=True,rng_banks=True,CEM_updates=True,all_candidate_costs=True,
        float32_commands_and_history=True,actual_module_ledger=True,pre_evaluation_bank_freeze=True,
        global_stable_selection=True,independent_triple_saved_paths=True),
        limits='Saved CUDA receipts/arrays only; no independent model forward, no physical/goal-achievement proof; zero AD is declared scope plus no AD ledger events, not profiler proof.',
        neural_calls=0,physics_calls=0,producer_optimize_or_consume_imported=False,
        equal_D2P_compute_claim=False,P1_admission=False)


def write_new(path,value):
    raw=io.canonical(value);p=Path(path).absolute()
    need(len(raw)<=150*1024 and not p.exists() and shutil.disk_usage(p.parent).free>=200*1024**2+len(raw),'NEW <=150KiB output +200MiB reserve')
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(output=str(p),sha256=io.sha(p),bytes=len(raw))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,default=REPORT)
    p.add_argument('--report-sha',default=REPORT_SHA);p.add_argument('--prepare-sha',default=PREPARE_SHA);p.add_argument('--output',type=Path)
    a=p.parse_args();value=audit(a.report,a.report_sha,a.prepare_sha)
    print(io.canonical(write_new(a.output,value) if a.output else value).decode())


if __name__=='__main__':main()
