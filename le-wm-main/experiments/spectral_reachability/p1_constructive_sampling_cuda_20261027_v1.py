"""Fixed blind constructive random/CEM baselines. MAIN-only CUDA, never physics."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as base

NAME='p1_constructive_sampling_cuda_20261027_v1'
ROOT=base.ROOT
gn,enc,core=base.gn,base.encode,base.encode.core
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
ARMS=('random_shooting','CEM')
BASE_SHA='e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27'
GN_SHA='825c148cf515ad42d59b187d0d8c83984826db67f58d66b2e2d4f289dcecf70a'
EXPORTS=[str(ROOT/f'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_{i:02d}.json.gz') for i in range(2)]
EXPORT_SHAS=['714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a',
             '70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2']


def budget():
    return dict(predict=134,action_encoder=134)


def seed(slot,arm):
    require(type(slot) is int and slot in (0,1) and arm in ARMS,'fixed slot/arm')
    return 20261027+slot*101+ARMS.index(arm)


def contract():
    return dict(arms=list(ARMS),roots=2,order='source_slot ascending, random_shooting then CEM',
        seed='20261027 + source_slot*101 + arm_index(random=0,CEM=1)',rng='NumPy PCG64 default_rng',
        random=dict(candidates=64,distribution='iid standard Gaussian logits, packet-ball projection'),
        CEM=dict(iterations=4,samples=16,topk=4,initial_mean=0.,initial_std=1.,
                 update='elite projected logits mean/std ddof0, no smoothing, std clip[.05,1]',
                 incumbent_injection=False),
        logits_shape=[2,10],raw_commands_shape=[10,2],packet_logit_cap=float(.75*np.sqrt(5)),
        commands='float32 NumPy tanh then reshape(10,2); no dispatch clip',
        selection='stable global argmin(cost,index) over ALL64, then independent nominal/whole/segmented H2 replay',
        per_arm_root_budget=budget(),total_budget={k:4*v for k,v in budget().items()},
        candidate_predict=128,winner_predict=6,AD_API=0,mathematical_VJP=0,jacobian=0,svd=0,solve=0,
        image_encoder=0,image_projector=0,physical_steps=0,inherited_encoding='separate historical cost, not free',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),training=False,new_Q=False,
        equal_D2P_compute_claim=False,P1_admission=False,new_independent_confirmation=False,
        inputs='ONLY exact two approved blind task exports; no sourcepack/witness/physical outcomes',
        output_cap_bytes=base.CAP,reserve_bytes=base.FLOOR)


def build():
    require(sha(base.__file__)==BASE_SHA and sha(gn.__file__)==GN_SHA,'sealed blind/GN helpers')
    p=base.build(EXPORTS,EXPORT_SHAS)
    slots=[]
    for i,s in enumerate(p['slots']):
        e=read(s['task_export'])
        require(e['source_slot']==i and e['blind_source_sha256']==e['blind_sha256']==e['task_contract']['blind_source_sha256']==list(enc.APPROVED_INPUTS.values())[i],'exact blind source binding')
        slots.append(dict(s,source_slot=i,blind_sha256=e['blind_sha256']))
    bindings=deepcopy(p['bindings_sha256'])
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f.resolve())]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'prepare TOCTOU')
    return dict(schema=NAME+'.prepare',slots=slots,policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        contract=contract(),bindings_sha256=bindings,global_freshness='existing_two_calibration_tasks_only')


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=base.FLOOR+len(raw),'NEW small prepare +200MiB floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path)
    require(p==build(),'entire source/export/model/protocol reconstruction');return p


def proposals(rng,mean,std,n,cap):
    return [gn.screen.projected_logits(x,cap) for x in rng.normal(mean,std,size=(n,2,10))]


def elite_update(bank):
    require(len(bank)==16 and all(np.isfinite(c['cost']) and c['cost']>=0 for c in bank),'all16 finite CEM costs')
    elite=sorted(range(16),key=lambda k:(bank[k]['cost'],bank[k]['index']))[:4]
    x=np.asarray([bank[k]['logits'] for k in elite])
    return x.mean(axis=0),np.clip(x.std(axis=0,ddof=0),.05,1.),elite


def compact(p):
    return dict(logits=p['logits'],commands=p['commands'],cost=p['cost'],terminal=p['contexts'][-1]['latent_history'][-1])


class Meter(gn.Meter):
    def __init__(self,sink,sync=lambda:None):
        gn.screen.Meter.__init__(self,budget(),sink);self.sync=sync

    def emit(self,kind,**values):
        # Full per-packet contexts already durable in native_packet. Avoid
        # duplicating complete rolling histories in evaluation_complete.
        if kind=='evaluation_complete':values=dict(values,prediction=compact(values['prediction']))
        super().emit(kind,**values)


def optimize(task,slot,arm,backend,meter):
    core.validate_task(task);rng=np.random.default_rng(seed(slot,arm));bank=[];updates=[]
    mean=np.zeros((2,10));std=np.ones((2,10));cap=task['packet_logit_cap']
    for iteration in range(1 if arm==ARMS[0] else 4):
        xs=proposals(rng,mean,std,64 if arm==ARMS[0] else 16,cap)
        meter.emit('bank_frozen',iteration=iteration,mean=mean.tolist(),std=std.tolist(),logits=[x.tolist() for x in xs])
        chunk=[]
        for x in xs:
            index=len(bank);meter.group=f'candidate_{index}'
            p=gn.single(task,x,backend,meter,'whole_replay')
            c=dict(index=index,iteration=iteration,**compact(p));bank.append(c);chunk.append(c)
            meter.emit('candidate_saved',candidate=c)
        if arm==ARMS[1]:
            mean,std,elite=elite_update(chunk)
            u=dict(iteration=iteration,elite_indices=[chunk[k]['index'] for k in elite],mean=mean.tolist(),std=std.tolist())
            updates.append(u);meter.emit('CEM_update',**u)
    require(len(bank)==64,'complete fullH2 denominator')
    index=min(range(64),key=lambda k:(bank[k]['cost'],k));chosen=bank[index]
    meter.emit('winner_frozen_before_replay',index=index,cost=chosen['cost'],commands=chosen['commands'])
    meter.group='winner_triple_replay';fresh=gn.verified(task,chosen['logits'],backend,meter)
    require(compact(fresh)=={k:chosen[k] for k in compact(fresh)},'MATH search/replay exact commands/terminal/cost')
    base.exact_counts(meter.counts(),budget())
    return dict(task_id=core.task_id(task),source_slot=slot,arm=arm,seed=seed(slot,arm),candidates=bank,
                updates=updates,winner_index=index,final_prediction=fresh,resources=meter.counts())


def choice(task,result,slot,arm):
    require(result['task_id']==core.task_id(task) and result['source_slot']==slot and result['arm']==arm and result['seed']==seed(slot,arm),'task/arm/seed identity')
    base.exact_counts(result['resources'],budget());bank=result['candidates'];require(len(bank)==64,'ALL64')
    rng=np.random.default_rng(seed(slot,arm));mean=np.zeros((2,10));std=np.ones((2,10));updates=[]
    for it in range(1 if arm==ARMS[0] else 4):
        n=64 if arm==ARMS[0] else 16;chunk=bank[it*n:(it+1)*n]
        xs=proposals(rng,mean,std,n,task['packet_logit_cap'])
        for j,(c,x) in enumerate(zip(chunk,xs,strict=True)):
            terminal=core.array(c['terminal'],(192,))
            require(c['index']==it*n+j and c['iteration']==it and c['logits']==x.tolist() and
                c['commands']==np.tanh(x.astype(np.float32)).reshape(10,2).astype(float).tolist() and
                c['cost']==core.model_cost(task,terminal),'exact deterministic candidate/rawcommands/own-Q cost')
        if arm==ARMS[1]:
            mean,std,elite=elite_update(chunk)
            updates.append(dict(iteration=it,elite_indices=[chunk[k]['index'] for k in elite],mean=mean.tolist(),std=std.tolist()))
    require(result['updates']==updates,'own CEM causal update chain')
    index=min(range(64),key=lambda k:(bank[k]['cost'],k));p=result['final_prediction'];core.check_prediction(task,p)
    require(result['winner_index']==index and compact(p)=={k:bank[index][k] for k in compact(p)},'stable global winner exact triple replay')
    return dict(task_id=core.task_id(task),logits=p['logits'],commands=p['commands'],prediction=p['contexts'][-1],model_terminal_cost=p['cost'])


def execute(task,slot,arm,model,device,torch,sink,row):
    meter=Meter(sink,lambda:torch.cuda.synchronize(device))
    try:
        backend=gn.screen.NativeCuda(model,device,task)
        with torch.inference_mode(),backend.guard(meter):return optimize(task,slot,arm,backend,meter)
    finally:row['resources']=meter.counts()


def journal_check(task,slot,arm,result,group):
    """Pure array replay of every recorded H2 path, not repeated receipt IDs."""
    requests=[];current=None;packets=[];completed=[];attempts=[];returns=[];pending=None;frozen=0;winner_sealed=False
    for e in group:
        require('forbidden' not in e['kind'] and 'error' not in e,'no forbidden/error event')
        if e['kind']=='bank_frozen':
            require(current is None and len(completed)==(0 if arm==ARMS[0] else frozen*16) and
                    frozen<(1 if arm==ARMS[0] else 4),'bank seal precedes its first evaluation')
            frozen+=1
        elif e['kind']=='winner_frozen_before_replay':
            require(not winner_sealed and current is None and len(completed)==64,'selection after ALL64 and before fresh replay')
            winner_sealed=True
        elif e['kind']=='operation_attempted':
            v=e['event'];require(pending is None and v['index']==len(attempts) and not v['completed'],'sequential native attempt')
            attempts.append(v);pending=v
        elif e['kind']=='operation_completed':
            v=e['event'];require(pending is not None and v==dict(pending,completed=True),'native attempt/return identity')
            returns.append(v);pending=None
        elif e['kind']=='evaluation_request':
            require(current is None and frozen>=1 and (len(requests)<(64 if arm==ARMS[0] else 16*frozen) if len(requests)<64 else winner_sealed),'bank/winner seal before dispatch, no overlapping H2')
            current=e;packets=[];requests.append(e)
        elif e['kind']=='native_packet':
            require(current is not None and e['evaluation_id']==current['evaluation_id'] and e['mode']==current['mode'] and e['packet']==len(packets),'exact H2 packet scope/order')
            packets.append(e['context'])
        elif e['kind']=='evaluation_complete':
            require(current is not None and len(packets)==2 and e['evaluation_id']==current['evaluation_id'] and e['mode']==current['mode'],'both real H2 packets before cost')
            gn.saved.check_path(task,np.asarray(current['commands']),packets)
            p=dict(logits=current['logits'],commands=current['commands'],contexts=packets,
                   cost=core.model_cost(task,packets[-1]['latent_history'][-1]))
            require(e['prediction']==compact(p),'actual saved packets recompute terminal/own-Q cost')
            completed.append(dict(request=current,prediction=compact(p),contexts=deepcopy(packets)));current=None
    require(current is None and pending is None and len(completed)==67 and len(attempts)==len(returns)==268,'complete actual H2 and native counts')
    base.exact_counts({k:dict(attempted=sum(e['resource']==k for e in attempts),completed=sum(e['resource']==k for e in returns)) for k in budget()},budget())
    for i,(c,e) in enumerate(zip(result['candidates'],completed[:64],strict=True)):
        require(e['request']['evaluation_id']==f'candidate_{i}' and e['request']['mode']=='whole_replay' and
                e['prediction']=={k:c[k] for k in e['prediction']},'actual cost and commands for every bank slot')
    p=result['final_prediction']
    for e,mode in zip(completed[-3:],('nominal','whole_replay','segmented_replay'),strict=True):
        require(e['request']['evaluation_id']=='winner_triple_replay' and e['request']['mode']==mode and
                e['prediction']==compact(p) and e['contexts']==p['contexts'],'three genuinely recomputed selected trajectories')
    require([e['candidate'] for e in group if e['kind']=='candidate_saved']==result['candidates'],'all candidates durable')
    require([{k:e[k] for k in ('iteration','elite_indices','mean','std')} for e in group if e['kind']=='CEM_update']==result['updates'],'causal CEM updates durable')
    rng=np.random.default_rng(seed(slot,arm));mean=np.zeros((2,10));std=np.ones((2,10));want=[]
    for it in range(1 if arm==ARMS[0] else 4):
        n=64 if arm==ARMS[0] else 16
        xs=proposals(rng,mean,std,n,task['packet_logit_cap'])
        want.append(dict(kind='bank_frozen',evaluation_id=None if it==0 else f'candidate_{it*16-1}',iteration=it,mean=mean.tolist(),std=std.tolist(),logits=[x.tolist() for x in xs]))
        if arm==ARMS[1]:mean,std,_=elite_update(result['candidates'][it*16:it*16+16])
    require([e for e in group if e['kind']=='bank_frozen']==want,'all frozen pre-evaluation proposal arrays')
    w=result['candidates'][result['winner_index']]
    require([e for e in group if e['kind']=='winner_frozen_before_replay']==[dict(kind='winner_frozen_before_replay',evaluation_id='candidate_63',index=w['index'],cost=w['cost'],commands=w['commands'])],'single pre-replay selection seal')


def plan_all(p,store,runner,rows):
    for row in rows:
        slot=p['slots'][row['source_slot']];arm=row['arm'];task=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        row['status']='attempted'
        def sink(e):store.event(dict(source_slot=slot['source_slot'],arm=arm,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',seed=seed(slot['source_slot'],arm),task_sha256=slot['task_sha256']))
        result=runner(task,slot['source_slot'],arm,sink,row);selected=choice(task,result,slot['source_slot'],arm)
        base.exact_counts(row['resources'],budget());stem=f'root_{slot["source_slot"]}_{arm}'
        store.put(stem+'.result.json.gz',result)
        plan=dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=arm,blind_sha256=slot['blind_sha256'],
            task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],
            with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**selected)
        plan['selection_sha256']=core.digest(plan);store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    p=verify(prepared,expected);store=base.Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=a,blind_sha256=s['blind_sha256'],task_id=s['task_id'],status='not_started',commands=None,resources={}) for s in p['slots'] for a in ARMS]
    try:
        print('Fixed blind same-task random64/CEM4x16, model-only selection; not D2P compute equivalence or P1 admission.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'frozen original model')
        with sdpa_kernel(SDPBackend.MATH):plan_all(p,store,lambda t,s,a,k,r:execute(t,s,a,model,device,torch,k,r),rows)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(enc.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:errors.append(enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(enc.error('final_source',e))
        try:store.close()
        except BaseException as e:errors.append(enc.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:errors.append(enc.error('artifact_listing',e));complete=False;files=[]
        for f in files:
            try:
                if f.is_file():hashes[f.name]=sha(f)
            except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in budget()}
        report=dict(schema=NAME+'.report',status='sampling_model_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,resources=resources,errors=errors,first_error=errors[0] if errors else None,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,contract=p['contract'],commands_released=complete,new_images=0,new_physics=0,training=False,P1_admission=False,equal_D2P_compute_claim=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:complete=False;errors.append(enc.error('final_report',e));print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial preserved, no retry: '+str(errors));return report


def consume(output,expected):
    output=Path(output).absolute();require(sha(output/'report.json')==expected,'external actual report SHA')
    r=read(output/'report.json');require(r['schema']==NAME+'.report' and r['status']=='sampling_model_plans_sealed_not_admitted' and
        r['commands_released'] is True and r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and
        r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==enc.MODEL_TENSOR_SHA,'whole CUDA closure')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and len(r['rows'])==4,'own complete four branches')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{a}.{suffix}' for s in (0,1) for a in ARMS for suffix in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'complete exact artifact set')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'saved artifact hash')
    plain,eof=base.inflate((output/'events.jsonl.gz').read_bytes());require(eof and plain.endswith(b'\n'),'complete gzip journal')
    events=[base.decode(x) for x in plain.splitlines()];plans=[]
    require(all(set(e)=={'source_slot','arm','task_id','event'} for e in events),'no global error journal')
    groups=[]
    for e in events:
        key=(e['source_slot'],e['arm'])
        if not groups or groups[-1]!=key:groups.append(key)
    require(groups==[(s,a) for s in (0,1) for a in ARMS],'fixed causal scope order')
    for row,(s,a) in zip(r['rows'],groups,strict=True):
        slot=p['slots'][s];stem=f'root_{s}_{a}'
        require(row['source_slot']==s and row['arm']==a and row['blind_sha256']==slot['blind_sha256'] and row['task_id']==slot['task_id'] and
            row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','direct stable branch identity')
        task=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];result=read(output/row['result_file']);selected=choice(task,result,s,a)
        plan=read(output/row['plan_file']);ident=plan.pop('selection_sha256')
        want=dict(schema=NAME+'.model_plan',source_slot=s,arm=a,blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
            task_contract_sha256=task['task_contract_sha256'],with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**selected)
        require(plan==want and core.digest(plan)==ident==row['selection_sha256'] and row['commands']==selected['commands'],'selection frozen before physics')
        base.exact_counts(row['resources'],budget())
        group=[e['event'] for e in events if e['source_slot']==s and e['arm']==a]
        require(all(e['task_id']==slot['task_id'] for e in events if e['source_slot']==s and e['arm']==a),'journal task binding')
        require(group[0]==dict(kind='arm_started',seed=seed(s,a),task_sha256=slot['task_sha256']) and
            group[-1]==dict(kind='model_selection_sealed',selection_sha256=ident,commands=selected['commands']),'durable lifecycle')
        journal_check(task,s,a,result,group)
        plans.append(dict(plan,selection_sha256=ident,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],p['contract']['total_budget'])
    require(sha(output/'report.json')==expected,'report TOCTOU')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final source/input binding')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'final artifact TOCTOU')
    return dict(schema=NAME+'.consume',rows=plans,P1_admission=False,new_physics=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='consume':result=consume(a.output,a.report_sha)
    else:result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode().strip())


if __name__=='__main__':main()
