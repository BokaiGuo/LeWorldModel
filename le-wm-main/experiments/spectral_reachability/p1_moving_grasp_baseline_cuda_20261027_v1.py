"""Fixed moving2 adapter for frozen GRASP_H3_bounded. MAIN-only HOST CUDA.

H3 is history length, NOT a future15-step optimizer truncated to10 steps.
Original future H2=10,750 joint+150 sync updates; no algorithm/outcome tuning.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import sysconfig
import shutil
from experiments.spectral_reachability import p1_constructive_grasp_baseline_20261027_v1 as grasp
from experiments.spectral_reachability import p1_moving_gradient_baseline_cuda_20261027_v1 as binding

NAME='p1_moving_grasp_baseline_cuda_20261027_v1'
ROOT,ARM,core,base=grasp.ROOT,grasp.ARM,grasp.core,grasp.base
require,sha,read,canonical=grasp.require,grasp.sha,grasp.read,grasp.canonical
Config,budget,schedule,seed=grasp.Config,grasp.budget,grasp.schedule,grasp.seed
optimize,choice,journal_check=grasp.optimize,grasp.choice,grasp.journal_check
Dynamics,Meter,Store=grasp.Dynamics,grasp.Meter,grasp.Store
Limits= binding.Limits
CAP,FLOOR,FINAL=grasp.CAP,grasp.FLOOR,grasp.FINAL
GRASP_SHA='ae9d2dd59bf613bc10c81858cb767dad1d92dcf9b7b7d103bedb018e0bd1fa28'
BINDING_SHA='3224d6a094e74b3845c2c695c6c85fe29c6522cd4a88c91c57f2cc03d127a70e'
load_task=binding.load_task


def contract():
    c=deepcopy(grasp.contract())
    c.update(interface_source=binding.moving.NAME,history_source=binding.moving.contract()['history'],
        horizon_semantics=dict(observed_history_frames=3,imagined_future_packets=2,imagined_future_public_steps=10,
            released_commands=10,physical_steps_this_producer=0,third_future_packet=False),
        global_freshness='fixed existing moving2 calibration tasks, not independent confirmation',
        sourcepack_access='SHA only; no witness/native state/outcome parse',
        input_evidence='each1802 actual predictor input latents and raw-normalized packet tokens/root',
        backward_accounting='900 scalar loss.backward/root, not192-dimensional VJP; no separate extra backward charge',
        wall_seconds=600,allocated_bytes=3221225472,reserved_bytes=4294967296,final_reserve_bytes=FINAL,
        feedback_MPC=False,cross_contract_reported_comparison=False)
    return c


def build(**arguments):
    require(sha(grasp.__file__)==GRASP_SHA and sha(binding.__file__)==BINDING_SHA,'frozen GRASP core and moving binding helper')
    p=binding.build(**arguments)  # SHA/identity only; NEVER zero-only grasp.build.
    site=Path(sysconfig.get_paths()['purelib'])
    files=[Path(grasp.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        ROOT/'tests'/('test_'+grasp.NAME+'.py'),ROOT/'innerdocs'/(grasp.NAME+'.md')]
    files += [site/x for x in ('torch/optim/adam.py','torch/optim/optimizer.py','torch/optim/_functional.py','torch/random.py')]
    bindings={**p['bindings_sha256'],**grasp.official_bindings(),**{str(f.resolve()):sha(f) for f in files}}
    for path,h in bindings.items():require(sha(path)==h,'final own source/export/model/official/optimizer closure')
    return dict(schema=NAME+'.prepare',arguments=p['arguments'],slots=p['slots'],policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        source_host=p['source_host'],contract=contract(),bindings_sha256=bindings)


def prepare(output,**arguments):
    p=build(**arguments);raw=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(raw)<=256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive <=256KiB prepare +200MiB')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    binding.gradient.io.fsync_directory(path.parent)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external own prepare SHA');p=read(path)
    require(p==build(**p['arguments']),'complete own moving GRASP closure');return p


@contextmanager
def api_trace(model,sink):
    original=model.predict;entries=[];primary=None
    def call(*args,**kwargs):
        require(len(entries)<budget()['predict'],'1802 model.predict API/root cap')
        e=dict(index=len(entries),completed=False);entries.append(e);sink(dict(kind='predict_API_attempt',index=e['index']))
        result=original(*args,**kwargs);e['completed']=True;sink(dict(kind='predict_API_completed',index=e['index']));return result
    model.predict=call
    try:yield entries
    except BaseException as exc:primary=exc;raise
    finally:
        try:model.predict=original
        except BaseException as exc:
            if primary is not None:primary.add_note('predict API restore '+str(exc))
            else:raise


def history_check(task,events,cfg=Config()):
    # The frozen GRASP journal_check separately verifies full joint/lifted/sync
    # chronology. This joins actual native input tensors to each requested edge.
    expected=[];current=None
    for e in events:
        if e['kind']=='transition_attempt':
            require(current is None,'single active native transition');current=e
            before=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                latent_history=e['latent_history'][0],action_history=e['action_history'][0])
            expected.append(binding.moving.incoming(task,before,e['commands']))
        elif e['kind']=='transition_completed':
            require(current is not None,'actual requested transition');current=None
    n=budget(cfg)['predict'];traces=[e for e in events if e['kind']=='native_context_attempt']
    attempts=[e['index'] for e in events if e['kind']=='predict_API_attempt'];done=[e['index'] for e in events if e['kind']=='predict_API_completed']
    require(current is None and len(expected)==n and [e['index'] for e in traces]==attempts==done==list(range(n)) and
        [e['context'] for e in traces]==expected,'ALL actual lifted/serial native inputs match original moving history')
    sequence=[(e['kind'],e.get('index')) for e in events if e['kind'] in ('predict_API_attempt','native_context_attempt','predict_API_completed')]
    require(sequence==[(k,i) for i in range(n) for k in ('predict_API_attempt','native_context_attempt','predict_API_completed')],'actual API/context chronology')
    return dict(native_contexts=n,predict_API=n,all_nonzero_history_paths_verified=True)


def execute(task,slot,model,device,torch,sink,row,limits):
    def checked(e):limits.check();sink(e)
    meter=Meter(checked,lambda:torch.cuda.synchronize(device));api=[]
    try:
        backend=grasp.gn.screen.NativeCuda(model,device,task)
        dynamics=Dynamics(task,device,lambda z,h,a,m:backend._step(task,z,h,a))
        with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False),backend.guard(meter),api_trace(model,checked) as api,binding.moving.context_trace(model,checked):
            return optimize(task,slot,dynamics,meter)
    finally:
        row['resources']=meter.counts();row['predict_API_resources']=dict(predict=dict(attempted=len(api),completed=sum(e['completed'] for e in api)))


def model_plan(slot,task,selected):
    p=dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=ARM,blind_sha256=slot['blind_sha256'],
        task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],
        with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**selected)
    return dict(p,selection_sha256=core.digest(p))


def plan_all(p,store,runner,rows):
    for slot,row in zip(p['slots'],rows,strict=True):
        task=load_task(slot['task_export'],slot['task_sha256']);require(core.task_id(task)==slot['task_id'],'own moving task identity')
        row['status']='attempted';events=[];s=slot['source_slot']
        def sink(e):events.append(deepcopy(e));store.event(dict(source_slot=s,arm=ARM,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',seed=seed(s),task_sha256=slot['task_sha256']))
        result=runner(task,s,sink,row);selected=choice(task,result,s)
        base.exact_counts(row['resources'],budget());base.exact_counts(row['predict_API_resources'],dict(predict=1802))
        journal_check(task,result,events);row['history_verification']=history_check(task,events)
        stem=f'root_{s}_{ARM}';store.put(stem+'.result.json.gz',result);plan=model_plan(slot,task,selected);store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    limits=Limits(free=lambda:shutil.disk_usage(Path(output).absolute().parent).free)
    p=verify(prepared,expected);limits.check();store=Store(output)
    errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=ARM,blind_sha256=s['blind_sha256'],task_id=s['task_id'],status='not_started',
        commands=None,resources={},predict_API_resources={}) for s in p['slots']]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        for key in ('name','torch_cuda','torch_version'):require(host[key]==p['source_host'][key],'original HOST CUDA numerics '+key)
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device));limits.check()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        limits.check();grasp.enc.frozen(model);before=grasp.enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model hash')
        with sdpa_kernel(SDPBackend.MATH):plan_all(p,store,lambda t,s,k,r:execute(t,s,model,device,torch,k,r,limits),rows)
    except BaseException as exc:
        errors.append(grasp.enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(grasp.enc.error('error_journal',secondary))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:grasp.enc.frozen(model);after=grasp.enc.tensor_hash(model)
            except BaseException as exc:errors.append(grasp.enc.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(grasp.enc.error('final_source',exc))
        try:limits.check()
        except BaseException as exc:errors.append(grasp.enc.error('final_limits',exc))
        try:store.close()
        except BaseException as exc:errors.append(grasp.enc.error('journal_finalize',exc))
        hashes={}
        try:
            for f in store.output.iterdir():
                if f.is_file():hashes[f.name]=sha(f)
        except BaseException as exc:errors.append(grasp.enc.error('artifact_hash',exc))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),
            completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in budget()}
        report=dict(schema=NAME+'.report',status='moving_grasp_model_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,resources=resources,
            predict_API_resources=dict(predict=dict(attempted=sum(r['predict_API_resources'].get('predict',{}).get('attempted',0) for r in rows),
                completed=sum(r['predict_API_resources'].get('predict',{}).get('completed',0) for r in rows))),
            errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,host=host,
            contract=p['contract'],limits=limits.receipt(),commands_released=complete,new_images=0,new_physics=0,
            training=False,P1_admission=False,official_equivalence=False)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:complete=False;errors.append(grasp.enc.error('final_report',exc));print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete and not errors,'partial moving GRASP saved; no retry: '+str(errors));return report


def consume(output,expected):
    """Own command consumer; never accepts old zero-schema or performs physics."""
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'external own report SHA');r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='moving_grasp_model_plans_sealed_not_admitted' and r['commands_released'] is True and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==grasp.enc.MODEL_TENSOR_SHA and r['P1_admission'] is False and
        r['official_equivalence'] is False and r['training'] is False and r['new_images']==r['new_physics']==0,'whole own complete run')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and len(r['rows'])==2,'fixed own moving2 contract')
    l=r['limits'];require(l['checks']>0 and l['minimum_observed_free_bytes']>=FLOOR and l['peak_allocated_bytes']<=3221225472 and
        l['peak_reserved_bytes']<=4294967296 and 0<=l['elapsed_seconds']<600,'actual bounded engineering receipt')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{ARM}.{ext}' for s in (0,1) for ext in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact own output census')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h,'artifact SHA')
    require(sum(f.stat().st_size for f in output.iterdir())<=CAP,'whole64MiB output')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'own execution binding')
    raw,eof=base.inflate((output/'events.jsonl.gz').read_bytes(),limit=256*1024**2)
    require(eof and raw.endswith(b'\n'),'complete bounded single-stream journal');events=[base.decode(line) for line in raw.splitlines()]
    require(all(set(e)=={'source_slot','arm','task_id','event'} and type(e['source_slot']) is int and e['source_slot'] in (0,1) and e['arm']==ARM for e in events),'own branch scope')
    require([e['source_slot'] for i,e in enumerate(events) if i==0 or e['source_slot']!=events[i-1]['source_slot']]==[0,1],'ordered complete two-root census')
    plans=[]
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        s=slot['source_slot'];stem=f'root_{s}_{ARM}';task=load_task(slot['task_export'],slot['task_sha256'])
        require(row['source_slot']==s and row['arm']==ARM and row['blind_sha256']==slot['blind_sha256'] and row['task_id']==slot['task_id'] and
            row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','own moving row/arm/task')
        base.exact_counts(row['resources'],budget());base.exact_counts(row['predict_API_resources'],dict(predict=1802))
        result=read(output/row['result_file']);selected=choice(task,result,s);group=[e['event'] for e in events if e['source_slot']==s]
        require(all(e['task_id']==slot['task_id'] for e in events if e['source_slot']==s),'own journal task')
        journal_check(task,result,group);require(row['history_verification']==history_check(task,group),'all actual moving native inputs')
        plan=read(output/row['plan_file']);require(plan==model_plan(slot,task,selected) and plan['selection_sha256']==row['selection_sha256'] and row['commands']==selected['commands'],'own plan/selection SHA')
        require(group[0]==dict(kind='arm_started',seed=seed(s),task_sha256=slot['task_sha256']) and
            group[-1]==dict(kind='model_selection_sealed',selection_sha256=row['selection_sha256'],commands=selected['commands']),'own durable lifecycle')
        plans.append(dict(plan,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],p['contract']['total_budget']);base.exact_counts(r['predict_API_resources'],dict(predict=3604))
    require(sha(output/'report.json')==expected,'final actual report SHA')
    for path,h in {**p['bindings_sha256'],**{str(output/n):h for n,h in r['artifact_sha256'].items()}}.items():require(sha(path)==h,'final source/artifact closure')
    return dict(schema=NAME+'.consume',status='own_moving_grasp_commands_verified_not_admitted',rows=plans,P1_admission=False,new_physics=0,official_equivalence=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    p.add_argument('--task-exports',nargs=2);p.add_argument('--task-shas',nargs=2)
    for key in ('loader-sha','encoder-report','encoder-sha','sourcepack','source-sha','prepared','prepare-sha','output','report-sha'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output,task_exports=a.task_exports,task_shas=a.task_shas,loader_sha=a.loader_sha,
        encoder_report=a.encoder_report,encoder_sha=a.encoder_sha,sourcepack=a.sourcepack,source_sha=a.source_sha)
    elif a.command=='verify':r=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='consume':r=consume(a.output,a.report_sha)
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode().strip())


if __name__=='__main__':main()
