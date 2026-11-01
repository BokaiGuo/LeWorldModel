"""Fixed moving2 full-H2 open-loop gradient baseline. MAIN-only HOST CUDA.

Own schemas/consumer, frozen gradient math, no old zero-only task loader.
No witness, outcome, training, feedback control or resource-fairness claim.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_gradient_mpc_cuda_20261027_v1 as gradient
from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as moving

NAME='p1_moving_gradient_baseline_cuda_20261027_v1'
ARM='moving_gradient_open_loop'
ROOT,core,base=gradient.ROOT,gradient.core,gradient.base
require,sha,read,canonical=gradient.require,gradient.sha,gradient.read,gradient.canonical
UPDATES,STEP_SIZE=gradient.UPDATES,gradient.STEP_SIZE
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,256*1024
GRADIENT_SHA='2c874aa56cc0b7727dbc20fbcbe60c3c0265bd368a6f5e1d050301f095be5be6'
MOVING_SHA='f8f1d5507e2474fbbbfa56cae98f7bfa99f5929829946a2d7760c2c30ccc03c1'
ENCODER_SHA='e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6'
SOURCE_SHA='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
ENCODER=ROOT/'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json'
SOURCE=ROOT/'results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json'
budget=gradient.budget
optimize,choice,proposal,journal_check=gradient.optimize,gradient.choice,gradient.proposal,gradient.journal_check
Store=base.Store


def contract():
    c=deepcopy(gradient.contract())
    c.update(arm=ARM,interface_source=moving.NAME,history=moving.contract()['history'],
        global_freshness='fixed existing moving two calibration roots; NOT independent confirmation',
        context_evidence='all136 actual predictor latent/token inputs per root, including AD forwards and triple replays',
        predict_accounting='predict meter means predictor.forward; own API trace reconciles model.predict one-to-one',
        output_cap_bytes=CAP,reserve_bytes=FLOOR,final_reserve_bytes=FINAL,
        wall_seconds=600,allocated_bytes=3221225472,reserved_bytes=4294967296,
        official_reproduction=False,feedback_MPC=False,sourcepack_access='SHA only, never parsed or used as planner input')
    return c


def load_task(path,expected):
    # This public loader is moving-specific, with strict nonzero raw10 validation.
    return moving.load_export(path,expected)['gn_task']


def build(task_exports,task_shas,loader_sha,encoder_report,encoder_sha,sourcepack,source_sha):
    require(sha(gradient.__file__)==GRADIENT_SHA and sha(moving.__file__)==MOVING_SHA,'frozen gradient/math and moving interfaces')
    require(Path(encoder_report).absolute()==ENCODER and encoder_sha==ENCODER_SHA and sha(ENCODER)==encoder_sha,'explicit fixed actual moving encoder SHA')
    require(Path(sourcepack).absolute()==SOURCE and source_sha==SOURCE_SHA and sha(SOURCE)==source_sha,'sourcepack SHA-only provenance binding')
    p=moving.build(task_exports,task_shas,loader_sha)  # NEVER original gradient.build/base.load_export.
    r=read(ENCODER)
    require(r['schema']==moving.moving.NAME+'.report' and r['status']=='moving_blind_tasks_encoded_and_sealed' and
        r['errors']==[] and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and len(r['rows'])==2,'complete existing moving encoder')
    require(r['resources']=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')},'inherited paid encoding8/8, not new cost')
    for s,row in zip(p['slots'],r['rows'],strict=True):
        require((row['order'],row['source_slot'],row['status'])==(s['order'],s['source_slot'],'task_sealed') and
            row['task_path']==s['task_export'] and row['task_file_sha256']==s['task_sha256'] and row['task_id']==s['task_id'] and
            row['blind_source_sha256']==s['blind_sha256'],'fixed encoder row / own task export mapping')
    files=set(base.source_files()+moving.moving.source_files()+[Path(gradient.__file__),Path(moving.__file__),Path(__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        ROOT/'tests'/('test_'+gradient.NAME+'.py'),ROOT/'innerdocs'/(gradient.NAME+'.md')])
    bindings=deepcopy(p['bindings_sha256']);bindings.update({str(f.resolve()):sha(f) for f in files})
    bindings.update({str(ENCODER):encoder_sha,str(SOURCE):source_sha})
    for path,h in bindings.items():require(sha(path)==h,'final source/model/export closure')
    arguments=dict(task_exports=list(task_exports),task_shas=list(task_shas),loader_sha=loader_sha,
        encoder_report=str(ENCODER),encoder_sha=encoder_sha,sourcepack=str(SOURCE),source_sha=source_sha)
    return dict(schema=NAME+'.prepare',arguments=arguments,slots=p['slots'],contract=contract(),bindings_sha256=bindings,
        policy=p['policy'],expected_model_hash=p['expected_model_hash'],source_host=r['host'])


def prepare(output,**arguments):
    p=build(**arguments);raw=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(raw)<=256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive <=256KiB prepare +200MiB floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    gradient.io.fsync_directory(path.parent)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external own prepare SHA');p=read(path)
    require(p==build(**p['arguments']),'complete own prepare reconstruction');return p


class Limits:
    def __init__(self,free,clock=time.monotonic,memory=lambda:(0,0)):
        self.free,self.clock,self.memory=free,clock,memory;self.start=clock();self.minimum=None;self.allocated=self.reserved=self.checks=0
    def check(self):
        self.checks+=1;require(self.clock()-self.start<600,'wall600s')
        a,r=self.memory();self.allocated=max(self.allocated,a);self.reserved=max(self.reserved,r)
        f=self.free();self.minimum=f if self.minimum is None else min(self.minimum,f)
        require(a<=3221225472 and r<=4294967296,'allocated3GiB/reserved4GiB');require(f>=FLOOR,'200MiB disk floor')
    def receipt(self):
        return dict(checks=self.checks,minimum_observed_free_bytes=self.minimum,peak_allocated_bytes=self.allocated,
            peak_reserved_bytes=self.reserved,elapsed_seconds=self.clock()-self.start)


class NativeGradient(gradient.NativeGradient):
    def _start(self,task):
        require(np.array_equal(task['root_raw_action_history'],moving.fixed_raw()) and
            np.array_equal(task['root_context']['action_history'],core.normalized_tokens(task,moving.fixed_raw())),
            'actual nonzero moving rawhistory/tokens, never zero substitute')
        return super()._start(task)


@contextmanager
def api_trace(model,sink):
    original=model.predict;index=0;primary=None;entries=[]
    def call(*args,**kwargs):
        nonlocal index
        require(index<budget()['predict'],'136 model.predict API/root cap')
        i=index;index+=1;entry=dict(index=i,completed=False);entries.append(entry)
        sink(dict(kind='predict_API_attempt',index=i))
        result=original(*args,**kwargs)
        entry['completed']=True;sink(dict(kind='predict_API_completed',index=i));return result
    model.predict=call
    try:yield entries
    except BaseException as exc:primary=exc;raise
    finally:
        try:model.predict=original
        except BaseException as exc:
            if primary is not None:primary.add_note('predict API restore '+str(exc))
            else:raise


def history_check(task,events):
    """Own release contract: all neural input receipts vs actual H2 graph paths."""
    expected=[];commands=None;path=[];active=None
    for e in events:
        kind=e['kind']
        if kind in ('evaluation_request','gradient_request'):
            require(active is None,'no overlapping native path');active=kind;commands=e['commands'];path=[]
        elif kind in ('native_packet','gradient_forward_packet'):
            require(active==('evaluation_request' if kind=='native_packet' else 'gradient_request') and e['packet']==len(path),'exact native path scope')
            n=len(path);before=task['root_context'] if n==0 else path[-1];raw=np.asarray(commands,np.float32)[5*n:5*n+5]
            expected.append(moving.incoming(task,before,raw));moving.advance(task,before,raw,e['context']);path.append(e['context'])
        elif kind in ('evaluation_complete','gradient_forward_complete'):
            require(active is not None and len(path)==2,'full H2 history before completion');active=None
    traces=[e for e in events if e['kind']=='native_context_attempt']
    aa=[e['index'] for e in events if e['kind']=='predict_API_attempt'];rr=[e['index'] for e in events if e['kind']=='predict_API_completed']
    require(active is None and len(expected)==budget()['predict'] and [e['index'] for e in traces]==aa==rr==list(range(136)) and
        [e['context'] for e in traces]==expected,'all136 actual latent+token/API inputs include original moving context')
    # API -> action_encoder -> predictor -> return chronology, not just a census.
    sequence=[(e['kind'],e.get('index')) for e in events if e['kind'] in ('predict_API_attempt','native_context_attempt','predict_API_completed')]
    require(sequence==[(k,i) for i in range(136) for k in ('predict_API_attempt','native_context_attempt','predict_API_completed')],'causal API/context order')
    return dict(native_contexts=136,predict_API=136,all_nonzero_history_paths_verified=True)


def execute(task,model,device,torch,sink,row,limits):
    def checked(e):limits.check();sink(e)
    meter=gradient.Meter(checked,lambda:torch.cuda.synchronize(device));gate=gradient.GradGate(torch,meter)
    api=[]
    try:
        backend=NativeGradient(model,device,task)
        with gate,backend.guard(meter),api_trace(model,checked) as api,moving.context_trace(model,checked):
            return optimize(task,backend,meter)
    finally:
        row['resources']=meter.counts();row['AD_API_resources']=gate.counts()
        row['predict_API_resources']=dict(predict=dict(attempted=len(api),completed=sum(e['completed'] for e in api)))


def model_plan(slot,task,selected):
    plan=dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=ARM,blind_sha256=slot['blind_sha256'],
        task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],
        with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**selected)
    return dict(plan,selection_sha256=core.digest(plan))


def plan_all(p,store,runner,rows):
    for slot,row in zip(p['slots'],rows,strict=True):
        task=load_task(slot['task_export'],slot['task_sha256']);require(core.task_id(task)==slot['task_id'],'own moving task identity')
        row['status']='attempted';events=[]
        def sink(e):events.append(deepcopy(e));store.event(dict(source_slot=slot['source_slot'],arm=ARM,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',task_sha256=slot['task_sha256'],updates=UPDATES,step_size=STEP_SIZE))
        result=runner(task,sink,row);selected=choice(task,result)
        base.exact_counts(row['resources'],budget());base.exact_counts(row['AD_API_resources'],dict(grad=32))
        base.exact_counts(row['predict_API_resources'],dict(predict=136))
        journal_check(task,result,events);row['history_verification']=history_check(task,events)
        stem=f'root_{slot["source_slot"]}_{ARM}';store.put(stem+'.result.json.gz',result)
        plan=model_plan(slot,task,selected);store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',
            plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    limits=Limits(free=lambda:shutil.disk_usage(Path(output).absolute().parent).free)
    p=verify(prepared,expected);limits.check();store=Store(output)
    errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=ARM,blind_sha256=s['blind_sha256'],task_id=s['task_id'],status='not_started',
        commands=None,resources={},AD_API_resources={},predict_API_resources={}) for s in p['slots']]
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
        limits.check();gradient.enc.frozen(model);before=gradient.enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen192 tensor hash')
        with sdpa_kernel(SDPBackend.MATH):plan_all(p,store,lambda t,s,r:execute(t,model,device,torch,s,r,limits),rows)
    except BaseException as exc:
        errors.append(gradient.enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(gradient.enc.error('error_journal',secondary))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:gradient.enc.frozen(model);after=gradient.enc.tensor_hash(model)
            except BaseException as exc:errors.append(gradient.enc.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(gradient.enc.error('final_source',exc))
        try:limits.check()
        except BaseException as exc:errors.append(gradient.enc.error('final_limits',exc))
        try:store.close()
        except BaseException as exc:errors.append(gradient.enc.error('journal_finalize',exc))
        hashes={}
        try:
            for f in store.output.iterdir():
                if f.is_file():hashes[f.name]=sha(f)
        except BaseException as exc:errors.append(gradient.enc.error('artifact_hash',exc))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),
            completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in budget()}
        report=dict(schema=NAME+'.report',status='moving_gradient_model_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,resources=resources,
            predict_API_resources=dict(predict=dict(attempted=sum(r['predict_API_resources'].get('predict',{}).get('attempted',0) for r in rows),
                completed=sum(r['predict_API_resources'].get('predict',{}).get('completed',0) for r in rows))),
            AD_API_resources=dict(grad=dict(attempted=sum(r['AD_API_resources'].get('grad',{}).get('attempted',0) for r in rows),
                completed=sum(r['AD_API_resources'].get('grad',{}).get('completed',0) for r in rows))),
            errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,contract=p['contract'],limits=limits.receipt(),commands_released=complete,new_images=0,new_physics=0,
            training=False,new_Q=False,P1_admission=False,equal_total_resource_claim=False)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:errors.append(gradient.enc.error('final_report',exc));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete and not errors,'partial moving gradient saved; no retry: '+str(errors));return report


def consume(output,expected):
    """Own saved command-release contract, not admission or physical execution."""
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'external actual report SHA')
    r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='moving_gradient_model_plans_sealed_not_admitted' and
        r['commands_released'] is True and r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and
        r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==gradient.enc.MODEL_TENSOR_SHA,'whole own completed CUDA run')
    require(all(r[k] is False for k in ('P1_admission','new_Q','training','equal_total_resource_claim')) and r['new_images']==r['new_physics']==0,'no scope upgrade')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and len(r['rows'])==2,'own complete two-root contract')
    l=r['limits'];require(l['checks']>0 and l['minimum_observed_free_bytes']>=FLOOR and l['peak_allocated_bytes']<=3221225472 and
        l['peak_reserved_bytes']<=4294967296 and 0<=l['elapsed_seconds']<600,'actual bounded resource receipt')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{ARM}.{ext}' for s in (0,1) for ext in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact own artifact census')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'artifact SHA')
    require(sum(f.stat().st_size for f in output.iterdir())<=CAP,'8MiB whole run')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'binding identity')
    raw,eof=base.inflate((output/'events.jsonl.gz').read_bytes());require(eof and raw.endswith(b'\n'),'complete gzip journal')
    events=[base.decode(line) for line in raw.splitlines()]
    require(all(set(e)=={'source_slot','arm','task_id','event'} and type(e['source_slot']) is int and e['source_slot'] in (0,1) and e['arm']==ARM for e in events),'own scoped journal only')
    require([e['source_slot'] for k,e in enumerate(events) if k==0 or e['source_slot']!=events[k-1]['source_slot']]==[0,1],'causal two-root census')
    plans=[]
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        s=slot['source_slot'];stem=f'root_{s}_{ARM}';task=load_task(slot['task_export'],slot['task_sha256'])
        require(row['source_slot']==s and row['task_id']==slot['task_id'] and row['blind_sha256']==slot['blind_sha256'] and row['arm']==ARM and
            row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','own row/task/arm identity')
        base.exact_counts(row['resources'],budget());base.exact_counts(row['AD_API_resources'],dict(grad=32))
        base.exact_counts(row['predict_API_resources'],dict(predict=136))
        result=read(output/row['result_file']);selected=choice(task,result);group=[e['event'] for e in events if e['source_slot']==s]
        require(all(e['task_id']==slot['task_id'] for e in events if e['source_slot']==s),'own scoped task id')
        journal_check(task,result,group);require(row['history_verification']==history_check(task,group),'actual neural input chronology')
        require(group[0]==dict(kind='arm_started',task_sha256=slot['task_sha256'],updates=UPDATES,step_size=STEP_SIZE) and
            group[-1]==dict(kind='model_selection_sealed',selection_sha256=row['selection_sha256'],commands=selected['commands']),'durable own lifecycle')
        plan=read(output/row['plan_file']);require(plan==model_plan(slot,task,selected) and plan['selection_sha256']==row['selection_sha256'] and row['commands']==selected['commands'],'own selection content/SHA')
        plans.append(dict(plan,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],p['contract']['total_budget']);base.exact_counts(r['AD_API_resources'],dict(grad=64))
    base.exact_counts(r['predict_API_resources'],dict(predict=272))
    require(sha(output/'report.json')==expected,'final report SHA')
    for path,h in {**p['bindings_sha256'],**{str(output/n):h for n,h in r['artifact_sha256'].items()}}.items():require(sha(path)==h,'final closure drift')
    return dict(schema=NAME+'.consume',status='own_moving_gradient_commands_verified_not_admitted',rows=plans,new_physics=0,P1_admission=False)


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
