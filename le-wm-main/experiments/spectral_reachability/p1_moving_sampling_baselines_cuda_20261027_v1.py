"""Moving-history CEM/nominal-MPPI adaptation, own schema; MAIN CUDA only.

No zero-only interface/build/consumer, no new encoding/physics/AD or official
MPPI/equal-compute claim. Complete per-arm prepare/run/consume closure.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_sampling_cuda_20261027_v1 as cem
from experiments.spectral_reachability import p1_constructive_mppi_cuda_20261027_v2 as nominal
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving

NAME='p1_moving_sampling_baselines_cuda_20261027_v1'
base,gn,enc,core=cem.base,cem.gn,cem.enc,moving.core
ROOT=base.ROOT
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
CAP,FLOOR,FINAL=base.CAP,base.FLOOR,base.FINAL
Store=base.Store
ARMS=('CEM',nominal.ARM)
REPORT=ROOT/'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json'
REPORT_SHA='e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6'
SOURCE_AUTHORITY_SHA='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
PRIMITIVES={cem.__file__:'5fe2af6256071d4a056a9f40908ee616c284e7b72e61a5c1fc5a79e359b86862',
    nominal.__file__:'f6bb9c882105d102c3292264f0018745c41ede8a8adcd2c3ab0d47dfcef6c920',
    nominal.kernels.__file__:'e1978044c81c6313b4b0a18fb7417cca4c46578860cc382909770641b4aafc04',
    moving.__file__:'246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63'}


def checked_arms(arms=None):
    arms=ARMS if arms is None else tuple(arms)
    require(arms in ((ARMS[0],),(ARMS[1],),ARMS),'fixed ordered one/both arms; no duplicates or unknown arm')
    return arms


def budget(arm):
    require(arm in ARMS,'own arm')
    return cem.budget() if arm==ARMS[0] else nominal.budget()


def seed(slot,arm):
    require(type(slot) is int and slot in (0,1) and arm in ARMS,'fixed two roots/own arm')
    return cem.seed(slot,'CEM') if arm==ARMS[0] else nominal.seed(slot)


def contract(arms=None):
    arms=checked_arms(arms)
    return dict(arms=list(arms),roots=2,order='source_slot ascending, CEM then nominal MPPI when selected',
        per_root_budget={a:budget(a) for a in arms},
        total_budget={k:2*sum(budget(a)[k] for a in arms) for k in ('predict','action_encoder')},
        seeds={a:[seed(s,a) for s in (0,1)] for a in arms},
        CEM=deepcopy(cem.contract()['CEM']),CEM_candidates=64,CEM_fresh_H2_replays=3,
        CEM_selection='stable global argmin over ALL64; independent nominal/whole/segmented H2',
        MPPI=deepcopy(nominal.contract()),
        MPPI_selection='last updated nominal index34, NOT best sample; one independent whole H2 replay',
        model_metric='original native projected CLS192, identity Q mean192 terminal squared error',
        horizon=2,packet_steps=5,packet_logit_cap=float(.75*np.sqrt(5)),
        raw='float32 tanh after sealed per-packet logit-ball projection; original normalizer, no second /100',
        history='three original observed latents and TWO actual nonzero normalized incoming packets; rolled each H1',
        with_target=False,numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        samples_per_native_call=1,context_tokens_per_native_call=3,
        inherited_encoding_once=dict(encoder=dict(attempted=8,completed=8),projector=dict(attempted=8,completed=8)),
        inherited_encoding_per_root=dict(encoder=4,projector=4),
        inherited_source_authority_sha256=SOURCE_AUTHORITY_SHA,source_authority_scope='MAIN attestation; sourcepack not opened by planners',
        encoder_report_sha256=REPORT_SHA,new_encoder=0,new_projector=0,AD=0,jacobian=0,VJP=0,SVD=0,solve=0,
        new_physics=0,training=False,new_Q=False,P1_admission=False,independent_confirmation=False,
        official_MPPI=False,importance_correction=False,projection_density_correction=False,
        equal_total_compute_claim=False,paper_strong_MPPI_requirement_closed=False,
        no_CPU_neural_fallback=True,no_retries=True,output_cap_bytes=CAP,reserve_bytes=FLOOR,
        final_reserve_bytes=FINAL,prepare_cap_bytes=256*1024,
        count_scope='actual predictor and action_encoder entries/returns; pred_proj enclosed in predict, not separately metered; not FLOPs',
        visual_cost='historical8+8 counted ONCE per invocation, not per arm and not free total pipeline')


def interface(value):
    require(set(value)=={'gn_task','d2p_packet','q','task_id'},'moving public loader fields only')
    t=value['gn_task'];core.validate_task(t)
    require(t['environment']=='swm/PushT-v1' and np.array_equal(t['projection'],np.eye(192)) and
        t['packet_logit_cap']==float(.75*np.sqrt(5)) and
        np.array_equal(t['root_raw_action_history'],moving.fixed_raw().astype(float)) and
        np.array_equal(t['root_context']['action_history'],core.normalized_tokens(t,moving.fixed_raw())) and
        np.array_equal(t['action_mean'],moving.normalization()['mean']) and
        np.array_equal(t['action_std'],moving.normalization()['std']),'own nonzero history/original cap/Q/normalizer')
    require(value['task_id']==core.task_id(t) and value['q']==t['projection'] and
        value['d2p_packet']==dict(root_context=t['root_context'],goal=t['goal']),'same public task')
    return deepcopy(value)


def load_export(path,expected):
    return interface(moving.load_task(path,expected))


def build(arms=None):
    arms=checked_arms(arms)
    for path,h in PRIMITIVES.items():require(sha(path)==h,'sealed primitive/loader source')
    require(sha(REPORT)==REPORT_SHA,'MAIN approved actual moving encoder report SHA')
    r=read(REPORT)
    require(r['schema']==moving.NAME+'.report' and r['status']=='moving_blind_tasks_encoded_and_sealed' and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==moving.low.MODEL_TENSOR_SHA and r['with_target'] is False and
        r['new_predict']==r['new_actionencoder']==r['new_AD']==r['new_physics']==0 and
        r['contract']==moving.contract() and len(r['rows'])==2,'actual complete frozen moving encoding')
    inherited=contract(arms)['inherited_encoding_once'];require(r['resources']==inherited,'exact inherited encoding cost')
    bindings={str(REPORT):REPORT_SHA,**moving.MODELS};slots=[]
    expected_names={'binding.json','image.events.jsonl','task_00.json.gz','task_01.json.gz'}|{f'image_{i:02d}.native192.npy' for i in range(8)}
    require(set(r['artifact_sha256'])==expected_names,'full original encoding artifact closure')
    for name,h in r['artifact_sha256'].items():
        f=REPORT.parent/name;require(sha(f)==h,'encoding artifact SHA');bindings[str(f)]=h
    for i,row in enumerate(r['rows']):
        require(type(row['source_slot']) is int and row['source_slot']==row['order']==i and
            row['status']=='task_sealed' and len(row['images'])==4 and
            row['task_path']==str(REPORT.parent/f'task_{i:02d}.json.gz') and
            row['task_file_sha256']==r['artifact_sha256'][f'task_{i:02d}.json.gz'],'two fixed moving export rows')
        path,h=row['task_path'],row['task_file_sha256'];v=load_export(path,h);e=read(path)
        require(e['source_slot']==i and e['blind_source_sha256']==e['blind_sha256']==row['blind_source_sha256'] and
            e['task_contract']['prepare_sha256']==r['prepare_sha256'] and
            e['task_contract']['manifest_sha256']==r['arguments']['manifest_sha'] and
            v['task_id']==row['task_id'],'moving export/encoder prepare/manifest/task identity')
        for j,im in enumerate(row['images']):
            name=f'image_{4*i+j:02d}.native192.npy'
            require(im==dict(path=str(REPORT.parent/name),sha256=r['artifact_sha256'][name],dtype='float32',shape=[192]),'exact four encoded image receipts')
        slots.append(dict(order=i,source_slot=i,task_export=path,task_sha256=h,task_id=v['task_id'],
            blind_sha256=e['blind_sha256'],task_contract_sha256=v['gn_task']['task_contract_sha256']))
    files=set(base.source_files()+moving.source_files()+[Path(__file__),Path(cem.__file__),Path(nominal.__file__),
        Path(nominal.kernels.__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')])
    for f in files:bindings[str(f.resolve())]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'build source/model/export TOCTOU')
    return dict(schema=NAME+'.prepare',arguments=dict(arms=list(arms)),slots=slots,policy=str(moving.POLICY),
        expected_model_hash=moving.low.MODEL_TENSOR_SHA,contract=contract(arms),bindings_sha256=bindings,
        inherited_encoding_report=str(REPORT),inherited_encoding_report_sha256=REPORT_SHA,
        global_freshness='fixed_old_two_moving_roots_not_independent_confirmation')


def prepare(output,arms=None):
    p=build(arms);raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'NEW small prepare +200MiB floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path)
    require(p==build(**p['arguments']),'entire OWN moving source/model/export/protocol reconstruction')
    return p


def optimize(task,slot,arm,backend,meter):
    require(arm in ARMS,'own optimizer arm')
    return cem.optimize(task,slot,'CEM',backend,meter) if arm==ARMS[0] else nominal.optimize(task,slot,backend,meter)


def choice(task,result,slot,arm):
    require(arm in ARMS,'own selection arm')
    return cem.choice(task,result,slot,'CEM') if arm==ARMS[0] else nominal.choice(task,result,slot)


def execute(task,slot,arm,model,device,torch,sink,row):
    require(arm in ARMS,'own native arm')
    meter=(cem.Meter if arm==ARMS[0] else nominal.Meter)(sink,lambda:torch.cuda.synchronize(device))
    try:
        backend=gn.screen.NativeCuda(model,device,task)
        with torch.inference_mode(),backend.guard(meter):return optimize(task,slot,arm,backend,meter)
    finally:row['resources']=meter.counts()


def journal_check(task,slot,arm,result,events):
    if arm==ARMS[0]:cem.journal_check(task,slot['source_slot'],'CEM',result,events)
    else:nominal.journal_check(task,result,slot,events)


def model_plan(slot,arm,task,selected):
    return dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=arm,blind_sha256=slot['blind_sha256'],
        task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],
        with_target=False,history_semantics='actual_moving_raw_between_observed0_5_10',
        encoder_report_sha256=REPORT_SHA,source_authority_sha256=SOURCE_AUTHORITY_SHA,
        physical_execution=False,selection_uses_physics=False,P1_admission=False,official_MPPI=False,**selected)


def plan_all(p,store,runner,rows):
    for row in rows:
        slot=p['slots'][row['source_slot']];arm=row['arm'];s=slot['source_slot']
        task=load_export(slot['task_export'],slot['task_sha256'])['gn_task'];row['status']='attempted'
        def sink(e):store.event(dict(source_slot=s,arm=arm,task_id=slot['task_id'],event=e))
        start=dict(kind='arm_started',task_sha256=slot['task_sha256'],seed=seed(s,arm))
        if arm==ARMS[1]:start['temperature']=nominal.TEMPERATURE
        sink(start)
        result=runner(task,s,arm,sink,row);selected=choice(task,result,s,arm)
        base.exact_counts(row['resources'],budget(arm));stem=f'root_{s}_{arm}'
        store.put(stem+'.result.json.gz',result)
        plan=model_plan(slot,arm,task,selected);plan['selection_sha256']=core.digest(plan)
        store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',
            plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=a,blind_sha256=s['blind_sha256'],task_id=s['task_id'],status='not_started',commands=None,resources={}) for s in p['slots'] for a in p['contract']['arms']]
    try:
        print('Fixed moving-history CEM/nominal MPPI, unequal forward budgets, no new encoding/AD/physics or official MPPI claim.',flush=True)
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
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in ('predict','action_encoder')}
        report=dict(schema=NAME+'.report',status='moving_sampling_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,resources=resources,errors=errors,first_error=errors[0] if errors else None,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,contract=p['contract'],inherited_encoding_once=p['contract']['inherited_encoding_once'],commands_released=complete,new_images=0,new_physics=0,training=False,P1_admission=False,equal_D2P_compute_claim=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:complete=False;errors.append(enc.error('final_report',e));print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial preserved, no retry: '+str(errors));return report


def consume(output,expected):
    output=Path(output).absolute();require(sha(output/'report.json')==expected,'external actual report SHA')
    r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='moving_sampling_plans_sealed_not_admitted' and
        r['commands_released'] is True and r['errors']==[] and r['first_error'] is None and
        r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==moving.low.MODEL_TENSOR_SHA,'complete own moving CUDA closure')
    p=verify(r['prepared'],r['prepare_sha256']);arms=checked_arms(p['contract']['arms'])
    order=[(s,a) for s in (0,1) for a in arms]
    require(r['contract']==p['contract'] and len(r['rows'])==len(order) and
        r['inherited_encoding_once']==p['contract']['inherited_encoding_once'],'fixed moving arms/root/inherited cost')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{a}.{suffix}' for s,a in order for suffix in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact own artifacts')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'saved artifact hash')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution prepare binding')
    plain,eof=base.inflate((output/'events.jsonl.gz').read_bytes())
    require(eof and plain.endswith(b'\n'),'complete bounded lossless journal')
    events=[base.decode(line) for line in plain.splitlines()];groups=[]
    for e in events:
        require(set(e)=={'source_slot','arm','task_id','event'},'only own scoped nonerror events')
        key=(e['source_slot'],e['arm'])
        if not groups or groups[-1]!=key:groups.append(key)
    require(groups==order,'no resumed/hidden/duplicated/out-of-order arm')
    plans=[]
    for row,(s,arm) in zip(r['rows'],order,strict=True):
        slot=p['slots'][s];stem=f'root_{s}_{arm}'
        require(row['source_slot']==s and row['arm']==arm and row['task_id']==slot['task_id'] and
            row['blind_sha256']==slot['blind_sha256'] and row['status']=='model_plan_sealed' and
            row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','strict moving branch join')
        task=load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        result=read(output/row['result_file']);selected=choice(task,result,s,arm)
        plan=read(output/row['plan_file']);ident=plan.pop('selection_sha256')
        require(plan==model_plan(slot,arm,task,selected) and core.digest(plan)==ident==row['selection_sha256'] and
            row['commands']==selected['commands'],'own schema source-bound chosen action witness')
        base.exact_counts(row['resources'],budget(arm))
        group=[e for e in events if e['source_slot']==s and e['arm']==arm]
        require(all(e['task_id']==slot['task_id'] for e in group),'all journal contexts belong to moving task')
        scope=[e['event'] for e in group]
        start=dict(kind='arm_started',task_sha256=slot['task_sha256'],seed=seed(s,arm))
        if arm==ARMS[1]:start['temperature']=nominal.TEMPERATURE
        require(scope[0]==start and scope[-1]==dict(kind='model_selection_sealed',selection_sha256=ident,commands=selected['commands']),
            'fixed lifecycle and post-replay selected-action seal')
        journal_check(task,slot,arm,result,scope)
        plans.append(dict(plan,selection_sha256=ident,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],p['contract']['total_budget'])
    require(sha(output/'report.json')==expected,'final report TOCTOU')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final source/model/input binding')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'final artifact TOCTOU')
    return dict(schema=NAME+'.consume',rows=plans,inherited_encoding_once=p['contract']['inherited_encoding_once'],
        resources=r['resources'],P1_admission=False,independent_confirmation=False,official_MPPI=False,
        equal_total_compute_claim=False,new_physics=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+key)
    parser.add_argument('--arm',choices=('both',*ARMS),default='both',help='prepare-time only; run uses sealed arms')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.output,ARMS if args.arm=='both' else (args.arm,))
    elif args.command=='verify':result=dict(status='verified_no_execution',contract=verify(args.prepared,args.prepare_sha)['contract'])
    elif args.command=='consume':result=consume(args.output,args.report_sha)
    else:
        require(args.arm=='both','run cannot override prepared arm selection')
        result=run(args.prepared,args.prepare_sha,args.output)
    print(canonical(result).decode().strip())


if __name__=='__main__':main()
