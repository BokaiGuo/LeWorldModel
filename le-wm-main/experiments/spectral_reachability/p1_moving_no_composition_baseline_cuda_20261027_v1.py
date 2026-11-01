"""MAIN-only moving-task joint spectral baseline, then packet-ball mapping.

Not a prefix compiler, random/CEM rename, original joint-ball distribution,
equal-compute comparison, or scientific admission. No physical inputs/outcomes.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_moving_sampling_baselines_cuda_20261027_v1 as inputs
from experiments.spectral_reachability import p2_no_composition_sra_20261027_v2 as spectral

NAME='p1_moving_no_composition_baseline_cuda_20261027_v1'
ARM='joint_H2_spectral_packet_mapped'
base,gn,enc,core=inputs.base,inputs.gn,inputs.enc,inputs.core
ROOT=inputs.ROOT
require,sha,read,canonical=inputs.require,inputs.sha,inputs.read,inputs.canonical
Store=inputs.Store
CAP,FLOOR,FINAL=inputs.CAP,inputs.FLOOR,inputs.FINAL
load_export=inputs.load_export


def budget():
    return dict(predict=88,action_encoder=88,jacobian=1,vjp_directions=192,svd=2,bounded_solve=1)


def contract():
    return dict(arm=ARM,roots=2,horizon=2,candidates=40,
        per_root_budget=budget(),total_budget={k:2*v for k,v in budget().items()},
        calls=dict(nominal=2,AD_forward=2,candidates=80,winner_whole=2,winner_segmented=2),
        metric='original native192 identity mean squared terminal goal error',
        linear_ridge='UNNORMALIZED residual squared + .002*norm(step)^2; mean192 equivalent damping .002/192',
        joint_cap=float(.75*np.sqrt(10)),joint_radii=[float(.25*np.sqrt(10)),float(.75*np.sqrt(10))],
        packet_cap=float(.75*np.sqrt(5)),bank='original zero1+DLS(.25,.5,1)3+top9*2radii*2sign36',
        mapping='joint bank THEN independent float32 packet-ball projection, float32 tanh; changed support, NOT original joint-ball distribution',
        selection='ALL40 full nonlinear original Q, stable first argmin; no Vreach/prefix score',
        Jacobian='one192x20 full-H2 zero-logit graph, no detach/recompile;192 scalar autograd.grad calls',
        SVD='bounded ridge one SVD, mode bank one SVD; bounded solve is not a dense solve API call',
        history='original3 observed states and2 actual nonzero incoming packets, rolling without second /100',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        inherited_encoding_once=inputs.contract(('CEM',))['inherited_encoding_once'],
        encoder_report_sha256=inputs.REPORT_SHA,source_authority_sha256=inputs.SOURCE_AUTHORITY_SHA,
        source_scope='MAIN attestation; sourcepack/witness/outcomes not opened',
        verification_CPU='consumer bank reconstruction repeats2 CPU SVD/1bounded solve/40packet projections per root, not a planner/neural call',
        CPU_work_per_root=dict(packet_pair_projections=40,planning_candidate_scores=40,
            planning_reference_replay_scores=3,run_verification_Q_scores=41,stable_selections=2),
        consume_CPU_per_root=dict(svd=2,bounded_solve=1,packet_pair_projections=40,Q_scores=84,stable_selection=1),
        new_encoder=0,new_projector=0,new_physics=0,training=False,new_Q=False,
        P1_admission=False,P2_admission=False,independent_confirmation=False,equal_total_compute_claim=False,
        output_cap_bytes=CAP,reserve_bytes=FLOOR,final_reserve_bytes=FINAL,prepare_cap_bytes=256*1024,
        no_CPU_neural_fallback=True,no_retries=True)


def build():
    # Authenticated moving input bridge ONLY; no CEM computation/results read.
    for path,h in ((inputs.__file__,'f30a1c5ac5ef618cfe3add37326aae29513278fe7c4319b5a743d9f6422300e6'),
                   (spectral.__file__,'6e1402ac8abe3aabfffefd24aa016c341e6eb4f40d3f60e233152460ca70e45a'),
                   (spectral.compiler.__file__,'f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a')):
        require(sha(path)==h,'approved immutable mathematical/input primitive')
    p=inputs.build(('CEM',))
    p.pop('arguments');p.update(schema=NAME+'.prepare',contract=contract())
    for f in (Path(__file__),Path(spectral.__file__),Path(spectral.compiler.__file__),
              ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        p['bindings_sha256'][str(f.resolve())]=sha(f)
    todo=[Path(__file__).resolve()];seen=set()
    while todo:
        f=todo.pop()
        if f in seen:continue
        seen.add(f);p['bindings_sha256'][str(f)]=sha(f)
        for node in ast.walk(ast.parse(f.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments'):
                names=[node.module]+[node.module+'.'+x.name for x in node.names]
            for name in names:
                if not name.startswith('experiments.'):continue
                child=(ROOT/Path(*name.split('.'))).with_suffix('.py').resolve()
                if child.is_file() and child not in seen:todo.append(child)
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'own source/model/input TOCTOU')
    return p


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<=256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'NEW prepare +200MiB floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path)
    require(p==build(),'OWN full moving source/model/task/protocol binding');return p


class Meter(gn.Meter):
    def __init__(self,sink,sync=lambda:None):
        gn.screen.Meter.__init__(self,budget(),sink);self.sync=sync

    def begin(self,resource):
        require(all(e['completed'] or e['resource'] in ('jacobian','bounded_solve') for e in self.extended),'failed work cannot retry')
        return gn.screen.Meter.begin(self,resource)


def bank(task,jacobian,terminal,meter=None):
    pending={}
    def journal(e):
        if meter is None:return
        name=e['kind']
        if 'wall_time_s' not in e:
            if name=='bounded_ridge_svd':pending['solve']=meter.begin('bounded_solve')
            pending[name]=meter.begin('svd')
        elif e['completed']:
            meter.end(pending.pop(name))
            if name=='bounded_ridge_svd':meter.end(pending.pop('solve'))
        meter.emit('spectral_operation',operation=e)
    original=spectral.spectral_bank(jacobian,terminal,task['goal'],task['projection'],journal=journal)
    xs=[gn.screen.projected_logits(np.asarray(x).reshape(2,10),task['packet_logit_cap']).tolist() for x in original['logits']]
    return dict(original_joint=original,logits=xs,
        commands=[np.tanh(np.asarray(x,np.float32)).reshape(10,2).astype(float).tolist() for x in xs])


def optimize(task,slot,backend,meter):
    core.validate_task(task);meter.group='nominal'
    nominal=gn.single(task,np.zeros((2,10)),backend,meter,'whole_replay')
    meter.group='joint_J'
    terminal,j=meter.call('jacobian',lambda:backend.jacobian(task,nominal,meter))
    require(np.array_equal(terminal,nominal['contexts'][-1]['latent_history'][-1]),'AD nominal exact')
    b=bank(task,j,terminal,meter);meter.emit('bank_frozen',bank=b)
    candidates=[]
    for i,x in enumerate(b['logits']):
        meter.group=f'candidate_{i}'
        p=gn.single(task,x,backend,meter,'whole_replay');candidates.append(p)
    index=min(range(40),key=lambda i:(candidates[i]['cost'],i))
    meter.emit('winner_frozen_before_replay',index=index)
    replays=[]
    for mode in ('whole_replay','segmented_replay'):
        meter.group='winner_'+mode
        p=gn.single(task,b['logits'][index],backend,meter,mode)
        require(p==candidates[index],'fresh H2 whole/segmented exact');replays.append(p)
    base.exact_counts(meter.counts(),budget())
    return dict(task_id=core.task_id(task),source_slot=slot,arm=ARM,nominal=nominal,jacobian=np.asarray(j).tolist(),
        bank=b,candidates=candidates,winner_index=index,replays=replays,resources=meter.counts())


def check_single(task,p):
    require(set(p)=={'task_id','logits','commands','contexts','cost'} and p['task_id']==core.task_id(task),'own single-H2 prediction schema')
    x=core.array(p['logits'],(2,10));raw=core.array(p['commands'],(10,2))
    require(np.array_equal(x,x.astype(np.float32)) and (np.linalg.norm(x,axis=1)<=task['packet_logit_cap']).all() and
        np.array_equal(raw,np.tanh(x.astype(np.float32)).reshape(10,2)),'float32 feasible logits/exact raw')
    gn.saved.check_path(task,raw,p['contexts'])
    require(p['cost']==core.model_cost(task,p['contexts'][-1]['latent_history'][-1]),'original full nonlinear Q')


def choice(task,result,slot,reconstruct=True):
    require(result['task_id']==core.task_id(task) and result['source_slot']==slot and result['arm']==ARM,'own moving root/arm')
    base.exact_counts(result['resources'],budget())
    n=result['nominal'];check_single(task,n)
    require(n['logits']==np.zeros((2,10)).tolist(),'zero nominal')
    b=bank(task,result['jacobian'],n['contexts'][-1]['latent_history'][-1]) if reconstruct else result['bank']
    require(result['bank']==b and len(result['candidates'])==40,'exact original joint and packet mapped ALL40')
    for i,p in enumerate(result['candidates']):
        check_single(task,p)
        require(p['logits']==b['logits'][i] and p['commands']==b['commands'][i],'all candidate actions')
    index=min(range(40),key=lambda i:(result['candidates'][i]['cost'],i));p=result['candidates'][index]
    require(result['winner_index']==index and result['replays']==[p,p],'stable nonlinear selection and double replay')
    return dict(task_id=core.task_id(task),logits=p['logits'],commands=p['commands'],prediction=p['contexts'][-1],model_terminal_cost=p['cost'])


def journal_check(task,result,events):
    attempts={};completed={};paths=[];current=None;packets=[]
    for e in events:
        k=e['kind'];require('error' not in k and 'forbidden' not in k,'no error/forbidden event')
        if k=='operation_attempted':
            v=e['event'];i=v['index'];require(i==len(attempts) and not v['completed'] and v['samples']==1 and v['resource'] in budget() and v['evaluation_id']==e['evaluation_id'],'ordered actual attempts');attempts[i]=v
        elif k=='operation_completed':
            v=e['event'];i=v['index'];require(i in attempts and i not in completed and v==dict(attempts[i],completed=True),'actual return identity');completed[i]=v
        elif k=='evaluation_request':
            require(current is None,'no overlapping evaluation');current=e;packets=[]
        elif k=='native_packet':
            require(current is not None and e['evaluation_id']==current['evaluation_id'] and e['mode']==current['mode'] and e['packet']==len(packets),'packet scope/order');packets.append(e['context'])
        elif k=='evaluation_complete':
            require(current is not None and len(packets)==2 and e['evaluation_id']==current['evaluation_id'] and e['mode']==current['mode'],'complete H2')
            gn.saved.check_path(task,np.asarray(current['commands']),packets)
            p=dict(task_id=core.task_id(task),logits=current['logits'],commands=current['commands'],contexts=packets,cost=core.model_cost(task,packets[-1]['latent_history'][-1]))
            require(e['prediction']==p,'saved contexts/cost');paths.append((current['evaluation_id'],current['mode'],p));current=None
    require(current is None and len(attempts)==len(completed),'no incomplete operation')
    base.exact_counts({k:dict(attempted=sum(v['resource']==k for v in attempts.values()),completed=sum(v['resource']==k for v in completed.values())) for k in budget()},budget())
    wanted=[('nominal','whole_replay',result['nominal'])]+[(f'candidate_{i}','whole_replay',p) for i,p in enumerate(result['candidates'])]+[('winner_'+mode,mode,p) for mode,p in zip(('whole_replay','segmented_replay'),result['replays'],strict=True)]
    require(paths==wanted,'exact nominal/40slots/independent whole+segmented paths')
    for ident,_,_ in wanted:
        native=[v for v in attempts.values() if v['evaluation_id']==ident]
        require(len(native)==4 and sorted(v['resource'] for v in native)==['action_encoder','action_encoder','predict','predict'],'two real native module pairs per H2')
    ad_ops=[v for v in attempts.values() if v['evaluation_id']=='joint_J']
    require(len(ad_ops)==200 and all(v['evaluation_id']=='joint_J' or v['resource'] in ('predict','action_encoder') for v in attempts.values()),'AD+compile operations separately scoped')
    require([e['jacobian'] for e in events if e['kind']=='joint_logit_jacobian']==[result['jacobian']],'durable full joint J')
    require(len([e for e in events if e['kind']=='jacobian_row'])==192 and len([e for e in events if e['kind']=='jacobian_forward_packet'])==2,'all AD rows/forward packets')
    raw_rows=[e for e in events if e['kind']=='jacobian_row']
    require([e['row'] for e in raw_rows]==list(range(192)) and
        np.array_equal(np.asarray([e['raw_control_derivative'] for e in raw_rows]),result['jacobian']),
        'zero-logit tanh chain is identity; all192 raw rows equal saved J')
    ad=[e for e in events if e['kind']=='jacobian_forward_packet']
    require([e['packet'] for e in ad]==[0,1] and [e['context'] for e in ad]==result['nominal']['contexts'],'AD nominal full H2 contexts')
    require([e['terminal'] for e in events if e['kind']=='jacobian_forward_complete']==[result['nominal']['contexts'][-1]['latent_history'][-1]],'AD terminal exact')
    frozen=[i for i,e in enumerate(events) if e['kind']=='bank_frozen'];selected=[i for i,e in enumerate(events) if e['kind']=='winner_frozen_before_replay']
    require(len(frozen)==len(selected)==1 and events[frozen[0]]['bank']==result['bank'] and events[selected[0]]['index']==result['winner_index'],'durable frozen bank/selection')
    requests=[i for i,e in enumerate(events) if e['kind']=='evaluation_request']
    require(requests[0]<frozen[0]<requests[1] and requests[40]<selected[0]<requests[41],'pre-dispatch bank/selection')


def model_plan(slot,task,selected):
    return dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=ARM,blind_sha256=slot['blind_sha256'],
        task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],
        encoder_report_sha256=inputs.REPORT_SHA,source_authority_sha256=inputs.SOURCE_AUTHORITY_SHA,
        with_target=False,history_semantics='actual_moving_raw_between_observed0_5_10',
        physical_execution=False,P1_admission=False,P2_admission=False,**selected)


def execute(task,model,torch,device,sink,row):
    meter=Meter(sink,lambda:torch.cuda.synchronize(device))
    try:
        backend=gn.screen.NativeCuda(model,device,task)
        with torch.no_grad(),backend.guard(meter):return optimize(task,row['source_slot'],backend,meter)
    finally:row['resources']=meter.counts()


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],task_id=s['task_id'],status='not_started',resources={}) for s in p['slots']]
    try:
        print('Fixed two moving roots: joint full-H2 spectral bank then packet projection; unequal total compute, no admission or physical claim.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original model')
        with sdpa_kernel(SDPBackend.MATH):
            for slot,row in zip(p['slots'],rows,strict=True):
                task=load_export(slot['task_export'],slot['task_sha256'])['gn_task'];row['status']='attempted'
                def sink(e):store.event(dict(source_slot=slot['source_slot'],task_id=slot['task_id'],event=e))
                result=execute(task,model,torch,device,sink,row);selected=choice(task,result,slot['source_slot'],reconstruct=False)
                # The independent saved-data consumer reconstructs; run does not repeat SVDs.
                stem=f"root_{slot['source_slot']}";store.put(stem+'.result.json.gz',result)
                plan=model_plan(slot,task,selected);plan['selection_sha256']=core.digest(plan)
                store.put(stem+'.plan.json',plan);row.update(status='model_plan_sealed',result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])
    except BaseException as e:
        errors.append(enc.error('execution',e))
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
        hashes={}
        try:
            for f in store.output.iterdir():
                if f.is_file():hashes[f.name]=sha(f)
        except BaseException as e:errors.append(enc.error('artifact_hash',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in budget()}
        report=dict(schema=NAME+'.report',status='moving_spectral_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,resources=resources,contract=p['contract'],
            errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,commands_released=complete,P1_admission=False,P2_admission=False,
            runtime_verification_CPU=dict(svd=0,bounded_solve=0))
        try:store.put('report.json',report,final=True)
        except BaseException as e:complete=False;errors.append(enc.error('final_report',e));print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial retained/no retry: '+str(errors));return report


def consume(output,expected):
    output=Path(output).absolute();require(sha(output/'report.json')==expected,'external actual report SHA');r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='moving_spectral_plans_sealed_not_admitted' and r['commands_released'] is True and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==inputs.moving.low.MODEL_TENSOR_SHA,'complete own CUDA closure')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==contract() and len(r['rows'])==2,'own fixed contract/roots')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}.{suffix}' for s in (0,1) for suffix in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact artifact closure')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'prepare binding')
    plain,eof=base.inflate((output/'events.jsonl.gz').read_bytes());require(eof and plain.endswith(b'\n'),'complete lossless journal')
    events=[base.decode(line) for line in plain.splitlines()];plans=[]
    require(all(set(e)=={'source_slot','task_id','event'} for e in events),'scoped journal only')
    require([e['source_slot'] for e in events]==sorted(e['source_slot'] for e in events),'no resumed root')
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        s=slot['source_slot'];stem=f'root_{s}'
        require(row['source_slot']==s and row['task_id']==slot['task_id'] and row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','root/identity/filenames')
        task=load_export(slot['task_export'],slot['task_sha256'])['gn_task'];result=read(output/row['result_file']);selected=choice(task,result,s)
        group=[e for e in events if e['source_slot']==s];require(all(e['task_id']==slot['task_id'] for e in group),'journal task binding')
        journal_check(task,result,[e['event'] for e in group]);plan=read(output/row['plan_file']);h=plan.pop('selection_sha256')
        require(plan==model_plan(slot,task,selected) and core.digest(plan)==h==row['selection_sha256'],'own selected witness seal')
        base.exact_counts(row['resources'],budget());plans.append(dict(plan,selection_sha256=h,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],contract()['total_budget'])
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final source/model/input TOCTOU')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'final artifact TOCTOU')
    require(sha(output/'report.json')==expected,'final report SHA')
    return dict(schema=NAME+'.consume',rows=plans,resources=r['resources'],inherited_encoding_once=contract()['inherited_encoding_once'],P1_admission=False,P2_admission=False,equal_total_compute_claim=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(status='verified_no_execution',contract=verify(a.prepared,a.prepare_sha)['contract'])
    elif a.command=='run':r=run(a.prepared,a.prepare_sha,a.output)
    else:r=consume(a.output,a.report_sha)
    print(canonical(r).decode().strip())


if __name__=='__main__':main()
