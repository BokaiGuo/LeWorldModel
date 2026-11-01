"""Fixed4 hybrid held-old2/actual-z5 suffix diagnostic; MAIN HOST CUDA only.

Four native predictions, zero encoder/AD/physics. Not true full RGB feedback.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_nonlinear_suffix_midpoint_rgb_20261028_v1 as mid

NAME='agent_nonlinear_suffix_history_control_cuda_20261028_v1'
AUTH='MAIN_FIXED4_HYBRID_HELD_OLD2_SUFFIX_HOST_CUDA_ONLY'
ROOT=mid.ROOT
suffix,bridge,producer=mid.suffix,mid.bridge,mid.producer
core,signal,transport=suffix.core,suffix.signal,suffix.transport
require,sha,raw,read=mid.require,mid.sha,mid.raw,mid.read
digest=core.digest
CAP,FINAL,FLOOR,WALL=64*1024**2,8*1024**2,200*1024**2,600
FLAGS=dict(P1_admission=False,freshness=False,scientific_admission=False,physical_release=False)
FROZEN=((mid,'b1371a0de4d805bac1b4b64cfb625b7d327a3772d61d68c092804430f94a7ece'),
    (suffix,'0a3e475af9271a0769a3829b4e714fa4e2d6f00eaed38ec8c3ea5d7b9a163fbc'),
    (transport,'3f166420ddbe6b25b04458427747a7b4dbde8397d8e4707242eef26be961f939'),*mid.FROZEN)
LABEL='hybrid_held_old2_suffix'


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=4,known_roots=2,fresh_roots=0,
        order=[list(k) for k in producer.ORDER],intervention='hybrid old-root last2 + new actual z5; NOT full RGB history or deploy recompile',
        latent=[1,3,192],action_history=[1,2,10],outgoing=[1,3,10],native_prediction=[1,3,192],
        suffix='original exact rawfloat32 plan6..10; no action selection/clip/tanh',
        incoming='real warmup6..10 + plan1..5, same saved observed-history inputs',
        goal='unchanged saved goal192/identity Q and original normalizer',
        compare='hybrid minus entire-observed-history prediction, same actual terminal and goal',
        primary='endpoint MSE to actual terminal192',secondary='absolute predicted goalQ error',
        history_drift='retain per-role old2 MSE/max_abs and endpoint response; no tolerance/causal-null shortcut',
        resource_caps=suffix.caps(4,0),per_row_caps=suffix.caps(1,0),
        new_encoder=0,new_projector=0,new_AD=0,new_physics=0,new_training=0,new_goal_encode=0,
        precision=mid.terminal.precision_contract(),CUDA_only=True,CPU_fallback=False,retries=0,
        output_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,wall_seconds=WALL,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,FLOPs=None,equal_FLOPs=False,**FLAGS)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']


def material(prepared,report,accepted,compiled,load):
    require(prepared['schema']==mid.NAME+'.prepare' and prepared['contract']==mid.contract() and
        report['schema']==mid.NAME+'.report' and report['status']=='actual_midpoint_suffix4_predicted_not_admitted' and
        accepted['schema']==mid.NAME+'.consume' and accepted['status']=='actual_midpoint_suffix4_verified_not_admitted' and
        accepted['fixed_denominator']==4 and accepted['known_roots']==2 and accepted['fresh_roots']==0 and
        compiled['schema']==suffix.NAME+'.prepare' and len(prepared['rows'])==len(report['rows'])==
        len(accepted['rows'])==len(compiled['slots'])==4,'actual full4 midpoint ABI, no partial/relabel')
    require(all(accepted[k] is False for k in ('P1_admission','scientific_admission','freshness')) and
        report['errors']==report['blocked']==[],'clean upstream engineering, never science admission')
    rows=[]
    for i,(spec,saved,end,slot) in enumerate(zip(prepared['rows'],report['rows'],accepted['rows'],compiled['slots'],strict=True)):
        task=core.validate_task(slot['task']);key=producer.ORDER[i]
        require(all(r['index']==i and (r['root_slot'],r['condition'])==key for r in (spec,saved,end,slot)) and
            spec['eligible'] is saved['eligible'] is end['eligible'] is True and saved['status']=='predicted' and
            spec['original_root_context']==task['root_context'] and spec['model_task_binding']==task['binding'] and
            spec['goal192']==task['goal'] and np.array_equal(task['projection'],np.eye(192)) and
            task['binding']['model_sha256']==prepared['expected_model_hash']==compiled['expected_model_hash'] and
            all(spec[k]==saved[k]==end[k] for k in ('task_sha256','selection_digest','raw_float32_sha256','goal_sha256','row_binding_sha256')),
            'all4 original goal/Q/model/root-context/task/raw identity')
        full=load(saved['actual_full3']);observed=load(saved['prediction'])
        require(full.dtype==observed.dtype==np.float32 and full.shape==observed.shape==(1,3,192),
            'exact saved native full3 context and prediction')
        bridge.same(full,load(saved['feedback']['inputs']['context']),'same actual encoded context')
        b=mid.tokens(spec)
        for k,v in b.items():bridge.same(load(saved['feedback']['inputs'][k]),v,'real causal saved '+k)
        require(end['conditioned_final192']==observed[0,-1].tolist() and
            end['actual_final192']==spec['actual_terminal192'] and spec['actual_terminal192'] is not None and
            end['actual_final_Q']==spec['actual_terminal_Q']==bridge.mse(spec['actual_terminal192'],task['goal']) and
            spec['action_mean']==task['action_mean'] and spec['action_std']==task['action_std'] and
            np.asarray(task['root_context']['action_history'],np.float32)[1].tobytes()==b['incoming'][0,0].tobytes(),
            'same observed final/actual target/goal and incoming action history')
        old=core.finite(task['root_context']['latent_history'],(3,192),True).astype(np.float32)
        context=deepcopy(task['root_context'])
        context.update(latent_history=np.concatenate((old[-2:],full[0,-1:])).tolist(),action_history=b['incoming'][0].tolist())
        core.context(context)
        row=dict(index=i,root_slot=key[0],condition=key[1],task=deepcopy(task),hybrid_context=context,
            full10=deepcopy(spec['full10']),suffix5=deepcopy(spec['full10'][5:]),
            original_root_context=deepcopy(task['root_context']),observed_full3=full[0].tolist(),
            observed_prediction3=observed[0].tolist(),actual_terminal192=deepcopy(spec['actual_terminal192']),
            source_identity={k:deepcopy(spec[k]) for k in ('task_sha256','selection_digest','raw_float32_sha256',
                'goal_sha256','row_binding_sha256','frames')},
            source_arrays=dict(actual_full3=deepcopy(saved['actual_full3']),prediction=deepcopy(saved['prediction']),
                feedback_inputs=deepcopy(saved['feedback']['inputs'])))
        row['request_sha256']=digest(row);rows.append(row)
    return rows


def build(midpoint_output,midpoint_report_sha,midpoint_receipt,midpoint_receipt_sha):
    args=dict(midpoint_output=midpoint_output,midpoint_report_sha=midpoint_report_sha,
        midpoint_receipt=midpoint_receipt,midpoint_receipt_sha=midpoint_receipt_sha)
    for k,v in args.items():
        if k.endswith('_sha'):producer.mixed.actual_hash(v)
        else:producer.mixed.absolute(v)
    for module,h in FROZEN:require(sha(module.__file__)==h,'frozen reviewed native/midpoint source')
    out=Path(midpoint_output)
    require(sha(out/'report.json')==midpoint_report_sha and sha(midpoint_receipt)==midpoint_receipt_sha,'actual external midpoint seals')
    accepted=mid.consume(out,midpoint_report_sha)
    require(raw(accepted)==raw(read(midpoint_receipt,producer.PREP_CAP)),'entire midpoint strict receipt reconstructed')
    r=read(out/'report.json',producer.PREP_CAP);p=read(r['prepared'],producer.PREP_CAP)
    require(sha(r['prepared'])==r['prepare_sha256'],'actual midpoint preparation')
    a=p['physical_authority'];path=a['producer_prepared'];compiled=read(path,16*1024**2)
    require(sha(path)==p['bindings'][path]==a['producer_prepare_sha256'],'same original suffix compiled task')
    rows=material(p,r,accepted,compiled,lambda ref:mid.load_saved_array(out,ref))
    bindings=deepcopy(p['bindings'])
    producer.mixed.merge(bindings,{str(out/'report.json'):midpoint_report_sha,str(Path(midpoint_receipt)):midpoint_receipt_sha,
        r['prepared']:r['prepare_sha256'],**{str(out/n):h for n,h in r['artifact_sha256'].items()}})
    producer.mixed.merge(bindings,{str(f):sha(f) for f in own_files()})
    bridge.verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=args,rows=rows,bindings=bindings,
        policy=p['policy'],expected_model_hash=p['expected_model_hash'],prediction_abi=p['prediction_abi'],
        inherited_resources=dict(midpoint=r['counts'],previous=p['inherited_resources']),
        upstream_analysis_protocol_sha256=mid.PROTOCOL_SHA)


def write(path,value):
    path=Path(producer.mixed.absolute(str(path)));data=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and
        not path.is_symlink() and len(data)<=FINAL and shutil.disk_usage(path.parent).free>=FLOOR+len(data),
        'exclusive bounded metadata, no overwrite')
    with path.open('xb') as f:require(f.write(data)==len(data),'full metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    producer.mixed.actual_hash(expected);require(sha(path)==expected,'external own prepare SHA')
    p=read(path,FINAL)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'whole hybrid-source/request rebuild')
    return p


class Store(transport.Store):
    def __init__(self,path,started):
        self.path=Path(producer.mixed.absolute(str(path)));self.started=started;self.finalizing=False;self.guard_index=0
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and not self.path.exists() and
            self.path.resolve()==self.path and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new exclusive bounded output')
        self.path.mkdir();signal.fsync_dir(self.path.parent)
    def check(self,size=4096):
        require(self.finalizing or time.monotonic()-self.started<WALL,'whole-stage wall cap')
        used=sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size<=CAP-(0 if self.finalizing else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+size,'output/finalreserve/diskfloor')
    def namespace(self,name):
        require(name in {f'row{i}' for i in range(4)},'fixed four native namespaces')
        path=self.path/name;require(not path.exists(),'exclusive namespace')
        path.mkdir();signal.fsync_dir(self.path);return path
    def put(self,name,value,final=False):return self.write(name,raw(value),final=final)
    def event(self,value):
        self.put(f'guard_{self.guard_index:04}.json',value);self.guard_index+=1


def roster():
    return [dict(index=i,root_slot=k[0],condition=k[1],status='not_started',result=None,resource_receipt=None,
        journal_file=None,journal_sha256=None,journal_tail=None) for i,k in enumerate(producer.ORDER)]


def dispatch(p,store,rows,check,factory):
    require(rows==roster() and len(p['rows'])==4,'fixed all4 initial denominator')
    for spec,row in zip(p['rows'],rows,strict=True):
        check();i=row['index'];row['status']='attempted'
        journal=transport.NativeJournal(store,f'row{i}',check);ledger=core.Ledger(suffix.caps(1,0),sink=journal.event)
        try:
            backend=factory(spec['task'],journal,ledger)
            with backend.observe():row['result']=backend.step(deepcopy(spec['hybrid_context']),deepcopy(spec['suffix5']),LABEL)
            require(ledger.attempted==ledger.completed==suffix.caps(1,0),'actual one native prediction/row')
            core.checked_path(spec['task'],spec['hybrid_context'],spec['suffix5'],[row['result']]);row['status']='predicted'
        finally:
            if row['status']!='predicted':row['status']='failed_partial'
            row['resource_receipt']=ledger.receipt();journal.close()
            row.update(journal_file=f'row{i}/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)
            store.put(f'row{i}_result.json',row)


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN HOST CUDA required before IO/imports')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output,started);rows=roster();errors=[];blocked=[]
    model=before=after=host=None;flags=None;source_ok=False;precision_before=precision_after=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model)
        require(before==p['expected_model_hash'],'original frozen checkpoint')
        for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'actual installed class source closure')
        def check():
            store.check();mid.terminal.checked_precision(torch);require(not blocked,'AD guard fail-stop')
            require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and
                torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'CUDA memory caps')
        with mid.no_AD(torch,blocked,store),torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            precision_before=mid.terminal.checked_precision(torch)
            dispatch(p,store,rows,check,lambda task,journal,ledger:suffix.Backend(model,device,task,journal,ledger,check))
            torch.cuda.synchronize(device);check();precision_after=mid.terminal.checked_precision(torch)
    except BaseException as exc:errors.append(dict(stage='hybrid_suffix4',error=repr(exc)))
    finally:
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');source_ok=True
            require(time.monotonic()-started<WALL,'whole-stage wall cap including final source check')
        except BaseException as exc:errors.append(dict(stage='final_closure',error=repr(exc)))
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        good=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            precision_before==precision_after==mid.terminal.precision_contract() and all(row['status']=='predicted' for row in rows))
        seal=dict(rows=rows,errors=errors,blocked=blocked,precision_before=precision_before,precision_after=precision_after)
        store.finalizing=True;h=store.put('native_seal.json',seal,final=True)
        artifacts={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()}
        r=dict(schema=NAME+'.report',contract=contract(),status='hybrid_suffix4_predicted_not_admitted' if good else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,**seal,host=host,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,native_seal_sha256=h,artifact_sha256=artifacts,
            elapsed_seconds=time.monotonic()-started,**FLAGS)
        store.put('report.json',r,final=True)
    require(good,'partial evidence retained, no retry '+repr(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row):
    hybrid=np.asarray(row['result']['latent_history'][-1],np.float32)
    observed=np.asarray(spec['observed_prediction3'][-1],np.float32)
    actual=spec['actual_terminal192'];goal=spec['task']['goal']
    qh,qo,qa=(bridge.mse(z,goal) for z in (hybrid,observed,actual))
    old=np.asarray(spec['original_root_context']['latent_history'],np.float32)[-2:]
    new=np.asarray(spec['observed_full3'],np.float32)[:2]
    response=hybrid.astype(float)-observed.astype(float)
    error=observed.astype(float)-np.asarray(actual,float)
    return dict(index=spec['index'],root_slot=spec['root_slot'],condition=spec['condition'],request_sha256=spec['request_sha256'],
        hybrid_final192=hybrid.tolist(),entire_observed_final192=observed.tolist(),actual_final192=actual,
        hybrid_endpoint_MSE=bridge.mse(hybrid,actual),entire_observed_endpoint_MSE=bridge.mse(observed,actual),
        endpoint_MSE_change=bridge.mse(hybrid,actual)-bridge.mse(observed,actual),
        hybrid_goalQ=qh,entire_observed_goalQ=qo,actual_goalQ=qa,
        hybrid_goalQ_absolute_error=abs(qh-qa),entire_observed_goalQ_absolute_error=abs(qo-qa),
        goalQ_absolute_error_change=abs(qh-qa)-abs(qo-qa),terminal_response= response.tolist(),
        terminal_response_MSE=float(np.mean(response**2)),terminal_response_max_abs=float(np.max(abs(response))),
        endpoint_error_cross_term=float(2*np.mean(error*response)),
        old2_drift=[dict(role=role,MSE=bridge.mse(a,b),max_abs=float(np.max(abs(a.astype(float)-b.astype(float)))))
            for role,a,b in zip(('warmup5','plan0'),old,new,strict=True)],
        diagnosis='explicit hybrid held-old2; not actual complete RGB history or deploy recompile',
        drift_cause_attributed=False,physical_benefit_claim=False,**FLAGS)


def validate(p,r,journals):
    require(p['schema']==NAME+'.prepare' and r['schema']==NAME+'.report' and p['contract']==r['contract']==contract() and
        len(p['rows'])==len(r['rows'])==len(journals)==4 and r['errors']==r['blocked']==[] and
        r['precision_before']==r['precision_after']==mid.terminal.precision_contract(),'full4 own schema/precision')
    totals=dict.fromkeys(core.AXES,0);out=[]
    allowed={'forward_started','forward_finished','program_packet','program_boundary','resource_attempt','resource_return','io_input','io_return'}
    for i,(spec,row,events) in enumerate(zip(p['rows'],r['rows'],journals,strict=True)):
        require(row['index']==spec['index']==i and (row['root_slot'],row['condition'])==producer.ORDER[i] and
            row['status']=='predicted' and spec['request_sha256']==digest({k:v for k,v in spec.items() if k!='request_sha256'}),
            'entire fixed request/row identity')
        require(all(e['kind'] in allowed for e in events),'all and only actual prediction events, no filtering')
        starts,ends,linked=suffix.audit_native(spec['task'],row['resource_receipt'],events,[(LABEL,spec['hybrid_context'],spec['suffix5'])])
        require(events[0]==starts[0] and events[-1]==ends[0] and ends[0]['context']==row['result'],'exact native endpoint and journal bookends')
        shapes={'action_encoder':([[1,3,10]],[1,3,192]),'predict':([[1,3,192],[1,3,192]],[1,3,192]),
            'predictor':([[1,3,192],[1,3,192]],[1,3,192]),'pred_proj':([[3,192]],[3,192])}
        for op in row['resource_receipt']['events']:
            ins,outs=linked[op['index']];wanted_in,wanted_out=shapes[op['resource']]
            require([b['shape'] for b in ins['arrays'].values()]==wanted_in and
                [b['shape'] for b in outs['arrays'].values()]==[wanted_out],'complete original native module shapes')
        core.checked_path(spec['task'],spec['hybrid_context'],spec['suffix5'],[row['result']])
        for k,v in row['resource_receipt']['completed'].items():totals[k]+=v
        out.append(analyze(spec,row))
    require(totals==suffix.caps(4,0),'full actual4 resource axes, no padding');return out,totals


def consume(output,expected):
    out=Path(producer.mixed.absolute(str(output)));producer.mixed.actual_hash(expected)
    require(out.resolve()==out and not out.is_symlink() and sha(out/'report.json')==expected,'actual external report SHA/path')
    r=read(out/'report.json',FINAL);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['status']=='hybrid_suffix4_predicted_not_admitted' and all(r[k] is False for k in FLAGS) and
        r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        0<=r['elapsed_seconds']<WALL,'unchanged original model/source/limits, science closed')
    actual={str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}
    wanted={'binding.json','initial_roster.json','native_seal.json'}|{f'row{i}_result.json' for i in range(4)}|{f'row{i}/events.jsonl' for i in range(4)}
    require(actual==set(r['artifact_sha256'])|{'report.json'} and set(r['artifact_sha256'])==wanted and
        {str(d.relative_to(out)) for d in out.rglob('*') if d.is_dir()}=={f'row{i}' for i in range(4)} and
        sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<=CAP,'exact full output census/cap')
    for name,h in r['artifact_sha256'].items():
        f=out/name;require(f.resolve()==f and not f.is_symlink() and sha(f)==h,'all artifact SHA/path')
    require(read(out/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        read(out/'initial_roster.json')==roster(),'exact own prepare/initial denominator binding')
    require(sha(out/'native_seal.json')==r['native_seal_sha256'] and read(out/'native_seal.json',FINAL)==
        {k:r[k] for k in ('rows','errors','blocked','precision_before','precision_after')},'native seal matches entire actual report')
    journals=[]
    for i,row in enumerate(r['rows']):
        require(row['journal_file']==f'row{i}/events.jsonl' and read(out/f'row{i}_result.json',FINAL)==row,'own durable row and fixed journal path')
        journals.append(suffix.load_journal(out,row))
    rows,counts=validate(p,r,journals)
    bridge.verify_bindings(p['bindings']);require(sha(out/'report.json')==expected and sha(r['prepared'])==r['prepare_sha256'],'final source/report/prepare unchanged')
    return dict(schema=NAME+'.consume',status='hybrid_suffix4_verified_not_admitted',rows=rows,counts=counts,
        report_sha256=expected,midpoint_report_sha256=p['arguments']['midpoint_report_sha'],
        inherited_resources=p['inherited_resources'],fixed_denominator=4,known_roots=2,fresh_roots=0,
        consume_new_neural_calls=0,consume_new_physics=0,FLOPs=None,**FLAGS)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','midpoint-output','midpoint-report-sha','midpoint-receipt','midpoint-receipt-sha','report-sha','receipt'):
        parser.add_argument('--'+k)
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build(a.midpoint_output,a.midpoint_report_sha,a.midpoint_receipt,a.midpoint_receipt_sha))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution')
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN HOST CUDA flag required');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:require(a.receipt,'new own receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
