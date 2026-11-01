"""Fixed four full15 plans x checkpoints5/10/15: actual PNG -> frozen native192.
Saved strict physical authority required; MAIN HOST CUDA only, no prediction/physics.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v1 as intervention

bridge=intervention.bridge
physical=bridge.physical
require,sha,raw=physical.require,physical.sha,physical.raw
ROOT=bridge.ROOT
from experiments.spectral_reachability import agent_p2_three_program_physical_20261029_v1 as producer
from experiments.spectral_reachability import agent_p2_three_program_20261028_v1 as planner
META=128*1024**2

def read(path):return producer.read(path,META)
NAME='agent_p2_three_program_rgb_20261029_v1'
AUTH='MAIN_THREE_PROGRAM12_RGB_HOST_CUDA_ONLY'


def contract():
    return dict(fixed_denominator=12,plan_denominator=4,known_roots=2,fresh_roots=0,checkpoints=[5,10,15],
        mapping='same selected raw15, each exact actual step and corresponding edge output; not terminal-Q at earlier steps',
        max_calls={k:12 if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        AD=0,new_physics=0,new_optimizer=0,training=False,Qfit=False,batch=1,no_dedup=True,no_padding=True,
        missing='null; no earlier terminal substitution or zero fill',native_hidden=[1,257,192],latent=192,
        Q='original identity192 float64 mean squared distance; original saved goal, no goal re-encode',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',total_FLOPs=None,equal_FLOPs=False,equal_total_compute=False,
        wall_seconds=600,output_cap_bytes=bridge.CAP,final_reserve_bytes=bridge.FINAL,disk_floor_bytes=bridge.FLOOR,
        metadata_cap_bytes=META,P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_pass=False,physical_release=False)


def fixed_rows(prep,strict,modelprep,output):
    require(strict['schema']==producer.NAME+'.consume' and strict['status']==producer.RECORDED
        and strict['fixed_denominator']==4 and strict['known_roots']==2 and strict['errors']==[]
        and all(strict[k] is False for k in ('P1_admission','P2_admission','physical_release')),
        'actual complete physical strict authority')
    producer.validate_execution(prep)
    require(len(strict['rows'])==len(prep['plans'])==4,'all4 plans, no subset')
    rows=[]
    for i,(plan,actual) in enumerate(zip(prep['plans'],strict['rows'],strict=True)):
        key=producer.ORDER[i];mp=plan['planner_plan']
        require((plan['root_slot'],plan['condition'],plan['arm'])==key and
            (actual['index'],actual['root_slot'],actual['condition'],actual['arm'])==(i,*key) and
            actual['plan']==plan and actual['selection_digest']==plan['selection_digest'] and
            actual['task_id']==plan['task_id'] and actual['status'] in ('horizon_completed','native_stopped'),
            'complete actual plan/root/arm/task/seal join')
        require(mp['model_hash']==modelprep['expected_model_hash'] and len(mp['edges'])==3
            and planner.digest({k:v for k,v in mp.items() if k not in ('selection_digest','planner_report_sha256','prepare_sha256')})==mp['selection_digest'],
            'original model plan seal and model identity')
        raw15=producer.raw15(plan['commands'])
        require(raw15.tobytes().hex()==mp['raw_float32_bits'] and mp['commands']==plan['commands'],'same raw15 model and native execution')
        goal=bridge.latent(mp['goal']);context=mp['root_context']
        for j,step in enumerate((5,10,15)):
            edge=mp['edges'][j]
            require(edge['index']==j and edge['input_context']==context and edge['commands']==plan['commands'][step-5:step]
                and edge['input_context_sha256']==planner.digest(context) and edge['goal_sha256']==planner.digest(mp['goal'])
                and edge['task_id']==mp['task_id'] and edge['source_report_sha256']==mp['source_report_sha256']
                and edge['witness_sha256']==planner.digest({k:v for k,v in edge.items() if k!='witness_sha256'}),
                'exact per-packet context/raw/goal/source witness')
            context=edge['output_context'];pred=bridge.latent(context['latent_history'][-1]);q=bridge.mse(pred,goal)
            if step==15:
                require(pred.tolist()==mp['predicted_terminal'] and q==mp['predicted_Q'],'terminal model-Q exact')
            cp=[c for c in actual['checkpoints'] if c['step']==step]
            frames=[f for f in actual['trajectory'] if f['phase']=='plan' and f['step']==step and f['overlay_index']==0]
            responses=[v for v in actual['responses'] if v['step']==step]
            present=step<=actual['own_returned']
            require(len(cp)==len(frames)==len(responses)==int(present) and
                (step in actual['missing_checkpoints']) is (not present),'exact physical step/missing denominator')
            source=None;native=None;response=None
            if present:
                f=frames[0];response=responses[0]
                require(cp[0]=={k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
                    and response['state']==f['state'] and Path(f['png']).name==f['png'],'same time canonical off RGB/native7')
                source=deepcopy(cp[0]);source['path']=str(Path(output).resolve()/source['png'])
                native=producer.native.metrics(plan['native_goal'],response['state'])
                require(response['native_metrics']==native['native'] and response['reward']==native['native7_reward']
                    and response['native_success'] is response['terminated'] is native['native_success'],'original native metrics/reward/done')
            history=np.asarray(mp['root_warm_raw10']+plan['commands'][:step],np.float32)[-10:]
            rows.append(dict(index=len(rows),plan_index=i,root_slot=key[0],condition=key[1],arm=key[2],task_id=plan['task_id'],
                checkpoint=step,selection_digest=plan['selection_digest'],model_selection_digest=mp['selection_digest'],
                model_hash=mp['model_hash'],task_file=mp['task_file'],task_sha256=mp['task_sha256'],
                blind_path=mp['blind_path'],blind_sha256=mp['blind_sha256'],normalization=mp['normalization'],
                source_report_sha256=mp['source_report_sha256'],edge=edge,
                full15=plan['commands'],raw_float32_bits=raw15.tobytes().hex(),
                prefix_actions=plan['commands'][:step],causal_raw10=history.tolist(),causal_raw10_bits=history.tobytes().hex(),
                predicted192=pred.tolist(),goal192=goal.tolist(),model_Q=q,
                physical_status=actual['status'],actual_returned=actual['own_returned'],native_metrics=native,response=response,
                source=source,status='not_started' if present else 'missing_physical_checkpoint',latent=None))
    return rows


def build(physical_output,physical_report_sha,physical_receipt,physical_receipt_sha,physical_source_sha):
    output=Path(producer.native.absolute(str(physical_output)));receipt=Path(producer.native.absolute(str(physical_receipt)))
    for h in (physical_report_sha,physical_receipt_sha,physical_source_sha):producer.native.actual_hash(h)
    require(sha(producer.__file__)==physical_source_sha,'actual physical source SHA')
    require(sha(output/'report.json')==physical_report_sha and sha(receipt)==physical_receipt_sha,'actual report/receipt SHA')
    strict=producer.consume(output,physical_report_sha)
    require(raw(strict)==raw(read(receipt)),'entire actual strict receipt exact; no envelope/status substitute')
    report=read(output/'report.json');b=report['execution_binding']
    prep,_=producer.verify(b['prepared'],b['prepare_sha256'])
    args=prep['arguments'];mr=read(Path(args['planner_output'])/'report.json')
    modelprep=planner.verify(mr['prepared'],mr['prepare_sha256'])
    rows=fixed_rows(prep,strict,modelprep,output)
    bindings=dict(strict['bindings'])
    producer.native.merge(bindings,{str(receipt):physical_receipt_sha,str(output/'report.json'):physical_report_sha,
        str(Path(producer.__file__).resolve()):physical_source_sha,b['prepared']:b['prepare_sha256']})
    files=bridge.source_files()+[Path(__file__).resolve(),Path(planner.__file__).resolve(),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),Path(intervention.__file__).resolve()]
    producer.native.merge(bindings,{str(f):sha(f) for f in files});producer.native.merge(bindings,bridge.rgb.MODELS)
    require(Path(modelprep['policy']).resolve()==bridge.rgb.POLICY.resolve(),'original checkpoint/preprocess pairing')
    shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json'))
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    bridge.verify_bindings(bindings);n=sum(row['source'] is not None for row in rows)
    return dict(schema=NAME+'.prepare',arguments=dict(physical_output=str(output),physical_report_sha=physical_report_sha,
        physical_receipt=str(receipt),physical_receipt_sha=physical_receipt_sha,physical_source_sha=physical_source_sha),
        contract=contract(),rows=rows,bindings=bindings,shapes=shapes,
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        policy=modelprep['policy'],expected_model_hash=modelprep['expected_model_hash'],
        inherited_physical_counts=strict['counts'],inherited_model_resources=mr['actual_resources'],
        inherited_model_wall_seconds=mr.get('elapsed_seconds'),physical_report_sha256=physical_report_sha,
        physical_receipt_sha256=physical_receipt_sha)


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(payload)<=META
        and shutil.disk_usage(path.parent).free>=bridge.FLOOR+len(payload),'own exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    require(sha(path)==expected,'external own prepare SHA');p=read(path)
    require(raw(p)==raw(build(**p['arguments'])),'entire prepare reconstruction');return p


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_'),'own new flat output');super().__init__(output,budget);self.index=0
    def event(self,e):self.put(f'event_{self.index:04}.json',e);self.index+=1


def process(p,rows,store,encode,check=lambda:None):
    require(len(rows)==12,'all12 source rows including missing')
    require(rows==p['rows'],'immutable complete predispatch roster')
    ordinal=0
    for row in rows:
        if row['source'] is None:continue
        check();scope=dict(row=row['index'],step=row['checkpoint'],phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        pixels=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pixels)
        z=encode(pixels,row,scope,ordinal);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)));ordinal+=1


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);precision=None;errors=[];trace=guard=None;model=None;before=after=None;source_ok=False
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        precision=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        trace=bridge.Trace(model,torch,store,store.check,p['expected_calls'])
        guard=bridge.native.Ledger(3,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pixels,row,scope,ordinal):
            guard.image=ordinal
            with physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pixels,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pixels),'original preprocessing exactly')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,bridge.native.guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            process(p,rows,store,encode,store.check)
    except BaseException as e:errors.append(physical.kernel.error('encode',e))
    finally:
        if precision is not None:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=precision
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        # native seal precedes any analysis of the new encoded outputs.
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='terminal_encoded_science_blocked' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,P1_admission=False,P2_admission=False,physical_release=False)
        store.put('report.json',report,final=True)
    require(complete,'partial retained/no retries '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row,load):
    z=load(row['latent']) if row['latent'] is not None else None
    actual=bridge.mse(z,spec['goal192']) if z is not None else None
    return dict(index=spec['index'],plan_index=spec['plan_index'],root_slot=spec['root_slot'],condition=spec['condition'],
        arm=spec['arm'],task_id=spec['task_id'],checkpoint=spec['checkpoint'],selection_digest=spec['selection_digest'],
        model_selection_digest=spec['model_selection_digest'],model_Q=spec['model_Q'],actual_encoded_Q=actual,
        model_minus_actual_Q=spec['model_Q']-actual if actual is not None else None,
        prediction_vs_encoded_MSE=bridge.mse(z,spec['predicted192']) if z is not None else None,
        latent_missing=z is None,source=spec['source'],native_metrics=spec['native_metrics'],response=spec['response'],
        causal_raw10_bits=spec['causal_raw10_bits'],physical_status=spec['physical_status'],
        physical_benefit_claim=False,P2_mechanism=False)



def validate(p,r,events,load):
    require(r['contract']==p['contract']==contract() and len(r['rows'])==12 and r['errors']==r['blocked']==[],'all12 clean checkpoint report')
    modules=[];guards=[];want=[];ordinal=0
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'original plan/new witness/goal/frame identity')
        if spec['source'] is None:require(row==spec,'missing not replaced or padded');continue
        require(row['status']=='encoded','available encoded once');pix=load(row['pixels'])
        bridge.same(pix,bridge.load_pixels(spec['source']),'actual new terminal PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'exact preprocessed input')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=row['checkpoint'],phase='encode')
        mods=row['modules'];require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),'full scoped actual encoder/projector')
        en,proj=mods
        physical.saved.overlay.visual_bridge(pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z,[1,257,192])
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules))) and r['guards']==guards
        and events==want and r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'entire native+forbidden-API guard chronology/counts, no unknown filtered')
    return [analyze(spec,row,load) for spec,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual encoder report SHA')
    r=read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='terminal_encoded_science_blocked' and
        r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False and 0<=r['elapsed_seconds']<=600,'complete frozen CUDA only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=bridge.CAP,'complete bounded output census')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'all artifact file SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        read(output/'initial_roster.json')==p['rows'],'original entire execution roster/binding')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and read(output/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native output sealed before saved analysis')
    rows=validate(p,r,[read(output/f'event_{i:04}.json') for i in range(r['event_count'])],
        lambda e:bridge.rgb.load_array(e,blob))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='all12_slots_RGB_model_native_join_verified_NOT_science',rows=rows,report_sha256=expected,
        counts=r['counts'],fixed_denominator=12,plan_denominator=4,known_roots=2,fresh_roots=0,total_FLOPs=None,
        physical_report_sha256=p['physical_report_sha256'],physical_receipt_sha256=p['physical_receipt_sha256'],
        inherited_physical_counts=p['inherited_physical_counts'],inherited_model_resources=p['inherited_model_resources'],
        inherited_model_wall_seconds=p['inherited_model_wall_seconds'],consume_new_neural_calls=0,consume_new_physics=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,equal_total_compute=False,physical_release=False)



def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-output','physical-report-sha','physical-receipt','physical-receipt-sha','physical-source-sha','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.physical_output,a.physical_report_sha,a.physical_receipt,a.physical_receipt_sha,a.physical_source_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN explicit CUDA flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'own saved receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()

