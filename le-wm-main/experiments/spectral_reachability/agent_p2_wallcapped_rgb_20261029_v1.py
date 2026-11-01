"""Own wallcapped four-plan x cp5/10/15 RGB companion. MAIN HOST CUDA only.

Twelve fixed slots, at most eleven real encoder/projector calls, no predict/AD/physics.
No old edges or receipt relabeling; checkpoints are exact own contexts[0:3].
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
from experiments.spectral_reachability import agent_p2_contextual_physical_rgb_encode_20261028_v1 as bridge
from experiments.spectral_reachability import agent_p2_wallcapped_physical_20261029_v1 as producer

NAME='agent_p2_wallcapped_rgb_20261029_v1'
AUTH='MAIN_WALLCAPPED11_RGB_HOST_CUDA_ONLY'
ROOT=producer.ROOT
require,sha,raw=producer.require,producer.sha,producer.raw
META=128*1024**2
visual=bridge.physical.saved.overlay


def read(path):return producer.read(path,META)


def contract():
    return dict(fixed_denominator=12,plan_denominator=4,pair_denominator=6,known_roots=2,fresh_roots=0,
        checkpoints=[5,10,15],order=[list(x) for x in producer.ORDER],
        mapping='own wallcapped contexts[j] vs actual step5*(j+1), never edges or earlier terminal',
        max_calls={k:11 if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        AD=0,new_physics=0,new_optimizer=0,training=False,Qfit=False,batch=1,no_dedup=True,no_padding=True,retries=0,
        missing='null; actual root1 joint cp15 unavailable after native stop14; no replacement',
        Q='original saved goal192, float64 identity192 mean squared distance; no new goal encode',
        native_hidden=[1,257,192],latent=192,CUDA_only=True,TF32=False,autocast=False,SDP='MATH',
        wall_seconds=600,output_cap_bytes=bridge.CAP,final_reserve_bytes=bridge.FINAL,disk_floor_bytes=bridge.FLOOR,
        metadata_cap_bytes=META,allocated_cap_bytes=3*1024**3,reserved_cap_bytes=4*1024**3,
        total_FLOPs=None,equal_FLOPs=False,equal_total_compute=False,selection=False,
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_pass=False,physical_release=False)


def fixed_rows(prep,strict,modelprep,output):
    require(strict['schema']==producer.NAME+'.consume' and strict['status']==producer.RECORDED
        and strict['fixed_denominator']==4 and strict['known_roots']==2 and strict['fresh_roots']==0
        and strict['errors']==[] and all(strict[k] is False for k in ('P1_admission','P2_admission','physical_release')),
        'actual own complete physical strict authority')
    producer.validate_execution(prep)
    require(len(strict['rows'])==len(prep['plans'])==4 and all(p is not None for p in prep['plans']),
        'four actual sealed exports, no subset or substitute')
    require(modelprep['schema']==producer.PRODUCER+'.prepare'
        and modelprep['expected_model_hash']==prep['expected_model_hash']
        and modelprep['slots']==prep['model_slots'],'exact original model input authority')
    rows=[]
    for i,(plan,actual) in enumerate(zip(prep['plans'],strict['rows'],strict=True)):
        key=producer.ORDER[i];mp=plan['planner_plan'];slot=prep['model_slots'][key[0]]
        producer.check_planner_plan(mp,slot,prep['arguments']['planner_report_sha'],modelprep['expected_model_hash'])
        require((actual['index'],actual['root_slot'],actual['condition'],actual['arm'])==(i,*key)
            and actual['plan']==plan and actual['task_id']==plan['task_id']
            and actual['selection_digest']==plan['selection_digest']
            and actual['status'] in ('horizon_completed','native_stopped') and actual['initial_exact'] is True,
            'whole actual model/native branch and warmup/root authentication')
        a=producer.raw15(plan['commands'])
        require(a.tobytes().hex()==mp['raw_float32_bits'] and plan['commands']==mp['commands'], 'literal model/native raw15')
        context=deepcopy(mp['root_context']);goal=bridge.latent(mp['goal'])
        for j,step in enumerate((5,10,15)):
            out=mp['contexts'][j]
            # Frozen normalization/shift checker is independent of any planner selection.
            producer.normalization.validate_context(out)
            producer.normalization.validate_advance(context,a[step-5:step],out)
            context=deepcopy(out);pred=bridge.latent(out['latent_history'][-1]);q=bridge.mse(pred,goal)
            if step==15:require(q==mp['model_Q'],'original terminal model Q exact')
            cp=[c for c in actual['checkpoints'] if c['step']==step]
            frames=[f for f in actual['trajectory'] if f['phase']=='plan' and f['step']==step and f['overlay_index']==0]
            responses=[v for v in actual['responses'] if v['step']==step]
            present=step<=actual['own_returned']
            require(len(cp)==len(frames)==len(responses)==int(present)
                and (step in actual['missing_checkpoints']) is (not present),'exact same-step presence/missing, no early terminal replacement')
            source=native=response=None
            if present:
                f=frames[0];response=responses[0]
                require(cp[0]=={k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
                    and response['state']==f['state'] and Path(f['png']).name==f['png']
                    and f.get('condition','off')=='off','same off checkpoint PNG/state/snapshot/response')
                source=deepcopy(cp[0]);source['path']=str(Path(output).resolve()/source['png'])
                native=producer.native.metrics(plan['native_goal'],response['state'])
                require(response['native_metrics']==native['native'] and response['reward']==native['native7_reward']
                    and response['native_success'] is response['terminated'] is native['native_success'], 'original native labels without refit')
            history=np.asarray(mp['warm_raw10']+plan['commands'][:step],np.float32)[-10:]
            rows.append(dict(index=len(rows),plan_index=i,root_slot=key[0],condition=key[1],arm=key[2],checkpoint=step,
                task_id=plan['task_id'],task_file=slot['task_file'],task_sha256=slot['task_sha256'],
                blind_path=slot['blind_path'],blind_sha256=slot['blind_sha256'],model_hash=mp['model_hash'],
                selection_digest=plan['selection_digest'],model_selection_digest=mp['selection_digest'],
                model_report_sha256=mp['report_sha256'],root_context=deepcopy(mp['root_context']),
                input_context=deepcopy(mp['root_context'] if j==0 else mp['contexts'][j-1]),output_context=deepcopy(out),
                full15=plan['commands'],raw_float32_bits=a.tobytes().hex(),prefix_actions=plan['commands'][:step],
                causal_raw10=history.tolist(),causal_raw10_bits=history.tobytes().hex(),
                predicted192=pred.tolist(),goal192=goal.tolist(),model_Q=q,native_goal=plan['native_goal'],
                physical_status=actual['status'],actual_returned=actual['own_returned'],native_metrics=native,response=response,
                source=source,status='not_started' if present else 'missing_physical_checkpoint',latent=None))
    require(len(rows)==12 and rows[11]['source'] is None and sum(r['source'] is not None for r in rows)<=11,
        'fixed twelve slots, actual root1 joint cp15 missing and max11 images')
    return rows


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(physical_output,physical_report_sha,physical_receipt,physical_receipt_sha,physical_source_sha):
    output=Path(producer.native.absolute(str(physical_output)));receipt=Path(producer.native.absolute(str(physical_receipt)))
    for h in (physical_report_sha,physical_receipt_sha,physical_source_sha):producer.native.actual_hash(h)
    require(sha(producer.__file__)==physical_source_sha,'actual physical source before consume')
    require(sha(output/'report.json')==physical_report_sha and sha(receipt)==physical_receipt_sha,'actual own physical report/strict SHA')
    strict=producer.consume(output,physical_report_sha)
    require(raw(strict)==raw(read(receipt)),'entire own physical.consume reconstructed, no status substitute')
    report=read(output/'report.json');b=report['execution_binding'];prep,_=producer.verify(b['prepared'],b['prepare_sha256'])
    modelreport_path=Path(prep['arguments']['planner_output'])/'report.json';mr=read(modelreport_path)
    require(prep['bindings'][str(modelreport_path)]==sha(modelreport_path)==prep['arguments']['planner_report_sha']
        and prep['bindings'][mr['prepared']]==sha(mr['prepared'])==mr['prepare_sha256'],'original model report/preparation closure')
    modelprep=read(mr['prepared']);rows=fixed_rows(prep,strict,modelprep,output)
    bindings=deepcopy(strict['bindings'])
    producer.native.merge(bindings,{str(receipt):physical_receipt_sha,str(output/'report.json'):physical_report_sha,
        str(Path(producer.__file__).resolve()):physical_source_sha,b['prepared']:b['prepare_sha256']})
    producer.native.merge(bindings,{str(f):sha(f) for f in bridge.source_files()+own_files()})
    producer.native.merge(bindings,bridge.rgb.MODELS)
    require(Path(modelprep['policy']).resolve()==bridge.rgb.POLICY.resolve(),'same original checkpoint/preprocess')
    shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json'))
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    bridge.verify_bindings(bindings);n=sum(r['source'] is not None for r in rows)
    return dict(schema=NAME+'.prepare',arguments=dict(physical_output=str(output),physical_report_sha=physical_report_sha,
        physical_receipt=str(receipt),physical_receipt_sha=physical_receipt_sha,physical_source_sha=physical_source_sha),
        contract=contract(),rows=rows,bindings=bindings,shapes=shapes,
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        policy=modelprep['policy'],expected_model_hash=modelprep['expected_model_hash'],
        inherited_physical_counts=strict['counts'],inherited_model_accounts=prep['planner_authority']['accounts'],
        inherited_model_wall_seconds=mr['wall_seconds'],physical_report_sha256=physical_report_sha,
        physical_receipt_sha256=physical_receipt_sha)


def write(path,value):
    path=Path(path).absolute();body=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(body)<=META
        and shutil.disk_usage(path.parent).free>=bridge.FLOOR+len(body),'own new128MiB metadata/floor')
    with path.open('xb') as f:require(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path))


def verify(path,expected):
    require(sha(path)==expected,'actual own prepare SHA');p=read(path)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'whole own prepare rebuilt');return p


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_') and shutil.disk_usage(Path(output).parent).free>=bridge.CAP+bridge.FLOOR,
            'own full-cap exclusive directory')
        super().__init__(output,budget);self.index=0
    def event(self,e):self.put(f'event_{self.index:04}.json',e);self.index+=1


def process(p,rows,store,encode,check=lambda:None):
    require(len(rows)==12 and rows==p['rows'] and rows[11]['source'] is None,'immutable twelve-row predispatch roster')
    ordinal=0
    for row in rows:
        if row['source'] is None:continue
        require(ordinal<11,'maximum eleven real images');check()
        scope=dict(row=row['index'],step=row['checkpoint'],phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        pixels=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pixels)
        z=encode(pixels,row,scope,ordinal);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)));ordinal+=1


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA before reads/imports')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);precision=None;errors=[];trace=guard=None;model=None;before=after=None;source_ok=False
    peaks=dict(allocated=0,reserved=0)
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats(device)
        def check():
            store.check();torch.cuda.synchronize(device)
            peaks['allocated']=int(torch.cuda.max_memory_allocated(device));peaks['reserved']=int(torch.cuda.max_memory_reserved(device))
            require(peaks['allocated']<=contract()['allocated_cap_bytes'] and peaks['reserved']<=contract()['reserved_cap_bytes'],'actual allocator limits')
        precision=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model');check()
        trace=bridge.Trace(model,torch,store,check,p['expected_calls'])
        guard=bridge.native.Ledger(3,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pixels,row,scope,ordinal):
            guard.image=ordinal
            with visual.traced_scope(trace,scope,row):
                x=p0._image_tensor(pixels,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pixels),'original pixel preprocessing exact')
                z=model.encode({'pixels':x})['emb'];require(tuple(z.shape)==(1,1,192),'native projected CLS192 only')
                return z[0,0].detach().cpu().numpy().copy()
        with trace,bridge.native.guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            process(p,rows,store,encode,check)
        check()
    except BaseException as e:errors.append(bridge.physical.kernel.error('encode',e))
    finally:
        if precision is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=precision
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(bridge.physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash']
            and bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='wallcapped_RGB_encoded_NOT_science' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,peak_allocated_bytes=peaks['allocated'],peak_reserved_bytes=peaks['reserved'],
            P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False)
        store.put('report.json',report,final=True)
    require(complete,'partial retained/no retry '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row,load):
    z=load(row['latent']) if row['latent'] is not None else None
    actual=None if z is None else bridge.mse(z,spec['goal192'])
    return dict(index=spec['index'],plan_index=spec['plan_index'],root_slot=spec['root_slot'],condition=spec['condition'],arm=spec['arm'],
        checkpoint=spec['checkpoint'],task_id=spec['task_id'],task_sha256=spec['task_sha256'],selection_digest=spec['selection_digest'],
        model_selection_digest=spec['model_selection_digest'],model_Q=spec['model_Q'],real_Q=actual,
        model_minus_real_Q=None if actual is None else spec['model_Q']-actual,
        prediction_vs_encoded_MSE=None if z is None else bridge.mse(z,spec['predicted192']),latent_missing=z is None,
        source=spec['source'],native_metrics=spec['native_metrics'],response=spec['response'],
        causal_raw10_bits=spec['causal_raw10_bits'],physical_status=spec['physical_status'],actual_returned=spec['actual_returned'],
        physical_benefit_claim=False,P2_mechanism=False)


def paired(rows):
    require(len(rows)==12,'twelve rows before six pairs');pairs=[]
    for root in (0,1):
        for step in (5,10,15):
            sides={r['arm']:r for r in rows if r['root_slot']==root and r['checkpoint']==step}
            require(set(sides)=={'joint','composition'},'both original arms including missing')
            c,j=sides['composition'],sides['joint']
            equal_task=all(c[k]==j[k] for k in ('task_id','task_sha256','condition'))
            require(equal_task,'same original physical/model task per pair')
            def delta(key):return None if c[key] is None or j[key] is None else c[key]-j[key]
            physical={k:None if c['native_metrics'] is None or j['native_metrics'] is None
                else c['native_metrics'][k]-j['native_metrics'][k] for k in
                ('joint_xy_error','angle_native_term','angle_wrapped_error','velocity_error','native7_distance','native7_reward')}
            pairs.append(dict(root_slot=root,checkpoint=step,direction='composition - joint',
                composition_index=c['index'],joint_index=j['index'],task_id=c['task_id'],
                model_Q=dict(composition=c['model_Q'],joint=j['model_Q'],delta=delta('model_Q')),
                real_Q=dict(composition=c['real_Q'],joint=j['real_Q'],delta=delta('real_Q')),
                physical_metrics=dict(composition=c['native_metrics'],joint=j['native_metrics'],delta=physical),
                native_success={a:None if x['native_metrics'] is None else x['native_metrics']['native_success'] for a,x in sides.items()},
                real_pair_missing=c['latent_missing'] or j['latent_missing'],physical_benefit_claim=False))
    return pairs


def validate(p,r,events,load):
    require(r['contract']==p['contract']==contract() and len(r['rows'])==12 and r['errors']==r['blocked']==[],'all twelve clean slots')
    modules=[];guards=[];want=[];ordinal=0
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'immutable own model/native task/context/goal/raw/frame')
        if spec['source'] is None:require(row==spec,'missing not padded or substituted');continue
        require(row['status']=='encoded','encode available once');pix=load(row['pixels'])
        bridge.same(pix,bridge.load_pixels(spec['source']),'actual source PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'independent original preprocessing')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=row['checkpoint'],phase='encode')
        mods=row['modules'];require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),'full scoped native encoder/projector')
        en,proj=mods
        visual.visual_bridge(pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z,[1,257,192])
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules))) and r['guards']==guards
        and events==want and r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()}
        and ordinal<=11,'entire native+guard event sequence/counts, reject unknowns')
    return [analyze(s,row,load) for s,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual own RGB report SHA')
    r=read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='wallcapped_RGB_encoded_NOT_science'
        and r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and all(r[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','physical_release'))
        and 0<=r['elapsed_seconds']<=600 and 0<=r['peak_allocated_bytes']<=contract()['allocated_cap_bytes']
        and r['peak_allocated_bytes']<=r['peak_reserved_bytes']<=contract()['reserved_cap_bytes'],'complete frozen bounded CUDA encoding')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'}
        and sum(f.stat().st_size for f in output.iterdir())<=bridge.CAP,'complete bounded artifact census')
    used={'binding.json','initial_roster.json','native_seal.json'}
    def blob(n):
        used.add(n);return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():
        require(hashlib.sha256(bridge.rgb.flat(output,n).read_bytes()).hexdigest()==h,'all actual artifact hashes')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'])
        and read(output/'initial_roster.json')==p['rows'],'whole original roster and execution binding')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and read(output/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native seal before any derived analysis')
    event_names=[f'event_{i:04}.json' for i in range(r['event_count'])];used.update(event_names)
    rows=validate(p,r,[read(output/n) for n in event_names],lambda e:bridge.rgb.load_array(e,blob))
    require(used==set(r['artifact_sha256']),'all and only referenced native artifacts, no resealed extras')
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'report unchanged after consume')
    return dict(schema=NAME+'.consume',status='all12_wallcapped_RGB_slots_verified_NOT_science',rows=rows,pairs=paired(rows),
        report_sha256=expected,counts=r['counts'],fixed_denominator=12,plan_denominator=4,pair_denominator=6,
        missing_slots=[x['index'] for x in rows if x['latent_missing']],real_pairs_available=sum(not x['real_pair_missing'] for x in paired(rows)),
        known_roots=2,fresh_roots=0,total_FLOPs=None,physical_report_sha256=p['physical_report_sha256'],
        physical_receipt_sha256=p['physical_receipt_sha256'],inherited_physical_counts=p['inherited_physical_counts'],
        inherited_model_accounts=p['inherited_model_accounts'],inherited_model_wall_seconds=p['inherited_model_wall_seconds'],
        new_neural_calls=r['counts'],consume_new_neural_calls=0,consume_new_physics=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,equal_total_compute=False,physical_release=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-output','physical-report-sha','physical-receipt','physical-receipt-sha','physical-source-sha','report-sha','receipt'):ap.add_argument('--'+k)
    ap.add_argument('--authorize-host-cuda',action='store_true');a=ap.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.physical_output,a.physical_report_sha,a.physical_receipt,a.physical_receipt_sha,a.physical_source_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(verified=True,expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN explicit HOST CUDA flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'own strict receipt path required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()
