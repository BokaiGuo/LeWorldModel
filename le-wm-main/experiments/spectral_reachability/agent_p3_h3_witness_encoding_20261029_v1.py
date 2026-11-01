"""Four fixed H3 same-action plans x cp5/10/15/20, MAIN HOST CUDA only.

Own physical authority and encoder consumer; original frozen native192/Q.
No predictions, training, physics or scientific admission in this stage.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import inspect
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
from experiments.spectral_reachability import agent_p3_h3_witness_physical_20261029_v1 as producer
from experiments.spectral_reachability import agent_p3_shared_panel_20261029_v1 as planner
META=128*1024**2
CP=(5,10,15,20)
PRODUCER_SHA='f06772639f5477d9bd9bbe962a7b37ece34878226c30b0b43a0efa3a30eb7fe0'

def read(path):return producer.read(path,META)
NAME='agent_p3_h3_witness_encoding_20261029_v1'
AUTH='MAIN_H3_WITNESS16_RGB_HOST_CUDA_ONLY'


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=16,plan_denominator=4,known_roots=2,fresh_roots=0,
        checkpoints=list(CP),raw_control_steps=20,original_prefix=5,H3_suffix=15,
        mapping='cp5 observed anchor only; b5 H3 path[0/1/2] versus actual cp10/15/20 same actions',
        max_calls={k:16 if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        AD=0,new_physics=0,new_optimizer=0,training=False,Qfit=False,batch=1,no_dedup=True,no_padding=True,
        missing='null, no earlier terminal substitution, prediction is not an actual label',
        native_hidden=[1,257,192],latent=192,encoding='same native model.encode RGB224 -> CLS -> projector; no latent renormalization',
        Q='original saved goal192 identity; float64 mean squared distance; no goal encode',
        raw_normalizer='original task float32 subtract/divide once; context roll from real raw10 only',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',total_FLOPs=None,equal_FLOPs=False,equal_total_compute=False,
        wall_seconds=600,output_cap_bytes=bridge.CAP,final_reserve_bytes=bridge.FINAL,disk_floor_bytes=bridge.FLOOR,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,metadata_cap_bytes=META,
        P1_admission=False,P3_admission=False,C3=False,scientific_pass=False,physical_release=False)


def fixed_rows(prep,strict,modelprep,output):
    """Pure fully authenticated physical/model join; preserves all16 slots."""
    require(strict['schema']==producer.NAME+'.consume' and strict['status']==producer.RECORDED
        and strict['fixed_denominator']==4 and strict['known_roots']==2 and strict['errors']==[]
        and all(strict[k] is False for k in ('P1_admission','P3_admission','C3','physical_release')),
        'complete own physical strict authority; not an old schema')
    producer.validate_execution(prep)
    require(len(strict['rows'])==len(strict['plans'])==4 and strict['plans']==prep['plans']
        and modelprep['schema']==planner.NAME+'.prepare' and modelprep['contract']==planner.contract(),
        'all4 fixed plans, original actual shared-panel model preparation')
    rows=[]
    for i,(plan,actual) in enumerate(zip(prep['plans'],strict['rows'],strict=True)):
        key=producer.ORDER[i];w=plan['source_witness'];task=w['task']
        require((actual['index'],actual['root_slot'],actual['condition'],actual['arm'])==(i,*key)
            and actual['plan']==plan and actual['selection_digest']==plan['selection_digest']
            and actual['task_id']==plan['task_id'] and actual['status'] in ('horizon_completed','native_stopped'),
            'actual fixed root/bank/plan/raw20 identity')
        require(task==modelprep['tasks'][key[0]] and task['binding']['model_sha256']==modelprep['expected_model_hash']
            and w['prefix']==planner.bridge.prefix_binding(modelprep['layers'][key[0]]),
            'original model/goal/Q/normalizer/prefix authority')
        planner.validate_request(w['request'],task);planner.core.validate_task(task)
        require(np.array_equal(task['projection'],np.eye(192)),'original identity Q, not new fitting')
        commands=producer.raw20(plan['commands']);suffix=producer.raw15(w['winner']['actions'])
        require(commands[5:].tobytes()==suffix.tobytes() and w['winner']==w['result']['panels'][1]['winner']
            and w['result']['selected_horizon']==3,'selected original b5 full H3, not later MPC')
        anchor=w['request']['task']['root_context'];path=w['winner']['whole']
        planner.core.checked_path(task,anchor,suffix.tolist(),path)
        goal=bridge.latent(task['goal'])
        require(bridge.mse(path[-1]['latent_history'][-1],goal)==w['winner']['cost'],
            'original H3 terminal cost exactly identity meanQ')
        warm=producer.moving.warmup_commands()
        cp5raw=np.concatenate((warm,commands[:5]))[-10:]
        require(cp5raw.tobytes()==producer.literal(w['request']['raw10'],10).tobytes(),
            'original actual b5 raw10 is warmup tail plus fixed prefix5')
        for j,step in enumerate(CP):
            cp=[c for c in actual['checkpoints'] if c['step']==step]
            frames=[f for f in actual['trajectory'] if f['phase']=='plan' and f['step']==step and f['overlay_index']==0]
            responses=[v for v in actual['responses'] if v['step']==step]
            present=step<=actual['own_returned'];source=None;native=None;response=None
            require(len(cp)==len(frames)==len(responses)==int(present)
                and (step in actual['missing_checkpoints']) is (not present),
                'exact checkpoint/missing slots, no padding or forecast labels')
            if present:
                f=frames[0];response=responses[0]
                require(cp[0]=={k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
                    and f['condition']=='off' and f['state']==f['snapshot']['state']
                    and response['state']==f['state'] and Path(f['png']).name==f['png'],
                    'canonical same-time off RGB/native7 checkpoint')
                if step==5:
                    require(all(producer.raw(cp[0][k])==producer.raw(w['actual_cp5'][k]) for k in
                        ('state','snapshot','pixel_sha256')),'same original observed cp5, not another branch')
                source=deepcopy(cp[0]);source['path']=str(Path(output).resolve()/source['png'])
                native=producer.native.metrics(plan['native_goal'],response['state'])
                require(response['native_metrics']==native['native'] and response['reward']==native['native7_reward']
                    and response['native_success'] is response['terminated'] is native['native_success'],
                    'original full native7 reward/goal/success')
            history=np.concatenate((warm,commands[:step]))[-10:]
            pred=None if j==0 else bridge.latent(path[j-1]['latent_history'][-1]).tolist()
            rows.append(dict(index=len(rows),plan_index=i,root_slot=key[0],condition=key[1],arm=key[2],
                task_id=plan['task_id'],checkpoint=step,selection_digest=plan['selection_digest'],
                witness_sha256=w['witness_sha256'],model_hash=modelprep['expected_model_hash'],
                task_binding=deepcopy(task['binding']),goal192=goal.tolist(),predicted192=pred,
                observed_anchor192=bridge.latent(anchor['latent_history'][-1]).tolist() if j==0 else None,
                model_Q=None if pred is None else bridge.mse(pred,goal),
                full20=plan['commands'],raw_float32_bits=commands.tobytes().hex(),
                causal_raw10=history.tolist(),causal_raw10_bits=history.tobytes().hex(),
                normalized_action_history=planner.moving.normalized(task,history),
                physical_status=actual['status'],actual_returned=actual['own_returned'],
                native_metrics=native,response=response,source=source,
                status='not_started' if present else 'missing_physical_checkpoint',latent=None))
    return rows


def build(physical_output,physical_report_sha,physical_receipt,physical_receipt_sha,physical_source_sha):
    output=Path(producer.native.absolute(str(physical_output)));receipt=Path(producer.native.absolute(str(physical_receipt)))
    for h in (physical_report_sha,physical_receipt_sha,physical_source_sha):producer.native.actual_hash(h)
    require(sha(producer.__file__)==physical_source_sha==PRODUCER_SHA,'frozen own physical source before full strict')
    require(sha(output/'report.json')==physical_report_sha and sha(receipt)==physical_receipt_sha,'actual report/receipt SHA')
    strict=producer.consume(output,physical_report_sha)
    require(raw(strict)==raw(read(receipt)),'entire real physical strict receipt, not status-only')
    report=read(output/'report.json');b=report['execution_binding']
    # producer.consume already fully rebuilt this prepare; authenticated read avoids another duplicate closure.
    require(sha(b['prepared'])==b['prepare_sha256']==strict['bindings'][b['prepared']],'strict-bound original physical prepare')
    prep=read(b['prepared']);args=prep['arguments'];mrpath=Path(args['planner_output'])/'report.json'
    require(sha(mrpath)==producer.REPORT_SHA==strict['bindings'][str(mrpath)],'actual original shared-panel report')
    mr=producer.read(mrpath,planner.contract()['final_reserve_bytes'])
    require(sha(mr['prepared'])==mr['prepare_sha256']==producer.PREPARE_SHA==strict['bindings'][mr['prepared']],
        'strict-bound original shared-panel prepare/model')
    modelprep=read(mr['prepared']);rows=fixed_rows(prep,strict,modelprep,output)
    bindings=dict(strict['bindings'])
    producer.native.merge(bindings,{str(receipt):physical_receipt_sha,str(output/'report.json'):physical_report_sha,
        str(Path(producer.__file__).resolve()):physical_source_sha,b['prepared']:b['prepare_sha256']})
    files=bridge.source_files()+own_files()+[Path(planner.__file__).resolve(),Path(intervention.__file__).resolve()]
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
        plans=deepcopy(prep['plans']),inherited_physical_counts=strict['counts'],
        inherited_model_resources=mr['per_episode_neural'],inherited_model_wall_seconds=mr['wall_seconds'],
        physical_report_sha256=physical_report_sha,physical_receipt_sha256=physical_receipt_sha,
        upstream_fullstrict_reconstructed_here=True)


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
    require(len(rows)==16,'all16 source rows including missing')
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


def precision_state(torch):
    return dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
        autocast=torch.is_autocast_enabled('cuda'),math=torch.backends.cuda.math_sdp_enabled(),
        flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())


def model_inventory(model,bindings):
    result=[]
    for module in model.modules():
        path=str(Path(inspect.getfile(type(module))).resolve())
        require(path in bindings and sha(path)==bindings[path],'presealed installed model class source')
        result.append(dict(type=type(module).__module__+'.'+type(module).__qualname__,path=path,sha256=bindings[path]))
    require(result,'complete module inventory');return result


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);precision=None;errors=[];trace=guard=None;model=None;before=after=None;source_ok=False
    numeric_before=numeric_after=None;classes_before=classes_after=None;peak=None
    def check():
        store.check()
        require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and
            torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'actual CUDA memory caps')
        require(precision_state(torch)==planner.precision_contract(),'actual TF32/autocast/MATH contract')
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats(device)
        precision=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        classes_before=model_inventory(model,p['bindings'])
        trace=bridge.Trace(model,torch,store,check,p['expected_calls'])
        guard=bridge.native.Ledger(4,lambda e:store.event(dict(kind='guard_event',event=e)),
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
            numeric_before=precision_state(torch);check()
            process(p,rows,store,encode,check)
            torch.cuda.synchronize(device);check();numeric_after=precision_state(torch)
            peak=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
    except BaseException as e:errors.append(physical.kernel.error('encode',e))
    finally:
        if precision is not None:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=precision
        try:
            if model is not None:
                bridge.native.frozen(model);after=bridge.native.tensor_hash(model);classes_after=model_inventory(model,p['bindings'])
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash']
            and classes_before==classes_after and numeric_before==numeric_after==planner.precision_contract()
            and peak is not None and all(v<=contract()[k] for v,k in zip(peak,('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)) and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        # native seal precedes any analysis of the new encoded outputs.
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='terminal_encoded_science_blocked' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,precision_before=numeric_before,precision_after=numeric_after,
            model_inventory_before=classes_before,model_inventory_after=classes_after,CUDA_peak_bytes=peak,
            P1_admission=False,P3_admission=False,C3=False,physical_release=False)
        store.put('report.json',report,final=True)
    require(complete,'partial retained/no retries '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row,load):
    z=load(row['latent']) if row['latent'] is not None else None
    actual=bridge.mse(z,spec['goal192']) if z is not None else None
    anchor=spec['checkpoint']==5
    if z is not None and anchor:bridge.same(z,bridge.latent(spec['observed_anchor192']),'exact original observed cp5 re-encode')
    pred=spec['model_Q']
    return dict(index=spec['index'],plan_index=spec['plan_index'],root_slot=spec['root_slot'],condition='off',
        arm=spec['arm'],task_id=spec['task_id'],checkpoint=spec['checkpoint'],selection_digest=spec['selection_digest'],
        role='observed_anchor_reencode' if anchor else 'same_action_H3_prediction',
        model_Q=pred,actual_encoded_Q=actual,
        model_minus_actual_Q=pred-actual if pred is not None and actual is not None else None,
        absolute_Q_error=abs(pred-actual) if pred is not None and actual is not None else None,
        prediction_vs_encoded_MSE=bridge.mse(z,spec['predicted192']) if z is not None and not anchor else None,
        observed_anchor_reencode_MSE=bridge.mse(z,spec['observed_anchor192']) if z is not None and anchor else None,
        actual192=None if z is None else z.tolist(),predicted192=spec['predicted192'],goal192=spec['goal192'],
        latent_missing=z is None,source=spec['source'],native_metrics=spec['native_metrics'],response=spec['response'],
        causal_raw10_bits=spec['causal_raw10_bits'],physical_status=spec['physical_status'],
        physical_benefit_claim=False,P1_admission=False,P3_admission=False,C3=False)


def context_chains(p,results):
    """Actual feedback chain only; never a new forecast or deployment."""
    chains=[]
    for i,plan in enumerate(p['plans']):
        w=plan['source_witness'];context=deepcopy(w['request']['task']['root_context']);items=[]
        for j,step in enumerate(CP):
            row=results[4*i+j];spec=p['rows'][4*i+j]
            require(row['plan_index']==i and row['checkpoint']==step,'fixed context chain order')
            z=row['actual192']
            if z is None:context=None
            elif context is not None:
                if step==5:context['latent_history'][-1]=deepcopy(z)
                else:context=planner.moving.roll(context,bridge.latent(z),
                    np.asarray(spec['causal_raw10'],np.float32),w['task'])
                require(context['action_history']==spec['normalized_action_history'],
                    'observed context has original causal action tokens once')
            items.append(dict(step=step,context=deepcopy(context),context_missing=context is None))
        chains.append(dict(plan_index=i,root_slot=plan['root_slot'],bank=plan['arm'],checkpoints=items,
            scope='actual observed histories, no model prediction or policy intervention'))
    return chains



def validate(p,r,events,load):
    require(r['contract']==p['contract']==contract() and len(p['rows'])==len(r['rows'])==16 and r['errors']==r['blocked']==[],'all16 clean checkpoint report')
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
        buffers=[pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z]
        require(all(x.dtype==np.float32 and np.isfinite(x).all() for x in buffers),'literal full native float32 buffers')
        physical.saved.overlay.visual_bridge(*buffers,[1,257,192])
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
        and r['P1_admission'] is r['P3_admission'] is r['C3'] is r['physical_release'] is False and 0<=r['elapsed_seconds']<=600,'complete frozen CUDA only')
    require(r['precision_before']==r['precision_after']==planner.precision_contract()
        and r['model_inventory_before']==r['model_inventory_after'] and r['model_inventory_before'],
        'actual numeric flags and presealed model inventory before/after')
    for entry in r['model_inventory_before']:
        require(p['bindings'][entry['path']]==entry['sha256'],'bound installed class identity')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2
        and all(type(v) is int and 0<=v<=contract()[k] for v,k in zip(r['CUDA_peak_bytes'],
            ('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'actual CUDA peak resource axes')
    require(all(f.is_file() and not f.is_symlink() for f in output.iterdir()),'regular flat output artifacts')
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
    return dict(schema=NAME+'.consume',status='all16_H3_witness_RGB_verified_NOT_admitted',rows=rows,report_sha256=expected,
        counts=r['counts'],fixed_denominator=16,plan_denominator=4,known_roots=2,fresh_roots=0,total_FLOPs=None,actual_context_chains=context_chains(p,rows),
        physical_report_sha256=p['physical_report_sha256'],physical_receipt_sha256=p['physical_receipt_sha256'],
        inherited_physical_counts=p['inherited_physical_counts'],inherited_model_resources=p['inherited_model_resources'],
        inherited_model_wall_seconds=p['inherited_model_wall_seconds'],consume_new_neural_calls=0,consume_new_physics=0,
        P1_admission=False,P3_admission=False,C3=False,equal_total_compute=False,physical_release=False)



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


