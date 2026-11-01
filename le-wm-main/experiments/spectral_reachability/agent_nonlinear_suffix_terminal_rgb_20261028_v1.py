"""Nonlinear-suffix known4 actual cp10 RGB -> original frozen latent192.

Own physical/suffix-v2 authority, no terminal-witness schema impersonation.
HOST CUDA by MAIN only. No prediction, AD, physics, fitting or action selection.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from experiments.spectral_reachability import agent_nonlinear_suffix_physical_20261028_v1 as producer
from experiments.spectral_reachability import agent_terminal_witness_terminal_rgb_20261028_v1 as rgb_base

NAME='agent_nonlinear_suffix_terminal_rgb_20261028_v1'
AUTH='MAIN_NONLINEAR_SUFFIX4_TERMINAL_RGB_HOST_CUDA_ONLY'
ROOT=producer.ROOT
require,sha,raw,read=producer.require,producer.sha,producer.raw,producer.read
bridge=rgb_base.bridge
suffix=producer.suffix
CAP,FINAL,FLOOR,WALL=bridge.CAP,bridge.FINAL,bridge.FLOOR,bridge.WALL
FROZEN_HELPERS=((producer,'9108512019f0a7154a5eec56dfc55c2af9035362c5687130eb48eae5d7c9dc25'),
    (rgb_base,'310eda7b2fcc91e069a917f5963a962b610e9ef239a5796d47cc2b466dbec396'),
    *rgb_base.FROZEN_HELPERS[1:])
ARGUMENTS={'physical_output','physical_report_sha','physical_prepared','physical_prepare_sha',
    'physical_receipt','physical_receipt_sha'}


def contract():
    return dict(schema=NAME+'.contract',physical_schema=producer.NAME+'.consume',
        planner_schema='agent_p1_nonlinear_suffix_library_20261028_v2.consume',
        fixed_denominator=4,known_roots=2,fresh_roots=0,checkpoint=10,
        scope='all4 nonlinear-suffix final plans; only actual same-condition new cp10; no action selection',
        max_calls=dict(encoder=4,projector=4,predict=0,action_encoder=0,pred_proj=0,AD=0),
        missing='null4 denominator; native early stop only; no earlier PNG, old latent, dedup or padding',
        Q='unchanged identity192 float64 terminal mean squared distance to original saved goal192',
        normalization='original action mean/std unchanged; no extra tanh/clip/scale',
        batch=1,latent=192,encoder_hidden=[1,257,192],CUDA_only=True,TF32=False,autocast=False,SDP='MATH',
        new_physics=0,new_prediction=0,new_optimizer=0,training=False,Qfit=False,retries=0,
        state_use='provenance/evaluation only, never model.encode inputs',
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        total_FLOPs=None,equal_FLOPs=False,P1_admission=False,scientific_admission=False,freshness=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']


def hash_raw(value):
    return hashlib.sha256(producer.mixed.raw10(value).tobytes()).hexdigest()


def fixed_rows(prep,strict,compiled,model_report,output):
    """All4 exact own-schema joins; state/outcome never choose or alter plans."""
    producer.validate_execution(prep)
    require(strict['schema']==producer.NAME+'.consume' and strict['status']==producer.RECORDED and
        strict['errors']==[] and strict['fixed_denominator']==4 and strict['known_roots']==2 and
        strict['fresh_roots']==strict['inherited']==0 and all(strict[k] is False for k in
        ('P1_admission','scientific_admission','freshness','independent_confirmation','equal_FLOPs')),
        'complete actual nonlinear physical consume, never old selector receipt')
    authority=prep['planner_authority']
    require(strict['planner_authority']==authority and authority['producer']==suffix.NAME and
        authority['complete_all168'] is authority['full_source_build_verified'] is authority['persisted_receipt_equal'] is True and
        compiled['schema']==suffix.NAME+'.prepare' and model_report['schema']==suffix.NAME+'.report' and
        model_report['status']=='all168_nonlinear_suffix_sealed_not_admitted' and model_report['errors']==[] and
        model_report['prepared']==authority['producer_prepared'] and
        model_report['prepare_sha256']==authority['producer_prepare_sha256'] and
        model_report['contract']==compiled['contract']==suffix.contract() and
        model_report['fixed_denominator']==len(model_report['rows'])==168 and model_report['slot_denominator']==4 and
        all(row['status']=='sealed' and row['slot']==i//42 and row['prefix_index']==i%42
            for i,row in enumerate(model_report['rows'])), 'complete168 real suffix-v2 source/prepare/finals')
    require(len(prep['plans'])==len(strict['rows'])==len(compiled['slots'])==len(model_report['finals'])==4,
        'all4 original known root-condition finals, no subset')
    rows=[]
    for i,(plan,actual,slot,final) in enumerate(zip(prep['plans'],strict['rows'],compiled['slots'],model_report['finals'],strict=True)):
        key=producer.ORDER[i];task=slot['task'];binding=task['binding'];native=prep['source_roots'][key[0]]
        suffix.core.validate_task(task)
        require(plan['schema']==producer.NAME+'.execution_plan' and
            type(actual['index']) is type(slot['index']) is type(final['slot']) is int and
            plan['index']==actual['index']==slot['index']==final['slot']==i and
            (plan['root_slot'],plan['condition'])==(actual['root_slot'],actual['condition'])==
            (slot['root_slot'],slot['condition'])==key and actual['plan']==plan and
            actual['selection_digest']==plan['selection_digest'] and actual['task_id']==plan['task_id'] and
            actual['arm']==plan['arm'] and plan['arm']==producer.contract()['arm'] and final['status']=='sealed',
            'exact nonlinear newplan/raw/row/final identity')
        require(plan['model_task_binding']==binding and all(plan[k]==binding[k] for k in
            ('task_id','task_sha256','blind_sha256','context_sha256','goal_sha256')) and
            plan['q_sha256']==binding['projection_sha256'] and
            np.array_equal(task['projection'],np.eye(192)) and
            binding['model_sha256']==compiled['expected_model_hash'] and
            task['action_mean']==suffix.source.kernel.ACTION_MEAN.tolist() and
            task['action_std']==suffix.source.kernel.ACTION_STD.tolist(),
            'same original task/context/goal/identityQ/model/normalizer')
        chosen=final['plan'];commands=producer.mixed.raw10(plan['commands']);j=plan['selected_index']
        require(chosen['prefix_index']==j and chosen['iterate_index']==plan['iterate_index'] and
            chosen['selection_digest']==plan['producer_selection_digest'] and
            chosen['context']==plan['final_context'] and chosen['physical_release'] is False and
            raw(chosen['commands'])==raw(plan['commands']) and hash_raw(chosen['commands'])==
            plan['raw_float32_sha256']==hash_raw(plan['commands']) and
            commands[:5].tobytes()==np.asarray(slot['candidates'][j]['prefix'],np.float32).tobytes(),
            'original final full10 actions/prefix and predicted context exactly preserved')
        goal=bridge.latent(task['goal']);pred=bridge.latent(chosen['context']['latent_history'][-1])
        require(bridge.mse(pred,goal)==chosen['Q']==plan['nonlinear_terminal_Q'],
            'original exact terminal identityQ, not a ridge or terminal replacement')
        require(actual['status'] in ('horizon_completed','native_stopped') and actual['initial_exact'] is True and
            actual['root_sha256']==native['root_sha256']==plan['native_root_sha256'] and
            actual['root_state']==native['root_state'] and
            actual['goal']==native['goal']==plan['native_goal'] and
            actual['goal_pose']==native['goal_pose']==plan['native_goal_pose'] and
            actual['goal_pixel_sha256']==native['goal_pixels'][key[1]]==plan['native_goal_pixel_sha256'],
            'original native root/goal/pose/condition identity')
        cps=actual['checkpoints'];steps=[f['step'] for f in cps]
        require(len(set(steps))==len(steps) and all(type(s) is int for s in steps) and
            sorted(steps)==[s for s in (0,5,10) if s not in actual['missing_checkpoints']] and
            sorted(actual['missing_checkpoints'])==[s for s in (0,5,10) if s not in steps],
            'exact actual checkpoint/missing full denominator')
        cp=[f for f in cps if f['step']==10];source=None
        if cp:
            f=cp[0];found=[t for t in actual['trajectory'] if t['phase']=='plan' and t['step']==10]
            require(actual['own_returned']==10 and len(found)==1 and found[0]['condition']==key[1] and
                found[0]['overlay_index']==(1 if key[1]=='on' else 0) and
                all(found[0][k]==f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')) and
                f['state']==actual['final_state']==f['snapshot']['state'] and len(f['state'])==7 and
                Path(f['png']).name==f['png'] and f['png'] not in ('','.','..'),
                'true cp10 same-condition state/snapshot/time and actual PNG identity')
            source=dict(deepcopy(f),path=str(Path(output)/f['png']))
        else:
            require(actual['status']=='native_stopped' and 0<=actual['own_returned']<10 and
                (actual['terminated'] is True or actual['truncated'] is True),
                'missing cp10 only genuine native early stop, never engineering omission')
        rows.append(dict(index=i,root_slot=key[0],condition=key[1],arm=plan['arm'],task_id=plan['task_id'],
            task_sha256=plan['task_sha256'],blind_sha256=plan['blind_sha256'],selection_digest=plan['selection_digest'],
            selected_index=j,iterate_index=plan['iterate_index'],producer_selection_digest=plan['producer_selection_digest'],
            planner_report_sha256=plan['planner_report_sha256'],planner_receipt_sha256=plan['planner_receipt_sha256'],
            source_report_sha256=plan['source_report_sha256'],full10=deepcopy(plan['commands']),
            raw_float32_sha256=plan['raw_float32_sha256'],context_sha256=plan['context_sha256'],
            goal_sha256=plan['goal_sha256'],q_sha256=plan['q_sha256'],model_task_binding=deepcopy(binding),
            goal192=goal.tolist(),predicted192=pred.tolist(),predicted_Q=plan['nonlinear_terminal_Q'],
            native_root_sha256=native['root_sha256'],native_goal=deepcopy(native['goal']),
            native_goal_pose=deepcopy(native['goal_pose']),native_goal_pixel_sha256=plan['native_goal_pixel_sha256'],
            raw_history10=plan['commands'],checkpoint_step=10,physical_status=actual['status'],
            evaluation={k:deepcopy(actual[k]) for k in
                ('final_state','final_metrics','native_success','terminated','truncated')},
            source=source,status='not_started' if source is not None else 'missing_physical_checkpoint10',latent=None))
    return rows


def build(physical_output,physical_report_sha,physical_prepared,physical_prepare_sha,physical_receipt,physical_receipt_sha):
    arguments=dict(physical_output=physical_output,physical_report_sha=physical_report_sha,
        physical_prepared=physical_prepared,physical_prepare_sha=physical_prepare_sha,
        physical_receipt=physical_receipt,physical_receipt_sha=physical_receipt_sha)
    for k,v in arguments.items():
        if k.endswith('_sha'):producer.mixed.actual_hash(v)
        else:producer.mixed.absolute(v)
    for module,h in FROZEN_HELPERS:require(sha(module.__file__)==h,'frozen real physical/image helper source')
    output=Path(physical_output);receipt=Path(physical_receipt);ppath=Path(physical_prepared)
    require(sha(receipt)==physical_receipt_sha and sha(output/'report.json')==physical_report_sha and
        sha(ppath)==physical_prepare_sha,'all actual external physical seals, no future defaults')
    strict=producer.consume(output,physical_report_sha)  # includes actual verify -> material -> suffix.consume -> full build
    require(raw(strict)==raw(read(receipt,producer.PREP_CAP)),'entire real physical strict receipt reconstructed')
    report=read(output/'report.json',producer.CAP);binding=report['execution_binding']
    require(binding['prepared']==physical_prepared and binding['prepare_sha256']==physical_prepare_sha and
        strict['physical_report_sha256']==physical_report_sha and strict['rows']==report['rows'],
        'actual physical report/prepare/strict binding')
    pp=read(ppath,producer.PREP_CAP)  # actual producer.consume already executed verify on this exact prepare
    a=pp['planner_authority']
    for key in ('producer_prepared','producer_receipt'):
        require(strict['bindings'][a[key]]==sha(a[key])==a[key.replace('prepared','prepare_sha256').replace('receipt','receipt_sha256')],
            'full inherited suffix prepare/receipt seals')
    model_path=Path(a['producer_output'])/'report.json'
    require(sha(model_path)==strict['bindings'][str(model_path)]==a['producer_report_sha256'],
        'full inherited actual suffix report seal')
    compiled=read(a['producer_prepared'],16*1024**2);model_report=read(model_path,suffix.contract()['final_reserve_bytes'])
    rows=fixed_rows(pp,strict,compiled,model_report,output)
    bindings=deepcopy(strict['bindings'])
    producer.mixed.merge(bindings,{str(receipt):physical_receipt_sha,str(ppath):physical_prepare_sha,
        str(output/'report.json'):physical_report_sha})
    producer.mixed.merge(bindings,bridge.rgb.MODELS)
    files=own_files()+producer.own_files()+[Path(rgb_base.__file__).resolve(),
        Path(bridge.__file__).resolve(),Path(bridge.rgb.__file__).resolve(),
        Path(bridge.native.__file__).resolve(),Path(bridge.pixels.__file__).resolve(),
        Path(bridge.api.__file__).resolve(),Path(bridge.physical.saved.overlay.__file__).resolve()]
    producer.mixed.merge(bindings,{str(f):sha(f) for f in files})
    suffix.source.kernel.validate_normalizer_source(ROOT)
    require(str(Path(compiled['policy']).resolve())==str(bridge.rgb.POLICY.resolve()) and
        compiled['expected_model_hash']==bridge.rgb.MODEL_TENSOR_SHA,'same original frozen encoder/projector model')
    shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json'))
    require(shapes['hidden']==[1,257,192],'config-derived224/patch14 native257 ABI')
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    bridge.verify_bindings(bindings);n=sum(r['source'] is not None for r in rows)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=arguments,rows=rows,bindings=bindings,
        policy=compiled['policy'],expected_model_hash=compiled['expected_model_hash'],shapes=shapes,
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        action_mean=suffix.source.kernel.ACTION_MEAN.tolist(),action_std=suffix.source.kernel.ACTION_STD.tolist(),
        physical_authority=deepcopy(a),actual_new_physical_counts=strict['counts'],
        inherited_suffix_resources=deepcopy(a['actual_new_compute']),
        inherited_original_resources=deepcopy(a['inherited_compute']))


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
        len(payload)<=producer.PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'new exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata write');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    require(sha(path)==expected,'external own actual prepare SHA');p=read(path,producer.PREP_CAP)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,
        'entire own preparation reconstructed without neural calls');return p


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_') and shutil.disk_usage(Path(output).parent).free>=CAP+FLOOR,
            'new own full-cap output')
        super().__init__(output,budget);self.index=0
    def event(self,e):self.put(f'event_{self.index:04}.json',e);self.index+=1


def process(p,rows,store,encode,check=lambda:None):
    require(len(rows)==len(p['rows'])==4,'all four rows including missing')
    ordinal=0
    for row in rows:
        if row['source'] is None:continue
        check();scope=dict(row=row['index'],step=10,phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        pix=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pix)
        z=encode(pix,row,scope,ordinal);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)));ordinal+=1


encode_guard=rgb_base.encode_guard  # Frozen instance hooks/profile probe, no globals patches.


def precision_contract():
    return dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,grad=False,
        math_SDP=True,flash_SDP=False,memory_efficient_SDP=False)


def checked_precision(torch):
    value=dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_TF32=torch.backends.cudnn.allow_tf32,autocast=torch.is_autocast_enabled('cuda'),
        grad=torch.is_grad_enabled(),math_SDP=torch.backends.cuda.math_sdp_enabled(),
        flash_SDP=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_SDP=torch.backends.cuda.mem_efficient_sdp_enabled())
    require(value==precision_contract(),'actual MATH/noTF32/noautocast/inference precision guard')
    return value


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit host CUDA authority before I/O, never CPU fallback')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=guard=None;model=None;before=after=None;source_ok=False;flags=None
    precision_before=precision_after=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen weights')
        for module in model.modules():
            require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'inherited full native model class source closure')
        trace=bridge.Trace(model,torch,store,store.check,p['expected_calls'])
        guard=bridge.native.Ledger(1,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pix,row,scope,ordinal):
            checked_precision(torch)
            guard.image=ordinal
            with bridge.physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pix,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pix),'independent original pixels->preprocessing')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'original native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,encode_guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            precision_before=checked_precision(torch)
            process(p,rows,store,encode,store.check)
            precision_after=checked_precision(torch)
    except BaseException as e:errors.append(bridge.physical.kernel.error('terminal_encode',e))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(bridge.physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            precision_before==precision_after==precision_contract() and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors,
            precision_before=precision_before,precision_after=precision_after)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='nonlinear_suffix4_encoded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,
            model_unchanged=before is not None and before==after,source_verified=source_ok,native_seal_sha256=h,
            artifact_sha256=deepcopy(store.hashes),event_count=store.index,elapsed_seconds=time.monotonic()-budget.started,
            P1_admission=False,scientific_admission=False,freshness=False)
        store.put('report.json',report,final=True)
    require(complete,'partial evidence retained, no retry '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row,load):
    z=None if row['latent'] is None else load(row['latent'])
    actual=None if z is None else bridge.mse(z,spec['goal192'])
    return dict(index=spec['index'],root_slot=spec['root_slot'],condition=spec['condition'],arm=spec['arm'],
        task_sha256=spec['task_sha256'],selection_digest=spec['selection_digest'],selected_index=spec['selected_index'],
        raw_float32_sha256=spec['raw_float32_sha256'],goal_sha256=spec['goal_sha256'],
        predicted_Q=spec['predicted_Q'],actual_Q=actual,
        prediction_minus_actual_Q=None if actual is None else spec['predicted_Q']-actual,
        endpoint_MSE=None if z is None else bridge.mse(z,spec['predicted192']),
        actual192=None if z is None else z.tolist(),goal192=spec['goal192'],latent_missing=z is None,
        iterate_index=spec['iterate_index'],planner_report_sha256=spec['planner_report_sha256'],
        source_report_sha256=spec['source_report_sha256'],evaluation=spec['evaluation'],
        physical_benefit_claim=False,P1_admission=False)


def validate(p,r,events,load):
    require(p['schema']==NAME+'.prepare' and r['schema']==NAME+'.report' and
        p['contract']==r['contract']==contract() and len(p['rows'])==len(r['rows'])==4 and r['errors']==r['blocked']==[],
        'own exact schemas/full4 clean CUDA report')
    require(r['precision_before']==r['precision_after']==precision_contract() and
        p['shapes']['hidden']==[1,257,192],'persisted original config/precision ABI')
    modules=[];guards=[];want=[];ordinal=0
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require(raw({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')})==
            raw({k:v for k,v in spec.items() if k not in ('status','latent')}),'whole task/raw/goal/physical frame identity')
        if spec['source'] is None:require(row==spec,'missing unchanged, never padded');continue
        require(row['status']=='encoded','available encoded exactly once');pix=load(row['pixels'])
        bridge.same(pix,bridge.load_pixels(spec['source']),'actual new physical cp10 PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'independent original pixels->preprocessing')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=10,phase='encode')
        mods=row['modules'];require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector'] and
            all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),'actual full encoder/projector chain')
        en,proj=mods;ein=load(en['inputs'][0]);eout=load(en['output']);pin=load(proj['inputs'][0]);pout=load(proj['output'])
        bridge.same(ein,pre.reshape(1,3,224,224),'exact encoder input bits')
        require(eout.shape==(1,257,192),'native patch14/257 hidden, no shape relaxation')
        bridge.same(pin,eout[:,0],'exact original CLS bridge');bridge.same(pout[0],z,'exact projector->saved latent bits')
        require(pout.shape==(1,192),'original projector shape')
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),
                dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules))) and
        r['guards']==guards and events==want and
        r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'entire event/native/guard chronology and resources; never filter unknown events')
    return [analyze(spec,row,load) for spec,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual terminal encoder report SHA')
    r=read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='nonlinear_suffix4_encoded_not_admitted' and
        r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['P1_admission'] is r['scientific_admission'] is r['freshness'] is False and 0<=r['elapsed_seconds']<=WALL,
        'full actual frozen host CUDA report')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'complete output census/cap')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'all saved artifact SHAs')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        read(output/'initial_roster.json')==p['rows'],'own execution binding and full original roster')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and read(output/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors','precision_before','precision_after')},'native seal before saved evaluation')
    rows=validate(p,r,[read(output/f'event_{i:04}.json') for i in range(r['event_count'])],
        lambda a:bridge.rgb.load_array(a,blob))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='nonlinear_suffix4_RGB_verified_not_admitted',rows=rows,
        report_sha256=expected,physical_report_sha256=p['arguments']['physical_report_sha'],
        physical_receipt_sha256=p['arguments']['physical_receipt_sha'],
        physical_prepare_sha256=p['arguments']['physical_prepare_sha'],physical_authority=p['physical_authority'],
        counts=r['counts'],fixed_denominator=4,known_roots=2,fresh_roots=0,total_FLOPs=None,
        actual_new_physical_counts=p['actual_new_physical_counts'],inherited_suffix_resources=p['inherited_suffix_resources'],inherited_original_resources=p['inherited_original_resources'],
        consume_new_neural_calls=0,consume_new_physics=0,P1_admission=False,scientific_admission=False,freshness=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-output','physical-report-sha','physical-prepared',
              'physical-prepare-sha','physical-receipt','physical-receipt-sha','report-sha','receipt'):
        parser.add_argument('--'+k)
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':
        v=write(a.output,build(a.physical_output,a.physical_report_sha,a.physical_prepared,a.physical_prepare_sha,
            a.physical_receipt,a.physical_receipt_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN host CUDA flag required');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'new own consume receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())

