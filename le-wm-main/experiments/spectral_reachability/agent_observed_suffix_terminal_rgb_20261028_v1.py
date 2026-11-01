"""Actual observed-suffix known4 cp10 RGB -> original frozen latent192.
Own physical/observed-v2 authority. MAIN HOST CUDA only; no predict/AD/physics.
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
from experiments.spectral_reachability import agent_observed_suffix_physical_20261028_v1 as producer
from experiments.spectral_reachability import agent_nonlinear_suffix_terminal_rgb_20261028_v1 as template

NAME='agent_observed_suffix_terminal_rgb_20261028_v1'
AUTH='MAIN_OBSERVED_SUFFIX4_TERMINAL_RGB_HOST_CUDA_ONLY'
ROOT=producer.ROOT
bridge,rgb_base=template.bridge,template.rgb_base
require,sha,raw,read=producer.require,producer.sha,producer.raw,producer.read
CAP,FINAL,FLOOR,WALL=template.CAP,template.FINAL,template.FLOOR,template.WALL
META_CAP=producer.META_CAP
ARGUMENTS={'physical_output','physical_report_sha','physical_prepared','physical_prepare_sha',
    'physical_receipt','physical_receipt_sha','physical_source_sha'}
FROZEN=((producer,'246739a509c8727bb9381d0c5df3c9649ab6dd9b484ec8f4b8f9bb5891e45a65'),
    (template,'9ba540aa20f1c0a3d45a452444bbd648e254b22a629ed71a103e9d40453e5cd1'),*template.FROZEN_HELPERS)


def contract():
    return dict(schema=NAME+'.contract',physical_schema=producer.NAME+'.consume',planner_schema=producer.compiler.NAME+'.consume',
        fixed_denominator=4,known_roots=2,fresh_roots=0,checkpoint=10,
        scope='all4 new observed-suffix physical cp10 PNG; no old latent, plan inheritance or action selection',
        source_sha='MAIN actual frozen physical source CLI, must match original physical prepare bindings',
        digest_domains='compiler.digest for compiler plans/goal/context; producer.digest for physical execution/ref/guard',
        max_calls=dict(encoder=4,projector=4,predict=0,action_encoder=0,pred_proj=0,AD=0),
        missing='genuine native stop null4 only; no earlier PNG, old latent, duplicate or padding',
        Q='original saved goal192/identity Q; float64 mean squared distance',
        batch=1,encoder_hidden=[1,257,192],latent=192,CUDA_only=True,CPU_fallback=False,
        precision=template.precision_contract(),new_physics=0,new_prediction=0,new_optimizer=0,training=False,Qfit=False,retries=0,
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,wall_seconds=WALL,
        P1_admission=False,freshness=False,scientific_admission=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']


def fixed_rows(p,strict,compiled,report,events,output):
    producer.validate_execution(p)
    require(strict['schema']==producer.NAME+'.consume' and strict['status']==producer.RECORDED and
        strict['errors']==[] and strict['fixed_denominator']==4 and strict['known_roots']==2 and
        strict['fresh_roots']==strict['inherited']==0 and all(strict[k] is False for k in
        ('P1_admission','P3_admission','scientific_admission','freshness','equal_FLOPs','equal_total_compute','independent_confirmation')) and
        strict['planner_authority']==p['planner_authority'],'actual entire new observed physical consume/authority')
    require(report['schema']==producer.NAME+'.report' and report['status']==producer.RECORDED and
        report['errors']==[] and report['rows']==strict['rows'] and
        compiled['schema']==producer.compiler.NAME+'.prepare' and compiled['contract']==producer.compiler.contract() and
        compiled['production_ready'] is True and len(p['plans'])==len(strict['rows'])==len(compiled['rows'])==4,
        'all4 actual observed schema, never old nonlinear or partial compiler')
    producer.audit_guard(report,events,p)
    rows=[]
    for i,(plan,actual,spec,ref) in enumerate(zip(p['plans'],strict['rows'],compiled['rows'],p['prefix_reference'],strict=True)):
        key=producer.ORDER[i];task=producer.compiler.core.validate_task(plan['model_task']);binding=task['binding']
        chosen=plan['compiler_plan'];request=spec['request']
        require(plan['index']==actual['index']==spec['index']==chosen['slot']==i and
            all((x['root_slot'],x['condition'])==key for x in (plan,actual,spec,chosen)) and actual['plan']==plan and
            actual['task_id']==plan['task_id']==task['binding']['task_id'] and
            actual['selection_digest']==plan['selection_digest'] and actual['arm']==plan['arm'] and
            plan['model_task']==request['task'] and plan['model_task_binding']==chosen['model_task_binding']==binding and
            task['binding']['model_sha256']==compiled['expected_model_hash'] and np.array_equal(task['projection'],np.eye(192)),
            'exact new task/model/Q/plan/raw identity')
        # Never hash another schema using the physical serializer.
        require(chosen['schema']==producer.compiler.NAME+'.plan' and
            chosen['selection_digest']==producer.compiler.digest({k:v for k,v in chosen.items() if k!='selection_digest'}) and
            chosen['goal_sha256']==producer.compiler.digest(task['goal']) and
            chosen['observed_context_sha256']==producer.compiler.digest(request['observed_context']) and
            chosen['commands']==plan['commands'] and chosen['new_suffix_Q']==plan['new_suffix_Q'] and
            chosen['winner_iteration']==plan['iterate_index'],'compiler-owned compact digest and indivisible new witness')
        goal=bridge.latent(task['goal']);pred=bridge.latent(chosen['new_terminal_context']['latent_history'][-1])
        require(bridge.mse(pred,goal)==chosen['new_suffix_Q'] and template.hash_raw(plan['commands'])==plan['raw_float32_sha256'],
            'original saved goal/Q and actual new predicted endpoint arithmetic')
        native=p['source_roots'][key[0]]
        require(all(plan[k]==native[v] for k,v in (('native_goal','goal'),('native_goal_pose','goal_pose'),('native_root_sha256','root_sha256'))) and
            plan['native_goal_pixel_sha256']==native['goal_pixels'][key[1]],'same original physical goal/root/condition')
        require(actual['status'] in ('horizon_completed','native_stopped') and actual['first_error'] is None and actual['secondary_errors']==[],
            'native-only terminal/missing, not an engineering omission')
        if actual['initial_exact']:
            require(all(actual[k]==native[k] for k in ('root_state','root_sha256','goal','goal_pose')) and
                actual['goal_pixel_sha256']==native['goal_pixels'][key[1]],'actual original initial state/goal identity')
        cps=actual['checkpoints'];steps=[c['step'] for c in cps]
        require(len(set(steps))==len(steps) and sorted(steps)==[s for s in (0,5,10) if s not in actual['missing_checkpoints']] and
            sorted(actual['missing_checkpoints'])==[s for s in (0,5,10) if s not in steps],'complete checkpoint/missing denominator')
        cp5=[c for c in cps if c['step']==5]
        guard=[e for e in events if e.get('branch')==i and e['kind']=='cp5_guard']
        if actual['own_returned']>5:
            require(len(guard)==1 and guard[0]['accepted'] is True and guard[0]['error'] is None and
                actual['prefix_reference_exact'] is True and producer.prefix_match(actual,ref),'causal exact cp5 before any new suffix')
        else:require(not guard,'no suffix guard fabricated before step6')
        cp10=[c for c in cps if c['step']==10];source=None
        if cp10:
            f=cp10[0];found=[t for t in actual['trajectory'] if t['phase']=='plan' and t['step']==10]
            require(actual['own_returned']==10 and actual['initial_exact'] is True and len(found)==1 and
                found[0]['condition']==key[1] and found[0]['overlay_index']==(1 if key[1]=='on' else 0) and
                all(found[0][k]==f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')) and
                len(f['state'])==7 and f['state']==f['snapshot']['state']==actual['final_state'] and
                Path(f['png']).name==f['png'] and f['png'] not in ('','.','..'),'actual new cp10 same-condition native7/state/time/PNG')
            source=dict(deepcopy(f),path=str(Path(output)/f['png']))
        else:
            require(actual['status']=='native_stopped' and 0<=actual['own_returned']<10 and
                (actual['terminated'] is True or actual['truncated'] is True or
                 actual['initial_exact'] is False and actual['warmup_returned']<10),'missing cp10 only genuine native early stop')
        rows.append(dict(index=i,root_slot=key[0],condition=key[1],arm=plan['arm'],task_id=plan['task_id'],
            task_sha256=binding['task_sha256'],model_task_binding=deepcopy(binding),original_task=deepcopy(task),
            selection_digest=plan['selection_digest'],compiler_selection_digest=chosen['selection_digest'],
            selected_index=plan['selected_index'],iterate_index=plan['iterate_index'],full10=deepcopy(plan['commands']),
            raw_float32_sha256=plan['raw_float32_sha256'],goal_sha256=chosen['goal_sha256'],q_sha256=binding['projection_sha256'],
            goal192=goal.tolist(),predicted192=pred.tolist(),predicted_Q=chosen['new_suffix_Q'],
            observed_context=deepcopy(request['observed_context']),observed_context_sha256=chosen['observed_context_sha256'],
            prefix_reference_sha256=plan['prefix_reference_sha256'],actual_cp5=deepcopy(cp5),
            actual_cp5_guard=deepcopy(guard),prefix_reference_exact=actual['prefix_reference_exact'],
            compiler_plan=deepcopy(chosen),planner_report_sha256=plan['planner_report_sha256'],
            planner_receipt_sha256=plan['planner_receipt_sha256'],source_report_sha256=plan['source_report_sha256'],
            native_root_sha256=plan['native_root_sha256'],native_goal=deepcopy(plan['native_goal']),
            native_goal_pose=deepcopy(plan['native_goal_pose']),native_goal_pixel_sha256=plan['native_goal_pixel_sha256'],
            physical_status=actual['status'],evaluation={k:deepcopy(actual[k]) for k in
                ('final_state','final_metrics','native_success','terminated','truncated')},
            source=source,status='not_started' if source is not None else 'missing_physical_checkpoint10',latent=None))
    return rows


def build(physical_output,physical_report_sha,physical_prepared,physical_prepare_sha,physical_receipt,physical_receipt_sha,physical_source_sha):
    args=dict(physical_output=physical_output,physical_report_sha=physical_report_sha,physical_prepared=physical_prepared,
        physical_prepare_sha=physical_prepare_sha,physical_receipt=physical_receipt,physical_receipt_sha=physical_receipt_sha,
        physical_source_sha=physical_source_sha)
    for k,v in args.items():
        if k.endswith('_sha'):producer.mixed.actual_hash(v)
        else:producer.mixed.absolute(v)
    for module,h in FROZEN:require(sha(module.__file__)==h,'frozen reviewed encoding helper')
    code=str(Path(producer.__file__).resolve())
    require(sha(code)==physical_source_sha,'MAIN actual frozen new physical source before consume')
    output=Path(physical_output)
    require(sha(output/'report.json')==physical_report_sha and sha(physical_prepared)==physical_prepare_sha and
        sha(physical_receipt)==physical_receipt_sha,'all external actual physical seals, no future defaults')
    strict=producer.consume(output,physical_report_sha)
    require(raw(strict)==raw(read(physical_receipt,META_CAP)),'entire own physical persisted receipt reconstructed')
    report=read(output/'report.json',producer.CAP);p=read(physical_prepared,META_CAP)
    require(report['execution_binding']['prepared']==physical_prepared and
        report['execution_binding']['prepare_sha256']==physical_prepare_sha and
        strict['physical_report_sha256']==physical_report_sha and
        strict['bindings'][code]==p['bindings'][code]==physical_source_sha,'exact new physical prepare/source/execution')
    a=p['planner_authority'];cp=Path(a['compile_output'])/'report.json'
    require(sha(cp)==strict['bindings'][str(cp)]==a['compile_report_sha'] and
        sha(a['compile_receipt'])==strict['bindings'][a['compile_receipt']]==a['compile_receipt_sha'] and
        sha(producer.compiler.__file__)==a['compile_source_sha'],'same complete observed-v2 compiler report/receipt/source')
    cr=read(cp,META_CAP);compiled=read(cr['prepared'],META_CAP)
    require(sha(cr['prepared'])==strict['bindings'][cr['prepared']]==cr['prepare_sha256'],'same actual observed compiler preparation')
    body=bridge.rgb.flat(output,'journal.jsonl').read_bytes()
    require(len(body)<=producer.CAP and hashlib.sha256(body).hexdigest()==report['journal_sha256'] and body.endswith(b'\n'),'bound actual physical journal')
    events=[producer.mixed.decode(line) for line in body.splitlines()]
    rows=fixed_rows(p,strict,compiled,report,events,output)
    bindings=deepcopy(strict['bindings'])
    producer.mixed.merge(bindings,{str(Path(physical_receipt)):physical_receipt_sha,str(Path(physical_prepared)):physical_prepare_sha,
        str(output/'report.json'):physical_report_sha,code:physical_source_sha})
    producer.mixed.merge(bindings,bridge.rgb.MODELS)
    files=own_files()+producer.own_files()+[Path(bridge.__file__).resolve(),Path(bridge.native.__file__).resolve(),
        Path(bridge.rgb.__file__).resolve(),Path(bridge.pixels.__file__).resolve(),Path(rgb_base.__file__).resolve(),Path(template.__file__).resolve()]
    producer.mixed.merge(bindings,{str(f):sha(f) for f in files})
    producer.compiler.F.source.kernel.validate_normalizer_source(ROOT)
    require(str(Path(compiled['policy']).resolve())==str(bridge.rgb.POLICY.resolve()) and
        compiled['expected_model_hash']==bridge.rgb.MODEL_TENSOR_SHA,'same original frozen encoder model')
    shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json'))
    require(shapes['hidden']==[1,257,192],'original224/patch14/257 ABI')
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    bridge.verify_bindings(bindings);n=sum(row['source'] is not None for row in rows)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=args,rows=rows,bindings=bindings,
        policy=compiled['policy'],expected_model_hash=compiled['expected_model_hash'],shapes=shapes,
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        physical_authority=deepcopy(a),actual_new_physical_counts=strict['counts'],
        inherited_recompile_resources=deepcopy(a['actual_new_compute']),inherited_resources=deepcopy(a['inherited_resources']))


def write(path,value):
    path=Path(producer.mixed.absolute(str(path)));payload=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
        len(payload)<=META_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'new exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    producer.mixed.actual_hash(expected);require(sha(path)==expected,'actual own prepare SHA');p=read(path,META_CAP)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'whole actual observed source/physical/compile reconstruction')
    return p


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_') and shutil.disk_usage(Path(output).parent).free>=CAP+FLOOR,'new own full-cap output')
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
        report=dict(schema=NAME+'.report',status='observed_suffix4_encoded_not_admitted' if complete else 'partial_or_failed',
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
        compiler_selection_digest=spec['compiler_selection_digest'],observed_context_sha256=spec['observed_context_sha256'],
        prefix_reference_sha256=spec['prefix_reference_sha256'],actual_cp5=spec['actual_cp5'],
        actual_cp5_guard=spec['actual_cp5_guard'],prefix_reference_exact=spec['prefix_reference_exact'],
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
    output=Path(producer.mixed.absolute(str(output)));producer.mixed.actual_hash(expected)
    require(output.resolve()==output and sha(output/'report.json')==expected,'actual own terminal encoder report SHA/path')
    r=read(output/'report.json',FINAL);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='observed_suffix4_encoded_not_admitted' and
        r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['P1_admission'] is r['scientific_admission'] is r['freshness'] is False and 0<=r['elapsed_seconds']<WALL,'clean original model/source/whole-stage limits')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'full output census/cap')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'all actual artifact SHAs')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        read(output/'initial_roster.json',FINAL)==p['rows'],'own exact execution/initial denominator')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and read(output/'native_seal.json',FINAL)==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors','precision_before','precision_after')},'native durable seal before analysis')
    rows=validate(p,r,[read(output/f'event_{i:04}.json') for i in range(r['event_count'])],lambda a:bridge.rgb.load_array(a,blob))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected and sha(r['prepared'])==r['prepare_sha256'],'final report/prepare/source unchanged')
    return dict(schema=NAME+'.consume',status='observed_suffix4_RGB_verified_not_admitted',rows=rows,counts=r['counts'],
        report_sha256=expected,physical_report_sha256=p['arguments']['physical_report_sha'],
        physical_prepare_sha256=p['arguments']['physical_prepare_sha'],physical_receipt_sha256=p['arguments']['physical_receipt_sha'],
        physical_source_sha256=p['arguments']['physical_source_sha'],physical_authority=p['physical_authority'],
        actual_new_physical_counts=p['actual_new_physical_counts'],inherited_recompile_resources=p['inherited_recompile_resources'],
        inherited_resources=p['inherited_resources'],fixed_denominator=4,known_roots=2,fresh_roots=0,total_FLOPs=None,
        consume_new_neural_calls=0,consume_new_physics=0,P1_admission=False,scientific_admission=False,freshness=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-output','physical-report-sha','physical-prepared','physical-prepare-sha',
        'physical-receipt','physical-receipt-sha','physical-source-sha','report-sha','receipt'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':
        v=write(a.output,build(a.physical_output,a.physical_report_sha,a.physical_prepared,a.physical_prepare_sha,
            a.physical_receipt,a.physical_receipt_sha,a.physical_source_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN host CUDA flag required');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:require(a.receipt,'new own receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())

