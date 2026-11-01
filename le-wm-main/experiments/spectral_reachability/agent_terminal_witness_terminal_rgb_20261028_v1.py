"""Terminal-witness mixed222 four NEW physical cp10 PNGs -> frozen latent192.

Own prepare/report/consume, actual producer authority, no schema impersonation.
MAIN HOST CUDA only; worker tests are RAM. No planning, physics, AD or fitting.
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
from experiments.spectral_reachability import agent_terminal_witness_physical_20261028_v1 as producer
from experiments.spectral_reachability import agent_p1_terminal_witness_selector_20261028_v1 as selector
from experiments.spectral_reachability import agent_p2_contextual_physical_rgb_encode_20261028_v1 as bridge

NAME='agent_terminal_witness_terminal_rgb_20261028_v1'
AUTH='MAIN_TERMINAL_WITNESS4_TERMINAL_RGB_HOST_CUDA_ONLY'
ROOT=producer.ROOT
require,sha,raw,read=producer.require,producer.sha,producer.raw,producer.read
CAP,FINAL,FLOOR,WALL=bridge.CAP,bridge.FINAL,bridge.FLOOR,bridge.WALL
PHYSICAL_PREPARE=ROOT/'results/agent_terminal_witness_physical_20261028_v1_prepare_MAIN01.json'
PHYSICAL_PREPARE_SHA='b644dc1954f560448bca62a0131cee71319dc5f0af02c709672b91082af1f7a9'
PHYSICAL_REPORT_SHA='d02868f33f5b4b348c53d3684f30d2681124fc391b4f8d0f9b1a9e8992a04850'
PHYSICAL_RECEIPT_SHA='0f644f89d5864f818805ce95b663d19d4a5d094b65e8f320ef30e3ef3491e8d4'
FROZEN_HELPERS=((producer,'113961de2b48c0a9cb8d0c5829f676df9084fd11a0f2b16cee139f48adb08a7e'),
    (bridge,'fa1d44fb2b753b2deb2c1743cfa0d0496b4fdea42f92d4190cce231260d5375e'),
    (bridge.rgb,'70c03508d775b8b488c12133da929bae607d08d6163d5e1c6578edd9a2422bcc'),
    (bridge.pixels,'f7e3abb842cc185c77fe217922cd83ce709f4a6bf8fce0d9337c87bf4f66f75b'))


def contract():
    return dict(schema=NAME+'.contract',physical_schema=producer.NAME+'.consume',
        selector_schema=selector.NAME+'.analysis',selector_analysis_sha256=producer.ANALYSIS_SHA,
        physical_prepare_sha256=PHYSICAL_PREPARE_SHA,
        physical_report_sha256=PHYSICAL_REPORT_SHA,physical_receipt_sha256=PHYSICAL_RECEIPT_SHA,
        fixed_denominator=4,known_roots=2,fresh_roots=0,family='mixedmode',checkpoint=10,
        scope='only actual new selected terminal cp10, original condition, goal192/Q; no action selection',
        max_calls=dict(encoder=4,projector=4,predict=0,action_encoder=0,pred_proj=0,AD=0),
        missing='null; no early terminal, old PNG/latent, off/on substitution, dedup or padding',
        Q='unchanged identity192 float64 mean squared distance; original saved goal, never re-encode',
        normalization='original float32 action mean/std; no tanh/clip/scale applied by this bridge',
        batch=1,latent=192,encoder_hidden=[1,257,192],CUDA_only=True,TF32=False,autocast=False,SDP='MATH',
        new_physics=0,new_prediction=0,new_optimizer=0,training=False,Qfit=False,retries=0,
        state_use='evaluation and provenance only, never planner/model.encode inputs',
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        total_FLOPs=None,equal_FLOPs=False,P1_admission=False,scientific_admission=False,freshness=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']


def hash_raw(value):
    return hashlib.sha256(producer.mixed.raw10(value).tobytes()).hexdigest()


def fixed_rows(prep,strict,analysis,compiled,model_report,output):
    """Actual physical schema joins; native labels do not alter chosen actions."""
    require(prep['schema']==producer.NAME+'.prepare' and strict['schema']==producer.NAME+'.consume' and
        strict['status']==producer.RECORDED and strict['errors']==[] and strict['fixed_denominator']==4 and
        strict['known_roots']==2 and strict['fresh_roots']==strict['inherited']==0 and
        strict['P1_admission'] is strict['scientific_admission'] is strict['freshness'] is False,
        'own actual known4 physical strict authority, not old mixed receipt')
    authority=prep['selector_authority']
    require(strict['selector_authority']==authority and authority['analysis_sha256']==producer.ANALYSIS_SHA and
        authority['plans_sha256']==producer.mixed.core.digest(prep['plans']) and
        analysis['schema']==selector.NAME+'.analysis','same sealed model-only selector authority')
    panels=[p for p in analysis['panels'] if p['family']=='mixedmode']
    require(len(prep['plans'])==len(strict['rows'])==len(compiled['slots'])==len(panels)==len(model_report['rows'])==4,
        'all four original root-condition plans, no subset')
    rows=[]
    for i,(plan,actual,slot,panel,model) in enumerate(zip(prep['plans'],strict['rows'],compiled['slots'],panels,model_report['rows'],strict=True)):
        key=producer.ORDER[i]
        require(plan['schema']==producer.NAME+'.execution_plan' and plan['index']==actual['index']==slot['index']==panel['index']==model['index']==i and
            (plan['root_slot'],plan['condition'])==(actual['root_slot'],actual['condition'])==
            (slot['root_slot'],slot['condition'])==(panel['root_slot'],panel['condition'])==key and
            raw(actual['plan'])==raw(plan) and actual['selection_digest']==plan['selection_digest']==
            producer.mixed.core.digest({k:v for k,v in plan.items() if k!='selection_digest'}) and
            plan['selector_analysis_sha256']==producer.ANALYSIS_SHA and plan['source_report_sha256']==producer.mixed.SOURCE_SHA and
            plan['selection_uses_physics'] is plan['P1_admission'] is False and plan['raw_steps']==10,
            'exact new plan digest/root/condition/action authority')
        require(all(plan[k]==slot[k] for k in ('task_id','task_sha256','blind_sha256')) and
            actual['task_id']==plan['task_id'] and actual['arm']==plan['arm'] and panel['task_sha256']==slot['task_sha256'] and
            plan['arm']=='terminal_witness_mixed222','original task/blind and own new arm identity')
        require(panel['candidate_denominator']==panel['terminal_available']==len(panel['candidates'])==len(model['result']['candidates'])==222 and
            panel['literal_forward_requests']==[] and panel['status']=='complete_model_only' and panel['P1_admission'] is False,
            'complete frozen mixed222 witness library')
        require(plan['context_sha256']==panel['context_sha256']==selector.sha_object(slot['packet']['root_context']) and
            plan['goal_sha256']==panel['goal_sha256']==selector.sha_object(slot['packet']['goal']) and
            plan['q_sha256']==panel['q_sha256']==selector.base.digest(slot['q']),
            'same exact raw context, goal and Q binding, no teacher replacement')
        selector.base.validate_q(slot['q'])
        selected=panel['nonlinear_terminal_Q'];j=plan['selected_index']
        require(type(j) is int and 0<=j<222 and selected['index']==j and
            hash_raw(plan['commands'])==plan['raw_float32_sha256']==selected['raw_float32_sha256'] and
            hash_raw(selected['commands'])==plan['raw_float32_sha256'],'same selected raw10 float32 bits')
        c=model['result']['candidates'][j];goal=bridge.latent(slot['packet']['goal']);pred=bridge.latent(c['predicted_terminal'])
        require(c['index']==j and hash_raw(c['commands'])==plan['raw_float32_sha256'] and
            selector.sha_object(c['predicted_terminal'])==panel['candidates'][j]['predicted_terminal_sha256'] and
            bridge.mse(pred,goal)==c['nonlinear_terminal_cost']==plan['nonlinear_terminal_Q']==selected['score'],
            'actual selected nonlinear endpoint witness and original terminal Q, not ridge proxy')
        cps=actual['checkpoints'];steps=[f['step'] for f in cps]
        require(len(set(steps))==len(steps) and sorted(steps)==[s for s in (0,5,10) if s not in actual['missing_checkpoints']] and
            sorted(actual['missing_checkpoints'])==[s for s in (0,5,10) if s not in steps],
            'exact actual checkpoint/missing denominator')
        cp=[f for f in cps if f['step']==10];source=None
        if cp:
            f=cp[0];found=[t for t in actual['trajectory'] if t['phase']=='plan' and t['step']==10]
            require(actual['own_returned']==10 and actual['initial_exact'] is True and len(found)==1 and
                found[0]['condition']==key[1] and found[0]['overlay_index']==(1 if key[1]=='on' else 0) and
                all(found[0][k]==f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')) and
                f['state']==actual['final_state'] and Path(f['png']).name==f['png'] and f['png'] not in ('','.','..'),
                'true cp10 same-condition selected trajectory/state PNG, no early terminal padding')
            source=dict(deepcopy(f),path=str(Path(output).resolve()/f['png']))
        rows.append(dict(index=i,root_slot=key[0],condition=key[1],arm=plan['arm'],task_id=plan['task_id'],
            task_sha256=plan['task_sha256'],blind_sha256=plan['blind_sha256'],selection_digest=plan['selection_digest'],
            selected_index=j,selector_analysis_sha256=producer.ANALYSIS_SHA,full10=deepcopy(plan['commands']),
            raw_float32_sha256=plan['raw_float32_sha256'],context_sha256=plan['context_sha256'],
            goal_sha256=plan['goal_sha256'],q_sha256=plan['q_sha256'],goal192=goal.tolist(),predicted192=pred.tolist(),
            predicted_Q=plan['nonlinear_terminal_Q'],model_prefix_uphill_from_anchor=plan['model_prefix_uphill_from_anchor'],
            physical_status=actual['status'],evaluation={k:deepcopy(actual[k]) for k in
                ('final_state','final_metrics','native_success','terminated','truncated')},
            source=source,status='not_started' if source is not None else 'missing_physical_checkpoint10',latent=None))
    return rows


def build(physical_output,physical_report_sha,physical_receipt,physical_receipt_sha):
    for module,expected in FROZEN_HELPERS:
        require(sha(module.__file__)==expected,'frozen actual producer/image low-level source')
    producer.mixed.actual_hash(physical_report_sha);producer.mixed.actual_hash(physical_receipt_sha)
    require(physical_report_sha==PHYSICAL_REPORT_SHA and physical_receipt_sha==PHYSICAL_RECEIPT_SHA,
        'actual completed fixed4 report and full264PNG strict receipt, no future seals')
    output=Path(physical_output).resolve();receipt=Path(physical_receipt).resolve()
    require(sha(receipt)==physical_receipt_sha,'external actual new physical receipt SHA')
    strict=producer.consume(output,physical_report_sha)
    require(raw(strict)==raw(read(receipt)),'entire real physical strict consume reconstructed')
    report=read(output/'report.json');binding=report['execution_binding']
    require(Path(binding['prepared']).resolve()==PHYSICAL_PREPARE and
        sha(binding['prepared'])==binding['prepare_sha256']==PHYSICAL_PREPARE_SHA,'actual frozen MAIN physical prepare identity')
    pp=read(binding['prepared'])  # producer.consume has already verified it.
    require(sha(producer.ANALYSIS)==producer.ANALYSIS_SHA,'actual fixed selector analysis')
    analysis=read(producer.ANALYSIS);item=next(x for x in analysis['inputs'] if x['family']=='mixedmode')
    require(sha(item['prepared'])==item['prepare_sha256']==producer.mixed.PREP_SHA and
        sha(item['report'])==item['report_sha256']==producer.mixed.REPORT_SHA,'original frozen mixed model witnesses')
    compiled=selector.base.overlay.read(item['prepared'],selector.mixed.PREP_CAP)
    model_report=selector.base.overlay.read(item['report'],selector.mixed.CAP)
    rows=fixed_rows(pp,strict,analysis,compiled,model_report,output)
    bindings=deepcopy(strict['bindings'])
    producer.mixed.merge(bindings,{str(receipt):physical_receipt_sha,str(producer.ANALYSIS):producer.ANALYSIS_SHA,
        item['prepared']:item['prepare_sha256'],item['report']:item['report_sha256']})
    producer.mixed.merge(bindings,bridge.rgb.MODELS)
    files=own_files()+producer.own_files()+[Path(bridge.__file__).resolve(),Path(bridge.rgb.__file__).resolve(),
        Path(bridge.native.__file__).resolve(),Path(bridge.pixels.__file__).resolve(),
        Path(bridge.api.__file__).resolve(),Path(bridge.physical.saved.overlay.__file__).resolve()]
    producer.mixed.merge(bindings,{str(f):sha(f) for f in files})
    selector.base.kernel.validate_normalizer_source(ROOT)
    require(str(Path(compiled['policy']).resolve())==str(bridge.rgb.POLICY.resolve()) and
        compiled['expected_model_hash']==bridge.rgb.MODEL_TENSOR_SHA,'same original frozen encoder/projector policy')
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    bridge.verify_bindings(bindings);n=sum(r['source'] is not None for r in rows)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=dict(physical_output=str(output),
        physical_report_sha=physical_report_sha,physical_receipt=str(receipt),physical_receipt_sha=physical_receipt_sha),
        rows=rows,bindings=bindings,policy=compiled['policy'],expected_model_hash=compiled['expected_model_hash'],
        shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json')),
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        action_mean=selector.base.kernel.ACTION_MEAN.tolist(),action_std=selector.base.kernel.ACTION_STD.tolist(),
        actual_new_physical_counts=strict['counts'],inherited_selector_resources=analysis['resources'])


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
        len(payload)<=producer.PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'new exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata write');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    require(sha(path)==expected,'external own actual prepare SHA');p=read(path,producer.PREP_CAP)
    require(raw(p)==raw(build(**p['arguments'])),'entire own preparation reconstructed without neural calls');return p


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


@contextmanager
def encode_guard(model,torch,ledger,errors):
    """Instance wrappers + thread-local public AD call probes; no torch patch.

    A forbidden call is terminal for this run (ledger.blocked is irreversible).
    CPython may disable a profile callback which raises; no inference may resume
    after that first violation. This is not an internal-backward/FLOPs counter.
    """
    undo=[];previous=sys.getprofile()
    apis=[(torch.autograd,n) for n in ('grad','backward')]+[(torch.Tensor,'backward')]
    apis += [(torch.autograd.functional,n) for n in ('jacobian','hessian','jvp','vjp','hvp','vhp')]
    apis += [(torch.func,n) for n in ('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value')]
    codes={};builtins={}
    for obj,name in apis:
        fn=getattr(obj,name);unwrapped=inspect.unwrap(fn);code=getattr(unwrapped,'__code__',None)
        if code is not None:codes[code]=name
        else:builtins[id(fn)]=name
    def probe(frame,event,arg):
        name=codes.get(frame.f_code) if event=='call' else builtins.get(id(arg)) if event=='c_call' else None
        if name is not None:ledger.forbid('AD:'+name)()
        if previous is not None:previous(frame,event,arg)
    def patch_instance(obj,name,value):
        undo.append((obj,name,getattr(obj,name),name in vars(obj)));setattr(obj,name,value)
    try:
        for kind,module in (('encoder',model.encoder),('projector',model.projector)):
            old=module.forward
            def forward(*args,_kind=kind,_old=old,**kw):
                require(not ledger.blocked,'no inference after forbidden call')
                return ledger.dispatch(_kind,args[0],_old,*args,**kw)
            patch_instance(module,'forward',forward)
        for obj,name in ((model,'predict'),(model.action_encoder,'forward'),(model.predictor,'forward'),(model.pred_proj,'forward')):
            patch_instance(obj,name,ledger.forbid(name))
        sys.setprofile(probe)
        yield
    finally:
        sys.setprofile(previous)
        for obj,name,old,had_own in reversed(undo):
            try:
                if had_own:setattr(obj,name,old)
                else:delattr(obj,name)
            except BaseException as exc:errors.append(bridge.physical.kernel.error('restore_'+name,exc))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit host CUDA authority before I/O, never CPU fallback')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=guard=None;model=None;before=after=None;source_ok=False;flags=None
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
        trace=bridge.Trace(model,torch,store,store.check,p['expected_calls'])
        guard=bridge.native.Ledger(1,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pix,row,scope,ordinal):
            guard.image=ordinal
            with bridge.physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pix,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pix),'independent original pixels->preprocessing')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'original native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,encode_guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            process(p,rows,store,encode,store.check)
    except BaseException as e:errors.append(bridge.physical.kernel.error('terminal_encode',e))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(bridge.physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='terminal_witness4_encoded_not_admitted' if complete else 'partial_or_failed',
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
        model_prefix_uphill_from_anchor=spec['model_prefix_uphill_from_anchor'],evaluation=spec['evaluation'],
        physical_benefit_claim=False,P1_admission=False)


def validate(p,r,events,load):
    require(p['schema']==NAME+'.prepare' and r['schema']==NAME+'.report' and
        p['contract']==r['contract']==contract() and len(p['rows'])==len(r['rows'])==4 and r['errors']==r['blocked']==[],
        'own exact schemas/full4 clean CUDA report')
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
    require(r['schema']==NAME+'.report' and r['status']=='terminal_witness4_encoded_not_admitted' and
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
        {k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native seal before saved evaluation')
    rows=validate(p,r,[read(output/f'event_{i:04}.json') for i in range(r['event_count'])],
        lambda a:bridge.rgb.load_array(a,blob))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='terminal_witness4_RGB_verified_not_admitted',rows=rows,
        report_sha256=expected,physical_report_sha256=p['arguments']['physical_report_sha'],
        physical_receipt_sha256=p['arguments']['physical_receipt_sha'],selector_analysis_sha256=producer.ANALYSIS_SHA,
        counts=r['counts'],fixed_denominator=4,known_roots=2,fresh_roots=0,total_FLOPs=None,
        actual_new_physical_counts=p['actual_new_physical_counts'],inherited_selector_resources=p['inherited_selector_resources'],
        consume_new_neural_calls=0,consume_new_physics=0,P1_admission=False,scientific_admission=False,freshness=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-output','physical-report-sha',
              'physical-receipt','physical-receipt-sha','report-sha','receipt'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build(a.physical_output,a.physical_report_sha,a.physical_receipt,a.physical_receipt_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN host CUDA flag required');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'new own consume receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
