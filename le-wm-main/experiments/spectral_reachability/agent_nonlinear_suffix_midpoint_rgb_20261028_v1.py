"""Actual known4 midpoint history3 -> frozen same-suffix5 prediction, MAIN CUDA only.

macro5 history = real warmup5, plan0, plan5; no latent teacher substitution.
Own prepare/run/consume; no physics, action/Q changes, training or admission.
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
from experiments.spectral_reachability import agent_nonlinear_suffix_terminal_rgb_20261028_v1 as terminal
from experiments.spectral_reachability import agent_p2_midpoint_teacher_suffix_20261028_v1 as abi

NAME='agent_nonlinear_suffix_midpoint_rgb_20261028_v1'
AUTH='MAIN_NONLINEAR_SUFFIX4_ACTUAL_MIDPOINT_HOST_CUDA_ONLY'
ROOT=terminal.ROOT
producer,bridge,suffix=terminal.producer,terminal.bridge,terminal.suffix
require,sha,raw,read=terminal.require,terminal.sha,terminal.raw,terminal.read
CAP,FINAL,FLOOR,WALL=terminal.CAP,terminal.FINAL,terminal.FLOOR,terminal.WALL
KINDS=bridge.KINDS
FROZEN=((terminal,'9ba540aa20f1c0a3d45a452444bbd648e254b22a629ed71a103e9d40453e5cd1'),
    *terminal.FROZEN_HELPERS)
ARGUMENTS={'terminal_output','terminal_report_sha','terminal_receipt','terminal_receipt_sha'}
ROLES=(('warmup',5),('plan',0),('plan',5))
PROTOCOL=ROOT/'results/main_nonlinear_suffix_midpoint_effect_20261028_v1_protocol_MAIN01.json'
PROTOCOL_SHA='d421793a24ff9c38789f9baa419ff183ecbb36d2bd0082884f313ef763461a18'


def contract():
    return dict(schema=NAME+'.contract',known_roots=2,fresh_roots=0,fixed_denominator=4,
        order=[list(x) for x in producer.ORDER],history3=[list(x) for x in ROLES],macro_raw_steps=5,
        incoming_history='warmup6..10 then original plan1..5',next_packet='original plan6..10 only',
        encode_API=4,encoder_calls=4,encoder_samples=12,projector_calls=4,projector_samples=12,
        predict_API=4,predict_samples=4,predict_context_tokens=12,action_encoder=4,predictor=4,pred_proj=4,
        max_calls={k:4 for k in KINDS},new_physics=0,new_AD=0,new_training=0,new_goal_encode=0,
        encode_ABI=[1,3,3,224,224],hidden_ABI=[3,257,192],latent_ABI=[1,3,192],outgoing_ABI=[1,3,10],
        prediction_semantics='full native3 outputs are not shifted history; terminal output[-1] only is future5',
        intervention='reencode and replace entire actual history3, NOT held-old2/last-only',
        raw='original float32 subtract/divide once, original mean/std; no tanh/clip/rescale',
        Q='original saved goal192/identity Q, float64 mean squared error; never re-fit or re-encode',
        missing='keep null4; only genuinely unavailable history or terminal; never copy/render/pad',
        state_input='provenance/evaluation only; native predictor receives RGB latent3 and action tokens only',
        analysis_protocol_sha256=PROTOCOL_SHA,primary='actual endpoint MSE',secondary='absolute predicted goalQ error',
        precision=terminal.precision_contract(),CUDA_only=True,CPU_fallback=False,retries=0,
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,wall_seconds=WALL,
        total_FLOPs=None,equal_FLOPs=False,P1_admission=False,freshness=False,scientific_admission=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']


def tokens(row):
    """Exact causal incoming2/outgoing3; no stale root bootstrap token."""
    actions=producer.mixed.raw10(row['full10'])
    warm=np.asarray(producer.moving.warmup_commands(),np.float32)
    mean=np.asarray(row['action_mean'],np.float32);std=np.asarray(row['action_std'],np.float32)
    require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.all(std>0),'original action mean/std')
    packets=np.stack((warm[5:],actions[:5],actions[5:]))
    outgoing=((packets-mean)/std).reshape(1,3,10)
    require(outgoing.dtype==np.float32 and np.isfinite(outgoing).all(),'original finite normalized tokens')
    return dict(raw_packets=packets,action_mean=mean,action_std=std,outgoing=outgoing,incoming=outgoing[:,:2].copy())


def fixed_rows(tp,accepted,physical,pp,compiled,output):
    require(tp['schema']==terminal.NAME+'.prepare' and tp['contract']==terminal.contract() and
        accepted['schema']==terminal.NAME+'.consume' and accepted['status']=='nonlinear_suffix4_RGB_verified_not_admitted' and
        accepted['fixed_denominator']==4 and accepted['known_roots']==2 and accepted['fresh_roots']==0 and
        all(accepted[k] is False for k in ('P1_admission','scientific_admission','freshness')) and
        accepted['physical_authority']==tp['physical_authority']==pp['planner_authority'],
        'actual entire terminal RGB/physical/suffix-v2 authority')
    producer.validate_execution(pp)
    require(physical['schema']==producer.NAME+'.report' and physical['status']==producer.RECORDED and
        physical['errors']==[] and len(tp['rows'])==len(accepted['rows'])==len(physical['rows'])==
        len(pp['plans'])==len(compiled['slots'])==4,'all4 known task rows, never subset')
    rows=[]
    for i,(base,end,actual,plan,slot) in enumerate(zip(tp['rows'],accepted['rows'],physical['rows'],pp['plans'],compiled['slots'],strict=True)):
        key=producer.ORDER[i];task=slot['task'];suffix.core.validate_task(task)
        require(base['index']==end['index']==actual['index']==plan['index']==slot['index']==i and
            all((x['root_slot'],x['condition'])==key for x in (base,end,actual,plan,slot)) and actual['plan']==plan and
            all(base[k]==end[k]==plan[k] for k in ('task_sha256','selection_digest','selected_index','raw_float32_sha256','goal_sha256')) and
            base['task_id']==actual['task_id']==plan['task_id']==task['binding']['task_id'] and
            base['model_task_binding']==plan['model_task_binding']==task['binding'] and
            base['full10']==plan['commands'] and terminal.hash_raw(base['full10'])==base['raw_float32_sha256'] and
            base['goal192']==task['goal']==end['goal192'] and base['q_sha256']==task['binding']['projection_sha256'] and
            np.array_equal(task['projection'],np.eye(192)) and base['predicted192']==plan['final_context']['latent_history'][-1],
            'exact original raw10/task/context/goal/Q/model/final plan')
        require(tp['action_mean']==task['action_mean']==suffix.source.kernel.ACTION_MEAN.tolist() and
            tp['action_std']==task['action_std']==suffix.source.kernel.ACTION_STD.tolist(),
            'same original action normalizer')
        frames=[];missing=[]
        for phase,step in ROLES:
            matches=[f for f in actual['trajectory'] if f['phase']==phase and f['step']==step]
            require(len(matches)<=1,'unique actual condition-specific temporal observation')
            if not matches:
                require(actual['status']=='native_stopped' and actual['own_returned']<step and phase=='plan',
                    'history missing only actual native early stop')
                frames.append(None);missing.append([phase,step]);continue
            f=matches[0];snapshot=f['snapshot']
            require(f['condition']==key[1] and f['overlay_index']==(1 if key[1]=='on' else 0) and
                len(f['state'])==7 and f['state']==snapshot['state'] and Path(f['png']).name==f['png'] and
                f['png'] not in ('','.','..'),'exact original phase/time/condition/state/snapshot/frame')
            if phase=='plan':
                cp=[c for c in actual['checkpoints'] if c['step']==step]
                require(len(cp)==1 and all(cp[0][k]==f[k] for k in
                    ('step','png','file_sha256','pixel_sha256','state','snapshot')),'actual cp0/cp5 canonical frame identity')
            frames.append(dict(deepcopy(f),path=str(Path(output)/f['png'])))
        known=[f for f in frames if f is not None]
        require(len({f['path'] for f in known})==len(known),'distinct observation IDs; equal pixels permitted, no frame aliases')
        actual_z=end['actual192']
        if actual_z is not None:
            bridge.latent(actual_z)
            require(base['source'] is not None and actual['own_returned']==10 and
                end['actual_Q']==bridge.mse(actual_z,base['goal192']) and base['source']['step']==10,
                'same strict actual final encoder coordinate and original goal')
        row=deepcopy(base)
        row.update(frames=frames,missing_history=missing,action_mean=deepcopy(tp['action_mean']),action_std=deepcopy(tp['action_std']),
            original_root_context=deepcopy(task['root_context']),actual_terminal192=deepcopy(actual_z),
            actual_terminal_Q=end['actual_Q'],inherited_terminal_endpoint_MSE=end['endpoint_MSE'],
            eligible=not missing,status='not_started' if not missing else 'missing_actual_history',
            actual_full3=None,prediction=None,encoding={},feedback={})
        row.pop('latent');row['source']=deepcopy(base['source'])
        b=tokens(row)
        require(np.asarray(task['root_context']['action_history'],np.float32)[1].tobytes()==b['incoming'][0,0].tobytes(),
            'warmup6..10 incoming token matches original root context, no guessed bootstrap')
        require(bridge.mse(base['predicted192'],base['goal192'])==base['predicted_Q'],
            'same imagined final and original goal arithmetic')
        row['row_binding_sha256']=producer.mixed.core.digest(row);rows.append(row)
    return rows


def build(terminal_output,terminal_report_sha,terminal_receipt,terminal_receipt_sha):
    args=dict(terminal_output=terminal_output,terminal_report_sha=terminal_report_sha,
        terminal_receipt=terminal_receipt,terminal_receipt_sha=terminal_receipt_sha)
    for k,v in args.items():
        if k.endswith('_sha'):producer.mixed.actual_hash(v)
        else:producer.mixed.absolute(v)
    for module,h in FROZEN:require(sha(module.__file__)==h,'frozen actual physical/terminal/array helper')
    require(sha(PROTOCOL)==PROTOCOL_SHA,'MAIN pre-result fixed analysis protocol SHA')
    protocol=read(PROTOCOL,2*1024**2)
    require(protocol['schema']=='main_nonlinear_suffix_midpoint_effect_20261028_v1.protocol' and
        protocol['actual_report'] is None and protocol['status']=='analysis_protocol_only_no_actual_results',
        'pre-result protocol, not a selected post-result metric')
    out=Path(terminal_output);receipt=Path(terminal_receipt)
    require(sha(out/'report.json')==terminal_report_sha and sha(receipt)==terminal_receipt_sha,'external actual terminal seals')
    accepted=terminal.consume(out,terminal_report_sha)
    require(raw(accepted)==raw(read(receipt,producer.PREP_CAP)),'entire real terminal strict receipt reconstructed')
    report=read(out/'report.json',producer.PREP_CAP);tp=read(report['prepared'],producer.PREP_CAP)
    require(sha(report['prepared'])==report['prepare_sha256'],'actual fully verified terminal preparation')
    pa=tp['arguments'];pout=Path(pa['physical_output']);physical=read(pout/'report.json',producer.CAP)
    require(sha(pout/'report.json')==tp['bindings'][str(pout/'report.json')]==pa['physical_report_sha'] and
        physical['execution_binding']['prepared']==pa['physical_prepared'] and
        physical['execution_binding']['prepare_sha256']==pa['physical_prepare_sha'],'exact original physical binding')
    pp=read(pa['physical_prepared'],producer.PREP_CAP)
    require(sha(pa['physical_prepared'])==tp['bindings'][pa['physical_prepared']]==pa['physical_prepare_sha'],'same physical prep')
    path=pp['planner_authority']['producer_prepared']
    require(sha(path)==tp['bindings'][path]==pp['planner_authority']['producer_prepare_sha256'],'same original full168 prepare')
    compiled=read(path,16*1024**2)
    rows=fixed_rows(tp,accepted,physical,pp,compiled,pout)
    bindings=deepcopy(tp['bindings'])
    producer.mixed.merge(bindings,{str(PROTOCOL):PROTOCOL_SHA,**protocol['bindings']})
    producer.mixed.merge(bindings,{str(out/'report.json'):terminal_report_sha,str(receipt):terminal_receipt_sha,
        report['prepared']:report['prepare_sha256']})
    for n,h in report['artifact_sha256'].items():producer.mixed.merge(bindings,{str(out/n):h})
    files=own_files()+[Path(abi.__file__).resolve(),Path(bridge.__file__).resolve()]
    producer.mixed.merge(bindings,{str(f):sha(f) for f in files})
    for row in rows:
        for frame in row['frames']:
            if frame is not None:bridge.load_pixels(frame)
    config=bridge.rgb.read(Path(tp['policy'])/'config.json')
    encoder=bridge.rgb.config_shapes(config);prediction=abi.native_abi(config)
    require(encoder['hidden']==[1,257,192] and tp['expected_model_hash']==compiled['expected_model_hash']==
        bridge.rgb.MODEL_TENSOR_SHA,'original config/full3 ABI/model, no alternate encoder')
    bridge.verify_bindings(bindings);n=sum(row['eligible'] for row in rows)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=args,rows=rows,bindings=bindings,
        policy=tp['policy'],expected_model_hash=tp['expected_model_hash'],encoder_abi=encoder,prediction_abi=prediction,
        expected_calls={k:n for k in KINDS},expected_encode_API=n,
        inherited_resources=dict(physical=accepted['actual_new_physical_counts'],suffix=accepted['inherited_suffix_resources'],
            original=accepted['inherited_original_resources'],terminal_encoding=accepted['counts']),
        physical_authority=accepted['physical_authority'],analysis_protocol=protocol['protocol'])


def write(path,value):
    path=Path(producer.mixed.absolute(str(path)));payload=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and
        not path.is_symlink() and len(payload)<=producer.PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),
        'new exclusive bounded metadata, no overwrite')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata write');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    producer.mixed.actual_hash(expected);require(sha(path)==expected,'external own prepare SHA')
    p=read(path,producer.PREP_CAP)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'whole real midpoint source/frames/raw/goal preparation')
    return p


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_') and shutil.disk_usage(Path(output).parent).free>=CAP+FLOOR,
            'new own full-cap output')
        super().__init__(output,budget);self.index=0
    def event(self,e):self.put(f'event_{self.index:04}.json',e);self.index+=1


class Trace(bridge.Trace):
    """New batch3 constructor; reuse reviewed hook bodies, never old globals."""
    def __init__(self,model,torch,store,check,expected):
        self.model,self.torch,self.store,self.check=model,torch,store,check
        self.encoder_shape=(3,257,192);self.entries=[];self.scope=None;self.pending={};self.hooks=[]
        self.expected=expected;self.api_hooks=bridge.api.mid.Hooks()


@contextmanager
def no_AD(torch,blocked,store):
    previous=sys.getprofile();codes={};builtins={}
    apis=[(torch.autograd,k) for k in ('grad','backward')]+[(torch.Tensor,'backward')]
    apis += [(torch.autograd.functional,k) for k in ('jacobian','hessian','jvp','vjp','hvp','vhp')]
    apis += [(torch.func,k) for k in ('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value')]
    for obj,key in apis:
        fn=inspect.unwrap(getattr(obj,key));code=getattr(fn,'__code__',None)
        if code is None:builtins[id(fn)]=key
        else:codes[code]=key
    def probe(frame,event,arg):
        key=codes.get(frame.f_code) if event=='call' else builtins.get(id(arg)) if event=='c_call' else None
        if key is not None:
            blocked.append('AD:'+key);store.event(dict(kind='forbidden',resource='AD:'+key));raise ValueError('AD forbidden')
        if previous is not None:previous(frame,event,arg)
    try:sys.setprofile(probe);yield
    finally:sys.setprofile(previous)


def process(p,rows,store,encode,predict,check=lambda:None):
    require(len(rows)==len(p['rows'])==4,'fixed four rows, no outcome filtering')
    for row in rows:
        if not row['eligible']:continue
        check();i=row['index'];scope=dict(row=i,step=5,phase='encode');row['status']='encoding'
        store.event(dict(kind='triplet_attempt',scope=scope,row_binding_sha256=row['row_binding_sha256']))
        pix=np.stack([bridge.load_pixels(f) for f in row['frames']])
        row['encoding']['pixels']=store.array(f'r{i}_triplet_pixels',pix)
        full=encode(pix,row['encoding'],scope)
        require(full.dtype==np.float32 and full.shape==(1,3,192) and np.isfinite(full).all(),'native actual full3 encode')
        row['actual_full3']=store.array(f'r{i}_actual_full3',full)
        store.event(dict(kind='triplet_return',scope=scope,encoding=row['encoding'],full3=row['actual_full3']))
        check();scope=dict(row=i,step=5,phase='same_suffix_prediction');b=tokens(row)
        inputs=dict(context=full,**b);feedback=row['feedback']
        feedback['inputs']={k:store.array(f'r{i}_{k}',v) for k,v in inputs.items()}
        row['status']='predicting';store.event(dict(kind='feedback_attempt',scope=scope))
        result=predict(inputs,feedback,scope)
        require(result.dtype==np.float32 and result.shape==(1,3,192) and np.isfinite(result).all(),'full native prediction3')
        row['prediction']=store.array(f'r{i}_prediction3',result);row['status']='predicted'
        store.event(dict(kind='feedback_return',scope=scope,feedback=feedback,prediction=row['prediction']))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN HOST CUDA authority before input reads/imports; no CPU fallback')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];blocked=[];trace=None;model=before=after=None;flags=None;source_ok=False
    precision_before=precision_after=None;host=None;enc_calls=[]
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model)
        require(before==p['expected_model_hash'],'original unchanged weights/buffers')
        for module in model.modules():
            require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'full installed actual model class source closure')
        def check():
            store.check();terminal.checked_precision(torch);require(not blocked,'no continuation after forbidden AD')
        trace=Trace(model,torch,store,check,p['expected_calls']);transform=p0.image_transform()
        def encode(pixels,encoding,scope):
            check()
            with bridge.physical.saved.overlay.traced_scope(trace,scope,encoding):
                x=torch.cat([p0._image_tensor(pic,transform=transform,device=device) for pic in pixels],dim=1)
                encoding['preprocessed']=trace.save(x,f'r{scope["row"]}_preprocessed')
                independent=np.concatenate([bridge.pixels.preprocess_pixels(pic) for pic in pixels],axis=1)
                bridge.same(x.detach().cpu().numpy(),independent,'original actual triplet independent preprocessing')
                e=dict(index=len(enc_calls),scope=scope,completed=False);enc_calls.append(e)
                store.event(dict(kind='encode_API_attempt',entry=deepcopy(e)))
                result=model.encode({'pixels':x})
                require(set(result)=={'pixels','emb'},'original native encode return whitelist')
                encoding['api_pixels']=trace.save(result['pixels'],f'r{scope["row"]}_encode_pixels')
                encoding['api_emb']=trace.save(result['emb'],f'r{scope["row"]}_encode_emb')
                e['completed']=True;store.event(dict(kind='encode_API_return',entry=deepcopy(e)))
                return result['emb'].detach().cpu().numpy().copy()
        def predict(inputs,feedback,scope):
            check()
            with bridge.physical.saved.overlay.traced_scope(trace,scope,feedback):
                lat=torch.tensor(inputs['context'],dtype=torch.float32,device=device)
                outgoing=torch.tensor(inputs['outgoing'],dtype=torch.float32,device=device)
                value=model.predict(lat,model.action_encoder(outgoing))
                return value.detach().cpu().numpy().copy()
        with trace,no_AD(torch,blocked,store),torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            precision_before=terminal.checked_precision(torch)
            process(p,rows,store,encode,predict,check)
            precision_after=terminal.checked_precision(torch)
    except BaseException as exc:errors.append(bridge.physical.kernel.error('midpoint_encode_predict',exc))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'own prepare unchanged');budget.check();source_ok=True
        except BaseException as exc:errors.append(bridge.physical.kernel.error('final_sources_model_limits',exc))
        modules=[] if trace is None else trace.entries
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            precision_before==precision_after==terminal.precision_contract() and
            bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()} and
            len(enc_calls)==p['expected_encode_API'] and all(e['completed'] for e in enc_calls) and
            all(r['status']=='predicted' if r['eligible'] else r['status']=='missing_actual_history' for r in rows))
        seal=dict(rows=rows,modules=modules,encode_API=enc_calls,errors=errors,blocked=blocked,
            precision_before=precision_before,precision_after=precision_after)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='actual_midpoint_suffix4_predicted_not_admitted' if complete else 'partial_or_failed',
            contract=contract(),prepared=str(Path(prepared).resolve()),prepare_sha256=expected,**seal,
            counts=resource_counts(modules,enc_calls),host=host,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,P1_admission=False,scientific_admission=False,freshness=False)
        store.put('report.json',report,final=True)
    require(complete,'partial evidence retained; no retry '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def resource_counts(modules,enc_calls):
    counts=bridge.counts(modules)
    for kind in KINDS:
        es=[e for e in modules if e['resource']==kind]
        counts[kind]['input_axis0_samples']=sum(e['inputs'][0]['shape'][0] for e in es if e['completed'])
        counts[kind]['input_temporal_rows']=sum(e['inputs'][0]['shape'][0]*
            (1 if kind in ('encoder','projector','pred_proj') else e['inputs'][0]['shape'][1]) for e in es if e['completed'])
    counts['encode_API']=dict(attempted=len(enc_calls),completed=sum(e['completed'] for e in enc_calls),
        input_axis0_samples=sum(e['completed'] for e in enc_calls),input_temporal_rows=3*sum(e['completed'] for e in enc_calls))
    return counts


def analyze(spec,row,load):
    predicted=None if row['prediction'] is None else load(row['prediction'])[0,-1]
    actual=spec['actual_terminal192'];goal=spec['goal192']
    conditioned_Q=None if predicted is None else bridge.mse(predicted,goal)
    imagined_error=None if actual is None else bridge.mse(spec['predicted192'],actual)
    conditioned_error=None if predicted is None or actual is None else bridge.mse(predicted,actual)
    return dict(index=spec['index'],root_slot=spec['root_slot'],condition=spec['condition'],task_sha256=spec['task_sha256'],
        selection_digest=spec['selection_digest'],raw_float32_sha256=spec['raw_float32_sha256'],
        goal_sha256=spec['goal_sha256'],row_binding_sha256=spec['row_binding_sha256'],
        imagined_final_Q=spec['predicted_Q'],actual_final_Q=spec['actual_terminal_Q'],conditioned_final_Q=conditioned_Q,
        imagined_endpoint_MSE=imagined_error,conditioned_endpoint_MSE=conditioned_error,
        endpoint_MSE_change=None if conditioned_error is None else conditioned_error-imagined_error,
        imagined_goalQ_absolute_error=None if spec['actual_terminal_Q'] is None else abs(spec['predicted_Q']-spec['actual_terminal_Q']),
        conditioned_goalQ_absolute_error=None if conditioned_Q is None or spec['actual_terminal_Q'] is None else
            abs(conditioned_Q-spec['actual_terminal_Q']),
        conditioned_final192=None if predicted is None else predicted.tolist(),actual_final192=actual,
        eligible=spec['eligible'],missing_history=spec['missing_history'],native_evaluation=spec['evaluation'],
        P1_admission=False,scientific_admission=False,physical_benefit_claim=False)


def load_saved_array(output,reference):
    """Public saved-array reader; caller must first consume the whole report.

    References live in report rows.actual_full3 and rows.feedback.inputs; this
    reader verifies the bounded native dtype/shape/hash, never imports a model.
    """
    out=Path(producer.mixed.absolute(str(output)))
    return bridge.rgb.load_array(reference,lambda name:bridge.rgb.flat(out,name).read_bytes())


def validate(p,r,events,load):
    require(p['schema']==NAME+'.prepare' and r['schema']==NAME+'.report' and
        p['contract']==r['contract']==contract() and len(p['rows'])==len(r['rows'])==4 and r['errors']==r['blocked']==[] and
        r['precision_before']==r['precision_after']==terminal.precision_contract(),'own full4 clean precision/schema')
    expected=[];modules=[];enc_calls=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        dynamic=('status','actual_full3','prediction','encoding','feedback')
        require(raw({k:v for k,v in row.items() if k not in dynamic})==raw({k:v for k,v in spec.items() if k not in dynamic}) and
            spec['row_binding_sha256']==producer.mixed.core.digest({k:v for k,v in spec.items() if k!='row_binding_sha256'}),
            'entire original row/frame/history/raw/task/goal/actual-final identity')
        if not spec['eligible']:require(row==spec,'missing row unchanged null, not padded');continue
        require(row['status']=='predicted','all eligible rows predicted')
        i=row['index'];enc=row['encoding'];fb=row['feedback'];scope=dict(row=i,step=5,phase='encode')
        pix=load(enc['pixels']);wanted=np.stack([bridge.load_pixels(f) for f in spec['frames']])
        bridge.same(pix,wanted,'each original PNG and temporal order')
        pre=load(enc['preprocessed']);independent=np.concatenate([bridge.pixels.preprocess_pixels(x) for x in wanted],axis=1)
        bridge.same(pre,independent,'independent true triplet preprocessing')
        en,proj=enc['modules'];require([en['resource'],proj['resource']]==['encoder','projector'],'one native3 encoder/projector')
        for m in (en,proj):require(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1,'actual encode modules')
        ein,eout,pin,pout=[load(x) for x in (en['inputs'][0],en['output'],proj['inputs'][0],proj['output'])]
        bridge.same(ein,pre.reshape(3,3,224,224),'encode flattened original full3 time axis')
        require(eout.shape==(3,257,192) and pin.shape==pout.shape==(3,192),'config224/patch14/257/CLS192 exact')
        bridge.same(pin,eout[:,0],'native full3 CLS');bridge.same(pout[None],load(row['actual_full3']),'projected full3 context')
        bridge.same(load(enc['api_pixels']),pre,'native encode preserved actual pixels')
        bridge.same(load(enc['api_emb']),load(row['actual_full3']),'native encode returned all3 latents')
        expected.append(dict(kind='triplet_attempt',scope=scope,row_binding_sha256=spec['row_binding_sha256']))
        e=dict(index=len(enc_calls),scope=scope,completed=True);enc_calls.append(e)
        expected.append(dict(kind='encode_API_attempt',entry=dict(e,completed=False)))
        def begin(m):return dict(kind='module_attempt',entry={k:m[k] for k in ('index','scope','resource')}|dict(completed=False))
        def end(m):return dict(kind='module_return',entry=m)
        expected += [begin(en),end(en),begin(proj),end(proj),dict(kind='encode_API_return',entry=e),
            dict(kind='triplet_return',scope=scope,encoding=enc,full3=row['actual_full3'])]
        scope=dict(row=i,step=5,phase='same_suffix_prediction')
        expected.append(dict(kind='feedback_attempt',scope=scope))
        inp={k:load(v) for k,v in fb['inputs'].items()};want=dict(context=load(row['actual_full3']),**tokens(spec))
        require(set(inp)==set(want),'all and only causal prediction inputs')
        for k,value in want.items():bridge.same(inp[k],value,'exact causal '+k)
        ae,api,pr,pp=fb['modules']
        require([m['resource'] for m in fb['modules']]==['action_encoder','predict_API','predictor','pred_proj'] and
            all(m['scope']==scope and m['completed'] is True for m in fb['modules']),'actual suffix5 native module graph')
        require([len(m['inputs']) for m in fb['modules']]==[1,2,2,1],'full native argument census')
        ain,aout,api0,api1,apiout,pr0,pr1,prout,ppin,ppout=[load(x) for x in
            (ae['inputs'][0],ae['output'],*api['inputs'],api['output'],*pr['inputs'],pr['output'],pp['inputs'][0],pp['output'])]
        require(ain.shape==(1,3,10) and all(x.shape==(1,3,192) for x in
            (aout,api0,api1,apiout,pr0,pr1,prout)) and ppin.shape==ppout.shape==(3,192),'native full3 prediction ABI')
        for a,b,label in ((ain,inp['outgoing'],'AE outgoing3'),(api0,inp['context'],'actual latent3'),
            (api1,aout,'AE->predict'),(pr0,api0,'predictor latents'),(pr1,api1,'predictor actions'),
            (ppin,prout[0],'predictor->pred_proj'),(apiout,ppout[None],'pred_proj->predict'),
            (load(row['prediction']),apiout,'saved full native3')):
            bridge.same(a,b,label)
        expected += [begin(ae),end(ae),begin(api),begin(pr),end(pr),begin(pp),end(pp),end(api),
            dict(kind='feedback_return',scope=scope,feedback=fb,prediction=row['prediction'])]
        modules += [en,proj,ae,api,pr,pp]
    require(events==expected and r['modules']==modules and [m['index'] for m in modules]==list(range(len(modules))) and
        r['encode_API']==enc_calls and r['counts']==resource_counts(modules,enc_calls) and
        bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()} and
        len(enc_calls)==p['expected_encode_API'],'complete raw event/native/API/resource census, no filtering')
    return [analyze(spec,row,load) for spec,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    out=Path(producer.mixed.absolute(str(output)));producer.mixed.actual_hash(expected)
    require(sha(out/'report.json')==expected,'external actual midpoint report SHA')
    r=read(out/'report.json',producer.PREP_CAP);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='actual_midpoint_suffix4_predicted_not_admitted' and
        r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        all(r[k] is False for k in ('P1_admission','scientific_admission','freshness')) and
        0<=r['elapsed_seconds']<WALL,'completed original model/source/job/precision report')
    require({f.name for f in out.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in out.iterdir())<=CAP,'full bounded output census')
    def blob(n):return bridge.rgb.flat(out,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'all actual artifact SHAs')
    require(read(out/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        read(out/'initial_roster.json')==p['rows'],'exact own prepare execution/roster binding')
    require(sha(out/'native_seal.json')==r['native_seal_sha256'] and read(out/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','encode_API','errors','blocked','precision_before','precision_after')},
        'durable native seal before saved analysis')
    rows=validate(p,r,[read(out/f'event_{i:04}.json') for i in range(r['event_count'])],
        lambda x:bridge.rgb.load_array(x,blob))
    bridge.verify_bindings(p['bindings']);require(sha(out/'report.json')==expected and sha(r['prepared'])==r['prepare_sha256'],
        'final saved source/report/prepare seals')
    return dict(schema=NAME+'.consume',status='actual_midpoint_suffix4_verified_not_admitted',rows=rows,
        report_sha256=expected,terminal_report_sha256=p['arguments']['terminal_report_sha'],
        counts=r['counts'],fixed_denominator=4,known_roots=2,fresh_roots=0,total_FLOPs=None,
        inherited_resources=p['inherited_resources'],analysis_protocol_sha256=PROTOCOL_SHA,
        consume_new_neural_calls=0,consume_new_physics=0,
        P1_admission=False,scientific_admission=False,freshness=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','terminal-output','terminal-report-sha','terminal-receipt',
        'terminal-receipt-sha','report-sha','receipt'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build(a.terminal_output,a.terminal_report_sha,a.terminal_receipt,a.terminal_receipt_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN HOST CUDA flag required');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:require(a.receipt,'new own receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())

