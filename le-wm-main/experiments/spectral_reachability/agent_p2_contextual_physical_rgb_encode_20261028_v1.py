"""Eight contextual plans: true RGB checkpoints and same-suffix mid feedback.

MAIN HOST CUDA only. No physics, selector, recompilation, fitting or new roots.
prepare/verify/consume are saved arithmetic; missing checkpoints remain null.
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
from experiments.spectral_reachability import agent_p2_contextual_composition_physical_handoff_20261028_v1 as physical
from experiments.spectral_reachability import p1_moving_overlay_recompile_rgb_cuda_20261027_v1 as rgb
from experiments.spectral_reachability import agent_p1_teacher_library40_encode_20261028_v2 as pixels
from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as native
from experiments.spectral_reachability import agent_p1_teacher_library40_api_equivalence_20261028_v1 as api

NAME='agent_p2_contextual_physical_rgb_encode_20261028_v1'
AUTH='MAIN_CONTEXTUAL_PHYSICAL_RGB_AND_FIXED_SUFFIX_HOST_CUDA_ONLY'
ROOT=physical.ROOT
require,sha,canonical=physical.require,physical.sha,physical.raw
CAP,FINAL,FLOOR,WALL=64*1024**2,8*1024**2,200*1024**2,600
META_CAP=8*1024**2
PHYSICAL_SHA='84d688824a1cb2b51b1c5b9d28cfe8bcc66ea5393b62fec2c1a8d101435136a5'
KINDS=('encoder','projector','action_encoder','predict_API','predictor','pred_proj')


def contract():
    return dict(fixed_denominator=8,checkpoints=[0,5,10],known_roots=2,fresh_roots=0,
        max_calls=dict(encoder=24,projector=24,action_encoder=8,predict_API=8,predictor=8,pred_proj=8),
        no_dedup=True,no_padding=True,retries=0,goal='original saved native192 identity Q, never re-encode/replace goal',
        actual_mid_context='original same-condition warm5 latent + newly encoded actual checkpoint0 + actual checkpoint5',
        actual_mid_actions='actual cp5 incoming raw10, append SAME originally selected suffix5; one native predict endpoint',
        semantic='native prediction3 temporal rows are NOT shifted history3; compare last endpoints only',
        raw='original float32 subtract/divide once; no extra /100, clipping or tanh',
        new_physics=0,new_optimizer=0,new_AD_requested=0,hidden_backward_engine_calls=None,training=False,Qfit=False,
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',batch=1,
        latent_dimension=192,encoder_hidden=[1,257,192],
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,wall_seconds=WALL,
        total_FLOPs=None,equal_FLOPs=False,equal_total_compute=False,
        P1_admission=False,P2_admission=False,physical_release=False,scientific_pass=False)


def same(a,b,message):
    a=np.asarray(a);b=np.asarray(b)
    require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),message)


def latent(value):
    a=np.asarray(value,np.float32)
    require(a.shape==(192,) and np.isfinite(a).all() and np.array_equal(a.astype(float),np.asarray(value)),
        'lossless finite native192 float32')
    return a


def mse(a,b):return float(np.mean((np.asarray(a,np.float64)-np.asarray(b,np.float64))**2))


def verify_bindings(bindings):
    for path,h in bindings.items():
        require(Path(path).is_file() and not Path(path).is_symlink() and sha(path)==h,'full input/model/source seal '+path)


def source_files():
    return sorted(set(physical.source_files()+native.source_files()+pixels.own_files()+api.sources()+
        [Path(__file__).resolve(),Path(rgb.__file__),Path(pixels.__file__),Path(api.__file__),
         ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]))


def fixed_rows(prep,strict,planner):
    """Pure source join; no physical labels can change a plan or suffix."""
    require(strict['schema']==physical.NAME+'.consume' and strict['status']=='engineering_verified_science_blocked'
        and strict['fixed_denominator']==8 and strict['known_roots']==2 and len(strict['rows'])==8
        and strict['P1_admission'] is strict['P2_admission'] is strict['physical_release'] is False,'actual full strict physical authority')
    require(len(prep['plans'])==8 and len(planner)==4,'all8 plans/four scientific slots')
    rows=[]
    for i,(plan,actual) in enumerate(zip(prep['plans'],strict['rows'],strict=True)):
        slot=i//2;result=planner[slot]['results'][i%2];contexts=result['winner']['segmented_contexts']
        require(len(contexts)==3,'original two-packet replay has root/mid/terminal contexts')
        key=(*physical.saved.producer.ORDER[slot],physical.ARMS[i%2])
        require((plan['root_slot'],plan['condition'],plan['arm'])==key and
            (actual['index'],actual['root_slot'],actual['condition'],actual['arm'])==(i,*key) and
            actual['selection_digest']==plan['selection_digest'] and actual['compiler_selection_digest']==plan['compiler_selection_digest'],
            'exact selected original full10 / physical slot identity')
        require(result['arm']==plan['arm'] and result['winner']['witness']==plan['witness'] and contexts[0]==plan['root_context']
            and contexts[-1]==plan['witness']['receipt']['terminal_context'],'strict original imagined replay context, not observed teacher15')
        require(physical.bits(plan['commands'])==plan['raw_float32_bits'] and
            result['winner']['cost']==plan['predicted_Q'],'unchanged selected full10 and original Q')
        cp={x['step']:deepcopy(x) for x in actual['rgb_encoding_inputs']}
        require(len(cp)==len(actual['rgb_encoding_inputs']) and sorted(cp)==[x for x in (0,5,10) if x not in actual['missing_checkpoints']],
            'all available/missing checkpoint slots explicit')
        for step,f in cp.items():
            require(f['condition']==plan['condition'] and f['task_id']==plan['task_id'] and
                f['compiler_selection_digest']==plan['compiler_selection_digest'] and
                f['original_task_file']==plan['original_task_file'] and f['original_task_sha256']==plan['original_task_sha256']
                and f['original_blind_path']==plan['blind_path'] and f['original_blind_sha256']==plan['blind_sha256']
                and f['encoder_identity']==plan['encoder_identity'] and f['model_bound_by_prepare'] is True
                and f['encoding_performed'] is False,'exact condition/action/model/task frame join')
            require(Path(f['path']).is_absolute() and Path(f['path']).name==f['png'],'actual original PNG absolute path')
            history=np.concatenate((physical.kernel.moving.warmup_commands(),np.asarray(plan['commands'],np.float32)[:step]))[-10:]
            require(physical.bits(f['raw_action_history10'])==physical.bits(history.tolist())==f['raw_action_history_bits'],
                'checkpoint raw10 causally equals warm10 + this actual executed prefix')
        goal=latent(plan['goal']);im=latent(contexts[1]['latent_history'][-1]);end=latent(contexts[2]['latent_history'][-1])
        require(mse(end,goal)==plan['predicted_Q'],'original identity-Q exact arithmetic')
        frames=[dict(step=step,source=cp.get(step),status='not_started' if step in cp else 'missing_physical_checkpoint',
            latent=None,Q=None) for step in (0,5,10)]
        rows.append(dict(index=i,root_slot=key[0],condition=key[1],arm=key[2],task_id=plan['task_id'],
            selection_digest=plan['selection_digest'],compiler_selection_digest=plan['compiler_selection_digest'],
            full10=deepcopy(plan['commands']),raw_float32_bits=plan['raw_float32_bits'],suffix_bits=bytes.fromhex(plan['raw_float32_bits'])[40:].hex(),
            root_context=deepcopy(contexts[0]),imagined_mid_context=deepcopy(contexts[1]),imagined_terminal_context=deepcopy(contexts[2]),
            imagined_mid192=im.tolist(),imagined_terminal192=end.tolist(),goal192=goal.tolist(),original_predicted_Q=plan['predicted_Q'],
            encoder_identity=deepcopy(plan['encoder_identity']),frames=frames,physical_status=actual['status'],
            actual_native_metrics=deepcopy(actual['actual_metrics']),initial_native_metrics=deepcopy(actual['initial_metrics']),
            final_native_metrics=deepcopy(actual['final_metrics']),full_horizon=actual['full_horizon'],
            missing_checkpoints=deepcopy(actual['missing_checkpoints']),predict_eligible=0 in cp and 5 in cp))
    return rows


def build(physical_output,physical_report_sha,physical_receipt,physical_receipt_sha):
    require(sha(physical.__file__)==PHYSICAL_SHA,'frozen handoff source')
    output=Path(physical_output).resolve();receipt=Path(physical_receipt).resolve()
    physical.kernel.hash_value(physical_report_sha);physical.kernel.hash_value(physical_receipt_sha)
    require(sha(receipt)==physical_receipt_sha,'external actual physical consume receipt SHA')
    accepted=physical.read(receipt);strict=physical.consume(output,physical_report_sha)
    require(canonical(accepted)==canonical(strict),'entire actual physical strict consumer reconstructed, not hash/status alone')
    r=physical.read(output/'report.json');b=r['execution_binding'];p,_=physical.verify(b['prepared'],b['prepare_sha256'])
    planner=[physical.saved.overlay.read(Path(p['planner_output'])/f'slot{i}_result.json.gz',physical.saved.EXPANDED) for i in range(4)]
    rows=fixed_rows(p,strict,planner);bindings={};physical.merge(bindings,p['bindings'])
    physical.merge(bindings,{str(output/'report.json'):physical_report_sha,str(receipt):physical_receipt_sha,
        b['prepared']:b['prepare_sha256']})
    for file in output.iterdir():physical.merge(bindings,{str(file):sha(file)})
    for row in rows:
        for f in row['frames']:
            if f['source'] is not None:load_pixels(f['source'])
    physical.merge(bindings,rgb.MODELS)
    physical.merge(bindings,{str(f.resolve()):sha(f) for f in source_files()})
    shapes=rgb.config_shapes(rgb.read(rgb.POLICY/'config.json'))
    require(all(row['encoder_identity']['model_sha256']==rgb.MODELS[str(rgb.POLICY/'weights.pt')] and
        row['encoder_identity']['model_config_sha256']==rgb.MODELS[str(rgb.POLICY/'config.json')] for row in rows),
        'same original frozen model/config, not alternate representation')
    counts={k:sum(f['source'] is not None for row in rows for f in row['frames']) if k in ('encoder','projector')
        else sum(row['predict_eligible'] for row in rows) for k in KINDS}
    verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=dict(physical_output=str(output),physical_report_sha=physical_report_sha,
        physical_receipt=str(receipt),physical_receipt_sha=physical_receipt_sha),rows=rows,bindings=bindings,
        expected_calls=counts,shapes=shapes,policy=str(rgb.POLICY),expected_model_hash=rgb.MODEL_TENSOR_SHA,
        inherited_compiler_resources=p['compiler_resources'],inherited_physical_counts=strict['counts'])


def load_pixels(frame):
    item=dict(path=frame['path'],file_sha256=frame['file_sha256'],pixel_sha256=frame['pixel_sha256'],kind='checkpoint')
    return rgb.load_image(item)[0]


def write_metadata(path,value):
    path=Path(path).absolute();b=canonical(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(b)<=META_CAP
        and shutil.disk_usage(path.parent).free>=FLOOR+len(b),'exclusive bounded own metadata')
    with path.open('xb') as f:require(f.write(b)==len(b),'short metadata');f.flush();os.fsync(f.fileno())
    rgb.syncdir(path.parent)
    return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared_no_execution'))


def prepare(output,**arguments):return write_metadata(output,build(**arguments))


def verify(path,expected):
    require(sha(path)==expected,'external own prepare SHA');p=physical.read(path)
    require(canonical(p)==canonical(build(**p['arguments'])) and sha(path)==expected,'complete actual saved preparation reconstruction')
    return p


class Budget(rgb.Budget):
    def check(self,size=0,final=False):
        require(final or self.clock()-self.started<=WALL,'600s whole job; no retry')
        require(self.used+size<=CAP-(0 if final else FINAL) and self.free()>=FLOOR+size,'64MiB cap/8MiB final reserve/200MiB floor')


class Store(rgb.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_') and shutil.disk_usage(Path(output).parent).free>=CAP+FLOOR,'own exclusive full-cap output')
        super().__init__(output,budget);self.index=0
    def event(self,e):
        self.put(f'event_{self.index:04}.json',e);self.index+=1


def mid_inputs(row,z0,z5):
    """Only RGB native latents and fixed raw actions; no state/native metrics."""
    z0=latent(z0);z5=latent(z5);root=row['root_context']
    context=np.asarray([root['latent_history'][1],z0,z5],np.float32)[None]
    frame=row['frames'][1]['source'];require(frame is not None,'real checkpoint5 required')
    history=np.asarray(frame['raw_action_history10'],np.float32)
    commands=np.asarray(row['full10'],np.float32)[5:]
    require(commands.tobytes().hex()==row['suffix_bits'],'same originally selected suffix5 bits')
    token=physical.saved.frontier._normalized_token
    outgoing=np.asarray([token(history[:5]),token(history[5:]),token(commands)],np.float32)[None]
    return dict(context=context,outgoing=outgoing)


def process(p,store,encode,predict,rows,check=lambda:None):
    require(len(rows)==8,'all eight original plan denominator')
    for row in rows:
        row['feedback']=dict(status='not_started' if row['predict_eligible'] else 'missing_mid_inputs',prediction=None)
    for spec,row in zip(p['rows'],rows,strict=True):
        for f in row['frames']:
            if f['source'] is None:continue
            check();scope=dict(row=row['index'],step=f['step'],phase='encode')
            f['status']='attempted';store.event(dict(kind='image_attempt',scope=scope))
            pix=load_pixels(f['source']);f['pixels']=store.array(f'r{row["index"]}_s{f["step"]}_pixels',pix)
            z=encode(pix,f,scope);latent(z)
            f['latent']=store.array(f'r{row["index"]}_s{f["step"]}_latent',z);f['status']='encoded'
            store.event(dict(kind='image_return',scope=scope,frame=deepcopy(f)))
    for row in rows:
        if not row['predict_eligible']:continue
        check();scope=dict(row=row['index'],phase='same_suffix_prediction')
        def load(r):return rgb.load_array(r,lambda n:store.files[n] if hasattr(store,'files') else (store.path/n).read_bytes())
        inputs=mid_inputs(row,load(row['frames'][0]['latent']),load(row['frames'][1]['latent']))
        feedback=row['feedback'];feedback['status']='attempted'
        store.event(dict(kind='feedback_attempt',scope=scope))
        feedback['inputs']={k:store.array(f'r{row["index"]}_mid_{k}',v) for k,v in inputs.items()}
        value=predict(inputs,feedback,scope)
        require(value.dtype==np.float32 and value.shape==(1,3,192) and np.isfinite(value).all(),'full native3 predictions, not shifted history')
        feedback['prediction']=store.array(f'r{row["index"]}_same_suffix_native3',value);feedback['status']='predicted'
        store.event(dict(kind='feedback_return',scope=scope,feedback=deepcopy(feedback)))


class Trace(physical.saved.overlay.NativeTrace):
    """Original full native tensor persistence, own fixed24/8 schedule/caps."""
    def __init__(self,model,torch,store,check,expected):
        super().__init__(model,torch,store,check,[1,257,192]);self.expected=expected;self.api_hooks=api.mid.Hooks()
    def __enter__(self):
        def pre(kind):
            def hook(module,args):
                self.check();require(not self.torch.is_grad_enabled() and self.scope is not None and module not in self.pending,'scoped CUDA inference')
                want=['encoder','projector'] if self.scope['phase']=='encode' else ['action_encoder','predict_API','predictor','pred_proj']
                previous=[e['resource'] for e in self.entries if e['scope']==self.scope]
                require(len(previous)<len(want) and kind==want[len(previous)] and
                    sum(e['resource']==kind for e in self.entries)<self.expected[kind],'exact planned module predispatch')
                e=dict(index=len(self.entries),scope=deepcopy(self.scope),resource=kind,completed=False)
                self.entries.append(e);self.pending[module]=e;self.store.event(dict(kind='module_attempt',entry=deepcopy(e)))
                e['inputs']=[self.save(x,f'm{e["index"]}_input{j}') for j,x in enumerate(args)]
            return hook
        def post(module,args,out):
            e=self.pending.pop(module)
            if hasattr(out,'last_hidden_state'):out=out.last_hidden_state
            if e['resource']=='encoder':require(tuple(out.shape)==self.encoder_shape,'exact patch14 hidden257, no197')
            e['output']=self.save(out,f'm{e["index"]}_output');self.torch.cuda.synchronize();self.check();e['completed']=True
            self.store.event(dict(kind='module_return',entry=deepcopy(e)))
        original=self.model.predict
        def counted(*args,**kw):
            require(not kw and len(args)==2,'literal cached predict API')
            pre('predict_API')(self.model,args)
            try:value=original(*args)
            except BaseException:raise
            post(self.model,args,value);return value
        try:
            for kind in ('encoder','projector','action_encoder','predictor','pred_proj'):
                module=getattr(self.model,kind);self.hooks.extend((module.register_forward_pre_hook(pre(kind)),module.register_forward_hook(post)))
            self.api_hooks.install(self.model,'predict',counted)
            return self
        except BaseException as e:self.__exit__(type(e),e,e.__traceback__);raise
    def __exit__(self,typ,exc,tb):
        errors=[];self.api_hooks.restore(errors)
        try:super().__exit__(typ,exc,tb)
        finally:require(not errors,'restore original predict API')


def counts(entries):
    return {k:dict(attempted=sum(e['resource']==k for e in entries),completed=sum(e['resource']==k and e['completed'] for e in entries)) for k in KINDS}


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN authority before prepare/model/output')
    budget=Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=None;model=None;before=after=host=None;source_ok=False
    store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        native.frozen(model);before=native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model tensor SHA')
        trace=Trace(model,torch,store,store.check,p['expected_calls']);transform=p0.image_transform()
        def encode(pix,f,scope):
            with physical.saved.overlay.traced_scope(trace,scope,f):
                x=p0._image_tensor(pix,transform=transform,device=device)
                f['preprocessed']=trace.save(x,f'r{scope["row"]}_s{scope["step"]}_preprocessed')
                same(x.detach().cpu().numpy(),pixels.preprocess_pixels(pix),'exact original CUDA preprocessing')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'native emb192')
                return out[0,0].detach().cpu().numpy().copy()
        def predict(inputs,feedback,scope):
            with physical.saved.overlay.traced_scope(trace,scope,feedback):
                z=torch.tensor(inputs['context'],dtype=torch.float32,device=device)
                tokens=torch.tensor(inputs['outgoing'],dtype=torch.float32,device=device)
                return model.predict(z,model.action_encoder(tokens)).detach().cpu().numpy().copy()
        with trace,torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            process(p,store,encode,predict,rows,store.check)
    except BaseException as e:errors.append(dict(stage='execution',type=type(e).__name__,message=str(e)))
    finally:
        if model is not None:
            try:native.frozen(model);after=native.tensor_hash(model)
            except BaseException as e:errors.append(dict(stage='final_model',message=str(e)))
        try:verify_bindings(p['bindings']);require(sha(prepared)==expected,'unchanged prepare');budget.check();source_ok=True
        except BaseException as e:errors.append(dict(stage='final_inputs_or_limits',message=str(e)))
        entries=[] if trace is None else trace.entries
        complete=not errors and source_ok and before==after==p['expected_model_hash'] and counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()}
        seal=dict(rows=rows,modules=entries,counts=counts(entries),errors=errors)
        seal_sha=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='encoded_and_same_suffix_predicted_science_blocked' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=contract(),rows=rows,modules=entries,counts=counts(entries),
            errors=errors,model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,host=host,native_seal_sha256=seal_sha,artifact_sha256=deepcopy(store.hashes),
            elapsed_seconds=time.monotonic()-budget.started,event_count=store.index,P1_admission=False,P2_admission=False,physical_release=False)
        store.put('report.json',report,final=True)
    require(complete,'partial/failed output retained; no retries: '+str(errors))
    return dict(status=report['status'],report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def native_chain(mods,inputs,value,load):
    require([m['resource'] for m in mods]==['action_encoder','predict_API','predictor','pred_proj'],'one actual cached suffix API chain')
    ae,api_,pr,pp=mods
    def values(m,n):
        require(len(m['inputs'])==n,'all original native input tensors')
        return [load(x) for x in m['inputs']]+[load(m['output'])]
    ai,ao=values(ae,1);zi,act,zo=values(api_,2);pi,pa,po=values(pr,2);qi,qo=values(pp,1)
    require(ai.shape==(1,3,10) and ao.shape==zi.shape==act.shape==zo.shape==pi.shape==pa.shape==po.shape==(1,3,192)
        and qi.shape==qo.shape==(3,192) and value.shape==(1,3,192),'full native cached shapes, not shifted context')
    same(ai,inputs['outgoing'],'actual normalized raw packets to AE');same(zi,inputs['context'],'actual mid full3 to native predict')
    same(act,ao,'AE to predict');same(pi,zi,'predict to predictor latent');same(pa,ao,'predict to predictor action')
    same(qi,po[0],'predictor to pred_proj');same(zo,qo[None],'projected native to predict API');same(value,zo,'full native prediction return')


def summarize(spec,row,load):
    zs={f['step']:load(f['latent']) if f['latent'] is not None else None for f in row['frames']}
    feedback=row['feedback'];out=load(feedback['prediction']) if feedback['prediction'] is not None else None
    q=spec['goal192'];actual=zs[10]
    return dict(index=spec['index'],root_slot=spec['root_slot'],condition=spec['condition'],arm=spec['arm'],
        selection_digest=spec['selection_digest'],compiler_selection_digest=spec['compiler_selection_digest'],
        original_predicted_Q=spec['original_predicted_Q'],
        original_root_reencode_MSE=mse(zs[0],spec['root_context']['latent_history'][-1]) if zs[0] is not None else None,
        imagined_prefix_MSE=mse(spec['imagined_mid192'],zs[5]) if zs[5] is not None else None,
        imagined_terminal_MSE=mse(spec['imagined_terminal192'],actual) if actual is not None else None,
        real_mid_same_suffix_MSE=mse(out[0,-1],actual) if out is not None and actual is not None else None,
        real_mid_same_suffix_predicted_Q=mse(out[0,-1],q) if out is not None else None,
        imagined_Q_error=spec['original_predicted_Q']-mse(actual,q) if actual is not None else None,
        real_mid_same_suffix_Q_error=mse(out[0,-1],q)-mse(actual,q) if out is not None and actual is not None else None,
        checkpoints=[dict(step=s,latent_Q=mse(z,q) if z is not None else None,latent_missing=z is None,
            native=spec['initial_native_metrics'] if s==0 else next((m for m in spec['actual_native_metrics'] if m['step']==s),None)) for s,z in zs.items()],
        final_native_metrics=spec['final_native_metrics'],physical_status=spec['physical_status'],missing_checkpoints=spec['missing_checkpoints'],
        no_control_intervention=True,physical_benefit_claim=False)


def validate(p,r,events,load):
    require(len(r['rows'])==8 and r['contract']==p['contract']==contract() and r['errors']==[],'all8 terminal only')
    modules=[];expected_events=[];outputs=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('frames','feedback')}=={k:v for k,v in spec.items() if k!='frames'},'original full plan/input identity')
        require(len(row['frames'])==3,'all3 checkpoints including missing')
        for original,f in zip(spec['frames'],row['frames'],strict=True):
            require(f['step']==original['step'] and f['source']==original['source'],'same original actual frame identity')
            if original['source'] is None:
                require(f==original,'missing frame untouched, no zero or replacement');continue
            require(f['status']=='encoded','all available frames encoded once')
            pix=load(f['pixels']);same(pix,load_pixels(f['source']),'source PNG pixels to encoding')
            pre=load(f['preprocessed']);same(pre,pixels.preprocess_pixels(pix),'original RGB normalization')
            z=load(f['latent']);latent(z);mods=f['modules'];scope=dict(row=row['index'],step=f['step'],phase='encode')
            require(len(mods)==2 and all(m['scope']==scope for m in mods),'same checkpoint scoped modules')
            # Reuse the native full hidden257 -> CLS -> projector validator.
            en,proj=mods;require([m['resource'] for m in mods]==['encoder','projector'] and len(en['inputs'])==len(proj['inputs'])==1,'exact visual module order')
            physical.saved.overlay.visual_bridge(pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z,[1,257,192])
            modules.extend(mods);expected_events.append(dict(kind='image_attempt',scope=scope))
            for m in mods:expected_events.extend([dict(kind='module_attempt',entry={k:m[k] for k in ('index','scope','resource')}|{'completed':False}),dict(kind='module_return',entry=m)])
            expected_events.append(dict(kind='image_return',scope=scope,frame=f))
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        f=row['feedback'];scope=dict(row=row['index'],phase='same_suffix_prediction')
        if not spec['predict_eligible']:
            require(f==dict(status='missing_mid_inputs',prediction=None),'missing inputs not predicted or padded');continue
        require(f['status']=='predicted','actual same-suffix prediction')
        inputs=mid_inputs(spec,load(row['frames'][0]['latent']),load(row['frames'][1]['latent']))
        require(set(f['inputs'])==set(inputs),'full feedback literal input buffers')
        for k,v in inputs.items():same(load(f['inputs'][k]),v,'original literal feedback input '+k)
        mods=f['modules'];require(all(m['scope']==scope for m in mods),'own prediction scope')
        native_chain(mods,inputs,load(f['prediction']),load);modules.extend(mods)
        ae,api_,pr,pp=mods
        def attempt(m):return dict(kind='module_attempt',entry={k:m[k] for k in ('index','scope','resource')}|{'completed':False})
        def returned(m):return dict(kind='module_return',entry=m)
        expected_events += [dict(kind='feedback_attempt',scope=scope),attempt(ae),returned(ae),attempt(api_),attempt(pr),returned(pr),attempt(pp),returned(pp),returned(api_),dict(kind='feedback_return',scope=scope,feedback=f)]
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules)))
        and all(m['completed'] is True for m in modules) and r['counts']==counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'complete actual per-module attempts/returns versus planned caps, no extra calls')
    require(events==expected_events,'exact actual image->modules->feedback chronology, no skipped bad events')
    return [summarize(spec,row,load) for spec,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external RGB report SHA')
    r=physical.read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='encoded_and_same_suffix_predicted_science_blocked'
        and r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False and r['elapsed_seconds']<=WALL,'actual terminal CUDA frozen report only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.iterdir())<=CAP,'entire output census')
    def blob(n):return rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'all full array/artifact SHAs')
    require(physical.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution prepare binding')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and physical.read(output/'native_seal.json')==
        dict(rows=r['rows'],modules=r['modules'],counts=r['counts'],errors=r['errors']),'native arrays sealed before saved analysis')
    events=[physical.read(output/f'event_{i:04}.json') for i in range(r['event_count'])]
    rows=validate(p,r,events,lambda receipt:rgb.load_array(receipt,blob))
    verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final input closure unchanged')
    return dict(schema=NAME+'.consume',status='actual_RGB_and_same_suffix_saved_verified_NOT_science',report_sha256=expected,
        rows=rows,counts=r['counts'],fixed_denominator=8,known_roots=2,fresh_roots=0,
        inherited_physical_counts=p['inherited_physical_counts'],inherited_compiler_resources=p['inherited_compiler_resources'],
        total_FLOPs=None,P1_admission=False,P2_admission=False,physical_release=False,new_neural_calls=0,new_physics=0)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-output','physical-report-sha','physical-receipt','physical-receipt-sha','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=prepare(a.output,physical_output=a.physical_output,physical_report_sha=a.physical_report_sha,
        physical_receipt=a.physical_receipt,physical_receipt_sha=a.physical_receipt_sha)
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',plans=8,expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN explicit CUDA authorization');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'exclusive saved consume receipt path required');v=write_metadata(a.receipt,consume(a.output,a.report_sha))
    print(canonical(v).decode(),end='',flush=True)


if __name__=='__main__':main()
