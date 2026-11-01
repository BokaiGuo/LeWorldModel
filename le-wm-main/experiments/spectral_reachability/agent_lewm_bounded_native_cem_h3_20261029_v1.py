"""New logit-Gaussian CEM/tanh-raw P2-known2 H3; not official/equal-budget/Hi scaler."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import numpy as np
from experiments.spectral_reachability import agent_lewm_finalmean_paid_evaluator_20261029_v1 as paid
from experiments.spectral_reachability import agent_lewm_bounded_native_cem_20261029_v1 as h2
from experiments.spectral_reachability import agent_p2_grasp_raw15_prepare_20261029_v1 as domain

old=paid.old
NAME='agent_lewm_bounded_native_cem_h3_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'innerdocs/agent_lewm_bounded_cem_horizon_comparison_protocol_20261029_v1.md'
PROTOCOL_SHA='354b5e8882989ba1e870a54fde32ad28725582b0ac46f184817b40434f1a6f84'
CAP,RESERVE,FLOOR,WALL=2*1024**3,128*1024**2,200*1024**2,900
N,I,K,SEED=300,30,30,0
AUTH='MAIN_BOUNDED_NATIVE_CEM_H3_HOSTCUDA_ONLY'
need,raw,read,sha,exact,durable=old.need,old.raw,old.read,old.sha,old.exact,old.durable
P_PREPARE=ROOT/'results/agent_p2_common_resource_grasp_cuda_20261029_v1_prepare_MAIN01.json'
P_SHA='7facba500439172ced280ec0b893659715139b1e5e22073ae7313665a1c0faec'
ENCODING=ROOT/'results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json'
ENCODING_SHA='2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381'
LAYOUTS=('whole','segmented')
KINDS=('action_encoder','predictor','pred_proj')
PINS={**old.PINS,Path(h2.__file__):'d0f249632793fb1ceca98849e7e716b7cd4d382d092a0adc1ba7c8fa99bab47f',
      Path(old.__file__):'87bb5269388ced3f4dc74d41550b4939c5cf5457946c1c1b63d0e272a8c88e26',
      Path(paid.__file__):'5e569c07037f23a2c2b2403a49f0bd090738f7b1da31327bf97657b1c151dd72',
      ROOT/'experiments/spectral_reachability/pusht_d2p_packet_contract.py':'eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef',PROTOCOL:PROTOCOL_SHA}


def resources():
    return dict(cost_oracle_API=2*I,model_get_cost_API=0,rollout_API=2*I+2,segmented_predict_API=6,
        criterion_API=2*I+4,action_encoder=2*I+8,AE_samples=2*I*N+8,AE_tokens=10*I*N+28,
        predictor=6*I+12,predictor_samples=6*I*N+12,predictor_context_tokens=18*I*N+36,
        pred_proj=6*I+12,pred_proj_rows=18*I*N+36,tanh_API=2*I+2,normalize_API=2*I+2,
        encoder=0,projector=0,AD=0,physics=0,training=0)


def whole_schedule(n):
    return [('action_encoder',[(n,5,10)],(n,5,192))]+[
        entry for _ in range(3) for entry in (
            ('predictor',[(n,3,192),(n,3,192)],(n,3,192)),
            ('pred_proj',[(n*3,192)],(n*3,192)))]


def delivery_schedule():
    return whole_schedule(1)+[
        entry for _ in range(3) for entry in (
            ('action_encoder',[(1,3,10)],(1,3,192)),
            ('predictor',[(1,3,192),(1,3,192)],(1,3,192)),
            ('pred_proj',[(3,192)],(3,192)))]


def shape_census(n=300,iterations=30,k=30):
    """Pure integer/shape proof: every Store.array call including duplicate captures."""
    base=[((1,3,3,224,224),4),((1,1,3,224,224),4),((1,3,192),4),
          ((1,1,192),4),((1,1,2,10),4)]
    plan=[((1,n,3,10),4)]*4+[
        ((1,n,5,10),4),((1,n,6,192),4),((1,n,3,10),4),((1,n),4)]
    callback=[((1,n,3,10),4),((1,n),4),((1,k),4),((1,k),8),((1,k,3,10),4)]+[((1,3,10),4)]*4
    def module_arrays(schedule):
        return [(shape,4) for _,args,out in schedule for shape in [*args,out]]
    final=[((1,3,10),4)]*4+[((15,2),4)]*2+[((1,3,5,2),4),((1,3,10),4)]
    inputs=base+[((1,3,10),4)]
    layouts=[((1,1,3,10),4),((1,1,6,192),4)]+[((1,3,192),4)]*3+[((1,1,6,192),4)]
    shapes=2*(base+iterations*(plan+callback+module_arrays(whole_schedule(n)))+
              final+inputs+module_arrays(delivery_schedule())+layouts+[((1,1),4)]*2)
    payload=sum(int(np.prod(shape,dtype=object))*size for shape,size in shapes)
    # All NPY headers <=256B (actual writer tested); each event <=2048B by schema.
    # 64KiB/array additionally covers report refs, inventory, scalar metadata.
    working=payload+len(shapes)*(256+65536)+len(expected_events('a'*64,'complete'))*2048
    return dict(array_count=len(shapes),payload_bytes=payload,largest_array_bytes=max(
        int(np.prod(shape,dtype=object))*size for shape,size in shapes),
        metadata_allowance_per_array=65536,npy_header_cap=256,event_bytes_cap=2048,
        working_upper_bytes=working,working_cap=CAP-RESERVE,total_upper_bytes=working+RESERVE,
        cap=CAP,full_precision=True)


def module_totals(rows):
    result={k:dict(attempted=0,returned=0,samples_attempted=0,samples_returned=0,
        tokens_attempted=0,tokens_returned=0) for k in KINDS}
    for row in rows:
        for phase in ('planning','delivery'):
            for e in row[phase]['modules']:
                value=result[e['kind']];value['attempted']+=1;value['returned']+=int(e['returned'])
                shape=e['inputs'][0]['shape'] if e['inputs'] else None
                for metric in ('samples','tokens'):
                    count=None if shape is None else (shape[0] if metric=='samples' or len(shape)==2 else shape[0]*shape[1])
                    for suffix,entered in (('attempted',True),('returned',e['returned'])):
                        if entered:
                            key=metric+'_'+suffix
                            value[key]=None if count is None or value[key] is None else value[key]+count
    return result


def contract():
    return dict(roots=[0,1],condition='off',known_only=True,fresh=False,N=N,iterations=I,topk=K,seed=SEED,
        seed_authority='MAIN prospective seed0; installed default remains1234',horizon=3,packet_raw_steps=5,
        method='new logit-Gaussian CEM with exactly one tanh to raw, then P2 normalization once',
        optimizer='pinned installed CEMSolver; mean/std ddof1 in logit space',
        raw_domain=[-1.,1.],initial_logit_mean=0.,initial_logit_std=1.,mapping='CUDA float32 tanh once; no clamp/ball/extra division',
        final_action='tanh(final logit elite mean), NOT raw elite mean or sampled winner',
        final_evaluation='seal raw15; reread exact file bytes; normalize; independent whole3/segmented1+1+1',
        seal='each candidate batch before planning criterion/update; all four final predictions before final criteria',
        criterion='official float32 terminal sum192; saved float64 sum192/mean192 separately',
        resources=resources(),output_cap=CAP,final_reserve=RESERVE,floor=FLOOR,wall_seconds=WALL,
        GPU_allocated=3*1024**3,GPU_reserved=4*1024**3,requested_SDP='MATH',actual_kernel_certificate=False,
        no_CPU_neural=True,no_retries=True,official_reproduction=False,equal_budget=False,equal_FLOPs=False,
        same_horizon_as_P2_common_H3=True,Hi_training_scaler_certified=False,training_provenance_verified=False,
        same_support_as_P2=False,same_planning_reduction_as_P2=False,fair_ranking=False,
        physical_release=False,scientific_admission=False)


def build():
    need((N,I,K,SEED)==(300,30,30,0),'fixed prospective production parameters')
    need(all(sha(f)==h for f,h in PINS.items()),'approved H3 protocol and frozen helpers')
    census=shape_census()
    need(census['working_upper_bytes']<CAP-RESERVE and census['largest_array_bytes']<16*1024**2,'complete full-precision shape census')
    p=read(P_PREPARE,P_SHA)
    need(p['schema']=='agent_p2_common_resource_grasp_cuda_20261029_v1.prepare' and
         [(x['root_slot'],x['condition']) for x in p['slots']]==[(0,'off'),(1,'off')],'exact P2 original two off task slots')
    slots=[]
    for slot in p['slots']:
        t=domain.parse(domain.read_bound(slot['task_file'],slot['task_sha256'],compressed=True))
        task=t['task'];need(t['task_id']==domain.core.task_id(task)==slot['task_id'],'original P2 task identity')
        need(raw(task['root_context'])==raw(slot['packet']['root_context']) and
             raw(task['goal'])==raw(slot['packet']['goal']) and raw(task['projection'])==raw(slot['q']) and
             raw(slot['q'])==raw(np.eye(192).tolist()) and raw(task['root_raw_action_history'])==raw(slot['warm_raw10']),'original goal/Q/context/raw history')
        norm=domain.normalization()
        need(task['action_mean']==norm['mean'] and task['action_std']==norm['std'] and
             slot['task_sha256']==domain.TASK_SHAS[slot['root_slot']],'original P2 engineering normalizer and task')
        blind=old.inputs.moving.load_blind(slot['blind_path'],slot['blind_sha256'])
        pixel_sha=[hashlib.sha256(x.tobytes()).hexdigest() for x in [*blind['rgb_history'],blind['goal_rgb']]]
        slots.append(dict(slot,source_slot=slot['root_slot'],blind_npz=slot['blind_path'],pixel_sha256=pixel_sha,
            emb=slot['packet']['root_context']['latent_history'],warm=slot['packet']['root_context']['action_history'],
            goal_emb=slot['packet']['goal'],mean=norm['mean'],std=norm['std']))
    enc=read(ENCODING,ENCODING_SHA)
    inherited={kind:dict(attempted=sum(e['resource']==kind for e in enc['module_ledger']),
        returned=sum(e['resource']==kind and e['completed'] is True for e in enc['module_ledger'])) for kind in ('encoder','projector')}
    need(all(v==dict(attempted=16,returned=16) for v in inherited.values()),'original four-task encoding16/16, not new calls')
    inherited.update(report=str(ENCODING),report_sha256=ENCODING_SHA,scope='four off/on tasks once; off-root attribution unknown',
        off_root_attribution=None,training_provenance_verified=False)
    sources=dict(p['bindings_sha256'])
    for f,h in PINS.items():
        need(str(f) not in sources or sources[str(f)]==h,'consistent frozen primitive binding')
        sources[str(f)]=h
    sources.update({str(P_PREPARE):P_SHA,str(ENCODING):ENCODING_SHA})
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),Path(domain.__file__)):
        h=sha(f);need(str(f) not in sources or sources[str(f)]==h,'no source authority override');sources[str(f)]=h
    need(all(sha(f)==h for f,h in sources.items()),'original task/model/scaler/source closure')
    return dict(schema=NAME+'.prepare',contract=contract(),slots=slots,sources=sources,policy=p['policy'],
        expected_model_hash=p['expected_model_hash'],inherited=inherited,protocol_sha256=PROTOCOL_SHA,
        P_input_authority=dict(path=str(P_PREPARE),sha256=P_SHA),shape_census=census,
        trust='bound original P prepare/task/encoding files; no recursive producer fullstrict')


def source_arrays(slot):
    blind=old.inputs.moving.load_blind(slot['blind_npz'],slot['blind_sha256'])
    need([old.inputs.moving.digest(x.tobytes()) for x in [*blind['rgb_history'],blind['goal_rgb']]]==slot['pixel_sha256'],'actual RGB source identity')
    return dict(pixels=np.stack([old.transformed_rgb(x) for x in blind['rgb_history']])[None],
        goal=old.transformed_rgb(blind['goal_rgb'])[None,None],emb=np.asarray(slot['emb'],np.float32)[None],
        goal_emb=np.asarray(slot['goal_emb'],np.float32)[None,None],warm=np.asarray(slot['warm'],np.float32)[None,None])


def own(path):
    p=Path(path).absolute();need(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and
        not any(x.is_symlink() for x in (p,*p.parents)),'exclusive new stem direct result child');return p


def prepare(output):
    p=own(output);v=build();b=raw(v);need(len(b)<=2*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'prepare bounds')
    durable(p,b);return dict(path=str(p),sha256=sha(p))


def verify(prepared,h):
    p=read(prepared,h);need(raw(p)==raw(build()),'complete saved identity closure, not fullstrict');return p


class Store(old.Store):
    def __init__(self,path,started=None):
        self.path=own(path);self.started=time.monotonic() if started is None else started
        need(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive2GiB+floor')
        self.path.mkdir();self.inventory={};self.events=[];self.total=0;self.tail=None;self.extra_guard=lambda:None
    def guard(self,size=0,final=False):
        need(final or time.monotonic()-self.started<WALL,'cooperative whole900s')
        need(self.total+size<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+size,'2GiB/128reserve/200floor')
        if not final:self.extra_guard()


class Trace:
    """New H3 schedules; no fixed-H2 hook or module/global replacement."""
    def __init__(self,model,store,row,planning=False):
        self.model,self.store,self.row,self.planning=model,store,row,planning
        self.handles=[];self.pending={}
    def __enter__(self):
        try:
            for kind in (*KINDS,'encoder','projector'):
                def pre(module,args,k=kind):
                    need(k in KINDS,'no new encoder/projector')
                    import torch
                    need(torch.is_inference_mode_enabled() and not torch.is_grad_enabled() and not torch.is_autocast_enabled('cuda') and
                        not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'observed inference/TF32/autocast')
                    need(all(x.device.type=='cuda' and x.dtype==torch.float32 and not x.requires_grad for x in args),'native CUDA float32')
                    index=len(self.row['modules']);schedule=whole_schedule(N) if self.planning else delivery_schedule()
                    pos=index%7 if self.planning else index
                    need(pos<len(schedule),'no excess H3 module calls')
                    expected,shapes,_=schedule[pos]
                    need(k==expected and [tuple(x.shape) for x in args]==shapes and k not in self.pending,'H3 module schedule/input')
                    if self.planning:
                        j=self.row['cost_attempted']-1
                        need(index//7==j and j<I,'planning actual iteration')
                        phase=dict(iteration=j)
                    else:
                        layout='whole' if index<7 else 'segmented'
                        need(self.row['active_layout']==layout,'final layout dispatch order')
                        phase=dict(layout=layout)
                    self.store.guard()
                    e=dict(index=index,kind=k,returned=False,inputs=[],**phase)
                    self.row['modules'].append(e);self.pending[k]=(e,schedule[pos][2])
                    self.store.event('module_intent',root=self.row['root'],index=index,kind_name=k)
                    e['inputs']=[self.store.array(x) for x in args]
                def post(module,args,out,k=kind):
                    import torch
                    need(k in self.pending,'return has actual intent')
                    e,shape=self.pending[k]
                    need(out.device.type=='cuda' and out.dtype==torch.float32 and not out.requires_grad and tuple(out.shape)==shape,'H3 native output')
                    e['output']=self.store.array(out);e['returned']=True;del self.pending[k]
                    self.store.event('module_return',root=self.row['root'],index=e['index'],kind_name=k)
                self.handles.extend([getattr(self.model,kind).register_forward_pre_hook(pre),
                    getattr(self.model,kind).register_forward_hook(post)])
        except BaseException:
            for handle in self.handles:handle.remove()
            raise
        return self
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        if args[0] is None:
            need(not self.pending and len(self.row['modules'])==(I*7 if self.planning else 16),'complete H3 module denominator')


def native_map(logits,store,root,iteration,final=False):
    import torch
    expected=(1,3,10) if final else (1,N,3,10)
    need(tuple(logits.shape)==expected and logits.dtype==torch.float32 and logits.device.type=='cuda','actual CUDA logit inputs')
    e=dict(returned=False,logits=store.array(logits));store.event('mapping_intent',root=root,iteration=iteration,final_mapping=final)
    mapped=torch.tanh(logits)  # the only tanh call; final raw later reread, never remapped
    need(bool(torch.isfinite(mapped).all()) and bool((mapped.abs()<=1).all()),'strict actual raw domain')
    out=mapped.reshape(15,2) if final else mapped.reshape(1,N,3,5,2)
    e['raw']=store.array(out);e['returned']=True;store.event('mapping_return',root=root,iteration=iteration,final_mapping=final)
    return out,e


def native_normalize(commands,slot,store,root,iteration):
    from experiments.spectral_reachability.pusht_d2p_packet_contract import normalize_live_packet
    e=dict(returned=False,raw=store.array(commands));store.event('normalize_intent',root=root,iteration=iteration)
    out=normalize_live_packet(commands,mean=slot['mean'],std=slot['std'])
    e['tokens']=store.array(out);e['returned']=True;store.event('normalize_return',root=root,iteration=iteration)
    return out,e


def run(prepared,h,output,authorization):
    need(authorization==AUTH and not os.environ.get('PYTHONPATH'),'MAIN CUDA-only authority, no import overlay')
    started=time.monotonic();p=verify(prepared,h);store=Store(output,started)
    rows=[dict(root=i,task_id=p['slots'][i]['task_id'],task_sha256=p['slots'][i]['task_sha256'],
        planning=dict(root=i,status='not_started',cost_attempted=0,cost_returned=0,costs=[],callbacks=[],modules=[]),
        delivery=dict(root=i,status='not_started',modules=[],layouts={k:dict(attempted=False,returned=False) for k in LAYOUTS},
            scores={k:dict(attempted=False,returned=False) for k in LAYOUTS})) for i in (0,1)]
    r=dict(schema=NAME+'.report',prepared=str(Path(prepared).absolute()),prepare_sha256=h,contract=contract(),rows=rows,
        inherited=p['inherited'],model_before=None,model_after=None,errors=[],source_verified=False,all_final_predictions_sealed=False,
        physical_release=False,scientific_admission=False);model=None
    try:
        store.event('started',prepare_sha256=h)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        need(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','fixed requested CUBLAS')
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from gymnasium.spaces import Box
        from stable_worldmodel.solver.cem import CEMSolver
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        r['CUDA']=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        need(Path(inspect.getfile(CEMSolver)).resolve()==old.CEM,'installed unchanged CEM class')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        need(Path(inspect.getfile(type(model))).resolve()==old.WM,'actual official LeWM class')
        r['model_before']=old.inputs.moving.low.tensor_hash(model);need(r['model_before']==p['expected_model_hash'],'original frozen model')
        store.extra_guard=lambda:need(torch.cuda.memory_allocated()<=3*1024**3 and torch.cuda.memory_reserved()<=4*1024**3,'GPU3/4GiB')
        tensor=lambda a:torch.as_tensor(a,device=device)
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            for slot,row in zip(p['slots'],rows,strict=True):
                plan=row['planning'];a=source_arrays(slot);plan['base_inputs']={k:store.array(v) for k,v in a.items()}
                info={k:tensor(a[k]) for k in ('pixels','emb')};warm=tensor(a['warm']);goal=tensor(a['goal_emb'])[:,None]
                store.event('planning_started',root=row['root']);plan['status']='attempted'
                class Cost:
                    def parameters(self):return model.parameters()
                    def get_cost(self,expanded,candidates):
                        j=plan['cost_attempted'];need(j<I,'no extra oracle call');plan['cost_attempted']+=1
                        store.guard();store.event('cost_intent',root=row['root'],iteration=j)
                        e=dict(iteration=j,returned=False,rollout_returned=False);plan['costs'].append(e)
                        commands,e['mapping']=native_map(candidates,store,row['root'],j)
                        tokens,e['normalization']=native_normalize(commands,slot,store,row['root'],j)
                        actions=torch.cat((warm.expand(1,N,2,10),tokens),dim=2);e['actions']=store.array(actions)
                        for k,v in info.items():need(torch.equal(expanded[k],v[:,None].expand(1,N,*v.shape[1:])),'actual cached input broadcast')
                        store.event('rollout_intent',root=row['root'],iteration=j)
                        actual=model.rollout(dict(expanded),actions);pred=actual['predicted_emb']
                        e.update(predicted_emb=store.array(pred),action_prefix=store.array(actual['action']),rollout_returned=True)
                        store.event('rollout_return',root=row['root'],iteration=j)
                        store.event('planning_predictions_sealed',root=row['root'],iteration=j)
                        store.event('criterion_intent',root=row['root'],layout='planning',iteration=j)
                        with paid.NoNeural(model):cost=model.criterion(dict(predicted_emb=pred,goal_emb=goal))
                        need(tuple(cost.shape)==(1,N) and cost.device.type=='cuda' and cost.dtype==torch.float32,'native planning cost')
                        e['cost']=store.array(cost);store.event('criterion_return',root=row['root'],layout='planning',iteration=j)
                        e['returned']=True;plan['cost_returned']+=1;store.event('cost_return',root=row['root'],iteration=j);return cost
                class Recorder:
                    output_key=NAME
                    def reset(self):self.history=[]
                    def start_batch(self):return None
                    def end_solve(self):return None
                    def __call__(self,**kw):
                        need(kw['step']==len(plan['callbacks']),'all CEM updates in fixed order')
                        plan['callbacks'].append({k:store.array(kw[k]) for k in old.KEYS});store.event('cem_update',root=row['root'],iteration=kw['step'])
                solver=CEMSolver(Cost(),device=device,seed=SEED,callbacks=[Recorder()])
                solver.configure(action_space=Box(-np.inf,np.inf,shape=(1,2),dtype=np.float32),n_envs=1,config=SimpleNamespace(horizon=3,action_block=5))
                need((solver.num_samples,solver.n_steps,solver.topk,solver.batch_size,solver.var_scale,solver.horizon,solver.action_dim)==(N,I,K,1,1,3,10),'fixed installed defaults except explicit MAIN seed0')
                with Trace(model,store,plan,planning=True):out=solver.solve(info,init_action=torch.zeros((1,3,10),device=device))
                need(plan['cost_attempted']==plan['cost_returned']==I and len(plan['callbacks'])==I and
                    len(plan['modules'])==7*I and all(e['returned'] for e in plan['modules']),'all actual planning slots completed')
                plan.update(returned_logits=store.array(out['actions']),mean=store.array(out['mean'][0]),std=store.array(out['var'][0]),
                    elite_average_cost=out['costs'],status='logit_mean_sealed')
                store.event('parameters_sealed',root=row['root'])
                _,row['final_mapping']=native_map(out['actions'].to(device),store,row['root'],None,True)
                store.event('final_raw_sealed',root=row['root']);store.event('root_planning_sealed',root=row['root'])
            store.event('all_planning_sealed');finals=[]
            for slot,row in zip(p['slots'],rows,strict=True):
                d=row['delivery'];a=source_arrays(slot)
                store.event('raw_reload_intent',root=row['root'])
                commands=old.load_array(store.path,row['final_mapping']['raw'],store.inventory)
                d['raw15']=store.array(commands);store.event('raw_reload_return',root=row['root'])
                tokens,d['normalization']=native_normalize(tensor(commands).reshape(1,3,5,2),slot,store,row['root'],None)
                a['actions']=tokens.cpu().numpy();d['inputs']={k:store.array(v) for k,v in a.items()};store.event('delivery_sealed',root=row['root'])
                emb=tensor(a['emb']);actions=torch.cat((tensor(a['warm']),tokens[:,None]),dim=2)
                with Trace(model,store,d):
                    for layout in LAYOUTS:
                        d['active_layout']=layout;e=d['layouts'][layout];e['attempted']=True
                        store.guard();store.event('layout_intent',root=row['root'],layout=layout)
                        if layout=='whole':
                            actual=model.rollout(dict(pixels=tensor(a['pixels'])[:,None],emb=emb[:,None]),actions)
                            pred=actual['predicted_emb'];e['action_prefix']=store.array(actual['action'])
                        else:
                            current=emb;future=[];e['full3']=[]
                            for step in range(3):
                                full=model.predict(current,model.action_encoder(actions[:,0,step:step+3]))
                                own=full[:,-1:];future.append(own);e['full3'].append(store.array(full))
                                current=torch.cat((current[:,1:],own),dim=1)
                            pred=torch.cat((emb,*future),dim=1)[:,None]
                        e['predicted_emb']=store.array(pred);e['returned']=True;finals.append((d,layout,pred,a['goal_emb']))
                        store.event('layout_return',root=row['root'],layout=layout)
                del d['active_layout'];d['status']='predictions_sealed';store.event('root_replays_sealed',root=row['root'])
            need(len(finals)==4,'all roots/layouts before scoring');r['all_final_predictions_sealed']=True;store.event('all_final_predictions_sealed')
            with paid.NoNeural(model):
                for d,layout,pred,g in finals:
                    store.guard();e=d['scores'][layout];e['attempted']=True;store.event('criterion_intent',root=d['root'],layout=layout,iteration=None)
                    cost=model.criterion(dict(predicted_emb=pred,goal_emb=tensor(g)[:,None]))
                    need(tuple(cost.shape)==(1,1) and cost.device.type=='cuda' and cost.dtype==torch.float32,'native final paid cost')
                    e['cost']=store.array(cost);e['returned']=True
                    store.event('criterion_return',root=d['root'],layout=layout,iteration=None)
            for row in rows:row['delivery']['status']='paid_finalmean_evaluated'
        r['model_after']=old.inputs.moving.low.tensor_hash(model);need(r['model_after']==r['model_before'],'unchanged original model')
        need(all(sha(f)==h0 for f,h0 in p['sources'].items()),'all bytes unchanged');r['source_verified']=True
        r['GPU_peak_allocated']=torch.cuda.max_memory_allocated();r['GPU_peak_reserved']=torch.cuda.max_memory_reserved()
        need(r['GPU_peak_allocated']<=3*1024**3 and r['GPU_peak_reserved']<=4*1024**3,'whole GPU peak cap')
        r['numerics']=dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_autocast=False,observed_TF32=False)
    except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)))
    finally:
        if model is not None and r['model_after'] is None:
            try:r['model_after']=old.inputs.moving.low.tensor_hash(model)
            except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)))
        r['elapsed_seconds']=time.monotonic()-started
        if r['elapsed_seconds']>=WALL:r['errors'].append(dict(type='wall_cap',message='whole900s'))
        r['status']='known2_H3_bounded_CEM_actual_raw_evaluated' if not r['errors'] and r['source_verified'] and r['all_final_predictions_sealed'] else 'partial_or_failed'
        store.event('finished',final=True,status=r['status']);journal=store.path/'journal.jsonl'
        store.inventory['journal.jsonl']=dict(sha256=sha(journal),file_bytes=journal.stat().st_size)
        r.update(artifacts=dict(store.inventory),journal_tail=store.tail,actual_modules=module_totals(rows))
        store.put('report.json',raw(r),final=True)
    return dict(status=r['status'],path=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def mapping_check(e,load,shape):
    need(e['returned'] is True,'actual once mapping returned');logits=load(e['logits']);commands=load(e['raw'])
    need(logits.shape==shape and logits.dtype==commands.dtype==np.float32 and np.isfinite(commands).all() and
        (np.abs(commands)<=1).all() and commands.size==logits.size,'complete bounded raw mapping')
    reference=np.tanh(logits.astype(np.float64)).astype(np.float32).reshape(commands.shape)
    # Saved source-bound CUDA mapping is authoritative. NumPy libm is diagnostic,
    # not an exact CUDA tanh certificate or adjustable equivalence threshold.
    return commands,dict(NumPy_tanh_diagnostic=paid.compare(commands,reference),actual_CUDA_tanh_from_bound_source=True)


def normalization_check(e,commands,slot,load):
    need(e['returned'] is True and exact(load(e['raw']),commands),'actual supplied raw normalizer input')
    normalized=load(e['tokens']);mean=np.tile(np.asarray(slot['mean'],np.float32),5);std=np.tile(np.asarray(slot['std'],np.float32),5)
    reference=(commands.reshape(*commands.shape[:-2],10)-mean)/std
    need(exact(normalized,reference),'P2 supplied-stat float32 subtract/divide exact; no Hi scaler or extra /100')
    return normalized


def check_modules(modules,schedule,load,planning=False,iteration=None):
    need(len(modules)==len(schedule),'all H3 native modules')
    values=[]
    for i,(e,(kind,shapes,outshape)) in enumerate(zip(modules,schedule,strict=True)):
        args=[load(x) for x in e['inputs']];out=load(e['output'])
        need(e['kind']==kind and e['returned'] is True and e['index']==(iteration*7+i if planning else i) and
            [a.shape for a in args]==shapes and out.shape==outshape and
            all(a.dtype==np.float32 for a in [*args,out]),'H3 native order/shapes/dtypes')
        need(e.get('iteration')==iteration if planning else e['layout']==('whole' if i<7 else 'segmented'),'native phase identity')
        values.append((args,out))
    return values


def whole_chain(values,pred,actions,base,n):
    need(pred.shape==(1,n,6,192) and exact(pred[:,:,:3],np.broadcast_to(base['emb'][:,None],(1,n,3,192))) and
        exact(values[0][0][0],actions.reshape(n,5,10)),'whole H3 original history/AE5')
    ae=values[0][1]
    for step in range(3):
        (pi,po),(qi,qo)=values[1+step*2:3+step*2]
        need(exact(pi[0],pred[:,:,step:step+3].reshape(n,3,192)) and exact(pi[1],ae[:,step:step+3]) and
            exact(qi[0],po.reshape(n*3,192)) and exact(qo.reshape(n,3,192)[:,-1],pred[:,:,3+step].reshape(n,192)),
            'whole H3 full3 causal chain')


def whole_check(modules,pred,actions,base,load,n,j):
    whole_chain(check_modules(modules,whole_schedule(n),load,True,j),pred,actions,base,n)


def validate_delivery(d,base,tokens,load):
    need(d['status']=='paid_finalmean_evaluated' and set(d['layouts'])==set(d['scores'])==set(LAYOUTS),'all four final layout slots')
    need(set(d['inputs'])==set(base)|{'actions'} and all(exact(load(d['inputs'][k]),v) for k,v in base.items())
        and exact(load(d['inputs']['actions']),tokens),'original deployment inputs')
    values=check_modules(d['modules'],delivery_schedule(),load)
    actions=np.concatenate((base['warm'],tokens[:,None]),axis=2);preds={};scores={}
    for layout in LAYOUTS:
        e=d['layouts'][layout];need(e['attempted'] is e['returned'] is True,'complete final layouts')
        pred=load(e['predicted_emb']);preds[layout]=pred
        need(pred.shape==(1,1,6,192) and pred.dtype==np.float32 and exact(pred[:,0,:3],base['emb']),'H3 final all6 frames')
        if layout=='whole':
            need(exact(load(e['action_prefix']),actions[:,:,:3]),'official action prefix history3')
            whole_chain(values[:7],pred,actions,base,1)
        else:
            need(len(e['full3'])==3,'all segmented full3 snapshots')
            for step in range(3):
                (ai,ao),(pi,po),(qi,qo)=values[7+3*step:10+3*step]
                full=load(e['full3'][step])
                need(exact(ai[0],actions[:,0,step:step+3]) and exact(pi[0],pred[:,0,step:step+3]) and
                    exact(pi[1],ao) and exact(qi[0],po.reshape(3,192)) and exact(full,qo.reshape(1,3,192)) and
                    exact(full[:,-1],pred[:,0,3+step]),'segmented own H3 context/action shift')
        score=d['scores'][layout];cost=load(score['cost'])
        total=np.sum((pred[:,:,-1].astype(np.float64)-base['goal_emb'][:,0,None].astype(np.float64))**2,axis=-1,dtype=np.float64)
        need(score['attempted'] is score['returned'] is True and cost.shape==(1,1) and cost.dtype==np.float32,'final criterion full denominator')
        old.bounded_reduction(cost,total,194)
        scores[layout]=dict(sum192_float32=cost.tolist(),sum192_float64=total.tolist(),mean192_float64=(total/192).tolist(),
            checkpoint_mean192_float64=[np.mean((pred[:,:,3+t].astype(np.float64)-base['goal_emb'][:,0,None].astype(np.float64))**2,axis=-1).tolist() for t in range(3)])
    return dict(root=d['root'],scores=scores,comparisons=dict(all_future=paid.compare(preds['whole'][:,:,3:],preds['segmented'][:,:,3:]),
        **{f'frame{15+5*t}':paid.compare(preds['whole'][:,:,3+t],preds['segmented'][:,:,3+t]) for t in range(3)}),
        returned_mean_evaluated_in_new_job=True,parity_upgraded=False,physical_release=False)


def validate_root(slot,row,load):
    root=slot['source_slot'];plan=row['planning'];d=row['delivery'];need(row['root']==plan['root']==d['root']==root and
        row['task_id']==slot['task_id'] and row['task_sha256']==slot['task_sha256'],'all fixed P root/task identities')
    expected=source_arrays(slot);base={k:load(v) for k,v in plan['base_inputs'].items()}
    need(set(base)==set(expected) and all(exact(base[k],expected[k]) for k in base),'original context/goal/RGB/warmup exact')
    need(plan['status']=='logit_mean_sealed' and plan['cost_attempted']==plan['cost_returned']==I and
        len(plan['costs'])==len(plan['callbacks'])==I and len(plan['modules'])==I*7,'all9000 candidates per root')
    mean=np.zeros((1,3,10),np.float32);std=np.ones_like(mean);scores=[];maps=[]
    for j,(e,c) in enumerate(zip(plan['costs'],plan['callbacks'],strict=True)):
        need(e['iteration']==j and e['returned'] is e['rollout_returned'] is True and set(c)==set(old.KEYS),'all actual iterations returned')
        raw_commands,diag=mapping_check(e['mapping'],load,(1,N,3,10));need(raw_commands.shape==(1,N,3,5,2),'all mapped raw slots')
        tokens=normalization_check(e['normalization'],raw_commands,slot,load);actions=load(e['actions']);pred=load(e['predicted_emb']);cost=load(e['cost'])
        v={k:load(x) for k,x in c.items()};logits=load(e['mapping']['logits'])
        need(exact(v['candidates'],logits) and exact(logits[:,0],mean) and exact(v['prev_mean'],mean) and exact(v['prev_var'],std) and
            actions.shape==(1,N,5,10) and exact(actions[:,:,:2],np.broadcast_to(base['warm'],(1,N,2,10))) and exact(actions[:,:,2:],tokens) and
            exact(load(e['action_prefix']),actions[:,:,:3]),'logit updates vs actual bounded normalized actions, no inverse/clipping')
        whole_check(plan['modules'][7*j:7*j+7],pred,actions,base,load,N,j)
        totals=np.sum((pred[:,:,-1].astype(np.float64)-base['goal_emb'][:,0,None].astype(np.float64))**2,axis=-1,dtype=np.float64)
        need(cost.shape==(1,N) and cost.dtype==np.float32 and exact(v['costs'],cost),'original official planning criterion delivered unchanged')
        old.bounded_reduction(cost,totals,194)
        inds=v['topk_inds'];need(inds.shape==(1,K) and inds.dtype==np.int64 and len(set(inds[0].tolist()))==K and
            (inds>=0).all() and (inds<N).all(),'all actual elite indices')
        need(exact(v['topk_vals'],np.take_along_axis(cost,inds,1)) and exact(v['topk_candidates'],logits[np.arange(1)[:,None],inds]) and
            exact(np.sort(v['topk_vals'],axis=1),np.sort(cost,axis=1)[:,:K]),'nonlinear Q over all raw candidates, logit elites')
        elite=v['topk_candidates'].astype(np.float64);old.bounded_reduction(v['mean'],elite.mean(axis=1),K);old.bounded_reduction(v['var'],elite.std(axis=1,ddof=1),K)
        mean,std=v['mean'],v['var'];scores.append(dict(iteration=j,sum192_float64=totals.tolist(),mean192_float64=(totals/192).tolist()));maps.append(diag)
    need(exact(load(plan['returned_logits']),mean) and exact(load(plan['mean']),mean) and exact(load(plan['std']),std),'actual returned logit mean/std')
    old.bounded_reduction(np.asarray(plan['elite_average_cost']),v['topk_vals'].astype(np.float64).mean(axis=1),K)
    commands,diag=mapping_check(row['final_mapping'],load,(1,3,10));need(commands.shape==(15,2) and
        exact(load(row['final_mapping']['logits']),mean) and exact(load(d['raw15']),commands),'exact sealed final raw15 reloaded, no clip/tanh again')
    tokens=normalization_check(d['normalization'],commands.reshape(1,3,5,2),slot,load)
    need(exact(load(d['inputs']['actions']),tokens),'actual deployment tokens used, not eliteavg or inverse-normalized proxy')
    for k in expected:need(exact(load(d['inputs'][k]),expected[k]),'delivery context/goal/history unchanged')
    validated=validate_delivery(d,expected,tokens,load)
    return dict(root=root,task_id=slot['task_id'],task_sha256=slot['task_sha256'],candidate_count=N*I,
        planning_scores=scores,mapping_diagnostics=maps,final_mapping_diagnostic=diag,
        actual_raw15=commands.tolist(),actual_raw15_bytes_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),
        delivery=validated,elite_average_is_not_delivery_cost=True,physical_release=False)


def expected_events(h,status):
    events=[dict(kind='started',prepare_sha256=h)]
    def add(kind,**kw):events.append(dict(kind=kind,**kw))
    for root in (0,1):
        add('planning_started',root=root)
        for j in range(I):
            add('cost_intent',root=root,iteration=j)
            for kind in ('mapping_intent','mapping_return'):add(kind,root=root,iteration=j,final_mapping=False)
            for kind in ('normalize_intent','normalize_return'):add(kind,root=root,iteration=j)
            add('rollout_intent',root=root,iteration=j)
            for i,(k,_,_) in enumerate(whole_schedule(N)):
                for kind in ('module_intent','module_return'):add(kind,root=root,index=7*j+i,kind_name=k)
            add('rollout_return',root=root,iteration=j);add('planning_predictions_sealed',root=root,iteration=j)
            for kind in ('criterion_intent','criterion_return'):add(kind,root=root,layout='planning',iteration=j)
            add('cost_return',root=root,iteration=j);add('cem_update',root=root,iteration=j)
        add('parameters_sealed',root=root)
        for kind in ('mapping_intent','mapping_return'):add(kind,root=root,iteration=None,final_mapping=True)
        add('final_raw_sealed',root=root);add('root_planning_sealed',root=root)
    add('all_planning_sealed')
    for root in (0,1):
        for kind in ('raw_reload_intent','raw_reload_return','normalize_intent','normalize_return'):
            add(kind,root=root,**({'iteration':None} if kind.startswith('normalize') else {}))
        add('delivery_sealed',root=root)
        for layout,indices in (('whole',range(7)),('segmented',range(7,16))):
            add('layout_intent',root=root,layout=layout)
            for i in indices:
                for kind in ('module_intent','module_return'):add(kind,root=root,index=i,kind_name=delivery_schedule()[i][0])
            add('layout_return',root=root,layout=layout)
        add('root_replays_sealed',root=root)
    add('all_final_predictions_sealed')
    for root in (0,1):
        for layout in LAYOUTS:
            for kind in ('criterion_intent','criterion_return'):add(kind,root=root,layout=layout,iteration=None)
    add('finished',status=status);return events


def consume(output,h):
    folder=own(output);r=read(folder/'report.json',h);p=verify(r['prepared'],r['prepare_sha256'])
    need(r['schema']==NAME+'.report' and r['contract']==contract() and r['status']=='known2_H3_bounded_CEM_actual_raw_evaluated' and r['errors']==[] and
        r['source_verified'] is r['all_final_predictions_sealed'] is True and r['model_before']==r['model_after']==p['expected_model_hash'] and
        r['inherited']==p['inherited'] and r['physical_release'] is r['scientific_admission'] is False and len(r['rows'])==2,'complete new bounded known2 job only')
    need(r['elapsed_seconds']<WALL and r['GPU_peak_allocated']<=3*1024**3 and r['GPU_peak_reserved']<=4*1024**3 and
        r['numerics']==dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_autocast=False,observed_TF32=False),'observed resource/numerical guards')
    inv=r['artifacts'];need(set(inv)|{'report.json'}=={x.name for x in folder.iterdir()} and
        sum(v['file_bytes'] for v in inv.values())+(folder/'report.json').stat().st_size<=CAP,'whole artifact census/output cap')
    for f,v in inv.items():need(Path(f).name==f and not (folder/f).is_symlink() and (folder/f).stat().st_size==v['file_bytes'] and sha(folder/f)==v['sha256'],'all original NPY/journal bytes')
    tail=None;events=[]
    with (folder/'journal.jsonl').open('rb') as stream:
        for line in stream:
            need(len(line)<=1024**2,'bounded journal');e=json.loads(line);h0=e.pop('event_sha256')
            need(e['sequence']==len(events) and e['previous_sha256']==tail and hashlib.sha256(raw(e)).hexdigest()==h0,'full durable hashchain')
            e['event_sha256']=h0;need(raw(e)==line,'canonical single newline');tail=h0
            events.append({k:v for k,v in e.items() if k not in ('sequence','previous_sha256','event_sha256')})
    need(events==expected_events(r['prepare_sha256'],r['status']) and tail==r['journal_tail'],'all raw mappings, native calls and seal-before-criterion orders')
    used=set()
    def load(rec):used.add(rec['file']);return old.load_array(folder,rec,inv)
    rows=[validate_root(s,row,load) for s,row in zip(p['slots'],r['rows'],strict=True)]
    need(used==set(inv)-{'journal.jsonl'},'all actual raw/normalized/native IO accounted')
    actual=module_totals(r['rows']);need(r['actual_modules']==actual,'actual H3 module resources reconstructed')
    for kind in KINDS:
        for suffix in ('attempted','returned'):
            need(actual[kind][suffix]==resources()[kind],'exact H3 native API budget')
            samples=resources()['AE_samples' if kind=='action_encoder' else 'predictor_samples' if kind=='predictor' else 'pred_proj_rows']
            tokens=resources()['AE_tokens' if kind=='action_encoder' else 'predictor_context_tokens' if kind=='predictor' else 'pred_proj_rows']
            need(actual[kind]['samples_'+suffix]==samples and actual[kind]['tokens_'+suffix]==tokens,'exact actual H3 sample/token budget')
    return dict(schema=NAME+'.consume',status='known2_H3_bounded_CEM_actual_raw_saved_verified',report_sha256=h,rows=rows,
        resources=resources(),actual_modules=actual,inherited=r['inherited'],same_horizon_as_P2_common_H3=True,official_reproduction=False,
        same_support_as_P2=False,same_planning_reduction_as_P2=False,fair_ranking=False,
        parity_upgraded=False,physical_release=False,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',required=True)
    for cmd in ('verify','run'):
        a=s.add_parser(cmd);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd=='run':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=s.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt',required=True)
    q=p.parse_args()
    if q.command=='prepare':v=prepare(q.output)
    elif q.command=='verify':verify(q.prepared,q.prepare_sha);v=dict(status='bounded_CEM_prepare_verified')
    elif q.command=='run':v=run(q.prepared,q.prepare_sha,q.output,q.authorization)
    else:
        v=consume(q.output,q.report_sha);p=own(q.receipt);b=raw(v);need(len(b)<=RESERVE and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'receipt bounds')
        durable(p,b);v=dict(status=v['status'],path=str(p),sha256=sha(p))
    print(raw(v).decode(),end='')
    if v.get('status')=='partial_or_failed':raise SystemExit(2)


if __name__=='__main__':main()

