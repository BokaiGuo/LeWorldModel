"""Fixed24 suffix compiler: own imagined/teacher context × explicit action scale."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import os
import time
import numpy as np
from experiments.spectral_reachability import moving_capability_direction_cuda_MAIN_20261027_v1 as source
from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as tracebase
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as packet
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v2 as svdtrace

NAME='capability_suffix_factorial_cuda_MAIN_20261027_v1'
ROOT=source.ROOT
AUTH='MAIN_FIXED24_SUFFIX_FACTORIAL_CUDA_ONLY'
VARIANTS=('open_full','teacher_full','teacher_small')
PREP=ROOT/'results/moving_capability_direction_cuda_MAIN_20261027_v1_prepare_MAIN01.json'
PREP_SHA='c76a5c8b50aa47887cdb4e354f182a5700b64aa89068b1a7fcb9ba830ea41dab'
require,sha=source.require,source.sha


def contract():
    return dict(anchors=8,cases=24,variants=list(VARIANTS),candidates=10,
        raw_scale=dict(open_full=1.,teacher_full=1.,teacher_small=.1),
        damping=packet.DAMPING,logit_l2_cap=packet.CAP,radii=list(packet.RADII),
        caps=dict(action_encoder=312,predictor=312,pred_proj=312),predict_API=312,
        encoder=0,projector=0,jacobians=24,mathematical_VJP_directions=4608,SVD=48,
        selection='nonlinear identity mean192; stable bank index',
        training=False,Qfit=False,new_physics=0,fresh_roots=0,P1_admission=False,
        total_FLOPs_matched=False,CUDA_only=True,TF32=False,wall_seconds=900,
        output_cap_bytes=64*1024**2,final_reserve_bytes=4*1024**2,disk_floor_bytes=200*1024**2)


def cost(predicted,goal):
    a,b=np.asarray(predicted,np.float64),np.asarray(goal,np.float64)
    require(a.shape==b.shape==(192,) and np.isfinite(a).all() and np.isfinite(b).all(),'native192 Q')
    return float(np.mean((a-b)**2))


def compile_case(case,terminal,jacobian,operation=lambda kind,fn:fn()):
    """Actual production core; terminal is own scale/context, no outcome inputs."""
    nominal=np.asarray(terminal(np.zeros(10,np.float32)),np.float64)
    j=np.asarray(operation('jacobian',jacobian),np.float64)
    require(nominal.shape==(192,) and j.shape==(192,10) and np.isfinite(j).all(),'own true Jacobian')
    goal=np.asarray(case['goal'],np.float64)
    bank=svdtrace.with_svd_receipts(lambda:packet.spectral_bank(j,nominal,goal,np.eye(192)),operation)
    candidates=[]
    for i,u in enumerate(bank['params']):
        params=np.asarray(u,np.float32)
        pred=np.asarray(terminal(params),np.float32)
        raw=case['scale']*np.tanh(params).reshape(5,2)
        candidates.append(dict(index=i,kind=bank['kinds'][i],params=params.tolist(),raw=raw.tolist(),
                               predicted=pred.tolist(),Q=cost(pred,goal)))
    winner=deepcopy(min(candidates,key=lambda c:(c['Q'],c['index'])))
    replay=np.asarray(terminal(np.asarray(winner['params'],np.float32)),np.float32)
    require(np.array_equal(replay,np.asarray(winner['predicted'],np.float32)),'independent winner replay exact')
    return dict(nominal=nominal.tolist(),jacobian=j.tolist(),bank=bank,candidates=candidates,winner=winner,replay_exact=True)


def build():
    require(sha(PREP)==PREP_SHA,'fixed real teacher/open preparation')
    p=source.verify(PREP,PREP_SHA); bindings=deepcopy(p['bindings']);bindings[str(PREP)]=PREP_SHA
    # Validate actual source CUDA modules without running a model.
    from experiments.spectral_reachability import consume_capability_direction_MAIN_20261027_v1 as consumer
    consumer.consume(ROOT/'results/moving_capability_direction_cuda_MAIN_20261027_v1_MAIN01',
                     'eac431fca369c977c57d38a32a600d1ef02ae2030b4112636d4c3df92bf143fe')
    rp=ROOT/'results/moving_capability_direction_cuda_MAIN_20261027_v1_MAIN01/report.json'
    rr=source.rgb.read(rp,8*1024**2);bindings[str(rp)]=sha(rp)
    for name,h in rr['artifact_sha256'].items(): bindings[str(rp.parent/name)]=h
    plans=tracebase.consume(source.PLANNER,source.PLANNER_SHA)
    cases=[]
    for anchor in range(8):
        r=p['rows'][5*anchor];plan=plans[anchor]
        require(r['probe']=='zero' and source.key(r)==source.key(plan) and r['selection_digest']==plan['selection_digest'],'same original prefix')
        for variant in VARIANTS:
            mode='open' if variant=='open_full' else 'teacher'
            cases.append(dict(index=len(cases),anchor=anchor,variant=variant,scale=.1 if variant=='teacher_small' else 1.,
                context=dict(latent_history=deepcopy(r['contexts'][mode]),action_history=deepcopy(r['contexts']['past'])),
                goal=deepcopy(r['goal']),prefix=deepcopy(plan['commands'][:5]),original_plan=deepcopy(plan),
                root_slot=r['root_slot'],condition=r['condition'],arm=r['arm'],task_id=r['task_id'],selection_digest=r['selection_digest']))
    files=[Path(__file__),Path(consumer.__file__),Path(packet.__file__),Path(svdtrace.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        ROOT.parent/'docs/D2P_CAPABILITY_SUFFIX_FACTORIAL_PROTOCOL_MAIN_20261027.md']
    for f in files: bindings[str(f.resolve())]=sha(f)
    for path,h in bindings.items(): require(sha(path)==h,'final dependency seal')
    return dict(schema=NAME+'.prepare',contract=contract(),cases=cases,bindings=bindings,policy=p['policy'],model_hash=p['model_hash'])


def prepare(output):
    p=build();data=source.rgb.canonical(p)
    require(len(data)<8*1024**2,'bounded prepare')
    with Path(output).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(Path(output).absolute()),sha256=sha(output))


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=source.rgb.read(path,8*1024**2)
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and len(p['cases'])==24,'fixed cases and contract')
    for path,h in p['bindings'].items():require(sha(path)==h,'dependency drift')
    return p


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA authorization')
    started=time.monotonic()
    def check():require(time.monotonic()-started<900,'whole job900s')
    p=verify(prepared,expected);store=tracebase.Store(output,check)
    rows=[dict(index=i,status='not_started') for i in range(24)]
    operations=[];errors=[];model=None;trace=None;before=after=None;handles=[];api=0;host=None
    def operation(kind,callback):
        cap=24 if kind=='jacobian' else 48
        require(sum(e['kind']==kind for e in operations)<cap,'predispatch math cap')
        e=dict(index=len(operations),case=current,kind=kind,completed=False);operations.append(e)
        store.event(dict(kind='math_attempt',entry=deepcopy(e)))
        try:value=callback();e['completed']=True;return value
        finally:store.event(dict(kind='math_return_or_failed',entry=deepcopy(e)))
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as native
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        native.frozen(model);before=native.tensor_hash(model);require(before==p['model_hash'],'original frozen model')
        trace=tracebase.NativeTrace(model,torch,store,check,[1,257,192])
        def pre(kind):
            def guard(module,args):
                require(kind in contract()['caps'],'no new visual calls')
                require(sum(e['resource']==kind for e in trace.entries)<312,'312 native predispatch cap')
            return guard
        for kind in ('encoder','projector','action_encoder','predictor','pred_proj'):
            handles.append(getattr(model,kind).register_forward_pre_hook(pre(kind)))
        transition=packet.make_transition(model,device)
        with sdpa_kernel(SDPBackend.MATH),trace:
            for case in p['cases']:
                current=case['index'];row=rows[current];row['status']='attempted'
                trace.scope=dict(plan_index=current)
                c=case['context'];emb=torch.tensor(c['latent_history'],device=device,dtype=torch.float32)[None]
                history=torch.tensor(c['action_history'],device=device,dtype=torch.float32)[None]
                def forward(u):
                    nonlocal api
                    check();require(api<312,'predict API predispatch cap');api+=1
                    return transition(emb,history,case['scale']*torch.tanh(u).reshape(5,2))[0][0,-1]
                def terminal(params):
                    with torch.no_grad():z=forward(torch.tensor(params,device=device,dtype=torch.float32))
                    return z.detach().cpu().numpy().copy()
                def jacobian():
                    zero=torch.zeros(10,device=device,dtype=torch.float32,requires_grad=True)
                    j=torch.autograd.functional.jacobian(forward,zero,vectorize=True)
                    return j.detach().cpu().numpy().copy()
                result=compile_case(case,terminal,jacobian,operation)
                row.update(status='sealed',result=result)
                store.put(f'case_{current:02}.json',row)
                store.event(dict(kind='case_sealed',index=current,winner=result['winner']['index']))
        for h in reversed(handles):h.remove()
        handles=[]
        native.frozen(model);after=native.tensor_hash(model);verify(prepared,expected);check()
        require(before==after and api==312 and len(operations)==72 and all(e['completed'] for e in operations),'frozen actual API/math census')
        require(all(sum(e['resource']==kind and e['completed'] for e in trace.entries)==312 for kind in contract()['caps']) and not trace.pending,'native complete census')
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)[:1800]))
        if model is not None:
            try:after=native.tensor_hash(model)
            except BaseException as exc2:errors.append(dict(type=type(exc2).__name__,message=str(exc2)[:1000]))
    finally:
        for h in reversed(handles):h.remove()
        try:store.close()
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)[:1000]))
    good=not errors and all(r['status']=='sealed' for r in rows) and before==after==p['model_hash']
    report=dict(schema=NAME+'.report',status='all24_plans_sealed' if good else 'partial_or_failed',rows=rows,
        errors=errors,operations=operations,ledger=[] if trace is None else trace.entries,
        prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=contract(),
        model_hash_before=before,model_hash_after=after,predict_API=api,host=host,wall_seconds=time.monotonic()-started,
        commands_released=False,P1_admission=False,artifact_sha256={f.name:sha(f) for f in store.output.iterdir() if f.is_file()})
    store.check=lambda:None;store.put('report.json',report,final=True)
    require(good,'partial evidence preserved '+str(errors))
    return dict(status=report['status'],sha256=sha(store.output/'report.json'),cases=24)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('output','prepared','prepare-sha','authorization'):parser.add_argument('--'+name)
    a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(status='verified_no_execution',cases=len(verify(a.prepared,a.prepare_sha)['cases']))
    else:result=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    print(json.dumps(result,sort_keys=True))
