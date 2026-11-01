"""Offline prepare; explicit host CUDA only. No environment imports or rollout.

GRASP is the sealed official-adapter mechanism subset, not internal action GD.
The action-Adam comparator starts from the same H1 input and zero action plan.
Transition/backward counts match; processed context and graph work do not.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

from experiments.spectral_reachability import p2_grasp_reference_core_20261027_v1 as core
from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME = 'p2_grasp_h1_cuda_runner_20261027_v1'
METHODS = ('grasp_h1', 'action_adam_h1')
COMPILATION = 'results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json'
COMPILATION_SHA = 'd5f544689a9d2b5c6807b2454c1a98b9db6d88bb374badc457bdcc6defa87c8a'
GOALS = 'results/pusht_d2p_goal_agent_intervention4_v2_20261026_cuda/report.json'
FRAMES = 'results/pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json'
CORE_PREPARE = 'results/p2_grasp_official_contract_20261027_v1/prepare/prepare.json'
CORE_PREPARE_SHA = '608bffb9d83a8e033e5ec0cc2b30c82a8084d1bfe5f51a7ab11a3cf3a70be151'
UPSTREAM = 'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_protocol_20261027.json'
UPSTREAM_ID = 'a18656bd92f5db33bb95227d8d1e643fcd30096856151641c25648a6d355e5c0'


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


MEAN = [f32(-.0087), f32(.0068)]
STD = [f32(.2019), f32(.2002)]
require = io.require


def contract():
    return dict(config=asdict(core.Config()), methods=list(METHODS), rollout_tokens=3,
        initial_history='H1_last_real_full_latent', latent_dim=192, action_dim=10,
        virtual_state='full original model output embedding, not Q projection',
        metric='mean((terminal_full_latent-independent_goal0)^2)', action_adam_steps=900,
        action_adam_lr=.005, initialization='zero normalized action for both arms',
        transition_samples_per_arm=2703, backward_per_arm=900, total_h1_transitions=21624,
        real_h3_diagnostic_transitions=24, total_with_diagnostic=21648,
        processed_context_tokens_per_arm=dict(grasp_h1=3156,action_adam_h1=5406),
        real_h3_diagnostic_context_tokens=72, visual_encoder_calls=0,
        feedback='cached initial state only; no observation, physics or replanning',
        real_h3_diagnostic='separate original three-frame context replay; never H1 objective or equal-budget H3 arm',
        normalization=dict(mean=MEAN,std=STD,source='experiments/run_pusht_counterfactual.py PUBLIC_ACTION_MEAN/STD float32',
            certification='public supplied stats, not certified checkpoint training stats', extra_div100=False),
        bounds='official unconstrained normalized optimization; preserve raw out-of-box, never clip',
        full_pipeline_equal_compute=False, official_equivalence=False,
        inherited_encodings_cost='unknown; inherited cached RGB/goal encoding, no fresh encoding',
        new_independent_roots=0, cohort='four existing calibration roots', P0_admission=False,P2_admission=False)


def phases(method, cfg=None):
    cfg = cfg or core.Config()
    budget = core.budget(cfg)
    if method == 'action_adam_h1':
        return ['action_adam'] * (budget['backward_api_calls_per_root']*cfg.horizon) + ['verification']*cfg.horizon
    require(method == 'grasp_h1', 'unknown method')
    out=[]
    for k in range(cfg.steps):
        out += ['soft']*cfg.horizon
        if k+1 in budget['sync_after_iterations_1based']:
            out += ['sync']*(cfg.sync_steps*cfg.horizon)
    return out + ['verification']*cfg.horizon


def finite_array(x, shape):
    if not shape:
        return type(x) in (int,float) and math.isfinite(x)
    return isinstance(x,list) and len(x)==shape[0] and all(finite_array(v,shape[1:]) for v in x)


def hexsha(x):
    return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)


def validate_packet(p):
    require(finite_array(p['context'],(3,192)), 'full original latent history [3,192] required')
    require(finite_array(p['history_actions'],(2,10)), 'between-frame action history required')
    require(finite_array(p['goal'],(192,)), 'full independent goal [192] required')
    require(all(hexsha(p[k]) for k in ('root_sha256','history_rgb_sha256','goal_rgb_sha256')), 'source frame/root hashes required')


def packets(compilation, goals, frames):
    roots=compilation['protocol']['roots']
    require(len(roots)==len(set(roots))==4, 'four unique source roots required')
    ci=io.index(compilation['rows'],('root_sha256',))
    gi=io.index(goals['rows'],('root_sha256',))
    fi=io.index(frames['rows'],('root_sha256',))
    require(set(ci)==set(gi)==set(fi)=={(r,) for r in roots}, 'source root joins differ')
    out=[]
    for r in roots:
        c=ci[r,]['compiled']; g=gi[r,]['variants'][0]; f=fi[r,]
        require(g['offset']==[0.,0.] and g['goal_vector']==c['goal_vectors'][0], 'independent goal0 binding mismatch')
        p=dict(root_sha256=r,context=c['input_context']['emb'][0],
            history_actions=c['input_context']['actions'][0][1:],goal=g['goal_vector'],
            history_rgb_sha256=f['history_rgb_sha256'],goal_rgb_sha256=g['goal_rgb_sha256'])
        validate_packet(p)
        p['input_sha256']=io.sha(io.canonical(p).encode())
        out.append(p)
    return out


def export_actions(a):
    require(finite_array(a,(3,10)), 'finite normalized [3,10] actions required')
    commands=[[token[2*j+i]*STD[i]+MEAN[i] for i in range(2)] for token in a for j in range(5)]
    require(finite_array(commands,(15,2)), 'finite raw actions required')
    return dict(normalized_actions=a,commands=commands,planned_commands=15,
        physical_box_supported=all(abs(v)<=1 for cmd in commands for v in cmd),
        clipped=False,layout='time_major_5x2',execution='five_separate_relative_env_steps',
        physical_execution_authorized=False,command_sha256=io.sha(io.canonical(commands).encode()))


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):
        return 'nonfinite:'+str(value)
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean(v) for v in value]
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(clean(value),sort_keys=True,indent=2,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


class Journal:
    """Exclusive append-only journal; every request flushed before its model call."""
    def __init__(self,path):
        self.stream=Path(path).open('x')
        self.sequence=0
    def __call__(self,event):
        self.stream.write(io.canonical(clean(dict(event,sequence=self.sequence)))+'\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence+=1
    def __enter__(self):
        return self
    def __exit__(self,*exc):
        self.stream.close()


def slots(roots):
    return [dict(root_sha256=r,method=m,status='not_started',failure=None,plan=None)
        for i,r in enumerate(roots) for m in (METHODS if i%2==0 else METHODS[::-1])]


def paired_h1(rows,roots):
    by=io.index(rows,('root_sha256','method'))
    require(set(by)=={(r,m) for r in roots for m in METHODS},'all root-arm slots required')
    pairs=[];differences=[]
    for root in roots:
        a,b=(by[root,m] for m in METHODS)
        complete=all(x['status']=='encoded_h1_completed_not_physical' for x in (a,b))
        delta=a['h1_terminal_mse']-b['h1_terminal_mse'] if complete else None
        pairs.append(dict(root_sha256=root,status='complete' if complete else 'incomplete',
            arm_status={m:by[root,m]['status'] for m in METHODS},
            difference_grasp_minus_action_adam=delta))
        if complete:
            differences.append(delta)
    return dict(metric='H1 full-latent terminal MSE only',rows=pairs,complete_pairs=len(differences),
        incomplete_pairs=len(roots)-len(differences),
        mean_difference_grasp_minus_action_adam=sum(differences)/len(differences) if differences else None,
        real_h3_diagnostic_pooled=False,full_pipeline_equal_compute=False,
        scope='descriptive calibration only; incomplete pair statuses retained; no held-out or physical inference')


def prepare(root, output):
    root=Path(root).resolve();output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW output required')
    e=io.Evidence(root)
    c=e.json(root/COMPILATION,COMPILATION_SHA)
    require(c['status']=='compiled_calibration' and all(v is True for v in c['gates'].values()),'source compilation gates')
    e.seal(c['source_seal'])
    bindings=c['protocol']['bindings']
    g=e.json(root/GOALS,bindings[str(root/GOALS)])
    f=e.json(root/FRAMES,bindings[str(root/FRAMES)])
    cp=e.json(root/CORE_PREPARE,CORE_PREPARE_SHA)
    e.raw((root/CORE_PREPARE).parent/'contract.json',cp['contract_sha256'])
    for p,h in cp['sources_sha256'].items():
        e.raw(Path(p),h)
    upstream=e.json(root/UPSTREAM)
    require(io.sha(io.canonical(upstream['protocol']).encode())==upstream['protocol_id']==UPSTREAM_ID,'upstream identity')
    # Includes the installed official model/predictor and compatibility loader.
    sources={Path(p) for p in upstream['protocol']['sources'] if '/site-packages/' in p}
    for p in sources:
        e.raw(p,upstream['protocol']['sources'][str(p)])
    sources.update([Path(__file__).resolve(),Path(core.__file__).resolve(),Path(io.__file__).resolve(),
        root/'experiments/checkpoint_compat.py',root/'experiments/gpu_preflight.py',
        root/'experiments/run_pusht_counterfactual.py',
        root/'experiments/spectral_reachability/d2p_prerun_source_seal.py',
        root/'tests'/('test_'+NAME+'.py'),root/'docs'/(NAME+'.md')])
    for p in sources:
        e.raw(p)
    # AST only: do not import the normalizer's physics-bearing module.
    import ast
    constants={}
    tree=ast.parse((root/'experiments/run_pusht_counterfactual.py').read_text())
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'):
            require(isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='np.asarray' and
                any(k.arg=='dtype' and ast.unparse(k.value)=='np.float32' for k in n.value.keywords),'normalizer float32 source')
            constants[n.targets[0].id]=[f32(v) for v in ast.literal_eval(n.value.args[0])]
    require(constants==dict(PUBLIC_ACTION_MEAN=MEAN,PUBLIC_ACTION_STD=STD),'normalizer source values changed')
    model_bindings={p:h for p,h in bindings.items() if not Path(p).is_relative_to(root.parent)}
    require(len(model_bindings)==2,'explicit local weights/config bindings')
    body=dict(schema=NAME+'.protocol',contract=contract(),packets=packets(c,g,f),
        roots=c['protocol']['roots'],core_prepare_sha256=CORE_PREPARE_SHA,
        model_bindings=model_bindings,inputs_sha256=dict(sorted(e.hashes.items())),
        sources_sha256={str(p):e.hashes[str(p)] for p in sorted(sources)},
        inherited_costs='RGB/history and goal encoding unknown; no fresh encoder calls',
        preflight='no weight loading, torch, CUDA or physics in prepare')
    identity=io.sha(io.canonical(body).encode())
    e.recheck();output.mkdir(exist_ok=False)
    write(output/'protocol.json',dict(protocol=body,protocol_id=identity))
    seal=create_seal(files=sorted(sources),args_json=io.canonical(dict(protocol_path=str(output/'protocol.json'),
        protocol_sha256=io.sha((output/'protocol.json').read_bytes()))),output_dir=output/'source_seal')
    receipt=dict(schema=NAME+'.prepare',status='prepared_not_executed',protocol_id=identity,
        protocol_sha256=io.sha((output/'protocol.json').read_bytes()),source_seal=seal,
        torch_imported='torch' in sys.modules,model_loaded=False,new_predictions=0,new_physical_steps=0,
        expected_h1_transitions=21624,expected_diagnostic_transitions=24,scheduled_root_arms=8,
        P0_admission=False,P2_admission=False)
    e.recheck();write(output/'prepare.json',receipt)
    return receipt


def validate_prepared(prepared, expected_sha):
    prepared=Path(prepared).absolute()
    raw=(prepared/'prepare.json').read_bytes()
    require(io.sha(raw)==expected_sha,'prepare SHA256 mismatch')
    receipt=io.decode(raw)
    require(receipt['status']=='prepared_not_executed' and receipt['model_loaded'] is False,'prepared-only receipt')
    raw=(prepared/'protocol.json').read_bytes()
    require(io.sha(raw)==receipt['protocol_sha256'],'protocol SHA mismatch')
    envelope=io.decode(raw);p=envelope['protocol']
    require(io.sha(io.canonical(p).encode())==envelope['protocol_id']==receipt['protocol_id'],'protocol identity')
    require(p['contract']==contract() and p['core_prepare_sha256']==CORE_PREPARE_SHA,'fixed reviewed contract changed')
    require(len(p['packets'])==len(set(p['roots']))==4 and [x['root_sha256'] for x in p['packets']]==p['roots'],'four root census')
    for packet in p['packets']:
        validate_packet(packet)
        require(io.sha(io.canonical({k:v for k,v in packet.items() if k!='input_sha256'}).encode())==packet['input_sha256'],'packet identity')
    for path,h in p['inputs_sha256'].items():
        require(io.sha(Path(path).read_bytes())==h,'input/source SHA changed: '+path)
    seal=receipt['source_seal']
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    return p,receipt['protocol_id']


def action_adam(wm,start,goal,cfg):
    import torch
    n=core.budget(cfg)['backward_api_calls_per_root']
    a=start.new_zeros((1,cfg.horizon,wm.action_dim),requires_grad=True)
    receipt=dict(status='started',backward_requested=0,backward_completed=0,iterations=[])
    try:
        opt=torch.optim.Adam([a],lr=cfg.lr_a)
        for k in range(n):
            opt.zero_grad()
            loss=(core.serial(wm,start,a)[:,-1]-goal).square().mean()
            require(bool(torch.isfinite(loss)),'finite comparator loss')
            receipt['backward_requested']+=1
            loss.backward()
            receipt['backward_completed']+=1
            opt.step()
            receipt['iterations'].append(dict(iteration_1based=k+1,pre_update_terminal_mse=float(loss.detach())))
        with torch.no_grad():
            verification=core.serial(wm,start,a)
        receipt['status']='encoded_core_completed_not_physical'
        return dict(actions=a.detach(),verification=verification,receipt=receipt)
    except Exception as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise core.PlanningFailure(str(exc),receipt) from exc


def run_arm(model, packet, method, sink, *, analytic_fixture=False, cfg=None):
    """CPU is available ONLY to plain, non-Module algebra test doubles.

    All transition-submitted actions and completions are journaled; this is not
    an optimizer-state checkpoint/resume interface. Failed plans are not filled.
    """
    import torch
    require(method in METHODS,'unknown method')
    cfg=cfg or core.Config()
    if analytic_fixture:
        require(not isinstance(model,torch.nn.Module),'no neural CPU fixture')
    else:
        require(isinstance(model,torch.nn.Module),'CUDA model required; no CPU neural fallback')
        require(cfg==core.Config(),'production configuration cannot change')
    core._guard(model,analytic_fixture)
    validate_packet(packet)
    device='cpu' if analytic_fixture else 'cuda:0'
    start=torch.tensor([packet['context'][-1]],device=device,dtype=torch.float32)
    goal=torch.tensor([packet['goal']],device=device,dtype=torch.float32)
    schedule=phases(method,cfg)+['real_h3_diagnostic']*cfg.horizon
    counts={phase:dict(requested=0,completed=0,failed=0) for phase in set(schedule)}
    position=0
    result=dict(status='started',plan=None,receipt=None,verification=None,real_h3_diagnostic=None,
        transition_counts=counts,failure=None,official_equivalence=False)

    class Recorded(core.LeWMH1):
        def call(self,func,state,a):
            nonlocal position
            require(position<len(schedule),'transition schedule exceeded')
            phase=schedule[position];index=position;position+=1
            counts[phase]['requested']+=len(a)
            sink(dict(kind='transition_requested',phase=phase,index=index,
                normalized_action=a.detach().cpu().tolist()))
            try:
                out=func(state,a)
                pred=out[1] if isinstance(out,tuple) else out
                require(pred.shape==start.shape and bool(torch.isfinite(pred).all()),'finite aligned prediction required')
                counts[phase]['completed']+=len(a)
                sink(dict(kind='transition_completed',phase=phase,index=index))
                return out
            except Exception as exc:
                counts[phase]['failed']+=len(a)
                sink(dict(kind='transition_failed',phase=phase,index=index,error=type(exc).__name__+': '+str(exc)))
                raise
        def edge(self,s,a):
            return self.call(super().edge,s,a)
        def advance(self,s,a):
            return self.call(super().advance,s,a)

    wm=Recorded(model,history_size=3,action_dim=10,analytic_fixture=analytic_fixture)
    try:
        out=core.plan(wm,start,goal,cfg,analytic_fixture=analytic_fixture) if method=='grasp_h1' else action_adam(wm,start,goal,cfg)
        result['receipt']=out['receipt']
        result['verification']=out['verification'][0].cpu().tolist()
        result['h1_terminal_mse']=float((out['verification'][:,-1]-goal).square().mean())
        result['plan']=export_actions(out['actions'][0].cpu().tolist())
        sink(dict(kind='final_plan',plan=result['plan']))
        # Never substitute this for the H1 objective. Uses TWO between-state
        # actions and all THREE original real latent frames, not H1 bootstrap.
        z=torch.tensor([packet['context']],device=device,dtype=start.dtype)
        history=torch.tensor([packet['history_actions']],device=device,dtype=start.dtype)
        state=(z,history); trajectory=[z[:,-1]]
        with torch.no_grad():
            for t in range(cfg.horizon):
                state,pred=wm.advance(state,out['actions'][:,t])
                trajectory.append(pred)
        result['real_h3_diagnostic']=dict(states=torch.stack(trajectory,dim=1)[0].cpu().tolist(),
            terminal_mse=float((trajectory[-1]-goal).square().mean()),included_in_h1_comparison=False)
        require(position==len(schedule),'complete transition schedule required')
        result['status']='analytic_fixture_completed' if analytic_fixture else 'encoded_h1_completed_not_physical'
    except Exception as exc:
        if isinstance(exc,core.PlanningFailure):
            result['receipt']=exc.receipt
        result.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc)))
        sink(dict(kind='arm_failure',failure=result['failure'],partial_receipt=result['receipt']))
    return clean(result)


class ModuleWork:
    """Actual entry/completion hooks, separate from transition-boundary counts."""
    def __init__(self,model,sink):
        self.counts={};self.handles=[];self.sink=sink
        for name in ('predictor','action_encoder','encoder'):
            module=getattr(model,name)
            self.counts[name]=dict(requested_calls=0,completed_calls=0,requested_samples=0,completed_samples=0,requested_context_tokens=0,completed_context_tokens=0)
            def entry(mod,args,name=name):
                x=args[0];b=int(x.shape[0]);t=int(x.shape[1]) if name!='encoder' else 1
                c=self.counts[name]
                for k,v in (('requested_calls',1),('requested_samples',b),('requested_context_tokens',b*t)):
                    c[k]+=v
                self.sink(dict(kind='module_requested',module=name,batch=b,context_tokens=b*t))
            def done(mod,args,out,name=name):
                x=args[0];b=int(x.shape[0]);t=int(x.shape[1]) if name!='encoder' else 1
                c=self.counts[name]
                for k,v in (('completed_calls',1),('completed_samples',b),('completed_context_tokens',b*t)):
                    c[k]+=v
                self.sink(dict(kind='module_completed',module=name,batch=b,context_tokens=b*t))
            self.handles.append(module.register_forward_pre_hook(entry))
            self.handles.append(module.register_forward_hook(done))
    def close(self):
        for h in self.handles:
            h.remove()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    import torch
    result={}
    for name,items in (('parameters',model.named_parameters()),('buffers',model.named_buffers())):
        h=hashlib.sha256()
        for key,t in items:
            h.update(io.canonical([key,str(t.dtype),list(t.shape)]).encode())
            h.update(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        result[name]=h.hexdigest()
    return result


def reviewed_loaded_type(model, expected_class):
    """Reject same-state custom forward implementations at the injection exit."""
    require(type(model) is expected_class,'exact official LeWM class required')
    for name in ('predict','encode'):
        require(getattr(getattr(model,name),'__func__',None) is getattr(expected_class,name),
            'loaded model method overridden: '+name)
    for name in ('predictor','action_encoder','pred_proj'):
        part=getattr(model,name)
        require('forward' not in vars(part),'loaded module forward overridden: '+name)


def execute(prepared,expected_sha,output,*,model=None):
    """User-reviewed host entry. An existing loaded model may be passed explicitly.

    Loaded-model injection requires full live state equality with the hashed
    official checkpoint (compatibility key remapping included), not metadata.
    """
    p,identity=validate_prepared(prepared,expected_sha)
    output=Path(output).absolute();output.mkdir(exist_ok=False)
    rows=slots(p['roots']);failure=None;before=None;after=None;host=None;unchanged=False
    tf32_before=None
    write(output/'started.json',dict(protocol_id=identity,prepare_sha256=expected_sha,rows=rows))
    with Journal(output/'events.jsonl') as journal:
        try:
            for path,h in p['model_bindings'].items():
                require(digest(path)==h,'model/config file changed: '+path)
            for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
                os.environ[k]='1'
            import torch
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.checkpoint_compat import load_compatible_pretrained, remap_legacy_vit_keys
            from stable_worldmodel.wm.lewm import LeWM
            from stable_worldmodel.wm.lewm.module import Predictor, Embedder, MLP
            host=require_cuda_allocation('cuda')
            weight=next(Path(x) for x in p['model_bindings'] if Path(x).name=='weights.pt')
            if model is None:
                model=load_compatible_pretrained(str(weight.parent),device=torch.device('cuda:0'))
            core._guard(model,False)
            reviewed_loaded_type(model,LeWM)
            require(type(model.predictor) is Predictor and type(model.action_encoder) is Embedder and
                type(model.pred_proj) is MLP,'official predictor/action/prediction-projection classes required')
            # Validate already-loaded as well as newly-loaded model without any
            # learned CPU forward: host tensor comparison is not inference.
            expected,_=remap_legacy_vit_keys(torch.load(weight,map_location='cpu',weights_only=True))
            actual=model.state_dict()
            require(set(expected)==set(actual),'loaded checkpoint state keys differ')
            require(all(actual[k].dtype==v.dtype and actual[k].shape==v.shape and torch.equal(actual[k].detach().cpu(),v) for k,v in expected.items()),'loaded model differs from official checkpoint')
            del expected,actual
            before=state_hash(model)
            tf32_before=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            with torch.enable_grad(), torch.autocast(device_type='cuda',enabled=False):
                for i,slot in enumerate(rows):
                    packet=next(x for x in p['packets'] if x['root_sha256']==slot['root_sha256'])
                    tag=dict(root_sha256=slot['root_sha256'],method=slot['method'])
                    def sink(event):
                        journal(dict(event,**tag))
                    sink(dict(kind='arm_started'))
                    work=ModuleWork(model,sink)
                    try:
                        out=run_arm(model,packet,slot['method'],sink)
                        slot.update(out)
                    except BaseException as exc:
                        slot.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc)))
                        raise
                    finally:
                        work.close()
                        slot['module_work']=work.counts
                    try:
                        if slot['status']=='encoded_h1_completed_not_physical':
                            tokens=contract()['processed_context_tokens_per_arm'][slot['method']]+9
                            require(all(work.counts[n]['requested_calls']==work.counts[n]['completed_calls']==2706 and
                                work.counts[n]['requested_samples']==work.counts[n]['completed_samples']==2706 and
                                work.counts[n]['requested_context_tokens']==work.counts[n]['completed_context_tokens']==tokens
                                for n in ('predictor','action_encoder')),'actual predictor/action hook coordinates differ')
                            require(work.counts['encoder']['requested_calls']==0 and slot['receipt']['backward_completed']==900,'encoding/backward coordinates differ')
                    except Exception as exc:
                        slot.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc)))
                    finally:
                        write(output/f'arm_{i:02d}.json',slot)
                    # Abort is retained. No fresh model or fallback retry.
                    if slot['status']=='failed':
                        raise RuntimeError('arm failed; remaining slots retained not_started')
            after=state_hash(model);unchanged=before==after
            require(unchanged,'model parameters/buffers mutated')
        except BaseException as exc:
            failure=dict(type=type(exc).__name__,message=str(exc))
            journal(dict(kind='run_failure',failure=failure))
        finally:
            if tf32_before is not None:
                torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=tf32_before
            try:
                if model is not None and before is not None:
                    after=state_hash(model);unchanged=before==after
                validate_prepared(prepared,expected_sha)
                for path,h in p['model_bindings'].items():
                    require(digest(path)==h,'end model/config byte change')
            except Exception as exc:
                failure=dict(previous=failure,type=type(exc).__name__,message=str(exc))
            completed=all(x['status']=='encoded_h1_completed_not_physical' for x in rows)
            report=dict(schema=NAME+'.report',status='h1_calibration_recorded' if completed and unchanged and failure is None else 'failed',
                protocol_id=identity,prepare_sha256=expected_sha,host=host,failure=failure,rows=rows,
                paired_h1=paired_h1(rows,p['roots']),
                parameters_buffers_before=before,parameters_buffers_after=after,model_unchanged=unchanged,
                contract=contract(),journal_sha256=digest(output/'events.jsonl'),new_physical_steps=0,
                new_independent_roots=0,P0_admission=False,P2_admission=False,
                replay_export='per-arm plan.commands is complete 15-step raw path; requires NEW separately reviewed guarded physical consumer; never autoexecute')
            write(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--prepared')
    parser.add_argument('--expected-prepare-sha256')
    parser.add_argument('--output',required=True)
    a=parser.parse_args()
    require(a.prepare != a.execute_cuda,'choose exactly --prepare or --execute-cuda')
    if a.prepare:
        prepare(a.root,a.output)
        print(io.canonical(dict(prepare=str(Path(a.output).absolute()/'prepare.json'),sha256=digest(Path(a.output)/'prepare.json'))))
    else:
        require(a.prepared and a.expected_prepare_sha256,'explicit prepared path and SHA required')
        result=execute(a.prepared,a.expected_prepare_sha256,a.output)
        print(result['status'])
        if result['status']=='failed':
            raise SystemExit(1)


if __name__=='__main__':
    main()
