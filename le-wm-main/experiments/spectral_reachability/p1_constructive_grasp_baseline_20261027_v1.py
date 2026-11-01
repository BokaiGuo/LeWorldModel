"""GRASP mechanism, explicit real-H3/bounded-action extension. MAIN-only CUDA.

Official LeWM 94be3c4 / GRASP 64f988a: virtual states, soft dynamics,
state-input stopgrad, state noise, action-only serial sync, joint Adam reset.
Not official H1 numerical equivalence, paper reproduction or P1 admission.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sysconfig
import zlib
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as base

NAME='p1_constructive_grasp_baseline_20261027_v1'
ARM='GRASP_H3_bounded'
ROOT=base.ROOT
gn,enc,core=base.gn,base.encode,base.encode.core
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
BASE_SHA='e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27'
GN_SHA='825c148cf515ad42d59b187d0d8c83984826db67f58d66b2e2d4f289dcecf70a'
CAP,FLOOR,FINAL=64*1024**2,200*1024**2,256*1024
EXPORTS=[str(ROOT/f'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_{s:02d}.json.gz') for s in (0,1)]
EXPORT_SHAS=['714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a',
             '70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2']
OFFICIAL=ROOT/'results/p2_grasp_official_contract_20261027_v1/sources'
REVISIONS={'':'64f988aebf04ad9f674f9821bc6d2b9e22879105','lewm__':'94be3c42aa2f0867c6cb1e38bda498b293b6921e'}
PINS={
 'LICENSE.source.txt':'d54bc4cd9f49bd38a602c6aaff6ff3ba60062d026039e8844ebf4863ac9856c2',
 'grasp_pseudocode.py.source.txt':'6ae8616e8fb950c7d5bdef49f34e3df234b43cddfab8ea273a87493238932ad1',
 'commit.source.txt':'f1699f1b478f06db33422ea6e0fd7d0f69c34153576f6e134b1976803f5a2817',
 'tree.source.txt':'cb2b7173ffa5d865214c646d1f4ace9fca1fbe112e6f092a73e2f786e3940e5a',
 'lewm__grasp_solver.py.source.txt':'53e7761019e3eed0d776df46aa6e0f8e84c6a084d9588802b2528b006be690df',
 'lewm__config__eval__solver__grasp.yaml.source.txt':'e424acb6c4d5bbfdaf4ec24323ad3c5b3a6948ea13e4042ee243cf3a15d424f5',
 'lewm__commit.source.txt':'cd50b761547afad08fd09a229f73a08f3a44acea80158cbfcd979e0c7ceaef2d',
 'lewm__tree.source.txt':'fa98a31684ead1c3355e3d9d21c053ea46f53c0c0726939c1fb1db445ba4f100'}


@dataclass(frozen=True)
class Config:
    steps:int=750
    lr_s:float=.1
    lr_a:float=.005
    state_noise:float=.005
    sync_interval:int=50
    sync_steps:int=10
    sync_lr:float=.05
    def __post_init__(self):
        for k in ('steps','sync_interval','sync_steps'):
            require(type(getattr(self,k)) is int and getattr(self,k)>=1,'positive integer '+k)
        for k in ('lr_s','lr_a','state_noise','sync_lr'):
            v=getattr(self,k);require(type(v) in (int,float) and np.isfinite(v) and v>=0 and (k=='state_noise' or v>0),'finite rate '+k)


def schedule(cfg):return [k+1 for k in range(cfg.steps) if (k+1)%cfg.sync_interval==0 and k>0]
def budget(cfg=Config()):
    n=2*(cfg.steps+len(schedule(cfg))*cfg.sync_steps)+2
    return dict(predict=n,action_encoder=n,backward=cfg.steps+len(schedule(cfg))*cfg.sync_steps)
def seed(slot):
    require(type(slot) is int and slot in (0,1),'fixed source slot');return 20261027+101*slot


def contract():
    b=budget()
    return dict(arms=[ARM],roots=2,horizon=2,packet_steps=5,config=asdict(Config()),
        seed='20261027+101*source_slot; local torch.Generator on execution device',
        virtual_states='one full192 intermediate latent, initialized midpoint(z10,goal); never a witness',
        soft_dynamics='sum_edge MSE(pred,virtual_next)+sum_edge MSE(pred,goal); final next=goal',
        history='edge0 original H3; edge1 [observed5,observed10,virtual5], shifted actual history plus live first action token',
        detach='all lifted state inputs detached, residual target virtual state live, both action tokens live; serial sync full chain',
        noise='iid Gaussian on virtual state only after joint Adam; fixed sigma; no action noise/restarts',
        sync='(k+1)%50==0 and k>0; fresh action-only Adam10, fullH2 terminal MSE; replace actions/reset joint Adam, no virtual refresh',
        bounds='project each updated float32 logit packet with sealed projected_logits; NumPy float32 tanh forward, analytic tanh chain once',
        normalization='task original float32 (raw-mean)/std; no /100, no inverse-normalizer or action clipping',
        final='last optimized action, no best/outcome selection; freeze then fresh serial fullH2 verification',
        initialization='zero logits/raw; not official zero-normalized-action initialization',
        per_root_budget=b,total_budget={k:2*v for k,v in b.items()},
        processed_context_tokens_per_root={k:3*b[k] for k in ('predict','action_encoder')},
        lifted_per_root=1500,sync_per_root=300,verification_per_root=2,
        new_images=0,new_physics=0,new_Q=False,training=False,AD_jacobian=0,SVD=0,
        inherited_encoding='separate historical acquisition; not free total pipeline',
        official_revisions=REVISIONS,official_equivalence=False,paper_results_code_commit='unknown',
        adaptation='history-preserving bounded-coordinate GRASP mechanism extension, NOT official H1/DINO numerical equivalence',
        full_pipeline_equal_compute=False,forward_count_matches_D2P=False,
        no_restarts=True,no_early_exit=True,feedback='open_loop_model_only',with_target=False,
        P1_admission=False,new_independent_confirmation=False,scientific_pass_threshold=None,
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False,optimizer='torch Adam default betas/eps, foreach=False,fused=False'),
        output_cap_bytes=CAP,reserve_bytes=FLOOR)


def official_bindings():
    """Only immutable small text metadata, no remote execution or outcome read."""
    bodies={}
    for name,h in PINS.items():
        p=OFFICIAL/name;require(p.is_file() and not p.is_symlink() and p.stat().st_size<=256*1024,'missing official asset: '+str(p))
        require(sha(p)==h,'official asset SHA: '+name);bodies[name]=p.read_bytes()
    for prefix,rev in REVISIONS.items():
        commit=json.loads(bodies[prefix+'commit.source.txt']);tree=json.loads(bodies[prefix+'tree.source.txt'])
        require(commit['sha']==rev and not tree.get('truncated') and tree['sha'] in (rev,commit['commit']['tree']['sha']),'official version/tree')
        entries={r['path']:r for r in tree['tree'] if r['type']=='blob'}
        names=('grasp_solver.py','config/eval/solver/grasp.yaml') if prefix else ('LICENSE','grasp_pseudocode.py')
        for name in names:
            raw=bodies[prefix+name.replace('/','__')+'.source.txt']
            h=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            require(h==entries[name]['sha'] and len(raw)==entries[name]['size'],'official Git blob: '+name)
    return {str((OFFICIAL/k).resolve()):v for k,v in PINS.items()}


def build():
    require(sha(base.__file__)==BASE_SHA and sha(gn.__file__)==GN_SHA,'frozen blind/GN primitive sources')
    official=official_bindings();p=base.build(EXPORTS,EXPORT_SHAS);slots=[]
    for i,s in enumerate(p['slots']):
        e=read(s['task_export'])
        require(e['source_slot']==i and e['blind_sha256']==e['blind_source_sha256']==list(enc.APPROVED_INPUTS.values())[i],'fixed blind identity')
        slots.append(dict(s,source_slot=i,blind_sha256=e['blind_sha256']))
    files=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    site=Path(sysconfig.get_paths()['purelib'])
    files += [site/x for x in ('torch/optim/adam.py','torch/optim/optimizer.py','torch/optim/_functional.py','torch/random.py')]
    bindings={**p['bindings_sha256'],**official,**{str(f.resolve()):sha(f) for f in files}}
    for path,h in bindings.items():require(sha(path)==h,'prepare source TOCTOU')
    return dict(schema=NAME+'.prepare',slots=slots,policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        contract=contract(),bindings_sha256=bindings,global_freshness='unknown_existing_two_calibration_tasks')


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive small prepare +200MiB')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])
def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path);require(p==build(),'full input/source/protocol reconstruction');return p


class Store(base.Store):
    """Inherited gzip/fsync protocol, independent new capacity; no global patch."""
    def __init__(self,output):
        self.output=Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW64MiB+200MiB gate')
        self.output.mkdir();self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat output')
        used=sum(f.stat().st_size for f in self.output.iterdir() if f.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'cap/floor including partial')
        with (self.output/name).open('ab' if append else 'xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


class Meter(gn.Meter):
    def __init__(self,sink,sync=lambda:None,cfg=Config()):
        gn.screen.Meter.__init__(self,budget(cfg),sink);self.sync=sync


def host(t):return t.detach().cpu().tolist()


def exact_raw(x):
    """Exact existing NumPy raw forward; first-order analytical tanh chain once."""
    import torch
    a=np.tanh(x.detach().cpu().numpy().astype(np.float32))
    raw=torch.tensor(a,device=x.device,dtype=x.dtype)
    return raw+(x-x.detach())*(1-raw.square())


def projected(x,cap):
    import torch
    require(bool(torch.isfinite(x).all()),'nonfinite Adam logits')
    proposal=host(x)
    with torch.no_grad():x.copy_(x.new_tensor(gn.screen.projected_logits(proposal,cap)))
    return proposal


class Dynamics:
    """H3 lifted/serial tensor algebra. Production callback is NativeCuda._step.

    analytic_fixture=True is only for non-Module algebra doubles in tests;
    run/CLI never accept model/callback injection or select this mode.
    """
    def __init__(self,task,device,step,*,analytic_fixture=False):
        import torch
        core.validate_task(task)
        require(device.type=='cuda' or analytic_fixture,'no CPU neural fallback')
        require(not isinstance(getattr(step,'__self__',None),torch.nn.Module),'no Module analytic callback')
        self.task,self.step,self.analytic_fixture=task,step,analytic_fixture
        self.z=torch.tensor(task['root_context']['latent_history'],device=device,dtype=torch.float32)[None]
        self.h=torch.tensor(task['root_context']['action_history'],device=device,dtype=torch.float32)[None]
        self.goal=torch.tensor(task['goal'],device=device,dtype=torch.float32)[None]
        self.mean=self.z.new_tensor(task['action_mean']).repeat(5);self.std=self.z.new_tensor(task['action_std']).repeat(5)
    def advance(self,z,h,raw,meter,packet):
        meter.emit('transition_attempt',packet=packet,latent_history=host(z),action_history=host(h),commands=host(raw.reshape(5,2)))
        after,between=self.step(z,h,raw.reshape(5,2),meter)
        import torch
        require(tuple(after.shape)==(1,3,192) and tuple(between.shape)==(1,2,10) and bool(torch.isfinite(after).all()) and bool(torch.isfinite(between).all()),'finite H3 transition')
        require(torch.equal(after[:,:2],z[:,1:]) and torch.equal(between[:,:1],h[:,1:]),'literal rolling H3')
        token=(raw.reshape(10)-self.mean)/self.std
        require(torch.equal(between[:,1],token[None]),'normalizer/raw scale exact')
        meter.emit('transition_completed',packet=packet,context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=host(after[0]),action_history=host(between[0])))
        return after,between
    def soft(self,v,x,meter):
        import torch
        raw=exact_raw(x)
        z1,h1=self.advance(self.z.detach(),self.h.detach(),raw[0],meter,0)
        lifted=torch.cat((self.z[:,1:],v[:,None]),dim=1)
        z2,_=self.advance(lifted.detach(),h1,raw[1],meter,1)
        p0,p1=z1[:,-1],z2[:,-1]
        dyn=(p0-v).square().mean()+(p1-self.goal).square().mean()
        goal=(p0-self.goal).square().mean()+(p1-self.goal).square().mean()
        return dyn+goal,dyn,goal
    def serial(self,x,meter):
        z,h=self.z,self.h;paths=[];raw=exact_raw(x)
        for k in range(2):
            z,h=self.advance(z,h,raw[k],meter,k)
            paths.append(dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                latent_history=host(z[0]),action_history=host(h[0])))
        return z[:,-1],paths


def optimize(task,slot,dynamics,meter,*,cfg=Config(),analytic_fixture=False):
    import torch
    core.validate_task(task);require(analytic_fixture==dynamics.analytic_fixture,'explicit algebra fixture boundary')
    require(analytic_fixture or dynamics.z.device.type=='cuda','CUDA only')
    require(core.task_id(task)==core.task_id(dynamics.task),'same bound task')
    generator=torch.Generator(device=dynamics.z.device).manual_seed(seed(slot))
    v=((dynamics.z[:,-1]+dynamics.goal)/2).detach().clone().requires_grad_(True)
    x=dynamics.z.new_zeros((2,10),requires_grad=True)
    def adam(params,lr):return torch.optim.Adam(params,lr=lr,foreach=False,fused=False)
    def joint():return adam([{'params':v,'lr':cfg.lr_s},{'params':x,'lr':cfg.lr_a}],cfg.lr_a)
    def backward(loss,variables):
        require(bool(torch.isfinite(loss)),'finite optimization loss')
        meter.call('backward',loss.backward)
        require(all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in variables),'finite live optimizer gradients')
    opt=joint();syncs=[];losses=[]
    meter.emit('initialized',virtual=host(v),logits=host(x),seed=seed(slot),config=asdict(cfg))
    for k in range(cfg.steps):
        meter.group=f'joint_{k}';before=host(v);opt.zero_grad()
        meter.emit('joint_request',virtual=before,logits=host(x))
        loss,dyn,goal=dynamics.soft(v,x,meter);backward(loss,(v,x))
        grads=dict(virtual=host(v.grad),logits=host(x.grad))
        opt.step();proposal=projected(x,task['packet_logit_cap'])
        with torch.no_grad():
            noise=torch.randn(v.shape,device=v.device,dtype=v.dtype,generator=generator)*cfg.state_noise
            v.add_(noise)
        require(bool(torch.isfinite(v).all()),'finite updated virtual state')
        record=dict(iteration=k,pre_loss=float(loss.detach()),dynamics_loss=float(dyn.detach()),goal_loss=float(goal.detach()))
        losses.append(record)
        meter.emit('joint_completed',**record,gradients=grads,unprojected_logits=proposal,logits=host(x),virtual=host(v),noise=host(noise))
        if (k+1)%cfg.sync_interval==0 and k>0:
            syncs.append(k+1);x=x.detach().clone().requires_grad_(True);sopt=adam([x],cfg.sync_lr)
            meter.emit('sync_started',after_iteration=k+1,virtual_unchanged=host(v),logits=host(x))
            for j in range(cfg.sync_steps):
                meter.group=f'sync_{k+1}_{j}';sopt.zero_grad();meter.emit('sync_request',logits=host(x))
                terminal,_=dynamics.serial(x,meter);sloss=(terminal-dynamics.goal).square().mean()
                backward(sloss,(x,));grad=host(x.grad);sopt.step();proposal=projected(x,task['packet_logit_cap'])
                meter.emit('sync_completed',pre_loss=float(sloss.detach()),gradient=grad,unprojected_logits=proposal,logits=host(x))
            opt=joint();meter.emit('joint_optimizer_reset',after_iteration=k+1,virtual_unchanged=host(v),logits=host(x))
    raw=np.tanh(x.detach().cpu().numpy().astype(np.float32)).reshape(10,2)
    meter.emit('actions_frozen_before_verification',logits=host(x),commands=raw.astype(float).tolist())
    meter.group='verification'
    with torch.no_grad():terminal,path=dynamics.serial(x,meter)
    gn.saved.check_path(task,raw,path);base.exact_counts(meter.counts(),budget(cfg))
    return dict(task_id=core.task_id(task),source_slot=slot,arm=ARM,seed=seed(slot),config=asdict(cfg),
        logits=host(x),commands=raw.astype(float).tolist(),virtual=host(v),verification=path,
        terminal_cost=core.model_cost(task,host(terminal[0])),iterations=losses,sync_after_iterations=syncs,
        resources=meter.counts(),official_equivalence=False,P1_admission=False)


def choice(task,result,slot,cfg=Config()):
    require(result['task_id']==core.task_id(task) and result['source_slot']==slot and result['arm']==ARM and
        result['seed']==seed(slot) and result['config']==asdict(cfg),'bound GRASP task/slot/config')
    base.exact_counts(result['resources'],budget(cfg));x=core.array(result['logits'],(2,10));raw=core.array(result['commands'],(10,2))
    require(np.array_equal(x,x.astype(np.float32)) and (np.linalg.norm(x,axis=1)<=task['packet_logit_cap']).all(),'literal feasible logits')
    require(np.array_equal(raw,np.tanh(x.astype(np.float32)).reshape(10,2)),'exact commands')
    gn.saved.check_path(task,raw,result['verification']);core.array(result['virtual'],(1,192))
    require(len(result['iterations'])==cfg.steps and [r['iteration'] for r in result['iterations']]==list(range(cfg.steps)) and
        all(np.isfinite(r[k]) and r[k]>=0 for r in result['iterations'] for k in ('pre_loss','dynamics_loss','goal_loss')) and
        result['sync_after_iterations']==schedule(cfg),'complete GRASP iterations and sync schedule')
    require(result['terminal_cost']==core.model_cost(task,result['verification'][-1]['latent_history'][-1]) and
        result['official_equivalence'] is False and result['P1_admission'] is False,'terminal ownQ and evidence boundary')
    return dict(task_id=core.task_id(task),logits=result['logits'],commands=result['commands'],prediction=result['verification'][-1],model_terminal_cost=result['terminal_cost'])


def execute(task,slot,model,device,torch,sink,row):
    meter=Meter(sink,lambda:torch.cuda.synchronize(device))
    try:
        backend=gn.screen.NativeCuda(model,device,task)
        dynamics=Dynamics(task,device,lambda z,h,a,m:backend._step(task,z,h,a))
        with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False),backend.guard(meter):
            return optimize(task,slot,dynamics,meter)
    finally:row['resources']=meter.counts()


def plan_all(p,store,runner,rows):
    for row in rows:
        slot=p['slots'][row['source_slot']];task=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        row['status']='attempted'
        def sink(e):store.event(dict(source_slot=slot['source_slot'],arm=ARM,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',seed=seed(slot['source_slot']),task_sha256=slot['task_sha256']))
        result=runner(task,slot['source_slot'],sink,row);selected=choice(task,result,slot['source_slot'])
        base.exact_counts(row['resources'],budget());stem=f'root_{slot["source_slot"]}_{ARM}'
        store.put(stem+'.result.json.gz',result)
        plan=dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=ARM,blind_sha256=slot['blind_sha256'],
            task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],
            with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**selected)
        plan['selection_sha256']=core.digest(plan);store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host_info=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=ARM,blind_sha256=s['blind_sha256'],task_id=s['task_id'],status='not_started',commands=None,resources={}) for s in p['slots']]
    try:
        print('GRASP virtual-state H3 bounded extension; official-equivalence and equal-compute claims false; no physics/P1 admission.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host_info=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'frozen original model')
        with sdpa_kernel(SDPBackend.MATH):plan_all(p,store,lambda t,s,k,r:execute(t,s,model,device,torch,k,r),rows)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(enc.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:errors.append(enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(enc.error('final_source',e))
        try:store.close()
        except BaseException as e:errors.append(enc.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:errors.append(enc.error('artifact_listing',e));complete=False;files=[]
        for f in files:
            try:
                if f.is_file():hashes[f.name]=sha(f)
            except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in budget()}
        report=dict(schema=NAME+'.report',status='grasp_model_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,resources=resources,errors=errors,first_error=errors[0] if errors else None,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host_info,contract=p['contract'],commands_released=complete,new_images=0,new_physics=0,training=False,P1_admission=False,official_equivalence=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:complete=False;errors.append(enc.error('final_report',e));print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial preserved; no retries: '+str(errors));return report


def consume(output,expected):
    output=Path(output).absolute();require(sha(output/'report.json')==expected,'external actual report SHA');r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='grasp_model_plans_sealed_not_admitted' and r['commands_released'] is True and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==enc.MODEL_TENSOR_SHA and r['P1_admission'] is False,'complete CUDA closure')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and len(r['rows'])==2,'fixed two branches')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{ARM}.{ext}' for s in (0,1) for ext in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact output denominator')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'artifact SHA')
    plain,eof=base.inflate((output/'events.jsonl.gz').read_bytes(),limit=256*1024**2)
    require(eof and plain.endswith(b'\n'),'complete gzip journal');events=[base.decode(x) for x in plain.splitlines()]
    groups=[];plans=[]
    for e in events:
        require(set(e)=={'source_slot','arm','task_id','event'} and e['arm']==ARM,'only scoped completed journal')
        if not groups or groups[-1]!=e['source_slot']:groups.append(e['source_slot'])
    require(groups==[0,1],'fixed source order')
    for s,row in enumerate(r['rows']):
        slot=p['slots'][s];stem=f'root_{s}_{ARM}';task=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        require(row['source_slot']==s and row['arm']==ARM and row['task_id']==slot['task_id'] and row['blind_sha256']==slot['blind_sha256'] and
            row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','own complete plan identity')
        result=read(output/row['result_file']);selected=choice(task,result,s);plan=read(output/row['plan_file']);ident=plan.pop('selection_sha256')
        want=dict(schema=NAME+'.model_plan',source_slot=s,arm=ARM,blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
            task_contract_sha256=task['task_contract_sha256'],with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**selected)
        require(plan==want and core.digest(plan)==ident==row['selection_sha256'] and row['commands']==selected['commands'],'own sealed commands')
        base.exact_counts(row['resources'],budget())
        group=[e['event'] for e in events if e['source_slot']==s]
        require(all(e['task_id']==slot['task_id'] for e in events if e['source_slot']==s),'journal same task')
        journal_check(task,result,group)
        require(group[0]==dict(kind='arm_started',seed=seed(s),task_sha256=slot['task_sha256']) and group[-1]==dict(kind='model_selection_sealed',selection_sha256=ident,commands=selected['commands']),'durable lifecycle')
        plans.append(dict(plan,selection_sha256=ident,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],p['contract']['total_budget'])
    require(sha(output/'report.json')==expected,'final report TOCTOU')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final source binding')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'final artifact SHA')
    return dict(schema=NAME+'.consume',rows=plans,P1_admission=False,new_physics=0,official_equivalence=False)


def journal_check(task,result,events,cfg=Config()):
    """Producer consumer checks: saved bytes/counters, not a neural rerun."""
    attempts=[];done=[];pending=None;transitions=[];current=None;phase_edges={};frozen=False
    requests={};paths={};module_count=0;backward_phases=[]
    for e in events:
        kind=e['kind'];require('error' not in e and 'forbidden' not in kind,'no failure journal')
        if kind in ('joint_request','sync_request'):
            require(e['evaluation_id'] not in requests and current is None,'single phase request')
            requests[e['evaluation_id']]=e;paths[e['evaluation_id']]=[]
        elif kind=='operation_attempted':
            v=e['event'];require(pending is None and v['index']==len(attempts) and not v['completed'] and v['evaluation_id']==e['evaluation_id'],'sequential scoped actual attempt')
            if v['resource']=='backward':
                key=e['evaluation_id'];require(current is None and phase_edges.get(key)==2 and key not in backward_phases,'one backward after complete two-edge graph')
                backward_phases.append(key)
            else:
                require(current is not None and e['evaluation_id']==current['evaluation_id'] and module_count<2 and
                    v['resource']==('action_encoder','predict')[module_count],'native modules scoped to exact transition')
            attempts.append(v);pending=v
        elif kind=='operation_completed':
            v=e['event'];require(pending is not None and v==dict(pending,completed=True),'native/backward return identity');done.append(v);pending=None
            if v['resource']!='backward':module_count+=1
        elif kind=='transition_attempt':
            key=e['evaluation_id'];packet=e['packet'];require(current is None and pending is None and key in requests and packet in (0,1),'requested sequential transition')
            request=requests[key];raw=np.tanh(np.asarray(request['logits'],np.float32)).reshape(10,2)
            require(e['commands']==raw[5*packet:5*packet+5].astype(float).tolist(),'actual requested own raw packet')
            root=task['root_context']
            if packet==0:expected_z,expected_h=root['latent_history'],root['action_history']
            else:
                require(len(paths[key])==1,'previous packet returned')
                previous=paths[key][0];expected_h=previous['action_history']
                expected_z=root['latent_history'][1:]+request['virtual'] if key.startswith('joint_') else previous['latent_history']
            require(e['latent_history']==[expected_z] and e['action_history']==[expected_h],'actual original/lifted/serial H3 inputs')
            current=e;module_count=0
        elif kind=='transition_completed':
            require(current is not None and e['evaluation_id']==current['evaluation_id'] and e['packet']==current['packet'],'transition return scope')
            require(pending is None and module_count==2,'both actual native returns before transition completion')
            before=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                latent_history=current['latent_history'][0],action_history=current['action_history'][0])
            after=e['context'];core.context(before);core.context(after)
            a=core.array(current['commands'],(5,2));token=core.normalized_tokens(task,a)[0]
            require(after['latent_history'][:2]==before['latent_history'][1:] and after['action_history'][0]==before['action_history'][1] and np.array_equal(after['action_history'][1],token),'saved transition fullH3/normalizer')
            key=e['evaluation_id'];require(e['packet']==phase_edges.get(key,0),'both packets in order');phase_edges[key]=e['packet']+1
            transitions.append(e);paths[key].append(after);current=None
        elif kind=='actions_frozen_before_verification':
            require(not frozen and len(transitions)==budget(cfg)['predict']-2 and e['commands']==result['commands'] and e['logits']==result['logits'],'freeze after optimization before verification');frozen=True
            requests['verification']=e;paths['verification']=[]
    require(pending is None and current is None and frozen and len(transitions)==budget(cfg)['predict'],'complete planned transitions')
    base.exact_counts({k:dict(attempted=sum(v['resource']==k for v in attempts),completed=sum(v['resource']==k for v in done)) for k in budget(cfg)},budget(cfg))
    want=[]
    for i in range(cfg.steps):
        want += [f'joint_{i}']*2
        if i+1 in schedule(cfg):
            want += [f'sync_{i+1}_{j}' for j in range(cfg.sync_steps) for _ in range(2)]
    want += ['verification']*2
    require([e['evaluation_id'] for e in transitions]==want,'fixed optimization/sync/verification chronology')
    require([e['context'] for e in transitions[-2:]]==result['verification'],'actual final serial verification')
    require([e['iteration'] for e in events if e['kind']=='joint_completed']==list(range(cfg.steps)) and
        [e['after_iteration'] for e in events if e['kind']=='joint_optimizer_reset']==schedule(cfg),'full joint/sync/reset denominator')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(status='verified_without_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='consume':result=consume(a.output,a.report_sha)
    else:result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode().strip())


if __name__=='__main__':main()
