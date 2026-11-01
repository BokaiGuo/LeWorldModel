"""Selected physical checkpoints -> eight native CLS192 encodings, MAIN CUDA only.

No sourcepack/certificate/candidate-bank readers, prediction, optimization or physics.
Prepare fails closed unless all four selected branches have genuine step5/10 frames.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as planner

NAME='p1_constructive_selected_rgb_cuda_20261027_v1'
PHYSICAL='p1_constructive_plan_physical_20261027_v1'
HANDOFF='p1_constructive_plan_handoff_20261027_v1'
ROOT=planner.ROOT
enc=planner.encode
require=enc.require
sha,canonical=enc.sha,enc.canonical
FLOOR,CAP,FINAL=200*1024**2,2*1024**2,128*1024
PLANNER_SHA='e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27'


def read(path,expected=None):
    if expected is not None: require(sha(path)==expected,'external SHA: '+str(path))
    return planner.read(path)


def local(directory,name):
    require(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat artifact path')
    return Path(directory)/name


def vector(value,n):
    a=np.asarray(value,dtype=float)
    require(a.shape==(n,) and np.isfinite(a).all(),'finite vector'+str(n))
    return a


def png(path,file_sha,pixel_sha):
    from PIL import Image
    require(sha(path)==file_sha and Path(path).stat().st_size<=1024**2,'bounded selected PNG SHA')
    with Image.open(path) as im:
        require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'original RGB224 PNG, no conversion')
        a=np.array(im)
    require(a.dtype==np.uint8 and enc.digest(a.tobytes())==pixel_sha and sha(path)==file_sha,'pixel SHA and file readback')
    return a


def journal(path,expected):
    require(sha(path)==expected and Path(path).stat().st_size<=8*1024**2,'complete bounded physical journal SHA')
    raw=Path(path).read_bytes(); require(raw.endswith(b'\n'),'no torn physical journal')
    events=[];previous=None
    for index,line in enumerate(raw.splitlines()):
        e=planner.decode(line)
        require(e['index']==index and e['previous_sha256']==previous and
            enc.core.digest({k:v for k,v in e.items() if k!='event_sha256'})==e['event_sha256'],'physical hash chain')
        previous=e['event_sha256'];events.append(e)
    require(events and events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished','physical terminal journal')
    return events


def checkpoint(row,step):
    require([x['step'] for x in row['checkpoints']]==[0,5,10] and all(x['phase']=='plan' for x in row['checkpoints']),
        'missing/duplicate/partial checkpoints; require exactly plan0/5/10, no substitution')
    c=deepcopy(next(x for x in row['checkpoints'] if x['step']==step))
    state=vector(c['state'],7)
    require(np.array_equal(state,vector(c['observation_state'],7)) and
        np.array_equal(state,vector(c['snapshot']['state'],7)),'checkpoint state/observation/snapshot provenance')
    return c


def selected_task(plan,task):
    """Validate only selected plan; do not parse its result/candidate bank."""
    t=task['gn_task'];terminal=plan['prediction'];root=t['root_context']
    tokens=enc.core.normalized_tokens(t,np.asarray(plan['commands']))
    prefix=dict(root,latent_history=root['latent_history'][1:]+[terminal['latent_history'][-2]],
        action_history=[root['action_history'][-1],tokens[0].tolist()])
    enc.core.check_prediction(t,dict(task_id=task['task_id'],logits=plan['logits'],commands=plan['commands'],
        contexts=[prefix,terminal],whole_replay=terminal,segmented_replay=[prefix,terminal],cost=plan['model_terminal_cost']))
    return [vector(prefix['latent_history'][-1],192).tolist(),vector(terminal['latent_history'][-1],192).tolist()]


def assemble(physical,events,prepared,planner_report,plans,tasks):
    """Pure RAM contract join; no file discovery or privileged loaders."""
    require(physical['schema']==PHYSICAL+'.report' and physical['status']=='physical_calibration_recorded' and
        physical['errors']==[] and physical['P1_admission'] is False,'complete error-free physical receipt required')
    require(events[0]['prepare']==prepared and events[-1]['status']==physical['status'] and
        events[-1]['counts']==physical['counts'] and events[-1]['errors']==[],'physical prepare/terminal receipt alignment')
    expected=[(i,a) for i in range(2) for a in planner.ARMS]
    require([(r['root_slot'],r['arm']) for r in physical['rows']]==expected and
        [(p['root_slot'],p['arm']) for p in prepared['plans']]==expected,'exact two roots x GN/D2P identity/order')
    require(prepared['schema']==PHYSICAL+'.prepare' and prepared['sourcepack_sha256']==physical['sourcepack_sha256'] and
        prepared['contract']==physical['contract'] and prepared['contract']['with_target'] is False and
        prepared['contract']['own_horizon']==10 and prepared['contract']['branches']==4,'same physical source/control contract')
    counts={k:dict(attempted=0,completed=0) for k in physical['counts']}
    stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            k=e['resource'];field='attempted' if e['kind']=='operation_attempted' else 'completed';counts[k][field]+=1
            if field=='attempted':stack.append(k)
            else:require(stack and stack.pop()==k,'balanced physical operation nesting')
    require(not stack and counts==physical['counts'] and all(x['attempted']==x['completed'] for x in counts.values()),'actual physical operation census')
    require(counts=={k:dict(attempted=v,completed=v) for k,v in dict(reset=8,public_step=80,
        explicit_render=88,goal_render=8,set_state=16,physics_step=816,goal_bind=4).items()},'full four-branch physical budget, no missing operation')
    images=[]
    for b,(row,selection,plan,task) in enumerate(zip(physical['rows'],prepared['plans'],plans,tasks,strict=True)):
        require(row['status'] in ('horizon_completed','native_success') and row['first_error'] is None and
            row['secondary_errors']==[] and row['initial_exact'] is True and row['full_horizon'] is True and
            row['own_attempted']==row['own_returned']==10 and row['warmup_returned']==10 and not row['truncated'],
            'missing full10 endpoint: early success/partial is not substituted')
        require(selection['selection_digest']==row['selection_digest'] and
            enc.core.digest({k:v for k,v in selection.items() if k!='selection_digest'})==row['selection_digest'] and
            selection['commands']==plan['commands'] and selection['arm']==plan['arm'] and task['task_id']==plan['task_id'],
            'same selected raw commands/task/arm before physics')
        closed=[e for e in events if e['kind']=='branch_closed' and e['receipt'].get('selection_digest')==row['selection_digest']]
        require(len(closed)==1 and closed[0]['receipt']==row,'durable physical row receipt')
        starts=[e for e in events if e['kind']=='branch_started' and e.get('branch')==b]
        require(len(starts)==1 and starts[0]['plan']==selection,'exact precommitted branch')
        transitions=[e for e in events if e['kind']=='returned_transition' and e.get('branch')==b and e['phase']=='plan']
        require([e['step'] for e in transitions]==list(range(1,11)) and
            [e['command'] for e in transitions]==plan['commands'],'all10 executed selected commands')
        predictions=selected_task(plan,task);t=task['gn_task']
        rootc=checkpoint(row,0)
        require(rootc['state']==row['root_state'] and checkpoint(row,10)['state']==row['final_state'],'root/terminal state identity')
        root_hash=enc.digest(np.ascontiguousarray(np.concatenate((vector(row['root_state'],7),vector(row['goal'],7))),dtype=np.float64).tobytes())
        require(prepared['source_slots'][row['root_slot']]==dict(root_slot=row['root_slot'],root_sha256=root_hash),'physical native root+goal identity')
        attached=[e for e in events if e['kind']=='goal_attached' and e.get('branch')==b]
        require(len(attached)==1 and attached[0]['goal']==row['goal'] and
            attached[0]['goal_rgb_sha256']==t['goal_source_sha256'] and attached[0]['snapshot']==rootc['snapshot'] and
            all(x['returned_goal_state']==row['goal'] and x['returned_goal_rgb_sha256']==t['goal_source_sha256'] for x in transitions),
            'actual original task goal RGB/state bound throughout execution')
        for j,step in enumerate((5,10)):
            c=checkpoint(row,step)
            matches=[e for e in events if e['kind']=='fresh_observation' and e.get('branch')==b and e['receipt']==c]
            guards=[e for e in events if e['kind']=='guard_receipt' and e.get('branch')==b and e['phase']=='plan' and e['step']==step]
            require(len(matches)==len(guards)==1 and guards[0]['accepted'] is True and
                guards[0]['png']==c['png'] and guards[0]['file_sha256']==c['file_sha256'] and
                transitions[step-1]['state']==c['state'],'checkpoint journal/guard/transition provenance')
            images.append(dict(index=len(images),branch=b,root_slot=row['root_slot'],arm=row['arm'],task_id=task['task_id'],
                task_export=plan['task_export'],task_sha256=plan['task_sha256'],
                role='prefix5' if step==5 else 'terminal10',checkpoint=c,predicted=predictions[j],
                root=t['root_context']['latent_history'][-1],goal=t['goal'],Q='identity192',selection_digest=row['selection_digest']))
    require(len(images)==8,'exact eight images')
    return images


def build(physical_report,physical_sha,planner_report,planner_sha):
    arguments=dict(physical_report=str(Path(physical_report).absolute()),physical_sha=physical_sha,
        planner_report=str(Path(planner_report).absolute()),planner_sha=planner_sha)
    require(sha(planner.__file__)==PLANNER_SHA,'reviewed frozen planner source')
    pr=Path(arguments['planner_report']);r=read(pr,planner_sha)
    require(r['schema']==planner.NAME+'.report' and r['status']=='model_plans_sealed_not_admitted' and r['errors']==[] and
        r['commands_released'] is True and r['model_unchanged'] is True and r['source_verified'] is True and
        r['model_hash_before']==r['model_hash_after']==enc.MODEL_TENSOR_SHA,'complete frozen planner/model receipt')
    pp=planner.verify(r['prepared'],r['prepare_sha256'])
    require(len(pp['slots'])==2 and r['contract']==pp['contract'] and
        [(x['order'],x['arm']) for x in r['rows']]==[(i,a) for i in range(2) for a in planner.ARMS],'four approved planner arms')
    bindings={str(pr):planner_sha,str(Path(r['prepared']).absolute()):r['prepare_sha256']}
    bindings.update(pp['bindings_sha256']);plans=[];tasks=[];source_slots=[]
    for row in r['rows']:
        require(row['status']=='model_plan_sealed','all planner selections complete')
        planner.exact_counts(row['resources'],planner.budgets()[row['arm']])
        planner.exact_counts(row['AD_API_resources'],planner.contract(1)['AD_API_per_root'][row['arm']])
        path=local(pr.parent,row['plan_file']);plan=read(path,r['artifact_sha256'][row['plan_file']])
        require(enc.core.digest({k:v for k,v in plan.items() if k!='selection_sha256'})==plan['selection_sha256']==row['selection_sha256'] and
            plan['order']==row['order'] and plan['arm']==row['arm'] and plan['selection_uses_physics'] is False and
            plan['physical_execution'] is False and plan['with_target'] is False,'selected plan seal and role')
        slot=pp['slots'][row['order']]
        require(plan['task_export']==slot['task_export'] and plan['task_sha256']==slot['task_sha256'] and plan['task_id']==slot['task_id'],'own approved task join')
        task=planner.load_export(slot['task_export'],slot['task_sha256']);export=read(slot['task_export'],slot['task_sha256'])
        source_slots.append(export['source_slot']);plans.append(plan);tasks.append(task);bindings[str(path)]=sha(path)
    require(source_slots==[0,0,1,1],'this v1 fixed source root order')
    ph=Path(arguments['physical_report']);physical=read(ph,physical_sha);bindings[str(ph)]=physical_sha
    bp=ph.parent/'execution_binding.json';binding=read(bp);bindings[str(bp)]=sha(bp)
    prepared=read(binding['prepared'],binding['prepare_sha256']);bindings[str(Path(binding['prepared']).absolute())]=binding['prepare_sha256']
    # Verify physical implementation seals only, never open its privileged data bindings.
    for path,h in prepared['bindings'].items():
        if Path(path).suffix in ('.py','.md'):
            require(sha(path)==h,'physical source seal: '+path);bindings[str(Path(path).absolute())]=h
    jp=ph.parent/'journal.jsonl';events=journal(jp,physical['journal_sha256']);bindings[str(jp)]=physical['journal_sha256']
    images=assemble(physical,events,prepared,r,plans,tasks)
    # Authenticate the actual physical handoff, without its sourcepack/certificate reader.
    for s,plan,slot in zip(prepared['plans'],plans,source_slots,strict=True):
        hpath=Path(s['compiler_report']);h=read(hpath,s['compiler_report_SHA']);bindings[str(hpath.absolute())]=s['compiler_report_SHA']
        require(h['schema']==HANDOFF+'.handoff' and h['status']=='planner_verified_handoff_not_admitted' and h['errors']==[] and
            h['planner_report_sha256']==planner_sha and Path(h['planner_output']).absolute()==pr.parent,'physical handoff belongs to explicit planner')
        hr=next((x for x in h['rows'] if x['source_slot']==slot and x['arm']==plan['arm']),None)
        require(hr is not None and hr['commands']==plan['commands'] and hr['task_id']==plan['task_id'] and
            hr['planner_selection_sha256']==plan['selection_sha256'] and s['root_slot']==slot,'handoff root/task/selection identity')
    for item in images:
        c=item['checkpoint'];path=local(ph.parent,c['png']);png(path,c['file_sha256'],c['pixel_sha256'])
        item['png_path']=str(path);bindings[str(path)]=c['file_sha256']
    files=planner.source_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(Path(f).resolve()):sha(f) for f in files})
    return dict(schema=NAME+'.prepare',arguments=arguments,images=images,bindings=bindings,
        policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,
        contract=dict(images=8,encoder=8,projector=8,predict=0,action_encoder=0,AD=0,physics=0,training=False,
            Q='identity192',output_cap_bytes=CAP,disk_floor_bytes=FLOOR,P1_admission=False,
            inherited_costs='original eight root/goal image encodings; GN78 vs D2P66 predict/root plus AD; physical80public/816space.step; separate, not refunded',
            selection='already selected GN/D2P only; no candidate bank or certificate access',
            partial='missing prefix5 or terminal10 rejects before any CUDA; no early endpoint substitution'))


def prepare(output,**arguments):
    p=build(**arguments);path=Path(output);raw=canonical(p)
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive small prepare +200MiB')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path.absolute()),sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    p=read(path,expected);require(p==build(**p['arguments']),'complete selected frame/task/source reconstruction');return p


class Store:
    def __init__(self,output):
        self.output=Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW2MiB +200MiB reserve')
        self.output.mkdir()
    def write(self,name,raw,append=False,final=False):
        path=local(self.output,name);require(not path.is_symlink(),'no symlink output')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'whole output cap/floor including partial')
        with path.open('ab' if append else 'xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def event(self,e):self.write('events.jsonl',canonical(e),append=True)
    def put(self,name,value,final=False):self.write(name,canonical(value),final=final)
    def latent(self,index,z):
        enc.native(z);b=BytesIO();np.save(b,z,allow_pickle=False);name=f'image_{index:02d}.native192.npy'
        self.write(name,b.getvalue());return dict(file=name,sha256=sha(self.output/name),dtype='float32',shape=[192])


def compare(item,z):
    enc.native(z);real=z.astype(float);pred=vector(item['predicted'],192);root=vector(item['root'],192);goal=vector(item['goal'],192)
    q=lambda a,b:float(np.mean((a-b)**2))
    return dict(predicted_Q=q(pred,goal),actual_Q=q(real,goal),root_Q=q(root,goal),
        predicted_vs_real_mse=q(pred,real),actual_vs_root_mse=q(real,root),predicted_vs_root_mse=q(pred,root),
        actual_Q_change=q(real,goal)-q(root,goal),predicted_Q_change=q(pred,goal)-q(root,goal),
        Q_improvement_sign_reversed=bool((q(real,goal)-q(root,goal))*(q(pred,goal)-q(root,goal))<0))


def encode_images(p,callback,ledger,store,rows):
    for item,row in zip(p['images'],rows,strict=True):
        ledger.image=item['index'];row['status']='attempted'
        store.event(dict(kind='image_attempt',provenance=item))
        rgb=png(item['png_path'],item['checkpoint']['file_sha256'],item['checkpoint']['pixel_sha256'])
        z=callback(rgb.copy());enc.native(z)
        require([(e['resource'],e['completed']) for e in ledger.entries if e['image']==item['index']]==[('encoder',True),('projector',True)],'both native module completions, no inferred count')
        row.update(status='encoded',latent=store.latent(item['index'],z),metrics=compare(item,z))
        store.event(dict(kind='image_saved',receipt=deepcopy(row)));ledger.image=None


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    ledger=enc.Ledger(2,store.event);rows=[dict(index=i['index'],branch=i['branch'],root_slot=i['root_slot'],arm=i['arm'],
        role=i['role'],task_id=i['task_id'],task_export=i['task_export'],task_sha256=i['task_sha256'],
        status='not_started',provenance=i) for i in p['images']]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen tensor SHA')
        transform=p0.image_transform();ledger.sync=lambda:torch.cuda.synchronize(device);ledger.no_grad=lambda:not torch.is_grad_enabled()
        with sdpa_kernel(SDPBackend.MATH),enc.guard(model,torch,ledger,errors):
            def callback(rgb):
                with torch.inference_mode():
                    x=p0._image_tensor(rgb,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native batch1 CUDA RGB input')
                    z=model.encode({'pixels':x})['emb']
                    require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native projected CLS192')
                    return z[0,0].detach().cpu().numpy().copy()
            encode_images(p,callback,ledger,store,rows)
    except BaseException as exc:
        errors.insert(0,enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(enc.error('error_journal',e))
    finally:
        ledger.image=None
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:errors.append(enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(enc.error('final_source',e))
        complete=not errors and not ledger.blocked and source_ok and before is not None and before==after and all(x['status']=='encoded' for x in rows) and ledger.counts()=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')}
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try:hashes[f.name]=sha(f)
                except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        result=dict(schema=NAME+'.report',status='selected_rgb_encoded_not_admitted' if complete else 'partial_or_failed',
            prepare_sha256=expected,inputs=p['arguments'],rows=rows,resources=ledger.counts(),errors=errors,
            first_error=errors[0] if errors else None,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,host=host,artifact_sha256=hashes,contract=p['contract'],
            new_predict=0,new_AD=0,new_physics=0,training=False,P1_admission=False)
        try:store.put('report.json',result,final=True)
        except BaseException as e:errors.append(enc.error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial output retained; no retry: '+str(errors));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for key in ('physical-report','physical-sha','planner-report','planner-sha','prepared','prepare-sha','output'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output,physical_report=a.physical_report,physical_sha=a.physical_sha,planner_report=a.planner_report,planner_sha=a.planner_sha)
    elif a.command=='verify':r=dict(status='verified_no_execution',images=len(verify(a.prepared,a.prepare_sha)['images']))
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
