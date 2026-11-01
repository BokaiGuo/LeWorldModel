"""Generic selected physical RGB diagnostics: two opaque tasks x fixed1..4 arms.

Saved-data prepare/verify; MAIN-only CUDA run. No planner/physics execution,
certificate/sourcepack/witness reader, training, candidate search or selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_selected_rgb_cuda_20261027_v1 as old
from experiments.spectral_reachability import p1_constructive_plan_physical_20261027_v1 as physical_api

NAME='p1_constructive_selected_rgb_cuda_20261027_v2'
ROOT,enc,planner=old.ROOT,old.enc,old.planner
require,sha,canonical,read=old.require,old.sha,old.canonical,old.read
FLOOR,CAP,FINAL=old.FLOOR,old.CAP,old.FINAL
OLD_SHA='3508316e56515db38eca74c8516c4f434c50d273f1927830445767ef81400dbe'
PHYSICAL_SHA='9dea58dfc6418dd01bf9ab28de6d712189ca1154a5cf29b165efa4b8ef65ea96'
Store=old.Store


def fixed_arms(arms):
    require(isinstance(arms,list) and 1<=len(arms)<=4 and len(set(arms))==len(arms) and
        all(type(a) is str and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',a) for a in arms),'fixed1..4 distinct safe arms')
    return [(i,a) for i in range(2) for a in arms]


def counts_for(n):
    return {k:dict(attempted=v*n,completed=v*n) for k,v in dict(reset=2,public_step=20,
        explicit_render=22,goal_render=2,set_state=4,physics_step=204,goal_bind=1).items()}


def assemble(report,events,prepared,tasks,arms):
    """Array-only provenance join; task records are externally approved exports."""
    expected=fixed_arms(arms); n=len(expected)
    require(report['schema']==old.PHYSICAL+'.report' and report['status']=='physical_calibration_recorded' and
        report['errors']==[] and report['P1_admission'] is False,'complete error-free physical receipt')
    require(prepared['schema']==old.PHYSICAL+'.prepare' and prepared['sourcepack_sha256']==report['sourcepack_sha256'] and
        prepared['contract']==report['contract'] and prepared['contract']['with_target'] is False and
        prepared['contract']['own_horizon']==10 and prepared['contract']['branches']==n,'same physical contract')
    require(events[0]['kind']=='run_started' and events[0]['prepare']==prepared and
        events[-1]['kind']=='run_finished' and events[-1]['status']==report['status'] and
        events[-1]['counts']==report['counts'] and events[-1]['errors']==[],'terminal physical journal alignment')
    require([(r['root_slot'],r['arm']) for r in report['rows']]==expected and
        [(r['root_slot'],r['arm']) for r in prepared['plans']]==expected and
        all(type(r['root_slot']) is int for r in report['rows']+prepared['plans']),'complete fixed arms denominator/order')
    require([t['source_slot'] for t in tasks]==[0,1],'two opaque approved task slots')
    counts={k:dict(attempted=0,completed=0) for k in report['counts']}; stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            k=e['resource']; field='attempted' if e['kind']=='operation_attempted' else 'completed'
            counts[k][field]+=1
            if field=='attempted':stack.append(k)
            else:require(stack and stack.pop()==k,'balanced physical operation nesting')
    require(canonical(counts)==canonical(counts_for(n)) and not stack and
        canonical(counts)==canonical(report['counts']),'actual physical census')
    require(len([e for e in events if e['kind']=='branch_started'])==n and
        len([e for e in events if e['kind']=='branch_closed'])==n,'no extra/missing branches')
    images=[]
    for b,(row,selection) in enumerate(zip(report['rows'],prepared['plans'],strict=True)):
        slot=row['root_slot']; task=tasks[slot]; t=task['interface']['gn_task']
        physical_api.commands(selection['commands'])
        require(physical_api.source.digest({k:v for k,v in selection.items() if k!='selection_digest'})==
            selection['selection_digest']==row['selection_digest'],'default-spaces physical selection digest')
        require(row['status'] in ('horizon_completed','native_success') and row['first_error'] is None and
            row['secondary_errors']==[] and row['initial_exact'] is True and row['full_horizon'] is True and
            row['own_attempted']==row['own_returned']==row['warmup_returned']==10 and not row['truncated'],'full10 only; no early endpoint substitution')
        closed=[e for e in events if e['kind']=='branch_closed' and e['receipt'].get('selection_digest')==row['selection_digest']]
        starts=[e for e in events if e['kind']=='branch_started' and e.get('branch')==b]
        require(len(closed)==len(starts)==1 and closed[0]['receipt']==row and starts[0]['plan']==selection,'durable selected branch receipt')
        transitions=[e for e in events if e['kind']=='returned_transition' and e.get('branch')==b and e['phase']=='plan']
        require([e['step'] for e in transitions]==list(range(1,11)) and
            [e['command'] for e in transitions]==selection['commands'],'all10 selected literal commands')
        root=old.checkpoint(row,0); terminal=old.checkpoint(row,10)
        require(root['state']==row['root_state'] and terminal['state']==row['final_state'],'root/terminal state identity')
        h=enc.digest(np.ascontiguousarray(np.concatenate((old.vector(row['root_state'],7),old.vector(row['goal'],7))),dtype=np.float64).tobytes())
        require(prepared['source_slots'][slot]==dict(root_slot=slot,root_sha256=h),'physical root/goal identity')
        attached=[e for e in events if e['kind']=='goal_attached' and e.get('branch')==b]
        require(len(attached)==1 and attached[0]['goal']==row['goal'] and
            attached[0]['goal_rgb_sha256']==t['goal_source_sha256'] and attached[0]['snapshot']==root['snapshot'] and
            all(e['returned_goal_state']==row['goal'] and e['returned_goal_rgb_sha256']==t['goal_source_sha256'] for e in transitions),
            'own approved task goal attached throughout')
        for step in (5,10):
            c=old.checkpoint(row,step)
            observed=[e for e in events if e['kind']=='fresh_observation' and e.get('branch')==b and e['receipt']==c]
            guards=[e for e in events if e['kind']=='guard_receipt' and e.get('branch')==b and e['phase']=='plan' and e['step']==step]
            require(len(observed)==len(guards)==1 and guards[0]['accepted'] is True and guards[0]['png']==c['png'] and
                guards[0]['file_sha256']==c['file_sha256'] and transitions[step-1]['state']==c['state'],'checkpoint/guard/journal alignment')
            images.append(dict(index=len(images),branch=b,root_slot=slot,source_slot=slot,arm=row['arm'],
                task_id=task['interface']['task_id'],task_export=task['task_export'],task_sha256=task['task_sha256'],
                blind_sha256=task['blind_sha256'],role='prefix5' if step==5 else 'terminal10',checkpoint=c,
                root=t['root_context']['latent_history'][-1],goal=t['goal'],Q='identity192',predicted=None,
                selection_digest=row['selection_digest']))
    require(len(images)==2*n,'all fixed prefix5/terminal10 images')
    return images


def build(physical_report,physical_sha,task_exports,task_shas,arms):
    fixed_arms(arms); require(len(task_exports)==len(task_shas)==2,'two explicit approved original task exports')
    require(sha(old.__file__)==OLD_SHA and sha(physical_api.__file__)==PHYSICAL_SHA,'reviewed frozen infrastructure')
    for h in [physical_sha]+task_shas:enc.core.hash_value(h)
    arguments=dict(physical_report=str(Path(physical_report).absolute()),physical_sha=physical_sha,
        task_exports=[str(Path(p).absolute()) for p in task_exports],task_shas=task_shas,arms=arms)
    bindings={};tasks=[]
    for slot,(path,h) in enumerate(zip(arguments['task_exports'],task_shas,strict=True)):
        interface=planner.load_export(path,h); export=read(path,h); t=interface['gn_task']
        require(type(export['source_slot']) is int and export['source_slot']==slot and
            export['blind_sha256']==export['blind_source_sha256']==list(enc.APPROVED_INPUTS.values())[slot] and
            np.array_equal(t['projection'],np.eye(192)),'own opaque approved task/Q identity')
        tasks.append(dict(source_slot=slot,task_export=path,task_sha256=h,blind_sha256=export['blind_sha256'],interface=interface))
        bindings[path]=h
    ph=Path(arguments['physical_report']); report=read(ph,physical_sha); bindings[str(ph)]=physical_sha
    bp=ph.parent/'execution_binding.json'; binding=read(bp); bindings[str(bp)]=sha(bp)
    pp=Path(binding['prepared']).absolute(); prep=read(pp,binding['prepare_sha256']);bindings[str(pp)]=binding['prepare_sha256']
    # Load only the selected manifest, never the sourcepack/source prepared/certificate.
    mp=Path(prep['manifest']).absolute(); manifest=read(mp,prep['manifest_sha256']);bindings[str(mp)]=prep['manifest_sha256']
    require(manifest['arms']==arms and manifest['plans']==prep['plans'] and manifest['roots']==prep['source_slots'] and
        manifest['sourcepack_sha256']==prep['sourcepack_sha256'],'physical manifest/prepared selection chain')
    refs=[dict(root_sha256=s['root_sha256'],blind_sha256=t['blind_sha256']) for s,t in zip(prep['source_slots'],tasks,strict=True)]
    # selection() is saved-pointer-only; never invoke material/verify/load_source.
    plans,compiler_bindings=physical_api.selection(manifest,dict(references=refs))
    require(plans==prep['plans'],'own compiler pointer receipts');bindings.update(compiler_bindings)
    for path,h in compiler_bindings.items():
        compiler=read(path,h)
        for filename,artifact_sha in compiler.get('artifact_sha256',{}).items():
            f=old.local(Path(path).parent,filename);require(sha(f)==artifact_sha,'compiler artifact seal');bindings[str(f)]=artifact_sha
    for path,h in prep['bindings'].items():
        if Path(path).suffix in ('.py','.md'):
            require(sha(path)==h,'physical source seal');bindings[str(Path(path).absolute())]=h
    jp=ph.parent/'journal.jsonl';events=old.journal(jp,report['journal_sha256']);bindings[str(jp)]=report['journal_sha256']
    images=assemble(report,events,prep,tasks,arms)
    for item in images:
        c=item['checkpoint'];f=old.local(ph.parent,c['png']);old.png(f,c['file_sha256'],c['pixel_sha256'])
        item['png_path']=str(f);bindings[str(f)]=c['file_sha256']
    for path,h in enc.MODELS.items():require(sha(path)==h,'frozen model/config bytes');bindings[path]=h
    files=planner.source_files()+[Path(old.__file__),Path(physical_api.__file__),Path(__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(Path(f).resolve()):sha(f) for f in files})
    for path,h in bindings.items():require(sha(path)==h,'final saved input/source TOCTOU')
    return dict(schema=NAME+'.prepare',arguments=arguments,images=images,bindings=bindings,policy=str(enc.POLICY),
        expected_model_hash=enc.MODEL_TENSOR_SHA,contract=dict(roots=2,arms=arms,branches=2*len(arms),images=4*len(arms),
        encoder=4*len(arms),projector=4*len(arms),predict=0,action_encoder=0,AD=0,physics=0,training=False,
        Q='identity192_no_fit',selection=False,no_dedup=True,P1_admission=False,P2_admission=False,
        prediction='unknown; no model rollout or schema-dependent imagined-prefix substitution',
        inherited_costs='physical/compiler/original encoding costs remain separately paid; not refunded',
        numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),output_cap_bytes=CAP,disk_floor_bytes=FLOOR))


def prepare(output,**arguments):
    p=build(**arguments);path=Path(output).absolute();raw=canonical(p)
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive small prepare +200MiB')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    p=read(path,expected);require(p==build(**p['arguments']),'complete generic saved chain reconstruction');return p


def compare(item,z):
    enc.native(z);real=z.astype(float);root=old.vector(item['root'],192);goal=old.vector(item['goal'],192)
    q=lambda a,b:float(np.mean((a-b)**2))
    return dict(actual_Q=q(real,goal),root_Q=q(root,goal),actual_Q_change=q(real,goal)-q(root,goal),
        actual_vs_root_mse=q(real,root),predicted_Q=None,predicted_vs_real_mse=None,
        Q_improvement_sign_reversed=None,prediction_status='unknown_not_recomputed')


def encode_images(p,callback,ledger,store,rows):
    for item,row in zip(p['images'],rows,strict=True):
        ledger.image=item['index'];row['status']='attempted';store.event(dict(kind='image_attempt',provenance=item))
        a=old.png(item['png_path'],item['checkpoint']['file_sha256'],item['checkpoint']['pixel_sha256'])
        z=callback(a.copy());enc.native(z)
        require([(e['resource'],e['completed']) for e in ledger.entries if e['image']==item['index']]==
            [('encoder',True),('projector',True)],'both actual native module completions')
        row.update(status='encoded',latent=store.latent(item['index'],z),metrics=compare(item,z))
        store.event(dict(kind='image_saved',receipt=deepcopy(row)));ledger.image=None


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    n=p['contract']['images'];ledger=enc.Ledger(len(p['arguments']['arms']),store.event)
    rows=[dict(**{k:i[k] for k in ('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')},
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
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model tensor SHA')
        transform=p0.image_transform();ledger.sync=lambda:torch.cuda.synchronize(device);ledger.no_grad=lambda:not torch.is_grad_enabled()
        with sdpa_kernel(SDPBackend.MATH),enc.guard(model,torch,ledger,errors):
            def callback(rgb):
                with torch.inference_mode():
                    x=p0._image_tensor(rgb,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native CUDA RGB batch1')
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
        complete=not errors and not ledger.blocked and source_ok and before is not None and before==after and all(r['status']=='encoded' for r in rows) and ledger.counts()=={k:dict(attempted=n,completed=n) for k in ('encoder','projector')}
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try:hashes[f.name]=sha(f)
                except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='selected_rgb_encoded_not_admitted' if complete else 'partial_or_failed',
            prepare_sha256=expected,inputs=p['arguments'],rows=rows,resources=ledger.counts(),errors=errors,first_error=errors[0] if errors else None,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            host=host,artifact_sha256=hashes,contract=p['contract'],new_predict=0,new_AD=0,new_physics=0,training=False,P1_admission=False,P2_admission=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(enc.error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial output retained; no retry: '+str(errors));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for k in ('physical-report','physical-sha','prepared','prepare-sha','output'):p.add_argument('--'+k)
    p.add_argument('--task-export',action='append');p.add_argument('--task-sha',action='append');p.add_argument('--arm',action='append')
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output,physical_report=a.physical_report,physical_sha=a.physical_sha,task_exports=a.task_export,task_shas=a.task_sha,arms=a.arm)
    elif a.command=='verify':r=dict(status='verified_no_execution',images=len(verify(a.prepared,a.prepare_sha)['images']))
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
