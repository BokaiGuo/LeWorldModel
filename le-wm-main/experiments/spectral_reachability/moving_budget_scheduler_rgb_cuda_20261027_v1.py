"""Fixed scheduler8 actual prefix5/terminal10 -> raw PNG copies and native192; MAIN-only CUDA.

Independent authority: no old physical whitelist or execute path. Missing checkpoints
fail before CUDA. Predicted latents stay unknown; no predictions, AD or physics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import moving_budget_scheduler_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_constructive_moving_history_rgb_cuda_20261027_v1 as original

NAME='moving_budget_scheduler_rgb_cuda_20261027_v1'
ROOT,moving,rgb=original.ROOT,original.moving,original.rgb
enc=rgb.enc
require,sha,read,canonical=rgb.require,rgb.sha,rgb.read,rgb.canonical
CAP,FLOOR,FINAL=16*1024**2,200*1024**2,512*1024
PHYSICAL_SOURCE_SHA = '60ad315b96c2c9eda21cc79fe4b07710eafd6d16027d8b0ff46641bb794e3f1a'


ARMS=list(physical.ARMS)


def contract():
    return dict(roots=2,arms=list(ARMS),branches=8,images=16,encoder=16,projector=16,
        predict=0,action_encoder=0,AD=0,physics=0,training=False,Q='identity_mean192_no_fit',
        no_dedup=True,selection=False,P1_admission=False,P2_admission=False,P3_admission=False,
        new_independent_samples=0,prediction='unknown_not_recomputed',
        scope='same known moving2 x ALL prereg U/D/S/N; not independent confirmation or total-compute matching',
        partial='missing5/10 rejects before CUDA; fixed8 branches/16 images retained on encoding failure',
        numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),wall_seconds=300,
        wall_scope='encode_images loop only; excludes prepare/verify/model loading/finalization',
        events_per_completed_image=7,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,
        inherited_costs='source/root-goal encoder, scheduler and physical costs separately paid, never refunded',
        comparison='RGB absent is missing; explicit external actual report SHA needed, predicted remains null')


def key(row):
    require(type(row['root_slot']) is int and row['root_slot'] in (0,1),'literal physical root_slot')
    return row['root_slot'],row['arm'],row['selection_digest']


def plan_from_consumed(row,entry,roots):
    s=row['source_slot']
    require(type(s) is int and s in (0,1) and row['arm'] in ARMS and
        row['root_sha256']==roots[s]['root_sha256'] and entry['authority']==physical.AUTHORITY,
        'only fixed scheduler original roots/arms')
    p=dict(root_slot=s,root_sha256=row['root_sha256'],arm=row['arm'],
        commands=physical.kernel.commands(row['commands']).tolist(),
        task_id=row['task_id'],blind_sha256=row['blind_sha256'],task_export=row['task_export'],task_sha256=row['task_sha256'],
        task_contract_sha256=row['task_contract_sha256'],compiler_authority=physical.AUTHORITY,
        compiler_report=str(Path(physical.absolute(entry['output']))/'report.json'),
        compiler_report_sha256=entry['report_sha256'],compiler_selection_sha256=row['selection_sha256'],
        own_plan_sha256=physical.source.digest(row))
    return dict(p,selection_digest=physical.source.digest(p))


def assemble(report,events,p,manifest,tasks,selected):
    """Own scheduler schema join; saved provenance only, never control selection."""
    arms=list(ARMS);physical.fixed_arms(manifest['arms'])
    expected=[(s,a) for s in (0,1) for a in arms];n=len(expected)
    require(report['schema']==physical.NAME+'.report' and report['status']=='moving_budget_scheduler_physical_recorded' and
        report['errors']==[] and all(report[k] is False for k in ('P1_admission','P2_admission','P3_admission')) and report['fixed_branch_denominator']==n,'complete predeclared physical cohort')
    require(p['schema']==physical.NAME+'.prepare' and p['contract']==report['contract']==physical.contract(manifest['wall_seconds']) and
        manifest['arms']==arms and manifest['sourcepack_sha256']==p['sourcepack_sha256']==report['sourcepack_sha256']==physical.SOURCE_SHA,
        'same original target-free moving source/contract')
    require(p['source_prepare_id']==report['source_prepare_id'],'same source preparation identity')
    require([(r['root_slot'],r['arm']) for r in report['rows']]==expected and
        [(r['root_slot'],r['arm']) for r in p['plans']]==expected and len(tasks)==2,'all fixed physical branches in order')
    require(len(manifest['roots'])==2 and all(type(r['root_slot']) is int and r['root_slot']==s
        for s,r in enumerate(manifest['roots'])),'strict original evaluator root slots, not bool aliases')
    require(events[0]['kind']=='run_started' and events[0]['prepare']==p and events[-1]['kind']=='run_finished' and
        events[-1]['status']==report['status'] and events[-1]['counts']==report['counts'] and events[-1]['errors']==[] and
        events[-1]['event_sha256']==report['journal_chain_tail'],'physical prepare and terminal hashchain receipt')
    expected_counts=rgb.counts_for(n)
    expected_counts.update({k:dict(attempted=n,completed=n) for k in ('environment_construct','environment_close')})
    counts={k:dict(attempted=0,completed=0) for k in expected_counts};stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            resource=e['resource'];require(resource in counts,'known physical operation')
            field='attempted' if e['kind']=='operation_attempted' else 'completed';counts[resource][field]+=1
            if field=='attempted':stack.append(resource)
            else:require(stack and stack.pop()==resource,'actual nested operation completion')
    require(not stack and counts==report['counts']==expected_counts,'full actual reset/public/physics/render census')
    starts=[e for e in events if e['kind']=='branch_started'];closed=[e for e in events if e['kind']=='branch_closed']
    require(len(starts)==len(closed)==n and len({key(e) for e in starts})==n,'exact durable branch denominator')
    images=[]
    for b,(row,plan) in enumerate(zip(report['rows'],p['plans'],strict=True)):
        identity=key(row);require(identity==key(plan) and row['branch_index']==b and row['selected_plan']==plan,'physical selection join and exact returned selected_plan')
        slot,arm,_=identity;require(plan['root_sha256']==manifest['roots'][slot]['root_sha256'],'plan root binding');t=tasks[slot];g=t['interface']['gn_task'];tc=t['task_contract'];chosen=selected[(slot,arm)]
        require(chosen['source_slot']==slot and chosen['arm']==arm and chosen['task_id']==t['interface']['task_id']==plan['task_id'] and
            chosen['blind_sha256']==t['blind_sha256']==plan['blind_sha256'] and chosen['task_export']==t['task_export']==plan['task_export'] and
            chosen['task_sha256']==t['task_sha256']==plan['task_sha256'] and chosen['commands']==plan['commands'] and
            physical.source.digest(chosen)==plan['own_plan_sha256'],'actual ownconsume task/blind/raw commands/plan digest')
        require(chosen['root_sha256']==plan['root_sha256'] and
            chosen['task_contract_sha256']==plan['task_contract_sha256']==g['task_contract_sha256'] and
            chosen['selection_sha256']==plan['compiler_selection_sha256'] and
            chosen['whole_run_report_sha256']==plan['compiler_report_sha256'] and
            chosen['schema']==physical.AUTHORITY+'.model_plan' and
            chosen['physical_execution'] is False and chosen['selection_uses_physics'] is False,
            'original moving task contract/root and blind scheduler model_plan seal')
        require(physical.source.digest({k:v for k,v in chosen.items() if k not in ('selection_sha256','whole_run_report_sha256')})==
            chosen['selection_sha256'],'whole report SHA is outside original model_plan selection seal')
        require(physical.source.digest({k:v for k,v in plan.items() if k!='selection_digest'})==plan['selection_digest'],'default-spaces physical selection digest')
        require(row['status'] in ('horizon_completed','native_success') and row['initial_exact'] is True and row['full_horizon'] is True and
            row['first_error'] is None and row['secondary_errors']==[] and not row['truncated'] and row['PNG']==22 and
            row['warmup_returned']==row['own_returned']==row['own_attempted']==10,'full10 and required5/10; never synthesize missing image')
        start=[e for e in starts if key(e)==identity];close=[e for e in closed if key(e['receipt'])==identity]
        require(len(start)==len(close)==1 and start[0]['branch']==b and start[0]['plan']==plan and close[0]['receipt']==row,'exact durable branch/selected commands')
        scoped=[e for e in events if e.get('branch')==b];require(all(key(e)==identity for e in scoped),'all event identity scopes')
        warm=np.asarray(start[0]['raw_warmup'],np.float32)
        require(physical.kernel.moving.exact(warm,moving.fixed_raw()),'moving nonzero raw history')
        require(start[0]['raw_float32_bits']==warm.tobytes().hex(),'moving raw history bytes')
        wr=[e for e in scoped if e['kind']=='warmup_return']
        require([e['step'] for e in wr]==list(range(1,11)) and [e['raw_float32_bits'] for e in wr]==[a.tobytes().hex() for a in warm] and
            [e['raw_action'] for e in wr]==warm.astype(float).tolist(),'all10 actual moving warmup tokens')
        transitions=[e for e in scoped if e['kind']=='returned_transition' and e['phase']=='plan']
        require([e['step'] for e in transitions]==list(range(1,11)) and [e['command'] for e in transitions]==plan['commands'] and
            all(not e['terminated'] and not e['truncated'] for e in transitions[:-1]),'actual own full10 execution')
        observed=[e for e in scoped if e['kind']=='fresh_observation'];guards=[e for e in scoped if e['kind']=='guard_receipt']
        census=[(ph,s) for ph in ('warmup','plan') for s in range(11)]
        require([(e['receipt']['phase'],e['receipt']['step']) for e in observed]==census and
            [(e['phase'],e['step']) for e in guards]==census and all(e['accepted'] is True and e['render_preserved_dynamics'] is True for e in guards),
            'all22 real off frames and fresh guards')
        for o,guard in zip(observed,guards,strict=True):
            require(o['index']<guard['index'] and o['receipt']['state']==o['receipt']['observation_state']==o['receipt']['snapshot']['state'],'actual scene saved before guard')
        require([observed[s]['receipt']['pixel_sha256'] for s in (0,5,10)]==tc['pixel_sha256'][:3],'old moving RGB3 unchanged')
        cp=[rgb.old.checkpoint(row,s) for s in (0,5,10)];root=cp[0]
        require(root['state']==row['root_state'] and cp[-1]['state']==row['final_state'] and root['pixel_sha256']==tc['pixel_sha256'][2] and
            root['snapshot']==observed[10]['receipt']['snapshot'],'root and terminal actual identity')
        require(manifest['roots'][slot]==dict(root_slot=slot,root_sha256=physical.source.root_hash(np.asarray(row['root_state']),np.asarray(row['goal']))),'original root+goal identity')
        attached=[e for e in scoped if e['kind']=='goal_attached']
        require(len(attached)==1 and attached[0]['goal']==row['goal'] and attached[0]['snapshot']==root['snapshot'] and
            attached[0]['goal_rgb_sha256']==g['goal_source_sha256']==tc['pixel_sha256'][3] and
            all(e['returned_goal_state']==row['goal'] and e['returned_goal_rgb_sha256']==g['goal_source_sha256'] for e in transitions),'original off goal throughout physical commands')
        for step,c in zip((5,10),cp[1:],strict=True):
            require(observed[11+step]['receipt']==c and transitions[step-1]['state']==c['state'],'checkpoint tied to own returned command')
            images.append(dict(index=len(images),branch=b,root_slot=slot,source_slot=slot,arm=arm,
                task_id=t['interface']['task_id'],task_export=t['task_export'],task_sha256=t['task_sha256'],blind_sha256=t['blind_sha256'],
                role='prefix5' if step==5 else 'terminal10',checkpoint=c,root=g['root_context']['latent_history'][-1],goal=g['goal'],
                predicted=None,Q='identity192',selection_digest=plan['selection_digest'],own_plan_sha256=plan['own_plan_sha256'],
                root_sha256=plan['root_sha256'],task_contract_sha256=plan['task_contract_sha256'],
                compiler_selection_sha256=plan['compiler_selection_sha256'],compiler_authority=plan['compiler_authority'],compiler_report_sha256=plan['compiler_report_sha256']))
    require(len(images)==2*n,'full requested image denominator')
    return images


def build(physical_report,physical_sha,physical_prepare_sha,physical_source_sha):
    arms=list(ARMS)
    require(PHYSICAL_SOURCE_SHA is not None and physical_source_sha==PHYSICAL_SOURCE_SHA,'MAIN frozen scheduler physical source SHA required')
    for h in (physical_sha,physical_prepare_sha,physical_source_sha):moving.core.hash_value(h)
    arguments=dict(physical_report=str(Path(physical_report).resolve()),physical_sha=physical_sha,
        physical_prepare_sha=physical_prepare_sha,physical_source_sha=physical_source_sha)
    bindings={}
    def bind(path,h):
        path=str(Path(path).resolve());require(sha(path)==h and (path not in bindings or bindings[path]==h),'nonconflicting external seal '+path)
        bindings[path]=h;return Path(path)
    bind(physical.__file__,physical_source_sha)
    bind(moving.__file__,original.MOVING_SHA);bind(rgb.__file__,original.RGB_SHA);bind(rgb.old.__file__,rgb.OLD_SHA)
    ph=bind(arguments['physical_report'],physical_sha);report=read(ph,physical_sha)
    bp=bind(ph.parent/'execution_binding.json',sha(ph.parent/'execution_binding.json'));binding=read(bp)
    require(binding['prepare_sha256']==physical_prepare_sha,'external actual physical prepare')
    require(report['prepared']==binding['prepared'] and report['prepare_sha256']==physical_prepare_sha,'scheduler report execution binding')
    pp=bind(binding['prepared'],physical_prepare_sha);p=read(pp,physical_prepare_sha)
    mp=bind(p['manifest'],p['manifest_sha256']);manifest=read(mp,p['manifest_sha256'])
    require(manifest['schema']==physical.NAME+'.selection' and manifest['arms']==arms and
        manifest['selection_fixed_before_physics'] is True and manifest['physical_outcome_selection'] is False,'predeclared same arms, never post-hoc subset')
    tasks=[]
    for slot,(path,h) in enumerate(zip(original.EXPORTS,original.EXPORT_SHAS,strict=True)):
        bind(path,h);interface=moving.load_task(path,h);e=read(path,h)
        require(type(e['source_slot']) is int and e['source_slot']==slot and e['blind_sha256']==e['blind_source_sha256'] and
            np.array_equal(interface['q'],np.eye(192)),'strict original moving task / identity mean192Q')
        tasks.append(dict(task_export=path,task_sha256=h,blind_sha256=e['blind_sha256'],interface=interface,task_contract=e['task_contract']))
    selected={};reconstructed={}
    require(len(manifest['authorities'])==1 and manifest['authorities'][0]['authority']==physical.AUTHORITY,
        'only fixed final scheduler ownconsume authority')
    entry=manifest['authorities'][0]
    rows,bound=physical.ownconsume(entry)  # saved-only own authority; never neural/physics
    for row in rows:
        s,a=row['source_slot'],row['arm']
        require(type(s) is int and s in (0,1) and a in ARMS and (s,a) not in selected,
            'unique scheduler original root/strategy')
        selected[s,a]=row
        reconstructed[s,a]=plan_from_consumed(row,entry,manifest['roots'])
    for f,h in bound.items():bind(f,h)
    require(set(selected)=={(s,a) for s in (0,1) for a in ARMS} and
        p['plans']==[reconstructed[s,a] for s in (0,1) for a in ARMS],
        'all eight actual physical plans exactly from scheduler ownconsume')
    # Source/certificate/evaluator bytes are hashed only, not parsed to select controls.
    for f,h in p['bindings'].items():bind(f,h)
    jp=bind(ph.parent/'journal.jsonl',report['journal_sha256']);events=physical.kernel.journal(jp,report['journal_chain_tail'])
    images=assemble(report,events,p,manifest,tasks,selected)
    for e in events:
        if e['kind']=='fresh_observation':
            c=e['receipt'];f=bind(rgb.old.local(ph.parent,c['png']),c['file_sha256']);rgb.old.png(f,c['file_sha256'],c['pixel_sha256'])
    for item in images:item['png_path']=str(rgb.old.local(ph.parent,item['checkpoint']['png']))
    for f,h in enc.MODELS.items():bind(f,h)
    for f in moving.source_files()+[Path(original.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bind(f,sha(f))
    for f,h in list(bindings.items()):bind(f,h)
    return dict(schema=NAME+'.prepare',arguments=arguments,images=images,bindings=bindings,policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,contract=contract())


def prepare(output,**arguments):
    p=build(**arguments);value=canonical(p);path=Path(output).resolve()
    require(not path.exists() and len(value)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(value),'exclusive small prepare/200MiB floor')
    with path.open('xb') as f:require(f.write(value)==len(value),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(value))


def verify(path,expected):
    p=read(path,expected);require(p==build(**p['arguments']),'whole fixed RGB input/source reconstruction');return p


class Store(rgb.Store):
    def __init__(self,output):
        self.output=Path(output).absolute();require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW16MiB +200MiB reserve')
        self.output.mkdir()

    def write(self,name,value,append=False,final=False):
        path=rgb.old.local(self.output,name);require(not path.is_symlink(),'flat nonsymlink output')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(value)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(value),'whole16MiB/floor including partial/final')
        with path.open('ab' if append else 'xb') as f:require(f.write(value)==len(value),'short artifact write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


compare=rgb.compare  # Pure native192 actual meanQ; prediction explicitly unknown.


def encode_images(p,callback,ledger,store,rows):
    """Real original PNG bytes copied BEFORE neural dispatch; no state input."""
    n=p['contract']['images'];require(len(p['images'])==len(rows)==n,'whole declared image denominator')
    started=time.monotonic();prefix={}
    for item,row in zip(p['images'],rows,strict=True):
        require(time.monotonic()-started<p['contract']['wall_seconds'],'encoding wall cap')
        ledger.image=item['index'];row['status']='attempted';store.event(dict(kind='image_attempt',provenance=item))
        c=item['checkpoint'];path=Path(item['png_path'])
        a=rgb.old.png(path,c['file_sha256'],c['pixel_sha256'])
        require(path.stat().st_size<=1024**2,'bounded original RGB PNG')
        payload=path.read_bytes()
        require(enc.digest(payload)==c['file_sha256'] and sha(path)==c['file_sha256'],'exact raw PNG read seal')
        name=f"image_{item['index']:02d}.rgb.png";store.write(name,payload)
        row['rawpixel']=dict(file=name,sha256=c['file_sha256'],pixel_sha256=c['pixel_sha256'],dtype='uint8',shape=[224,224,3])
        store.event(dict(kind='raw_image_saved',index=item['index'],receipt=row['rawpixel']))
        z=callback(a.copy());enc.native(z)
        require([(e['resource'],e['completed']) for e in ledger.entries if e['image']==item['index']]==
            [('encoder',True),('projector',True)],'both actual module completions, no inferred counts')
        metrics=compare(item,z);identity=(item['root_slot'],item['arm'],item['selection_digest'])
        if item['role']=='prefix5':
            require(identity not in prefix,'one exact prefix per branch')
            prefix[identity]=(a.copy(),metrics['actual_Q']);pixel_mse=0.;q_change=0.
        else:
            require(item['role']=='terminal10' and identity in prefix,'terminal only after own prefix, no substitute')
            pixel_mse=float(np.mean((a.astype(np.float64)-prefix[identity][0].astype(np.float64))**2))
            q_change=metrics['actual_Q']-prefix[identity][1]
        metrics.update(rawpixel_vs_prefix5_mse=pixel_mse,actual_Q_change_from_prefix5=q_change,
            rawpixel_mean=float(np.mean(a.astype(np.float64))))
        row.update(status='encoded',latent=store.latent(item['index'],z),metrics=metrics)
        store.event(dict(kind='image_saved',receipt=deepcopy(row)));ledger.image=None
        require(time.monotonic()-started<p['contract']['wall_seconds'],'encoding loop completion wall cap')


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    n=p['contract']['images'];ledger=enc.Ledger(4,store.event)
    rows=[dict(**{k:i[k] for k in ('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')},status='not_started',provenance=i) for i in p['images']]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
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
            def callback(a):
                with torch.inference_mode():
                    x=p0._image_tensor(a,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native CUDA RGB224 batch1')
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
        complete=(not errors and not ledger.blocked and source_ok and before is not None and before==after and
            all(r['status']=='encoded' for r in rows) and ledger.counts()=={k:dict(attempted=n,completed=n) for k in ('encoder','projector')})
        hashes={}
        try:artifacts=list(store.output.iterdir())
        except BaseException as e:errors.append(enc.error('artifact_listing',e));complete=False;artifacts=[]
        for f in artifacts:
            try:
                if f.is_file():hashes[f.name]=sha(f)
            except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='moving_budget_scheduler_rgb_encoded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,inputs=p['arguments'],rows=rows,resources=ledger.counts(),
            errors=errors,first_error=errors[0] if errors else None,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,host=host,artifact_sha256=hashes,contract=p['contract'],
            fixed_branch_denominator=p['contract']['branches'],fixed_image_denominator=n,new_predict=0,new_AD=0,new_physics=0,
            training=False,P1_admission=False,P2_admission=False,P3_admission=False,new_independent_samples=0)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(enc.error('final_report',e));complete=False;report['status']='partial_or_failed';print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial output retained; no retry: '+str(errors));return report


def consume(output,report_sha):
    """Own saved-only comparison adapter. Missing RGB stays missing at the caller.

    Never encode/retry/reconstruct an unknown prediction. Actual report and its
    full closure are necessary; partial runs cannot silently shrink the cohort.
    """
    output=Path(output).absolute();require(not output.is_symlink(),'nonsymlink RGB output directory')
    report=read(output/'report.json',report_sha)
    require(report['schema']==NAME+'.report' and report['status']=='moving_budget_scheduler_rgb_encoded_not_admitted' and
        report['errors']==[] and report['first_error'] is None and report['source_verified'] is True and
        report['model_unchanged'] is True and report['model_hash_before']==report['model_hash_after']==enc.MODEL_TENSOR_SHA and
        report['contract']==contract() and report['fixed_branch_denominator']==8 and report['fixed_image_denominator']==16 and
        all(report[k] is False for k in ('P1_admission','P2_admission','P3_admission','training')) and
        all(report[k]==0 for k in ('new_predict','new_AD','new_physics','new_independent_samples')) and
        report['resources']=={k:dict(attempted=16,completed=16) for k in ('encoder','projector')},
        'complete original-model scheduler RGB report only')
    p=verify(report['prepared'],report['prepare_sha256'])
    require(report['inputs']==p['arguments'] and len(report['rows'])==16,'same full prepared image cohort')
    require(read(output/'binding.json')==dict(prepared=report['prepared'],prepare_sha256=report['prepare_sha256']),
        'actual RGB execution binding')
    expected_files={'binding.json','events.jsonl'}|{f'image_{i:02d}.{suffix}' for i in range(16) for suffix in ('rgb.png','native192.npy')}
    require(set(report['artifact_sha256'])==expected_files,'complete sixteen raw/latent artifact pairs')
    actual=list(output.iterdir())
    require(not output.is_symlink() and {f.name for f in actual}==expected_files|{'report.json'} and
        all(f.is_file() and not f.is_symlink() for f in actual) and sum(f.stat().st_size for f in actual)<=CAP,
        'exact flat directory census:34 artifacts+report, no symlink/directory/extra file, <=16MiB')
    for name,h in report['artifact_sha256'].items():require(sha(rgb.old.local(output,name))==h,'RGB artifact seal')
    path=output/'events.jsonl';require(path.stat().st_size<=CAP,'bounded RGB events')
    raw=path.read_bytes();require(raw.endswith(b'\n'),'no torn RGB journal')
    events=[rgb.old.planner.decode(line) for line in raw.splitlines()]
    require(len(events)==112,'seven actual events for EACH of sixteen images')
    prefix={};summaries=[]
    for item,row in zip(p['images'],report['rows'],strict=True):
        i=item['index'];c=item['checkpoint'];rp=row['rawpixel'];latent=row['latent']
        require(row['provenance']==item and row['status']=='encoded' and
            all(row[k]==item[k] for k in ('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')),
            'exact actual RGB root/task/selection/role join')
        require(rp==dict(file=f'image_{i:02d}.rgb.png',sha256=c['file_sha256'],pixel_sha256=c['pixel_sha256'],dtype='uint8',shape=[224,224,3]) and
            latent==dict(file=f'image_{i:02d}.native192.npy',sha256=report['artifact_sha256'][f'image_{i:02d}.native192.npy'],dtype='float32',shape=[192]),
            'original byte-identical PNG and native192 file receipts')
        a=rgb.old.png(output/rp['file'],rp['sha256'],rp['pixel_sha256'])
        npy=(output/latent['file']).read_bytes()
        require(len(npy)<=4096 and enc.digest(npy)==latent['sha256'],'bounded sealed native192 NPY bytes')
        header=BytesIO(npy)
        require(np.lib.format.read_magic(header)==(1,0),'native np.save NPY1.0 format')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(header)
        require(shape==(192,) and not fortran and dtype==np.dtype(np.float32) and
            len(npy)==header.tell()+192*4,'finite float32[192] NPY header/size, no trailing bytes before np.load')
        stream=BytesIO(npy);z=np.load(stream,allow_pickle=False)
        require(isinstance(z,np.ndarray) and z.dtype==np.float32 and z.shape==(192,) and
            np.isfinite(z).all() and stream.tell()==len(npy),'finite float32[192] NPY, no trailing bytes/array')
        metrics=compare(item,z)
        identity=(item['root_slot'],item['arm'],item['selection_digest'])
        if item['role']=='prefix5':prefix[identity]=(a.copy(),metrics['actual_Q']);pixel_mse=0.;change=0.
        else:
            require(identity in prefix and item['role']=='terminal10','own prefix/terminal pair')
            pixel_mse=float(np.mean((a.astype(np.float64)-prefix[identity][0].astype(np.float64))**2))
            change=metrics['actual_Q']-prefix[identity][1]
        metrics.update(rawpixel_vs_prefix5_mse=pixel_mse,actual_Q_change_from_prefix5=change,rawpixel_mean=float(np.mean(a.astype(np.float64))))
        require(row['metrics']==metrics,'saved actual Q/pixel arithmetic, no inferred prediction')
        expected=[dict(kind='image_attempt',provenance=item),dict(kind='raw_image_saved',index=i,receipt=rp)]
        for offset,resource in enumerate(('encoder','projector')):
            entry=dict(index=2*i+offset,image=i,resource=resource,completed=False)
            expected.extend([dict(kind='module_attempt',entry=entry),dict(kind='module_return',entry=dict(entry,completed=True))])
        expected.append(dict(kind='image_saved',receipt=row))
        require(events[7*i:7*(i+1)]==expected,'exact image/module attempted-returned/NPY event receipts')
        summaries.append(dict(root_slot=item['root_slot'],arm=item['arm'],role=item['role'],
            selection_digest=item['selection_digest'],task_id=item['task_id'],root_sha256=item['root_sha256'],
            compiler_report_sha256=item['compiler_report_sha256'],metrics=metrics,rawpixel=rp,latent=latent))
    require(sha(output/'report.json')==report_sha,'actual report final readback')
    return dict(schema=NAME+'.consume',status='scheduler_rgb_verified_not_admitted',report_sha256=report_sha,
        rows=summaries,errors=[],P1_admission=False,P2_admission=False,P3_admission=False,
        prediction_status='unknown_not_recomputed',fixed_branch_denominator=8,fixed_image_denominator=16)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('physical-report','physical-sha','physical-prepare-sha','physical-source-sha','prepared','prepare-sha','report-sha','output'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output,physical_report=a.physical_report,physical_sha=a.physical_sha,
        physical_prepare_sha=a.physical_prepare_sha,physical_source_sha=a.physical_source_sha)
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='consume':r=consume(a.output,a.report_sha)
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()

