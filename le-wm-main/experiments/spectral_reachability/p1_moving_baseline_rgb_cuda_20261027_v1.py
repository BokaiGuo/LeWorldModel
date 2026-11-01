"""Fixed moving baseline physical prefix5/terminal10 RGB; MAIN-only CUDA.

Predeclared gradient-only cohort is supported without waiting for other arms.
No outcome selection, old zero-history loader, new model prediction or physics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_baseline_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_constructive_moving_history_rgb_cuda_20261027_v1 as original

NAME='p1_moving_baseline_rgb_cuda_20261027_v1'
ROOT,moving,rgb=original.ROOT,original.moving,original.rgb
enc=rgb.enc
require,sha,read,canonical=rgb.require,rgb.sha,rgb.read,rgb.canonical
CAP,FLOOR,FINAL=16*1024**2,200*1024**2,512*1024
GRADIENT_REPORT_SHA='96c140caee14bba43bf766d981c21be1724a36c0047fde4abcc2892c24bf5b7d'
SAMPLING_REPORT_SHA='9ac889d6be76bf3241f85c9bfcac13936784dcf80171b3fe48abf5c9cfcd319c'


def contract(arms):
    arms=physical.checked_arms(arms);n=4*len(arms)
    return dict(roots=2,arms=arms,branches=2*len(arms),images=n,encoder=n,projector=n,
        predict=0,action_encoder=0,AD=0,physics=0,training=False,Q='identity_mean192_no_fit',
        no_dedup=True,selection=False,P1_admission=False,P2_admission=False,new_independent_samples=0,
        prediction='unknown_not_recomputed; this adapter evaluates actual RGB Q only',
        scope='predeclared arms of the same two known moving roots; not independent confirmation or full-compute matching',
        partial='missing5/10 rejects before CUDA; no endpoint substitution or denominator reduction',
        numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),wall_seconds=300,
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,
        inherited_costs='planner/physical/root-goal encoding separately paid, not refunded')


def key(row):
    require(type(row['root_slot']) is int and row['root_slot'] in (0,1),'literal physical root_slot')
    return row['root_slot'],row['arm'],row['selection_digest']


def plan_from_consumed(row,entry):
    p=dict(root_slot=row['source_slot'],arm=row['arm'],commands=physical.kernel.commands(row['commands']).tolist(),
        task_id=row['task_id'],blind_sha256=row['blind_sha256'],task_export=row['task_export'],task_sha256=row['task_sha256'],
        compiler_authority=entry['authority'],compiler_report=str(Path(entry['output'])/'report.json'),
        compiler_report_sha256=entry['report_sha256'],own_plan_sha256=physical.source.digest(row))
    return dict(p,selection_digest=physical.source.digest(p))


def assemble(report,events,p,manifest,tasks,selected,arms):
    """Pure-array mapping of actual Arendt physical schema, not its execute."""
    expected=[(s,a) for s in (0,1) for a in physical.checked_arms(arms)];n=len(expected)
    require(report['schema']==physical.NAME+'.report' and report['status']=='moving_baseline_physical_recorded' and
        report['errors']==[] and report['P1_admission'] is False and report['fixed_branch_denominator']==n,'complete predeclared physical cohort')
    require(p['schema']==physical.NAME+'.prepare' and p['contract']==report['contract']==physical.contract(arms,manifest['wall_seconds']) and
        manifest['arms']==arms and manifest['sourcepack_sha256']==p['sourcepack_sha256']==report['sourcepack_sha256']==physical.SOURCE_SHA,
        'same original target-free moving source/contract')
    require([(r['root_slot'],r['arm']) for r in report['rows']]==expected and
        [(r['root_slot'],r['arm']) for r in p['plans']]==expected and len(tasks)==2,'all fixed physical branches in order')
    require(len(manifest['roots'])==2 and all(type(r['root_slot']) is int and r['root_slot']==s
        for s,r in enumerate(manifest['roots'])),'strict original evaluator root slots, not bool aliases')
    require(events[0]['kind']=='run_started' and events[0]['prepare']==p and events[-1]['kind']=='run_finished' and
        events[-1]['status']==report['status'] and events[-1]['counts']==report['counts'] and events[-1]['errors']==[] and
        events[-1]['event_sha256']==report['journal_chain_tail'],'physical prepare and terminal hashchain receipt')
    counts={k:dict(attempted=0,completed=0) for k in rgb.counts_for(n)};stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            resource=e['resource'];require(resource in counts,'known physical operation')
            field='attempted' if e['kind']=='operation_attempted' else 'completed';counts[resource][field]+=1
            if field=='attempted':stack.append(resource)
            else:require(stack and stack.pop()==resource,'actual nested operation completion')
    require(not stack and counts==report['counts']==rgb.counts_for(n),'full actual reset/public/physics/render census')
    starts=[e for e in events if e['kind']=='branch_started'];closed=[e for e in events if e['kind']=='branch_closed']
    require(len(starts)==len(closed)==n and len({key(e) for e in starts})==n,'exact durable branch denominator')
    images=[]
    for b,(row,plan) in enumerate(zip(report['rows'],p['plans'],strict=True)):
        identity=key(row);require(identity==key(plan),'physical selection join')
        slot,arm,_=identity;t=tasks[slot];g=t['interface']['gn_task'];tc=t['task_contract'];chosen=selected[(slot,arm)]
        require(chosen['source_slot']==slot and chosen['arm']==arm and chosen['task_id']==t['interface']['task_id']==plan['task_id'] and
            chosen['blind_sha256']==t['blind_sha256']==plan['blind_sha256'] and chosen['task_export']==t['task_export']==plan['task_export'] and
            chosen['task_sha256']==t['task_sha256']==plan['task_sha256'] and chosen['commands']==plan['commands'] and
            physical.source.digest(chosen)==plan['own_plan_sha256'],'actual ownconsume task/blind/raw commands/plan digest')
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
                compiler_authority=plan['compiler_authority'],compiler_report_sha256=plan['compiler_report_sha256']))
    require(len(images)==2*n,'full requested image denominator')
    return images


def build(physical_report,physical_sha,physical_prepare_sha,physical_source_sha,arms):
    arms=physical.checked_arms(arms)
    for h in (physical_sha,physical_prepare_sha,physical_source_sha):moving.core.hash_value(h)
    arguments=dict(physical_report=str(Path(physical_report).resolve()),physical_sha=physical_sha,
        physical_prepare_sha=physical_prepare_sha,physical_source_sha=physical_source_sha,arms=arms)
    bindings={}
    def bind(path,h):
        path=str(Path(path).resolve());require(sha(path)==h and (path not in bindings or bindings[path]==h),'nonconflicting external seal '+path)
        bindings[path]=h;return Path(path)
    bind(physical.__file__,physical_source_sha)
    bind(moving.__file__,original.MOVING_SHA);bind(rgb.__file__,original.RGB_SHA);bind(rgb.old.__file__,rgb.OLD_SHA)
    ph=bind(arguments['physical_report'],physical_sha);report=read(ph,physical_sha)
    bp=bind(ph.parent/'execution_binding.json',sha(ph.parent/'execution_binding.json'));binding=read(bp)
    require(binding['prepare_sha256']==physical_prepare_sha,'external actual physical prepare')
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
    authority_order=[k for k,(_,_,names,_) in physical.AUTHORITIES.items() if any(a in arms for a in names)]
    require([x['authority'] for x in manifest['authorities']]==authority_order,'exact used ownconsume authority order')
    for entry in manifest['authorities']:
        if entry['authority']=='gradient':require(entry['report_sha256']==GRADIENT_REPORT_SHA,'MAIN fixed actual gradient model report')
        if entry['authority']=='sampling':require(entry['report_sha256']==SAMPLING_REPORT_SHA,'MAIN fixed actual sampling model report')
        rows,bound=physical.ownconsume(entry)  # own saved verification; NO neural/physics
        allowed=physical.AUTHORITIES[entry['authority']][2];seen=set()
        for row in rows:
            s,a=row['source_slot'],row['arm'];require(type(s) is int and s in (0,1) and a in allowed and (s,a) not in seen,'unique ownconsume slot/arm')
            seen.add((s,a))
            if a in arms:selected[s,a]=row;reconstructed[s,a]=plan_from_consumed(row,entry)
        require(seen=={(s,a) for s in (0,1) for a in {a for _,a in seen}},'both roots in ownconsume')
        for f,h in bound.items():bind(f,h)
    require(set(selected)=={(s,a) for s in (0,1) for a in arms} and p['plans']==[reconstructed[s,a] for s in (0,1) for a in arms],
        'every actual physical plan exactly from its ownconsume authority')
    # Source/certificate/evaluator bytes are hashed only, not parsed to select controls.
    for f,h in p['bindings'].items():bind(f,h)
    jp=bind(ph.parent/'journal.jsonl',report['journal_sha256']);events=physical.kernel.journal(jp,report['journal_chain_tail'])
    images=assemble(report,events,p,manifest,tasks,selected,arms)
    for e in events:
        if e['kind']=='fresh_observation':
            c=e['receipt'];f=bind(rgb.old.local(ph.parent,c['png']),c['file_sha256']);rgb.old.png(f,c['file_sha256'],c['pixel_sha256'])
    for item in images:item['png_path']=str(rgb.old.local(ph.parent,item['checkpoint']['png']))
    for f,h in enc.MODELS.items():bind(f,h)
    for f in moving.source_files()+[Path(original.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bind(f,sha(f))
    for f,h in list(bindings.items()):bind(f,h)
    return dict(schema=NAME+'.prepare',arguments=arguments,images=images,bindings=bindings,policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,contract=contract(arms))


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
    n=p['contract']['images'];require(len(p['images'])==len(rows)==n,'predeclared whole image denominator')
    started=time.monotonic()
    def bounded(a):
        require(time.monotonic()-started<p['contract']['wall_seconds'],'wall cap before actual encoding')
        return callback(a)
    rgb.encode_images(p,bounded,ledger,store,rows)  # PNG/array/module checks only; no loader or runtime verifier


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    n=p['contract']['images'];ledger=enc.Ledger(len(p['arguments']['arms']),store.event)
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
        report=dict(schema=NAME+'.report',status='moving_baseline_rgb_encoded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,inputs=p['arguments'],rows=rows,resources=ledger.counts(),
            errors=errors,first_error=errors[0] if errors else None,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,host=host,artifact_sha256=hashes,contract=p['contract'],
            fixed_branch_denominator=p['contract']['branches'],fixed_image_denominator=n,new_predict=0,new_AD=0,new_physics=0,
            training=False,P1_admission=False,P2_admission=False,new_independent_samples=0)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(enc.error('final_report',e));complete=False;report['status']='partial_or_failed';print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial output retained; no retry: '+str(errors));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for k in ('physical-report','physical-sha','physical-prepare-sha','physical-source-sha','prepared','prepare-sha','output'):parser.add_argument('--'+k)
    parser.add_argument('--arms',nargs='+');a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output,physical_report=a.physical_report,physical_sha=a.physical_sha,
        physical_prepare_sha=a.physical_prepare_sha,physical_source_sha=a.physical_source_sha,arms=a.arms)
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
