"""Known-root moving overlay fidelity core; MAIN-only CUDA, never P1 admission.

Explicit Arendt source/Jason plan receipt adapters; no producer execution.
The typed RAM boundary is internal, NOT a renamed producer report tag.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_archive_target_overlay_fidelity_cuda_20261027_v1 as native
from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as moving

NAME='p1_moving_target_overlay_fidelity_cuda_20261027_v1'
NATIVE_SHA='f536cf5fe9baa4c652b17888f6ff0112bccc5aa241c567581a034b5ec8ccc31c'
PHASES=('factual_witness','fixed_planner_terminal')
CONDITIONS=('off','on')
ARMS=tuple(moving.ARMS)
KINDS=('H1','fresh_H2_first','fresh_H2_terminal','teacher_forced_H1')
PACKET_KEYS={'rgb_history','raw_action_history','goal_rgb'}
REQUEST_KEYS={'slot','condition','arm','kind','latent_history','raw_packets','tokens'}
require,enc,canonical=native.require,native.enc,native.canonical
Store,save_array,NativeTrace=native.Store,native.save_array,native.NativeTrace


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def hash_value(value):enc.core.hash_value(value)


def packet(value):
    """Literal blind deployment packet; no state/witness/label/metadata keys."""
    require(type(value) is dict and set(value)==PACKET_KEYS,'strict blind rgb_history/raw_action_history/goal_rgb only')
    h,raw,goal=value['rgb_history'],value['raw_action_history'],value['goal_rgb']
    require(isinstance(h,np.ndarray) and h.dtype==np.uint8 and h.shape==(3,224,224,3),'unaltered history3 RGB224')
    native.image(goal)
    require(isinstance(raw,np.ndarray) and raw.dtype==np.float32 and raw.shape==(10,2) and np.isfinite(raw).all() and
        native.bits(raw,moving.fixed_raw()),'literal original moving warmup raw float32, no second /100')
    return value


def contract(phase):
    require(phase in PHASES,'explicit named phase')
    images,predictions=(20,16) if phase==PHASES[0] else (32,24)
    return dict(schema=NAME+'.contract',phase=phase,slots=[0,1],conditions=list(CONDITIONS),
        arms=['source_factual_witness'] if phase==PHASES[0] else list(ARMS),
        counts=dict(encoder=images,projector=images,predict=predictions,predictor=predictions,action_encoder=predictions,pred_proj=predictions),
        visual_policy=('own history3 + actual witness future5/10 per slot/condition' if phase==PHASES[0]
            else 'own history3 + own true goal + GN/D2P actual prefix5/terminal10 each per slot/condition'),
        goal_RGB=('bind unchanged packet goal; not encoded or a successor label' if phase==PHASES[0]
            else 'off unchanged original pixels; on producer same-body true goal render; encode each independently; original identity Q, no fit'),
        phases='separate exclusive runs; no implicit cross-phase encoding reuse; missing labels reject before encode',
        history='original warmup raw10 float32; original mean/std, no /100, clip, tanh or zero-only loader',
        H2='fresh first jump, then own predicted prefix; never teacher force open H2',
        label_policy='same condition and exact source/action/selection identity only; all predictions durable before scores',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False,CUDA_only=True),
        AD=0,physics=0,training=False,Q_fit=False,selection=False,no_retry=True,no_dedup=True,
        offreplay='validation-only duplicate, not an encoding sample; exact file/pixel/invariant checks',
        wall_seconds=600,allocated_bytes=3*1024**3,reserved_bytes=4*1024**3,
        witness_labels='source replay0 global15/20; missing on15 rejects phase; NEVER substitute goal or cross condition',
        native_outline='on default outline is NOT the constructed task goal; task goal RGB is same-body witness terminal rendering',
        output_cap_bytes=native.CAP,disk_floor_bytes=native.FLOOR,prepare_cap_bytes=native.PREPARE_CAP,
        scope='known-root paired rendering calibration, NOT independent confirmation',
        P1_admission=False,training_lineage_verified=False,ranking_repaired_claim=False)


def commands(raw):
    require(isinstance(raw,np.ndarray) and raw.dtype==np.float32 and raw.shape==(10,2) and np.isfinite(raw).all(),
        'literal preexisting actual/selected raw10, no numerical action repair')
    return raw


def validate_roots(roots):
    """Internal adapter output boundary, not a substitute for external file seals."""
    require(len(roots)==2 and [r['slot'] for r in roots]==[0,1] and all(type(r['slot']) is int for r in roots),'all two original slots')
    for root in roots:
        for key in ('root_id','original_task_id','producer_report_sha256'):hash_value(root[key])
        require(set(root['packets'])==set(CONDITIONS),'both fixed rendering conditions')
        old=packet(root['original_off_packet']);off=packet(root['packets']['off']);on=packet(root['packets']['on'])
        require(all(native.bits(off[k],old[k]) for k in PACKET_KEYS),'new off exactly reproduces old blind source packet')
        require(native.bits(off['raw_action_history'],on['raw_action_history']),'same actual raw history under overlay')
    return roots


def material(phase,source_report,source_sha,producer_source_sha,producer_prepare_sha,
             plans_report=None,plans_sha=None,plans_source_sha=None,plans_prepare_sha=None):
    """MAIN saved-only entry: external seals before importing any saved adapter."""
    from experiments.spectral_reachability import p1_moving_target_overlay_source_20261027_v1 as producer
    contract(phase)
    args=dict(phase=phase,source_report=str(Path(source_report).absolute()),source_sha=source_sha,
        producer_source_sha=producer_source_sha,producer_prepare_sha=producer_prepare_sha,
        plans_report=None if plans_report is None else str(Path(plans_report).absolute()),plans_sha=plans_sha,
        plans_source_sha=plans_source_sha,plans_prepare_sha=plans_prepare_sha)
    for h in (source_sha,producer_source_sha,producer_prepare_sha):hash_value(h)
    require(native.sha(producer.__file__)==producer_source_sha and native.sha(native.__file__)==NATIVE_SHA,'externally pinned source and frozen native trace')
    rp=Path(args['source_report']);r=native.read(rp,source_sha);bp=rp.parent/'execution_binding.json';eb=native.read(bp)
    require(set(eb)=={'prepared','prepare_file_sha256'} and eb['prepare_file_sha256']==producer_prepare_sha and Path(eb['prepared']).is_absolute(),
        'literal source execution binding and external prepare file SHA, not prepare_id')
    pp=Path(eb['prepared']);p=producer.load_prepared(pp,producer_prepare_sha);data=producer.verify(p)
    bindings=dict(p['bindings']);bindings.update({str(rp):source_sha,str(bp):native.sha(bp),str(pp):producer_prepare_sha})
    def sealed(path,expected=None,cap=16*1024**2):
        path=Path(path).absolute();require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded regular nonsymlink artifact')
        h=native.sha(path);require(expected is None or h==expected,'artifact external SHA')
        require(str(path) not in bindings or bindings[str(path)]==h,'nonconflicting artifact binding');bindings[str(path)]=h;return path
    def events(directory,report):
        jp=sealed(Path(directory)/'journal.jsonl');value=jp.read_bytes()
        require(native.sha(jp)==bindings[str(jp)],'journal read TOCTOU')
        return native.journal(value,report['journal_chain_tail'])
    def decoder(directory):
        def decode(f):
            require(type(f['png']) is str and Path(f['png']).name==f['png'],'flat literal PNG')
            path=sealed(Path(directory)/f['png'],f['file_sha256'],1024**2)
            return producer.old_image(dict(path=str(path),file_sha256=f['file_sha256'],pixel_sha256=f['pixel_sha256']))
        return decode
    def decode_packet(f):
        require(type(f['path']) is str and Path(f['path']).name==f['path'],'flat blind NPZ')
        path=sealed(rp.parent/f['path'],f['sha256'],2*1024**2)
        return moving.moving.load_blind(path,f['sha256'])
    def commit(key):return native.read(sealed(rp.parent/(key+'_commands.json')))
    interfaces=[]
    for path,h in zip(moving.EXPORTS,moving.EXPORT_SHAS,strict=True):
        sealed(path,h);interfaces.append(moving.load_export(path,h))
    roots=assemble_source(p,r,events(rp.parent,r),data,decoder(rp.parent),decode_packet,commit,interfaces,source_sha)
    for slot,(root,path) in enumerate(zip(roots,moving.EXPORTS,strict=True)):
        export=moving.read(path,2*1024**2);tc=export['task_contract'];off=root['packets']['off']
        require(export['source_slot']==tc['source_slot']==slot and tc['pixel_sha256']==[digest(a) for a in (*off['rgb_history'],off['goal_rgb'])] and
            tc['raw_action_sha256']==digest(off['raw_action_history']) and export['blind_sha256']==data['refs'][slot]['blind_sha256'],
            'original task identity and raw/pixel seals match source, not zero-history task')
    if phase==PHASES[1]:
        from experiments.spectral_reachability import p1_moving_target_overlay_plans_physical_20261027_v1 as plans
        for h in (plans_sha,plans_source_sha,plans_prepare_sha):hash_value(h)
        require(plans_report is not None and native.sha(plans.__file__)==plans_source_sha,'explicit external final plans source/report')
        pr=Path(args['plans_report']);rr=native.read(pr,plans_sha);bb=pr.parent/'execution_binding.json';binding=native.read(bb)
        require(set(binding)=={'prepared','prepare_sha256'} and binding['prepare_sha256']==plans_prepare_sha,'Jason own execution binding schema')
        ppath=Path(binding['prepared']);require(ppath.is_absolute(),'absolute plans prepare path')
        own=plans.verify(ppath,plans_prepare_sha);old,_,saved=plans.inputs();require(old==own['source'],'plans original source closure')
        bindings.update(own['bindings']);sealed(pr,plans_sha);sealed(bb);sealed(ppath,plans_prepare_sha)
        assemble_plans(roots,own,rr,events(pr.parent,rr),saved,decoder(pr.parent),plans_sha)
        # Verify root and selected compiler identity against sealed original task exports.
        for plan in own['source']['plans']:
            s=plan['root_slot'];require(plan['compiler_report_SHA']=='52c475ed499b077f261517bc592252b208926919909674cebf68fa4029cc8310' and
                plan['compiler_report']==str(moving.ROOT/'results/moving_history_planner_handoff_main01.json') and
                plan['compiler_schema']=='moving_history_planner_handoff.v1',
                'old fixed GN/D2P sealed command handoff, not raw CUDA report or newly optimized controls')
            require(own['source']['sourcepack_sha256']==producer.PINNED[str(producer.SD/'sourcepack.json')] and
                own['source']['source_slots'][s]['root_sha256']==roots[s]['root_id'],'source/plan stable physical root join')
    else:
        require(all(v is None for v in (plans_report,plans_sha,plans_source_sha,plans_prepare_sha)),
            'factual phase does not require/read/encode Jason or old GN/D2P endpoints')
    gate=prerequisites(roots,phase);require(gate['ready'],'missing actual phase prerequisites: '+str(gate['missing']))
    files=set(native.sealed.source_files()+moving.moving.source_files()+producer.files()+
        [Path(native.__file__),Path(native.fidelity.__file__),Path(native.fidelity.old.__file__),Path(moving.__file__),Path(__file__),
         moving.ROOT/'tests'/('test_'+NAME+'.py'),moving.ROOT/'innerdocs'/(NAME+'.md')])
    for f in files:sealed(f)
    bindings.update(enc.MODELS)
    normalization=enc.native_metadata()
    for path,h in bindings.items():require(native.sha(path)==h,'final saved source/model/PNG/command binding '+path)
    receipts=[dict(slot=root['slot'],root_id=root['root_id'],original_task_id=root['original_task_id'],
        packets={c:{k:native.array_id(v) for k,v in pk.items()} for c,pk in root['packets'].items()},
        witness_commands=None if 'witness' not in root else native.array_id(root['witness']['commands']),
        fixed_plan_selections=[v['selection_sha256'] for v in root.get('fixed_plans',[])]) for root in roots]
    own=dict(schema=NAME+'.prepare',arguments=args,contract=contract(phase),bindings=bindings,input_receipts=receipts,
        source_prepare_id=r['prepare_id'],normalization=normalization,policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA)
    require(len(canonical(own))<native.PREPARE_CAP,'prepare exceeds256KiB; report design problem, never raise cap')
    return own,roots


def prepare(output,**arguments):
    p,_=material(**arguments);b=canonical(p);path=Path(output).absolute()
    require(not path.exists() and shutil.disk_usage(path.parent).free>=native.FLOOR+len(b),'exclusive small prepare +200MiB floor')
    with path.open('xb') as f:require(f.write(b)==len(b),'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_sha256=native.sha(path),bytes=len(b))


def verify(prepared,expected):
    p=native.read(prepared,expected,native.PREPARE_CAP);now,data=material(**p['arguments'])
    require(p==now,'entire own prepare/source/label/model closure reconstruction');return p,data


def run(prepared,expected,output):
    p,data=verify(prepared,expected);phase=p['arguments']['phase'];store=Store(output)
    _,images,rows=census(data,phase);errors=[];callback=None;model=None;after=None;host=None;source_ok=False;forbidden=[]
    limits=native.fidelity.Limits(free=lambda:shutil.disk_usage(store.output).free)
    def error(where,exc):errors.append(enc.error(where,exc))
    try:
        limits.check();store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.put('input_receipts.json',p)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);limits.check()
        callback=CudaCallbacks(model,torch,device,store,limits)
        def forbidden_event(e):forbidden.append(deepcopy(e));store.event(e)
        with native.fidelity.old.no_ad(torch,forbidden_event,errors),sdpa_kernel(SDPBackend.MATH),callback.trace:
            analysis=evaluate(data,phase,callback.encode,callback.predict,store,images,rows);store.put('analysis.json',analysis)
        full_counts(callback.trace.counts(),p['contract']['counts']);require(not forbidden,'zero actual AD attempts')
    except BaseException as exc:
        errors.insert(0,enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as exc:error('error_journal',exc)
    finally:
        for row in images+rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as exc:error('final_model',exc)
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:error('final_source',exc)
        try:limits.check()
        except BaseException as exc:error('final_limits',exc)
        try:store.close()
        except BaseException as exc:error('journal_finalize',exc)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as exc:error('artifact_listing',exc);files=[]
        for f in files:
            try:require(f.is_file() and not f.is_symlink(),'flat actual output');hashes[f.name]=native.sha(f)
            except BaseException as exc:error('artifact_hash',exc)
        before=None if callback is None else callback.before
        complete=not errors and source_ok and before is not None and before==after and all(x['status']=='encoded' for x in images) and all(x['status']=='saved' for x in rows)
        report=dict(schema=NAME+'.report',status='moving_overlay_fidelity_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,phase=phase,contract=p['contract'],
            images=images,requests=rows,resources=callback.trace.counts() if callback else {k:dict(attempted=0,completed=0) for k in p['contract']['counts']},
            module_ledger=deepcopy(callback.trace.entries) if callback else [],limits=limits.receipt(),artifact_sha256=hashes,
            errors=errors,first_error=errors[0] if errors else None,forbidden_AD_attempts=forbidden,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,host=host,
            AD=0,physics=0,training=False,Q_fit=False,P1_admission=False,independent_confirmation=False,scientific_winner=None)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:error('final_report',exc);complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'failed/partial evidence retained, no retry: '+str(errors));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--phase',choices=PHASES)
    for key in ('source-report','source-sha','producer-source-sha','producer-prepare-sha','plans-report','plans-sha','plans-source-sha','plans-prepare-sha','prepared','prepare-sha','output'):
        parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare':
        require(a.phase is not None,'explicit factual or fixed-plan phase')
        keys=('phase','source_report','source_sha','producer_source_sha','producer_prepare_sha','plans_report','plans_sha','plans_source_sha','plans_prepare_sha')
        result=prepare(a.output,**{k:getattr(a,k) for k in keys})
    elif a.command=='verify':p,_=verify(a.prepared,a.prepare_sha);result=dict(status='verified_no_execution',contract=p['contract'])
    else:result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical({k:v for k,v in result.items() if k not in ('images','requests','module_ledger')}).decode(),end='')


def label(record,root,condition,raw,step,selection=None):
    """Every label is an actually rendered successor, never the packet goal."""
    require(type(record['slot']) is int and record['slot']==root['slot'] and record['root_id']==root['root_id'] and
        record['condition']==condition and record['step']==step and type(record['step']) is int and
        record['commands_sha256']==digest(raw) and record['source_report_sha256']==root['producer_report_sha256' if selection is None else 'plans_report_sha256'],
        'same source/root/condition/time/raw-action factual label')
    require(record['role']==('actual_witness_successor' if selection is None else 'actual_fixed_plan_prefix' if step==5 else 'actual_fixed_plan_terminal'), 'actual label acquisition role')
    if selection is not None:require(record['selection_sha256']==selection,'fixed planner selection label, no old endpoint')
    return native.image(record['pixels'])


def prerequisites(roots,phase):
    validate_roots(roots);contract(phase);missing=[]
    for root in roots:
        if phase==PHASES[0]:
            if root.get('witness') is None:
                missing.append(dict(slot=root['slot'],required='actual witness commands + off/on actual future5/10 receipts'));continue
            w=root['witness'];commands(w['commands'])
            for c in CONDITIONS:
                frames=w.get('labels',{}).get(c)
                if frames is None or len(frames)!=2:
                    missing.append(dict(slot=root['slot'],condition=c,required='two actual witness successor frames5/10'));continue
                for frame,step in zip(frames,(5,10),strict=True):label(frame,root,c,w['commands'],step)
        else:
            plans=root.get('fixed_plans')
            if plans is None:
                missing.append(dict(slot=root['slot'],required='frozen GN/D2P plans + actual off/on prefix/terminal receipts'));continue
            hash_value(root['plans_report_sha256'])
            require(isinstance(root['q'],np.ndarray) and root['q'].dtype==np.float32 and native.bits(root['q'],np.eye(192,dtype=np.float32)),
                'original fixed identity192 Q, no refit or alternate projection')
            require(root['goal_render_roles']==dict(off='original_source_goal_bit_exact',on='same_body_true_goal_render'), 'explicit original off vs same-body true on goal acquisition')
            require([p['arm'] for p in plans]==list(ARMS),'all two original fixed arms in order, no winners selected')
            for p in plans:
                commands(p['commands']);hash_value(p['selection_sha256']);hash_value(p['planner_report_sha256'])
                require(p['original_task_id']==root['original_task_id'],'same original old moving task identity, not a retuned task')
                for c in CONDITIONS:
                    frames=p.get('labels',{}).get(c)
                    if frames is None or set(frames)!={'prefix','terminal'}:
                        missing.append(dict(slot=root['slot'],condition=c,arm=p['arm'],required='actual same-condition selected prefix5 and terminal10'));continue
                    for role,step in (('prefix',5),('terminal',10)):label(frames[role],root,c,p['commands'],step,p['selection_sha256'])
    return dict(schema=NAME+'.prerequisites',phase=phase,ready=not missing,missing=missing,P1_admission=False)


def image_jobs(roots,phase):
    gate=prerequisites(roots,phase);require(gate['ready'],'missing actual labels before any encoding: '+str(gate['missing']))
    jobs=[]
    for root in roots:
        for c in CONDITIONS:
            for j,rgb in enumerate(root['packets'][c]['rgb_history']):
                jobs.append(dict(slot=root['slot'],condition=c,role='history',step=5*j,arm=None,pixels=rgb.copy()))
            if phase==PHASES[0]:
                for f in root['witness']['labels'][c]:
                    jobs.append(dict(slot=root['slot'],condition=c,role='actual_witness_successor',step=f['step'],arm=None,pixels=f['pixels'].copy()))
            else:
                jobs.append(dict(slot=root['slot'],condition=c,role='goal',step=None,arm=None,pixels=root['packets'][c]['goal_rgb'].copy()))
                for p in root['fixed_plans']:
                    for role,step in (('prefix',5),('terminal',10)):
                        jobs.append(dict(slot=root['slot'],condition=c,role='actual_fixed_plan_'+role,step=step,arm=p['arm'],pixels=p['labels'][c][role]['pixels'].copy()))
    require(len(jobs)==contract(phase)['counts']['encoder'],'fixed independent encoding samples');return jobs


def request(root,condition,arm,kind,history,raw_packets):
    require(history.dtype==np.float32 and history.shape==(3,192) and np.isfinite(history).all(),'native own-condition context3')
    return dict(slot=root['slot'],condition=condition,arm=arm,kind=kind,latent_history=history.copy(),
        raw_packets=raw_packets.copy(),tokens=native.fidelity.normalized(raw_packets))


def census(roots,phase):
    jobs=image_jobs(roots,phase)
    images=[dict(index=k,**{key:value for key,value in j.items() if key!='pixels'},status='not_started') for k,j in enumerate(jobs)]
    rows=[]
    for root in roots:
        for c in CONDITIONS:
            identities=[('source_factual_witness',kind) for kind in KINDS] if phase==PHASES[0] else [(a,k) for a in ARMS for k in ('fixed_H2_first','fixed_H2_terminal','fixed_teacher_H1')]
            for arm,kind in identities:rows.append(dict(index=len(rows),slot=root['slot'],condition=c,arm=arm,kind=kind,status='not_started'))
    require(len(rows)==contract(phase)['counts']['predict'],'fixed independent forwards');return jobs,images,rows


def evaluate(roots,phase,encode,predict,store,images,rows):
    """Pure scheduler; callbacks receive blind current inputs, NEVER goal/labels."""
    jobs,expected_images,expected_rows=census(roots,phase)
    require(images==expected_images and rows==expected_rows,'full predeclared fixed census, not selected subcases')
    zs={};predictions=[]
    for root in roots:
        save_array(store,f'slot_{root["slot"]}.raw_history',root['packets']['off']['raw_action_history'])
    for index,job in enumerate(jobs):
        row=images[index];row['status']='attempted'
        row['pixels']=save_array(store,f'image_{index:02d}.pixels',job['pixels'])
        store.event(dict(kind='image_attempt',receipt=deepcopy(row)))
        z=encode(job['pixels'].copy(),{k:v for k,v in job.items() if k!='pixels'},index);enc.native(z)
        row.update(status='encoded',latent=save_array(store,f'image_{index:02d}.native192',z))
        store.event(dict(kind='image_saved',receipt=deepcopy(row)))
        zs[job['slot'],job['condition'],job['role'],job['step'],job['arm']]=z.copy()
    index=0
    for root in roots:
        for c in CONDITIONS:
            original=np.stack([zs[root['slot'],c,'history',s,None] for s in (0,5,10)])
            warm=root['packets'][c]['raw_action_history'];arms=['source_factual_witness'] if phase==PHASES[0] else list(ARMS)
            for ai,arm in enumerate(arms):
                raw=root['witness']['commands'] if phase==PHASES[0] else root['fixed_plans'][ai]['commands']
                save_array(store,f'slot_{root["slot"]}.{c}.{arm}.raw_commands',raw)
                kinds=KINDS if phase==PHASES[0] else ('fixed_H2_first','fixed_H2_terminal','fixed_teacher_H1')
                h1=None;prefix=None
                for kind in kinds:
                    first=kind in ('H1','fresh_H2_first','fixed_H2_first')
                    if first:history=original.copy();packets=np.concatenate((warm,raw[:5])).reshape(3,5,2)
                    else:
                        if kind=='teacher_forced_H1':last=zs[root['slot'],c,'actual_witness_successor',5,None]
                        elif kind=='fixed_teacher_H1':last=zs[root['slot'],c,'actual_fixed_plan_prefix',5,arm]
                        else:last=prefix
                        enc.native(last);history=np.stack((original[1],original[2],last))
                        packets=np.concatenate((warm[5:],raw)).reshape(3,5,2)
                    r=request(root,c,arm,kind,history,packets);require(set(r)==REQUEST_KEYS,'strict predictor whitelist')
                    row=rows[index];require((row['slot'],row['condition'],row['arm'],row['kind'])==(root['slot'],c,arm,kind),'fixed case order')
                    row['status']='attempted';row['inputs']={k:save_array(store,f'request_{index:02d}.{k}',r[k]) for k in ('latent_history','raw_packets','tokens')}
                    store.event(dict(kind='prediction_attempt',receipt=deepcopy(row)))
                    out=predict(deepcopy(r),index)
                    require(isinstance(out,np.ndarray) and out.dtype==np.float32 and out.shape==(3,192) and np.isfinite(out).all(),'original full native3x192 output')
                    row.update(status='saved',output=save_array(store,f'request_{index:02d}.prediction',out))
                    store.event(dict(kind='prediction_saved',receipt=deepcopy(row)))
                    if kind=='H1':h1=out.copy()
                    if kind=='fresh_H2_first':require(native.bits(h1,out),'fresh H2 independently reproduces H1 full output exactly')
                    if kind in ('fresh_H2_first','fixed_H2_first'):prefix=out[-1].copy()
                    predictions.append(dict(slot=root['slot'],condition=c,arm=arm,kind=kind,last=out[-1].copy(),
                        last_input=history[-1].copy(),origin=original[-1].copy()))
                    index+=1
    require(index==contract(phase)['counts']['predict'],'all fixed predictions completed');store.event(dict(kind='all_predictions_durable_before_scoring',count=index))
    scores=[]
    for p in predictions:
        if phase==PHASES[0]:
            step=5 if p['kind'] in KINDS[:2] else 10;target=zs[p['slot'],p['condition'],'actual_witness_successor',step,None]
        else:
            step=5 if p['kind']=='fixed_H2_first' else 10
            target=zs[p['slot'],p['condition'],'actual_fixed_plan_prefix' if step==5 else 'actual_fixed_plan_terminal',step,p['arm']]
        scores.append(dict(**{k:p[k] for k in ('slot','condition','arm','kind')},root_relative_target_step=step,
            MSE=native.mse(p['last'],target),persistence_last_input_MSE=native.mse(p['last_input'],target),
            persistence_root_MSE=native.mse(p['origin'],target)))
    distances=[dict(slot=s,history_step=t,off_on_MSE=native.mse(zs[s,'off','history',t,None],zs[s,'on','history',t,None])) for s in (0,1) for t in (0,5,10)]
    goal_scores=[];rankings=[]
    if phase==PHASES[1]:
        for p in predictions:
            if p['kind']=='fixed_H2_first':continue
            q=roots[p['slot']]['q'];goal=zs[p['slot'],p['condition'],'goal',None,None]
            target=zs[p['slot'],p['condition'],'actual_fixed_plan_terminal',10,p['arm']]
            goal_scores.append(dict(**{k:p[k] for k in ('slot','condition','arm','kind')},
                predicted_Q=float(np.mean((q.astype(float)@(p['last'].astype(float)-goal))**2)),
                actual_Q=float(np.mean((q.astype(float)@(target.astype(float)-goal))**2))))
        for slot in (0,1):
            for c in CONDITIONS:
                for kind in ('fixed_H2_terminal','fixed_teacher_H1'):
                    pair=[next(s for s in goal_scores if (s['slot'],s['condition'],s['arm'],s['kind'])==(slot,c,a,kind)) for a in ARMS]
                    sign=lambda key:int(np.sign(pair[0][key]-pair[1][key]))
                    rankings.append(dict(slot=slot,condition=c,kind=kind,arm_order=list(ARMS),
                        predicted_GN_minus_D2P=pair[0]['predicted_Q']-pair[1]['predicted_Q'],
                        actual_GN_minus_D2P=pair[0]['actual_Q']-pair[1]['actual_Q'],
                        predicted_sign=sign('predicted_Q'),actual_sign=sign('actual_Q'),
                        rank_agreement=sign('predicted_Q')==sign('actual_Q')))
    return dict(schema=NAME+'.analysis',phase=phase,scores=scores,paired_history_distances=distances,
        goal_scores=goal_scores,rankings=rankings,P1_admission=False,independent_confirmation=False,selection=False)


class CudaCallbacks:
    """Frozen native calls; caller must bind producer before installing this scope.

    NativeTrace is faithfully reused without task/global replacement. This class
    never loads/encodes a goal or accepts a successor label in predictor inputs.
    """
    def __init__(self,model,torch,device,store,limits):
        require(getattr(device,'type',None)=='cuda','HOST CUDA only; no CPU neural fallback')
        require(native.sha(native.__file__)==NATIVE_SHA,'MAIN final sealed native capture helper')
        enc.frozen(model);self.model,self.torch,self.device,self.store,self.limits=model,torch,device,store,limits
        self.before=enc.tensor_hash(model);require(self.before==enc.MODEL_TENSOR_SHA,'original frozen checkpoint tensor hash')
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        self.p0=p0;self.transform=p0.image_transform()
        self.trace=NativeTrace(model,store,lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
    def encode(self,rgb,meta,index):
        self.limits.check();native.image(rgb);self.trace.active=('image',index,None)
        with self.torch.inference_mode():
            x=self.p0._image_tensor(rgb.copy(),transform=self.transform,device=self.device)
            require(tuple(x.shape)==(1,1,3,224,224) and x.device.type=='cuda','actual independent RGB CUDA batch1')
            z=self.model.encode({'pixels':x})['emb'];require(tuple(z.shape)==(1,1,192) and z.dtype==self.torch.float32,'native192 encoded output')
            result=z[0,0].detach().cpu().numpy().copy()
        self.trace.finish('image',index);self.limits.check();return result
    def predict(self,r,index):
        require(set(r)==REQUEST_KEYS,'no goal/label/state/task metadata reaches predictor')
        self.limits.check();self.trace.active=('prediction',index,r)
        with self.torch.inference_mode():
            z=self.torch.from_numpy(r['latent_history'].copy()).to(self.device)[None]
            tokens=self.torch.from_numpy(r['tokens'].copy()).to(self.device)[None]
            value=self.model.predict(z,self.model.action_encoder(tokens))
            require(tuple(value.shape)==(1,3,192) and value.dtype==self.torch.float32,'full native3x192 predicted output')
            result=value[0].detach().cpu().numpy().copy()
        self.trace.finish('prediction',index);self.limits.check();return result
    def final_model(self):
        enc.frozen(self.model);after=enc.tensor_hash(self.model)
        return dict(before=self.before,after=after,unchanged=self.before==after)


def full_counts(counts,expected):
    require(counts=={k:dict(attempted=v,completed=v) for k,v in expected.items()},'full actual attempted/completed resource census')


def actual_label(root,c,raw,step,pixels,selection=None):
    return dict(slot=root['slot'],root_id=root['root_id'],condition=c,step=step,
        commands_sha256=digest(raw),source_report_sha256=root['producer_report_sha256' if selection is None else 'plans_report_sha256'],
        role='actual_witness_successor' if selection is None else 'actual_fixed_plan_prefix' if step==5 else 'actual_fixed_plan_terminal',
        **({} if selection is None else dict(selection_sha256=selection)),pixels=native.image(pixels).copy())


def assemble_source(p,r,events,data,decode,decode_packet,commit,interfaces,report_sha):
    """Literal Arendt v1 receipts -> own blind/label boundary, entirely saved.

    decode/commit do bounded sealed reads. No producer episode or witness math
    is called here. Missing on15 is represented as a phase prerequisite gap.
    """
    prefix='p1_moving_target_overlay_source_20261027_v1'
    require(p['schema']==prefix+'.prepare' and r['schema']==prefix+'.report' and r['status']=='paired_moving_overlay_recorded' and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['full_protocol'] is True and
        r['prepare_id']==enc.core.digest(p) and r['contract']==p['contract'],'complete new source report/prepare, no partial native stop')
    ct=p['contract'];triplets=ct['source_triplets']
    require(triplets in ([0,5,10,20],[0,5,10,15,20]) and ct['flags']==[False,True,False] and ct['slots']==2 and
        ct['warmup']==ct['horizon']==10,'literal moving checkpoints/flags/denominator')
    expected_caps=dict(reset=12,public_step=120,physics_step=1224,set_state=24,goal_render=12,goal_setter=16,goal_bind=4,
        explicit_render=6*(21+2*len(triplets)))
    require(ct['caps']==expected_caps,'174 vs186 must be explicitly predeclared, never silently relabel')
    full_counts(r['counts'],expected_caps)
    require(events[0]['kind']=='run_started' and events[0]['prepare_id']==r['prepare_id'] and events[0]['contract']==ct and
        events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and events[-1]['counts']==r['counts'] and events[-1]['errors']==[],
        'source actual journal boundary')
    keys=[(s,ph) for s in (0,1) for ph in ('construction','replay0','replay1')]
    require([(x['source_slot'],x['phase']) for x in r['rows']]==keys,'all six source rows, fixed order')
    images={};raw_by_slot={}
    for row,(s,ph) in zip(r['rows'],keys,strict=True):
        key=f's{s}_{ph}';require(row['key']==key and row['kind']=='source' and row['status']=='source_episode_completed' and
            row['public_returned']==20 and row['missing_triplets']==[] and row['root_goal_sha256']==data['refs'][s]['root_sha256'] and
            row['native_success'] is (ph!='construction'),'complete original root goal identity, not selected favorable replay')
        own=[e for e in events if e.get('key')==key]
        frames=[e['receipt'] for e in own if e['kind']=='fresh_overlay_frame']
        order=[(str(t),i) for t in range(21) for i in (range(3) if t in triplets else range(1))]
        require([(f['reference_key'],f['overlay_index']) for f in frames]==order and
            row['frames']==[{k:f[k] for k in ('reference_key','overlay_index','png','file_sha256','pixel_sha256')} for f in frames],
            'complete identity-indexed actual source frames, not guessed indices')
        for f in frames:
            t,i=int(f['reference_key']),f['overlay_index'];ref=data['references'][key][str(t)]
            require(f['with_target'] is (i==1) and f['pre']==f['post'] and f['reference_error'] is None,
                'actual render preserved full invariant')
            a=native.image(decode(f));require(digest(a)==f['pixel_sha256'],'actual source PNG pixel SHA')
            if i!=1:
                require(f['old_off_match']==dict(file=True,pixels=True,state=True,snapshot=True) and
                    f['file_sha256']==ref['file_sha256'] and f['pixel_sha256']==ref['pixel_sha256'] and
                    f['post']['snapshot']==ref['snapshot'],'source off matches old sealed frame and full snapshot')
            else:require(f['old_off_match'] is None,'on is not old off')
            require(f['off_replay_match'] is (True if i==2 else None),'off replay receipt role')
            images[key,t,i]=a
            if i:
                first=next(x for x in frames if x['reference_key']==str(t) and x['overlay_index']==0)
                require(f['pre']==first['pre'],'triplet same body/goal/renderer')
            if i==2:require(native.bits(a,images[key,t,0]) and f['file_sha256']==first['file_sha256'],'off replay exact pixels and file')
        cp=commit(key);raw=commands(np.asarray(cp['commands'],np.float32))
        require(cp['schema']==prefix+'.witness_commit' and cp['root_snapshot']==data['references'][key]['10']['snapshot'],'actual witness commitment at source root')
        if s in raw_by_slot:require(native.bits(raw,raw_by_slot[s]),'same fixed witness across source episodes')
        raw_by_slot[s]=raw.copy()
        returns=[e for e in own if e['kind']=='native_return']
        require([(e['phase'],e['step']) for e in returns]==[(ph,t) for ph in ('warmup','witness') for t in range(1,11)],'full actual source transition chronology')
        for e,a in zip(returns,np.concatenate((moving.fixed_raw(),raw)),strict=True):
            require(e['requested_bits']==e['actual_bits']==a.tobytes().hex() and e['actual_dtype']==np.dtype(np.float32).str and
                native.bits(np.asarray(e['requested_raw'],np.float32),a) and native.bits(np.asarray(e['actual_raw'],np.float32),a),
                'actual warmup/witness raw float32 command bits')
    require([x['source_slot'] for x in r['blind_exports']]==[0,1],'both actual blind exports')
    roots=[]
    for s,export in enumerate(r['blind_exports']):
        ref=data['refs'][s];interface=interfaces[s]
        require(export['old_root_sha256']==ref['root_sha256'],'blind export stable physical root identity')
        original=packet(dict(rgb_history=ref['history'].copy(),raw_action_history=moving.fixed_raw(),goal_rgb=ref['goal_rgb'].copy()))
        packets={}
        for c,i in (('off',0),('on',1)):
            f=export['artifacts'][c];a=packet(decode_packet(f))
            require(f['fields']==['rgb_history','raw_action_history','goal_rgb'] and f['raw_action_sha256']==digest(moving.fixed_raw()) and
                f['rgb_history_pixel_sha256']==[digest(x) for x in a['rgb_history']] and f['goal_pixel_sha256']==digest(a['goal_rgb']),
                'literal blind artifact receipts')
            require(native.bits(a['rgb_history'],np.stack([images[f's{s}_replay0',t,i] for t in (0,5,10)])) and
                native.bits(a['rgb_history'],np.stack([images[f's{s}_replay1',t,i] for t in (0,5,10)])) and
                native.bits(a['goal_rgb'],images[f's{s}_construction',20,i]) and
                native.bits(a['goal_rgb'],images[f's{s}_replay0',20,i]) and native.bits(a['goal_rgb'],images[f's{s}_replay1',20,i]),
                'blind history and goal are true same-condition source renders, all replays exact')
            packets[c]=a
        root=dict(slot=s,root_id=ref['root_sha256'],original_task_id=interface['task_id'],producer_report_sha256=report_sha,
            original_off_packet=original,packets=packets,q=np.asarray(interface['q'],np.float32),
            goal_render_roles=dict(off='original_source_goal_bit_exact',on='same_body_true_goal_render'))
        # Do not fabricate on15 when the earlier four-triplet producer lacks it.
        if 15 in triplets:
            raw=raw_by_slot[s]
            root['witness']=dict(commands=raw,labels={c:[actual_label(root,c,raw,t-10,images[f's{s}_replay0',t,i]) for t in (15,20)] for c,i in (('off',0),('on',1))})
        roots.append(root)
    return validate_roots(roots)


def assemble_plans(roots,p,r,events,saved,decode,report_sha):
    """Jason's own plan checkpoint schema; no source-goal claim borrowed."""
    prefix='p1_moving_target_overlay_plans_physical_20261027_v1'
    require(p['schema']==prefix+'.prepare' and r['schema']==prefix+'.report' and r['status']=='moving_target_overlay_plans_recorded' and
        r['prepare_id']==enc.core.digest(p) and r['contract']==p['contract'] and r['errors']==[] and r['inputs_final_verified'] is True and
        r['full_horizon_count']==4 and r['complete_triplets']==12,'full new fixed-plan physical report, no partial')
    full_counts(r['counts'],dict(reset=8,public_step=80,physics_step=816,explicit_render=112,goal_render=8,set_state=16,goal_bind=4))
    require(events[0]['kind']=='run_started' and events[0]['prepare']==p and events[-1]['kind']=='run_finished' and
        events[-1]['counts']==r['counts'] and events[-1]['errors']==[],'actual fixed-plan journal boundary')
    expected=[(s,a) for s in (0,1) for a in ARMS]
    require([(x['root_slot'],x['arm']) for x in r['rows']]==expected and
        [(x['root_slot'],x['arm']) for x in p['source']['plans']]==expected,'both roots/all fixed arms, no outcome filtering')
    for root in roots:root.update(fixed_plans=[],plans_report_sha256=report_sha)
    for b,(row,plan,old) in enumerate(zip(r['rows'],p['source']['plans'],saved,strict=True)):
        s,arm=row['root_slot'],row['arm'];root=roots[s];raw=commands(np.asarray(plan['commands'],np.float32));selection=plan['selection_digest']
        hash_value(selection)
        require(row['selection_digest']==selection and row['initial_exact'] is True and row['full_horizon'] is True and row['first_error'] is None and
            row['secondary_errors']==[] and row['own_attempted']==row['own_returned']==row['warmup_returned']==10 and row['missing_checkpoints']==[] and
            row['status'] in ('horizon_completed','native_success'),'all fixed10 actual arms and initial exact')
        own=[e for e in events if e.get('branch')==b]
        starts=[e for e in own if e['kind']=='branch_started'];require(len(starts)==1 and starts[0]['plan']==plan,'actual original plan commitment')
        require(all(e.get('root_slot',s)==s and e.get('arm',arm)==arm and e.get('selection_digest',selection)==selection for e in own),'branch identity throughout journal')
        observations=[e['receipt'] for e in own if e['kind']=='fresh_observation']
        guards=[e for e in own if e['kind']=='guard_receipt']
        order=[('warmup',t,'off') for t in range(11)]+[( 'plan',t,mode) for t in range(11) for mode in (('off','on','off_after') if t in (0,5,10) else ('off',))]
        require([(f['phase'],f['step'],f['mode']) for f in observations]==order and
            [(g['phase'],g['step'],g['mode']) for g in guards]==order and all(g['accepted'] is True for g in guards),
            'all28 actual renders and guards per fixed plan')
        identity=lambda f:(f['phase'],f['step'],f['mode'],f['png'])
        guard_by_key={identity(g):g for g in guards}
        require(len(guard_by_key)==len(guards) and set(guard_by_key)=={identity(f) for f in observations},
            'one-to-one observation/guard join by phase,step,mode,png, not list position')
        returns=[e for e in own if e['kind']=='returned_transition']
        require([(e['phase'],e['step']) for e in returns]==[(ph,t) for ph in ('warmup','plan') for t in range(1,11)],'actual selected command chronology')
        for e,a in zip(returns,np.concatenate((moving.fixed_raw(),raw)),strict=True):
            require(native.bits(np.asarray(e['command'],np.float32),a) and native.bits(np.asarray(e['actual_command'],np.float32),a) and
                e['actual_float32_bits']==a.tobytes().hex(),'actual fixed-plan and warmup raw bits')
        pixels={}
        for f in observations:
            ph,t,mode=f['phase'],f['step'],f['mode'];a=native.image(decode(f));ref=old['off'][ph,t];g=guard_by_key[identity(f)]
            require(digest(a)==f['pixel_sha256'] and f['before']==f['after'] and f['with_target'] is (mode=='on'),'native same-body checkpoint render receipt')
            require(g['accepted'] is True and g['render_preserved_invariant'] is True and g['canonical_agent_pixels']>0,
                'matching fresh RGB observability guard actually accepted')
            if mode!='on':
                require(f['file_sha256']==ref['file_sha256'] and f['pixel_sha256']==ref['pixel_sha256'] and f['snapshot']==ref['snapshot'] and
                    g['old_off_match']==dict(file=True,pixels=True,state=True,snapshot=True,observation_state=True),'new off matches actual original fixed plan endpoint')
            else:require(g['old_off_match'] is None,'on not old off')
            pixels[ph,t,mode]=a
        require([x['step'] for x in row['checkpoints']]==[0,5,10],'all actual paired checkpoints')
        for cp in row['checkpoints']:
            t=cp['step'];require(cp['complete'] is True,'actual complete triplet')
            triple=[cp[k] for k in ('off_before','on','off_after')]
            for cp_receipt,mode in zip(triple,('off','on','off_after'),strict=True):
                observation=next(f for f in observations if f['phase']=='plan' and f['step']==t and f['mode']==mode)
                guard=guard_by_key[identity(observation)]
                require(all(cp_receipt.get(k)==v for k,v in observation.items()) and cp_receipt['old_off_match']==guard['old_off_match'],
                    'checkpoint shared observation fields plus post-observation guard-derived match')
            require(triple[0]['after']==triple[1]['after']==triple[2]['after'] and triple[0]['file_sha256']==triple[2]['file_sha256'] and
                native.bits(pixels['plan',t,'off'],pixels['plan',t,'off_after']),'literal off/on/off same-body checkpoint pointer')
        for c,mode in (('off','off'),('on','on')):
            require(native.bits(pixels['plan',0,mode],root['packets'][c]['rgb_history'][-1]),'plan root RGB exactly source same-condition context')
        root['fixed_plans'].append(dict(arm=arm,original_task_id=root['original_task_id'],commands=raw,
            selection_sha256=selection,planner_report_sha256=plan['compiler_report_SHA'],
            labels={c:{role:actual_label(root,c,raw,t,pixels['plan',t,mode],selection) for role,t in (('prefix',5),('terminal',10))} for c,mode in (('off','off'),('on','on'))}))
    return roots


if __name__=='__main__':main()
