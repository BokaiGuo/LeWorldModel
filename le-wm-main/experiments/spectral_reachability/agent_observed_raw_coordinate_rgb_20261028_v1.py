"""Actual fixed164 raw-coordinate cp10 PNG -> original frozen latent192.
MAIN HOST CUDA only. Own physical authority, never a projected old schema.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_observed_suffix_terminal_rgb_20261028_v1 as frozen
from experiments.spectral_reachability import agent_observed_raw_coordinate_physical_20261028_v1 as producer

NAME='agent_observed_raw_coordinate_rgb_20261028_v1'
AUTH='MAIN_OBSERVED_RAW164_TERMINAL_RGB_HOST_CUDA_ONLY'
ROOT=frozen.ROOT
bridge=frozen.bridge
require,sha,raw=frozen.require,frozen.sha,frozen.raw
CAP,FINAL,FLOOR,WALL=512*1024**2,128*1024**2,200*1024**2,1800
META_CAP=128*1024**2
ARGUMENTS=('physical_output','physical_report_sha','physical_prepared','physical_prepare_sha',
    'physical_receipt','physical_receipt_sha','physical_source_sha')
FROZEN=((frozen,'4c84f0d735981ec0a592d1df90ecaf6215390953f05ec7cd1dd438cb47b66427'),*frozen.FROZEN)
encode_guard=frozen.encode_guard
precision_contract=frozen.precision_contract
checked_precision=frozen.checked_precision


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=164,panels=4,jobs_per_panel=41,
        known_roots=2,fresh_roots=0,checkpoint=10,batch=1,
        baseline='four original suffix iterate0 controls; never optimized replacements',
        max_calls=dict(encoder=164,projector=164,predict=0,action_encoder=0,pred_proj=0,AD=0),
        no_dedup=True,no_padding=True,no_selection=True,retries=0,
        encoder_hidden=[1,257,192],latent=192,precision=precision_contract(),
        CUDA_only=True,CPU_fallback=False,new_physics=0,training=False,Qfit=False,
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,wall_seconds=WALL,
        P1_admission=False,freshness=False,scientific_admission=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


class Budget(bridge.rgb.Budget):
    def check(self,size=0,final=False):
        require(type(size) is int and size>=0,'nonnegative actual byte charge')
        require(final or self.clock()-self.started<=WALL,'whole164 job wall; no retry')
        require(self.used+size<=CAP-(0 if final else FINAL),'512MiB cap including128MiB final reserve')
        require(self.free()>=FLOOR+size,'200MiB floor before every write')


class Store(bridge.rgb.Store):
    """Reuse original exclusive writer/array codec, with our bounded ledger."""
    def __init__(self,output,budget):
        output=Path(output)
        require(output.is_absolute() and output.resolve()==output and output.parent==ROOT/'results'
            and output.name.startswith(NAME+'_'),'exclusive absolute own results path')
        require(type(budget) is Budget and shutil.disk_usage(output.parent).free>=CAP+FLOOR,
            'own164 full-cap resource ledger before directory creation')
        super().__init__(output,budget);self.index=0
    def event(self,event):
        self.put(f'event_{self.index:04}.json',event);self.index+=1


def expected_calls(rows):
    require(len(rows)==164 and [r['index'] for r in rows]==list(range(164)),
        'fixed ordered164 includes missing, never filtered or duplicated')
    n=sum(row['source'] is not None for row in rows)
    return {k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS}


def process(p,rows,store,encode,check=lambda:None):
    """Only already-authorized PNGs enter this native loop; no outcome routing."""
    require(raw(rows)==raw(p['rows']) and expected_calls(rows)==p['expected_calls'],
        'entire initial roster and planned resources')
    ordinal=0
    for row in rows:
        if row['source'] is None:continue
        check();scope=dict(row=row['index'],step=10,phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        pix=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pix)
        z=encode(pix,row,scope,ordinal);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)));ordinal+=1


def validate_native(p,r,events,load):
    """Exact arrays/events validation, independent of upstream JSON authority.

    build/fixed_rows establish which physical sources may enter p['rows'].
    This function does not certify a physical receipt, task or missing reason.
    """
    require(p['schema']==NAME+'.prepare' and r['schema']==NAME+'.report'
        and p['contract']==r['contract']==contract() and len(r['rows'])==164
        and expected_calls(p['rows'])==p['expected_calls'] and r['errors']==r['blocked']==[],
        'own exact schemas, complete164, no hidden failed resources')
    require(r['precision_before']==r['precision_after']==precision_contract()
        and p['shapes']['hidden']==[1,257,192],'original257/MATH/native precision ABI')
    modules=[];guards=[];want=[];ordinal=0;latents=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require(raw({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')})==
            raw({k:v for k,v in spec.items() if k not in ('status','latent')}),'whole source metadata unchanged')
        if spec['source'] is None:
            require(row==spec,'missing stays unchanged and null, never padded');latents.append(None);continue
        require(row['status']=='encoded','available image exactly once')
        pix=load(row['pixels']);bridge.same(pix,bridge.load_pixels(spec['source']),'actual physical PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'independent original PNG preprocessing')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=10,phase='encode')
        mods=row['modules']
        require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),
            'single-image actual encoder/projector module chain')
        en,proj=mods;ein=load(en['inputs'][0]);eout=load(en['output'])
        pin=load(proj['inputs'][0]);pout=load(proj['output'])
        bridge.same(ein,pre.reshape(1,3,224,224),'exact native encoder input bits')
        require(eout.shape==(1,257,192) and pout.shape==(1,192),'fixed real patch14/257 and projector192 ABI')
        bridge.same(pin,eout[:,0],'exact CLS, no averaging/pooling substitution')
        bridge.same(pout[0],z,'exact projector to saved192 bits')
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want.extend([dict(kind='module_attempt',entry=attempt),
                dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)])
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1;latents.append(z.tolist())
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules)))
        and r['guards']==guards and events==want
        and r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'full native event chronology/census/resources, unknown events rejected')
    return latents


def fixed_rows(p,strict,report,events,output):
    producer.validate_execution(p)
    require(strict['schema']==producer.NAME+'.consume' and strict['status']==producer.RECORDED
        and report['schema']==producer.NAME+'.report' and report['status']==producer.RECORDED
        and strict['errors']==report['errors']==[] and strict['rows']==report['rows']
        and strict['fixed_denominator']==164 and strict['known_roots']==2
        and strict['fresh_roots']==strict['inherited']==0 and strict['planner_authority']==p['planner_authority']
        and all(strict[k] is False for k in ('P1_admission','P3_admission','scientific_admission','freshness',
            'deployment_allowed','equal_total_compute','equal_FLOPs','independent_confirmation')),
        'actual full164 own physical receipt, not old suffix4 or partial')
    require(len(strict['rows'])==164,'all164 physical rows including native stops')
    producer.audit_guard(report,events,p);rows=[]
    for i,(plan,j,actual,ref) in enumerate(zip(p['plans'],p['jobs'],strict['rows'],p['prefix_reference'],strict=True)):
        require(actual['index']==plan['index']==j['index']==i and actual['plan']==plan
            and all(actual[k]==plan[k] for k in ('root_slot','condition','arm','task_id','selection_digest'))
            and actual['status'] in ('horizon_completed','native_stopped') and actual['first_error'] is None
            and actual['secondary_errors']==[],'exact task/raw plan identity, no engineering failure omission')
        task=producer.producer.core.validate_task(j['task']);goal=bridge.latent(task['goal'])
        native=p['source_roots'][j['root_slot']]
        require(all(plan[k]==native[v] for k,v in (('native_goal','goal'),('native_goal_pose','goal_pose'),
            ('native_root_sha256','root_sha256'))) and plan['native_goal_pixel_sha256']==native['goal_pixels'][j['condition']],
            'original native goal, root and condition')
        if actual['initial_exact']:
            require(all(actual[k]==native[k] for k in ('root_state','root_sha256','goal','goal_pose'))
                and actual['goal_pixel_sha256']==native['goal_pixels'][j['condition']],'actual original initial native identity')
        cps=actual['checkpoints'];steps=[c['step'] for c in cps]
        require(len(set(steps))==len(steps) and sorted(steps)==[s for s in (0,5,10) if s not in actual['missing_checkpoints']]
            and sorted(actual['missing_checkpoints'])==[s for s in (0,5,10) if s not in steps],'complete checkpoint census')
        guard=[e for e in events if e.get('branch')==i and e['kind']=='cp5_guard']
        if actual['own_returned']>5:
            require(len(guard)==1 and guard[0]['accepted'] is True and guard[0]['error'] is None
                and actual['prefix_reference_exact'] is True and producer.prefix_match(actual,ref),
                'exact genuine cp5 before every perturbation suffix')
        else:require(not guard,'no fabricated pre-suffix guard')
        cp10=[c for c in cps if c['step']==10];frame=None
        if cp10:
            f=cp10[0];t=[t for t in actual['trajectory'] if t['phase']=='plan' and t['step']==10]
            require(actual['own_returned']==10 and actual['initial_exact'] is True and len(t)==1
                and t[0]['condition']==j['condition'] and t[0]['overlay_index']==(1 if j['condition']=='on' else 0)
                and all(t[0][k]==f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot'))
                and len(f['state'])==7 and f['state']==f['snapshot']['state']==actual['final_state']
                and Path(f['png']).name==f['png'] and f['png'] not in ('','.','..'),'exact terminal PNG/native7/dispatch/time')
            frame=dict(deepcopy(f),path=str(Path(output)/f['png']))
        else:
            require(actual['status']=='native_stopped' and 0<=actual['own_returned']<10
                and (actual['terminated'] is True or actual['truncated'] is True
                    or actual['initial_exact'] is False and actual['warmup_returned']<10),'only genuine early stop missing')
        rows.append(dict(index=i,panel=j['slot'],local_index=j['local_index'],probe_id=j['id'],
            baseline=j['local_index']==0,root_slot=j['root_slot'],condition=j['condition'],arm=plan['arm'],
            coord=j['coord'],h=j['h'],sign=j['sign'],actual_delta=j['actual_delta'],
            task_id=plan['task_id'],original_task=deepcopy(task),model_task_binding=deepcopy(task['binding']),
            goal192=goal.tolist(),full10=deepcopy(plan['commands']),raw_float32_sha256=plan['raw_float32_sha256'],
            selection_digest=plan['selection_digest'],producer_job_sha256=j['job_sha256'],
            observed_context=deepcopy(j['observed_context']),observed_raw10=deepcopy(j['observed_raw10']),
            original_iterate0_context=deepcopy(j['original_iterate0_context']),provenance=deepcopy(j['provenance']),
            native_root_sha256=plan['native_root_sha256'],native_goal=deepcopy(plan['native_goal']),
            native_goal_pose=deepcopy(plan['native_goal_pose']),native_goal_pixel_sha256=plan['native_goal_pixel_sha256'],
            prefix_reference_sha256=plan['prefix_reference_sha256'],actual_cp5=[deepcopy(c) for c in cps if c['step']==5],
            actual_cp5_guard=deepcopy(guard),prefix_reference_exact=actual['prefix_reference_exact'],
            physical_status=actual['status'],evaluation={k:deepcopy(actual[k]) for k in
                ('final_state','final_metrics','native_success','terminated','truncated')},
            source=frame,status='not_started' if frame is not None else 'missing_physical_checkpoint10',latent=None))
    require([r['index'] for r in rows if r['baseline']]==[0,41,82,123],'all original four baseline controls')
    return rows


def build(physical_output,physical_report_sha,physical_prepared,physical_prepare_sha,physical_receipt,physical_receipt_sha,physical_source_sha):
    arguments=dict(zip(ARGUMENTS,(physical_output,physical_report_sha,physical_prepared,physical_prepare_sha,
        physical_receipt,physical_receipt_sha,physical_source_sha),strict=True))
    for k,v in arguments.items():
        if k.endswith('_sha'):producer.mixed.actual_hash(v)
        else:producer.mixed.absolute(v)
    for module,h in FROZEN:require(sha(module.__file__)==h,'frozen encoding chain source')
    code=str(Path(producer.__file__).resolve());output=Path(physical_output)
    require(sha(code)==physical_source_sha and sha(output/'report.json')==physical_report_sha
        and sha(physical_prepared)==physical_prepare_sha and sha(physical_receipt)==physical_receipt_sha,
        'all actual MAIN physical input SHA, no future defaults')
    strict=producer.consume(output,physical_report_sha)
    require(raw(strict)==raw(producer.read(physical_receipt,META_CAP)),'entire reconstructed164 persisted physical receipt')
    p=producer.read(physical_prepared,META_CAP);report=producer.read(output/'report.json',producer.META_CAP)
    require(report['execution_binding']['prepared']==physical_prepared
        and report['execution_binding']['prepare_sha256']==physical_prepare_sha
        and strict['physical_report_sha256']==physical_report_sha
        and p['bindings'][code]==strict['bindings'][code]==physical_source_sha,'own exact physical execution/source binding')
    events=[]
    with (output/'journal.jsonl').open('rb') as stream:
        while True:
            line=stream.readline(producer.META_CAP+1)
            if not line:break
            require(len(line)<=producer.META_CAP and line.endswith(b'\n'),'whole bounded native journal record')
            events.append(producer.mixed.decode(line))
    require(sha(output/'journal.jsonl')==report['journal_sha256'],'actual native journal bytes')
    rows=fixed_rows(p,strict,report,events,output);bindings=deepcopy(strict['bindings'])
    producer.mixed.merge(bindings,{physical_receipt:physical_receipt_sha,physical_prepared:physical_prepare_sha,
        str(output/'report.json'):physical_report_sha,code:physical_source_sha})
    producer.mixed.merge(bindings,bridge.rgb.MODELS)
    files=own_files()+producer.own_files()+[Path(m.__file__).resolve() for m in
        (frozen,frozen.template,bridge,bridge.native,bridge.rgb,bridge.pixels,frozen.rgb_base)]
    producer.mixed.merge(bindings,{str(f):sha(f) for f in files})
    fd=producer.read(p['arguments']['fd_prepared'],producer.META_CAP)
    require(sha(p['arguments']['fd_prepared'])==p['arguments']['fd_prepare_sha']
        and fd['expected_model_hash']==bridge.rgb.MODEL_TENSOR_SHA
        and Path(fd['policy']).resolve()==bridge.rgb.POLICY.resolve(),'original paid model/policy')
    producer.producer.F.source.kernel.validate_normalizer_source(ROOT)
    shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json'))
    require(shapes['hidden']==[1,257,192],'real224/patch14 config derived257 ABI')
    for row in rows:
        require(row['model_task_binding']['model_sha256']==fd['expected_model_hash'],'every task same frozen model')
        if row['source'] is not None:bridge.load_pixels(row['source'])
    bridge.verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=arguments,rows=rows,bindings=bindings,
        expected_calls=expected_calls(rows),shapes=shapes,policy=fd['policy'],expected_model_hash=fd['expected_model_hash'],
        physical_authority=deepcopy(p['planner_authority']),actual_new_physical_counts=deepcopy(strict['counts']),
        inherited_compute=deepcopy(fd['inherited_compute']),inherited_resources=deepcopy(fd['inherited_resources']))


def verify(path,expected):
    producer.mixed.actual_hash(expected);producer.mixed.absolute(str(path))
    require(sha(path)==expected,'own actual prepare SHA');p=producer.read(path,META_CAP)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'whole physical/source164 rebuilt')
    return p


def write(path,value):
    path=Path(producer.mixed.absolute(str(path)));body=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists()
        and not path.is_symlink() and len(body)<=META_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(body),
        'exclusive bounded metadata, no overwrite')
    with path.open('xb') as stream:require(stream.write(body)==len(body),'full metadata');stream.flush();os.fsync(stream.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit host CUDA authority before I/O')
    budget=Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=guard=model=None;before=after=None;source_ok=False;flags=None
    precision_before=precision_after=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model)
        require(before==p['expected_model_hash'],'original frozen model before first encode')
        for module in model.modules():
            require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'installed native class source closure')
        trace=bridge.Trace(model,torch,store,store.check,p['expected_calls'])
        # Frozen Ledger n counts four-image groups, not individual calls.
        guard=bridge.native.Ledger(41,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pix,row,scope,ordinal):
            checked_precision(torch);guard.image=ordinal
            with bridge.physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pix,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pix),'independent original pixel preprocessing')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'actual native192 ABI')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,encode_guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            precision_before=checked_precision(torch)
            process(p,rows,store,encode,store.check)
            precision_after=checked_precision(torch)
    except BaseException as e:errors.append(bridge.physical.kernel.error('terminal_encode164',e))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(bridge.physical.kernel.error('final_source_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash']
            and precision_before==precision_after==precision_contract()
            and bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors,
            precision_before=precision_before,precision_after=precision_after)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='raw164_encoded_not_admitted' if complete else 'partial_or_failed',
            contract=contract(),prepared=str(Path(prepared).resolve()),prepare_sha256=expected,**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,
            model_unchanged=before is not None and before==after,source_verified=source_ok,native_seal_sha256=h,
            artifact_sha256=deepcopy(store.hashes),event_count=store.index,elapsed_seconds=time.monotonic()-budget.started,
            P1_admission=False,scientific_admission=False,freshness=False)
        store.put('report.json',report,final=True)
    require(complete,'partial evidence retained; no retry '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,z):
    return dict(index=spec['index'],panel=spec['panel'],local_index=spec['local_index'],probe_id=spec['probe_id'],
        baseline=spec['baseline'],root_slot=spec['root_slot'],condition=spec['condition'],
        coord=spec['coord'],h=spec['h'],sign=spec['sign'],actual_delta=spec['actual_delta'],
        task_id=spec['task_id'],model_task_binding=spec['model_task_binding'],selection_digest=spec['selection_digest'],
        producer_job_sha256=spec['producer_job_sha256'],raw_float32_sha256=spec['raw_float32_sha256'],
        actual192=z,goal192=spec['goal192'],actual_Q=None if z is None else bridge.mse(z,spec['goal192']),
        source=spec['source'],
        latent_missing=z is None,actual_cp5=spec['actual_cp5'],actual_cp5_guard=spec['actual_cp5_guard'],
        prefix_reference_exact=spec['prefix_reference_exact'],evaluation=spec['evaluation'],
        physical_status=spec['physical_status'],baseline_is_original_suffix=True if spec['baseline'] else None,
        physical_benefit_claim=False,P1_admission=False)


def consume(output,expected):
    output=Path(producer.mixed.absolute(str(output)));producer.mixed.actual_hash(expected)
    require(output.resolve()==output and sha(output/'report.json')==expected,'own actual report path/SHA')
    r=producer.read(output/'report.json',FINAL);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='raw164_encoded_not_admitted'
        and r['model_unchanged'] is r['source_verified'] is True
        and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['P1_admission'] is r['scientific_admission'] is r['freshness'] is False
        and 0<=r['elapsed_seconds']<WALL,'clean original model/source/whole-stage limits')
    files=list(output.iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files)
        and {f.name for f in files}==set(r['artifact_sha256'])|{'report.json'}
        and sum(f.stat().st_size for f in files)<=CAP,'entire flat artifact census/cap')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'every actual artifact SHA')
    require(producer.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'])
        and producer.read(output/'initial_roster.json',FINAL)==p['rows'],'exact execution/initial164 roster')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and producer.read(output/'native_seal.json',FINAL)==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors','precision_before','precision_after')},'durable native seal')
    require(type(r['event_count']) is int and 0<=r['event_count']<=164*10,'bounded complete event count')
    z=validate_native(p,r,[producer.read(output/f'event_{i:04}.json',META_CAP) for i in range(r['event_count'])],
        lambda a:bridge.rgb.load_array(a,blob))
    rows=[analyze(spec,v) for spec,v in zip(p['rows'],z,strict=True)]
    bridge.verify_bindings(p['bindings'])
    require(sha(output/'report.json')==expected and sha(r['prepared'])==r['prepare_sha256'],'final report/prepare/source bytes')
    return dict(schema=NAME+'.consume',status='raw164_RGB_verified_not_admitted',rows=rows,counts=r['counts'],
        report_sha256=expected,physical_authority=p['physical_authority'],physical_input_sha256=p['arguments'],
        actual_new_physical_counts=p['actual_new_physical_counts'],inherited_compute=p['inherited_compute'],
        inherited_resources=p['inherited_resources'],fixed_denominator=164,panels=4,jobs_per_panel=41,baseline_indices=[0,41,82,123],
        known_roots=2,fresh_roots=0,total_FLOPs=None,consume_new_neural_calls=0,consume_new_physics=0,
        P1_admission=False,scientific_admission=False,freshness=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt',*(s.replace('_','-') for s in ARGUMENTS)):
        parser.add_argument('--'+k)
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build(**{k:getattr(a,k) for k in ARGUMENTS}))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN host CUDA flag required');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:require(a.receipt,'new durable receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
