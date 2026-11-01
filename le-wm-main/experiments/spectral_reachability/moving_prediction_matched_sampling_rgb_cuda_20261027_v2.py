"""Four fixed sampling plans: source goals + actual0/5/10 RGB, MAIN CUDA only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_overlay_recompile_rgb_cuda_20261027_v1 as arrays
from experiments.spectral_reachability import moving_prediction_matched_sampling_physical_20261027_v1 as physical
from experiments.spectral_reachability import moving_prediction_matched_sampling_cuda_20261027_v1 as planner

NAME='moving_prediction_matched_sampling_rgb_cuda_20261027_v2'
ROOT=planner.ROOT
require,sha,canonical,read=arrays.require,arrays.sha,arrays.canonical,arrays.read
digest,flat,decode=arrays.digest,arrays.flat,arrays.decode
POLICY,MODELS,MODEL_TENSOR_SHA=arrays.POLICY,arrays.MODELS,arrays.MODEL_TENSOR_SHA
PHYSICAL_SHA='50552d50ce3d97c9a25f938ef2b47e99bc66c88fa1644659611b5bb32843884a'
ARRAYS_SHA='70c03508d775b8b488c12133da929bae607d08d6163d5e1c6578edd9a2422bcc'
PLANNER_SHA='72195df2a7f1d10948c2bf5c494dd26e66400eaa2e269737332cbd876e599d40'
PLANNER_REPORT=ROOT/'results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json'
PLANNER_REPORT_SHA='60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8'
AUTH='MAIN_PREDICTION_MATCHED_SAMPLING_RGB_V2_CUDA_ONLY'
KEYS=[(s,a) for s in (0,1) for a in planner.ARMS]
CAP,FLOOR,FINAL,WALL=32*1024**2,200*1024**2,2*1024**2,600


def contract(goals=2):
    require(type(goals) is int and 1<=goals<=2,'actual source-provenance goals')
    return dict(fixed_denominator=4,arms=list(planner.ARMS),roots=2,checkpoints=[0,5,10],
        goals=goals,images=12+goals,caps=dict(encoder=12+goals,projector=12+goals),
        goal_dedup='identical source NPZ absolute path + file SHA + goal pixel SHA only; never arbitrary pixel dedup',
        checkpoint_dedup=False,all_encodings_sealed_before_Q=True,
        Q='float64 mean192 identity projected CLS, no fit',new_predict=0,new_action_encoder=0,
        new_pred_projector=0,new_AD=0,new_physics=0,training=False,Qfit=False,selection=False,
        CUDA_only=True,TF32=False,autocast=False,MATH=True,no_retry=True,
        output_cap_bytes=CAP,floor_bytes=FLOOR,final_reserve_bytes=FINAL,wall_seconds=WALL,
        wall_scope='verify/load/encode/final verify at bounded checkpoints; no preemption inside blocking calls',
        P1_admission=False,independent_confirmation=False,new_independent_samples=0,
        equal_total_compute=False,physical_and_planning_costs='inherited separate; not refunded/added to encode calls')


def identity(row):
    require(type(row['root_slot']) is int and row['root_slot'] in (0,1),'literal root slot')
    return row['root_slot'],row['arm'],row['selection_digest']



def validate_physical_meter(report,p,events):
    """Recount FINAL physical v1 receipt ABI, not obsolete overlay event names."""
    counts={k:dict(attempted=0,completed=0) for k in p['contract']['caps']}
    intents={};terminals={};stack=[]
    for e in events:
        kind=e['kind']
        require(kind not in ('operation_attempted','operation_completed'),'obsolete operation event ABI')
        if kind not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        r=e['receipt'];idx=r['operation_index'];resource=r['resource']
        require(type(idx) is int and resource in counts,'actual receipt resource/index')
        if kind=='operation_dispatch_intent':
            require(idx==len(intents) and r['status']=='pending','sequential dispatch intent')
            intents[idx]=r;stack.append(idx)
        else:
            require(kind=='operation_returned' and r['status']=='returned' and idx in intents
                and idx not in terminals and stack and stack.pop()==idx
                and dict(r,status='pending')==intents[idx],'exact nested successful native return')
            terminals[idx]=r
            counts[resource]['attempted']+=1;counts[resource]['completed']+=1
    require(not stack and len(intents)==len(terminals)
        and report['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))],
        'every actual entry and return durable, no intent-count substitution')
    require(counts==report['counts']=={k:dict(attempted=v,completed=v) for k,v in p['contract']['caps'].items()},
        'full actual four-plan physical census')
    return counts


def assemble(report,p,manifest,events,tasks,goals):
    """Pure saved schema mapping; all four complete rows or fail before CUDA."""
    require(report['schema']==physical.NAME+'.report' and report['status']=='matched_sampling_physical_recorded_not_admitted'
        and report['errors']==[] and report['fixed_branch_denominator']==4 and report['P1_admission'] is False,'actual complete fixed physical cohort')
    require(p['schema']==physical.NAME+'.prepare' and manifest['schema']==physical.NAME+'.selection' and
        manifest['arms']==list(planner.ARMS) and manifest['selection_fixed_before_physics'] is True and
        manifest['physical_outcome_selection'] is False and p['contract']==report['contract']==physical.contract(manifest['wall_seconds'])
        and report['sourcepack_sha256']==p['sourcepack_sha256']==manifest['sourcepack_sha256']==physical.SOURCE_SHA and
        report['source_prepare_id']==p['source_prepare_id'],'source/prepare/fixed arms')
    require([(r['root_slot'],r['arm']) for r in report['rows']]==KEYS and
        [(r['root_slot'],r['arm']) for r in p['plans']]==KEYS and len(tasks)==len(goals)==2,'four ordered plans, two source tasks')
    require(events[0]['kind']=='run_started' and events[0]['prepare']==p and
        events[-1]['kind']=='run_finished' and events[-1]['counts']==report['counts'] and
        events[-1]['status']==report['status'] and events[-1]['errors']==[],'durable start/terminal')
    validate_physical_meter(report,p,events)
    images=[];goal_indices={};origin={}
    for s,item in enumerate(goals):
        require(item['kind']=='goal' and item['root_slot']==s and item['pixel_sha256']==tasks[s]['interface']['gn_task']['goal_source_sha256'],'actual task goal pixels')
        key=(item['path'],item['file_sha256'],item['pixel_sha256'])
        if key not in origin:
            origin[key]=len(images);images.append(dict(deepcopy(item),index=len(images),source_slots=[s]))
        else:images[origin[key]]['source_slots'].append(s)
        goal_indices[s]=origin[key]
    rows=[]
    for i,(row,plan) in enumerate(zip(report['rows'],p['plans'],strict=True)):
        s,a=KEYS[i];t=tasks[s];g=t['interface']['gn_task'];tc=t['task_contract']
        require(identity(row)==identity(plan) and row['branch_index']==i and row['selected_plan']==plan and
            row['status'] in ('horizon_completed','native_success') and row['initial_exact'] is True and
            row['full_horizon'] is True and row['own_attempted']==row['own_returned']==row['warmup_returned']==10 and
            row['first_error'] is None and row['secondary_errors']==[] and not row['truncated'] and row['PNG']==22,'actual full10; missing endpoints fail closed')
        require(plan['task_id']==t['interface']['task_id'] and plan['task_sha256']==t['task_sha256'] and plan['task_export']==t['task_export'] and
            plan['blind_sha256']==t['blind_sha256'] and plan['task_contract_sha256']==g['task_contract_sha256'] and
            plan['root_sha256']==manifest['roots'][s]['root_sha256']==physical.source.root_hash(np.asarray(row['root_state']),np.asarray(row['goal'])) and
            physical.source.digest({k:v for k,v in plan.items() if k!='selection_digest'})==plan['selection_digest'],'own root/goal/task/selected plan')
        scoped=[e for e in events if e.get('branch')==i]
        require(all(identity(e)==identity(plan) for e in scoped),'no crossed event identities')
        starts=[e for e in scoped if e['kind']=='branch_started']
        closed=[e for e in events if e['kind']=='branch_closed' and e['receipt']['branch_index']==i]
        warm=planner.old.moving.fixed_raw()
        require(len(starts)==len(closed)==1 and starts[0]['plan']==plan and closed[0]['receipt']==row and
            starts[0]['raw_warmup']==warm.astype(float).tolist() and starts[0]['raw_float32_bits']==warm.tobytes().hex(),'actual nonzero history/branch close')
        wr=[e for e in scoped if e['kind']=='warmup_return']
        require([e['step'] for e in wr]==list(range(1,11)) and [e['raw_action'] for e in wr]==warm.astype(float).tolist()
            and [e['raw_float32_bits'] for e in wr]==[v.tobytes().hex() for v in warm],'ten literal moving warmup actions')
        observations=[e for e in scoped if e['kind']=='fresh_observation'];guards=[e for e in scoped if e['kind']=='guard_receipt']
        census=[(ph,k) for ph in ('warmup','plan') for k in range(11)]
        require([(e['receipt']['phase'],e['receipt']['step']) for e in observations]==census and
            [(e['phase'],e['step']) for e in guards]==census and all(e['accepted'] is True and e['render_preserved_dynamics'] is True for e in guards),'all22 actual captures/guards')
        for obs,guard in zip(observations,guards,strict=True):
            c=obs['receipt'];require(obs['index']<guard['index'] and c['state']==c['observation_state']==c['snapshot']['state'],'persisted scene then guard')
        require([observations[k]['receipt']['pixel_sha256'] for k in (0,5,10)]==tc['pixel_sha256'][:3],'original actual RGB3')
        trans=[e for e in scoped if e['kind']=='returned_transition' and e['phase']=='plan']
        require([e['step'] for e in trans]==list(range(1,11)) and [e['command'] for e in trans]==plan['commands'] and
            all(not e['terminated'] and not e['truncated'] for e in trans[:-1]),'own actual ten commands')
        attached=[e for e in scoped if e['kind']=='goal_attached']
        require(len(attached)==1 and attached[0]['goal']==row['goal'] and attached[0]['goal_rgb_sha256']==g['goal_source_sha256'] and
            all(e['returned_goal_state']==row['goal'] and e['returned_goal_rgb_sha256']==g['goal_source_sha256'] for e in trans),'actual offline goal throughout commands')
        cps=row['checkpoints'];require([c['step'] for c in cps]==[0,5,10] and all(type(c['step']) is int for c in cps),'all literal0/5/10, no substitution')
        out=dict(root_slot=s,arm=a,selection_digest=plan['selection_digest'],task_id=plan['task_id'],goal_image=goal_indices[s],
            physical_status=row['status'],initial_metrics=deepcopy(row['initial_metrics']),final_metrics=deepcopy(row['final_metrics']),checkpoints=[])
        for c in cps:
            k=c['step'];require(c==observations[11+k]['receipt'] and (k==0 or c['state']==trans[k-1]['state']),'checkpoint own command/journal bridge')
            if k==0:require(c['state']==row['root_state'] and c['pixel_sha256']==tc['pixel_sha256'][2] and c['snapshot']==observations[10]['receipt']['snapshot'],'actual root')
            if k==10:require(c['state']==row['final_state'],'actual terminal')
            idx=len(images);images.append(dict(index=idx,kind='checkpoint',row=i,root_slot=s,arm=a,step=k,
                selection_digest=plan['selection_digest'],path=c['png'],file_sha256=c['file_sha256'],pixel_sha256=c['pixel_sha256'],physical=deepcopy(c)))
            out['checkpoints'].append(dict(step=k,image=idx,latent=None,Q=None,status='not_started'))
        rows.append(out)
    return dict(rows=rows,images=images,contract=contract(len(origin)),physical_resources=deepcopy(report['counts']),
        physical_status=report['status'],physical_errors=deepcopy(report['errors']))


def build(physical_report,physical_sha,physical_prepare_sha,physical_source_sha):
    for h in (physical_sha,physical_prepare_sha,physical_source_sha):arrays.hash_value(h)
    require(physical_source_sha==PHYSICAL_SHA==sha(physical.__file__) and sha(arrays.__file__)==ARRAYS_SHA and
        sha(planner.__file__)==PLANNER_SHA,'frozen whitelisted provider/helpers')
    rp=Path(physical_report).absolute();require(sha(rp)==physical_sha,'external actual physical report')
    r=read(rp,32*1024**2);bp=rp.parent/'execution_binding.json';binding=read(bp)
    require(binding==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and binding['prepare_sha256']==physical_prepare_sha,'external physical prepare and execution binding')
    own=physical.consume(str(rp.parent),physical_sha)  # Full saved producer ABI; no simulator/model.
    p=own['prepared'];require(p==physical.verify(binding['prepared'],physical_prepare_sha)
        and own['checkpoint_complete'] is True,'actual own-consumed complete physical prepare')
    manifest=physical.read(p['manifest']);check_planner_authority(manifest,p)
    pack=physical.kernel.read(manifest['sourcepack'])
    require(len(pack['rows'])==2,'two actual source payloads')
    tasks=[];goals=[]
    for s in (0,1):
        plans=[x for x in p['plans'] if x['root_slot']==s];require(len(plans)==2,'both sampling arms per root')
        path,h=plans[0]['task_export'],plans[0]['task_sha256']
        interface=planner.old.load_export(path,h);e=planner.old.read(path)
        require(e['source_slot']==s and all(x['task_export']==path and x['task_sha256']==h for x in plans),'same original moving task per root')
        tasks.append(dict(interface=interface,task_export=path,task_sha256=h,blind_sha256=e['blind_sha256'],task_contract=e['task_contract']))
        row=pack['rows'][s];npz=physical.kernel.flat(Path(manifest['sourcepack']).parent,row['blind_payload'])
        require(sha(npz)==row['blind_payload_sha256']==e['blind_sha256'],'actual source NPZ identity')
        tc=e['task_contract'];goals.append(dict(kind='goal',root_slot=s,path=str(npz.absolute()),file_sha256=e['blind_sha256'],
            pixel_sha256=tc['pixel_sha256'][3],packet_metadata=dict(rgb_history_pixel_sha256=tc['pixel_sha256'][:3],raw_action_sha256=tc['raw_action_sha256'])))
    jp=rp.parent/'journal.jsonl';require(sha(jp)==r['journal_sha256'],'physical journal file SHA')
    events=physical.kernel.journal(jp,r['journal_chain_tail']);out=assemble(r,p,manifest,events,tasks,goals)
    bindings=deepcopy(p['bindings'])
    def bind(path,expected=None):
        path=str(Path(path).absolute());actual=sha(path)
        require((expected is None or actual==expected) and (path not in bindings or bindings[path]==actual),'nonconflicting actual input SHA')
        bindings[path]=actual
    for path,h in MODELS.items():bind(path,h)
    from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as native
    for path in native.source_files()+[Path(__file__),Path(arrays.__file__),Path(physical.__file__),Path(planner.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),rp,bp,jp,Path(binding['prepared'])]:bind(path)
    for ev in events:
        if ev['kind']=='fresh_observation':bind(flat(rp.parent,ev['receipt']['png']),ev['receipt']['file_sha256'])
    for item in out['images']:
        if item['kind']=='checkpoint':item['path']=str(flat(rp.parent,item['path']))
        bind(item['path'],item['file_sha256']);arrays.load_image(item)
    out.update(schema=NAME+'.prepare',arguments=dict(physical_report=str(rp),physical_sha=physical_sha,
        physical_prepare_sha=physical_prepare_sha,physical_source_sha=physical_source_sha),shapes=arrays.config_shapes(read(POLICY/'config.json')),
        bindings=bindings,policy=str(POLICY),expected_model_hash=MODEL_TENSOR_SHA)
    for path,h in bindings.items():require(sha(path)==h,'final complete prepare seal')
    return out


def check_planner_authority(manifest,p):
    entries=manifest['authorities']
    require(len(entries)==1 and entries[0]['authority']==planner.NAME and
        entries[0]['output']==str(PLANNER_REPORT.parent) and entries[0]['report_sha256']==PLANNER_REPORT_SHA and
        entries[0]['source_sha256']==PLANNER_SHA,'only MAIN actual matched sampling report, not old/alternative candidates')
    require(all(x['compiler_authority']==planner.NAME and x['compiler_report']==str(PLANNER_REPORT) and
        x['compiler_report_sha256']==PLANNER_REPORT_SHA for x in p['plans']),'all four selected plans same actual planner report')


def prepare(output,**arguments):
    p=build(**arguments);raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<=2*1024**2 and not path.exists() and not path.is_symlink() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'NEW prepare/2MiB/floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'full prepare write');f.flush();os.fsync(f.fileno())
    arrays.syncdir(path.parent);return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),images=len(p['images']))


def verify(prepared,expected):
    arrays.hash_value(expected);require(sha(prepared)==expected,'external RGB prepare SHA');p=read(prepared)
    require(p==build(**p['arguments']) and sha(prepared)==expected,'entire saved source/physical/model reconstruction');return p


class Budget(arrays.Budget):
    def start_gate(self):
        self.check();require(self.free()>=FLOOR+CAP,'full output allocation plus200MiB reserve before CUDA')
    def check(self,size=0,final=False):
        require(final or self.clock()-self.started<=WALL,'whole-job checkpoint wall600s')
        require(self.used+size<=CAP-(0 if final else FINAL) and self.free()>=FLOOR+size,'32MiB incl final /200MiB floor')


class Trace(arrays.Trace):
    def __init__(self,torch,shapes,store,check,limit):
        super().__init__(torch,shapes,store,check);self.limit=limit;self.fault=False
    def before(self,kind,index):
        self.check();require(not self.fault,'no retry after failed encoding')
        current=[e for e in self.entries if e['image']==index]
        require([(e['resource'],e['completed']) for e in current]==([] if kind=='encoder' else [('encoder',True)]),'encoder then projector once')
        require(sum(e['resource']==kind for e in self.entries)<self.limit,'actual declared image module cap')
    def dispatch(self,kind,tensor,callback,*args,**kwargs):
        self.before(kind,self.image);require(not self.torch.is_grad_enabled(),'inference only')
        ordinal=len(self.entries)+1;prefix=f'i{self.image:02d}_{kind}'
        e=dict(image=self.image,resource=kind,completed=False,inputs=[self.save(tensor,prefix+'_input')])
        require(e['inputs'][0]['shape']==self.shapes['pixels' if kind=='encoder' else 'cls'],'actual module input shape')
        if kind=='encoder':require(self.preprocessed is not None,'actual preprocess');e['preprocessed']=deepcopy(self.preprocessed)
        self.store.put(f'module_{ordinal:02d}_intent.json',e)
        self.entries.append(e)  # Actual callback entry; persistence failures above do not invent a call.
        try:
            result=callback(*args,**kwargs);self.torch.cuda.synchronize();e['completed']=True
            output=result.last_hidden_state if kind=='encoder' else result
            e['output']=self.save(output,prefix+'_output')
            require(e['output']['shape']==self.shapes['hidden' if kind=='encoder' else 'projected'],'config-derived actual output')
            self.store.put(f'module_{ordinal:02d}_returned.json',e);return result
        except BaseException as exc:
            self.fault=True;e['error']=str(exc)[:1000]
            try:self.store.put(f'module_{ordinal:02d}_failed.json',e,final=True)
            except BaseException as secondary:exc.add_note('module failure receipt: '+repr(secondary))
            raise


def evaluate(p,store,encode,loader,finalize,entries=None):
    rows=deepcopy(p['rows']);images=[dict(deepcopy(i),status='not_started',latent=None,modules=[]) for i in p['images']]
    values={};errors=[];meta={};finished=False
    try:
        for item in images:
            store.check();idx=item['index'];item['status']='attempted';pixels,original=loader(item)
            require(pixels.shape==(224,224,3) and pixels.dtype==np.uint8 and digest(pixels.tobytes())==item['pixel_sha256'] and digest(original)==item['file_sha256'],'unaltered original RGB bytes')
            filename=f'i{idx:02d}_original.'+('npz' if item['kind']=='goal' else 'png')
            item['original']=dict(file=filename,sha256=store.write(filename,original));item['rgb']=store.array(f'i{idx:02d}_rgb',pixels)
            z,receipts=encode(pixels.copy(),idx)
            require(z.shape==(192,) and z.dtype==np.float32 and np.isfinite(z).all(),'native projected192')
            arrays.validate_chain(receipts,z,p['shapes'],store.blob)
            require(all(e['image']==idx for e in receipts),'actual module image identity')
            item.update(latent=store.array(f'i{idx:02d}_latent',z),modules=deepcopy(receipts),status='encoded');values[idx]=z.copy()
        meta=finalize();finished=True
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)[:1500]))
        for item in images:
            if item['status']=='attempted':item['status']='failed'
    if not finished:
        try:meta=finalize()
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)[:1500]))
    actual=deepcopy(entries if entries is not None else [e for im in images for e in im['modules']])
    counts={k:dict(attempted=sum(e['resource']==k for e in actual),completed=sum(e['resource']==k and e['completed'] for e in actual)) for k in ('encoder','projector')}
    good=not errors and meta.get('source_verified') is True and meta.get('model_hash_before')==meta.get('model_hash_after')==p['expected_model_hash'] and not meta.get('blocked') and all(i['status']=='encoded' for i in images)
    good=good and counts=={k:dict(attempted=len(images),completed=len(images)) for k in counts}
    seal=dict(schema=NAME+'.encoding_seal',images=images,entries=actual,counts=counts,meta=meta,errors=deepcopy(errors),
        input_binding=p['arguments'],contract=p['contract'],shapes=p['shapes'])
    seal_sha=store.put('encoding_seal.json',seal,final=True)
    if good:
        try:store.check()
        except BaseException as exc:good=False;errors.append(dict(type=type(exc).__name__,message=str(exc)))
    for row in rows:
        for cp in row['checkpoints']:
            im=images[cp['image']];cp.update(latent=im['latent'],status=im['status'])
            if good:cp['Q']=float(np.mean((values[cp['image']].astype(np.float64)-values[row['goal_image']].astype(np.float64))**2))
    if good:store.put('metrics.json',dict(encoding_seal_sha256=seal_sha,rows=rows),final=True)
    r=dict(schema=NAME+'.report',status='encoded_sealed' if good else 'partial_or_failed',rows=rows,images=images,
        counts=counts,errors=errors,meta=meta,encoding_seal_sha256=seal_sha,input_binding=p['arguments'],contract=p['contract'],shapes=p['shapes'],
        physical_resources=p['physical_resources'],physical_status=p['physical_status'],physical_errors=p['physical_errors'],
        new_predict=0,new_AD=0,new_physics=0,training=False,P1_admission=False,artifact_sha256=deepcopy(store.hashes))
    store.put('report.json',r,final=True);return r


class Store(arrays.Store):
    def blob(self,name):return flat(self.path,name).read_bytes()


def run(prepared,expected,output,authorization=None):
    require(authorization==AUTH,'MAIN explicit CUDA-only authorization')
    budget=Budget();p=verify(prepared,expected)
    budget.free=lambda:shutil.disk_usage(Path(output).absolute().parent).free
    budget.start_gate();store=Store(output,budget)
    store.put('execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    stack=ExitStack();trace=None;model=None;before=None;host=None;restore=[]
    def finalize():
        stack.close();require(not restore,'native guard restoration failure');budget.check()
        require(model is not None,'model unavailable')
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and not torch.is_autocast_enabled('cuda'),'TF32/autocast off')
        native.frozen(model);after=native.tensor_hash(model);verify(prepared,expected);budget.check()
        return dict(model_hash_before=before,model_hash_after=after,source_verified=True,cuda_verified=True,host=host,
            MATH=True,TF32=False,autocast=False,blocked=[] if trace is None else trace.blocked)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as native
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        native.frozen(model);before=native.tensor_hash(model);require(before==MODEL_TENSOR_SHA,'frozen original tensor SHA')
        trace=Trace(torch,p['shapes'],store,store.check,len(p['images']));transform=p0.image_transform()
        stack.enter_context(native.guard(model,torch,trace,restore));stack.enter_context(torch.inference_mode());stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        def encode(pixels,index):
            trace.image=index;x=p0._image_tensor(pixels,transform=transform,device=device)
            require(x.device.type=='cuda' and x.dtype==torch.float32 and tuple(x.shape)==(1,1,3,224,224),'original CUDA transform')
            trace.preprocessed=trace.save(x,f'i{index:02d}_preprocessed')
            z=model.encode({'pixels':x})['emb'];torch.cuda.synchronize(device)
            require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'original CUDA model.encode')
            return z[0,0].detach().cpu().numpy().copy(),[deepcopy(e) for e in trace.entries if e['image']==index]
        return evaluate(p,store,encode,arrays.load_image,finalize,trace.entries)
    except BaseException as exc:
        try:store.put('run_error.json',dict(type=type(exc).__name__,message=str(exc)),final=True)
        except BaseException as secondary:exc.add_note('run error persistence: '+repr(secondary))
        if 'report.json' not in store.hashes:
            def failed(item):raise ValueError('initialization unavailable: '+str(exc))
            try:evaluate(p,store,None,failed,finalize,[] if trace is None else trace.entries)
            except BaseException as secondary:exc.add_note('partial report persistence: '+repr(secondary))
        raise
    finally:stack.close()


def consume_report(r,p,blob):
    require(r['schema']==NAME+'.report' and r['status']=='encoded_sealed' and r['errors']==[] and r['P1_admission'] is False and
        r['contract']==p['contract']==contract(p['contract']['goals']) and r['shapes']==p['shapes'] and r['input_binding']==p['arguments'],'own successful bound report')
    meta=r['meta'];require(meta['cuda_verified'] is True and meta['source_verified'] is True and meta['model_hash_before']==meta['model_hash_after']==MODEL_TENSOR_SHA and
        meta['MATH'] is True and meta['TF32'] is meta['autocast'] is False and meta['blocked']==[],'actual frozen CUDA receipts, not RAM evidence')
    expected_files={'execution_binding.json','encoding_seal.json','metrics.json'}
    for i,item in enumerate(r['images']):
        expected_files.update((item['original']['file'],item['rgb']['file'],item['latent']['file']))
        for ordinal,e in enumerate(item['modules'],start=2*i+1):
            expected_files.update((f'module_{ordinal:02d}_intent.json',f'module_{ordinal:02d}_returned.json',e['output']['file']))
            expected_files.update(x['file'] for x in e['inputs'])
            if 'preprocessed' in e:expected_files.add(e['preprocessed']['file'])
    require(set(r['artifact_sha256'])==expected_files,'full original/module/array/seal artifact population')
    for name,h in r['artifact_sha256'].items():require(digest(blob(name))==h,'every artifact SHA')
    seal=decode(blob('encoding_seal.json'));require(digest(blob('encoding_seal.json'))==r['encoding_seal_sha256'] and
        seal==dict(schema=NAME+'.encoding_seal',images=r['images'],entries=[e for i in r['images'] for e in i['modules']],
        counts=r['counts'],meta=meta,errors=[],input_binding=p['arguments'],contract=p['contract'],shapes=p['shapes']),'full seal before Q')
    require(len(r['images'])==len(p['images'])==p['contract']['images'] and r['counts']=={k:dict(attempted=len(p['images']),completed=len(p['images'])) for k in ('encoder','projector')},'actual all declared image calls')
    values={}
    for i,(item,expected) in enumerate(zip(r['images'],p['images'],strict=True)):
        require(all(item[k]==v for k,v in expected.items()) and item['index']==i and item['status']=='encoded','prepared exact image provenance')
        original=blob(item['original']['file']);require(digest(original)==item['original']['sha256']==item['file_sha256'],'saved exact original bytes')
        pixels=arrays.load_array(item['rgb'],blob);z=arrays.load_array(item['latent'],blob)
        require(pixels.dtype==np.uint8 and pixels.shape==(224,224,3) and digest(pixels.tobytes())==item['pixel_sha256'] and z.dtype==np.float32 and z.shape==(192,),'rawpixel/native192')
        if item['kind']=='goal':require(np.array_equal(arrays.packet(original)['goal_rgb'],pixels),'actual source goal not default renderer')
        else:
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(original)) as im:require(im.format=='PNG' and im.mode=='RGB' and np.array_equal(np.asarray(im),pixels),'original PNG byte/pixel bridge')
        arrays.validate_chain(item['modules'],z,p['shapes'],blob)
        for ordinal,e in enumerate(item['modules'],start=2*i+1):
            intent=decode(blob(f'module_{ordinal:02d}_intent.json'));ret=decode(blob(f'module_{ordinal:02d}_returned.json'))
            require(ret==e and intent=={k:v for k,v in dict(e,completed=False).items() if k!='output'} and e['image']==i,'durable actual module intent/return')
        values[i]=z
    require(len(r['rows'])==len(p['rows'])==4,'all fixed four denominator')
    out=deepcopy(p['rows'])
    for row in out:
        for cp in row['checkpoints']:
            idx=cp['image'];cp.update(latent=r['images'][idx]['latent'],status='encoded',Q=float(np.mean((values[idx].astype(np.float64)-values[row['goal_image']].astype(np.float64))**2)))
    require(r['rows']==out and decode(blob('metrics.json'))==dict(encoding_seal_sha256=r['encoding_seal_sha256'],rows=out),'saved Q exact post-seal arithmetic')
    return dict(schema=NAME+'.consume',rows=out,counts=r['counts'],P1_admission=False,new_independent_samples=0)


def consume(output,expected):
    arrays.hash_value(expected);output=Path(output).absolute();require(sha(output/'report.json')==expected,'external RGB report SHA')
    require(sum(f.stat().st_size for f in output.iterdir() if f.is_file())<=CAP,'whole output cap')
    r=read(output/'report.json');binding=read(output/'execution_binding.json');p=verify(binding['prepared'],binding['prepare_sha256'])
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'complete artifact population')
    def blob(name):
        f=flat(output,name);require(f.stat().st_size<=CAP,'bounded saved artifact');return f.read_bytes()
    out=consume_report(r,p,blob)
    for path,h in p['bindings'].items():require(sha(path)==h,'final bound source/model/physical SHA')
    require(sha(output/'report.json')==expected,'report TOCTOU');return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('physical-report','physical-sha','physical-prepare-sha','physical-source-sha','prepared','prepare-sha','output','report-sha','authorization'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output,physical_report=a.physical_report,physical_sha=a.physical_sha,physical_prepare_sha=a.physical_prepare_sha,physical_source_sha=a.physical_source_sha)
    elif a.command=='verify':r=dict(images=len(verify(a.prepared,a.prepare_sha)['images']))
    elif a.command=='consume':r=consume(a.output,a.report_sha)
    else:r=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    print(canonical(r).decode().strip())
    if a.command=='run' and r['status']!='encoded_sealed':raise SystemExit(2)


if __name__=='__main__':main()
