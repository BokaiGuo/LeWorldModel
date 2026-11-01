"""Fixed two beam witnesses -> saved RGB bridge and MAIN-only CUDA encoding.

No planner replay, physics, training, Q fit or admission. Import/build/consume
use saved metadata/NumPy only. Shared helpers are sealed, never monkeypatched.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import agent_kinodynamic_matched_physical_consume_20261028_v1 as saved
from experiments.spectral_reachability import moving_prediction_matched_sampling_rgb_cuda_20261027_v2 as encoding
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving

NAME='agent_kinodynamic_matched_rgb_20261028_v1'
ROOT,physical,producer=saved.ROOT,saved.physical,saved.physical.producer
arrays=encoding.arrays
require,sha,read,canonical,digest=arrays.require,arrays.sha,arrays.read,arrays.canonical,arrays.digest
ARM=producer.ARM
AUTH='MAIN_MATCHED_DOMAIN_BEAM_RGB_CUDA_ONLY'
REPORT_SHA='01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7'
PREPARE_SHA='2430db7624f3d24ba80af78313b32ac2be168d45383a484f3aed91d5270fab43'
HELPERS={str(Path(saved.__file__).resolve()):'92f31a292a76a88e0e93bd908ac377f2430bb760b8e5b972758717d570f3241b',
         str(Path(encoding.__file__).resolve()):'dde4a59a92583161e488c906aaffe64f5f1e33c3be6bfcae9c46bd8172c83acc',
         str(Path(arrays.__file__).resolve()):'70c03508d775b8b488c12133da929bae607d08d6163d5e1c6578edd9a2422bcc'}
CAP,FLOOR,FINAL,WALL=32*1024**2,200*1024**2,2*1024**2,600


def contract(images=8):
    require(type(images) is int and 4<=images<=8,'actual source-provenance dedup population')
    return dict(roots=2,arms=[ARM],fixed_branch_denominator=2,checkpoints=[0,5,10],images=images,
        module_caps={k:dict(API=images,samples=images,frames=images) for k in ('encoder','projector')},
        dedup='same role + same source path/file SHA/pixel SHA; checkpoints unique by root/selection/step; no arbitrary pixel dedup',
        batch=1,frames_per_encode=1,all_encodings_sealed_before_Q=True,
        metric='float64 mean192 identity; original task goal AND freshly encoded goal separately',
        inherited_encoding_once=producer.contract()['inherited_encoding_once'],
        inherited_planning=dict(predict_API=288,predictor_samples=288,action_encoder=288,pred_proj=288),
        new_predict=0,new_action_encoder=0,new_pred_projector=0,new_AD=0,new_physics=0,
        training=False,Qfit=False,selection=False,retries=0,CUDA_only=True,MATH=True,TF32=False,autocast=False,
        output_cap_bytes=CAP,floor_bytes=FLOOR,final_reserve_bytes=FINAL,wall_seconds=WALL,
        wall_scope='Budget starts before verify/load; checked dispatch/write/finalize, not preemptive timeout',
        scientific_admission=False,P1_admission=False,independent_confirmation=False,new_independent_samples=0,equal_FLOPs=False)


def latent(value):
    a=np.asarray(value)
    require(a.shape==(192,) and a.dtype.kind in 'fi' and np.isfinite(a).all() and
        np.array_equal(a,a.astype(np.float32)),'literal finite native192 float32')
    return a.astype(np.float32)


def assemble(own,tasks,results,goals):
    """Pure, fixed TWO identity joins after strict physical saved consumption."""
    r,p=own['report'],own['prepared']
    require(own['physical_report_sha256']==REPORT_SHA and own['checkpoint_complete'] is True and
        own['fixed_branch_denominator']==2 and r['errors']==[] and r['full_horizons']==2 and
        r['status']=='matched_domain_beam_physical_recorded_not_admitted' and
        r['prepare_sha256']==PREPARE_SHA and r['planner_report_sha256']==physical.REPORT_SHA,'complete actual fixed physical cohort')
    require(len(tasks)==len(results)==len(goals)==len(r['rows'])==len(p['plans'])==2,'never reduce denominator')
    images=[];seen={};rows=[]
    def image(item,key):
        if key not in seen:
            seen[key]=len(images);images.append(dict(deepcopy(item),index=len(images),aliases=[]))
        i=seen[key];images[i]['aliases'].append(dict(root_slot=item['root_slot'],role=item['role']))
        return i
    for s,(row,plan,t,res,g) in enumerate(zip(r['rows'],p['plans'],tasks,results,goals,strict=True)):
        require(type(row['root_slot']) is int and row['root_slot']==plan['root_slot']==s and
            row['arm']==plan['arm']==ARM and row['selected_plan']==plan and row['initial_exact'] is True and
            row['full_horizon'] is True and row['own_returned']==row['warmup_returned']==10 and
            row['first_error'] is None and row['secondary_errors']==[],'own actual full10, never other arm')
        require(t['source_slot']==s and t['task_export']==plan['task_export'] and t['task_sha256']==plan['task_sha256'] and
            t['task_id']==plan['task_id'] and t['blind_sha256']==plan['blind_sha256'] and
            t['task_contract']['normalization']==moving.normalization() and
            t['task_contract']['model_bindings']==arrays.MODELS and t['task_contract']['with_target'] is False,'same moving task/model/normalizer')
        require(res['source_slot']==s and res['arm']==ARM and res['commands']==plan['commands'] and
            res['selected_path']==plan['selected_path'] and res['task_id']==t['task_id'] and
            res['whole_contexts']==res['segmented_contexts'] and len(res['whole_contexts'])==2 and
            res['prediction']==res['whole_contexts'][-1] and res['prediction']==t['prediction'],'whole/segmented BOTH actual sealed contexts')
        for context in res['whole_contexts']+res['segmented_contexts']:producer.checked_context(context)
        goal=latent(t['task']['goal']);root=latent(t['task']['root_context']['latent_history'][-1])
        require(res['model_terminal_cost']==float(np.mean((latent(res['prediction']['latent_history'][-1]).astype(float)-goal.astype(float))**2)),
            'saved model terminal Q arithmetic')
        tc=t['task_contract'];require(g['kind']=='goal' and g['root_slot']==s and
            g['file_sha256']==t['blind_sha256'] and g['pixel_sha256']==tc['pixel_sha256'][3],'actual task goal provenance')
        gi=image(dict(g,role='goal'),('goal',g['path'],g['file_sha256'],g['pixel_sha256']))
        cps=row['checkpoints'];require([c['step'] for c in cps]==[0,5,10],'missing physical checkpoint: no substitute')
        out=dict(root_slot=s,arm=ARM,selection_digest=plan['selection_digest'],task_id=t['task_id'],
            root_sha256=plan['root_sha256'],commands=deepcopy(plan['commands']),goal_image=gi,checkpoints=[],
            initial_metrics=deepcopy(row['initial_metrics']),final_metrics=deepcopy(row['final_metrics']),
            native_success=row['native_success'],physical_status=row['status'])
        for c in cps:
            step=c['step'];require(c['phase']=='plan' and c['state']==c['observation_state']==c['snapshot']['state'],'actual checkpoint state/snapshot')
            if step==0:require(c['pixel_sha256']==tc['pixel_sha256'][2] and c['state']==row['root_state'],'actual original moving root RGB')
            if step==10:require(c['state']==row['final_state'],'actual terminal state')
            item=dict(kind='checkpoint',role='root' if step==0 else 'prefix5' if step==5 else 'terminal10',root_slot=s,
                arm=ARM,step=step,selection_digest=plan['selection_digest'],path=c['png'],
                file_sha256=c['file_sha256'],pixel_sha256=c['pixel_sha256'],physical=deepcopy(c))
            key=('checkpoint',s,plan['selection_digest'],step)
            idx=image(item,key);out['checkpoints'].append(dict(step=step,image=idx,status='not_started',latent=None,Q=None))
        t.update(original_root=root.tolist(),original_goal=goal.tolist(),whole_contexts=deepcopy(res['whole_contexts']),
            segmented_contexts=deepcopy(res['segmented_contexts']),model_terminal_cost=res['model_terminal_cost'])
        rows.append(out)
    return dict(rows=rows,images=images,tasks=tasks,contract=contract(len(images)),
        physical_resources=deepcopy(r['counts']),physical_status=r['status'],physical_errors=deepcopy(r['errors']))


def build(physical_report,physical_sha):
    require(physical_sha==REPORT_SHA,'only fixed actual beam physical report')
    rp=Path(physical_report).absolute();require(sha(rp)==physical_sha,'external physical SHA')
    for f,h in HELPERS.items():require(sha(f)==h,'sealed shared saved/encoding helpers')
    own=saved.consume(rp.parent,physical_sha)  # Saved journal/44PNG/native7/counters; never physics.
    p=own['prepared'];m=read(p['manifest']);pack=read(m['sourcepack'])
    plans,bindings=physical.ownconsume(m['authorities'][0])  # Saved native transitions, no new inference.
    bindings=deepcopy(bindings);physical.merge(bindings,own['bindings_sha256'])
    tasks=[];results=[];goals=[]
    for s,plan in enumerate(plans):
        interface=moving.load_task(plan['task_export'],plan['task_sha256']);e=moving.read(plan['task_export']);t=interface['gn_task']
        require(e['source_slot']==plan['source_slot']==s and e['task_id']==plan['task_id'],'original source slot/task loader')
        prediction=plan['prediction']
        tasks.append(dict(source_slot=s,task_export=plan['task_export'],task_sha256=plan['task_sha256'],task_id=plan['task_id'],
            blind_sha256=plan['blind_sha256'],task_contract=deepcopy(e['task_contract']),task=deepcopy(t),prediction=deepcopy(prediction)))
        result=producer.read(Path(m['authorities'][0]['output'])/f'root_{s}.result.json.gz');results.append(result)
        row=pack['rows'][s];npz=arrays.flat(Path(m['sourcepack']).parent,row['blind_payload'])
        require(sha(npz)==row['blind_payload_sha256']==plan['blind_sha256'],'actual goal blind source')
        tc=e['task_contract'];goals.append(dict(kind='goal',root_slot=s,path=str(npz),file_sha256=plan['blind_sha256'],
            pixel_sha256=tc['pixel_sha256'][3],packet_metadata=dict(rgb_history_pixel_sha256=tc['pixel_sha256'][:3],raw_action_sha256=tc['raw_action_sha256'])))
    out=assemble(own,tasks,results,goals)
    def bind(path,h=None):
        path=str(Path(path).absolute());actual=sha(path)
        require((h is None or h==actual) and (path not in bindings or bindings[path]==actual),'nonconflicting source SHA '+path)
        bindings[path]=actual
    for f,h in arrays.MODELS.items():bind(f,h)
    from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as native
    for f in native.source_files()+moving.source_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bind(f)
    for f,h in HELPERS.items():bind(f,h)
    for im in out['images']:
        if im['kind']=='checkpoint':im['path']=str(arrays.flat(rp.parent,im['path']))
        bind(im['path'],im['file_sha256']);arrays.load_image(im)
    out.update(schema=NAME+'.prepare',arguments=dict(physical_report=str(rp),physical_sha=physical_sha),bindings=bindings,
        policy=str(arrays.POLICY),shapes=arrays.config_shapes(read(arrays.POLICY/'config.json')),expected_model_hash=arrays.MODEL_TENSOR_SHA)
    for f,h in bindings.items():require(sha(f)==h,'final input/source/model SHA')
    return out


def prepare(output,physical_report,physical_sha):
    p=build(physical_report,physical_sha);raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<=2*1024**2 and not path.exists() and not path.is_symlink() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'NEW <=2MiB prepare/200MiB floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'full prepare write');f.flush();os.fsync(f.fileno())
    arrays.syncdir(path.parent);return dict(prepared=str(path),prepare_sha256=sha(path),images=len(p['images']),bytes=len(raw))


def verify(prepared,expected):
    arrays.hash_value(expected);require(sha(prepared)==expected,'external RGB prepare SHA');p=read(prepared)
    require(p==build(**p['arguments']) and sha(prepared)==expected,'entire reconstructed source/model/task/physical closure');return p


class Budget(arrays.Budget):
    def start_gate(self):
        self.check();require(self.free()>=FLOOR+CAP,'full32MiB allocation plus200MiB reserve before CUDA')
    def check(self,size=0,final=False):
        require(final or self.clock()-self.started<=WALL,'600s bounded-check wall scope')
        require(self.used+size<=CAP-(0 if final else FINAL) and self.free()>=FLOOR+size,'32MiB total/200MiB floor')


class Store(encoding.Store):
    def write(self,name,raw,final=False):
        require(name not in self.hashes,'exclusive append-only artifact')
        return super().write(name,raw,final)


def census(entries):
    return {k:{axis:dict(attempted=sum(e['resource']==k for e in entries),
        completed=sum(e['resource']==k and e['completed'] for e in entries)) for axis in ('API','samples','frames')} for k in ('encoder','projector')}


def score_rows(p,images,values):
    rows=deepcopy(p['rows'])
    for row,t in zip(rows,p['tasks'],strict=True):
        target=latent(t['original_goal']).astype(np.float64);fresh=values[row['goal_image']].astype(np.float64)
        row['goal_reencode']=dict(bitwise=values[row['goal_image']].tobytes()==latent(t['original_goal']).tobytes(),maxabs=float(np.max(np.abs(fresh-target))))
        for cp in row['checkpoints']:
            idx=cp['image'];actual=values[idx].astype(np.float64)
            pred=latent(t['original_root'] if cp['step']==0 else t['whole_contexts'][cp['step']//5-1]['latent_history'][-1]).astype(np.float64)
            cp.update(status='encoded',latent=images[idx]['latent'],Q=float(np.mean((actual-target)**2)),
                Q_encoded_goal=float(np.mean((actual-fresh)**2)),model_Q=float(np.mean((pred-target)**2)),
                prediction_MSE=float(np.mean((pred-actual)**2)),prediction_bitwise=pred.astype(np.float32).tobytes()==values[idx].tobytes(),
                prediction_maxabs=float(np.max(np.abs(pred-actual))))
    return rows


def evaluate(p,store,encode,loader,finalize,entries):
    images=[dict(deepcopy(x),status='not_started',latent=None,modules=[]) for x in p['images']]
    rows=deepcopy(p['rows']);values={};errors=[];meta={};seal_sha=None
    def error(stage,exc):errors.append(dict(stage=stage,type=type(exc).__name__,message=str(exc)[:1500]))
    try:
        store.put('tasks.json',p['tasks'])
        for im in images:
            store.check();idx=im['index'];im['status']='attempted';pixels,original=loader(im)
            require(pixels.dtype==np.uint8 and pixels.shape==(224,224,3) and digest(pixels.tobytes())==im['pixel_sha256'] and digest(original)==im['file_sha256'],'literal RGB byte/pixel bridge')
            fn=f'i{idx:02d}_original.'+('npz' if im['kind']=='goal' else 'png')
            im['original']=dict(file=fn,sha256=store.write(fn,original));im['rgb']=store.array(f'i{idx:02d}_rgb',pixels)
            z,receipts=encode(pixels.copy(),idx);z=latent(z)
            arrays.validate_chain(receipts,z,p['shapes'],store.blob)
            require(all(e['image']==idx for e in receipts),'module image identity')
            im.update(status='encoded',latent=store.array(f'i{idx:02d}_latent',z),modules=deepcopy(receipts));values[idx]=z.copy()
    except BaseException as exc:
        error('encoding',exc)
        for im in images:
            if im['status']=='attempted':im['status']='failed'
    try:meta=finalize()
    except BaseException as exc:error('finalize',exc)
    actual=deepcopy(entries);counts=census(actual);n=len(images)
    good=not errors and all(x['status']=='encoded' for x in images) and meta.get('source_verified') is True and meta.get('cuda_verified') is True and \
        meta.get('model_hash_before')==meta.get('model_hash_after')==p['expected_model_hash'] and meta.get('blocked')==[] and \
        meta.get('MATH') is True and meta.get('TF32') is False and meta.get('autocast') is False and \
        counts=={k:{a:dict(attempted=n,completed=n) for a in ('API','samples','frames')} for k in ('encoder','projector')}
    if not good and not errors:error('gate',ValueError('incomplete encoding/model/authority/counts'))
    try:
        seal_sha=store.put('encoding_seal.json',dict(schema=NAME+'.encoding_seal',images=images,entries=actual,counts=counts,meta=meta,
            errors=deepcopy(errors),input_binding=p['arguments'],contract=p['contract'],shapes=p['shapes']),final=True)
    except BaseException as exc:good=False;error('seal',exc)
    if good:
        try:
            store.check();rows=score_rows(p,images,values)
            store.put('metrics.json',dict(encoding_seal_sha256=seal_sha,rows=rows),final=True)
        except BaseException as exc:good=False;error('metrics',exc)
    if not good:
        rows=deepcopy(p['rows'])
        for row in rows:
            for cp in row['checkpoints']:
                im=images[cp['image']];cp.update(status=im['status'],latent=im['latent'])
    report=dict(schema=NAME+'.report',status='encoded_sealed' if good else 'partial_or_failed',rows=rows,images=images,
        counts=counts,meta=meta,errors=errors,first_error=errors[0] if errors else None,encoding_seal_sha256=seal_sha,
        contract=p['contract'],shapes=p['shapes'],input_binding=p['arguments'],physical_resources=p['physical_resources'],
        physical_status=p['physical_status'],physical_errors=p['physical_errors'],new_predict=0,new_AD=0,new_physics=0,
        training=False,P1_admission=False,artifact_sha256=deepcopy(store.hashes))
    store.put('report.json',report,final=True);return report


def run(prepared,expected,output,authorization=None):
    require(authorization==AUTH,'explicit MAIN HOSTCUDA authorization')
    budget=Budget();p=verify(prepared,expected);budget.free=lambda:shutil.disk_usage(Path(output).absolute().parent).free
    budget.start_gate();store=Store(output,budget)
    store.put('execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    stack=ExitStack();trace=None;model=None;before=None;host=None;restore=[]
    def finalize():
        stack.close();require(not restore,'module guard restoration')
        budget.check();require(model is not None,'model unavailable')
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
        native.frozen(model);before=native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen tensor SHA')
        trace=encoding.Trace(torch,p['shapes'],store,store.check,len(p['images']));transform=p0.image_transform()
        stack.enter_context(native.guard(model,torch,trace,restore));stack.enter_context(torch.inference_mode());stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        def encode(pixels,index):
            trace.image=index;x=p0._image_tensor(pixels,transform=transform,device=device)
            require(x.device.type=='cuda' and x.dtype==torch.float32 and tuple(x.shape)==(1,1,3,224,224),'native CUDA transform')
            trace.preprocessed=trace.save(x,f'i{index:02d}_preprocessed')
            z=model.encode({'pixels':x})['emb'];torch.cuda.synchronize(device)
            require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native CUDA projected CLS192')
            return z[0,0].detach().cpu().numpy().copy(),[deepcopy(e) for e in trace.entries if e['image']==index]
        return evaluate(p,store,encode,arrays.load_image,finalize,trace.entries)
    except BaseException as exc:
        try:store.put('run_error.json',dict(type=type(exc).__name__,message=str(exc)),final=True)
        except BaseException as secondary:exc.add_note('run_error persistence: '+repr(secondary))
        if 'report.json' not in store.hashes and 'encoding_seal.json' not in store.hashes:
            def failed(item):raise ValueError('initialization failed: '+str(exc))
            try:evaluate(p,store,None,failed,finalize,[] if trace is None else trace.entries)
            except BaseException as secondary:exc.add_note('partial report persistence: '+repr(secondary))
        raise
    finally:stack.close()


def consume_report(r,p,blob):
    """Independent saved arithmetic: does NOT call evaluator/score_rows."""
    require(r['schema']==NAME+'.report' and r['status']=='encoded_sealed' and r['errors']==[] and r['first_error'] is None and
        r['contract']==p['contract']==contract(len(p['images'])) and r['shapes']==p['shapes'] and
        r['input_binding']==p['arguments'] and r['P1_admission'] is r['training'] is False and
        r['new_predict']==r['new_AD']==r['new_physics']==0,'successful own encoding-only fixed report')
    meta=r['meta'];require(meta['source_verified'] is meta['cuda_verified'] is meta['MATH'] is True and
        meta['TF32'] is meta['autocast'] is False and meta['blocked']==[] and
        meta['model_hash_before']==meta['model_hash_after']==p['expected_model_hash'],'actual frozen CUDA receipts')
    require(len(r['images'])==len(p['images'])==p['contract']['images'],'all image denominator')
    names={'execution_binding.json','tasks.json','encoding_seal.json','metrics.json'};values={};entries=[]
    for i,(im,expected) in enumerate(zip(r['images'],p['images'],strict=True)):
        require(all(im[k]==v for k,v in expected.items()) and im['index']==i and im['status']=='encoded','prepared exact image identity')
        original=blob(im['original']['file']);require(digest(original)==im['original']['sha256']==im['file_sha256'],'exact original source bytes')
        pixels=arrays.load_array(im['rgb'],blob);z=arrays.load_array(im['latent'],blob);latent(z)
        require(pixels.shape==(224,224,3) and pixels.dtype==np.uint8 and digest(pixels.tobytes())==im['pixel_sha256'],'actual RGB pixel SHA')
        if im['kind']=='goal':require(np.array_equal(arrays.packet(original)['goal_rgb'],pixels),'saved blind goal pixels')
        else:
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(original)) as image:require(image.format=='PNG' and image.mode=='RGB' and np.array_equal(np.asarray(image),pixels),'saved literal PNG pixels')
        arrays.validate_chain(im['modules'],z,p['shapes'],blob)
        names.update((im['original']['file'],im['rgb']['file'],im['latent']['file']))
        for ordinal,e in enumerate(im['modules'],start=2*i+1):
            require(e['image']==i and arrays.decode(blob(f'module_{ordinal:02d}_returned.json'))==e and
                arrays.decode(blob(f'module_{ordinal:02d}_intent.json'))=={k:v for k,v in dict(e,completed=False).items() if k!='output'},'durable actual module dispatch/return')
            names.update((f'module_{ordinal:02d}_intent.json',f'module_{ordinal:02d}_returned.json',e['output']['file']))
            names.update(x['file'] for x in e['inputs']);names.add(e['preprocessed']['file']) if 'preprocessed' in e else None
            entries.append(e)
        values[i]=z.astype(np.float64)
    n=len(p['images']);counts={k:{axis:dict(attempted=n,completed=n) for axis in ('API','samples','frames')} for k in ('encoder','projector')}
    require(r['counts']==counts and len(entries)==2*n,'independent module/sample/frame census')
    seal=arrays.decode(blob('encoding_seal.json'))
    require(digest(blob('encoding_seal.json'))==r['encoding_seal_sha256'] and seal==dict(schema=NAME+'.encoding_seal',images=r['images'],
        entries=entries,counts=counts,meta=meta,errors=[],input_binding=p['arguments'],contract=p['contract'],shapes=p['shapes']),'full encoding seal')
    require(arrays.decode(blob('tasks.json'))==p['tasks'],'complete original task/normalizer/whole+segmented contexts')
    out=deepcopy(p['rows']);require(len(out)==len(r['rows'])==2,'all fixed TWO branches')
    for row,t in zip(out,p['tasks'],strict=True):
        target=np.asarray(t['original_goal'],np.float64);fresh=values[row['goal_image']]
        row['goal_reencode']=dict(bitwise=fresh.astype(np.float32).tobytes()==np.asarray(t['original_goal'],np.float32).tobytes(),maxabs=float(np.max(np.abs(fresh-target))))
        for cp in row['checkpoints']:
            idx=cp['image'];z=values[idx]
            prediction=np.asarray(t['original_root'] if cp['step']==0 else t['whole_contexts'][cp['step']//5-1]['latent_history'][-1],np.float64)
            cp.update(status='encoded',latent=r['images'][idx]['latent'],Q=float(np.sum((z-target)**2)/192),
                Q_encoded_goal=float(np.sum((z-fresh)**2)/192),model_Q=float(np.sum((prediction-target)**2)/192),
                prediction_MSE=float(np.sum((prediction-z)**2)/192),prediction_bitwise=prediction.astype(np.float32).tobytes()==z.astype(np.float32).tobytes(),
                prediction_maxabs=float(np.max(np.abs(prediction-z))))
    require(out==r['rows'] and arrays.decode(blob('metrics.json'))==dict(encoding_seal_sha256=r['encoding_seal_sha256'],rows=out),'independent saved mean192 Q/model residuals')
    require(set(r['artifact_sha256'])==names,'exact full artifact population')
    for name,h in r['artifact_sha256'].items():require(digest(blob(name))==h,'every saved artifact SHA')
    require(r['physical_resources']==p['physical_resources'] and r['physical_status']==p['physical_status'] and r['physical_errors']==p['physical_errors'],'inherited actual physical receipts')
    return dict(schema=NAME+'.consume',rows=out,counts=counts,scientific_admission=False,P1_admission=False,new_independent_samples=0)


def consume(output,expected):
    arrays.hash_value(expected);output=Path(output).absolute();require(sha(output/'report.json')==expected,'external RGB report SHA')
    files=list(output.iterdir());require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files)<=CAP,'regular bounded output')
    r=read(output/'report.json');b=read(output/'execution_binding.json');p=verify(b['prepared'],b['prepare_sha256'])
    require({f.name for f in files}==set(r['artifact_sha256'])|{'report.json'},'exact directory census')
    def blob(name):
        f=arrays.flat(output,name);require(f.stat().st_size<=CAP,'bounded artifact');return f.read_bytes()
    result=consume_report(r,p,blob)
    for f,h in p['bindings'].items():require(sha(f)==h,'final external source/model/task SHA')
    require(sha(output/'report.json')==expected,'report TOCTOU');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('physical-report','physical-sha','prepared','prepare-sha','output','report-sha','authorization'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output,a.physical_report,a.physical_sha)
    elif a.command=='verify':r=dict(images=len(verify(a.prepared,a.prepare_sha)['images']))
    elif a.command=='consume':r=consume(a.output,a.report_sha)
    else:r=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    print(canonical(r).decode().strip(),flush=True)
    if a.command=='run' and r['status']!='encoded_sealed':raise SystemExit(2)


if __name__=='__main__':main()
