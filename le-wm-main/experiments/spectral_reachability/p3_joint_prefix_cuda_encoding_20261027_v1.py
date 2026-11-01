"""Seal 17 real saved P3 prefixes; explicit MAIN-only CUDA encoding, no physics.

prepare/verify/consume use only saved bytes, PIL and NumPy. No model import.
No old executor/exporter is imported, run or edited. No source outcome tuning.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sysconfig
import numpy as np
from PIL import Image
from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3_joint_prefix_cuda_encoding_20261027_v1'
REPO=Path(__file__).resolve().parents[2]
PHYSICAL=REPO/'results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01'
PHYSICAL_SHA='6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a'
PHYSICAL_PREPARE_SHA='1cfc5eb2ddcfc563176a3cce51a0ef90b88e92d298b3cf35d415c2e420317daf'
PLANNER=REPO/'results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01'
PLANNER_SHA='23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950'
PLANNER_PREPARE_SHA='7886406284604cf17ba97671030ef32fde9d32f6f4c18b8293db0c2129654a47'
METRICS=('old_Q','goal_patch_Q')
ARMS=tuple(q+'__'+s for q in METRICS for s in ('uniform','dynamic'))
META=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10)
RESERVE=512*1024
OUTPUT_CAP=4_900_000
DISK_RESERVE=200*1024*1024
MIN_FREE=DISK_RESERVE+OUTPUT_CAP
require=io.require
def digest(v):return io.sha(io.canonical(v).encode())
def same(a,b):return io.canonical(a)==io.canonical(b)


def regular(path,limit=None):
    p=Path(path).absolute()
    require('..' not in p.parts and not any(x.is_symlink() for x in (p,*p.parents)) and p.is_file(),'regular nonsymlink path')
    if limit is not None:require(p.stat().st_size<=limit,'bounded file')
    return p


def sha(path):
    p=regular(path);h=hashlib.sha256()
    with p.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def read(path):return io.decode(regular(path,32_000_000).read_bytes())


def rawjson(v):
    raw=(io.canonical(v)+'\n').encode();require(len(raw)<=RESERVE,'small JSON cap');return raw


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write(path,value):
    path=Path(path)
    with path.open('xb') as stream:stream.write(rawjson(value));stream.flush();os.fsync(stream.fileno())
    fsync_directory(path.parent)


def error(stage,exc,**extra):return dict(stage=stage,type=type(exc).__name__,message=str(exc)[:1000],**extra)


def pixels(path,expected_file=None,expected_pixel=None):
    p=regular(path,1024*1024)
    if expected_file is not None:require(sha(p)==expected_file,'PNG file SHA')
    with Image.open(p) as im:
        require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'original canonical PNG')
        rgb=np.asarray(im).copy()
    require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8,'uint8 saved RGB')
    pixel=io.sha(rgb.tobytes())
    if expected_pixel is not None:require(pixel==expected_pixel,'RGB pixel SHA')
    return rgb,pixel


def array(v,shape):
    a=np.asarray(v);require(a.dtype.kind in 'fi' and a.shape==shape and np.isfinite(a).all(),'finite real array')
    return a.astype(float)


def context(c):
    require(all(type(c.get(k)) is type(v) and c[k]==v for k,v in META.items()),'native full3 metadata')
    return array(c['latent_history'],(3,192)),array(c['action_history'],(2,10))


def history(before,after,actions,norm):
    lat,hist=context(before);_,actual=context(after);raw=np.asarray(actions)
    require(raw.shape in ((5,2),(10,2),(15,2)) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'raw time-major full packets')
    require(same({k:v for k,v in before.items() if k not in ('latent_history','action_history')},
        {k:v for k,v in after.items() if k not in ('latent_history','action_history')}),'context metadata retained')
    mean,std=(np.asarray(norm[k],np.float32) for k in ('mean','std'))
    require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>0),'float32 normalizer')
    tokens=((raw.astype(np.float32)-mean)/std).reshape(-1,10)
    expected=np.concatenate((hist.astype(np.float32),tokens),axis=0)[-2:]
    require(np.array_equal(actual,expected.astype(float)),'exact normalized token content/order')
    h=len(tokens)
    if h<3:require(np.array_equal(np.asarray(after['latent_history'])[:3-h],lat.astype(np.float32)[h:].astype(float)),'retained latent chronology')


def prefix_vectors(plan,source,record,norm):
    """Derive final-context offsets from rolling replay, never an assumed index."""
    winner=source['winner'];h=plan['planned_horizon_tokens'];before=record['packet']['root_context']
    require(type(h) is int and h in (2,3) and source['status']=='completed' and
        winner['completed'] is True and winner['horizon']==h and len(source['segments'])==h,'complete H2/H3 witness')
    require(winner==min(source['candidates'],key=lambda x:(x['cost'],x['horizon'],x['index'])),'actual stable selected candidate')
    require(same(winner['actions'],plan['commands']) and same(before,plan['predictor_context']) and
        same(winner['context'],plan['selected_predicted_context']) and same(winner['context'],source['whole_replay_context']) and
        same(winner['context'],source['segmented_replay_context']) and source['whole_and_segmented_exact'] is True,'full-context winner/whole/segment binding')
    actions=array(plan['commands'],(5*h,2));generated=[]
    for i,after in enumerate(source['segments']):
        history(before,after,actions[5*i:5*i+5],norm);generated.append(after['latent_history'][-1]);before=after
    require(same(before,winner['context']),'last segmented context identity')
    history(record['packet']['root_context'],winner['context'],actions,norm)
    expected=(record['packet']['root_context']['latent_history']+generated)[-3:]
    require(same(expected,winner['context']['latent_history']),'all retained predicted state chronology')
    out={}
    for k in range(1,h):
        index=3-h+k-1
        require(same(generated[k-1],winner['context']['latent_history'][index]),'prefix is selected path state, not terminal/nominal')
        out[5*k]=dict(predicted_history_index=index,predicted_vector=generated[k-1],
            segmented_context_sha256=digest(source['segments'][k-1]))
    return out


def contract():
    return dict(roots=4,branches=16,frames=17,all_branch_step5=16,sole_H3_step10=1,
        selection='all frozen physical plan order, ascending interior packet boundaries; no outcome filter',
        latent_dim=192,batch=1,time=1,encoder_cap=17,existing_projector_cap=17,per_frame_each_cap=1,
        predict_cap=0,AD_cap=0,physics_cap=0,training=False,device='cuda:0',cpu_neural_fallback=False,
        representation='same model.encode pixels -> native encoder CLS -> existing projector -> native192',
        normalization='original p0._image_tensor + public image_transform; frozen source-bound',
        retries=False,no_refit=True,output_bytes_cap=OUTPUT_CAP,remaining_disk_reserve_bytes=DISK_RESERVE,large_input_copies=False,
        P1_admission=False,P3_admission=False,efficacy_claim=False)


def prepare_read(e,path,expected):
    receipt=e.json(path/'prepare.json',expected);p=e.json(path/'protocol.json',receipt['protocol_sha256'])
    require(receipt['protocol_id']==digest(p),'original protocol identity')
    for seal in receipt['source_seals']:e.seal(seal)
    return receipt,p


def collect():
    """All actual prerequisites, read-only; no imports of old runtime/exporter."""
    e=io.Evidence(REPO);r=e.json(PHYSICAL/'report.json',PHYSICAL_SHA);c=e.json(PLANNER/'report.json',PLANNER_SHA)
    binding=e.json(PHYSICAL/'execution_protocol.json');cb=e.json(PLANNER/'execution_protocol.json')
    require(binding['prepare_sha256']==PHYSICAL_PREPARE_SHA and binding['planner_report_sha256']==PLANNER_SHA and
        binding['encode_terminals'] is True and cb['prepare_sha256']==PLANNER_PREPARE_SHA,'original launch bindings')
    _,p=prepare_read(e,Path(binding['prepared']),PHYSICAL_PREPARE_SHA)
    _,cp=prepare_read(e,Path(cb['prepared']),PLANNER_PREPARE_SHA)
    require(r['status']=='guarded_calibration_recorded' and r['failures']==[] and r['model_unchanged'] is True and
        r['source_verified'] is True and r['full_horizon_all_branches'] is True and
        r['consumer_prepare_sha256']==PHYSICAL_PREPARE_SHA and r['planner_report_sha256']==p['planner_report_sha256']==PLANNER_SHA,'successful actual physical source')
    require(c['status']=='cuda_completed' and c['errors']==[] and c['model_unchanged'] is True and
        c['protocol_id']==digest(cp) and c['contract']==cp['contract'],'successful actual planner source')
    require(same(p['model_bindings'],cp['model_bindings']),'same original frozen model')
    require(all(r[k] is c[k] is False for k in ('P1_admission','P2_admission','P3_admission','P4_admission')),'no admission upgrade')
    rows=io.index(r['rows'],('root_sha256','method'));plans=io.index(p['plans'],('root_sha256','method'))
    roots=io.index(c['roots'],('root_sha256',));records={x['packet']['root_sha256']:x for x in cp['records']}
    require(len(records)==len(cp['records'])==4 and set(rows)==set(plans)=={(root,arm) for root in records for arm in ARMS},'all four roots x four arms')
    expected=[(x['packet']['root_sha256'],arm) for i,x in enumerate(cp['records']) for arm in cp['arm_order'][i]]
    require([(x['root_sha256'],x['method']) for x in p['plans']]==expected,'frozen actual plan order')
    require(sum(x['planned_horizon_tokens']==3 for x in p['plans'])==1 and len(p['plans'])==16,'one H3 and fifteen H2 plans')
    for path,h in {**p['sources_sha256'],**cp['sources_sha256']}.items():e.raw(Path(path),h)
    frames=[]
    for plan in p['plans']:
        key=plan['root_sha256'],plan['method'];row=rows[key];record=records[key[0]];source=roots[key[0],]['arms'][key[1]]
        require(plan['selection_sha256']==digest({k:v for k,v in plan.items() if k!='selection_sha256'}),'physical selected plan digest')
        require(all(row[k]==plan[k] for k in ('root_index','root_sha256','method','selection_sha256','planned_horizon_tokens','planned_commands','active_metric_id')),'actual branch selected plan')
        require(row['status']=='completed' and row['failure'] is None and row['failures']==[] and row['endpoint_fresh'] is True and
            row['full_horizon_completed'] is True and row['physical_env_step_attempts']==row['physical_env_step_returns']==plan['planned_commands']==5*plan['planned_horizon_tokens'],'all complete full-path actual branches')
        require(same(record['packet']['root_context'],plan['predictor_context']) and same(record['packet']['goal'],plan['goal']) and
            plan['source_input_sha256']==record['packet']['input_sha256'] and plan['active_metric_id']==record['metrics'][plan['metric_label']]['active_metric_id'],'original context/goal/metric identity')
        require(same(p['root_metrics'][key[0]],dict(goal=record['packet']['goal'],metrics=record['metrics'])),'physical/planner original goal and both Q')
        vectors=prefix_vectors(plan,source,record,cp['normalization'])
        path=PHYSICAL/f'root_{plan["root_index"]}_{plan["method"]}.events.jsonl'
        require(row['journal_path']==str(path),'own original journal path')
        events=[io.decode(line) for line in e.raw(path,row['journal_sha256']).splitlines()]
        require(len(events)<=46,'bounded full H3 journal')
        obs=[v for v in events if v['kind']=='observation'];attempts=[v for v in events if v['kind']=='attempt']
        require(len(obs)==len(attempts)==plan['planned_commands'] and [v['primitive_step'] for v in obs]==list(range(1,plan['planned_commands']+1)) and
            same([v['command'] for v in attempts],plan['commands']) and
            same([{k:v for k,v in t.items() if k not in ('rgb_path','rgb_file_sha256')} for t in obs],row['trace']),'saved selected action/observation chronology')
        for step,witness in vectors.items():
            observation=obs[step-1];rgbpath=PHYSICAL/f'root_{plan["root_index"]}_{plan["method"]}_rgb_{step}.png'
            require(observation['rgb_path']==str(rgbpath) and observation['command']==plan['commands'][step-1],'exact prefix RGB/command')
            e.raw(rgbpath,observation['rgb_file_sha256']);_,pixel=pixels(rgbpath,observation['rgb_file_sha256'])
            frames.append(dict(frame_id=f'root_{plan["root_index"]}_{plan["method"]}_step_{step}',order=len(frames),root_index=plan['root_index'],
                root_sha256=key[0],arm=key[1],metric_label=plan['metric_label'],active_metric_id=plan['active_metric_id'],
                horizon=plan['planned_horizon_tokens'],primitive_step=step,selection_sha256=plan['selection_sha256'],
                source_input_sha256=plan['source_input_sha256'],rgb_path=str(rgbpath),rgb_file_sha256=observation['rgb_file_sha256'],
                pixel_sha256=pixel,actual_state=observation['state'],goal_rgb_sha256=row['context_identity']['goal_rgb_sha256'],
                journal_path=str(path),journal_sha256=row['journal_sha256'],**witness))
    require(len(frames)==17 and len({x['frame_id'] for x in frames})==17 and sum(x['primitive_step']==5 for x in frames)==16,'all 17 prefixes no filtering/deduplication')
    models=p['model_bindings'];require({Path(x).name for x in models}=={'config.json','weights.pt'},'local original checkpoint only')
    for path,h in models.items():require(sha(path)==h,'original checkpoint streaming SHA')
    policy=Path(next(x for x in models if Path(x).name=='weights.pt')).parent
    require(all(Path(x).parent==policy for x in models),'same model directory')
    config=read(policy/'config.json');require(config['_target_']=='stable_worldmodel.wm.lewm.LeWM' and
        config['encoder'].get('pretrained') is False and config['projector']['_target_']=='stable_worldmodel.wm.lewm.module.MLP','local native encoder + existing projector; no checkpoint download')
    e.recheck()
    return dict(schema=NAME+'.protocol',contract=contract(),frames=frames,policy=str(policy),model_bindings=models,
        physical_report=str(PHYSICAL/'report.json'),physical_report_sha256=PHYSICAL_SHA,planner_report=str(PLANNER/'report.json'),planner_report_sha256=PLANNER_SHA,
        goal_q_source=str(Path(binding['prepared'])/'protocol.json'),goal_q_source_sha256=sha(Path(binding['prepared'])/'protocol.json'),
        goal_q_sha256=digest(p['root_metrics']),bindings=dict(sorted(e.hashes.items())),
        original_prepare_sha256=PHYSICAL_PREPARE_SHA,planner_prepare_sha256=PLANNER_PREPARE_SHA,
        inherited_cost='original planner/physical/terminal encoder work not repeated or free; cached acquisition cost unknown')


def source_files():
    site=Path(sysconfig.get_paths()['purelib'])
    relative=['stable_worldmodel/wm/lewm/lewm.py','stable_worldmodel/wm/lewm/module.py','stable_pretraining/backbone/utils.py',
        'stable_pretraining/data/dataset_stats.py','transformers/models/vit/modeling_vit.py','transformers/models/vit/configuration_vit.py',
        'torchvision/transforms/v2/_container.py','torchvision/transforms/v2/_misc.py','torchvision/transforms/v2/_geometry.py',
        'torchvision/transforms/v2/functional/_misc.py','torchvision/transforms/v2/functional/_geometry.py','torchvision/transforms/v2/functional/_type_conversion.py']
    local=[Path(__file__).resolve(),REPO/'tests'/('test_'+NAME+'.py'),REPO/'innerdocs'/(NAME+'.md'),Path(io.__file__).resolve(),
        REPO/'experiments/checkpoint_compat.py',REPO/'experiments/gpu_preflight.py',REPO/'experiments/run_pusht_counterfactual.py',
        REPO/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py',REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    return sorted(set(local+[site/r for r in relative]))


def disk_bytes(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def prepare(output):
    output=Path(output).absolute();require(not output.exists(),'exclusive NEW prepare');require(shutil.disk_usage(output.parent).free>=MIN_FREE,'prepare headroom')
    p=collect();sources=source_files();require(sum(regular(f).stat().st_size for f in sources)<3_500_000,'small source closure')
    p['sources_sha256']={str(x):sha(x) for x in sources};output.mkdir(exist_ok=False);write(output/'protocol.json',p)
    seal=create_seal(files=sources,args_json=io.canonical(dict(protocol_sha256=sha(output/'protocol.json'),contract=contract())),output_dir=output/'source_seal')
    receipt=dict(schema=NAME+'.prepare',status='prepared_not_executed',protocol_sha256=sha(output/'protocol.json'),source_seal=seal,
        encoder_calls=0,projector_calls=0,predict_calls=0,AD_calls=0,physics_steps=0,P1_admission=False,P3_admission=False)
    write(output/'prepare.json',receipt);require(disk_bytes(output)<OUTPUT_CAP,'prepare output below5MB');return receipt


def verify(prepared,expected_sha):
    prepared=Path(prepared).absolute();require(sha(prepared/'prepare.json')==expected_sha,'prepare SHA')
    receipt=read(prepared/'prepare.json');p=read(prepared/'protocol.json')
    require(receipt['schema']==NAME+'.prepare' and receipt['status']=='prepared_not_executed' and sha(prepared/'protocol.json')==receipt['protocol_sha256'] and
        p['schema']==NAME+'.protocol' and p['contract']==contract(),'own frozen contract')
    seal=receipt['source_seal'];verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    for path,h in {**p['bindings'],**p['model_bindings'],**p['sources_sha256']}.items():require(sha(path)==h,'source/input/model drift: '+path)
    rebuilt=collect();rebuilt['sources_sha256']=p['sources_sha256'];require(same(rebuilt,p),'all saved prefixes/model/goalQ/order reconstructed')
    return p


class Ledger:
    """Nonrefundable per-frame1/global17 CUDA module caps; no failed retry."""
    def __init__(self,frames,journal,synchronize=lambda:None):
        self.order=[(f['frame_id'],k) for f in frames for k in ('encoder','existing_projector')]
        self.journal,self.synchronize=journal,synchronize;self.active=None;self.entries=[];self.blocked=[];self.journal_errors=[]

    def emit(self,event):
        try:self.journal(event)
        except BaseException as exc:self.journal_errors.append(error('journal',exc));raise

    def dispatch(self,kind,tensor,callback,args,kwargs):
        try:
            require(kind in ('encoder','existing_projector') and getattr(getattr(tensor,'device',None),'type',None)=='cuda','CUDA encoder/projector only')
            require(tuple(tensor.shape)==((1,3,224,224) if kind=='encoder' else (1,192)),'single native frame/module shape')
            require(len(self.entries)<len(self.order) and (self.active,kind)==self.order[len(self.entries)],'sealed frame/module sequence')
            require(sum(e['kind']==kind for e in self.entries)<17 and all(e['completed'] for e in self.entries) and not self.journal_errors,'nonrefundable cap/no retry')
        except BaseException as exc:
            e=dict(kind=kind,frame_id=self.active,error=str(exc)[:1000]);self.blocked.append(e)
            try:self.emit(dict(event='blocked',**e))
            except BaseException:raise exc
            raise
        e=dict(kind=kind,frame_id=self.active,index=len(self.entries),completed=False);self.entries.append(e);primary=None
        try:
            self.emit(dict(event='attempt',**e));result=callback(*args,**kwargs);self.synchronize();e['completed']=True;return result
        except BaseException as exc:primary=exc;e['error']=error('module',exc);raise
        finally:
            try:self.emit(dict(event='return_or_error',**deepcopy(e)))
            except BaseException:
                if primary is None:raise

    def forbid(self,kind):
        def denied(*args,**kwargs):
            exc=RuntimeError('forbidden '+kind);e=dict(kind=kind,frame_id=self.active,error=str(exc)[:1000]);self.blocked.append(e)
            try:self.emit(dict(event='blocked',**e))
            except BaseException:raise exc
            raise exc
        return denied

    def counts(self):
        return {k:dict(attempted=sum(e['kind']==k for e in self.entries),completed=sum(e['kind']==k and e['completed'] for e in self.entries)) for k in ('encoder','existing_projector')}


class Hooks:
    def __init__(self):self.originals=[]
    def install(self,obj,key,value):
        self.originals.append((obj,key,getattr(obj,key)));setattr(obj,key,value)
    def restore(self,errors):
        for obj,key,value in reversed(self.originals):
            try:setattr(obj,key,value)
            except BaseException as exc:errors.append(error('hook_restore',exc,attribute=key))


def save_native(path,value):
    with Path(path).open('xb') as stream:np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    fsync_directory(Path(path).parent)
    return sha(path)


def encode_frames(p,output,encode,ledger,journal,rows,errors):
    """Persist each NPY/hash/result before next frame; keep all blocked slots."""
    output=Path(output)
    for frame in p['frames']:
        row={k:v for k,v in frame.items() if k!='predicted_vector'};row.update(status='blocked_not_attempted',latent_path=None,latent_sha256=None)
        if not errors:
            ledger.active=frame['frame_id'];row.update(status='attempted',latent_path=str(output/(frame['frame_id']+'.native192.npy')))
            try:
                require(disk_bytes(output)<OUTPUT_CAP-RESERVE and shutil.disk_usage(output).free>=MIN_FREE,'small output/headroom before next encode')
                write(output/(frame['frame_id']+'.attempt.json'),row);journal(dict(event='endpoint_attempt',frame_id=frame['frame_id']))
                native=np.asarray(encode(frame));require(native.shape==(192,) and native.dtype==np.float32 and np.isfinite(native).all(),'native finite float32 latent192')
                row['latent_sha256']=save_native(row['latent_path'],native)
                row.update(status='encoded_and_saved',latent_dtype='float32',latent_shape=[192])
                # No Q arithmetic before the native file and hash are durable.
                write(output/(frame['frame_id']+'.latent.json'),row)
                journal(dict(event='endpoint_saved',frame_id=frame['frame_id'],latent_path=row['latent_path'],latent_sha256=row['latent_sha256']))
            except BaseException as exc:errors.append(error('endpoint',exc,frame_id=frame['frame_id']));row.update(status='failed',error=errors[-1])
            finally:ledger.active=None
        rows.append(row)
        try:write(output/(frame['frame_id']+'.result.json'),row)
        except BaseException as exc:errors.append(error('endpoint_result_write',exc,frame_id=frame['frame_id']))


def tensor_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()


def frozen(model):
    require(all(not m.training for m in model.modules()) and all(v.device.type=='cuda' and not v.requires_grad for v in model.parameters()),'frozen CUDA eval model')


def execute_cuda(prepared,expected_sha,output):
    """Explicit MAIN-only entry; never invoked by prepare/verify/consume."""
    prepared=Path(prepared).absolute();p=verify(prepared,expected_sha);output=Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free>=MIN_FREE,'exclusive derived run/headroom')
    # Separate NEW sibling launch claim: do not mutate the frozen preparation.
    write(prepared.with_name(prepared.name+'.launch.json'),dict(output=str(output),prepare_sha256=expected_sha))
    output.mkdir(exist_ok=False)
    with (output/'receipt.json').open('xb') as stream:os.posix_fallocate(stream.fileno(),0,RESERVE);os.fsync(stream.fileno())
    write(output/'execution_binding.json',dict(prepared=str(prepared),prepare_sha256=expected_sha,physical_report_sha256=PHYSICAL_SHA))
    rows=[];errors=[];model=None;before=after=None;host=None;source_ok=False;hooks=Hooks();complete=False
    with (output/'encoding.events.jsonl').open('x') as stream:
        def journal(v):
            raw=rawjson(v);require(stream.tell()+len(raw)<=RESERVE,'bounded fsync journal')
            stream.write(raw.decode());stream.flush();os.fsync(stream.fileno())
        ledger=Ledger(p['frames'],journal)
        try:
            import torch
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=False
            model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);frozen(model);before=tensor_hash(model)
            transform=p0.image_transform();ledger.synchronize=lambda:torch.cuda.synchronize(device)
            for kind,module in (('encoder',model.encoder),('existing_projector',model.projector)):
                original=module.forward
                def counted(*args,_kind=kind,_original=original,**kwargs):return ledger.dispatch(_kind,args[0],_original,args,kwargs)
                hooks.install(module,'forward',counted)
            for obj,key,kind in ((model,'predict','predict'),(model.predictor,'forward','predictor'),(model.action_encoder,'forward','action_encoder'),
                (model.pred_proj,'forward','pred_proj'),(torch.autograd,'grad','AD.grad'),(torch.autograd,'backward','AD.backward'),(torch.Tensor,'backward','AD.Tensor.backward')):
                hooks.install(obj,key,ledger.forbid(kind))
            for key in ('jacobian','hessian','jvp','vjp','hvp','vhp'):hooks.install(torch.autograd.functional,key,ledger.forbid('AD.'+key))
            for key in ('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value'):hooks.install(torch.func,key,ledger.forbid('AD.func.'+key))
            def encode(frame):
                rgb,_=pixels(frame['rgb_path'],frame['rgb_file_sha256'],frame['pixel_sha256'])
                with torch.inference_mode():
                    x=p0._image_tensor(rgb,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native CUDA input')
                    z=model.encode({'pixels':x})['emb']
                    require(z.device.type=='cuda' and tuple(z.shape)==(1,1,192) and z.dtype==torch.float32 and bool(torch.isfinite(z).all()),'native CUDA192 output')
                    require([(e['kind'],e['completed']) for e in ledger.entries if e['frame_id']==frame['frame_id']]==[('encoder',True),('existing_projector',True)],'exact native encoding entries')
                    return z[0,0].detach().cpu().numpy().copy()
            encode_frames(p,output,encode,ledger,journal,rows,errors);torch.cuda.synchronize(device)
        except BaseException as exc:errors.append(error('setup_or_execution',exc))
        finally:
            hooks.restore(errors)
            if model is not None:
                try:frozen(model);after=tensor_hash(model)
                except BaseException as exc:errors.append(error('final_model_hash',exc))
            try:verify(prepared,expected_sha);source_ok=True
            except BaseException as exc:errors.append(error('final_input_verify',exc))
            for frame in p['frames'][len(rows):]:rows.append(dict(frame_id=frame['frame_id'],status='blocked_not_attempted',latent_path=None,latent_sha256=None))
            model_ok=None if model is None else before is not None and before==after
            if model_ok is not True:errors.append(dict(stage='model_integrity',message='hash unavailable or model changed; not verified'))
            counts=ledger.counts()
            complete=(not errors and not ledger.blocked and not ledger.journal_errors and source_ok and model_ok is True and
                all(x==dict(attempted=17,completed=17) for x in counts.values()) and len(rows)==17 and all(x['status']=='encoded_and_saved' for x in rows))
            if not complete and not errors:errors.append(dict(stage='entry_integrity',message='incomplete module/frame census or forbidden attempt'))
            journal_hash=None
            try:journal_hash=sha(output/'encoding.events.jsonl')
            except BaseException as exc:errors.append(error('final_journal_hash',exc));complete=False
            receipt=dict(schema=NAME+'.derived_receipt',status='all_17_prefix_latents_saved' if complete else 'partial_or_failed',
                prepare_sha256=expected_sha,physical_report_sha256=PHYSICAL_SHA,planner_report_sha256=PLANNER_SHA,goal_q_sha256=p['goal_q_sha256'],
                rows=rows,counts=counts,entries=ledger.entries,blocked=ledger.blocked,journal_errors=ledger.journal_errors,
                first_error=errors[0] if errors else None,errors=errors,journal_sha256=journal_hash,
                model_hash_before=before,model_hash_after=after,model_unchanged=model_ok,model_bindings=p['model_bindings'],source_verified=source_ok,host=host,
                new_predict_calls=0,new_AD_calls=0,new_physical_steps=0,training=False,P1_admission=False,P3_admission=False,efficacy_claim=False,
                denominator=dict(roots=4,branches=16,prefix_frames=17,new_independent_roots=0),
                note='derived encoder samples from existing physical frames; not new rollouts or independent roots')
            # Receipt space reserved before model setup; finalization failures are
            # appended above, never raised from cleanup over the first error.
            raw=rawjson(receipt)
            with (output/'receipt.json').open('r+b') as target:target.write(raw);target.truncate();target.flush();os.fsync(target.fileno())
            fsync_directory(output);require(disk_bytes(output)<OUTPUT_CAP,'derived output below5MB')
    if not complete:raise RuntimeError('prefix encoding incomplete; inspect NEW receipt, no automatic retry')
    return receipt


def numeric(actual,predicted,goal,qs):
    a,p,g=(array(x,(192,)) for x in (actual,predicted,goal));out=dict(actual_Q_costs={},predicted_Q_costs={},actual_minus_predicted_same_Q_cost={},projected_prediction_mse={})
    for label in METRICS:
        q=array(qs[label],(192,192));require(np.allclose(q,q.T,atol=1e-8,rtol=0) and np.allclose(q@q,q,atol=1e-8,rtol=0) and np.trace(q)>.5,'original orthogonal Q')
        out['actual_Q_costs'][label]=float(np.mean((q@(a-g))**2));out['predicted_Q_costs'][label]=float(np.mean((q@(p-g))**2))
        out['actual_minus_predicted_same_Q_cost'][label]=out['actual_Q_costs'][label]-out['predicted_Q_costs'][label]
        out['projected_prediction_mse'][label]=float(np.mean((q@(a-p))**2))
    out['native_prediction_mse']=float(np.mean((a-p)**2));return out


def consume(folder,expected_sha,output):
    """Separate saved-only numerical consumer; never loads an encoder/model."""
    folder=Path(folder).absolute();output=Path(output).absolute();require(not output.exists(),'exclusive NEW numerical output')
    require(sha(folder/'receipt.json')==expected_sha,'derived receipt external SHA');r=read(folder/'receipt.json');binding=read(folder/'execution_binding.json')
    p=verify(binding['prepared'],binding['prepare_sha256'])
    require(r['schema']==NAME+'.derived_receipt' and r['status']=='all_17_prefix_latents_saved' and r['errors']==[] and
        r['first_error'] is None and r['blocked']==r['journal_errors']==[] and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after'] and r['model_hash_before'] is not None and r['source_verified'] is True,'completed derived receipt')
    require(r['physical_report_sha256']==PHYSICAL_SHA and r['planner_report_sha256']==PLANNER_SHA and
        r['prepare_sha256']==binding['prepare_sha256'] and r['goal_q_sha256']==p['goal_q_sha256'] and same(r['model_bindings'],p['model_bindings']),'same original source/model/goalQ')
    require(r['counts']=={k:dict(attempted=17,completed=17) for k in ('encoder','existing_projector')} and
        r['new_predict_calls']==r['new_AD_calls']==r['new_physical_steps']==0 and
        r['training'] is r['P1_admission'] is r['P3_admission'] is r['efficacy_claim'] is False,'derived module budget/scope')
    journal=folder/'encoding.events.jsonl';require(sha(journal)==r['journal_sha256'],'derived journal SHA')
    events=[io.decode(line) for line in regular(journal,RESERVE).read_bytes().splitlines()]
    require(len(r['rows'])==17 and len(r['entries'])==34,'all frame/module records')
    expected_events=[];index=0;results=[];metric=read(p['goal_q_source'])
    require(sha(p['goal_q_source'])==p['goal_q_source_sha256'] and digest(metric['root_metrics'])==p['goal_q_sha256'],'source actual goalQ bytes')
    for frame,row in zip(p['frames'],r['rows']):
        require(all(same(row[k],v) for k,v in frame.items() if k!='predicted_vector') and row['status']=='encoded_and_saved','all ordered own prefix identities')
        require(same(row,read(folder/(frame['frame_id']+'.result.json'))) and same(row,read(folder/(frame['frame_id']+'.latent.json'))),'durable per-frame result')
        attempt=read(folder/(frame['frame_id']+'.attempt.json'))
        require(same(attempt,{k:v for k,v in dict(row,status='attempted',latent_sha256=None).items() if k not in ('latent_dtype','latent_shape')}),'durable original frame attempt')
        expected_events.append(dict(event='endpoint_attempt',frame_id=frame['frame_id']))
        for kind in ('encoder','existing_projector'):
            entry=dict(kind=kind,frame_id=frame['frame_id'],index=index,completed=True);require(r['entries'][index]==entry,'ordered actual module entries');index+=1
            expected_events.extend((dict(event='attempt',**dict(entry,completed=False)),dict(event='return_or_error',**entry)))
        path=folder/(frame['frame_id']+'.native192.npy');require(row['latent_path']==str(path) and sha(path)==row['latent_sha256'],'own new NPY hash/path')
        a=np.load(regular(path,2048),allow_pickle=False);require(a.dtype==np.float32 and a.shape==(192,) and np.isfinite(a).all(),'new native float32 NPY')
        expected_events.append(dict(event='endpoint_saved',frame_id=frame['frame_id'],latent_path=str(path),latent_sha256=row['latent_sha256']))
        source=metric['root_metrics'][frame['root_sha256']];scores=numeric(a,frame['predicted_vector'],source['goal'],{q:source['metrics'][q]['projection'] for q in METRICS})
        results.append(dict(frame_id=frame['frame_id'],root_sha256=frame['root_sha256'],arm=frame['arm'],primitive_step=frame['primitive_step'],
            predicted_history_index=frame['predicted_history_index'],own_Q=frame['metric_label'],**scores))
    require(same(events,expected_events),'exact derived attempt/return/frame journal')
    # Reuse the ALREADY ENCODED real terminal witnesses, never another encode.
    # This makes the new prefixes useful for time localization, without claiming
    # that the first saved nonzero checkpoint is the actual error-onset substep.
    physical=read(p['physical_report']);require(sha(p['physical_report'])==PHYSICAL_SHA,'inherited terminal report')
    terminal_enc=io.index(physical['encodings'],('root_sha256','method'));physical_rows=io.index(physical['rows'],('root_sha256','method'))
    trajectories=[]
    for plan in metric['plans']:
        key=plan['root_sha256'],plan['method'];terminal=terminal_enc[key];actual_row=physical_rows[key]
        require(terminal['status']=='completed' and terminal['terminal_rgb_sha256']==actual_row['terminal_rgb_sha256'] and
            terminal['selection_sha256']==plan['selection_sha256'],'inherited actual terminal identity')
        source=metric['root_metrics'][key[0]];scores=numeric(terminal['terminal_vector'],plan['selected_predicted_context']['latent_history'][-1],
            source['goal'],{q:source['metrics'][q]['projection'] for q in METRICS})
        require(all(np.isclose(scores['actual_Q_costs'][q],terminal['actual_costs'][q],atol=1e-10,rtol=1e-10) for q in METRICS),'inherited terminal own-Q arithmetic')
        points=[dict(primitive_step=x['primitive_step'],source='new_prefix_encoding',**{k:x[k] for k in scores})
            for x in results if (x['root_sha256'],x['arm'])==key]
        points.append(dict(primitive_step=plan['planned_commands'],source='inherited_original_terminal_encoding',**scores))
        require([x['primitive_step'] for x in points]==list(range(5,plan['planned_commands']+1,5)),'all observed packet boundaries without terminal/prefix substitution')
        nonzero=[x['primitive_step'] for x in points if x['native_prediction_mse']>0]
        trajectories.append(dict(root_sha256=key[0],arm=key[1],horizon=plan['planned_horizon_tokens'],checkpoints=points,
            first_saved_nonzero_native_error_step=min(nonzero) if nonzero else None,
            interpretation='first observed nonzero boundary only; onset could be earlier within the unobserved packet, no fitted tolerance'))
    result=dict(schema=NAME+'.numerical_report',status='saved_prefix_numerics_verified',receipt_sha256=expected_sha,
        physical_report_sha256=PHYSICAL_SHA,rows=results,trajectories=trajectories,inherited_terminal_encoder_samples=16,
        new_neural_calls=0,new_physical_steps=0,P1_admission=False,P3_admission=False,
        gaps=['No physical causal/contact attribution from latent MSE alone','No cross-Q subtraction or task success claim',
            'Existing calibration frames, no new independent roots','Packet-boundary observations cannot localize the within-packet error-onset time'])
    output.mkdir(exist_ok=False);write(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    parser.add_argument('--run-dir',type=Path);parser.add_argument('--receipt-sha');args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.output)
    elif args.command=='verify':result=dict(status='verified_not_executed',frames=len(verify(args.prepared,args.prepare_sha)['frames']))
    elif args.command=='run':result=execute_cuda(args.prepared,args.prepare_sha,args.output)
    else:result=consume(args.run_dir,args.receipt_sha,args.output)
    print(io.canonical({k:v for k,v in result.items() if k not in ('rows','entries','model_bindings')}))


if __name__=='__main__':main()
