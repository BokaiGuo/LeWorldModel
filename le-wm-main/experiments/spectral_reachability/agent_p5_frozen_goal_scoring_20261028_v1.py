"""Frozen native JEPA visual goal scoring; MAIN HOST CUDA only.

Public numeric context/goal and sealed model predictions only, never private
physical outcomes. Goal encoding must be durably sealed before any scoring.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
from experiments.spectral_reachability import agent_p5_contact_action_response_20261028_v1 as base

TAG='agent_p5_frozen_goal_scoring_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
BASE_SHA='bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930'
MANIFEST_SHA='572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6'
REPORT_SHA='4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f'
RECEIPT_SHA='247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3'
WORDS=('xy','yx','x_minus_x','zero_zero')
SHAPE=(256,384)
CAP,FLOOR,WALL=16*1024**2,200*1024**2,600
EXPECTED=dict(official_encoder=1,proprio_encoder=0,predictor=0)
NATIVE={
 **base.owner.NATIVE,
 'external/jepa-wms/evals/simu_env_planning/planning/planning/objectives.py':'db8b8a4bac7b7cb6486bc09f6bed7501a73152c0e529abbb57e0454fd93d853f',
 'external/jepa-wms/evals/simu_env_planning/planning/gc_agent.py':'f0a1ccb99b70b69d166aa94f83717ba7a468a6ad3a77a4940a1856dbefb2b9dc'}
require,sha,digest,read_json,write_new=base.require,base.sha,base.digest,base.read_json,base.write_new


def protocol():
    return dict(schema=TAG,words=list(WORDS),denominator=4,shape=list(SHAPE),
        scoring='mean((saved_terminal_patch.astype(float64)-sealed_goal_patch.astype(float64))**2)',
        native_objective='ReprTargetDistMPCObjective visual-only, terminal only; no proprio goal',
        reduction='existing frozen word-contrast float64 saved arithmetic; not bitwise Torch float32 reduction',
        selection='minimum finite MSE, exact ties first in fixed word order, no epsilon',
        goal='public goal_rgb only; original official encoder and frozen context preprocessing',
        seal_before_score=True,preprocessing='uint8 BCHW float32 /255; original make_transforms224 no augmentation',
        new_api=EXPECTED,new_forward_samples=EXPECTED,new_context_frames=EXPECTED,
        reused_api=dict(official_encoder=1,proprio_encoder=1,predictor=8),
        reused_forward_samples=dict(official_encoder=3,proprio_encoder=1,predictor=8),
        reused_context_frames=dict(official_encoder=3,proprio_encoder=3,predictor=24),
        shared='source context cohort charged once, reused predictions not rerun/refunded per word',
        replay=0,padding=0,AD=0,backward=0,JVP=0,VJP=0,SVD=0,physics=0,FLOPs=None,
        training=False,fitting=False,adapter=False,cpu_neural_fallback=False,
        known_single_constructed_OOD_root=True,fresh=False,prospective=False,
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False,
        output_cap_bytes=CAP,free_floor_bytes=FLOOR,wall_seconds=WALL,free_cuda_scratch_bytes=4*1024**3)


def self_bindings():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),
        ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md'))}


def prepare(manifest_path,manifest_sha,report_path,report_sha,receipt_path,receipt_sha):
    require(sha(base.__file__)==BASE_SHA,'frozen numeric source')
    require((manifest_sha,report_sha,receipt_sha)==(MANIFEST_SHA,REPORT_SHA,RECEIPT_SHA),
            'actual four-word public model manifest/report/receipt SHA')
    m=read_json(manifest_path,manifest_sha)
    r=read_json(report_path,report_sha)
    receipt=read_json(receipt_path,receipt_sha)
    require(base.consume(m,report_path,report_sha)==receipt,'full saved numeric strict return')
    require(receipt['status']=='all4_saved_public_action_words_verified','actual strict completion')
    terminals={w:r['artifacts'][w+'_1_visual_out'] for w in WORDS}
    for rec in terminals.values(): base.owner.array(rec,(3,*SHAPE))
    bindings={str(Path(p).absolute()):h for p,h in (
        (manifest_path,manifest_sha),(report_path,report_sha),(receipt_path,receipt_sha),(base.__file__,BASE_SHA))}
    bindings.update({str(ROOT/p):h for p,h in NATIVE.items()})
    bindings.update(self_bindings())
    base.owner.verify_bindings(bindings)
    return dict(schema=TAG,status='prepared_goal_not_encoded_not_scored',protocol=protocol(),
        authority=dict(manifest_path=str(Path(manifest_path).absolute()),manifest_sha=manifest_sha,
            report_path=str(Path(report_path).absolute()),report_sha=report_sha,
            receipt_path=str(Path(receipt_path).absolute()),receipt_sha=receipt_sha),
        public_manifest=m,terminals=terminals,bindings=bindings,
        scientific_admission=False,C2_established=False,physical_release=False)


def verify(m):
    require(m==prepare(**m['authority']),'goal scoring manifest/source drift')
    _,arrays=base.public_inputs(m['public_manifest']['public_payload_path'],
        m['public_manifest']['public_payload_sha256'],m['public_manifest']['model_spec']['stats'])
    return arrays


def scores(goal,terminals):
    goal=base.owner.check_array(goal,SHAPE)
    require(tuple(terminals)==WORDS,'all four original predictions, fixed order')
    rows=[]
    for w,t in terminals.items():
        t=base.owner.check_array(t,SHAPE)
        delta=t.astype(np.float64)-goal.astype(np.float64)
        value=float(np.mean(delta*delta,dtype=np.float64))
        require(np.isfinite(value) and value>=0,'finite goal MSE')
        rows.append(dict(word=w,goal_patch_mse=value,status='scored',error=None))
    selected=min(range(4),key=lambda i:rows[i]['goal_patch_mse'])
    return dict(rows=rows,selected_word=WORDS[selected],tie_rule=protocol()['selection'])


class Meter:
    """Debit before dispatch, no failed-attempt refund; no extra forward allowed."""
    def __init__(self,event,guard=lambda:None):
        self.attempted=dict.fromkeys(EXPECTED,0); self.completed=dict.fromkeys(EXPECTED,0)
        self.pending=None; self.event,self.guard=event,guard

    def before(self,key,batch):
        self.guard()
        require(key in EXPECTED and type(batch) is int and batch==1,'one actual forward sample')
        require(self.pending is None and self.attempted[key]<EXPECTED[key],'predispatch goal cap/inflight')
        self.attempted[key]+=1; self.pending=key
        self.event('attempted',self.snapshot())

    def after(self,key,batch):
        require(type(batch) is int and batch==1 and self.pending==key,'actual matching forward return')
        self.completed[key]+=1; self.pending=None
        self.event('returned',self.snapshot())

    def snapshot(self):
        return dict(attempted=dict(self.attempted),completed=dict(self.completed),pending=self.pending,
            samples_attempted=dict(self.attempted),samples_completed=dict(self.completed),
            frames_attempted=dict(self.attempted),frames_completed=dict(self.completed))


class Store:
    def __init__(self,out):
        self.out=Path(out).absolute()
        require(self.out.parent==ROOT/'results' and self.out.name.startswith(TAG+'_')
            and not self.out.exists() and not any(p.is_symlink() for p in (self.out,*self.out.parents)),
            'exclusive new goal results child')
        self.out.mkdir(exist_ok=False)
        self.started=time.monotonic(); self.bytes=0; self.index=0; self.previous=None

    def guard(self,reserve=0):
        require(time.monotonic()-self.started<WALL,'wall cap')
        require(self.bytes+reserve+1024**2<CAP,'bounded payload with terminal reserve')
        require(shutil.disk_usage(self.out).free>=FLOOR+reserve+1024**2,'disk floor')

    def json(self,name,value):
        require('/' not in name,'local output member')
        self.guard(len(json.dumps(value,allow_nan=False).encode())*2+1024)
        write_new(self.out/name,value)
        self.bytes+=(self.out/name).stat().st_size
        self.sync_dir()

    def sync_dir(self):
        fd=os.open(self.out,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def event(self,kind,value):
        require(self.index<32,'bounded event count')
        e=dict(sequence=self.index,previous_sha256=self.previous,kind=kind,value=value)
        h=digest(e)
        self.json(f'event_{self.index:04d}.json',{**e,'event_sha256':h})
        self.index+=1; self.previous=h

    def goal(self,array):
        a=base.owner.check_array(array,SHAPE)
        self.guard(a.nbytes+1024)
        path=self.out/'goal_patch.npy'
        with path.open('xb') as f:
            np.save(f,a,allow_pickle=False); f.flush(); os.fsync(f.fileno())
        self.bytes+=path.stat().st_size
        self.sync_dir()
        rec=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,shape=list(SHAPE),dtype='float32')
        self.event('goal_artifact',rec)
        return rec


def seal_value(m,rec,before):
    return dict(schema=TAG+'.goal_encoding',manifest_sha256=digest(m),
        public_payload_sha256=m['public_manifest']['public_payload_sha256'],
        goal_rgb=m['public_manifest']['inputs']['goal_rgb'],goal_patch=rec,
        model_receipts=before,protocol_sha256=digest(protocol()))


def score_after_seal(m,store,rec,before):
    seal_path=store.out/'goal_encoding.json'
    seal=read_json(seal_path)
    require(seal==seal_value(m,rec,before),'durable goal encoding seal before score')
    goal=base.owner.array(seal['goal_patch'],SHAPE)
    terminal={w:base.owner.array(m['terminals'][w],(3,*SHAPE))[-1] for w in WORDS}
    result=scores(goal,terminal)
    result['goal_encoding_sha256']=sha(seal_path)
    return result


def cuda_goal(model,rgb):
    import torch
    from app.plan_common.datasets.transforms import make_transforms
    from torch.nn.attention import sdpa_kernel,SDPBackend
    require(rgb.dtype==np.uint8 and rgb.shape==(224,224,3),'public goal RGB ABI')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        pixels=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32)/255.
        transform=make_transforms(img_size=224,random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.),
            reprob=0.,auto_augment=False,motion_shift=False)
        result=model(transform(pixels),is_training=True)['x_norm_patchtokens']
        require(result.is_cuda and result.dtype==torch.float32 and tuple(result.shape)==(1,*SHAPE),'native goal feature ABI')
        return base.owner.check_array(result[0].detach().cpu().numpy().copy(),SHAPE)


def run_cuda(m,output,main_authorized=False):
    require(main_authorized,'explicit MAIN HOST CUDA authorization')
    arrays=verify(m)
    require(not os.environ.get('PYTHONPATH') and not any(k=='pymunk' or k.startswith('pymunk.') for k in sys.modules),
        'fresh process without physics overlay')
    store=Store(output)
    r=dict(schema=TAG+'.report',status='failed',manifest_sha256=digest(m),protocol=protocol(),
        words=[dict(word=w,status='not_scored',goal_patch_mse=None,error=None) for w in WORDS],
        selected_word=None,errors=[],finalization_errors=[],cuda_executed=False,
        scientific_admission=False,C2_established=False,physical_release=False,task_success=None)
    meter=Meter(store.event,store.guard)
    handles=[]; models=None; v4=None; before=None
    old_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame): raise TimeoutError('fixed600s wall; no retry')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(WALL)
    try:
        import torch
        require(torch.cuda.is_available(),'HOST CUDA required, no CPU neural fallback')
        require(torch.cuda.mem_get_info()[0]>=4*1024**3,'4GiB free CUDA scratch')
        loader=importlib.import_module('experiments.spectral_reachability.'+base.owner.LOADER)
        require(sha(loader.__file__)==base.owner.LOADER_SHA,'sealed CUDA loader')
        models,strict,sources,before=loader.load_cuda(ROOT,m['public_manifest']['model_spec']['loader_manifest'])
        _,v5,_=loader.helper(ROOT).dependencies(ROOT); v4=v5.import_v4(ROOT)
        require(set(models)==set(EXPECTED),'exact frozen models')
        r.update(cuda_executed=True,model_before=before,strict_loads=strict,imported_sources=sources)
        def pre(k):
            def hook(module,args):
                require(args[0].is_cuda and tuple(args[0].shape)==(1,3,224,224),'batch1 goal input')
                meter.before(k,int(args[0].shape[0]))
            return hook
        def post(k): return lambda module,args,out:meter.after(k,int(args[0].shape[0]))
        for k,model in models.items():
            handles.append(model.register_forward_pre_hook(pre(k)))
            handles.append(model.register_forward_hook(post(k)))
        rec=store.goal(cuda_goal(models['official_encoder'],arrays['goal_rgb']))
        seal=seal_value(m,rec,before)
        store.json('goal_encoding.json',seal)
        # Both feature and seal fsynced before opening any terminals for scoring.
        store.event('goal_sealed',dict(sha256=sha(store.out/'goal_encoding.json'),record=rec))
        result=score_after_seal(m,store,rec,before)
        store.event('scored',result)
        r.update(words=result['rows'],selected_word=result['selected_word'],
            goal_encoding_sha256=result['goal_encoding_sha256'],goal_patch=rec)
        require(meter.attempted==meter.completed==EXPECTED and meter.pending is None,'complete actual forward census')
        r['status']='all4_public_goal_scores_recorded'
    except Exception as exc:
        r['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old_alarm)
        for handle in handles:
            try: handle.remove()
            except Exception as exc: r['finalization_errors'].append('hook cleanup: '+str(exc))
        if models is not None and v4 is not None:
            try:
                r['model_after']={k:v4.model_receipt(v) for k,v in models.items()}
                require(r['model_after']==before==m['public_manifest']['model_spec']['loader_manifest']['expected_model_receipts'],
                    'frozen modules unchanged')
                v7=loader.helper(ROOT); _,_,v6=v7.dependencies(ROOT)
                sources={**v7.imported_sources('dinov2',v6.official_root(ROOT)),
                    **v7.imported_sources('app',v4.native_root(ROOT)),**v7.imported_sources('src',v4.native_root(ROOT))}
                seals=m['public_manifest']['model_spec']['loader_manifest']['seals']
                require(all(p in seals and sha(p)==seals[p] for p in sources.values()),'final imported source closure')
                r['imported_sources']=sources
            except Exception as exc: r['finalization_errors'].append('model identity: '+str(exc))
        try: verify(m); store.guard()
        except Exception as exc: r['finalization_errors'].append('final source/resource check: '+str(exc))
        if r['errors'] or r['finalization_errors']: r['status']='failed'
        r.update(meter=meter.snapshot(),wall_seconds=time.monotonic()-store.started,journal_terminal_sha256=store.previous)
        # Terminal report may use reserved1MiB even when ordinary guard failed.
        size=len(json.dumps(r,allow_nan=False).encode())*2+1024
        require(size<1024**2 and store.bytes+size<=CAP and shutil.disk_usage(store.out).free>=FLOOR+size,'terminal reserve/floor')
        write_new(store.out/'report.json',r)
    return r


def consume(m,report_path,expected):
    verify(m)
    path=Path(report_path).absolute(); r=read_json(path,expected)
    require(r['schema']==TAG+'.report' and r['manifest_sha256']==digest(m) and r['protocol']==protocol(),'report authority')
    require(r['status']=='all4_public_goal_scores_recorded' and r['errors']==r['finalization_errors']==[]
        and r['cuda_executed'] is True and r['scientific_admission'] is False and r['C2_established'] is False
        and r['physical_release'] is False and r['task_success'] is None,'complete actual CUDA report')
    require(r['model_before']==r['model_after']==m['public_manifest']['model_spec']['loader_manifest']['expected_model_receipts'],
        'frozen model before/after')
    require(np.isfinite(r['wall_seconds']) and 0<=r['wall_seconds']<WALL,'wall receipt')
    spec=m['public_manifest']['model_spec']['loader_manifest']
    require(set(r['strict_loads'])=={'predictor','proprio_encoder'}
        and all(v['exact_match'] is True for v in r['strict_loads'].values()),'strict checkpoint load receipt')
    require(r['imported_sources'] and all(p in spec['seals'] and sha(p)==spec['seals'][p]
        for p in r['imported_sources'].values()),'actual imported source closure')
    events=sorted(path.parent.glob('event_*.json'))
    require(len(events)==5,'exact goal encode/seal/score event census')
    previous=None; kinds=[]; meter=Meter(lambda *args:None)
    seal_path=path.parent/'goal_encoding.json'; seal=read_json(seal_path)
    rec=r['goal_patch']
    require(Path(rec['path'])==path.parent/'goal_patch.npy' and seal==seal_value(m,rec,r['model_before']),
        'exact local sealed goal identity')
    for i,p in enumerate(events):
        require(p.name==f'event_{i:04d}.json','event sequence')
        e=read_json(p)
        require(e['sequence']==i and e['previous_sha256']==previous
            and e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}),'event hash chain')
        previous=e['event_sha256']; kinds.append(e['kind']); v=e['value']
        if e['kind']=='attempted': meter.before('official_encoder',1); require(v==meter.snapshot(),'actual predispatch debit')
        elif e['kind']=='returned': meter.after('official_encoder',1); require(v==meter.snapshot(),'actual forward return')
        elif e['kind']=='goal_artifact': require(v==rec,'actual goal feature record')
        elif e['kind']=='goal_sealed': require(v==dict(sha256=sha(seal_path),record=rec),'seal before score receipt')
        elif e['kind']=='scored':
            actual=score_after_seal(m,SimpleStore(path.parent),rec,r['model_before'])
            require(v==actual and r['words']==actual['rows'] and r['selected_word']==actual['selected_word']
                and r['goal_encoding_sha256']==actual['goal_encoding_sha256'],'all4 deterministic goal scores/tie')
        else: require(False,'unknown goal event')
    require(kinds==['attempted','returned','goal_artifact','goal_sealed','scored'],'goal sealed strictly before scoring')
    require(previous==r['journal_terminal_sha256'] and meter.snapshot()==r['meter']
        and meter.attempted==meter.completed==EXPECTED,'durable full meter')
    wanted={p.name for p in events}|{'report.json','goal_patch.npy','goal_encoding.json'}
    require(path.name=='report.json' and {p.name for p in path.parent.iterdir()}==wanted,'exact output inventory')
    require(sum(p.stat().st_size for p in path.parent.iterdir())<=CAP,'output cap')
    return dict(schema=TAG+'.consume',status='all4_saved_goal_scores_verified',report_sha256=sha(path),
        manifest_sha256=digest(m),goal_encoding_sha256=sha(seal_path),words=r['words'],selected_word=r['selected_word'],
        new_meter=r['meter'],reused_meter=read_json(m['authority']['receipt_path'],RECEIPT_SHA)['meter'],
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False,
        interpretation='known single OOD calibration; model-goal score decision, not native task benefit')


class SimpleStore:
    def __init__(self,out): self.out=out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','cuda','consume'))
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--manifest-sha256')
    for key in ('prediction-manifest','prediction-report','prediction-receipt','report'):
        p.add_argument('--'+key,type=Path); p.add_argument('--'+key+'-sha256')
    p.add_argument('--output',type=Path); p.add_argument('--confirm-host-cuda',action='store_true')
    a=p.parse_args(argv)
    if a.command=='prepare':
        require(all(getattr(a,k) is not None for k in ('prediction_manifest','prediction_manifest_sha256',
            'prediction_report','prediction_report_sha256','prediction_receipt','prediction_receipt_sha256')),'actual prediction authorities required')
        value=prepare(a.prediction_manifest,a.prediction_manifest_sha256,a.prediction_report,a.prediction_report_sha256,
            a.prediction_receipt,a.prediction_receipt_sha256); write_new(a.manifest,value)
    else:
        require(a.manifest_sha256,'actual scoring manifest FILE SHA')
        m=read_json(a.manifest,a.manifest_sha256)
        if a.command=='verify': verify(m); value=dict(status='goal_scoring_plan_verified')
        elif a.command=='cuda':
            require(a.confirm_host_cuda and a.output is not None,'MAIN explicit CUDA confirmation/new output')
            value=run_cuda(m,a.output,main_authorized=True)
        else:
            require(a.report is not None and a.report_sha256 and a.output is not None,'actual CUDA report/new receipt')
            value=consume(m,a.report,a.report_sha256); write_new(a.output,value)
    print(json.dumps(dict(status=value['status'],scientific_admission=False),sort_keys=True))
    return int(value['status']=='failed')


if __name__=='__main__': raise SystemExit(main())
