"""Ten fixed static raster encodes, MAIN HOST CUDA only. No planner decision."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

import numpy as np
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v5 as owner
from experiments.spectral_reachability import agent_p5_renderer_factorial_20261028_v3 as renderer

TAG='agent_p5_target_raster_encoder_20261028_v1'
ROOT=owner.ROOT
OWNER_SHA='aece72c75efa24f8c5e2df12aff38a954df31307a0476edf28be2295b1320025'
RENDERER_SHA='b2b41d1aee81d37a1569fe4943fa1b31cb8c7dbee30790fa4c07ee7133a97fcd'
WORDS=owner.WORDS
STATES=(*WORDS,'goal')
RASTERS=('visible','white')
KEYS=tuple(s+'__'+r for s in STATES for r in RASTERS)
SHAPE=(256,384)
EXPECTED=dict(official_encoder=10,proprio_encoder=0,predictor=0)
CAP,FLOOR,WALL=32*1024**2,200*1024**2,600
require,sha,read_json,write_new,digest=owner.require,owner.impl.sha,owner.impl.read_json,owner.impl.write_new,owner.impl.digest


def protocol():
    return dict(schema=TAG,states=list(STATES),rasters=list(RASTERS),image_order=list(KEYS),images=10,
        cross_denominator=16,shape=list(SHAPE),batch=1,deduplicate=False,
        reduction='float64 mean((terminal_feature-goal_feature)**2) over all 256x384 elements',
        encoder='frozen P5 v5 official encoder/preprocessing/math SDPA; full x_norm_patchtokens only',
        goal_seal='all ten token files fsynced before any cross-condition score',
        new_API=EXPECTED,new_forward_samples=EXPECTED,new_frames=EXPECTED,
        AD=0,backward=0,JVP=0,VJP=0,SVD=0,physics=0,predictor=0,FLOPs=None,
        forward_scope='all attempts, batch1; no warmup/replay/padding/retry; failed attempts not refunded',
        wall_scope='saved preflight before timer; CUDA load/encode/final source verification timed',
        native_coverage='old sealed physics only, not a new observation and never used to choose inputs/words',
        complete_dynamic_restore=False,agent_hidden=False,planner_rescoring=False,selection=False,
        masks=False,fitting=False,training=False,cpu_neural_fallback=False,
        single_known_constructed_OOD_root=True,fresh=False,prospective=False,
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False,
        output_cap_bytes=CAP,free_floor_bytes=FLOOR,wall_seconds=WALL,cuda_free_scratch_bytes=4*1024**3)


def actual_sha(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'explicit actual SHA256 required')
    return value


def prepare(static_manifest,static_manifest_sha256,static_report,static_report_sha256,static_receipt,static_receipt_sha256):
    """Named actual authorities, no future SHA/path defaults and no model imports."""
    require(sha(owner.__file__)==OWNER_SHA and sha(renderer.__file__)==RENDERER_SHA,'frozen owner/static consumer sources')
    args=dict(static_manifest=str(Path(static_manifest).resolve()),static_manifest_sha256=actual_sha(static_manifest_sha256),
        static_report=str(Path(static_report).resolve()),static_report_sha256=actual_sha(static_report_sha256),
        static_receipt=str(Path(static_receipt).resolve()),static_receipt_sha256=actual_sha(static_receipt_sha256))
    p=renderer.verify(Path(args['static_manifest']),args['static_manifest_sha256'])
    r=read_json(args['static_report'],args['static_report_sha256'])
    c=read_json(args['static_receipt'],args['static_receipt_sha256'])
    exact=renderer.consume(Path(args['static_manifest']),args['static_manifest_sha256'],
        Path(args['static_report']),args['static_report_sha256'])
    require(digest(c)==digest(exact) and c['schema']==renderer.NAME+'.consume' and c['status']=='all10_saved_static_target_views_verified'
        and c['manifest_sha256']==args['static_manifest_sha256'] and c['report_sha256']==args['static_report_sha256'],
        'actual v3 full strict persisted return required')
    require(c['native_consumer']['denominator']==10 and c['native_consumer']['baselines']==5
        and c['native_consumer']['Space_step']==0 and c['native_consumer']['complete_dynamic_restore'] is False,
        'static render-only all-five source contract')
    native=p['native_manifest']
    v5path=native['authority']['manifest']
    v5=read_json(v5path,renderer.native.parent.PREP_SHA)
    owner.verify(v5)
    require([x['state_id'] for x in native['states']]==[x['state_id'] for x in r['rows']]==list(STATES),'all-five exact state order')
    bindings={}
    def bind(path,h):
        path=str(Path(path).resolve())
        require(path not in bindings or bindings[path]==h,'authority binding collision')
        require(sha(path)==h,'bound actual source/input SHA')
        bindings[path]=h
    for source in (p['bindings'],v5['bindings'],v5['authority']['model_spec']['loader_manifest']['seals']):
        for path,h in source.items(): bind(path,h)
    for name in ('static_manifest','static_report','static_receipt'):
        bind(args[name],args[name+'_sha256'])
    launch=Path(str(Path(args['static_report']).parent)+'.launch.json')
    bind(launch,sha(launch))
    for module,h in ((owner,OWNER_SHA),(renderer,RENDERER_SHA)): bind(module.__file__,h)
    for path in (Path(__file__),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')): bind(path,sha(path))
    inputs=[]; coverage=[]
    out=Path(args['static_report']).parent
    for source,row in zip(native['states'],r['rows']):
        s=source['state_id']; require(source['original_target']=='visible','source-inferred original target corrected')
        expected=renderer.native.view_state(source['source_state'])
        if s in WORDS:
            old=v5['authority']['terminals'][s]['native']
            require(source['source_state']==old['public_double'],'exact original word/full-public-state join')
            t=old['transition']; cov=old['public_double']['env']['coverage_arr'][-1]
            require(t['done'] is False and t['info']['final_coverage']==cov,'old native coverage provenance')
            coverage.append(dict(word=s,coverage=cov,reward=t['reward'],done=False,task_success=None,
                source='sealed original four-word physics, not new rendering',source_state_sha256=source['native_sha'],
                source_state_path=source['native_path'],source_transition_sha256=old['transition_sha']))
        for target in RASTERS:
            shot=next(x for x in row['shots'] if x['target']==target)
            refs={}
            for field in ('rgb','before','after'):
                rec=shot['files'][field]; path=out/rec['path']
                require(path.stat().st_size==rec['bytes'],'static member bytes')
                bind(path,rec['sha256']); refs[field]=dict(path=str(path.resolve()),sha256=rec['sha256'],bytes=rec['bytes'])
            require(read_json(refs['before']['path'],refs['before']['sha256'])==read_json(refs['after']['path'],refs['after']['sha256'])==expected,
                    'exact full static state invariance')
            owner.array(refs['rgb'],(224,224,3),np.uint8)
            inputs.append(dict(image_key=s+'__'+target,state_id=s,raster=target,**refs,
                source_state_sha256=source['native_sha'],static_state_digest=digest(expected),
                readonly_difference=deepcopy(row['restore']['readonly_difference'])))
    require(tuple(x['image_key'] for x in inputs)==KEYS and len(coverage)==4,'ten-image/four-word complete denominator')
    return dict(schema=TAG,status='prepared_actual_static_receipt_no_encodes',protocol=protocol(),static_authority=args,
        static_receipt=c,inputs=inputs,legacy_native_coverage=coverage,
        model_spec=deepcopy(v5['authority']['model_spec']),bindings=bindings,
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False)


def verify(m):
    require(m['schema']==TAG and m['protocol']==protocol(),'own exact protocol')
    require(all(sha(path)==h for path,h in m['bindings'].items()),'full binding drift')
    require(digest(m)==digest(prepare(**m['static_authority'])),'full ten-image authority recomputation')


def runtime_view(m):
    return dict(authority=dict(model_spec=m['model_spec']))


def load_runtime(m):
    require(sha(owner.__file__)==OWNER_SHA,'frozen v5 runtime/preprocessing source')
    return owner.load_runtime(runtime_view(m))


def encode(model,rgb):
    # No global substitution of old TAG/protocol/verify/Meter. This pure call
    # retains exact frozen native transform and full patch ABI, on HOST CUDA only.
    return owner.encode(model,rgb)


def source_closure(m,sources):
    owner.source_closure(runtime_view(m),sources)


def scores(features):
    require(tuple(features)==KEYS,'all ten sealed feature arrays in fixed order')
    for a in features.values():
        require(a.shape==SHAPE and a.dtype==np.float32 and np.isfinite(a).all(),'full float32 patch ABI')
    rows=[]
    for word in WORDS:
        for terminal in RASTERS:
            for goal in RASTERS:
                value=owner.mse(features[word+'__'+terminal],features['goal__'+goal])
                rows.append(dict(word=word,terminal_raster=terminal,goal_raster=goal,status='computed',
                                 full_patch_goal_mse=value))
    changes=[dict(state_id=s,visible_white_feature_mse=owner.mse(features[s+'__visible'],features[s+'__white'])) for s in STATES]
    return dict(cross_condition=rows,fixed_within_state_raster_difference=changes)


def empty_scores():
    return [dict(word=w,terminal_raster=t,goal_raster=g,status='not_computed',full_patch_goal_mse=None)
            for w in WORDS for t in RASTERS for g in RASTERS]


class Meter:
    def __init__(self,event,guard=lambda:None):
        self.attempted=dict.fromkeys(EXPECTED,0); self.completed=dict.fromkeys(EXPECTED,0)
        self.attempted_order=[]; self.completed_order=[]; self.pending=None; self.item=None
        self.event,self.guard=event,guard

    def before(self,key,batch):
        self.guard()
        require(key in EXPECTED and type(batch) is int and batch==1 and self.pending is None,'batch1/inflight contract')
        require(self.attempted[key]<EXPECTED[key],'predispatch module cap')
        require(len(self.attempted_order)<10 and self.item==KEYS[len(self.attempted_order)],'one forward per image, fixed ten, no replay')
        self.attempted[key]+=1; self.attempted_order.append(self.item); self.pending=(key,self.item)
        self.event('attempted',self.snapshot())

    def after(self,key,batch):
        require(type(batch) is int and batch==1 and self.pending==(key,self.item),'actual paired return')
        self.completed[key]+=1; self.completed_order.append(self.item); self.pending=None
        self.event('returned',self.snapshot())

    def snapshot(self):
        return dict(attempted=dict(self.attempted),completed=dict(self.completed),
            forward_samples_attempted=dict(self.attempted),forward_samples_completed=dict(self.completed),
            frames_attempted=dict(self.attempted),frames_completed=dict(self.completed),
            attempted_order=list(self.attempted_order),completed_order=list(self.completed_order),
            pending=None if self.pending is None else list(self.pending),image_key=self.item)


class Store:
    def __init__(self,out):
        self.out=Path(out).absolute()
        require(self.out.parent==ROOT/'results' and self.out.name.startswith(TAG+'_') and not self.out.exists()
            and not any(p.is_symlink() for p in (self.out,*self.out.parents)),'exclusive nonsymlink own results child')
        self.out.mkdir(); self.started=time.monotonic(); self.bytes=0; self.index=0; self.previous=None

    def guard(self,reserve=0):
        require(time.monotonic()-self.started<WALL,'600s wall cap')
        require(self.bytes+reserve+1024**2<=CAP,'32MiB payload including final reserve')
        require(shutil.disk_usage(self.out).free>=FLOOR+reserve+1024**2,'200MiB floor')

    def sync(self):
        fd=os.open(self.out,os.O_DIRECTORY|os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def event(self,kind,value):
        require(self.index<64,'bounded durable events')
        row=dict(sequence=self.index,previous_sha256=self.previous,kind=kind,value=value)
        row['event_sha256']=digest(row)
        self.guard(len(json.dumps(row,allow_nan=False).encode())*2+1024)
        path=self.out/f'event_{self.index:04d}.json'; write_new(path,row); self.sync()
        self.bytes+=path.stat().st_size; self.index+=1; self.previous=row['event_sha256']

    def emit(self,key,a):
        require(key in KEYS and a.shape==SHAPE and a.dtype==np.float32 and np.isfinite(a).all(),'full encoder array ABI')
        self.guard(a.nbytes+1024); path=self.out/(key+'_patch.npy')
        with path.open('xb') as f: np.save(f,a,allow_pickle=False); f.flush(); os.fsync(f.fileno())
        self.sync(); self.bytes+=path.stat().st_size
        rec=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,shape=list(SHAPE),dtype='float32')
        self.event('artifact',dict(image_key=key,record=rec))
        return rec


def run_cuda(m,out,confirm=False):
    require(confirm and out is not None,'MAIN HOST CUDA confirmation required')
    verify(m)
    require(not os.environ.get('PYTHONPATH') and not any(k=='pymunk' or k.startswith('pymunk.') for k in sys.modules),
            'fresh CUDA process without physics overlay/imports')
    require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'no existing wall timer')
    store=Store(out); meter=Meter(store.event,store.guard); handles=[]; receipts=imported=None
    r=dict(schema=TAG+'.report',status='failed',manifest_digest=digest(m),protocol=protocol(),
        items=[dict(image_key=k,status='not_run',error=None) for k in KEYS],token_artifacts={},
        cross_condition=empty_scores(),fixed_within_state_raster_difference=None,
        legacy_native_coverage=deepcopy(m['legacy_native_coverage']),errors=[],finalization_errors=[],
        cuda_executed=False,all10_tokens_sealed_before_scores=False,scientific_admission=False,
        C2_established=False,task_success=None,physical_release=False,new_physics=0,AD=0)
    old_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*args): raise TimeoutError('600s cap; no retry')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(WALL)
    try:
        models,strict,sources,before,receipts,imported=load_runtime(m)
        r.update(cuda_executed=True,model_before=before,strict_loads=strict,imported_sources=sources)
        require(set(models)==set(EXPECTED),'exact three frozen modules')
        require(before==m['model_spec']['loader_manifest']['expected_model_receipts'],'expected original model identity')
        source_closure(m,sources)
        def pre(key):
            def hook(module,args):
                require(key=='official_encoder','no new predictor/proprio forward')
                require(args[0].is_cuda and tuple(args[0].shape)==(1,3,224,224),'actual CUDA batch1 RGB')
                meter.before(key,int(args[0].shape[0]))
            return hook
        def post(key): return lambda module,args,out:meter.after(key,int(args[0].shape[0]))
        for key,model in models.items():
            handles.append(model.register_forward_pre_hook(pre(key)))
            handles.append(model.register_forward_hook(post(key)))
        features={}
        for i,item in enumerate(m['inputs']):
            key=item['image_key']; require(key==KEYS[i],'fixed input order')
            meter.item=key; r['items'][i]['status']='attempted'
            rgb=owner.array(item['rgb'],(224,224,3),np.uint8)
            features[key]=encode(models['official_encoder'],rgb)
            require(len(meter.completed_order)==i+1 and meter.pending is None,'one actual encoder return per image')
            r['token_artifacts'][key]=store.emit(key,features[key]); r['items'][i]['status']='completed'
        require(meter.attempted==meter.completed==EXPECTED and meter.pending is None,'complete ten actual forwards')
        store.event('all10_tokens_sealed',dict(records=r['token_artifacts'],order=list(KEYS)))
        r['all10_tokens_sealed_before_scores']=True
        r.update(scores(features)); r['status']='all10_raster_encodes_recorded'
    except Exception as exc:
        msg=type(exc).__name__+': '+str(exc); r['errors'].append(msg)
        for item in r['items']:
            if item['status']=='attempted': item.update(status='failed',error=msg)
    finally:
        for handle in handles:
            try: handle.remove()
            except Exception as exc: r['finalization_errors'].append('hook cleanup: '+str(exc))
        if receipts is not None:
            try:
                r['model_after']=receipts()
                require(r['model_after']==r['model_before']==m['model_spec']['loader_manifest']['expected_model_receipts'],'before/after/expected model strong equality')
                r['imported_sources']=imported(); source_closure(m,r['imported_sources'])
            except Exception as exc: r['finalization_errors'].append('model/source: '+str(exc))
        try: verify(m); store.guard()
        except Exception as exc: r['finalization_errors'].append('final saved authority/resources: '+str(exc))
        signal.alarm(0); signal.signal(signal.SIGALRM,old_alarm)
        if r['errors'] or r['finalization_errors']: r['status']='failed'
        r.update(meter=meter.snapshot(),wall_seconds=time.monotonic()-store.started,journal_terminal_sha256=store.previous)
        reserve=len(json.dumps(r,allow_nan=False).encode())*2+1024
        require(reserve<1024**2 and store.bytes+reserve<=CAP and shutil.disk_usage(store.out).free>=FLOOR+reserve,'report reserve/floor')
        write_new(store.out/'report.json',r); store.sync()
    return r


def consume(m,report,report_sha256):
    verify(m)
    path=Path(report).absolute(); r=read_json(path,actual_sha(report_sha256))
    require(r['schema']==TAG+'.report' and r['manifest_digest']==digest(m) and r['protocol']==protocol(),'report exact authority')
    require(r['status']=='all10_raster_encodes_recorded' and r['errors']==r['finalization_errors']==[]
        and r['cuda_executed'] is r['all10_tokens_sealed_before_scores'] is True,'complete HOST CUDA required')
    require(r['scientific_admission'] is r['C2_established'] is r['physical_release'] is False
        and r['task_success'] is None and r['new_physics']==r['AD']==0,'diagnostic only, no scientific admission')
    require(r['model_before']==r['model_after']==m['model_spec']['loader_manifest']['expected_model_receipts'],'frozen expected/before/after identity')
    source_closure(m,r['imported_sources'])
    require(set(r['strict_loads'])=={'predictor','proprio_encoder'} and all(x['exact_match'] is True for x in r['strict_loads'].values()),'original strict checkpoint loads')
    require(np.isfinite(r['wall_seconds']) and 0<=r['wall_seconds']<WALL,'bounded wall')
    require(r['legacy_native_coverage']==m['legacy_native_coverage'],'unchanged old physics parallel values')
    require(r['items']==[dict(image_key=k,status='completed',error=None) for k in KEYS] and set(r['token_artifacts'])==set(KEYS),'all ten files/returns')
    events=sorted(path.parent.glob('event_*.json')); require(len(events)==31,'ten intent/return/artifact triplets then seal')
    meter=Meter(lambda *args:None); records={}; previous=None
    for i,p in enumerate(events):
        row=read_json(p)
        require(p.name==f'event_{i:04d}.json' and row['sequence']==i and row['previous_sha256']==previous
            and row['event_sha256']==digest({k:v for k,v in row.items() if k!='event_sha256'}),'durable event chain')
        previous=row['event_sha256']
        if i==30:
            require(row['kind']=='all10_tokens_sealed' and row['value']==dict(records=records,order=list(KEYS)),'all tokens durable before scoring')
            continue
        meter.item=KEYS[i//3]; kind=('attempted','returned','artifact')[i%3]
        require(row['kind']==kind,'actual fixed image event order')
        if kind in ('attempted','returned'):
            (meter.before if kind=='attempted' else meter.after)('official_encoder',1)
            require(row['value']==meter.snapshot(),'API/sample/frame/order replay')
        else:
            require(row['value']['image_key']==meter.item,'fixed token image identity')
            records[meter.item]=row['value']['record']
    require(previous==r['journal_terminal_sha256'] and records==r['token_artifacts'] and r['meter']==meter.snapshot()
        and meter.attempted==meter.completed==EXPECTED and meter.pending is None,'complete actual ten-image meter/artifacts')
    features={}
    for key in KEYS:
        rec=records[key]
        require(Path(rec['path'])==path.parent/(key+'_patch.npy') and rec['shape']==list(SHAPE) and rec['dtype']=='float32','local full token ABI/path')
        features[key]=owner.array(rec,SHAPE)
    computed=scores(features)
    require(r['cross_condition']==computed['cross_condition'] and r['fixed_within_state_raster_difference']==computed['fixed_within_state_raster_difference'],
            'all16 cross-condition/full-token independent recomputation')
    names={p.name for p in events}|{'report.json'}|{k+'_patch.npy' for k in KEYS}
    require(path.name=='report.json' and {p.name for p in path.parent.iterdir()}==names,'exact producer layout')
    require(sum(p.stat().st_size for p in path.parent.iterdir())<=CAP,'saved output cap')
    verify(m); require(sha(path)==report_sha256,'final report drift')
    return dict(schema=TAG+'.consume',status='all10_saved_raster_encodes_and16_cross_scores_verified',
        report_sha256=report_sha256,manifest_digest=digest(m),cross_condition=computed['cross_condition'],
        fixed_within_state_raster_difference=computed['fixed_within_state_raster_difference'],
        legacy_native_coverage=m['legacy_native_coverage'],new_meter=r['meter'],AD=0,physics=0,
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False,
        interpretation='known single-root raster sensitivity diagnostic; not new planner score, causal mechanism or task benefit')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','cuda','consume')); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256'); p.add_argument('--output',type=Path); p.add_argument('--report',type=Path)
    p.add_argument('--report-sha256'); p.add_argument('--confirm-host-cuda',action='store_true')
    for key in ('static-manifest','static-report','static-receipt'):
        p.add_argument('--'+key,type=Path); p.add_argument('--'+key+'-sha256')
    a=p.parse_args(argv)
    if a.command=='prepare':
        names=('static_manifest','static_manifest_sha256','static_report','static_report_sha256','static_receipt','static_receipt_sha256')
        args={k:getattr(a,k) for k in names}; require(all(v is not None for v in args.values()),'six explicit actual static authority arguments required')
        result=prepare(**args); write_new(a.manifest,result)
    else:
        require(a.manifest_sha256 and sha(a.manifest)==actual_sha(a.manifest_sha256),'actual own prepare file SHA')
        m=read_json(a.manifest,a.manifest_sha256)
        if a.command=='verify': verify(m); result=dict(status='ten_image_encoder_plan_verified')
        elif a.command=='cuda': result=run_cuda(m,a.output,a.confirm_host_cuda)
        else:
            require(a.report and a.report_sha256 and a.output,'actual CUDA report SHA/receipt output required')
            result=consume(m,a.report,a.report_sha256); write_new(a.output,result)
    print(json.dumps({'status':result['status'],'scientific_admission':False}))
    return 2 if result['status']=='failed' else 0


if __name__=='__main__': raise SystemExit(main())
