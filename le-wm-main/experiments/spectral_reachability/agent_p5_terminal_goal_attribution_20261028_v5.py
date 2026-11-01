"""v5 order-correct strict terminal RGB diagnostic.

Public authority order is goal, physical, numeric. The frozen v2 delegate
requires goal, numeric, physical; this adapter performs that explicit swap.
"""
from __future__ import annotations
import argparse, json, os, signal, sys, time, shutil, importlib
from copy import deepcopy
import numpy as np
from pathlib import Path
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v2 as impl

TAG='agent_p5_terminal_goal_attribution_20261028_v5'; ROOT=impl.ROOT
BASE_PROTOCOL=impl.protocol()
IMPL_SHA='849d7647110b663414d7ac989ce923887c4fec556021f701408e8c4ca9c42a25'
WORDS=('xy','yx','x_minus_x','zero_zero'); SHAPE=(256,384)
EXPECTED=dict(official_encoder=4,proprio_encoder=0,predictor=0)
CAP,FLOOR,WALL=32*1024**2,200*1024**2,600
ORDER=('goal_prepare','goal_report','goal_receipt','physical_manifest','physical_report','physical_receipt','numeric_manifest','numeric_report','numeric_receipt')

def require(ok,msg):
    if not ok: raise ValueError(msg)
def protocol():
    p=deepcopy(BASE_PROTOCOL); p.update(schema=TAG,authority_order=list(ORDER),v2_unchanged=True,v3_unchanged=True,
        reduction='float64 saved MSE; no CPU neural execution',AD=0,backward=0,JVP=0,VJP=0,SVD=0,
        wall_scope='saved preflight before timer; CUDA load/4 terminal encodes/final verification timed',
        forward_scope='all attempts, no warmup/replay/padding; failed attempts not refunded',
        raw_rgb_attribution='unknown: agent/block/goal raster confounded, no causal claim')
    return p
def paths(values):
    require(len(values)==9,'authority requires goal(3), physical(3), numeric(3)')
    return dict(zip(ORDER,map(Path,values)))
def fixed_authorities(gp,gr,gc,pm,pr,pc,sm,sr,sc):
    """v2 join with distinct path/report variable names; all saved only."""
    require((impl.sha(gp),impl.sha(gr),impl.sha(gc))==(impl.GOAL_PREP,impl.GOAL_REPORT,impl.GOAL_RECEIPT),'goal authority SHA')
    require((impl.sha(pm),impl.sha(pr),impl.sha(pc))==(impl.NUM_MANIFEST_SHA,impl.NUM_REPORT_SHA,impl.NUM_RECEIPT_SHA),'numeric authority SHA')
    require((impl.sha(sm),impl.sha(sr),impl.sha(sc))==(impl.PHYS_MANIFEST_SHA,impl.PHYS_REPORT_SHA,impl.PHYS_RECEIPT_SHA),'physical authority SHA')
    gm=impl.read_json(gp,impl.GOAL_PREP); goal_report=impl.read_json(gr,impl.GOAL_REPORT); goal_receipt=impl.read_json(gc,impl.GOAL_RECEIPT)
    nm=impl.read_json(pm,impl.NUM_MANIFEST_SHA); num_report=impl.read_json(pr,impl.NUM_REPORT_SHA); num_receipt=impl.read_json(pc,impl.NUM_RECEIPT_SHA)
    smv=impl.read_json(sm,impl.PHYS_MANIFEST_SHA); phys_report=impl.read_json(sr,impl.PHYS_REPORT_SHA); phys_receipt=impl.read_json(sc,impl.PHYS_RECEIPT_SHA)
    goalmod=impl.strict_module(impl.ROOT/'experiments/spectral_reachability'/(impl.GOAL_SOURCE+'.py'),impl.GOAL_SOURCE_SHA,'goal_v5')
    nummod=impl.strict_module(impl.ROOT/'experiments/spectral_reachability'/(impl.NUM+'.py'),impl.NUM_SOURCE_SHA,'numeric_v5')
    wrapper=impl.strict_module(impl.ROOT/'experiments/spectral_reachability/consume_contact_words_MAIN_20261028_v1.py',impl.PHYS_CONSUMER_SHA,'physical_v5')
    require(goalmod.consume(gm,gr,impl.GOAL_REPORT)==goal_receipt,'goal report/receipt exact return')
    require(nummod.consume(nm,pr,impl.NUM_REPORT_SHA)==num_receipt,'numeric report/receipt exact return')
    require(wrapper.consume(sm,impl.PHYS_MANIFEST_SHA,sr,impl.PHYS_REPORT_SHA)==phys_receipt,'physical report/receipt exact return')
    require(goal_receipt['status']=='all4_saved_goal_scores_verified' and num_receipt['status']=='all4_saved_public_action_words_verified' and phys_receipt['status']=='all4_saved_native_words_verified','all strict statuses')
    goalrec=goal_report['goal_patch']; require(goalrec['shape']==list(impl.SHAPE) and impl.sha(goalrec['path'])==goalrec['sha256'] and Path(goalrec['path']).stat().st_size==goalrec['bytes'],'goal patch SHA/bytes')
    prediction={w:num_report['artifacts'][f'{w}_1_visual_out'] for w in impl.WORDS}
    for rec in prediction.values(): require(rec['shape']==[3,*impl.SHAPE] and impl.sha(rec['path'])==rec['sha256'] and Path(rec['path']).stat().st_size==rec['bytes'],'prediction SHA/bytes')
    require(phys_report['status']=='all4_native_word_traces_recorded' and phys_report['global_error'] is None and [x['arm'] for x in phys_report['rows']]==list(impl.WORDS),'physical rows')
    terminals={}
    for row in phys_report['rows']:
        rec=next(x for x in row['records'] if x['frame']==20); base=Path(sr).parent/row['arm']/'frame_20'; files=rec['files']; members={k:base/k for k in ('rgb.npy','full_public_state.json','transition.json')}
        for k,p in members.items(): require(impl.sha(p)==files[k]['sha256'] and p.stat().st_size==files[k]['bytes'],'terminal SHA/bytes')
        terminals[row['arm']]=dict(rgb=dict(path=str(members['rgb.npy'].resolve()),sha256=files['rgb.npy']['sha256'],bytes=files['rgb.npy']['bytes']),native=dict(state=rec['state'],proprio=rec['proprio'],state_file=str(members['full_public_state.json'].resolve()),transition_file=str(members['transition.json'].resolve()),state_sha=files['full_public_state.json']['sha256'],transition_sha=files['transition.json']['sha256']))
    return dict(gm=gm,gr=goal_receipt,greport=goal_report,nm=nm,nr=num_receipt,nreport=num_report,sm=smv,sr=phys_receipt,sreport=phys_report,goalrec=goalrec,prediction=prediction,terminals=terminals,model_spec=nm['model_spec'],paths=dict(goal_prepare=str(Path(gp).resolve()),goal_report=str(Path(gr).resolve()),goal_receipt=str(Path(gc).resolve()),num_manifest=str(Path(pm).resolve()),num_report=str(Path(pr).resolve()),num_receipt=str(Path(pc).resolve()),phys_manifest=str(Path(sm).resolve()),phys_report=str(Path(sr).resolve()),phys_receipt=str(Path(sc).resolve())))
def prepare(*values):
    require(impl.sha(impl.__file__)==IMPL_SHA,'frozen constants/helper source')
    p=paths(values)
    # IMPORTANT: v2 internal ABI is goal,numeric,physical.
    a=fixed_authorities(p['goal_prepare'],p['goal_report'],p['goal_receipt'],
        p['numeric_manifest'],p['numeric_report'],p['numeric_receipt'],
        p['physical_manifest'],p['physical_report'],p['physical_receipt'])
    bindings={str(x.resolve()):impl.sha(x) for x in p.values()}
    for name in (impl.GOAL_SOURCE,impl.NUM,impl.PHYS,'consume_contact_words_MAIN_20261028_v1'):
        source=ROOT/'experiments/spectral_reachability'/(name+'.py')
        bindings[str(source)]=impl.sha(source)
    bindings[str(Path(impl.__file__).resolve())]=IMPL_SHA
    for rec in a['terminals'].values():
        rec['native']['public_double']=impl.read_json(rec['native']['state_file'],rec['native']['state_sha'])
        rec['native']['transition']=impl.read_json(rec['native']['transition_file'],rec['native']['transition_sha'])
    for rec in (a['goalrec'],*a['prediction'].values()):
        bindings[rec['path']]=rec['sha256']
    for term in a['terminals'].values():
        bindings[term['rgb']['path']]=term['rgb']['sha256']
        bindings[term['native']['state_file']]=term['native']['state_sha']
        bindings[term['native']['transition_file']]=term['native']['transition_sha']
    value=dict(status='prepared_strict_saved_join',authority=a,bindings=bindings,
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False)
    value['schema']=TAG; value['protocol']=protocol(); value['authority_order']=list(ORDER)
    value['v5_source_sha256']=impl.sha(Path(__file__))
    value['bindings'][str(Path(__file__).resolve())]=impl.sha(Path(__file__))
    value['bindings'][str(ROOT/'tests'/('test_'+TAG+'.py'))]=impl.sha(ROOT/'tests'/('test_'+TAG+'.py'))
    value['bindings'][str(ROOT/'innerdocs'/(TAG+'.md'))]=impl.sha(ROOT/'innerdocs'/(TAG+'.md'))
    require(all(impl.sha(p)==h for p,h in value['bindings'].items()),'final prepared authority drift')
    return value
def verify(value):
    require(value['schema']==TAG and value['protocol']==protocol() and value['authority_order']==list(ORDER),'v5 manifest/order')
    require(all(impl.sha(p)==h for p,h in value['bindings'].items()),'v5 binding drift')
    q=value['authority']['paths']
    fresh=prepare(q['goal_prepare'],q['goal_report'],q['goal_receipt'],q['phys_manifest'],q['phys_report'],q['phys_receipt'],q['num_manifest'],q['num_report'],q['num_receipt'])
    # authority paths in the delegate are canonicalized under its internal names;
    # compare all fields after replacing only v5 identity fields.
    require(fresh==value,'full v5 authority recomputation')
def array(rec,shape,dtype=np.float32):
    p=Path(rec['path'])
    require(impl.sha(p)==rec['sha256'] and p.stat().st_size==rec['bytes'],'array SHA/bytes')
    require(p.stat().st_size<=int(np.prod(shape))*np.dtype(dtype).itemsize+1024,'bounded array')
    a=np.load(p,allow_pickle=False,max_header_size=1024)
    require(a.shape==shape and a.dtype==np.dtype(dtype) and np.isfinite(a).all(),'array shape/dtype/finite')
    return a

def mse(x,y):
    return float(np.mean((x.astype(np.float64)-y.astype(np.float64))**2))

def metrics(a,real):
    require(set(real)==set(WORDS),'all four real token denominator')
    goal=array(a['goalrec'],SHAPE); rows=[]
    for w in WORDS:
        r=real[w]; require(r.shape==SHAPE and r.dtype==np.float32 and np.isfinite(r).all(),'full terminal feature ABI')
        p=array(a['prediction'][w],(3,*SHAPE))[-1]
        native=a['terminals'][w]['native']; public=native['public_double']; t=native['transition']
        require(t['done'] is False,'native done=False')
        coverage=public['env']['coverage_arr'][-1]
        require(t['info']['final_coverage']==coverage,'native terminal coverage')
        pm,rm=mse(p,goal),mse(r,goal)
        rows.append(dict(word=w,status='completed',error=None,predicted_goal_mse=pm,
            real_terminal_goal_mse=rm,prediction_to_real_feature_mse=mse(p,r),
            predicted_minus_real_goal_mse=pm-rm,
            native=dict(state7=native['state'],proprio=native['proprio'],
                agent_double=public['bodies']['agent'],block_double=public['bodies']['block'],
                coverage=coverage,reward=t['reward'],done=False,task_success=None)))
    return rows

class Meter:
    def __init__(self,event,guard=lambda:None):
        self.attempted=dict.fromkeys(EXPECTED,0); self.completed=dict.fromkeys(EXPECTED,0)
        self.pending=None; self.event,self.guard=event,guard
    def before(self,key,batch):
        self.guard(); require(key in EXPECTED and type(batch) is int and batch==1,'fixed batch1')
        require(self.pending is None and self.attempted[key]<EXPECTED[key],'predispatch cap/inflight')
        self.attempted[key]+=1; self.pending=key; self.event('attempted',self.snapshot())
    def after(self,key,batch):
        require(self.pending==key and type(batch) is int and batch==1,'paired actual return')
        self.completed[key]+=1; self.pending=None; self.event('returned',self.snapshot())
    def snapshot(self):
        return dict(attempted=dict(self.attempted),completed=dict(self.completed),pending=self.pending,
            samples_attempted=dict(self.attempted),samples_completed=dict(self.completed),
            frames_attempted=dict(self.attempted),frames_completed=dict(self.completed))

class Store:
    def __init__(self,out):
        self.out=Path(out).absolute()
        require(self.out.parent==ROOT/'results' and self.out.name.startswith(TAG+'_') and not self.out.exists()
            and not any(p.is_symlink() for p in (self.out,*self.out.parents)),'exclusive nonsymlink results child')
        self.out.mkdir(); self.started=time.monotonic(); self.bytes=0; self.index=0; self.previous=None
    def guard(self,reserve=0):
        require(time.monotonic()-self.started<WALL,'600s wall cap')
        require(self.bytes+reserve+1024**2<=CAP,'payload cap plus terminal reserve')
        require(shutil.disk_usage(self.out).free>=FLOOR+reserve+1024**2,'disk floor')
    def sync(self):
        fd=os.open(self.out,os.O_DIRECTORY|os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    def event(self,kind,value):
        require(self.index<32,'event cap')
        e=dict(sequence=self.index,previous_sha256=self.previous,kind=kind,value=value); h=impl.digest(e)
        self.guard(len(json.dumps(e,allow_nan=False).encode())*2+1024)
        p=self.out/f'event_{self.index:04d}.json'; impl.write_new(p,{**e,'event_sha256':h}); self.sync()
        self.bytes+=p.stat().st_size; self.index+=1; self.previous=h
    def emit(self,word,value):
        require(value.dtype==np.float32 and value.shape==SHAPE and np.isfinite(value).all(),'original full token array')
        self.guard(value.nbytes+1024); p=self.out/(word+'_terminal_features.npy')
        with p.open('xb') as f: np.save(f,value,allow_pickle=False); f.flush(); os.fsync(f.fileno())
        self.sync(); self.bytes+=p.stat().st_size
        rec=dict(path=str(p),sha256=impl.sha(p),bytes=p.stat().st_size,shape=list(SHAPE),dtype='float32')
        self.event('artifact',dict(word=word,record=rec)); return rec

def load_runtime(value):
    """Only called inside explicit MAIN CUDA run, never saved prepare/verify."""
    import torch
    require(torch.cuda.is_available(),'HOST CUDA required; no CPU neural fallback')
    require(torch.cuda.mem_get_info()[0]>=4*1024**3,'4GiB free CUDA scratch')
    numeric=impl.strict_module(ROOT/'experiments/spectral_reachability'/(impl.NUM+'.py'),impl.NUM_SOURCE_SHA,'runtime_numeric_v5')
    loader=importlib.import_module('experiments.spectral_reachability.'+numeric.owner.LOADER)
    require(impl.sha(loader.__file__)==numeric.owner.LOADER_SHA,'bound CUDA loader')
    models,strict,sources,before=loader.load_cuda(ROOT,value['authority']['model_spec']['loader_manifest'])
    v7=loader.helper(ROOT); _,v5,v6=v7.dependencies(ROOT); v4=v5.import_v4(ROOT)
    def receipts(): return {k:v4.model_receipt(model) for k,model in models.items()}
    def imported():
        return {**v7.imported_sources('dinov2',v6.official_root(ROOT)),
            **v7.imported_sources('app',v4.native_root(ROOT)),**v7.imported_sources('src',v4.native_root(ROOT))}
    return models,strict,sources,before,receipts,imported

def encode(model,rgb):
    import torch
    from app.plan_common.datasets.transforms import make_transforms
    from torch.nn.attention import sdpa_kernel,SDPBackend
    require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8,'raw terminal RGB')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        x=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32)/255.
        t=make_transforms(img_size=224,random_horizontal_flip=False,random_resize_aspect_ratio=(1.,1.),
            random_resize_scale=(1.,1.),reprob=0.,auto_augment=False,motion_shift=False)
        y=model(t(x),is_training=True)['x_norm_patchtokens']
        require(y.is_cuda and y.dtype==torch.float32 and tuple(y.shape)==(1,*SHAPE),'native terminal output')
        return y[0].detach().cpu().numpy().copy()

def source_closure(value,sources):
    seals=value['authority']['model_spec']['loader_manifest']['seals']
    require(sources and all(p in seals and impl.sha(p)==seals[p] for p in sources.values()),'imported source closure')

def run_cuda(value,out,confirm=False):
    require(confirm and out is not None,'MAIN HOST CUDA confirmation and output required')
    verify(value)
    require(not os.environ.get('PYTHONPATH') and not any(k=='pymunk' or k.startswith('pymunk.') for k in sys.modules),
        'fresh process without physics overlay')
    store=Store(out); meter=Meter(store.event,store.guard); handles=[]; models=None; receipts=None; imported=None
    r=dict(schema=TAG+'.report',status='failed',manifest_sha256=impl.digest(value),protocol=protocol(),
        words=[dict(word=w,status='not_run',error=None) for w in WORDS],terminal_features={},errors=[],finalization_errors=[],
        cuda_executed=False,scientific_admission=False,C2_established=False,physical_release=False,task_success=None)
    old_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame): raise TimeoutError('600s cap, no retry')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(WALL)
    try:
        models,strict,sources,before,receipts,imported=load_runtime(value)
        r.update(cuda_executed=True,model_before=before,strict_loads=strict,imported_sources=sources)
        require(set(models)==set(EXPECTED),'exact three frozen modules')
        require(before==value['authority']['model_spec']['loader_manifest']['expected_model_receipts'],'expected model identity')
        source_closure(value,sources)
        def pre(key):
            def hook(module,args):
                # Proprio/predictor are forbidden even if a caller fabricates their shape.
                require(key=='official_encoder','no new proprio/predictor forward')
                require(args[0].is_cuda and tuple(args[0].shape)==(1,3,224,224),'actual CUDA batch1 RGB')
                meter.before(key,int(args[0].shape[0]))
            return hook
        def post(key): return lambda module,args,out:meter.after(key,int(args[0].shape[0]))
        for k,model in models.items():
            handles.append(model.register_forward_pre_hook(pre(k)))
            handles.append(model.register_forward_hook(post(k)))
        real={}
        for i,w in enumerate(WORDS):
            r['words'][i]['status']='attempted'
            rgb=array(value['authority']['terminals'][w]['rgb'],(224,224,3),np.uint8)
            real[w]=encode(models['official_encoder'],rgb)
            r['terminal_features'][w]=store.emit(w,real[w]); r['words'][i]['status']='completed'
        r['words']=metrics(value['authority'],real)
        require(meter.attempted==meter.completed==EXPECTED and meter.pending is None,'complete four forward census')
        r['status']='all4_terminal_rgb_attribution_recorded'
    except Exception as exc:
        msg=type(exc).__name__+': '+str(exc); r['errors'].append(msg)
        for row in r['words']:
            if row['status']=='attempted': row.update(status='failed',error=msg)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old_alarm)
        for h in handles:
            try: h.remove()
            except Exception as exc: r['finalization_errors'].append('hook cleanup: '+str(exc))
        if receipts is not None:
            try:
                r['model_after']=receipts()
                require(r['model_after']==r['model_before']==value['authority']['model_spec']['loader_manifest']['expected_model_receipts'],
                    'model before/after/expected equality')
                r['imported_sources']=imported(); source_closure(value,r['imported_sources'])
            except Exception as exc: r['finalization_errors'].append('model/source: '+str(exc))
        try: verify(value); store.guard()
        except Exception as exc: r['finalization_errors'].append('final authority/resources: '+str(exc))
        if r['errors'] or r['finalization_errors']: r['status']='failed'
        r.update(meter=meter.snapshot(),wall_seconds=time.monotonic()-store.started,journal_terminal_sha256=store.previous)
        size=len(json.dumps(r,allow_nan=False).encode())*2+1024
        require(size<1024**2 and store.bytes+size<=CAP and shutil.disk_usage(store.out).free>=FLOOR+size,'terminal reserve/floor')
        impl.write_new(store.out/'report.json',r); store.sync()
    return r

def consume(value,report,report_sha):
    verify(value); p=Path(report).absolute(); r=impl.read_json(p,report_sha)
    require(r['schema']==TAG+'.report' and r['manifest_sha256']==impl.digest(value) and r['protocol']==protocol(),'report authority')
    require(r['status']=='all4_terminal_rgb_attribution_recorded' and r['errors']==r['finalization_errors']==[]
        and r['cuda_executed'] is True and r['scientific_admission'] is False and r['C2_established'] is False
        and r['physical_release'] is False and r['task_success'] is None,'complete actual CUDA required')
    require(r['model_before']==r['model_after']==value['authority']['model_spec']['loader_manifest']['expected_model_receipts'],'frozen model expected/before/after')
    source_closure(value,r['imported_sources'])
    require(set(r['strict_loads'])=={'predictor','proprio_encoder'} and all(x['exact_match'] is True for x in r['strict_loads'].values()),'strict checkpoint loads')
    require(np.isfinite(r['wall_seconds']) and 0<=r['wall_seconds']<WALL,'wall bound')
    require(set(r['terminal_features'])==set(WORDS),'four token denominator')
    events=sorted(p.parent.glob('event_*.json')); require(len(events)==12,'four intent/return/artifact groups')
    previous=None; meter=Meter(lambda *args:None); records={}
    for i,path in enumerate(events):
        e=impl.read_json(path)
        require(path.name==f'event_{i:04d}.json' and e['sequence']==i and e['previous_sha256']==previous
            and e['event_sha256']==impl.digest({k:v for k,v in e.items() if k!='event_sha256'}),'durable event chain')
        previous=e['event_sha256']; kind=e['kind']; val=e['value']
        expected_kind=('attempted','returned','artifact')[i%3]; require(kind==expected_kind,'fixed actual event order')
        if kind in ('attempted','returned'):
            (meter.before if kind=='attempted' else meter.after)('official_encoder',1)
            require(val==meter.snapshot(),'API/sample/frame event replay')
        else:
            require(val['word']==WORDS[i//3],'fixed four-word artifact order')
            records[val['word']]=val['record']
    require(previous==r['journal_terminal_sha256'] and records==r['terminal_features'] and r['meter']==meter.snapshot()
        and meter.attempted==meter.completed==EXPECTED and meter.pending is None,'complete saved meter/records')
    real={}
    for w in WORDS:
        rec=records[w]
        require(Path(rec['path'])==p.parent/(w+'_terminal_features.npy') and rec['shape']==list(SHAPE) and rec['dtype']=='float32','exact local token record')
        real[w]=array(rec,SHAPE)
    require(r['words']==metrics(value['authority'],real),'complete four-row full token recomputation')
    wanted={x.name for x in events}|{'report.json'}|{w+'_terminal_features.npy' for w in WORDS}
    require(p.name=='report.json' and {x.name for x in p.parent.iterdir()}==wanted,'exact saved output inventory')
    require(sum(x.stat().st_size for x in p.parent.iterdir())<=CAP,'saved output cap')
    verify(value); require(impl.sha(p)==report_sha,'final report drift')
    return dict(schema=TAG+'.consume',status='all4_saved_terminal_rgb_attribution_verified',
        report_sha256=report_sha,manifest_sha256=impl.digest(value),words=r['words'],new_meter=r['meter'],
        reused_goal_meter=value['authority']['gr']['new_meter'],reused_numeric_meter=value['authority']['nr']['meter'],
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False,
        interpretation='single known OOD diagnostic; feature-goal metric is not native coverage or decision benefit')
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','verify','cuda','consume')); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--manifest-sha256'); p.add_argument('--output',type=Path); p.add_argument('--report',type=Path); p.add_argument('--report-sha256'); p.add_argument('--confirm-host-cuda',action='store_true'); p.add_argument('--authority',nargs=9); a=p.parse_args(argv)
    if a.command=='prepare': require(a.authority is not None,'authority order is goal(3), physical(3), numeric(3)'); result=prepare(*a.authority); impl.write_new(a.manifest,result)
    else:
        require(a.manifest_sha256 and impl.sha(a.manifest)==a.manifest_sha256,'actual v5 manifest SHA'); value=impl.read_json(a.manifest,a.manifest_sha256)
        if a.command=='verify': verify(value); result={'status':'v5_verified'}
        elif a.command=='cuda': result=run_cuda(value,a.output,a.confirm_host_cuda)
        else: require(a.report is not None and a.report_sha256 and a.output is not None,'report/output required'); result=consume(value,a.report,a.report_sha256); impl.write_new(a.output,result)
    print(json.dumps({'status':result['status'],'scientific_admission':False},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())

