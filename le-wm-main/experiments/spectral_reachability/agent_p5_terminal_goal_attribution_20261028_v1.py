"""Saved physical terminal RGB versus sealed JEPA goal attribution.

MAIN HOST CUDA only: four terminal RGB encodes, no goal re-encode, predictor,
physics, private publication, outcome selection, or CPU neural fallback.
"""
from __future__ import annotations
import argparse, importlib, json, os, signal, sys, time
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p5_contact_action_response_20261028_v1 as numeric

TAG='agent_p5_terminal_goal_attribution_20261028_v1'; ROOT=Path(__file__).resolve().parents[2]
WORDS=('xy','yx','x_minus_x','zero_zero'); SHAPE=(256,384); CAP=32*1024**2; FLOOR=200*1024**2; WALL=600
GOAL_SOURCE='agent_p5_frozen_goal_scoring_20261028_v1'; GOAL_SOURCE_SHA='d987d0146627ad6067c11665d0a2e86f27d9fbda86f400db0ccf3d2d9f970e9c'
GOAL_PREP='aecd86b684ab664254a695c4ca10f131a90cb8e8f77e651eae8864b4ed2b07ac'; GOAL_REPORT='a732ba40426ff56d3d82db59c30759f7c2db7a54f970d33bc6dc6412c81407e5'; GOAL_RECEIPT='22b8319b443d6dcf5d475e0149df9df3909bcb7106db57fbfb01e77778bd76d4'
PHYS='agent_p5_contact_action_words_physical_20261028_v1'; PHYS_SOURCE_SHA='0a00d0f7864f03fc8e9ddf1fa7cef749a1d57f6c977357ce78c309262aa4556f'; PHYS_MANIFEST_SHA='46f8cf63d045123edfa61b257ad11bd94b3741b47dfe2244463c5c5096df0ab7'; PHYS_REPORT_SHA='a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633'; PHYS_RECEIPT_SHA='f6cd1f53d7537a6292c49647a3e20cfd61caf8a4b5e5e9d395223fc5845a39ee'
NUM='agent_p5_contact_action_response_20261028_v1'; NUM_SOURCE_SHA='bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930'; NUM_MANIFEST_SHA='572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6'; NUM_REPORT_SHA='4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f'; NUM_RECEIPT_SHA='247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3'
EXPECTED={'official_encoder':4,'proprio_encoder':0,'predictor':0}

require,sha,digest,read_json,write_new=numeric.require,numeric.sha,numeric.digest,numeric.read_json,numeric.write_new

def protocol():
    return dict(schema=TAG,words=list(WORDS),denominator=4,frame=20,shape=list(SHAPE),
      new_api=EXPECTED,new_forward_samples=EXPECTED,new_context_frames=EXPECTED,
      reused_goal_encoder=1,reused_predictor_forward=0,goal_reencoded=False,
      scoring='mean(float64(terminal_RGB_feature)-float64(sealed_goal_feature))^2',
      prediction_error='sealed model terminal prediction minus sealed goal feature MSE; no new predictor',
      native_metric='native terminal coverage/reward/done and full double state retained separately',
      tie_rule='none; diagnostic rows preserve fixed word order, no selected word',
      raw_rgb='uint8 terminal frame20, exact file SHA; no RGB attribution claim',
      private_physics_not_planner=True,known_single_OOD_root=True,fresh=False,
      scientific_admission=False,C2_established=False,task_success=None,physical_release=False,
      AD=0,backward=0,JVP=0,VJP=0,SVD=0,physics=0,FLOPs=None,
      output_cap_bytes=CAP,free_floor_bytes=FLOOR,wall_seconds=WALL)

def self_bindings():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md'))}

def _goal_paths():
    return (ROOT/'results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json',
            ROOT/'results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json',
            ROOT/'results/agent_p5_frozen_goal_scoring_20261028_v1_consume_MAIN01.json')

def prepare(goal_prepare,goal_report,goal_receipt,phys_manifest,phys_report,phys_receipt,num_manifest,num_report,num_receipt):
    require(sha(ROOT/'experiments/spectral_reachability'/f'{GOAL_SOURCE}.py')==GOAL_SOURCE_SHA,'sealed scorer source')
    require((sha(goal_prepare),sha(goal_report),sha(goal_receipt))==(GOAL_PREP,GOAL_REPORT,GOAL_RECEIPT),'actual scorer authorities')
    require((sha(phys_manifest),sha(phys_report),sha(phys_receipt))==(PHYS_MANIFEST_SHA,PHYS_REPORT_SHA,PHYS_RECEIPT_SHA),'actual physical authorities')
    gm,gr,gc=(read_json(goal_prepare,GOAL_PREP),read_json(goal_report,GOAL_REPORT),read_json(goal_receipt,GOAL_RECEIPT))
    require(gc['status']=='all4_saved_goal_scores_verified' and gr['status']=='all4_public_goal_scores_recorded','strict scorer complete')
    require(gr['goal_patch']['shape']==list(SHAPE) and sha(gr['goal_patch']['path'])==gr['goal_patch']['sha256'],'sealed goal feature')
    require(read_json(phys_receipt,PHYS_RECEIPT_SHA)['status']=='all4_saved_native_words_verified','strict physical complete')
    nm,nr,nc=read_json(num_manifest,NUM_MANIFEST_SHA),read_json(num_report,NUM_REPORT_SHA),read_json(num_receipt,NUM_RECEIPT_SHA)
    require(sha(ROOT/'experiments/spectral_reachability'/f'{NUM}.py')==NUM_SOURCE_SHA and nc['status']=='all4_saved_public_action_words_verified','strict numeric complete')
    require(nr['status']=='public_native_action_response_recorded' and nr['errors']==nr['finalization_errors']==[],'numeric report')
    prediction={w:nr['artifacts'][f'{w}_1_visual_out'] for w in WORDS}
    for rec in prediction.values(): require(rec['shape']==[3,*SHAPE] and sha(rec['path'])==rec['sha256'],'prediction token closure')
    pr=read_json(phys_report,PHYS_REPORT_SHA); require(pr['status']=='all4_native_word_traces_recorded' and pr['global_error'] is None,'physical report')
    rows=pr['rows']; require([r['arm'] for r in rows]==list(WORDS) and all(r['status']=='completed' for r in rows),'all4 physical rows')
    terminal={}
    for row in rows:
        rec=[x for x in row['records'] if x['frame']==20][0]; base=Path(phys_report).parent/row['arm']/'frame_20'
        f=rec['files']; rgb=base/'rgb.npy'; state=base/'full_public_state.json'; trans=base/'transition.json'
        for name,p in [('rgb.npy',rgb),('full_public_state.json',state),('transition.json',trans)]:
            require(sha(p)==f[name]['sha256'] and p.stat().st_size==f[name]['bytes'],'terminal file closure')
        native_public=read_json(state); transition=read_json(trans)
        terminal[row['arm']]=dict(rgb=dict(path=str(rgb.resolve()),sha256=f['rgb.npy']['sha256'],bytes=f['rgb.npy']['bytes'],shape=[224,224,3],dtype='uint8'),
            state=dict(path=str(state.resolve()),sha256=f['full_public_state.json']['sha256'],bytes=f['full_public_state.json']['bytes']),
            transition=dict(path=str(trans.resolve()),sha256=f['transition.json']['sha256'],bytes=f['transition.json']['bytes']),
            native=dict(state=rec['state'],proprio=rec['proprio'],bodies=native_public['bodies'],env=native_public['env'],transition=transition))
    bindings={str(Path(x).resolve()):h for x,h in ((goal_prepare,GOAL_PREP),(goal_report,GOAL_REPORT),(goal_receipt,GOAL_RECEIPT),(phys_manifest,PHYS_MANIFEST_SHA),(phys_report,PHYS_REPORT_SHA),(phys_receipt,PHYS_RECEIPT_SHA))}
    bindings.update({str(ROOT/'experiments/spectral_reachability'/f'{PHYS}.py'):PHYS_SOURCE_SHA,str(ROOT/'experiments/spectral_reachability'/f'{GOAL_SOURCE}.py'):GOAL_SOURCE_SHA,str(ROOT/'experiments/spectral_reachability'/f'{NUM}.py'):NUM_SOURCE_SHA})
    bindings.update({str(Path(x).resolve()):h for x,h in ((num_manifest,NUM_MANIFEST_SHA),(num_report,NUM_REPORT_SHA),(num_receipt,NUM_RECEIPT_SHA))})
    bindings.update(self_bindings()); require(all(sha(p)==h for p,h in bindings.items()),'authority drift')
    return dict(schema=TAG,status='prepared_terminal_rgb_only',protocol=protocol(),goal=dict(report=gr,receipt=gc),
      prediction=dict(manifest=nm,report=nr,receipt=nc,artifacts=prediction),model_spec=nm['model_spec'],
      physical=dict(report=pr,receipt=read_json(phys_receipt,PHYS_RECEIPT_SHA)),terminal=terminal,bindings=bindings,
      scientific_admission=False,C2_established=False,task_success=None,physical_release=False)

def verify(m):
    require(m['schema']==TAG and m['status']=='prepared_terminal_rgb_only' and m['protocol']==protocol(),'manifest contract')
    for p,h in m['bindings'].items(): require(sha(p)==h,'sealed binding drift')
    for w,rec in m['prediction']['artifacts'].items(): require(sha(rec['path'])==rec['sha256'],'prediction drift')
    for w,rec in m['terminal'].items():
        for key in ('rgb','state','transition'): require(sha(rec[key]['path'])==rec[key]['sha256'],'terminal drift')
    return True

def metrics(goal,pred,real,native):
    goal=np.asarray(goal); require(goal.shape==SHAPE and goal.dtype==np.float32 and np.isfinite(goal).all(),'goal ABI')
    require(set(pred)==set(WORDS) and set(real)==set(WORDS) and set(native)==set(WORDS),'complete four-word metric inputs')
    out=[]
    for w in WORDS:
        p=np.asarray(pred[w]); r=np.asarray(real[w]); require(p.shape==r.shape==(3,*SHAPE) and p.dtype==r.dtype==np.float32 and np.isfinite(p).all() and np.isfinite(r).all(),'prediction/real token ABI')
        pm=float(np.mean((p[-1].astype(np.float64)-goal.astype(np.float64))**2)); rm=float(np.mean((r[-1].astype(np.float64)-goal.astype(np.float64))**2))
        out.append(dict(word=w,predicted_goal_mse=pm,real_terminal_goal_mse=rm,model_prediction_minus_real=pm-rm,
          native= native[w],task_success=None))
    return out

class Meter:
    def __init__(self,event): self.attempted={'official_encoder':0}; self.completed={'official_encoder':0}; self.event=event; self.pending=False
    def before(self,batch): require(batch==1 and not self.pending and self.attempted['official_encoder']<4,'encoder cap'); self.attempted['official_encoder']+=1; self.pending=True; self.event('attempted',self.snapshot())
    def after(self,batch): require(batch==1 and self.pending,'encoder return'); self.completed['official_encoder']+=1; self.pending=False; self.event('returned',self.snapshot())
    def snapshot(self): return dict(attempted=dict(self.attempted),completed=dict(self.completed),pending=self.pending,samples_attempted=dict(self.attempted),samples_completed=dict(self.completed),frames_attempted=dict(self.attempted),frames_completed=dict(self.completed))

class Store:
    def __init__(self,out):
        self.out=Path(out).absolute(); require(self.out.parent==ROOT/'results' and self.out.name.startswith(TAG+'_') and not self.out.exists(),'exclusive output'); self.out.mkdir(); self.n=0; self.prev=None; self.bytes=0; self.start=time.monotonic()
    def guard(self): require(time.monotonic()-self.start<WALL and self.bytes+1024**2<CAP and __import__('shutil').disk_usage(self.out).free>=FLOOR+1024**2,'resource cap')
    def event(self,k,v): self.guard(); e=dict(sequence=self.n,previous_sha256=self.prev,kind=k,value=v); h=digest(e); write_new(self.out/f'event_{self.n:04d}.json',{**e,'event_sha256':h}); self.prev=h; self.n+=1
    def array(self,name,a): self.guard(); p=self.out/name; a=np.asarray(a); require(a.dtype==np.float32 and a.shape==SHAPE,'full token shape'); np.save(p,a,allow_pickle=False); rec=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,shape=list(SHAPE),dtype='float32'); self.bytes+=p.stat().st_size; return rec

def encode(model,rgb):
    import torch
    from app.plan_common.datasets.transforms import make_transforms
    from torch.nn.attention import sdpa_kernel,SDPBackend
    require(rgb.dtype==np.uint8 and rgb.shape==(224,224,3),'terminal RGB ABI')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        x=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32)/255.; t=make_transforms(img_size=224,random_horizontal_flip=False,random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.),reprob=0.,auto_augment=False,motion_shift=False); y=model(t(x),is_training=True)['x_norm_patchtokens']; require(y.is_cuda and y.dtype==torch.float32 and tuple(y.shape)==(1,*SHAPE),'native encoder output'); return y[0].detach().cpu().numpy().copy()

def run_cuda(m,out,main_authorized=False):
    require(main_authorized,'MAIN HOST CUDA authorization'); goal=np.load(m['goal']['report']['goal_patch']['path'],allow_pickle=False); require(goal.shape==SHAPE and goal.dtype==np.float32,'sealed goal file'); store=Store(out); r=dict(schema=TAG+'.report',status='failed',manifest_sha256=digest(m),protocol=protocol(),words=[],errors=[],finalization_errors=[],cuda_executed=False,scientific_admission=False,C2_established=False,task_success=None,physical_release=False)
    meter=Meter(store.event); models=None; handles=[]; old=signal.getsignal(signal.SIGALRM); signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('wall cap'))); signal.alarm(WALL)
    try:
        import torch; require(torch.cuda.is_available(),'CUDA required')
        loader=importlib.import_module('experiments.spectral_reachability.'+numeric.base.owner.LOADER); models,strict,sources,before=loader.load_cuda(ROOT,m['model_spec']['loader_manifest']); require(set(models)=={'official_encoder','proprio_encoder','predictor'},'sealed models')
        r.update(cuda_executed=True,model_before=before,strict_loads=strict,imported_sources=sources)
        def pre(mod,args): require(args[0].is_cuda and tuple(args[0].shape)==(1,3,224,224),'encoder input'); meter.before(1)
        def post(mod,args,out): meter.after(1)
        handles=[models['official_encoder'].register_forward_pre_hook(pre),models['official_encoder'].register_forward_hook(post)]
        real={w:np.load(m['terminal'][w]['rgb']['path'],allow_pickle=False) for w in WORDS}; pred={w:np.load(m['prediction']['artifacts'][w]['path'],allow_pickle=False) for w in WORDS}
        real_features={}; feature_records={}
        for w in WORDS:
            real_features[w]=encode(models['official_encoder'],real[w])
            feature_records[w]=store.array(w+'_terminal_features.npy',real_features[w])
        native={w:dict(state=m['terminal'][w]['native']['state'],proprio=m['terminal'][w]['native']['proprio']) for w in WORDS}; r['words']=metrics(goal,pred,real_features,native); r['terminal_features']=feature_records; r['status']='all4_terminal_rgb_attribution_recorded'
        require(meter.attempted==meter.completed=={'official_encoder':4} and not meter.pending,'complete encoder meter')
    except Exception as e: r['errors'].append(type(e).__name__+': '+str(e))
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old)
        for h in handles:
            try:h.remove()
            except Exception as e:r['finalization_errors'].append(str(e))
        if models is not None:
            try:r['model_after']={k:numeric.base.owner.helpers()[1].model_receipt(v) for k,v in models.items()}
            except Exception as e:r['finalization_errors'].append(str(e))
        r.update(meter=meter.snapshot(),wall_seconds=time.monotonic()-store.start,journal_terminal_sha256=store.prev)
        write_new(store.out/'report.json',r)
    return r

def consume(m,report_path,report_sha):
    verify(m); r=read_json(report_path,report_sha)
    require(r['schema']==TAG+'.report' and r['status']=='all4_terminal_rgb_attribution_recorded'
      and r['errors']==r['finalization_errors']==[] and r['cuda_executed'] is True
      and r['scientific_admission'] is False and r['C2_established'] is False and r['task_success'] is None,'complete attribution report')
    require(r['manifest_sha256']==digest(m) and r['protocol']==protocol() and r['meter']['completed']=={'official_encoder':4},'report authority/meter')
    require(set(r['terminal_features'])==set(WORDS),'all four terminal feature records')
    for w,rec in r['terminal_features'].items(): require(sha(rec['path'])==rec['sha256'] and rec['shape']==list(SHAPE),'terminal token closure')
    events=sorted(Path(report_path).parent.glob('event_*.json')); require(len(events)==8,'four encoder event pairs')
    prev=None; attempts=returns=0
    for i,p in enumerate(events):
        e=read_json(p); require(e['sequence']==i and e['previous_sha256']==prev and e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}),'event chain')
        prev=e['event_sha256']
        if e['kind']=='attempted': attempts+=1
        elif e['kind']=='returned': returns+=1
        else: require(False,'unknown event')
    require(attempts==returns==4 and prev==r['journal_terminal_sha256'],'event census')
    for w in WORDS:
        rec=r['words'][WORDS.index(w)]; saved=metrics(np.load(m['goal']['report']['goal_patch']['path'],allow_pickle=False),
          {x:np.load(m['prediction']['artifacts'][x]['path'],allow_pickle=False) for x in WORDS},
          {x:np.load(Path(report_path).parent/(x+'_terminal_features.npy'),allow_pickle=False) for x in WORDS},
          {x:m['terminal'][x]['native'] for x in WORDS})[WORDS.index(w)]
        require(rec==saved,'deterministic per-word attribution')
    wanted={p.name for p in events}|{'report.json'}|{w+'_terminal_features.npy' for w in WORDS}
    require({p.name for p in Path(report_path).parent.iterdir()}==wanted,'exact output inventory')
    return dict(schema=TAG+'.consume',status='all4_saved_terminal_rgb_attribution_verified',report_sha256=sha(report_path),
      manifest_sha256=digest(m),words=r['words'],meter=r['meter'],scientific_admission=False,C2_established=False,task_success=None,
      interpretation='diagnostic model prediction versus real RGB feature-goal distance; not native coverage equivalence or task benefit')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','verify','cuda','consume')); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--manifest-sha256'); p.add_argument('--goal-prepare',type=Path); p.add_argument('--goal-report',type=Path); p.add_argument('--goal-receipt',type=Path); p.add_argument('--physical-manifest',type=Path); p.add_argument('--physical-report',type=Path); p.add_argument('--physical-receipt',type=Path); p.add_argument('--prediction-manifest',type=Path); p.add_argument('--prediction-report',type=Path); p.add_argument('--prediction-receipt',type=Path); p.add_argument('--output',type=Path); p.add_argument('--report',type=Path); p.add_argument('--report-sha256'); p.add_argument('--confirm-host-cuda',action='store_true'); a=p.parse_args(argv)
    if a.command=='prepare':
        require(all(x is not None for x in (a.goal_prepare,a.goal_report,a.goal_receipt,a.physical_manifest,a.physical_report,a.physical_receipt,a.prediction_manifest,a.prediction_report,a.prediction_receipt)),'all external authorities required')
        value=prepare(a.goal_prepare,a.goal_report,a.goal_receipt,a.physical_manifest,a.physical_report,a.physical_receipt,a.prediction_manifest,a.prediction_report,a.prediction_receipt); write_new(a.manifest,value)
    else:
        require(a.manifest_sha256,'actual manifest SHA'); m=read_json(a.manifest,a.manifest_sha256)
        if a.command=='verify': verify(m); value={'status':'verified'}
        elif a.command=='cuda': value=run_cuda(m,a.output,a.confirm_host_cuda)
        else: require(a.report is not None and a.report_sha256 and a.output is not None,'actual report/output required'); value=consume(m,a.report,a.report_sha256); write_new(a.output,value)
    print(json.dumps(value if a.command=='verify' else {'status':value['status'],'scientific_admission':False},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
