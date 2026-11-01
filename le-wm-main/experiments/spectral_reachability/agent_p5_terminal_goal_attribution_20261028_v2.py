"""Strict v2 saved join and MAIN-only four-terminal RGB encoder diagnostic."""
from __future__ import annotations
import argparse, importlib, json, os, signal, shutil, time
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v1 as old

TAG='agent_p5_terminal_goal_attribution_20261028_v2'; ROOT=old.ROOT; WORDS=old.WORDS; SHAPE=old.SHAPE
CAP,FLOOR,WALL=32*1024**2,200*1024**2,600; EXPECTED={'official_encoder':4,'proprio_encoder':0,'predictor':0}
V1_SHA='f2432a33b1dd7a7f7270e650aeefec1873d265bb823036db104597222aca0c78'
GOAL_SOURCE='agent_p5_frozen_goal_scoring_20261028_v1'; GOAL_SOURCE_SHA='d987d0146627ad6067c11665d0a2e86f27d9fbda86f400db0ccf3d2d9f970e9c'; GOAL_PREP='aecd86b684ab664254a695c4ca10f131a90cb8e8f77e651eae8864b4ed2b07ac'; GOAL_REPORT='a732ba40426ff56d3d82db59c30759f7c2db7a54f970d33bc6dc6412c81407e5'; GOAL_RECEIPT='22b8319b443d6dcf5d475e0149df9df3909bcb7106db57fbfb01e77778bd76d4'
NUM='agent_p5_contact_action_response_20261028_v1'; NUM_SOURCE_SHA='bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930'; NUM_MANIFEST_SHA='572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6'; NUM_REPORT_SHA='4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f'; NUM_RECEIPT_SHA='247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3'
PHYS='agent_p5_contact_action_words_physical_20261028_v1'; PHYS_SOURCE_SHA='0a00d0f7864f03fc8e9ddf1fa7cef749a1d57f6c977357ce78c309262aa4556f'; PHYS_MANIFEST_SHA='46f8cf63d045123edfa61b257ad11bd94b3741b47dfe2244463c5c5096df0ab7'; PHYS_REPORT_SHA='a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633'; PHYS_RECEIPT_SHA='f6cd1f53d7537a6292c49647a3e20cfd61caf8a4b5e5e9d395223fc5845a39ee'; PHYS_CONSUMER_SHA='8bb51bddb3b2836d43ce5a51e0162b05d3f98ad2aa7f6cb4de23739c8b4fe7d5'
require,sha,digest,read_json,write_new=old.require,old.sha,old.digest,old.read_json,old.write_new

def protocol(): return dict(schema=TAG,words=list(WORDS),denominator=4,frame=20,new_api=EXPECTED,new_forward_samples=EXPECTED,goal_reencoded=False,predictor_rerun=False,score='float64 terminal-v-goal patch MSE',native_metrics_separate=True,tie_rule='none; fixed four-word order retained',known_single_OOD_root=True,scientific_admission=False,C2_established=False,task_success=None,physics=0,FLOPs=None,output_cap_bytes=CAP,free_floor_bytes=FLOOR,wall_seconds=WALL)
def own(): return {str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md'))}
def strict_module(path,expected,name):
    require(sha(path)==expected,name+' source SHA'); spec=importlib.util.spec_from_file_location(TAG+'_'+name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def exact_authorities(gp,gr,gc,pm,pr,pc,sm,sr,sc):
    require((sha(gp),sha(gr),sha(gc))==(GOAL_PREP,GOAL_REPORT,GOAL_RECEIPT),'goal authority SHA')
    require((sha(pm),sha(pr),sha(pc))==(NUM_MANIFEST_SHA,NUM_REPORT_SHA,NUM_RECEIPT_SHA),'numeric authority SHA')
    require((sha(sm),sha(sr),sha(sc))==(PHYS_MANIFEST_SHA,PHYS_REPORT_SHA,PHYS_RECEIPT_SHA),'physical authority SHA')
    gm,grv,gcv=read_json(gp,GOAL_PREP),read_json(gr,GOAL_REPORT),read_json(gc,GOAL_RECEIPT)
    nm,nrv,ncv=read_json(pm,NUM_MANIFEST_SHA),read_json(pr,NUM_REPORT_SHA),read_json(pc,NUM_RECEIPT_SHA)
    smv,srv,scv=read_json(sm,PHYS_MANIFEST_SHA),read_json(sr,PHYS_REPORT_SHA),read_json(sc,PHYS_RECEIPT_SHA)
    goalmod=strict_module(ROOT/'experiments/spectral_reachability'/(GOAL_SOURCE+'.py'),GOAL_SOURCE_SHA,'goal')
    nummod=strict_module(ROOT/'experiments/spectral_reachability'/(NUM+'.py'),NUM_SOURCE_SHA,'numeric')
    wrapper=strict_module(ROOT/'experiments/spectral_reachability/consume_contact_words_MAIN_20261028_v1.py',PHYS_CONSUMER_SHA,'physical_consumer')
    require(goalmod.consume(gm,gr,GOAL_REPORT)==gcv,'goal report/receipt exact return')
    require(nummod.consume(nm,pr,NUM_REPORT_SHA)==ncv,'numeric report/receipt exact return')
    require(wrapper.consume(sm,PHYS_MANIFEST_SHA,sr,PHYS_REPORT_SHA)==scv,'physical report/receipt exact return')
    require(gcv['status']=='all4_saved_goal_scores_verified' and ncv['status']=='all4_saved_public_action_words_verified' and scv['status']=='all4_saved_native_words_verified','all strict statuses')
    goalrec=grv['goal_patch']; require(goalrec['shape']==list(SHAPE) and sha(goalrec['path'])==goalrec['sha256'] and Path(goalrec['path']).stat().st_size==goalrec['bytes'],'goal patch SHA/bytes')
    prediction={w:nrv['artifacts'][f'{w}_1_visual_out'] for w in WORDS}
    for rec in prediction.values(): require(rec['shape']==[3,*SHAPE] and sha(rec['path'])==rec['sha256'] and Path(rec['path']).stat().st_size==rec['bytes'],'prediction SHA/bytes')
    pr=srv; require(pr['status']=='all4_native_word_traces_recorded' and pr['global_error'] is None and [x['arm'] for x in pr['rows']]==list(WORDS),'physical rows')
    terminals={}
    for row in pr['rows']:
        rec=next(x for x in row['records'] if x['frame']==20); base=Path(sr).parent/row['arm']/'frame_20'; files=rec['files']
        members={k:base/k for k in ('rgb.npy','full_public_state.json','transition.json')}
        for k,p in members.items(): require(sha(p)==files[k]['sha256'] and p.stat().st_size==files[k]['bytes'],'terminal SHA/bytes')
        terminals[row['arm']]=dict(rgb=dict(path=str(members['rgb.npy'].resolve()),sha256=files['rgb.npy']['sha256'],bytes=files['rgb.npy']['bytes']),native=dict(state=rec['state'],proprio=rec['proprio'],state_file=str(members['full_public_state.json'].resolve()),transition_file=str(members['transition.json'].resolve()),state_sha=files['full_public_state.json']['sha256'],transition_sha=files['transition.json']['sha256']))
    return dict(gm=gm,gr=gcv,greport=grv,nm=nm,nr=ncv,nreport=nrv,sm=smv,sr=scv,sreport=srv,goalrec=goalrec,prediction=prediction,terminals=terminals,model_spec=nm['model_spec'],paths=dict(goal_prepare=str(Path(gp).resolve()),goal_report=str(Path(gr).resolve()),goal_receipt=str(Path(gc).resolve()),num_manifest=str(Path(pm).resolve()),num_report=str(Path(pr).resolve()),num_receipt=str(Path(pc).resolve()),phys_manifest=str(Path(sm).resolve()),phys_report=str(Path(sr).resolve()),phys_receipt=str(Path(sc).resolve())))
def prepare(*paths):
    a=exact_authorities(*paths); b={k:v for k,v in a['paths'].items()}; b.update({str(ROOT/'experiments/spectral_reachability'/f'{GOAL_SOURCE}.py'):GOAL_SOURCE_SHA,str(ROOT/'experiments/spectral_reachability'/f'{NUM}.py'):NUM_SOURCE_SHA,str(ROOT/'experiments/spectral_reachability'/f'{PHYS}.py'):PHYS_SOURCE_SHA,str(ROOT/'experiments/spectral_reachability/consume_contact_words_MAIN_20261028_v1.py'):PHYS_CONSUMER_SHA,str(ROOT/'experiments/spectral_reachability'/f'{old.TAG}.py'):V1_SHA}); b.update(own()); b[str(a['goalrec']['path'])]=a['goalrec']['sha256']; b.update({r['path']:r['sha256'] for r in a['prediction'].values()}); b.update({x['rgb']['path']:x['rgb']['sha256'] for x in a['terminals'].values()}); require(all(sha(p)==h for p,h in b.items()),'complete prepare bindings')
    return dict(schema=TAG,status='prepared_strict_saved_join',protocol=protocol(),authority=a,bindings=b,scientific_admission=False,C2_established=False,task_success=None,physical_release=False)
def verify(m):
    require(m['schema']==TAG and m['status']=='prepared_strict_saved_join' and m['protocol']==protocol(),'manifest contract'); require(all(sha(p)==h for p,h in m['bindings'].items()),'binding drift'); a=m['authority']; paths=[a['paths'][k] for k in ('goal_prepare','goal_report','goal_receipt','phys_manifest','phys_report','phys_receipt','num_manifest','num_report','num_receipt')]; fresh=prepare(*paths); require(fresh==m,'full authority join recomputation')
def metric(goal,pred,real,native):
    require(goal.shape==SHAPE and goal.dtype==np.float32 and np.isfinite(goal).all() and set(pred)==set(real)==set(native)==set(WORDS),'four metric inputs')
    out=[]
    for w in WORDS:
        p,r=pred[w],real[w]; require(p.shape==r.shape==(3,*SHAPE) and p.dtype==r.dtype==np.float32 and np.isfinite(p).all() and np.isfinite(r).all(),'token ABI')
        out.append(dict(word=w,predicted_goal_mse=float(np.mean((p[-1].astype(np.float64)-goal.astype(np.float64))**2)),real_terminal_goal_mse=float(np.mean((r[-1].astype(np.float64)-goal.astype(np.float64))**2)),native=native[w],task_success=None))
    return out
class Meter:
    def __init__(self,event): self.a=0; self.c=0; self.pending=False; self.event=event
    def before(self): require(self.a<4 and not self.pending,'encoder predispatch cap'); self.a+=1; self.pending=True; self.event('attempted',self.snapshot())
    def after(self): require(self.pending,'encoder return pairing'); self.c+=1; self.pending=False; self.event('returned',self.snapshot())
    def snapshot(self): return dict(attempted={'official_encoder':self.a},completed={'official_encoder':self.c},pending=self.pending,samples_attempted={'official_encoder':self.a},samples_completed={'official_encoder':self.c},frames_attempted={'official_encoder':self.a},frames_completed={'official_encoder':self.c})
def run_cuda(m,out,confirm=False):
    require(confirm,'MAIN HOST CUDA confirmation'); verify(m); out=Path(out).absolute(); require(out.parent==ROOT/'results' and not out.exists(),'exclusive output'); out.mkdir(); events=[]; prev=None; started=time.monotonic(); meter=Meter(lambda k,v:events.append((k,v))); report=dict(schema=TAG+'.report',status='failed',manifest_sha256=digest(m),protocol=protocol(),words=[],errors=[],finalization_errors=[],cuda_executed=False,scientific_admission=False,C2_established=False,task_success=None,physical_release=False)
    models=None; handles=[]
    try:
        import torch; require(torch.cuda.is_available(),'CUDA required; no CPU fallback'); loader=importlib.import_module('experiments.spectral_reachability.'+old.numeric.owner.LOADER); models,strict,sources,before=loader.load_cuda(ROOT,m['authority']['model_spec']['loader_manifest']); require(set(models)=={'official_encoder','proprio_encoder','predictor'},'model set'); v7=loader.helper(ROOT); _,v5,_=v7.dependencies(ROOT); v4=v5.import_v4(ROOT)
        report.update(cuda_executed=True,model_before=before,strict_loads=strict,imported_sources=sources)
        def pre(mod,args): require(args[0].is_cuda and tuple(args[0].shape)==(1,3,224,224),'encoder shape'); meter.before()
        def post(mod,args,out): meter.after()
        handles=[models['official_encoder'].register_forward_pre_hook(pre),models['official_encoder'].register_forward_hook(post)]
        goal=np.load(m['authority']['goalrec']['path'],allow_pickle=False); pred={w:np.load(r['path'],allow_pickle=False) for w,r in m['authority']['prediction'].items()}; real={w:np.load(m['authority']['terminals'][w]['rgb']['path'],allow_pickle=False) for w in WORDS}; features={}
        for w in WORDS: features[w]=old.encode(models['official_encoder'],real[w]); np.save(out/(w+'_terminal_features.npy'),features[w],allow_pickle=False)
        native={w:m['authority']['terminals'][w]['native'] for w in WORDS}; report['words']=metric(goal,pred,features,native); report['terminal_features']={w:dict(path=str((out/(w+'_terminal_features.npy')).resolve()),sha256=sha(out/(w+'_terminal_features.npy')),bytes=(out/(w+'_terminal_features.npy')).stat().st_size,shape=list(SHAPE),dtype='float32') for w in WORDS}; report['status']='all4_terminal_rgb_attribution_recorded'; require(meter.a==meter.c==4 and not meter.pending,'complete encoder budget')
    except Exception as e: report['errors'].append(type(e).__name__+': '+str(e))
    finally:
        for h in handles:
            try:h.remove()
            except Exception as e: report['finalization_errors'].append(str(e))
        if models is not None:
            try: report['model_after']={k:v4.model_receipt(v) for k,v in models.items()}; require(report['model_after']==report.get('model_before'),'model before/after')
            except Exception as e: report['finalization_errors'].append(str(e))
        report['meter']=meter.snapshot(); report['wall_seconds']=time.monotonic()-started; report['journal_terminal_sha256']=None
        if report['errors'] or report['finalization_errors']: report['status']='failed'
        write_new(out/'report.json',report)
    return report
def consume(m,report,report_sha):
    verify(m); r=read_json(report,report_sha); require(r['schema']==TAG+'.report' and r['status']=='all4_terminal_rgb_attribution_recorded' and r['errors']==r['finalization_errors']==[] and r['model_after']==r['model_before'] and r['model_before']==m['authority']['model_spec']['loader_manifest']['expected_model_receipts'] and r['meter']['completed']=={'official_encoder':4},'strict v2 report');
    require(r['imported_sources'] and all(p in m['authority']['model_spec']['loader_manifest']['seals'] and sha(p)==m['authority']['model_spec']['loader_manifest']['seals'][p] for p in r['imported_sources'].values()),'imported source closure'); goal=np.load(m['authority']['goalrec']['path'],allow_pickle=False); pred={w:np.load(x['path'],allow_pickle=False) for w,x in m['authority']['prediction'].items()}; real={w:np.load(Path(report).parent/(w+'_terminal_features.npy'),allow_pickle=False) for w in WORDS}; expected=metric(goal,pred,real,{w:m['authority']['terminals'][w]['native'] for w in WORDS}); require(r['words']==expected,'full token metric recomputation');
    for w,x in r['terminal_features'].items(): require(sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes'],'terminal feature SHA/bytes')
    return dict(schema=TAG+'.consume',status='all4_saved_terminal_rgb_attribution_verified',report_sha256=sha(report),words=r['words'],meter=r['meter'],scientific_admission=False,C2_established=False,task_success=None)
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('prepare','verify','cuda','consume')); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--manifest-sha256'); p.add_argument('--output',type=Path); p.add_argument('--report',type=Path); p.add_argument('--report-sha256'); p.add_argument('--confirm-host-cuda',action='store_true'); p.add_argument('--authority',nargs=9); a=p.parse_args(argv)
    if a.command=='prepare': require(a.authority is not None,'nine authority paths required'); value=prepare(*a.authority); write_new(a.manifest,value)
    else: require(a.manifest_sha256 and sha(a.manifest)==a.manifest_sha256,'actual manifest SHA'); m=read_json(a.manifest,a.manifest_sha256); verify(m) if a.command=='verify' else None; value={'status':'verified'} if a.command=='verify' else (run_cuda(m,a.output,a.confirm_host_cuda) if a.command=='cuda' else consume(m,a.report,a.report_sha256));
    if a.command=='consume': write_new(a.output,value)
    print(json.dumps({'status':value['status'],'scientific_admission':False},sort_keys=True))
if __name__=='__main__': raise SystemExit(main())
