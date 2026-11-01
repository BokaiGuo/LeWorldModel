"""Same-time frame20 native raster diagnostic; five fixed HOST CUDA encodes, no physics."""
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

TAG='agent_p5_same_time_frame20_20261028_v4'
ROOT=owner.ROOT
OWNER_SHA='aece72c75efa24f8c5e2df12aff38a954df31307a0476edf28be2295b1320025'
KERNEL_SOURCE_SHA='11e97cd7ca2da60a9c625b524bfd564b3bf6976459ff9394605bda0c19978996'
WORDS=owner.WORDS
STATES=(*WORDS,'goal')
KEYS=tuple(w+'__frame20' for w in WORDS)+('goal__native',)
SHAPE=(256,384)
EXPECTED=dict(official_encoder=5,proprio_encoder=0,predictor=0)
CAP,FLOOR,WALL=32*1024**2,200*1024**2,600
require,sha,read_json,write_new,digest=owner.require,owner.impl.sha,owner.impl.read_json,owner.impl.write_new,owner.impl.digest


PHYSICAL_SHA='a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633'
PHYSICAL_PREP_SHA='46f8cf63d045123edfa61b257ad11bd94b3741b47dfe2244463c5c5096df0ab7'
GOAL_PREP_SHA='c2ecf01381cf339bf8f7e89fa428f4f62ac46075b10458ee7cfc8deac661941d'
MODEL_PREP_SHA='6c0fa9420bfed269880279ff6ce53aef479364870f12278796fe6016825627a1'
PREDICTION_PREP_SHA='79c3c2f1aa1b8f1b9223e8d87f22986f5c3437144e10bbe78a6cc556093beafd'


def protocol():
    return dict(schema=TAG,words=list(WORDS),image_order=list(KEYS),images=5,
        frame=20,word_denominator=4,unordered_pairs=6,directed_pairs=12,
        shape=list(SHAPE),batch=1,new_API=EXPECTED,new_forward_samples=EXPECTED,new_frames=EXPECTED,
        AD=0,physics=0,predictor=0,training=False,selection=False,FLOPs=None,
        reduction='float64 mean squared error over all 256x384 patch elements',
        seal='all five complete token arrays durable before scoring',
        direction='predicted and observed latent MSE lower; native final coverage and reward higher; exact ties; no tolerance',
        color_intervention_frame20=None,goal_first_stage='null pixels, not encoder invariance',
        scientific_admission=False,C2_established=False,task_success=None,
        prospective_scope='fixed analysis/encoding dispatch on previously saved known-root assets, not new acquisition',
        output_cap_bytes=CAP,free_floor_bytes=FLOOR,wall_seconds=WALL,
        wall_scope='saved verify before timer; runtime load/encode/final verification timed',
        cpu_neural_fallback=False,retry=False,hidden_body=False,synthetic_pixels=False)

def actual_sha(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'explicit actual SHA256 required')
    return value


def source_paths():
    return dict(physical_report=ROOT/'results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json',
        physical_prepare=ROOT/'results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json',
        goal_prepare=ROOT/'results/agent_p5_replay_target_renderer_20261028_v1_prepare_MAIN01.json',
        model_prepare=ROOT/'results/main_p5_live_raster_encoder_20261028_v2_prepare_MAIN01.json',
        prediction_prepare=ROOT/'results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json')


def prepare():
    """Fixed actual authorities, no model/physics imports or outcome-dependent input selection."""
    paths=source_paths(); hashes=dict(physical_report=PHYSICAL_SHA,physical_prepare=PHYSICAL_PREP_SHA,
                                    goal_prepare=GOAL_PREP_SHA,model_prepare=MODEL_PREP_SHA,prediction_prepare=PREDICTION_PREP_SHA)
    bindings={}
    def bind(path,h):
        p=Path(path).resolve()
        require(sha(p)==h,'actual authority bytes')
        require(str(p) not in bindings or bindings[str(p)]==h,'binding collision')
        bindings[str(p)]=h
    docs={}
    for key,path in paths.items():
        bind(path,hashes[key]);docs[key]=read_json(path,hashes[key])
    pr,pp,gp,mp=(docs[k] for k in ('physical_report','physical_prepare','goal_prepare','model_prepare'))
    pred=docs['prediction_prepare']
    require(sha(owner.__file__)==OWNER_SHA,'frozen model runtime source')
    kernel=ROOT/'experiments/spectral_reachability/agent_p5_target_raster_encoder_20261028_v1.py'
    bind(kernel,KERNEL_SOURCE_SHA)
    for source in (pp['bindings'],gp['bindings'],mp['bindings'],pred['bindings'],mp['model_spec']['loader_manifest']['seals']):
        for path,h in source.items():bind(path,h)
    require(pr['schema']=='agent_p5_contact_action_words_physical_20261028_v1.report'
        and pr['status']=='all4_native_word_traces_recorded' and pr['global_error'] is None
        and pr['manifest_sha256']==PHYSICAL_PREP_SHA,'complete original physical authority')
    require([r['arm'] for r in pr['rows']]==list(WORDS),'all four original words')
    plans=pp['source_contract']['generator_private']['plans']
    require([p['arm'] for p in plans]==list(WORDS),'original plan identities')
    dispatch=read_json(paths['physical_report'].parent/'dispatch.json')
    require(dispatch['plans']==plans and dispatch['manifest_sha256']==PHYSICAL_PREP_SHA,'actual dispatched commands')
    for rel,rec in pr['artifacts'].items():
        path=paths['physical_report'].parent/rel
        require(path.resolve().is_relative_to(paths['physical_report'].parent.resolve()),'local original artifact')
        bind(path,rec['sha256']);require(path.stat().st_size==rec['bytes'],'complete original inventory')
    inputs=[];native=[]
    for row,plan in zip(pr['rows'],plans,strict=True):
        word=row['arm']
        require(row['status']=='completed' and row['error'] is None and
                [f['frame'] for f in row['records']]==list(range(21)),'full frame chronology')
        require(np.asarray(plan['prefix_raw_u']).shape==(10,2) and
                np.asarray(plan['reference_suffix_raw_u']).shape==(10,2),'full raw20 action witnesses')
        rec=row['records'][20];base=paths['physical_report'].parent/word/'frame_20'
        files={}
        for name,r in rec['files'].items():
            p=base/name;bind(p,r['sha256']);require(p.stat().st_size==r['bytes'],'frame20 receipt')
            files[name]=dict(path=str(p),**r)
        state=owner.array(files['state.npy'],(7,),np.float32)
        proprio=owner.array(files['proprio.npy'],(4,),np.float32)
        require(state.tolist()==rec['state'] and proprio.tolist()==rec['proprio'],'original full state/proprio')
        public=read_json(files['full_public_state.json']['path'],files['full_public_state.json']['sha256'])
        transition=read_json(files['transition.json']['path'],files['transition.json']['sha256'])
        coverage=public['env']['coverage_arr']
        require(len(coverage)==20 and transition['info']['final_coverage']==coverage[-1] and
                transition['info']['state']['values']==state.tolist() and transition['done'] is False,'same frame native labels')
        require(public['env']['goal_pose']==transition['info']['goal_pose'],'unchanged actual goal')
        owner.array(files['rgb.npy'],(224,224,3),np.uint8)
        inputs.append(dict(image_key=word+'__frame20',state_id=word,frame=20,rgb=files['rgb.npy'],files=files))
        native.append(dict(word=word,frame=20,coverage=coverage[-1],reward=transition['reward'],
            done=transition['done'],task_success=None,transition=transition,public_state=public,
            state=state.tolist(),proprio=proprio.tolist(),source_transition_sha256=files['transition.json']['sha256']))
    goal=gp['frames']['goal'][0]
    for rec in goal.values():bind(rec['path'],rec['sha256'])
    owner.array(goal['rgb.npy'],(224,224,3),np.uint8)
    goal_public=read_json(goal['full_public_state.json']['path'],goal['full_public_state.json']['sha256'])
    require(all(n['public_state']['env']['goal_pose']==goal_public['env']['goal_pose'] for n in native),'same native goal pose')
    inputs.append(dict(image_key='goal__native',state_id='goal',frame=0,rgb=goal['rgb.npy'],files=goal))
    predictions=pred['authority']['prediction']
    require(set(predictions)==set(WORDS) and pred['authority']['model_spec']==mp['model_spec'],'original prediction/model identity')
    for item in inputs[:-1]:
        require(item['rgb']==pred['authority']['terminals'][item['state_id']]['rgb'],'original full-word endpoint identity')
    old_goal=[i for i in mp['inputs'] if i['state_id']=='goal']
    require(len(old_goal)==2,'both actual goal color cells')
    original_goal=owner.array(goal['rgb.npy'],(224,224,3),np.uint8)
    for item in old_goal:
        a=owner.array(item['rgb'],(224,224,3),np.uint8)
        require(a.tobytes()==original_goal.tobytes(),'actual goal first-stage null, not invariance')
    for rec in predictions.values():
        bind(rec['path'],rec['sha256'])
        require(rec['shape']==[3,*SHAPE] and rec['dtype']=='float32','sealed original predicted patch ABI')
    for path in (Path(__file__),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')):bind(path,sha(path))
    return dict(schema=TAG,status='prepared_five_images_no_encodes',protocol=protocol(),
        authority={k:dict(path=str(paths[k]),sha256=hashes[k]) for k in paths},
        inputs=inputs,legacy_native_coverage=native,plans=plans,predictions=predictions,
        missing=[dict(word=w,frame=20,white_rgb=None,color_effect=None,reason='no authenticated frame20 color intervention') for w in WORDS],
        goal_first_stage='null_pixel_first_stage_no_encoder_invariance_test',
        model_spec=deepcopy(mp['model_spec']),bindings=bindings,
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False)


def verify(m):
    require(m['schema']==TAG and m['protocol']==protocol(),'own exact protocol')
    require(all(sha(p)==h for p,h in m['bindings'].items()),'full binding drift')
    require(digest(m)==digest(prepare()),'fixed whole input authority recomputation')


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


def direction(x):
    require(np.isfinite(x),'finite exact comparison')
    return int(x>0)-int(x<0)


def scores(features,native,predictions):
    require(tuple(features)==KEYS and [r['word'] for r in native]==list(WORDS),'fixed four/five denominator/order')
    for a in features.values():
        require(a.shape==SHAPE and a.dtype==np.float32 and np.isfinite(a).all(),'full float32 patch ABI')
    rows=[dict(word=w,frame=20,status='computed',
               full_patch_goal_mse=owner.mse(features[w+'__frame20'],features['goal__native'])) for w in WORDS]
    require(set(predictions)==set(WORDS),'all original whole-word predictions')
    for row in rows:
        p=predictions[row['word']]
        require(p.shape==SHAPE and p.dtype==np.float32 and np.isfinite(p).all(),'original terminal predicted patch')
        row['predicted_goal_mse']=owner.mse(p,features['goal__native'])
        row['prediction_to_real_feature_mse']=owner.mse(p,features[row['word']+'__frame20'])
    pairs=[]
    for i in range(4):
        for j in range(4):
            if i==j:continue
            dc=rows[i]['full_patch_goal_mse']-rows[j]['full_patch_goal_mse']
            dp=rows[i]['predicted_goal_mse']-rows[j]['predicted_goal_mse']
            contrasts={}
            for metric in ('coverage','reward'):
                dn=native[i][metric]-native[j][metric]
                ls,ps,ns=-direction(dc),-direction(dp),direction(dn)
                contrasts[metric]=dict(delta=dn,latent_preference=ls,native_preference=ns,
                    predicted_preference=ps,
                    predicted_relation='tie' if 0 in (ps,ns) else ('agreement' if ps==ns else 'reversal'),
                    relation='tie' if 0 in (ls,ns) else ('agreement' if ls==ns else 'reversal'))
            pairs.append(dict(left=WORDS[i],right=WORDS[j],latent_cost_delta=dc,predicted_cost_delta=dp,**contrasts))
    return dict(cross_condition=rows,fixed_within_state_raster_difference=None,same_time_rankings=pairs)


def evaluate(features,m):
    predictions={w:owner.array(m['predictions'][w],(3,*SHAPE))[-1] for w in WORDS}
    return scores(features,m['legacy_native_coverage'],predictions)


def empty_scores():
    return [dict(word=w,frame=20,status='not_computed',full_patch_goal_mse=None,
        predicted_goal_mse=None,prediction_to_real_feature_mse=None) for w in WORDS]


class Meter:
    def __init__(self,event,guard=lambda:None):
        self.attempted=dict.fromkeys(EXPECTED,0); self.completed=dict.fromkeys(EXPECTED,0)
        self.attempted_order=[]; self.completed_order=[]; self.pending=None; self.item=None
        self.event,self.guard=event,guard

    def before(self,key,batch):
        self.guard()
        require(key in EXPECTED and type(batch) is int and batch==1 and self.pending is None,'batch1/inflight contract')
        require(self.attempted[key]<EXPECTED[key],'predispatch module cap')
        require(len(self.attempted_order)<5 and self.item==KEYS[len(self.attempted_order)],'one forward per image, fixed five, no replay')
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
        cross_condition=empty_scores(),fixed_within_state_raster_difference=None,same_time_rankings=None,
        legacy_native_coverage=deepcopy(m['legacy_native_coverage']),errors=[],finalization_errors=[],
        missing=deepcopy(m['missing']),goal_first_stage=m['goal_first_stage'],
        cuda_executed=False,all5_tokens_sealed_before_scores=False,scientific_admission=False,
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
        require(meter.attempted==meter.completed==EXPECTED and meter.pending is None,'complete five actual forwards')
        store.event('all5_tokens_sealed',dict(records=r['token_artifacts'],order=list(KEYS)))
        r['all5_tokens_sealed_before_scores']=True
        r.update(evaluate(features,m)); r['status']='all5_raster_encodes_recorded'
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
    require(r['status']=='all5_raster_encodes_recorded' and r['errors']==r['finalization_errors']==[]
        and r['cuda_executed'] is r['all5_tokens_sealed_before_scores'] is True,'complete HOST CUDA required')
    require(r['scientific_admission'] is r['C2_established'] is r['physical_release'] is False
        and r['task_success'] is None and r['new_physics']==r['AD']==0,'diagnostic only, no scientific admission')
    require(r['model_before']==r['model_after']==m['model_spec']['loader_manifest']['expected_model_receipts'],'frozen expected/before/after identity')
    source_closure(m,r['imported_sources'])
    require(set(r['strict_loads'])=={'predictor','proprio_encoder'} and all(x['exact_match'] is True for x in r['strict_loads'].values()),'original strict checkpoint loads')
    require(np.isfinite(r['wall_seconds']) and 0<=r['wall_seconds']<WALL,'bounded wall')
    require(r['legacy_native_coverage']==m['legacy_native_coverage'],'unchanged old physics parallel values')
    require(r['missing']==m['missing'] and r['goal_first_stage']==m['goal_first_stage'],'no imputed color effects or invariance claim')
    require(r['items']==[dict(image_key=k,status='completed',error=None) for k in KEYS] and set(r['token_artifacts'])==set(KEYS),'all five files/returns')
    events=sorted(path.parent.glob('event_*.json')); require(len(events)==16,'five intent/return/artifact triplets then seal')
    meter=Meter(lambda *args:None); records={}; previous=None
    for i,p in enumerate(events):
        row=read_json(p)
        require(p.name==f'event_{i:04d}.json' and row['sequence']==i and row['previous_sha256']==previous
            and row['event_sha256']==digest({k:v for k,v in row.items() if k!='event_sha256'}),'durable event chain')
        previous=row['event_sha256']
        if i==15:
            require(row['kind']=='all5_tokens_sealed' and row['value']==dict(records=records,order=list(KEYS)),'all tokens durable before scoring')
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
        and meter.attempted==meter.completed==EXPECTED and meter.pending is None,'complete actual five-image meter/artifacts')
    features={}
    for key in KEYS:
        rec=records[key]
        require(Path(rec['path'])==path.parent/(key+'_patch.npy') and rec['shape']==list(SHAPE) and rec['dtype']=='float32','local full token ABI/path')
        features[key]=owner.array(rec,SHAPE)
    computed=evaluate(features,m)
    require(all(r[k]==computed[k] for k in computed),
            'all4 cross-condition/full-token independent recomputation')
    names={p.name for p in events}|{'report.json'}|{k+'_patch.npy' for k in KEYS}
    require(path.name=='report.json' and {p.name for p in path.parent.iterdir()}==names,'exact producer layout')
    require(sum(p.stat().st_size for p in path.parent.iterdir())<=CAP,'saved output cap')
    verify(m); require(sha(path)==report_sha256,'final report drift')
    return dict(schema=TAG+'.consume',status='all5_frame20_encodes_and4_scores_12_rankings_verified',
        report_sha256=report_sha256,manifest_digest=digest(m),cross_condition=computed['cross_condition'],
        fixed_within_state_raster_difference=computed['fixed_within_state_raster_difference'],same_time_rankings=computed['same_time_rankings'],
        missing=m['missing'],goal_first_stage=m['goal_first_stage'],
        legacy_native_coverage=m['legacy_native_coverage'],new_meter=r['meter'],AD=0,physics=0,
        scientific_admission=False,C2_established=False,task_success=None,physical_release=False,
        interpretation='same-time frame20 fixed-four descriptive ranking; no color invariance, task success or scientific admission')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','run','consume')); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256'); p.add_argument('--output',type=Path); p.add_argument('--report',type=Path)
    p.add_argument('--report-sha256'); p.add_argument('--confirm-host-cuda',action='store_true')
    a=p.parse_args(argv)
    if a.command=='prepare':
        result=prepare(); write_new(a.manifest,result)
    else:
        require(a.manifest_sha256 and sha(a.manifest)==actual_sha(a.manifest_sha256),'actual own prepare file SHA')
        m=read_json(a.manifest,a.manifest_sha256)
        if a.command=='verify': verify(m); result=dict(status='five_image_encoder_plan_verified')
        elif a.command=='run': result=run_cuda(m,a.output,a.confirm_host_cuda)
        else:
            require(a.report and a.report_sha256 and a.output,'actual CUDA report SHA/receipt output required')
            result=consume(m,a.report,a.report_sha256); write_new(a.output,result)
    print(json.dumps({'status':result['status'],'scientific_admission':False}))
    return 2 if result['status']=='failed' else 0


if __name__=='__main__': raise SystemExit(main())




