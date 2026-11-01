"""Eight-arm saved RGB -> frozen native CUDA encodings; seal before saved Q.

Import/prepare/consume are NumPy metadata only. MAIN alone authorizes run.
No prediction, optimization, training, image edits, rendering or physics.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np

NAME='p1_moving_overlay_recompile_rgb_cuda_20261027_v1'
PHYSICAL_NAME='p1_moving_overlay_recompile_physical_20261027_v1'
SOURCE_NAME='p1_moving_target_overlay_source_20261027_v1'
SOURCE_SHA='85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a'
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results'/ (SOURCE_NAME+'_main01')/'report.json'
POLICY=Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
MODELS={str(POLICY/'config.json'):'2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09',
        str(POLICY/'weights.pt'):'48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'}
MODEL_TENSOR_SHA='4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546'
AUTH='MAIN_MOVING_OVERLAY_RECOMPILE_RGB_CUDA_ONLY'
KEYS=[(s,c,a) for s in (0,1) for c in ('off','on') for a in ('line_search_GN','full10_conditional_D2P')]
PHYSICAL_CAPS=dict(constructor=8,close=8,reset=16,public_step=160,physics_step=1632,
                   set_state=32,goal_setter=24,goal_render=16,goal_bind=8,explicit_render=528)
CAP,FLOOR,FINAL,WALL=32*1024**2,200*1024**2,2*1024**2,600


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def hash_value(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'literal external SHA256 required')


def flat(base,name):
    require(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat artifact name')
    p=Path(base)/name; require(not p.is_symlink(),'no symlink artifact'); return p


def sha(path):
    p=Path(path); require(p.is_file() and p.resolve()==p.absolute(),'literal regular input')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def decode(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(v):raise ValueError('nonfinite JSON '+v)
    v=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad);canonical(v);return v


def read(path,limit=2*1024**2):
    require(Path(path).stat().st_size<=limit,'bounded metadata')
    return decode(Path(path).read_bytes())


def config_shapes(config):
    e=config['encoder'];p=config['projector'];size=e['image_size'];patch=e['patch_size'];width=p['input_dim']
    require(all(type(x) is int and x>0 for x in (size,patch,width)) and size%patch==0,'config-derived patch grid')
    require(size==224 and patch==14 and width==p['output_dim']==192 and e['pretrained'] is False,'original native offline224/patch14 checkpoint configuration')
    return dict(pixels=[1,3,size,size],hidden=[1,(size//patch)**2+1,width],cls=[1,width],projected=[1,p['output_dim']])


def contract():
    return dict(fixed_denominator=8,checkpoints=[0,5,10],caps=dict(encoder=28,projector=28),
        goal_policy='four actual same-condition paired endpoint goal_rgb; default goal_pose is not the constructed goal',
        no_dedup=True,no_padding=True,encoding_only=True,prediction='not_recomputed',
        Q='float64 mean squared distance of original projected native192; no fit',
        new_predict=0,new_AD=0,new_control=0,new_physics=0,training=False,CUDA_only=True,TF32=False,
        wall_seconds=WALL,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,
        physical_resources='nested categories retained separately, never summed',P1_admission=False)


def assemble(report,prep,events,source_report,source_base):
    """Pure saved joins, not a physics replay or a scientific admission test."""
    require(report['schema']==PHYSICAL_NAME+'.report' and prep['schema']==PHYSICAL_NAME+'.prepare','new physical schema')
    require(report['fixed_denominator']==8 and report['P1_admission'] is False,'physical denominator')
    require(report['source_report_sha256']==prep['source_report_sha256']==SOURCE_SHA and
            report['planner_report_sha256']==prep['planner_report_sha256'],'source/planner identities')
    require(source_report['schema']==SOURCE_NAME+'.report' and source_report['status']=='paired_moving_overlay_recorded'
            and source_report['full_protocol'] is True and source_report['source_verified'] is True and not source_report['errors'],'original paired source')
    require(prep['contract']['flags']==source_report['contract']['flags']==[False,True,False]
            and prep['contract']['no_goal_pose_edit'] is source_report['contract']['no_goal_pose_edit'] is True,'unaltered overlay protocol')
    require(len(report['rows'])==len(prep['plans'])==8 and len(source_report['blind_exports'])==2,'eight rows/four condition packets')
    require(events and events[0]['kind']=='run_started' and events[0]['plans']==prep['plans'] and
            events[0]['contract']==prep['contract'] and events[-1]['kind']=='run_finished' and
            events[-1]['counts']==report['counts'],'physical journal joins')
    require(set(report['counts'])==set(PHYSICAL_CAPS),'nested physical counters')
    for k,cap in PHYSICAL_CAPS.items():
        c=report['counts'][k];require(all(type(c[t]) is int for t in ('attempted','completed')) and
            0<=c['completed']<=c['attempted']<=cap,'bounded physical '+k)
        require(c==dict(attempted=sum(e['kind']=='operation_attempted' and e.get('resource')==k for e in events),
                       completed=sum(e['kind']=='operation_completed' and e.get('resource')==k for e in events)),
                'separate nested physical journal counters '+k)
    rows=[];images=[];exports={}
    for s,entry in enumerate(source_report['blind_exports']):
        require(type(entry['source_slot']) is int and entry['source_slot']==s,'paired slot order')
        for c in ('off','on'):
            e=entry['artifacts'][c];idx=len(images);path=flat(source_base,e['path'])
            exports[s,c]=(entry,e,idx)
            images.append(dict(index=idx,kind='goal',root_slot=s,condition=c,path=str(path),
                file_sha256=e['sha256'],pixel_sha256=e['goal_pixel_sha256'],packet_metadata=deepcopy(e)))
    for i,(row,plan,key) in enumerate(zip(report['rows'],prep['plans'],KEYS,strict=True)):
        require(type(row['root_slot']) is int and type(plan['root_slot']) is int and
            (row['root_slot'],row['condition'],row['arm'])==(plan['root_slot'],plan['condition'],plan['arm'])==key and row['index']==i,'fixed row identity')
        for k in ('selection_digest','task_id'):
            hash_value(row[k]);require(row[k]==plan[k],'row/plan '+k)
        s,c,_=key;entry,e,goal=exports[s,c]
        require(plan['source_report_sha256']==SOURCE_SHA and plan['planner_report_sha256']==report['planner_report_sha256']
            and plan['blind_path']==str(flat(source_base,e['path'])) and plan['blind_sha256']==e['sha256'],'same-condition source plan')
        cps=row['checkpoints'];steps=[x['step'] for x in cps]
        require(all(type(t) is int and t in (0,5,10) for t in steps) and steps==sorted(set(steps)) and
            row['missing_checkpoints']==[t for t in (0,5,10) if t not in steps],'checkpoint missing labels, no replacement')
        if cps or 'root_state' in row:
            state=np.asarray(row['root_state'],np.float64);target=np.asarray(row['goal'],np.float64)
            require(state.shape==target.shape==(7,) and np.isfinite(state).all() and np.isfinite(target).all()
                and digest(np.concatenate([state,target]).tobytes())==entry['old_root_sha256'],'physical root/true goal binding')
            require(row['goal_pixel_sha256']==e['goal_pixel_sha256'],'condition-specific true goal')
        out=dict(root_slot=s,condition=c,arm=key[2],selection_digest=row['selection_digest'],
                 physical_status=row['status'],goal_image=goal,checkpoints=[])
        for t in (0,5,10):
            cp=next((x for x in cps if x['step']==t),None);cpout=dict(step=t,image=None,latent=None,Q=None,status='missing_physical_checkpoint')
            if cp is not None:
                require(set(cp)=={'step','png','file_sha256','pixel_sha256','state','snapshot'} and cp['state']==cp['snapshot']['state'],'checkpoint snapshot binding')
                selected=[ev for ev in events if ev['kind']=='fresh_observation' and ev.get('branch')==i
                    and ev['receipt'].get('phase')=='plan' and ev['receipt'].get('step')==t
                    and ev['receipt'].get('overlay_index')==(1 if c=='on' else 0)]
                require(len(selected)==1,'exact selected real PNG journal entry')
                ev=selected[0];rc=ev['receipt']
                require(all(ev.get(k)==row[k] for k in ('root_slot','condition','arm','selection_digest')) and rc['condition']==c
                    and all(rc[k]==v for k,v in cp.items()),'checkpoint journal/source identity')
                if t==0:require(cp['pixel_sha256']==e['rgb_history_pixel_sha256'][-1],'observed warmup endpoint, not padded')
                idx=len(images);flat('.',cp['png'])
                images.append(dict(index=idx,kind='checkpoint',row=i,root_slot=s,condition=c,step=t,
                    path=cp['png'],file_sha256=cp['file_sha256'],pixel_sha256=cp['pixel_sha256'],physical=deepcopy(cp)))
                cpout.update(image=idx,status='not_started')
            out['checkpoints'].append(cpout)
        rows.append(out)
    return dict(rows=rows,images=images,contract=contract(),physical_resources=deepcopy(report['counts']),
                physical_status=report['status'],physical_errors=deepcopy(report['errors']),
                source_report_sha256=SOURCE_SHA,planner_report_sha256=report['planner_report_sha256'])


def packet(raw):
    require(len(raw)<=2*1024**2,'bounded blind bytes')
    with zipfile.ZipFile(BytesIO(raw)) as z:
        info=z.infolist();require(len(info)==3 and {x.filename for x in info}=={'rgb_history.npy','goal_rgb.npy','raw_action_history.npy'}
            and all(x.file_size<=512*1024 and not x.flag_bits&1 for x in info) and sum(x.file_size for x in info)<=1024**2,'blind-only bounded archive')
    with np.load(BytesIO(raw),allow_pickle=False) as f:a={k:f[k] for k in f.files}
    fixedraw=np.asarray([[.1,0]]*5+[[0,.1]]*5,dtype=np.float32)
    require(a['rgb_history'].dtype==a['goal_rgb'].dtype==np.uint8 and a['rgb_history'].shape==(3,224,224,3)
        and a['goal_rgb'].shape==(224,224,3) and a['raw_action_history'].dtype==np.float32 and
        a['raw_action_history'].shape==(10,2) and np.array_equal(a['raw_action_history'],fixedraw),
        'exact paired moving float32 raw history: first5[.1,0], last5[0,.1]; no reordering/rescaling')
    return a


def load_image(item):
    p=Path(item['path']);require(p.stat().st_size<=2*1024**2,'bounded RGB source');raw=p.read_bytes()
    require(digest(raw)==item['file_sha256'],'input file read SHA')
    if item['kind']=='goal':
        a=packet(raw);meta=item['packet_metadata']
        require([digest(x.tobytes()) for x in a['rgb_history']]==meta['rgb_history_pixel_sha256'] and
            digest(a['raw_action_history'].tobytes())==meta['raw_action_sha256'],'paired packet complete pixel/raw metadata')
        rgb=a['goal_rgb']
    else:
        from PIL import Image
        with Image.open(BytesIO(raw)) as im:
            require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'unaltered actual RGB PNG224')
            rgb=np.asarray(im).copy()
    require(rgb.dtype==np.uint8 and rgb.shape==(224,224,3) and digest(rgb.tobytes())==item['pixel_sha256'],'actual condition pixels')
    return rgb,raw


def journal(path,tail):
    require(Path(path).stat().st_size<=16*1024**2,'bounded physical journal')
    return decode_journal(Path(path).read_bytes(),tail)


def decode_journal(raw,tail):
    require(len(raw)<=16*1024**2 and raw.endswith(b'\n'),'bounded complete physical journal')
    events=[];previous=None
    for i,line in enumerate(raw.splitlines()):
        e=decode(line);unsigned={k:v for k,v in e.items() if k!='event_sha256'}
        require(e['index']==i and e['previous_sha256']==previous and
            e['event_sha256']==digest(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()),'physical journal hashchain')
        previous=e['event_sha256'];events.append(e)
    require(previous==tail,'physical journal terminal SHA');return events


def build(physical_report,physical_sha):
    """Saved metadata only. Existing future report + MAIN external SHA mandatory."""
    hash_value(physical_sha);rp=Path(physical_report).absolute();require(sha(rp)==physical_sha,'external real physical report SHA')
    report=read(rp);binding=report['execution_binding'];bp=rp.parent/'execution_binding.json'
    require(sha(bp)==report['execution_binding_sha256'] and read(bp)==binding,'physical execution binding')
    pp=Path(binding['prepared']);require(pp.is_absolute() and sha(pp)==binding['prepare_sha256']==report['prepare_sha256'],'physical prepare binding')
    prep=read(pp);require(binding['source_report_sha256']==SOURCE_SHA and
        binding['planner_report_sha256']==prep['planner_report_sha256'],'physical bound inputs')
    require(sha(SOURCE)==SOURCE_SHA,'fixed original paired report SHA');src=read(SOURCE)
    jp=rp.parent/'journal.jsonl';require(sha(jp)==report['journal_sha256'],'physical journal file SHA')
    p=assemble(report,prep,journal(jp,report['journal_chain_tail']),src,str(SOURCE.parent))
    # Use only compiler saved consume, never its CUDA runner/old producer execution.
    from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as compiler
    plans=compiler.consume(prep['planner_output'],prep['planner_report_sha256'])
    require(len(plans)==8 and all(all(a[k]==b[k] for k in b) for a,b in zip(plans,prep['plans'],strict=True)),'actual compiler saved plan seal')
    bindings={}
    def bind(path,h=None):
        path=str(Path(path).absolute());actual=sha(path);hash_value(actual)
        require(h is None or h==actual,'input SHA mismatch '+path)
        require(path not in bindings or bindings[path]==actual,'path/hash conflict');bindings[path]=actual
    for path,h in prep['bindings'].items():bind(path,h)
    for path,h in MODELS.items():bind(path,h)
    for path in (rp,bp,pp,SOURCE,jp,Path(compiler.__file__)):bind(path)
    from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as native
    for path in native.source_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bind(path)
    for item in p['images']:
        if item['kind']=='checkpoint':item['path']=str(flat(rp.parent,item['path']))
        bind(item['path'],item['file_sha256']);load_image(item)
    p.update(schema=NAME+'.prepare',arguments=dict(physical_report=str(rp),physical_sha=physical_sha),
        shapes=config_shapes(read(POLICY/'config.json')),expected_model_hash=MODEL_TENSOR_SHA,bindings=bindings,policy=str(POLICY))
    for path,h in bindings.items():require(sha(path)==h,'prepare final source drift')
    return p


def syncdir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def prepare(output,physical_report,physical_sha):
    p=build(physical_report,physical_sha);raw=canonical(p);output=Path(output).absolute()
    require(len(raw)<=2*1024**2 and not output.exists() and not output.is_symlink(),'new exclusive bounded prepare')
    require(shutil.disk_usage(output.parent).free>=FLOOR+len(raw),'prepare disk floor')
    with output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    syncdir(output.parent);return dict(prepared=str(output),prepare_sha256=sha(output))


def verify(prepared,expected):
    hash_value(expected);require(sha(prepared)==expected,'external prepare SHA');p=read(prepared)
    require(p==build(**p['arguments']),'complete saved source/config/plan reconstruction');return p


class Budget:
    def __init__(self,clock=time.monotonic,free=lambda:shutil.disk_usage('.').free):
        self.clock=clock;self.free=free;self.started=clock();self.used=0
    def check(self,size=0,final=False):
        require(final or self.clock()-self.started<=WALL,'whole run wall600s; final error receipts only after exhaustion')
        require(self.used+size<=CAP-(0 if final else FINAL),'32MiB total output cap')
        require(self.free()>=FLOOR+size,'200MiB free disk floor')


class Artifacts:
    def put(self,name,value,final=False):return self.write(name,canonical(value),final=final)
    def array(self,name,a):
        a=np.ascontiguousarray(a);require(a.dtype in (np.dtype('float32'),np.dtype('uint8')) and np.isfinite(a).all(),'original finite native array')
        f=BytesIO();np.save(f,a,allow_pickle=False);file=name+'.npy.gz';raw=gzip.compress(f.getvalue(),mtime=0)
        h=self.write(file,raw)
        return dict(file=file,sha256=h,array_sha256=digest(a.tobytes()),shape=list(a.shape),dtype=str(a.dtype))


class Store(Artifacts):
    def __init__(self,output,budget):
        self.path=Path(output).absolute();require(not self.path.exists() and not self.path.is_symlink(),'NEW exclusive output directory')
        self.path.mkdir();syncdir(self.path.parent);self.hashes={};self.budget=budget
        budget.free=lambda:shutil.disk_usage(self.path).free
    def check(self,size=0,final=False):self.budget.check(size,final)
    def write(self,name,raw,final=False):
        path=flat(self.path,name);self.check(len(raw),final)
        with path.open('xb') as f:
            require(f.write(raw)==len(raw),'full artifact write');f.flush();os.fsync(f.fileno())
        syncdir(self.path);self.budget.used+=len(raw);self.hashes[name]=digest(raw);return self.hashes[name]


def load_array(receipt,read_blob):
    flat('.',receipt['file']);raw=read_blob(receipt['file']);require(len(raw)<=2*1024**2 and digest(raw)==receipt['sha256'],'array file SHA')
    with gzip.GzipFile(fileobj=BytesIO(raw)) as f:data=f.read(2*1024**2+1)
    require(len(data)<=2*1024**2,'bounded expanded array')
    stream=BytesIO(data);a=np.load(stream,allow_pickle=False)
    require(stream.tell()==len(data) and str(a.dtype)==receipt['dtype'] and list(a.shape)==receipt['shape']
        and a.dtype in (np.dtype('float32'),np.dtype('uint8')) and np.isfinite(a).all()
        and digest(np.ascontiguousarray(a).tobytes())==receipt['array_sha256'],'original array SHA/dtype/shape')
    return a


def validate_chain(receipts,latent,shapes,read_blob):
    require(len(receipts)==2 and [x['resource'] for x in receipts]==['encoder','projector'] and
        all(x['completed'] is True and len(x['inputs'])==1 for x in receipts),'exact completed native module chain')
    enc,proj=receipts
    pre=load_array(enc['preprocessed'],read_blob)
    arrays=[load_array(x,read_blob) for x in (enc['inputs'][0],enc['output'],proj['inputs'][0],proj['output'])]
    require([list(x.shape) for x in arrays]==[shapes[k] for k in ('pixels','hidden','cls','projected')]
        and all(x.dtype==np.float32 for x in arrays),'config-derived full hidden/CLS/projector shapes, never assumed197')
    require(pre.dtype==np.float32 and list(pre.shape)==[1,*shapes['pixels']] and
        np.array_equal(pre.reshape(shapes['pixels']),arrays[0]),'actual original transform -> encoder input exact')
    require(np.array_equal(arrays[1][:,0],arrays[2]) and np.array_equal(arrays[3][0],latent),'actual full hidden CLS -> projector -> latent exact')


class Trace:
    def __init__(self,torch,shapes,store,check):
        self.torch=torch;self.shapes=shapes;self.store=store;self.check=check;self.entries=[];self.image=None;self.blocked=[];self.preprocessed=None
    def save(self,tensor,name):
        require(getattr(getattr(tensor,'device',None),'type',None)=='cuda','CUDA native tensor only')
        require(tensor.dtype==self.torch.float32,'native float32, no autocast')
        self.torch.cuda.synchronize();return self.store.array(name,tensor.detach().cpu().contiguous().numpy().copy())
    def before(self,kind,index):
        self.check();current=[e for e in self.entries if e['image']==index]
        require([(e['resource'],e['completed']) for e in current]==([] if kind=='encoder' else [('encoder',True)]),'one encoder then one projector, no retry/dedup')
        require(sum(e['resource']==kind for e in self.entries)<28,'module cap28')
    def dispatch(self,kind,tensor,callback,*args,**kwargs):
        self.before(kind,self.image);require(not self.torch.is_grad_enabled(),'inference only')
        e=dict(image=self.image,resource=kind,completed=False,inputs=[]);self.entries.append(e)
        self.store.put(f'module_{len(self.entries):02d}_attempt.json',deepcopy(e))
        prefix=f'i{self.image:02d}_{kind}'
        try:
            if kind=='encoder':
                require(self.preprocessed is not None,'actual original transformed pixels receipt')
                e['preprocessed']=deepcopy(self.preprocessed)
            e['inputs']=[self.save(tensor,prefix+'_input')]
            require(e['inputs'][0]['shape']==self.shapes['pixels' if kind=='encoder' else 'cls'],'module input shape')
            result=callback(*args,**kwargs);self.torch.cuda.synchronize()
            output=result.last_hidden_state if kind=='encoder' else result
            e['output']=self.save(output,prefix+'_output')
            require(e['output']['shape']==self.shapes['hidden' if kind=='encoder' else 'projected'],'config-derived module output')
            e['completed']=True;self.store.put(f'module_{len(self.entries):02d}_completed.json',deepcopy(e));return result
        except BaseException as exc:
            e['error']=str(exc)[:1000]
            self.store.put(f'module_{len(self.entries):02d}_failed.json',deepcopy(e),final=True);raise
    def forbid(self,name):
        def denied(*a,**kw):
            self.blocked.append(name);raise ValueError('forbidden '+name)
        return denied


def evaluate(p,store,encode,loader,finalize,entries=None):
    rows=deepcopy(p['rows']);images=[dict(deepcopy(i),status='not_started',latent=None,modules=[]) for i in p['images']]
    values={};errors=[];meta={};counts={k:dict(attempted=0,completed=0) for k in ('encoder','projector')}
    try:
        for item in images:
            store.check();idx=item['index'];item['status']='attempted'
            pixels,original=loader(item)
            require(pixels.dtype==np.uint8 and pixels.shape==(224,224,3) and digest(pixels.tobytes())==item['pixel_sha256']
                and digest(original)==item['file_sha256'],'unaltered original input')
            file=f'i{idx:02d}_original.'+('npz' if item['kind']=='goal' else 'png')
            item['original']=dict(file=file,sha256=store.write(file,original));item['rgb']=store.array(f'i{idx:02d}_rgb',pixels)
            value,receipts=encode(pixels.copy(),idx)
            require(value.dtype==np.float32 and value.shape==(192,) and np.isfinite(value).all(),'actual native latent192')
            # The callback must expose the original full hidden, not a fabricated CLS-only receipt.
            validate_chain(receipts,value,p['shapes'],lambda name:store.files[name] if hasattr(store,'files') else flat(store.path,name).read_bytes())
            require(all(e['image']==idx for e in receipts),'module image identity')
            item['modules']=deepcopy(receipts);item['latent']=store.array(f'i{idx:02d}_latent',value)
            item['status']='encoded';values[idx]=value.copy()
        meta=finalize()
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)[:1500]))
        for item in images:
            if item['status']=='attempted':item['status']='failed'
        try:meta=finalize()
        except BaseException as final_exc:errors.append(dict(type=type(final_exc).__name__,message=str(final_exc)[:1000]))
    actual=deepcopy(entries if entries is not None else [e for im in images for e in im['modules']])
    for k in counts:
        counts[k]=dict(attempted=sum(e['resource']==k for e in actual),completed=sum(e['resource']==k and e['completed'] for e in actual))
    unchanged=meta.get('model_hash_before')==meta.get('model_hash_after')==p.get('expected_model_hash',MODEL_TENSOR_SHA)
    good=not errors and unchanged and meta.get('source_verified') is True and not meta.get('blocked') and all(i['status']=='encoded' for i in images)
    # This write is durable before *any* Q arithmetic; no metric can influence encodings.
    seal=dict(schema=NAME+'.encoding_seal',images=deepcopy(images),counts=counts,entries=actual,meta=meta,errors=errors,
        shapes=p['shapes'],model_unchanged=unchanged,input_binding=p.get('arguments'),contract=p['contract'])
    seal_sha=store.put('encoding_seal.json',seal,final=True)
    if good:
        try:store.check()
        except BaseException as exc:
            good=False;errors.append(dict(type=type(exc).__name__,message=str(exc)[:1500]))
    for row in rows:
        for cp in row['checkpoints']:
            if cp['image'] is not None:
                item=images[cp['image']];cp['latent']=item['latent'];cp['status']=item['status']
            if good and cp['latent'] is not None:
                cp['Q']=float(np.mean((values[cp['image']].astype(np.float64)-values[row['goal_image']].astype(np.float64))**2))
    if good:store.put('metrics.json',dict(encoding_seal_sha256=seal_sha,rows=rows),final=True)
    result=dict(schema=NAME+'.report',status='encoded_sealed' if good else 'partial_or_failed',rows=rows,images=images,
        counts=counts,errors=errors,model_unchanged=unchanged,**meta,encoding_seal_sha256=seal_sha,shapes=p['shapes'],
        physical_resources=p['physical_resources'],physical_status=p['physical_status'],physical_errors=p['physical_errors'],
        contract=p['contract'],input_binding=p.get('arguments'),
        prediction_status='not_recomputed_encoding_only',P1_admission=False,training=False,
        artifact_sha256=deepcopy(store.hashes))
    store.put('report.json',result,final=True);return result


def run(prepared,expected,output,*,authorization=None):
    require(authorization==AUTH,'explicit MAIN CUDA authorization required before verification')
    budget=Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    store.put('execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    trace=None;model=None;before=None;host=None;restore_errors=[]
    def finalize():
        budget.check();require(not restore_errors,'module restoration errors')
        require(model is not None,'CUDA model unavailable')
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
            and not torch.is_autocast_enabled(),'actual TF32/autocast off')
        native.frozen(model);after=native.tensor_hash(model);verify(prepared,expected);budget.check()
        return dict(model_hash_before=before,model_hash_after=after,source_verified=True,cuda_verified=True,
                    host=host,TF32=False,autocast=False,SDP='MATH',blocked=[] if trace is None else trace.blocked)
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
        native.frozen(model);before=native.tensor_hash(model);require(before==MODEL_TENSOR_SHA,'original frozen tensor hash')
        trace=Trace(torch,p['shapes'],store,store.check);transform=p0.image_transform()
        with native.guard(model,torch,trace,restore_errors),torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            def encode(pixels,index):
                trace.image=index;x=p0._image_tensor(pixels,transform=transform,device=device)
                require(x.device.type=='cuda' and x.dtype==torch.float32 and tuple(x.shape)==(1,1,3,224,224),'native original transform CUDA pixels')
                trace.preprocessed=trace.save(x,f'i{index:02d}_preprocessed')
                result=model.encode({'pixels':x})['emb'];torch.cuda.synchronize(device)
                require(result.device.type=='cuda' and result.dtype==torch.float32 and tuple(result.shape)==(1,1,192),'native original emb')
                z=result[0,0].detach().cpu().numpy().copy()
                return z,[deepcopy(e) for e in trace.entries if e['image']==index]
            result=evaluate(p,store,encode,load_image,finalize,trace.entries)
        require(not restore_errors,'failed to restore forbidden hooks');return result
    except BaseException as exc:
        # Initialization/pre-encoding failure still leaves a durable NEW raw error receipt.
        if 'report.json' not in store.hashes:
            store.put('initialization_error.json',dict(type=type(exc).__name__,message=str(exc)[:1500]),final=True)
            if 'encoding_seal.json' not in store.hashes:
                def failed_loader(item):raise ValueError('initialization failed: '+str(exc)[:1000])
                evaluate(p,store,None,failed_loader,lambda:{},[] if trace is None else trace.entries)
        raise


def consume_report(report,read_blob):
    require(report['schema']==NAME+'.report' and report['status']=='encoded_sealed' and report['cuda_verified'] is True
        and report['model_unchanged'] is True and report['model_hash_before']==report['model_hash_after']==MODEL_TENSOR_SHA
        and report['source_verified'] is True and not report['errors'] and report['P1_admission'] is False
        and report.get('TF32') is False and report.get('autocast') is False and report.get('SDP')=='MATH'
        and report.get('blocked')==[],'real CUDA sealed saved encoding required, RAM is not evidence')
    require(report['contract']==contract() and report['shapes']==dict(pixels=[1,3,224,224],hidden=[1,257,192],cls=[1,192],projected=[1,192]),'bound original patch14 full shape')
    for name,h in report['artifact_sha256'].items():
        flat('.',name);require(digest(read_blob(name))==h,'sealed artifact SHA')
    raw=read_blob('encoding_seal.json');require(digest(raw)==report['encoding_seal_sha256'],'encoding seal SHA')
    seal=decode(raw);require(seal['images']==report['images'] and seal['counts']==report['counts'] and
        seal['meta']['cuda_verified'] is True and all(report[k]==v for k,v in seal['meta'].items()) and
        seal['shapes']==report['shapes'] and seal['contract']==report['contract'],'seal/report original modules identity')
    require(len(report['rows'])==8 and len(report['images'])<=28,'fixed eight denominator')
    entries=[e for im in report['images'] for e in im['modules']]
    require(seal['entries']==entries and report['counts']=={k:dict(attempted=len(report['images']),completed=len(report['images']))
        for k in ('encoder','projector')},'actual sealed module attempts/completions, no hidden extra calls')
    values={}
    for idx,item in enumerate(report['images']):
        require(item['index']==idx and item['status']=='encoded','all declared images encoded')
        pixels=load_array(item['rgb'],read_blob);require(pixels.shape==(224,224,3) and pixels.dtype==np.uint8
            and digest(pixels.tobytes())==item['pixel_sha256'],'saved original pixels')
        z=load_array(item['latent'],read_blob);require(z.shape==(192,) and z.dtype==np.float32,'native saved192')
        validate_chain(item['modules'],z,report['shapes'],read_blob);values[idx]=z
    require(len(report['images'])>=4 and [(i['kind'],i['root_slot'],i['condition']) for i in report['images'][:4]]==
        [('goal',s,c) for s in (0,1) for c in ('off','on')],'four separate true condition goals')
    out=[];used=[]
    for row_index,(row,key) in enumerate(zip(report['rows'],KEYS,strict=True)):
        require((row['root_slot'],row['condition'],row['arm'])==key and [c['step'] for c in row['checkpoints']]==[0,5,10],'stable consume row order')
        require(row['goal_image']==key[0]*2+(key[1]=='on'),'own condition goal index')
        for cp in row['checkpoints']:
            if cp['image'] is None:require(cp['latent'] is None and cp['Q'] is None,'unknown not zero')
            else:
                item=report['images'][cp['image']]
                require(item['kind']=='checkpoint' and (item['root_slot'],item['condition'],item['step'])==(key[0],key[1],cp['step'])
                    and item['row']==row_index and cp['latent']==item['latent'],'checkpoint image binding')
                used.append(cp['image'])
                q=float(np.mean((values[cp['image']].astype(np.float64)-values[row['goal_image']].astype(np.float64))**2))
                require(type(cp['Q']) is float and cp['Q']==q,'saved Q recompiled only after seal')
        out.append({k:deepcopy(row[k]) for k in ('root_slot','condition','arm','selection_digest','checkpoints')})
    require(sorted(used)==list(range(4,len(report['images']))),'each actual checkpoint used once, never silently dropped/duplicated')
    metrics=decode(read_blob('metrics.json'));require(metrics==dict(encoding_seal_sha256=report['encoding_seal_sha256'],rows=report['rows']),'metrics sealed-after-encoding binding')
    return out


def consume(output,report_sha256):
    hash_value(report_sha256);base=Path(output).absolute();rp=base/'report.json';require(sha(rp)==report_sha256,'external evaluator report SHA')
    require(sum(f.stat().st_size for f in base.iterdir() if f.is_file())<=CAP,'total saved output32MiB')
    def blob(name):
        p=flat(base,name);require(p.stat().st_size<=CAP,'bounded saved blob');return p.read_bytes()
    report=read(rp);binding=decode(blob('execution_binding.json'));p=verify(binding['prepared'],binding['prepare_sha256'])
    require(report['input_binding']==p['arguments'] and report['shapes']==p['shapes'],'saved current prepare/config binding')
    for a,b in zip(report['images'],p['images'],strict=True):
        require(all(a[k]==v for k,v in b.items()),'saved input metadata exactly prepared')
    require(len(report['rows'])==len(p['rows'])==8,'prepared row denominator')
    for a,b in zip(report['rows'],p['rows'],strict=True):
        require(all(a[k]==v for k,v in b.items() if k!='checkpoints') and
            [c['image'] for c in a['checkpoints']]==[c['image'] for c in b['checkpoints']],'prepared selection/condition/arm/checkpoint binding')
    return consume_report(report,blob)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('physical-report','physical-sha','prepared','prepare-sha','output','report-sha','authorization'):parser.add_argument('--'+k)
    a=parser.parse_args(argv)
    if a.command=='prepare':v=prepare(a.output,a.physical_report,a.physical_sha)
    elif a.command=='verify':v=dict(status='verified_saved_only',images=len(verify(a.prepared,a.prepare_sha)['images']))
    elif a.command=='consume':v=consume(a.output,a.report_sha)
    else:v=run(a.prepared,a.prepare_sha,a.output,authorization=a.authorization)
    print(canonical(v).decode().strip())


if __name__=='__main__':main()
