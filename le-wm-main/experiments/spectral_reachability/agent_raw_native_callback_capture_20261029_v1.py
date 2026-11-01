"""MAIN-only fixed164 native replay with transparent actual raster observation.

Cold import / prepare / consume never imports a simulator or neural runtime.
The actor is the frozen mixed.episode, NOT any old executor. No hydrate/retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import zlib

import numpy as np
from experiments.spectral_reachability import agent_raw_raster_factor_20261028_v1 as factor
from experiments.spectral_reachability import agent_observed_raw_coordinate_physical_20261028_v1 as old

NAME='agent_raw_native_callback_capture_20261029_v1'
AUTH='MAIN_RAW164_NATIVE_CALLBACK_REPLAY_ONLY'
ROOT=old.ROOT
mixed,source=old.mixed,old.source
require,sha,raw=old.require,old.sha,factor.raw
CAP,FLOOR,FINAL,WALL=4*1024**3,1024**3,256*1024**2,3600
META_CAP,SHOT_CAP=256*1024**2,2*1024**2
PRIMITIVES=('aaline','aalines','arc','circle','ellipse','line','lines','polygon','rect')
CALLBACKS=('draw_circle','draw_segment','draw_fat_segment','draw_polygon','draw_dot','color_for_shape')
MAX_RENDER=11152  # 164*(66 explicit off/on/off + 2 genuine reset-goal renders)
MAX_PRIMITIVES,MAX_CALLBACK_EVENTS=1024,2048


def digest(x):return hashlib.sha256(raw(x)).hexdigest()


def literal(x):
    if isinstance(x,np.ndarray):return literal(x.tolist())
    if isinstance(x,np.generic):return literal(x.item())
    if isinstance(x,(list,tuple)):return [literal(v) for v in x]
    if isinstance(x,dict):return {str(k):literal(v) for k,v in x.items()}
    require(x is None or type(x) in (bool,int,float,str),'literal actual native arguments, no repr guessing')
    if type(x) is float:require(np.isfinite(x),'finite native scalar')
    return x


def read(path,expected=None,cap=META_CAP):
    p=Path(path);require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)) and p.stat().st_size<=cap,'bounded nonsymlink JSON')
    before=sha(p);require(expected is None or before==expected,'external exact SHA')
    value=mixed.decode(p.read_bytes());require(sha(p)==before,'source unchanged after read');return value


def contract():
    return dict(schema=NAME+'.contract',jobs=164,pairs=80,known_roots=2,fresh_roots=0,
        pairs_schedule=factor.schedule(),plans='all fixed original physical prepare plans, unchanged raw10; no selection',
        actor='original mixed.episode reset/placement/warmup/raw10; never poststate reset/hydrate',
        caps=deepcopy(old.contract()['caps']),render_cap=MAX_RENDER,primitive_cap_per_render=MAX_PRIMITIVES,
        callback_event_cap_per_render=MAX_CALLBACK_EVENTS,
        transparent_tap='sys.setprofile actual Python debug callback args/returns + scoped pygame.draw/cv2.resize forward-once observers',
        observation='actual canvas512 and actual resize input/output; original cv2.resize called once; no rerasterization',
        old_gate='all original observation receipts/PNG bytes, transitions, actions, rows and actual native resource receipts exact',
        failure='all164 statuses/all80 pairs retained, no retries or tolerance; incomplete pair unknown',
        inference='new replay callback membership only; original historical callbacks/caches512 remain unsealed',
        wall_seconds=WALL,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,
        metadata_cap_bytes=META_CAP,shot_metadata_cap_bytes=SHOT_CAP,
        new_neural_calls=0,training=False,gate_changed=False,P1_admission=False,correction_deployed=False)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def source_material():
    """Bound saved authority, complete existing strict receipt, no detached geometry."""
    fixed={k:factor.read(ROOT/'results'/n,h) for k,(n,h) in factor.PINS.items()}
    p,r,c,a=(fixed[k] for k in ('physical_prepare','physical_report','physical_receipt','observability'))
    factor.validate_rows(p,r,c,a);old.validate_execution(p)
    require(c['schema']==old.NAME+'.consume' and c['status']==r['status']==old.RECORDED and
        c['counts']==r['counts'] and r['counts']=={k:dict(attempted=v,completed=v) for k,v in old.contract()['caps'].items()},'existing strict full native census')
    directory=(ROOT/'results'/factor.PINS['physical_report'][0]).parent
    require(sha(directory/'journal.jsonl')==r['journal_sha256'],'entire original journal SHA')
    # All original artifacts are frozen by the complete existing receipt. This
    # pass really checks every listed byte; does not claim to re-run old consume.
    bindings=deepcopy(c['bindings'])
    mixed.merge(bindings,p['bindings'])
    mixed.merge(bindings,{str(ROOT/'results'/n):h for n,h in factor.PINS.values()})
    mixed.merge(bindings,{str(directory/'journal.jsonl'):r['journal_sha256']})
    for file,h in bindings.items():require(sha(file)==h,'all saved source/artifact closure bytes')
    require(bindings[str(factor.ENV)]==factor.ENV_SHA and bindings[str(factor.UTILS)]==factor.UTILS_SHA,'original environment/renderer source')
    data=mixed.source_material()
    require(p['source_slots']==[r['slot'] for r in data['roots']] and p['source_roots']==mixed.allmode.root_metadata(data)
        and p['warm_traces']==[r['warm_trace'] for r in data['roots']],'original source/reset/history/goal identity')
    for file in own_files()+[Path(old.__file__),Path(factor.__file__),Path(mixed.__file__)]:bindings[str(file)]=sha(file)
    for relative in ('pymunk/space_debug_draw_options.py','pymunk/_callbacks.py','pymunk/space.py',
                     'pymunk/pygame_util.py','pygame/draw.cpython-312-x86_64-linux-gnu.so',
                     'pygame/surface.cpython-312-x86_64-linux-gnu.so','pygame/surfarray.py',
                     'pygame/__init__.py','cv2/__init__.py'):
        path=factor.SITE/relative
        if path.is_file():bindings[str(path)]=sha(path)
        else:require(not relative.startswith('pymunk/'),'required native callback source')
    binaries=list((factor.SITE/'pymunk').glob('_chipmunk*.so'))+list((factor.SITE/'cv2').glob('*.so'))+list((factor.SITE/'pygame').glob('*.so'))
    require(len(binaries)>=2,'native Chipmunk and cv2 binary closure')
    for file in binaries:bindings[str(file)]=sha(file)
    q=dict(schema=NAME+'.prepare',contract=contract(),bindings=bindings,
        original_prepare=str(ROOT/'results'/factor.PINS['physical_prepare'][0]),original_prepare_sha256=factor.PINS['physical_prepare'][1],
        original_report=str(directory/'report.json'),original_report_sha256=factor.PINS['physical_report'][1],
        plans_sha256=digest(p['plans']),jobs_sha256=digest(p['jobs']),source_roots_sha256=digest(p['source_roots']),
        old_scene_evidence='shape-only raster factor is NOT full callback history',P1_admission=False)
    return q,p,r,data


def write_new(path,value):
    p=Path(path).absolute();b=raw(value)+b'\n'
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and
        not any(x.is_symlink() for x in (p,*p.parents)) and len(b)<=META_CAP and
        shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive bounded own metadata/floor')
    with p.open('xb') as f:require(f.write(b)==len(b),'full write');f.flush();os.fsync(f.fileno())
    old.paired.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p),status=value.get('status','prepared_no_execution'))


def prepare(output):return write_new(output,source_material()[0])


def verify(prepared,expected):
    q=read(prepared,expected);now,p,r,data=source_material()
    require(q==now and sha(prepared)==expected,'full prepared source and schedule identity');return q,p,r,data


class Store(old.Store):
    def __init__(self,path):
        self.path=Path(path).absolute()
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and not self.path.exists()
            and not any(x.is_symlink() for x in (self.path,*self.path.parents)) and
            shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'NEW4GiB +1GiB floor')
        self.path.mkdir();old.paired.fsync_dir(self.path.parent)
        self.previous=None;self.index=0;self.bytes=0;self.hashes={};self.sizes={};self.started=time.monotonic()
    def check(self,size=4096,final=False):
        require(self.bytes+size<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+size,'4GiB/256MiB final reserve/1GiB floor')
        require(final or time.monotonic()-self.started<WALL,'3600s wall before any work/write')
    def emit(self,kind,**fields):
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(fields));e['event_sha256']=source.digest(e)
        self.write('journal.jsonl',old.raw(e)+b'\n',append=True,final=kind in ('run_finished','failure'))
        self.previous=e['event_sha256'];self.index+=1
    def frame(self,name,value):
        require(isinstance(value,np.ndarray) and value.dtype==np.uint8 and value.ndim==3 and value.shape[-1]==3,'unaltered uint8 RGB')
        return super().frame(name,value)


class Meter(old.Meter):
    """Original native resource axes/guard, own3600s job wall, no old globals."""
    def call(self,kind,fn,*,cleanup=False,**fields):
        self.guard_call(kind,fn,cleanup=cleanup,**fields)
        require(not cleanup or kind=='close','only close may bypass floor/wall fault')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'own wall/fault before dispatch');self.store.check()
        count=self.counts[kind];require(count['attempted']<contract()['caps'][kind],'native cap before dispatch')
        receipt=dict(index=len(self.receipts),resource=kind,**deepcopy(self.where),**deepcopy(fields),status='pending')
        try:self.store.emit('operation_dispatch_intent',receipt=receipt)
        except BaseException as exc:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(mixed.base.error('cleanup_intent',exc))
        self.receipts.append(receipt);count['attempted']+=1;self.active[kind]+=1
        try:value=fn();count['completed']+=1;receipt['status']='returned'
        except BaseException as exc:
            receipt.update(status='failed',error=mixed.base.error(kind,exc))
            try:self.store.emit('operation_failed',receipt=receipt)
            except BaseException as extra:self.fault=True;self.secondary.append(mixed.base.error('failure_journal',extra))
            raise
        finally:self.active[kind]-=1
        try:self.store.emit('operation_returned',receipt=receipt)
        except BaseException as exc:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(mixed.base.error('cleanup_return',exc))
        return value


def rng_state():
    n=np.random.get_state()
    return dict(python=literal(random.getstate()),numpy=[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])])


def rgb_id(a):
    require(a.dtype==np.uint8 and a.ndim==3 and a.shape[-1]==3,'actual uint8 RGB')
    return dict(shape=list(a.shape),dtype=a.dtype.str,pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def shape_arg(shape,body):
    # Native object identity is retained separately, never used for pair equality.
    owner=shape.body
    role='agent' if owner is body.agent else 'block' if owner is body.block else 'other'
    result=dict(body_role=role,shape_type=type(shape).__name__)
    if hasattr(shape,'get_vertices'):result['local_vertices']=literal(list(shape.get_vertices()))
    elif hasattr(shape,'offset'):result['offset']=literal(tuple(shape.offset))
    if hasattr(shape,'radius'):result['radius']=float(shape.radius)
    return result


class Tap:
    """Forward exact original args ONCE; never replace callback/renderer logic.

    Scoped process-local pygame.draw and cv2.resize observer bindings are restored
    exactly. sys.setprofile sees original Python callback bodies including nested
    polygon->fat-segment calls, and color_for_shape actual return. No callback is
    re-run or evaluated from a reconstructed body/shape.
    """
    def __init__(self,pygame,cv2,store,meter,utils_path,options_path,render_path):
        self.pg,self.cv,self.store,self.meter=pygame,cv2,store,meter
        self.paths={str(Path(utils_path).resolve()),str(Path(options_path).resolve())}
        self.render_path=str(Path(render_path).resolve());self.body=None;self.branch=None
        self.current=None;self.captures=[];self.failed=[];self.saved={};self.restore_errors=[]
        self.owner=threading.get_ident();self.total_primitives=0;self.total_callbacks=0
        self.installed={};self.profile_installed=False
    def bind(self,body,branch):self.body,self.branch=body,branch
    def stamp(self):
        b=self.body;goal=getattr(b,'_goal',None)
        # First reset goal render precedes assignment of _goal. Keep literal
        # missing/None, not a fabricated goal image or poststate hydrate.
        return dict(snapshot=old.moving.snapshot(b),goal_state=literal(b.goal_state),goal_pose=literal(b.goal_pose),
            goal_rgb=None if goal is None else rgb_id(goal),with_target=b.with_target,
            render_mode=b.render_mode,render_size=b.render_size,window_size=b.window_size,render_action=b.render_action,
            latest_action=literal(b.latest_action),variations=literal(b.variation_space.value),
            env_rng=literal(b.rng.bit_generator.state),global_rng=rng_state())
    def event(self,kind,**fields):
        shot=self.current;require(shot is not None,'scoped native render')
        require(len(shot['events'])<2*MAX_CALLBACK_EVENTS+2*MAX_PRIMITIVES+16,'bounded full ordered transcript')
        shot['events'].append(dict(index=len(shot['events']),kind=kind,**fields))
    def profile(self,frame,event,arg):
        try:self._profile(frame,event,arg)
        except BaseException as exc:
            # CFFI may swallow Python profiler exceptions. Latch and reject at
            # resize/end; never silently drop a callback and release a receipt.
            self.failed.append(mixed.base.error('profile',exc))
    def _profile(self,frame,event,arg):
        # Hot observer: no resolve/stat/AST work per Python/C profile event.
        # Runtime source paths are absolute and explicitly sealed in prepare.
        code=frame.f_code;path=code.co_filename
        if path==self.render_path and code.co_name=='_render_frame':
            if event=='call':
                require(self.current is None and self.body is frame.f_locals['self'] and self.branch is not None,'single original live render')
                require(threading.get_ident()==self.owner and len(self.captures)<MAX_RENDER,'single-thread/render cap')
                self.current=dict(index=len(self.captures),branch=self.branch,operation_scope=deepcopy(self.meter.where),
                    operation_stack=self.pending_operations(),
                    mode=frame.f_locals['mode'],before=self.stamp(),events=[],stack=[],resizes=0,primitives=0,callbacks=0)
                self.store.emit('native_render_intent',index_in_capture=self.current['index'],branch=self.branch)
            elif event=='return':self.finish(arg)
            return
        if self.current is None or path not in self.paths or code.co_name not in CALLBACKS:return
        s=self.current
        if event=='call':
            require(s['callbacks']<MAX_CALLBACK_EVENTS,'callback predispatch cap');obj=frame.f_locals['self']
            args={}
            for name in code.co_varnames[1:code.co_argcount]:
                value=frame.f_locals[name]
                args[name]=shape_arg(value,self.body) if name=='shape' else literal(value)
            n=s['callbacks'];s['callbacks']+=1;self.total_callbacks+=1;s['stack'].append((id(frame),n))
            self.event('callback_call',callback=n,name=code.co_name,source=path,source_line=code.co_firstlineno,
                args=args,flags=int(obj.flags),transform=literal(tuple(obj.transform)))
        elif event=='return':
            require(s['stack'] and s['stack'][-1][0]==id(frame),'complete nested original callback return')
            _,n=s['stack'].pop();self.event('callback_return',callback=n,value=literal(arg))
    def pending_operations(self):
        count=sum(getattr(self.meter,'active',{}).values());found=[]
        if count:
            for op in reversed(self.meter.receipts):
                if op['status']=='pending':found.append(deepcopy(op))
                if len(found)==count:break
        require(len(found)==count,'actual pending native operation stack')
        return list(reversed(found))
    def initial(self,surface):
        s=self.current
        require(surface is self.body.screen and tuple(surface.get_size())==(512,512),'actual original canvas512 surface identity')
        if 'background' not in s:
            image=np.transpose(self.pg.surfarray.array3d(surface),(1,0,2)).copy()
            color=np.asarray(self.body.variation_space['background']['color'].value,dtype=np.uint8)
            require(color.shape==(3,) and np.all(image==color),'actual canvas after original fill before first draw')
            s['background']=dict(color=color.tolist(),**rgb_id(image))
    def draw(self,name,fn):
        def call(*args,**kwargs):
            if self.current is None:return fn(*args,**kwargs)
            s=self.current;require(not self.failed and threading.get_ident()==self.owner,'tap failure/thread guard')
            self.initial(args[0] if args else kwargs['surface'])
            require(s['primitives']<MAX_PRIMITIVES,'primitive predispatch cap')
            values=literal(args[1:]);kw={k:literal(v) for k,v in kwargs.items() if k!='surface'}
            n=s['primitives'];s['primitives']+=1;self.total_primitives+=1
            self.event('primitive_intent',primitive=n,op=name,args=values,kwargs=kw,
                callback=s['stack'][-1][1] if s['stack'] else None)
            value=fn(*args,**kwargs) # unchanged original values, one invocation
            self.event('primitive_return',primitive=n,rect=literal(tuple(value)))
            return value
        return call
    def resize(self,fn):
        def call(*args,**kwargs):
            if self.current is None:return fn(*args,**kwargs)
            s=self.current;require(not self.failed and s['resizes']==0 and not s['stack'],'complete callback stream before resize')
            self.initial(self.body.screen)
            image=args[0] if args else kwargs['src']
            actual=np.transpose(self.pg.surfarray.array3d(self.body.screen),(1,0,2)).copy()
            require(image.dtype==np.uint8 and image.shape==(512,512,3) and np.array_equal(image,actual),'actual512 surface == original cv2 source bytes')
            s['resize_args']=dict(args=literal(args[1:]),kwargs={k:literal(v) for k,v in kwargs.items() if k!='src'})
            self.event('resize_intent',source=rgb_id(actual),**s['resize_args'])
            s['surface512']=dict(file=f'tap_{s["index"]:05d}_512.png',**rgb_id(actual))
            s['surface512']['sha256']=self.store.frame(s['surface512']['file'],actual)
            result=fn(*args,**kwargs) # original cv2 code, flags/interpolation untouched
            require(result.shape==(224,224,3) and result.dtype==np.uint8,'actual original resize224')
            s['rgb224']=dict(file=f'tap_{s["index"]:05d}_224.png',**rgb_id(result))
            s['rgb224']['sha256']=self.store.frame(s['rgb224']['file'],result)
            s['resizes']+=1;self.event('resize_return',result=rgb_id(result));return result
        return call
    def finish(self,result):
        s=self.current
        require(s is not None and not self.failed and not s['stack'] and s['resizes']==1,'one complete original render/no swallowed callback error')
        require(rgb_id(result)=={k:s['rgb224'][k] for k in ('shape','dtype','pixel_sha256')},'returned render exactly captured resize output')
        after=self.stamp();require(s['before']==after,'tap render state/goal/renderer/RNG exact before-after')
        s['after']=after;s.pop('stack');self.event('render_return')
        transcript=dict(schema=NAME+'.transcript',**s)
        payload=raw(transcript)+b'\n';require(len(payload)<=SHOT_CAP,'complete transcript2MiB, never truncate')
        name=f'tap_{s["index"]:05d}.json.gz';self.store.write(name,gzip.compress(payload,mtime=0))
        ref={k:deepcopy(s[k]) for k in ('index','branch','operation_scope','operation_stack','surface512','rgb224','primitives','callbacks')}
        ref.update(file=name,sha256=sha(self.store.path/name),transcript_digest=digest(transcript))
        self.captures.append(ref);self.store.emit('native_render_return',receipt=ref);self.current=None
    def __enter__(self):
        require(sys.getprofile() is None and threading.get_ident()==self.owner,'no existing profile lost/single owner')
        try:
            for name in PRIMITIVES:
                fn=getattr(self.pg.draw,name);self.saved[name]=fn;wrapped=self.draw(name,fn)
                self.installed[name]=wrapped;setattr(self.pg.draw,name,wrapped)
            self.saved['resize']=self.cv.resize;self.installed['resize']=self.resize(self.cv.resize);self.cv.resize=self.installed['resize']
            sys.setprofile(self.profile);self.profile_installed=True;return self
        except BaseException:self.__exit__(*sys.exc_info());raise
    def __exit__(self,*exc):
        if self.profile_installed and sys.getprofile()!=self.profile:self.restore_errors.append(dict(phase='profile_drift',message='unexpected profile observer replaced'))
        sys.setprofile(None)
        for name,fn in self.saved.items():
            try:
                owner=self.cv if name=='resize' else self.pg.draw
                if getattr(owner,name) is not self.installed[name]:
                    self.restore_errors.append(dict(phase='binding_drift',name=name,message='observer replaced before restore'))
                setattr(owner,name,fn)
            except BaseException as error:self.restore_errors.append(mixed.base.error('tap_restore',error))
        return False


def transcript_load(directory,rec):
    name=rec['file'];require(Path(name).name==name and sha(directory/name)==rec['sha256'],'actual transcript file SHA')
    rawbytes=(directory/name).read_bytes();require(len(rawbytes)<=SHOT_CAP,'compressed shot cap')
    dec=zlib.decompressobj(31);body=dec.decompress(rawbytes,SHOT_CAP+1)
    require(len(body)<=SHOT_CAP and dec.eof and not dec.unused_data and not dec.unconsumed_tail,'single bounded complete transcript gzip')
    value=mixed.decode(body);require(digest(value)==rec['transcript_digest'],'full transcript semantic digest');return value


def load_rgb(directory,rec):
    from PIL import Image
    name=rec['file'];require(Path(name).name==name and sha(directory/name)==rec['sha256'],'actual surface PNG bytes')
    with Image.open(directory/name) as im:
        require(im.mode=='RGB' and im.format=='PNG','native RGB PNG');a=np.asarray(im).copy()
    require(rgb_id(a)=={k:rec[k] for k in ('shape','dtype','pixel_sha256')},'every saved raster pixel');return a


def validate_transcript(t):
    require(t['schema']==NAME+'.transcript' and t['before']==t['after'] and t['mode']=='rgb_array'
        and t['resizes']==1 and t['surface512']['shape']==[512,512,3] and t['rgb224']['shape']==[224,224,3],
        'fresh native raster/state/RNG and actual512->224')
    require(t['events'] and t['events'][-1]['kind']=='render_return','explicit final native render return')
    stack=[];primitives={};returned=set();callbacks=0;resize=0;pending=None
    for i,e in enumerate(t['events']):
        require(e['index']==i,'full ordered transcript')
        k=e['kind']
        if k=='callback_call':
            require(resize==0 and e['callback']==callbacks and e['name'] in CALLBACKS and
                e['source'] in (str(factor.UTILS),str(factor.SITE/'pymunk/space_debug_draw_options.py')),'actual callback census/source');stack.append(callbacks);callbacks+=1
        elif k=='callback_return':require(stack and stack.pop()==e['callback'],'actual nested callback return')
        elif k=='primitive_intent':
            require(resize==0 and pending is None and e['primitive']==len(primitives) and e['op'] in PRIMITIVES and
                e['callback']==(stack[-1] if stack else None),'actual ordered pygame primitive/callback association')
            primitives[e['primitive']]=e;pending=e['primitive']
        elif k=='primitive_return':
            require(e['primitive']==pending and e['primitive'] not in returned,'one original primitive return');returned.add(e['primitive']);pending=None
        elif k=='resize_intent':require(not stack and pending is None and resize==0 and e['source']['pixel_sha256']==t['surface512']['pixel_sha256'],'all drawing before original resize');resize=1
        elif k=='resize_return':require(resize==1 and e['result']['pixel_sha256']==t['rgb224']['pixel_sha256'],'actual resize return');resize=2
        else:require(k=='render_return' and i==len(t['events'])-1 and resize==2,'only one final native render return')
    require(not stack and callbacks==t['callbacks']<=MAX_CALLBACK_EVENTS and
        len(primitives)==len(returned)==t['primitives']<=MAX_PRIMITIVES,'complete actual callback/primitive counts')
    require(t['background']['shape']==[512,512,3] and t['resize_args']==dict(args=[[224,224]],kwargs={}),
        'original uniform background and unmodified cv2.resize default interpolation')


def pair_analysis(captures,directory):
    selected={}
    for shot in captures:
        key=shot.get('original_observation')
        if key is not None and key['phase']=='plan' and key['step']==10 and key['condition']==old.ORDER[shot['branch']//41][1]:
            # off triplet repeats index2: fixed original selected frame is0.
            wanted=1 if key['condition']=='on' else 0
            if key['overlay_index']==wanted:
                require(shot['branch'] not in selected,'one selected terminal per original branch');selected[shot['branch']]=shot
    rows=[]
    for schedule in factor.schedule():
        a,b=(selected.get(schedule[k]) for k in ('minus_index','plus_index'))
        row=dict(**schedule,callbacks_equal=None,integer_primitives_equal=None,surface512_equal=None,rgb224_equal=None,
            first_loss_stage=None,missing=[])
        if a is None or b is None:
            row['missing']=[k for k,x in (('minus',a),('plus',b)) if x is None];rows.append(row);continue
        ta,tb=transcript_load(directory,a),transcript_load(directory,b)
        def callback_values(t):return [{k:v for k,v in e.items() if k not in ('index','source','source_line')} for e in t['events'] if e['kind'] in ('callback_call','callback_return')]
        def primitive_values(t):return [dict(op=e['op'],args=e['args'],kwargs=e['kwargs']) for e in t['events'] if e['kind']=='primitive_intent']
        row.update(callbacks_equal=callback_values(ta)==callback_values(tb),
            integer_primitives_equal=primitive_values(ta)==primitive_values(tb) and ta['background']==tb['background'],
            surface512_equal=bool(np.array_equal(load_rgb(directory,a['surface512']),load_rgb(directory,b['surface512']))),
            rgb224_equal=bool(np.array_equal(load_rgb(directory,a['rgb224']),load_rgb(directory,b['rgb224']))))
        if row['integer_primitives_equal'] and not row['surface512_equal']:row['first_loss_stage']='unexplained_same_primitives_changed512'
        elif row['surface512_equal'] and not row['rgb224_equal']:row['first_loss_stage']='unexplained_same512_changed224'
        elif not row['callbacks_equal'] and row['integer_primitives_equal']:row['first_loss_stage']='callback_differences_erased_before_integer_primitives'
        elif not row['integer_primitives_equal'] and row['surface512_equal']:row['first_loss_stage']='different_primitives_same512'
        elif not row['surface512_equal'] and row['rgb224_equal']:row['first_loss_stage']='different512_same224'
        else:row['first_loss_stage']='no_disappearance_at_observed_layers'
        rows.append(row)
    require(len(rows)==80,'fixed complete80 pairs, missing never excluded');return rows


def old_events(directory):
    with (directory/'journal.jsonl').open('rb') as f:
        for line in f:
            require(len(line)<=2*1024**2 and line.endswith(b'\n'),'bounded complete original event')
            yield mixed.decode(line)


class GateStore(Store):
    """Actor writes original names/PNG encoder. New tap artifacts stay separate."""
    def reference(self,report):
        self.old_directory=Path(report['execution_binding']['prepared']).parent/factor.PINS['physical_report'][0].split('/')[0]
        self.original=report;self.next_observation=(e for e in old_events(self.old_directory) if e['kind']=='fresh_observation')
        self.tap=None;self.observations=0
    def emit(self,kind,**fields):
        if kind=='fresh_observation':
            want=next(self.next_observation,None)
            same=want is not None and all(fields[k]==want[k] for k in fields)
            frame=fields['receipt'];same=bool(same and sha(self.path/frame['png'])==frame['file_sha256']==sha(self.old_directory/frame['png']))
            require(self.tap is not None and self.tap.captures,'actual render before actor observation')
            shot=self.tap.captures[-1]
            require(shot['branch']==fields['branch'] and shot['rgb224']['pixel_sha256']==frame['pixel_sha256'] and
                sha(self.path/shot['rgb224']['file'])==frame['file_sha256'],'captured return matches original actual PNG')
            shot['original_observation']={k:deepcopy(frame[k]) for k in ('phase','step','condition','overlay_index','png','file_sha256','pixel_sha256')}
            self.observations+=1
            super().emit('observation_gate',branch=fields['branch'],png=frame['png'],exact=same,
                expected_digest=None if want is None else digest({k:want[k] for k in fields}),actual_digest=digest(fields))
            result=super().emit(kind,**fields)
            require(same,'entire original PNG/state/snapshot/goal exact; actual failed frame retained')
            return result
        return super().emit(kind,**fields)


def run(prepared,expected,output,authorization=None):
    require(authorization==AUTH,'MAIN authority before any simulator import/output')
    q,p,prior,data=verify(prepared,expected);store=GateStore(output);store.reference(prior)
    meter=Meter(store,store.started)
    rows=old.rows_for(p);errors=[];tap=None
    binding=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,original_report_sha256=q['original_report_sha256'])
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',contract=contract(),execution_binding=binding)
        import gymnasium as gym
        import stable_worldmodel  # original registration, MAIN only
        import pygame
        import cv2
        runtime_paths={
            'pygame':Path(pygame.__file__).resolve(),'pygame.draw':Path(pygame.draw.__file__).resolve(),
            'cv2':Path(cv2.__file__).resolve()}
        require(all(str(path) in q['bindings'] and sha(path)==q['bindings'][str(path)] for path in runtime_paths.values()),'actual loaded runtime source/binary paths bound')
        tap=Tap(pygame,cv2,store,meter,factor.UTILS,factor.SITE/'pymunk/space_debug_draw_options.py',factor.ENV)
        store.tap=tap
        with tap:
            for row,plan in zip(rows,p['plans'],strict=True):
                if meter.fault or time.monotonic()-store.started>=WALL:break
                meter.where=dict(branch=row['index'],**{k:row[k] for k in ('root_slot','condition','arm','selection_digest')})
                meter.guard_context(row,p['prefix_reference'][row['index']]);env=None;row['status']='running'
                try:
                    env=meter.call('constructor',lambda:gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False))
                    mixed.validate_env(env.unwrapped);tap.bind(env.unwrapped,row['index'])
                    require(env.unwrapped._render_frame.__code__.co_filename==str(factor.ENV) and
                        Path(sys.modules[type(env.unwrapped).__module__].__file__).resolve()==factor.ENV,'bound original native renderer class/code path')
                    if row['index']==0:
                        runtime_paths.update(environment=factor.ENV,renderer=Path(sys.modules['stable_worldmodel.envs.utils'].__file__).resolve())
                        require(all(str(path) in q['bindings'] and sha(path)==q['bindings'][str(path)] for path in runtime_paths.values()),'actual loaded renderer path bound')
                        store.put('runtime_paths.json',{k:str(v) for k,v in runtime_paths.items()})
                    require(env.unwrapped.render_action is False and env.unwrapped.render_size==224,'original renderer/control')
                    with mixed.native_hooks(env,meter,row['secondary_errors']):
                        mixed.episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store)
                    require(not row['secondary_errors'] and not tap.failed,'all observer/native cleanup/failure latches')
                except BaseException as exc:
                    error=mixed.base.error('branch',exc);errors.append(error);row.update(status='engineering_failed',first_error=error)
                finally:
                    try:row['prefix_reference_exact']=old.prefix_match(row,p['prefix_reference'][row['index']])
                    except BaseException as exc:errors.append(mixed.base.error('prefix',exc))
                    if env is not None:
                        try:meter.call('close',env.close,cleanup=True)
                        except BaseException as exc:errors.append(mixed.base.error('close',exc))
                    for phase,label in (('warmup','warmup'),('plan','own')):
                        ops=[o for o in meter.receipts if o['branch']==row['index'] and o['resource']=='public_step' and o['phase']==phase]
                        row[label+'_attempted']=len(ops);row[label+'_returned']=sum(o['status']=='returned' for o in ops)
                # A failed branch stops further native dispatch, never advances
                # the reference stream to a convenient subsequent job.
                store.emit('branch_closed',receipt=row)
                require(row==prior['rows'][row['index']],'entire replay row equal original; no historical equivalence tolerance')
    except BaseException as exc:errors.append(mixed.base.error('execution',exc))
    finally:
        errors.extend(meter.secondary)
        if tap is not None:errors.extend(tap.failed+tap.restore_errors)
        source_ok=False
        try:
            for file,h in q['bindings'].items():require(sha(file)==h,'final exact input/source closure')
            source_ok=True
        except BaseException as exc:errors.append(mixed.base.error('final_source',exc))
        captures=[] if tap is None else tap.captures
        exact=not errors and source_ok and rows==prior['rows'] and meter.receipts==prior['actual_operation_receipts'] and meter.counts==prior['counts'] and len(captures)==MAX_RENDER
        if time.monotonic()-store.started>=WALL:errors.append(dict(phase='wall',message='3600s cap'));exact=False
        status='raw164_callback_capture_exact' if exact else 'partial_or_failed'
        try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
        except BaseException as exc:errors.append(mixed.base.error('journal_finalize',exc));status='partial_or_failed';exact=False
        report=dict(schema=NAME+'.report',status=status,contract=contract(),execution_binding=binding,rows=rows,captures=captures,
            errors=errors,counts=meter.counts,actual_operation_receipts=meter.receipts,original_observations=store.observations,
            full_replay_exact=exact,source_verified=source_ok,journal_chain_tail=store.previous,
            capture_primitives=0 if tap is None else tap.total_primitives,capture_callbacks=0 if tap is None else tap.total_callbacks,
            partial_capture=None if tap is None else deepcopy(tap.current),
            wall_seconds=time.monotonic()-store.started,P1_admission=False,historical_callback_membership=None,
            old_raster_release=False,new_neural_calls=0,training=False,
            artifacts={n:h.hexdigest() for n,h in store.hashes.items()})
        store.put('report.json',report,final=True)
    return report


def consume(output,expected):
    directory=Path(output).absolute();r=read(directory/'report.json',expected)
    binding=r['execution_binding'];q,p,prior,_=verify(binding['prepared'],binding['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['contract']==contract() and len(r['rows'])==164 and
        r['P1_admission'] is False and r['historical_callback_membership'] is None and r['old_raster_release'] is False,'own fixed denominator/new-only evidence')
    inv=r['artifacts'];require(set(inv)|{'report.json'}=={f.name for f in directory.iterdir()} and
        all(Path(n).name==n and sha(directory/n)==h for n,h in inv.items()) and
        sum(f.stat().st_size for f in directory.iterdir())<=CAP,'complete flat saved output SHA/byte census')
    require(read(directory/'execution_binding.json')==binding,'durable own execution binding')
    events=list(old_events(directory));tail=None
    for i,e in enumerate(events):
        require(e['index']==i and e['previous_sha256']==tail and e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'complete new journal hashchain')
        tail=e['event_sha256']
    require(tail==r['journal_chain_tail'] and events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished','whole native journal bookends')
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==[x for x in r['rows'] if x['status']!='not_started'],'all completed/partial branches preserved')
    shots=r['captures'];returned=[e['receipt'] for e in events if e['kind']=='native_render_return']
    require(len(shots)==len(returned) and all({k:v for k,v in s.items() if k!='original_observation'}==ret for s,ret in zip(shots,returned,strict=True)),
        'all actual native render returns associated without forged historical keys')
    names={'execution_binding.json','journal.jsonl'};primitives=callbacks=0;render_operations=[]
    if 'runtime_paths.json' in inv:
        runtime=read(directory/'runtime_paths.json');names.add('runtime_paths.json')
        require(all(path in q['bindings'] and sha(path)==q['bindings'][path] for path in runtime.values()),'all actual runtime paths sealed')
    for i,s in enumerate(shots):
        require(s['index']==i and 0<=s['branch']<164,'ordered bounded native renders')
        t=transcript_load(directory,s);validate_transcript(t)
        require(all(t[k]==s[k] for k in ('index','branch','operation_scope','operation_stack','surface512','rgb224','primitives','callbacks')),'full shot/transcript association')
        stack=s['operation_stack'];require(stack,'actual original native render scope')
        for op in stack:
            n=op['index'];require(0<=n<len(r['actual_operation_receipts']) and
                op==dict(r['actual_operation_receipts'][n],status='pending') and op['branch']==s['branch'],
                'each pending native render operation tied to actual return receipt')
        inner=stack[-1];require(inner['resource'] in ('goal_render','explicit_render') and inner['index'] not in render_operations,'one actual capture per native render operation')
        render_operations.append(inner['index'])
        for k in ('surface512','rgb224'):load_rgb(directory,s[k]);names.add(s[k]['file'])
        names.add(s['file']);primitives+=s['primitives'];callbacks+=s['callbacks']
    fresh=[e for e in events if e['kind']=='fresh_observation']
    originals=[e for e in old_events(Path(q['original_report']).parent) if e['kind']=='fresh_observation']
    require(len(fresh)==r['original_observations'] and len(fresh)<=len(originals),'full actual observation prefix')
    exact=r['status']=='raw164_callback_capture_exact'
    gates=[e for e in events if e['kind']=='observation_gate']
    require(len(gates)==len(fresh),'each observation gate durable even on mismatch')
    by_png={}
    for s in shots:
        if 'original_observation' in s:
            name=s['original_observation']['png'];require(name not in by_png,'unique observed PNG to render');by_png[name]=s
    for e,want,gate in zip(fresh,originals,gates):
        same={k:v for k,v in e.items() if k not in ('index','previous_sha256','event_sha256')}=={
            k:v for k,v in want.items() if k not in ('index','previous_sha256','event_sha256')}
        f=e['receipt'];name=f['png'];names.add(name)
        factor.png(directory/name,f)
        same=bool(same and sha(directory/name)==sha(Path(q['original_report']).parent/name))
        require(gate['png']==name and gate['exact'] is same,'exact old gate independently recomputed')
        if exact:require(same,'all saved original observations exact')
        shot=by_png.get(name)
        require(shot is not None and shot['branch']==e['branch'] and shot['rgb224']['sha256']==f['file_sha256'],
            'one true rendered return per durable original PNG')
    unmatched=sorted(set(inv)-names)
    partial=r['partial_capture'];pp=pc=0
    if partial is not None and partial['index']==len(shots):pp=partial.get('primitives',0);pc=partial.get('callbacks',0)
    require(names<=set(inv) and primitives+pp==r['capture_primitives'] and callbacks+pc==r['capture_callbacks'],'all saved complete/partial actual shot calls')
    if exact:require(not unmatched,'no unreferenced output on complete success')
    if exact:
        require(r['full_replay_exact'] is r['source_verified'] is True and r['errors']==[] and r['rows']==prior['rows'] and
            r['counts']==prior['counts'] and r['actual_operation_receipts']==prior['actual_operation_receipts'] and
            len(fresh)==10824 and len(shots)==MAX_RENDER and partial is None and 0<=r['wall_seconds']<WALL,'all164 replay native/state/action/meter exact, no missing callbacks')
        require('runtime_paths.json' in names and set(runtime)=={'environment','renderer','pygame','pygame.draw','cv2'},'complete actual runtime identity paths')
        require(render_operations==[o['index'] for o in r['actual_operation_receipts'] if o['resource'] in ('goal_render','explicit_render')],
            'every native render and no extra direct/missing render captured in operation order')
        # Compare every old operation/transition event as well as row summaries.
        for kind in ('operation_dispatch_intent','operation_returned','operation_failed','returned_transition','placement_recipe','goal_attached','cp5_guard'):
            clean=lambda e:{k:v for k,v in e.items() if k not in ('index','previous_sha256','event_sha256')}
            original=[clean(e) for e in old_events(Path(q['original_report']).parent) if e['kind']==kind]
            require([clean(e) for e in events if e['kind']==kind]==original,'entire actual action/transition/resource trace exact')
    else:require(r['status']=='partial_or_failed' and not r['full_replay_exact'],'failed replay cannot be released')
    pairs=pair_analysis(shots,directory)
    return dict(schema=NAME+'.consume',status='all164_native_callback_capture_verified' if exact else 'blocked_partial_replay',
        report_sha256=expected,prepare_sha256=binding['prepare_sha256'],fixed_jobs=164,fixed_pairs=80,
        captures=len(shots),primitives=primitives,callbacks=callbacks,counts=r['counts'],pairs=pairs,
        unreferenced_partial_artifacts=unmatched,full_replay_exact=exact,P1_admission=False,historical_callback_membership=None,old_raster_release=False,
        sufficient_original_cause_certified=False,training=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha','receipt','authorization'):q.add_argument('--'+k)
    a=q.parse_args(argv)
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);result=dict(status='verified_no_execution')
    elif a.command=='consume':result=write_new(a.receipt,consume(a.output,a.report_sha))
    else:
        r=run(a.prepared,a.prepare_sha,a.output,a.authorization);result=dict(status=r['status'],counts=r['counts'],errors=r['errors'])
    print(json.dumps(result,sort_keys=True));return 2 if result.get('status') in ('partial_or_failed','blocked_partial_replay') else 0


if __name__=='__main__':raise SystemExit(main())
