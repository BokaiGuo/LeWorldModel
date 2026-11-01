"""Source-first conjunctive-budget HOST adapter. Import/fixtures never import torch.

MAIN owns production. No mutation of old modules, model methods or global state.
Saved validation certifies recorded calls/math, not an independent AD implementation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_p2_common_resource_intersection_20261029_v1 as box
from experiments.spectral_reachability import agent_p2_common_resource_grasp_cuda_20261029_v1 as common

NAME = 'agent_p2_intersection_host_cuda_20261029_v1'
AUTH = 'MAIN_P2_INTERSECTION_HOST_CUDA_ONLY'
ROOT = box.ROOT
CAP, RESERVE, FLOOR, OUTER = 1536*1024**2, 96*1024**2, 512*1024**2, 3600
need, equal = box.need, common.equal
old, wall, grasp = common.old, common.wall, common.grasp
PINS = {
    ROOT/'experiments/spectral_reachability'/ (box.NAME+'.py'):
        'fe02ea6917e49444f6bbae01b85533fb059e0739e032034e08fc11c710d332c3',
    ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py':
        'ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2',
}


def sha(path):
    p=Path(path);need(p.is_file() and not p.is_symlink(), 'regular source/artifact')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''):h.update(b)
    return h.hexdigest()


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


@lru_cache(maxsize=1)
def byte_proof():
    """Integer allocation proof; no compression-ratio assumption or result input.

    Arrays/metadata are separate hard partitions. Oversized *real* records refuse
    before writing/dispatch. These are transport limits, never extra model quota.
    """
    counts={a:sum(box.envelope(a,l)['predict_API'] for l in box.schedule(a))
            for a in ('joint','composition','grasp')}
    packets=2*sum(counts.values())
    # Original IO: AE30->576, predictor(576,576)->576, lowproj576->576.
    module_floats=30+576+576+576+576+576+576
    # Causal before/after z576+h20 and raw10, each packet; not final-token only.
    context_floats=2*(576+20)+10
    payload=packets*(4*module_floats+8*context_floats)
    jacobian=2*489600*4+2*247680*4
    # GRASP all Adam states/gradients/virtual/noise with source-specific bound.
    optimizer=2*632*(16*384*4+16*30*8)
    other_arrays=8*1024**2
    arrays_partition=368*1024**2
    need(payload+jacobian+optimizer+other_arrays < arrays_partition, 'native IO/AD/Adam fit array partition')
    # Independently exercised complete schedule census with one score per
    # non-bank unit. Five additional source scores per local unit are included
    # in the explicit 40MiB allowance below, along with optimizer native events.
    census=dict(joint=dict(a=129,aux=6,ci=11412,cr=11412,d=24940,r=2116,s=1,u=129,c=9985,done=9985,clock=72969),
      composition=dict(a=129,aux=6,ci=6372,cr=6372,d=14020,r=1276,s=1,u=129,c=5785,done=5785,clock=41049),
      grasp=dict(a=633,aux=2328,ci=11412,cr=11412,d=25992,r=3168,s=1,u=633,c=13206,done=13206,clock=78403))
    budget_max=2*sum(g['a']*2048+(g['aux']+g['ci']+g['r']+g['s'])*640+g['cr']*768+
        g['d']*1400+g['u']*1400+g['c']*512+g['done']*384+g['clock']*384 for g in census.values())
    native_events_max=packets*21*1400
    extra_metadata=40*1024**2
    extra_breakdown=dict(local_score_events=2*2*84*5*(512+2*384+1400+640+384),
        grasp_nonpacket_events=2*632*(4096+11*1400)+2*20*1400,
        spectral_jacobian_events=2*2*85*4*1400,composition_selection_events=2*84*1400,
        fixed_setup_and_failure_allowance=1024**2)
    need(sum(extra_breakdown.values())<=extra_metadata,'complete source nonpacket metadata envelope')
    journal_partition=784*1024**2
    need(budget_max+native_events_max+extra_metadata<=journal_partition,'bounded normal complete source-event census')
    rows_partition=192*1024**2
    row_max=1902*8192+85*8192+5706*1024+4000*512+1024**2+4096
    need(row_max<32*1024**2 and 6*row_max+16*1024**2<=rows_partition,'six row plus loader metadata envelope')
    partitions=arrays_partition+journal_partition+rows_partition+RESERVE
    need(partitions<=CAP, 'integer partitions <= old1536MiB; no cap expansion')
    return dict(packet_by_method=counts,packets=packets,module_floats=module_floats,
        causal_floats=context_floats,native_payload_bytes=payload,
        jacobian_bytes=jacobian,optimizer_payload_upper_bytes=optimizer,other_array_allowance=other_arrays,
        budget_event_census=census,budget_event_bytes_upper=budget_max,
        packet_native_events_per_forward_upper=21,packet_native_bytes_upper=native_events_max,
        extra_scores_optimizer_failure_metadata_allowance=extra_metadata,
        extra_metadata_breakdown=extra_breakdown,
        array_partition=arrays_partition,journal_partition=journal_partition,
        row_partition=rows_partition,row_per_slot_upper=row_max,loader_and_binding_allowance=16*1024**2,
        final_reserve=RESERVE,total_partitions=partitions,
        cap_bytes=CAP,compression_assumed=False,
        array_header_bytes=0,archive='contiguous little-endian raw; dtype/shape/offset/bytes/SHA in metadata',
        metadata_admission='separate exact byte partitions, bounded event schemas, durable write failure stops callbacks',
        full_completion_guaranteed=False,
        limit='shape/census transport bound; wall, OS failures, source-output ABI refusal may still leave a partial')


def contract():
    return dict(schema=NAME+'.contract',budget=box.protocol(),bytes=byte_proof(),
        order=[list(x) for x in box.ORDER],fixed_denominator=6,known_roots=2,fresh=False,
        raw_steps=15,context_length=3,batch=1,search_seconds=450.,outer_seconds=OUTER,
        allocated_cap_bytes=4*1024**3,reserved_cap_bytes=6*1024**3,
        whole_and_segmented=[3,3],support='unchanged raw tanh/logit30 original ball',
        criterion='original identity192 goal MSE; strict improvement, earlier tie',
        total_FLOPs=None,engine_internal_backward_calls=None,actual_cost_equality=False,
        CPU_capture_scope='module inputs and outputs only; raw/context/J/optimizer copies included in search wall',
        grasp_score_execution='source scalar terms summed in source order after three predictions; no CUDA parity assumed',
        equal_FLOPs=False,scientific_admission=False,P1_admission=False,P2_mechanism=False,
        physical_release=False,official_GRASP_reproduction=False,
        inherited_encoding_allocation=None,metadata_partition_exhaustion='partial, no refund/retry')


class Store:
    """Exclusive durable raw archive + canonical bounded JSONL; no pickle/gzip."""
    def __init__(self, output, *, clock=time.monotonic, floor=FLOOR, start=None):
        self.output=Path(output).absolute();self.clock=clock;self.start=clock() if start is None else start;self.floor=floor
        need(self.output.name.startswith(NAME+'_') and not self.output.exists()
             and not self.output.is_symlink(), 'exclusive own output')
        need(shutil.disk_usage(self.output.parent).free>=CAP+floor, 'total cap + disk floor')
        self.output.mkdir();self.totals=dict(array=0,journal=0,row=0,final=0)
        self.tail=None;self.count=0;self.cache={};self.closed=False
        self._write('arrays.bin', b'', 'array');self._write('events.jsonl', b'', 'journal')
    def check(self):need(self.clock()-self.start<OUTER, 'outer3600 guard')
    def _write(self,name,data,kind,append=False):
        need(Path(name).name==name and name not in ('','.','..') and kind in self.totals, 'flat own artifact')
        if kind!='final':self.check()
        limit=dict(array=byte_proof()['array_partition'],journal=byte_proof()['journal_partition'],
                   row=byte_proof()['row_partition'],final=RESERVE)[kind]
        need(self.totals[kind]+len(data)<=limit and sum(self.totals.values())+len(data)<=CAP,
             'hard byte partition incl partial')
        need(not (self.output/name).is_symlink() and shutil.disk_usage(self.output).free>=self.floor+len(data), 'disk floor')
        with (self.output/name).open('ab' if append else 'xb') as f:
            need(f.write(data)==len(data), 'complete durable write');f.flush();os.fsync(f.fileno())
        old.fsync_dir(self.output);self.totals[kind]+=len(data)
    def array(self,a):
        a=np.asarray(a);need(a.dtype in (np.dtype('float32'),np.dtype('float64')) and np.isfinite(a).all(), 'finite f32/f64 array')
        shape=a.shape;a=np.ascontiguousarray(a.astype(a.dtype.newbyteorder('<'),copy=False)).reshape(shape);b=a.tobytes()
        key=(str(a.dtype),tuple(a.shape),hashlib.sha256(b).hexdigest())
        if key in self.cache:return deepcopy(self.cache[key]) # output byte dedup only, never model reuse
        d=dict(file='arrays.bin',offset=self.totals['array'],bytes=len(b),dtype=str(a.dtype),shape=list(a.shape),sha256=key[2])
        self._write('arrays.bin',b,'array',True);self.cache[key]=d;return deepcopy(d)
    def encode(self,v):
        if isinstance(v,np.ndarray):return {'__array__':self.array(v),'restore':'numpy'}
        if isinstance(v,dict) and set(v)=={'shape','dtype','raw_hex','sha256'}:
            return {'__array__':self.array(old.unblob(v)),'restore':'old_blob'}
        if isinstance(v,dict) and set(v)=={'shape','dtype','hex','sha256'}:
            return {'__array__':self.array(grasp.unpack(v)),'restore':'grasp_blob'}
        if isinstance(v,dict):return {k:self.encode(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):
            # Decimal context/history arrays become lossless float64 transport.
            def only_floats(x):
                if isinstance(x,(list,tuple)):return bool(x) and all(only_floats(y) for y in x)
                return type(x) is float
            if v and only_floats(v):
                try:a=np.asarray(v)
                except (ValueError,TypeError):a=np.asarray([],dtype=object)
                if a.dtype.kind=='f' and a.size>=10:return {'__array__':self.array(a.astype(np.float64)),'restore':'list'}
            return [self.encode(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    @staticmethod
    def check_row(value):
            need(len(value['forwards'])<=1902 and len(value['AD'])<=85 and len(value['neural'])<=5706,
                 'bounded original row shape/census')
            need(all(len(box.raw(f))<=8192 for f in value['forwards']) and
                 all(len(box.raw(a))<=8192 for a in value['AD']) and
                 all(len(box.raw(n))<=1024 for n in value['neural']), 'per-record metadata bounds')
            ops=value['accounting']['operations'];need(len(ops)<=4000 and all(len(box.raw(x))<=512 for x in ops), 'budget operation metadata bound')
            need(len(box.raw({k:x for k,x in value.items() if k not in ('forwards','AD','neural','accounting')}))+
                 len(box.raw({k:x for k,x in value['accounting'].items() if k!='operations'}))<=1024**2,'bounded row remainder')
            need(len(box.raw(value))<=32*1024**2,'per-slot row partition')
    def put(self,name,v,*,final=False):
        value=self.encode(v)
        if name.startswith('arm') and name.endswith('.json') and not final:self.check_row(value)
        self._write(name,box.raw(value), 'final' if final else 'row')
    @staticmethod
    def event_limit(v):
        payload=v.get('payload',v) if v.get('kind')=='arm_event' else v
        kind=payload.get('kind')
        if kind=='clock':return 384
        if kind=='budget_command':return 512
        if kind=='budget_done':return 384
        if kind=='budget':
            k=payload['event']['kind']
            if k=='admission':return 2048
            if k in ('debit','unit_close'):return 1400
            if k=='capture_return':return 768
            return 640
        if kind=='optimizer_return':return 4096
        return 1400
    def event(self,v):
        need(not self.closed, 'sealed journal')
        e=dict(index=self.count,previous_sha256=self.tail,payload=self.encode(v));e['sha256']=box.digest(e)
        b=box.raw(e)+b'\n';need(len(b)<=self.event_limit(v), 'bounded source-specific event schema')
        self._write('events.jsonl',b,'journal',True);self.count+=1;self.tail=e['sha256']
    def close(self):self.closed=True


class Reader:
    def __init__(self, output):
        self.output=Path(output);self.used=set();self.segments={}
    def array(self,d):
        need(set(d)=={'file','offset','bytes','dtype','shape','sha256'} and d['file']=='arrays.bin', 'exact archive descriptor')
        need(d['dtype'] in ('float32','float64') and type(d['offset']) is type(d['bytes']) is int
             and d['offset']>=0 and len(d['shape'])<=3 and all(type(n) is int and 0<n<=192 for n in d['shape']), 'literal shape/offset/dtype')
        dtype=np.dtype(d['dtype']).newbyteorder('<');size=math.prod(d['shape'])*dtype.itemsize
        p=self.output/'arrays.bin';need(p.is_file() and not p.is_symlink() and size==d['bytes']
            and d['offset']+size<=p.stat().st_size, 'bounded complete slice')
        with p.open('rb') as f:f.seek(d['offset']);b=f.read(size)
        need(len(b)==size and hashlib.sha256(b).hexdigest()==d['sha256'], 'array raw SHA')
        a=np.frombuffer(b,dtype=dtype).reshape(d['shape']).copy();need(np.isfinite(a).all(), 'finite saved array')
        key=(d['offset'],size);need(key not in self.segments or self.segments[key]==d, 'no alias descriptor rebinding')
        self.segments[key]=deepcopy(d);self.used.add('arrays.bin');return a
    def decode(self,v):
        if isinstance(v,dict) and set(v)=={'file','offset','bytes','dtype','shape','sha256'} and v.get('file')=='arrays.bin':
            self.array(v);return v
        if isinstance(v,dict) and '__array__' in v:
            need(set(v)=={'__array__','restore'} and v['restore'] in ('numpy','list','old_blob','grasp_blob'), 'exact array transport')
            a=self.array(v['__array__'])
            if v['restore']=='list':return a.tolist()
            if v['restore']=='old_blob':return old.blob(a)
            if v['restore']=='grasp_blob':return grasp.pack(a)
            return a
        if isinstance(v,dict):return {k:self.decode(x) for k,x in v.items()}
        if isinstance(v,list):return [self.decode(x) for x in v]
        return v
    def read(self,name,cap):
        p=self.output/name;need(Path(name).name==name and p.is_file() and not p.is_symlink()
            and p.stat().st_size<=cap, 'bounded regular metadata')
        b=p.read_bytes();v=json.loads(b);need(b==box.raw(v), 'canonical full metadata');self.used.add(name)
        if name.startswith('arm') and name.endswith('.json'):Store.check_row(v)
        return self.decode(v)
    def events(self):
        p=self.output/'events.jsonl';need(p.stat().st_size<=byte_proof()['journal_partition'], 'journal byte cap')
        self.used.add('events.jsonl');tail=None
        with p.open('rb') as f:
            for i,line in enumerate(f):
                need(len(line)<=64*1024 and line.endswith(b'\n'), 'bounded complete line')
                e=json.loads(line);need(set(e)=={'index','previous_sha256','payload','sha256'} and e['index']==i
                    and e['previous_sha256']==tail and line==box.raw(e)+b'\n'
                    and e['sha256']==box.digest({k:v for k,v in e.items() if k!='sha256'})
                    and len(line)<=Store.event_limit(e['payload']), 'full journal chain and event bound')
                tail=e['sha256'];yield self.decode(e['payload'])
    def complete_archive(self):
        end=0
        for (offset,size),d in sorted(self.segments.items()):
            need(offset==end, 'every archive byte covered; no overlap/gap/unknown suffix');end+=size
        need(end==(self.output/'arrays.bin').stat().st_size, 'no unreferenced array suffix')


class StagedLedger(box.Ledger):
    """Frozen unit/counter mathematics, explicit input-before-dispatch capture.

    Unlike the frozen module helper, this split records both input and output
    capture. A returned kernel is durable before output transport begins.
    """
    def call(self,kind,callback,**fields):
        if kind!='predict':return super().call(kind,callback,**fields)
        need(not fields and (not self.stack or self.stack[-1]['kind'] in ('J','replay'))
             and self.unit is not None, 'packet nesting inside admitted unit')
        op=dict(index=len(self.operations),kind='predict',status='pending',modules=[],fields={},api=False)
        self.operations.append(op);self.stack.append(op)
        try:
            value=callback()
            need(op['modules']==['AE','predictor','lowproj'] and op['api'],'actual complete predict API and 3 modules')
            elapsed=self.tick();self._emit('return',operation=op['index'],elapsed=elapsed)
            op['status']='returned';return value
        except BaseException as exc:
            op['status']='failed';self.stop=self.stop or 'operation_failure'
            if not self.failed_transport:self._emit('failure',operation=op['index'],error=type(exc).__name__)
            raise
        finally:
            if self.stack and self.stack[-1] is op:self.stack.pop()
    def predict_api(self):
        need(self.stack and self.stack[-1]['kind']=='predict' and not self.stack[-1]['api']
             and self.stack[-1]['modules']==['AE'], 'one actual predict API after completed AE')
        delta=box.zeros();delta['predict_API']=1
        if any(x['kind']=='J' for x in self.stack) or (self.arm=='grasp' and self.unit!='replay'):delta['AD_primal']=1
        self._charge(delta,'predict_API');self.stack[-1]['api']=True
    def capture(self,name,callback):
        elapsed=self.tick()
        if self.stop is not None and not(self.stop=='resource_cap' and self.unit=='replay'):raise box.Refused(self.stop)
        self._emit('capture_intent',name=name,elapsed=elapsed)
        begin=float(self.clock());result,desc=callback();end=float(self.clock())
        need(math.isfinite(begin) and math.isfinite(end) and end>=begin>=self.last
             and set(desc)=={'bytes','sha256'} and type(desc['bytes']) is int and desc['bytes']>=0
             and len(desc['sha256'])==64, 'actual capture interval/descriptor')
        self.side['CPU_capture_bytes']+=desc['bytes'];self.side['CPU_capture_seconds']+=end-begin
        self._emit('capture_return',name=name,elapsed=self.tick(),descriptor=desc,capture_seconds=end-begin)
        return result,desc
    def module(self,name,callback,capture,*,shapes,input_capture):
        need(name in box.MODULE_SHAPES and self.stack and self.stack[-1]['kind']=='predict'
             and shapes==box.MODULE_SHAPES[name], 'actual native tensor ABI inside predict')
        parent=self.stack[-1];need(name==('AE','predictor','lowproj')[len(parent['modules'])], 'native module order')
        need(name=='AE' or parent['api'], 'actual predict API entered before predictor/projection')
        d=box.zeros();d[name+'_attempts']=1;self._charge(d,'module_attempt',name=name)
        self.capture(name+'/inputs',input_capture)
        d=box.zeros();d[name+'_dispatched']=1
        if name=='AE':d.update(AE_rows=1,AE_tokens=3)
        elif name=='predictor':d.update(predictor_samples=1,predictor_context_tokens=3)
        else:d['lowproj_rows']=3
        self._charge(d,'module_dispatch',name=name)
        result=callback()
        d=box.zeros();d[name+'_returned']=1;self._record_completion(d,'module_return',name=name)
        _,desc=self.capture(name+'/output',lambda:capture(result))
        d=box.zeros();d[name+'_captured']=1
        self._record_completion(d,'module_capture',name=name,descriptor=desc)
        parent['modules'].append(name);return result


class Tape:
    """Durable clock + typed command tape, replayed by the real saved algorithm.

    No trusting declared deltas: the saved side re-runs StagedLedger methods.
    """
    def __init__(self,root,arm,emit,clock):
        self.emit,self.real_clock=emit,clock
        def tick():
            x=float(clock());emit(dict(kind='clock',value=x));return x
        self.ledger=StagedLedger(root,arm,clock=tick,sink=lambda e:emit(dict(kind='budget',event=e)))
    def invoke(self,method,*args,callback=None,capture=None,input_capture=None,**kw):
        self.emit(dict(kind='budget_command',method=method,args=list(args),kwargs=kw))
        try:
            if method=='call':r=self.ledger.call(*args,callback,**kw)
            elif method=='auxiliary':r=self.ledger.auxiliary(*args,callback)
            elif method=='module':r=self.ledger.module(*args,callback,capture,input_capture=input_capture,**kw)
            else:r=getattr(self.ledger,method)(*args,**kw)
        except BaseException as exc:
            self.emit(dict(kind='budget_exception',type=type(exc).__name__));raise
        self.emit(dict(kind='budget_done',result=r if method in ('admit','commit') else None));return r
    def call(self,kind,fn,**fields):return self.invoke('call',kind,callback=fn,**fields)
    def auxiliary(self,kind,fn):return self.invoke('auxiliary',kind,callback=fn)
    def module(self,name,fn,capture,*,shapes,input_capture):
        return self.invoke('module',name,callback=fn,capture=capture,input_capture=input_capture,shapes=shapes)
    def admit(self,label):return self.invoke('admit',label)
    def commit(self):return self.invoke('commit')
    def predict_api(self):return self.invoke('predict_api')


class Cursor:
    def __init__(self,events):self.events=iter(events);self.i=0;self.pending=None
    def peek(self):
        if self.pending is None:
            try:self.pending=next(self.events)
            except StopIteration:raise ValueError('missing saved event') from None
        return self.pending
    def get(self,kind):
        v=self.peek();self.pending=None;self.i+=1
        need(v['kind']==kind, 'unfiltered chronology expected '+kind+' got '+str(v.get('kind')));return v
    def emit(self,e):equal(self.get(e['kind']),e,'entire saved event/fields')
    def clock(self):
        v=self.get('clock');need(set(v)=={'kind','value'} and type(v['value']) in (int,float)
            and math.isfinite(v['value']), 'actual synchronized clock sample');return v['value']
    def complete(self):
        need(self.pending is None, 'unconsumed saved event')
        try:next(self.events)
        except StopIteration:return
        raise ValueError('unknown trailing saved events')


class SavedTape(Tape):
    def __init__(self,root,arm,cursor):
        self.emit=cursor.emit;self.real_clock=cursor.clock
        self.ledger=StagedLedger(root,arm,clock=cursor.clock,sink=lambda e:cursor.emit(dict(kind='budget',event=e)))


def packed(a):return grasp.pack(np.asarray(a))


def capture_desc(values):
    return dict(bytes=sum((old.unblob(d) if 'raw_hex' in d else grasp.unpack(d)).nbytes for d in values),sha256=old.digest(values))


class ModelFacade:
    """Local wrappers only: installed original LeWM.predict, untouched graph.

    No monkeypatching hooks/forward/predict; no parameter copy or new network.
    """
    def __init__(self,model,backend):self.original=model;self.b=backend
    def modules(self):return self.original.modules()
    def parameters(self):return self.original.parameters()
    def action_encoder(self,x):return self.b.module('AE',self.original.action_encoder,(x,))
    def predictor(self,z,a):return self.b.module('predictor',self.original.predictor,(z,a))
    def pred_proj(self,x):return self.b.module('lowproj',self.original.pred_proj,(x,))
    def predict(self,z,a):
        import inspect
        method=self.original.predict.__func__
        source=ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py'
        need(method.__qualname__=='LeWM.predict' and method.__module__=='stable_worldmodel.wm.lewm.lewm'
             and Path(inspect.getsourcefile(method)).resolve()==source,
             'only authenticated installed LeWM.predict body')
        self.b.budget.predict_api()
        if not getattr(self.b,'grasp_mode',False):self.b.event(dict(kind='predict_API_intent',forward=self.b.current['index'],completed=False))
        out=method(self,z,a)
        if not getattr(self.b,'grasp_mode',False):self.b.event(dict(kind='predict_API_return',forward=self.b.current['index'],completed=True))
        return out


class HostBackend(old.HostBackend):
    def __init__(self,model,device,t,store,row,event,budget):
        self.budget=budget;self.original_model=model
        super().__init__(ModelFacade(model,self),device,t,store,row,event)
    @contextmanager
    def observe(self):
        # All three calls pass through a local facade, not mutable model hooks.
        yield
    def check(self):
        self.store.check();self.t.cuda.synchronize(self.device)
        a=self.t.cuda.memory_allocated(self.device);r=self.t.cuda.memory_reserved(self.device)
        need(a<=4*1024**3 and r<=6*1024**3, 'actual allocator4/6GiB')
        self.event(dict(kind='memory',allocated=a,reserved=r))
    def module(self,name,fn,args):
        shapes=[list(x.shape) for x in args];inputs=[];output=[]
        ordinal=3*self.current['index']+len(self.current['modules'])
        resource={'AE':'action_encoder','predictor':'predictor','lowproj':'pred_proj'}[name]
        def incoming():
            self.check();inputs.extend(self.array(x) for x in args)
            self.event(dict(kind='module_intent',index=ordinal,resource=resource,completed=False))
            self.event(dict(kind='module_inputs',index=ordinal,inputs=inputs))
            return None,capture_desc(inputs)
        def outgoing(value):
            expected=(3,192) if name=='lowproj' else (1,3,192)
            need(tuple(value.shape)==expected,'native module exact output shape before capture')
            self.check();output.append(self.array(value))
            entry=old.module_receipt(ordinal,resource,deepcopy(inputs),output[0])
            self.current['modules'].append(entry);self.event(dict(kind='module_return',entry=entry))
            return None,capture_desc(output)
        return self.budget.module(name,lambda:fn(*args),outgoing,shapes=shapes,input_capture=incoming)
    def _rollout(self,c,commands,tag,raw_tensor=None):
        t=self.t;state=t.tensor(c['latent_history'],device=self.device,dtype=t.float32)[None]
        history=t.tensor(c['action_history'],device=self.device,dtype=t.float32)[None]
        actions=t.tensor(commands,device=self.device,dtype=t.float32) if raw_tensor is None else raw_tensor
        paths=[]
        for k in range(len(commands)):
            def packet():
                nonlocal state,history,c
                self.check();a=actions[k];a_np=a.detach().cpu().numpy().copy();n=len(self.row['forwards'])
                req=wall.packet_request(n,c,a_np,tag+'/'+str(k))
                self.event(dict(kind='forward_intent',request=req))
                f=dict(index=n,tag=tag+'/'+str(k),request=req,input=deepcopy(c),raw=old.blob(a_np),modules=[],completed=False)
                self.row['forwards'].append(f);self.current=f
                self.event(dict(kind='forward_inputs',index=n,context=c,raw=f['raw']))
                state,history=self.transition(state,history,a)
                c=dict(c,latent_history=state[0].detach().cpu().tolist(),action_history=history[0].detach().cpu().tolist())
                f.update(completed=True,output=deepcopy(c));wall.validate_packet(f)
                self.event(dict(kind='forward_return',index=n,context=c,completed=True));self.current=None
            self.budget.call('predict',packet);paths.append(deepcopy(c))
        return state,paths
    def jacobian(self,c,u,tag,logit):
        def work():
            self.check();n=len(self.row['AD'])
            req=dict(index=n,tag=tag,input_context_sha256=old.digest(c),at_logits=np.asarray(u).tolist(),raw=old.chart(u).tolist(),
                coordinates='logit_zero' if logit else 'raw_incumbent',shape=[192,len(u)],mathematical_VJP=192,vectorize=True,completed=False)
            self.event(dict(kind='AD_intent',request=req));entry=dict(request=req,completed=False);self.row['AD'].append(entry)
            t=self.t;x=t.zeros(len(u),device=self.device,dtype=t.float32) if logit else t.tensor(old.chart(u).reshape(-1),device=self.device,dtype=t.float32)
            def fn(value):
                a=t.tanh(value).reshape(-1,5,2) if logit else value.reshape(-1,5,2)
                state,path=self._rollout(c,old.chart(u),tag+'/forward',a);entry['contexts']=path;return state[0,-1]
            with t.enable_grad():j=t.autograd.functional.jacobian(fn,x.requires_grad_(True),vectorize=True)
            self.check();a=j.detach().cpu().numpy().copy();desc=self.store.array(a)
            entry.update(array=desc,completed=True);self.last_ad_contexts=deepcopy(entry['contexts'])
            self.event(dict(kind='AD_return',index=n,array=desc,contexts=entry['contexts'],completed=True));return a.astype(np.float64)
        return self.budget.call('J',work,horizon=len(u)//10,dimension=len(u))
    def cpu(self,kind,fn,*args,**kw):
        return self.budget.auxiliary(kind,lambda:fn(*args,**kw))


def score(budget,path,goal,scope='root_full_path'):
    return budget.call('score',lambda:old.objective(path[-1],goal),trajectories=1,scope=scope)


def local_step(context,prefix,u,goal,b,tag,joint):
    """Frozen proposal/criterion, now with every scalar evaluation paid explicitly."""
    u=np.asarray(u,np.float64);path=b.rollout(context,old.chart(u),tag+'/incumbent')
    j=b.jacobian(context,u,tag+'/J',False);equal(b.last_ad_contexts,path,'literal AD nominal context')
    if joint:
        z=old.finite(path[-1]['latent_history'][-1],(192,));g=old.finite(goal,(192,))
        grad=(2/192)*(j*(1-np.tanh(u)**2)[None,:]).T@(z-g);norm=float(np.linalg.norm(grad))
        target=u.copy() if norm==0 else -.5*old.CAP_LOGIT*grad/norm;direction=target-u
        if float(grad@direction)>=0:direction=np.zeros_like(u)
        values=[u]+[u+a*direction for a in old.ALPHAS]
        need(all(old.legal([],v) for v in values), 'same full30 proposal support')
    else:values=old.proposals(prefix,u,path[-1]['latent_history'][-1],j,goal)['logits']
    scope='root_full_path' if joint else 'serial_terminal'
    evaluations=[dict(index=0,logits=u.tolist(),contexts=path,cost=score(b.budget,path,goal,scope))]
    for k,v in enumerate(values[1:],1):
        p=b.rollout(context,old.chart(v),tag+'/candidate'+str(k))
        evaluations.append(dict(index=k,logits=np.asarray(v).tolist(),contexts=p,cost=score(b.budget,p,goal,scope)))
    winner=min(evaluations,key=lambda e:(e['cost'],e['index']))
    if not joint:
        proposals=old.proposals(prefix,u,path[-1]['latent_history'][-1],j,goal)
        b.event(dict(kind='local_selection',tag=tag,selected=winner['index'],costs=[e['cost'] for e in evaluations],
            input_context_sha256=old.digest(context),goal_sha256=old.digest(goal),proposals_sha256=old.digest(proposals)))
    replay=b.rollout(context,old.chart(winner['logits']),tag+'/selected_replay')
    equal(replay,winner['contexts'],'selected literal replay; no rescoring/repair')
    return dict(logits=winner['logits'],contexts=replay)


def drive_spectral(slot,arm,b):
    budget=b.budget;c=slot['packet']['root_context'];g=slot['packet']['goal']
    states={};nominal=bank=None;incumbent=None;export=None;aborted=None
    labels=iter(box.schedule(arm));forced_replay=False
    while True:
        try:label='replay' if forced_replay else next(labels)
        except StopIteration:break
        if not budget.admit(label):
            if budget.ledger.stop=='resource_cap' and label!='replay' and incumbent is not None:
                forced_replay=True;continue
            break
        witness=None
        try:
            if label=='nominal':
                nominal=b.rollout(c,old.chart(np.zeros(30)),label);witness=dict(logits=[0.]*30,contexts=nominal)
            elif label=='bank':
                j=b.jacobian(c,np.zeros(30),'bank',True);equal(b.last_ad_contexts,nominal,'bank nominal all contexts')
                bank=wall.make_bank(j,nominal[-1]['latent_history'][-1],g,b)
            elif label.startswith('seed:'):
                i=int(label.split(':')[1]);u=bank['params'][i]
                states[i]=dict(logits=u,contexts=b.rollout(c,old.chart(u),label));witness=deepcopy(states[i])
            elif label.startswith('joint:'):
                i=int(label.split(':')[2]);states[i]=local_step(c,[],states[i]['logits'],g,b,label,True);witness=deepcopy(states[i])
            elif label.startswith('H2:'):
                i=int(label.split(':')[1]);u=np.asarray(states[i]['logits'],np.float64)
                first=b.rollout(c,old.chart(u[:10]),label+'/prefix')[0]
                r=local_step(first,u[:10],u[10:],g,b,label,False)
                states[i]=dict(logits=np.concatenate((u[:10],r['logits'])).tolist(),contexts=[first]+r['contexts']);witness=deepcopy(states[i])
            elif label.startswith('H1:'):
                i=int(label.split(':')[1]);u=np.asarray(states[i]['logits'],np.float64);first,mid=states[i]['contexts'][:2]
                r=local_step(mid,u[:20],u[20:],g,b,label,False)
                states[i]=dict(logits=np.concatenate((u[:20],r['logits'])).tolist(),contexts=[first,mid]+r['contexts']);witness=deepcopy(states[i])
            else:
                need(label=='replay' and incumbent is not None,'final replay requires complete incumbent')
                whole=budget.call('replay',lambda:b.rollout(c,old.chart(incumbent['witness']['logits']),'export/whole'),part='whole')
                def segments():
                    current=deepcopy(c);paths=[]
                    for k,a in enumerate(old.chart(incumbent['witness']['logits'])):
                        current=b.rollout(current,a[None],'export/segment'+str(k))[0];paths.append(current)
                    return paths
                segmented=budget.call('replay',segments,part='segmented')
                equal(whole,segmented,'all3 final fresh contexts');equal(whole,incumbent['witness']['contexts'],'incumbent replay')
            q=None if witness is None else score(budget,witness['contexts'],g)
            accepted=budget.commit()
            if accepted and witness is not None and (incumbent is None or q<incumbent['cost']):
                incumbent=dict(cost=q,witness=witness,label=label)
            if accepted and label=='replay':export=deepcopy(incumbent['witness']);break
            if not accepted:break
        except (box.Refused,ValueError,RuntimeError,OSError) as exc:
            aborted=label
            if budget.ledger.stop is None:budget.ledger.stop='operation_failure'
            b.event(dict(kind='algorithm_exception',type=type(exc).__name__,message=str(exc)));break
    return dict(export=export,aborted_unit=aborted,incumbent=incumbent)


class GraspBackend(grasp.Native):
    grasp_mode=True
    def __init__(self,model,device,task,event,store,budget,row):
        super().__init__(model,device,task,event,store)
        self.original_model=model;self.model=ModelFacade(model,self);self.budget=budget;self.row=row;self.current=None
    def check(self):
        self.store.check();self.t.cuda.synchronize(self.device)
        a=self.t.cuda.memory_allocated(self.device);r=self.t.cuda.memory_reserved(self.device)
        need(a<=4*1024**3 and r<=6*1024**3,'actual allocator4/6GiB')
        self.event(dict(kind='memory',allocated=a,reserved=r))
    @contextmanager
    def observe(self):yield
    def module(self,name,fn,args):
        inputs=[];resource={'AE':'action_encoder','predictor':'predictor','lowproj':'pred_proj'}[name]
        shapes=[list(x.shape) for x in args];entry=dict(index=self.n,resource=resource,inputs=None,output=None)
        self.row['neural'].append(entry)
        def incoming():
            self.check();inputs.extend(self.snap(x) for x in args);entry['inputs']=deepcopy(inputs)
            self.event(dict(kind='module_attempt',index=self.n,resource=resource,inputs=deepcopy(inputs)))
            return None,capture_desc(inputs)
        def outgoing(value):
            expected=(3,192) if name=='lowproj' else (1,3,192)
            need(tuple(value.shape)==expected,'native module exact output shape before capture')
            self.check();output=self.snap(value);entry['output']=output
            self.event(dict(kind='module_return',index=self.n,resource=resource,output=output));self.active+=1
            return None,capture_desc([output])
        return self.budget.module(name,lambda:fn(*args),outgoing,shapes=shapes,input_capture=incoming)
    def step(self,z,h,a,label):
        self.current=dict(index=self.n)
        try:return self.budget.call('predict',lambda:super(GraspBackend,self).step(z,h,a,label))
        finally:self.current=None
    def soft(self,v,x):
        # Original lifted objective, all source-order f32 additions unchanged.
        # Charge its actual scalar evaluation once (rather than a later no-op).
        t=self.t;z,h=self.root();a=self.exactraw(x);goal=self.tensor(self.task['goal'])[None]
        pairs=[];dyn=goal.new_zeros(());dense=goal.new_zeros(())
        for k in range(3):
            z,h=self.step(z.detach(),h,a[k],'lifted/'+str(k));pred=z[:,-1]
            pairs.append((pred,v[:,k] if k<2 else goal))
            if k<2:z=t.cat((z[:,:-1],v[:,k:k+1]),1)
        def evaluate():
            nonlocal dyn,dense
            for pred,target in pairs:
                dyn=dyn+(pred-target).square().mean();dense=dense+(pred-goal).square().mean()
            return dyn+dense,dyn,dense
        return self.budget.call('score',evaluate,trajectories=1,scope='lifted_loss')


def drive_grasp(task,slot,b):
    """Literal532 joint/100 sync source algorithm, split before each real operation."""
    t=b.t;budget=b.budget;z,_=b.root();goal=b.tensor(task['goal'])[None]
    fraction=t.linspace(0,1,4,device=b.device)[1:-1][None,:,None]
    v=(z[:,-1,None]+fraction*(goal[:,None]-z[:,-1,None])).detach().requires_grad_(True)
    x=t.zeros(30,device=b.device,dtype=t.float64,requires_grad=True)
    t.manual_seed(grasp.seed(slot));t.cuda.manual_seed_all(grasp.seed(slot))
    def adam(joint):
        return t.optim.Adam([{'params':[v],'lr':.1},{'params':[x],'lr':.005}] if joint else [x],
            lr=.005 if joint else .05,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False)
    opt=adam(True);sopt=None;snapshot=np.zeros(30,np.float64);export=None;failure=None
    b.event(dict(kind='initialized',seed=grasp.seed(slot),fraction=b.snap(fraction),x=b.snap(x),v=b.snap(v)))
    iterator=iter(common.phases());force_replay=False
    while True:
        if force_replay:phase,k,j='replay',None,None
        else:
            try:phase,k,j=next(iterator)
            except StopIteration:phase,k,j='replay',None,None
        label='replay' if phase=='replay' else f'joint:{k}' if phase=='joint' else f'sync:{k}:{j}'
        if not budget.admit(label):
            if budget.ledger.stop=='resource_cap' and phase!='replay' and budget.ledger.committed:
                force_replay=True;continue
            break
        try:
            if phase=='replay':
                b.event(dict(kind='actions_frozen',x=grasp.pack(snapshot),commands=grasp.pack(grasp.chart(snapshot).reshape(15,2))))
                with t.no_grad():
                    sx=b.tensor(snapshot,t.float64)
                    _,whole=budget.call('replay',lambda:b.serial(sx,'whole'),part='whole')
                    _,seg=budget.call('replay',lambda:b.serial(sx,'segmented',True),part='segmented')
                equal(whole,seg,'all3 fresh replay boundaries')
                witness=dict(logits=snapshot.tolist(),contexts=whole)
                accepted=budget.commit()
                b.event(dict(kind='export_complete',witness_sha256=old.digest(witness),accepted=accepted))
                if accepted:export=witness
                break
            if phase=='sync' and j==0:sopt=adam(False)
            current=opt if phase=='joint' else sopt;current.zero_grad(set_to_none=True)
            b.event(dict(kind='optimization_attempt',phase=phase,iteration=k,sync_step=j,x=b.snap(x),v=b.snap(v)))
            if phase=='joint':loss,dyn,dense=b.soft(v,x)
            else:
                terminal,_=b.serial(x,'sync')
                loss=budget.call('score',lambda:(terminal-goal).square().mean(),trajectories=1,scope='serial_terminal');dyn=dense=None
            def backward():
                b.event(dict(kind='backward_attempt',phase=phase,iteration=k,sync_step=j,loss=b.snap(loss),
                    dyn=None if dyn is None else b.snap(dyn),dense=None if dense is None else b.snap(dense)))
                need(bool(t.isfinite(loss)),'finite objective');loss.backward();b.check()
                need(x.grad is not None and bool(t.isfinite(x.grad).all()) and
                    (phase!='joint' or v.grad is not None and bool(t.isfinite(v.grad).all())),'finite gradients')
                b.event(dict(kind='backward_return',gx=b.snap(x.grad),gv=b.snap(v.grad) if phase=='joint' else None))
            budget.call('backward',backward)
            def update():
                bx=grasp.optimizer_snapshot(current,x,b.snap);bv=grasp.optimizer_snapshot(current,v,b.snap) if phase=='joint' else None
                b.event(dict(kind='optimizer_attempt',phase=phase,iteration=k,sync_step=j))
                current.step();b.check();proposal=b.snap(x)
                with t.no_grad():x.copy_(b.tensor(grasp.project(x.detach().cpu().numpy()),t.float64))
                b.event(dict(kind='optimizer_return',x_before=bx,v_before=bv,x_after=grasp.optimizer_snapshot(current,x,b.snap),
                    v_after=grasp.optimizer_snapshot(current,v,b.snap) if phase=='joint' else None,proposal=proposal,x=b.snap(x),v=b.snap(v)))
            budget.auxiliary('Adam_step',update)
            if phase=='joint':
                def noise_step():
                    b.event(dict(kind='noise_attempt'))
                    with t.no_grad():noise=.005*t.randn_like(v);v.add_(noise)
                    b.event(dict(kind='noise',noise=b.snap(noise),v=b.snap(v)))
                budget.auxiliary('virtual_noise',noise_step)
            if phase=='sync' and j==9:
                x=x.detach().clone().requires_grad_(True);opt=adam(True)
                b.event(dict(kind='joint_optimizer_reset',after=k+1,x=b.snap(x),v=b.snap(v)))
            saved=b.snap(x)
            if budget.commit():snapshot=grasp.unpack(saved,(30,),np.float64).copy()
            else:break
        except (box.Refused,ValueError,RuntimeError,OSError) as exc:
            failure=dict(type=type(exc).__name__,message=str(exc))
            if budget.ledger.stop is None:budget.ledger.stop='operation_failure'
            b.event(dict(kind='algorithm_exception',**failure));break
    return dict(export=export,error=failure)


class SavedSpectral:
    """Replay old math using saved full native IO/J; performs no model calls."""
    def __init__(self,row,budget,cursor,reader):
        self.row,self.budget,self.cursor,self.reader=row,budget,cursor,reader
        self.n=self.ad=0
    def event(self,e):self.cursor.emit(e)
    def check(self):
        e=self.cursor.get('memory')
        need(set(e)=={'kind','allocated','reserved'} and type(e['allocated']) is type(e['reserved']) is int
             and 0<=e['allocated']<=e['reserved'] and e['allocated']<=4*1024**3
             and e['reserved']<=6*1024**3,'saved actual allocator sample')
    def module(self,name,inputs,entry,ordinal):
        mapping={'AE':'action_encoder','predictor':'predictor','lowproj':'pred_proj'};resource=mapping[name]
        if entry is not None:
            need(entry['index']==ordinal and entry['resource']==resource and entry['completed'] is True,'exact saved module ordinal')
            equal(entry['inputs'],inputs,'actual source module inputs');out=entry['output']
        else:out=None # Sentinel only; cannot be consumed as a model output.
        shapes=[list(old.unblob(x).shape) for x in inputs]
        def incoming():
            self.check();self.event(dict(kind='module_intent',index=ordinal,resource=resource,completed=False))
            self.event(dict(kind='module_inputs',index=ordinal,inputs=inputs));return None,capture_desc(inputs)
        def outgoing(value):
            need(out is not None,'missing actual output cannot be captured')
            self.check();equal(value,out,'saved output bytes');self.event(dict(kind='module_return',entry=entry))
            return None,capture_desc([value])
        result=self.budget.module(name,lambda:out,outgoing,shapes=shapes,input_capture=incoming)
        need(result is not None,'missing output cannot advance state');return old.unblob(result)
    def rollout(self,c,commands,tag):
        paths=[]
        for k,a in enumerate(commands):
            def packet():
                nonlocal c
                self.check();need(self.n<len(self.row['forwards']),'actual forward exists')
                f=self.row['forwards'][self.n];need(f['index']==self.n,'contiguous actual forward ordinal');self.n+=1
                req=wall.packet_request(f['index'],c,np.asarray(a,np.float32),tag+'/'+str(k))
                equal(f['request'],req,'scheduled context/action/tag');equal(f['input'],c,'complete context input')
                need(old.unblob(f['raw']).tobytes()==np.asarray(a,np.float32).tobytes(),'exact raw action bits')
                self.event(dict(kind='forward_intent',request=req))
                self.event(dict(kind='forward_inputs',index=f['index'],context=c,raw=f['raw']))
                need(type(f['completed']) is bool and len(f['modules'])<=3,'bounded actual module prefix')
                tokens=np.concatenate((np.asarray(c['action_history'],np.float32),old.kernel.normalized_history(np.asarray(a,np.float32))))[None]
                z=np.asarray(c['latent_history'],np.float32)[None];outputs=[]
                for i,name in enumerate(('AE','predictor','lowproj')):
                    if i==1:
                        self.budget.predict_api();self.event(dict(kind='predict_API_intent',forward=f['index'],completed=False))
                    inputs=[old.blob(tokens)] if i==0 else [old.blob(z),old.blob(outputs[0])] if i==1 else [old.blob(outputs[1].reshape(3,192))]
                    entry=f['modules'][i] if i<len(f['modules']) else None
                    outputs.append(self.module(name,inputs,entry,3*f['index']+i))
                    if i==2:self.event(dict(kind='predict_API_return',forward=f['index'],completed=True))
                need(f['completed'] is True and len(f['modules'])==3,'missing packet cannot advance')
                c=wall.validate_packet(f)
                self.event(dict(kind='forward_return',index=f['index'],context=c,completed=True))
            self.budget.call('predict',packet);paths.append(deepcopy(c))
        return paths
    def jacobian(self,c,u,tag,logit):
        def work():
            self.check();need(self.ad<len(self.row['AD']),'actual J exists')
            e=self.row['AD'][self.ad];n=self.ad;self.ad+=1
            req=dict(index=n,tag=tag,input_context_sha256=old.digest(c),at_logits=np.asarray(u).tolist(),raw=old.chart(u).tolist(),
                coordinates='logit_zero' if logit else 'raw_incumbent',shape=[192,len(u)],mathematical_VJP=192,vectorize=True,completed=False)
            equal(e['request'],req,'fixed J request');self.event(dict(kind='AD_intent',request=req))
            paths=self.rollout(c,old.chart(u),tag+'/forward');equal(e['contexts'],paths,'actual primal context')
            self.check();need(e['completed'] is True,'partial J cannot certify complete');a=self.reader.array(e['array'])
            need(a.shape==(192,len(u)) and a.dtype==np.float32,'original full native J array')
            self.last_ad_contexts=deepcopy(paths)
            self.event(dict(kind='AD_return',index=n,array=e['array'],contexts=paths,completed=True));return a.astype(np.float64)
        return self.budget.call('J',work,horizon=len(u)//10,dimension=len(u))
    def cpu(self,kind,fn,*args,**kw):return self.budget.auxiliary(kind,lambda:fn(*args,**kw))
    def complete(self):
        need(self.n==len(self.row['forwards']) and self.ad==len(self.row['AD']),'all recorded forward/J consumed')


class SavedGrasp(grasp.Saved):
    """Saved source IO replay with paid real-shape module callbacks (no NN)."""
    def __init__(self,task,cursor,budget,row):
        self.task,self.c,self.budget,self.row=task,cursor,budget,row;self.n=0;self.module_index=0
    def check(self):
        e=self.c.get('memory');need(set(e)=={'kind','allocated','reserved'} and type(e['allocated']) is type(e['reserved']) is int
            and 0<=e['allocated']<=e['reserved'] and e['allocated']<=4*1024**3 and e['reserved']<=6*1024**3,'actual allocator')
    def event(self,e):self.c.emit(e)
    def module(self,name,inputs):
        need(self.module_index<len(self.row['neural']),'actual module output exists')
        e=self.row['neural'][self.module_index];self.module_index+=1
        resource={'AE':'action_encoder','predictor':'predictor','lowproj':'pred_proj'}[name]
        need(e['index']==self.n and e['resource']==resource,'actual native module ordinal')
        packs=[grasp.pack(x) for x in inputs]
        if e['inputs'] is not None:equal(e['inputs'],packs,'module input bits source wiring')
        output=e['output'];shapes=[list(x.shape) for x in inputs]
        def incoming():
            self.check();self.event(dict(kind='module_attempt',index=self.n,resource=resource,inputs=packs))
            return None,capture_desc(packs)
        def outgoing(value):
            need(output is not None,'missing actual output cannot be captured')
            self.check();equal(value,output,'module output bytes');self.event(dict(kind='module_return',index=self.n,resource=resource,output=output))
            return None,capture_desc([value])
        self.budget.module(name,lambda:output,outgoing,shapes=shapes,input_capture=incoming)
        need(output is not None,'missing output cannot advance state')
        a=grasp.unpack(output,(3,192) if name=='lowproj' else (1,3,192),np.float32);return a
    def step(self,z,h,a,label):
        def packet():
            self.check();self.event(dict(kind='forward_attempt',index=self.n,label=label,z=grasp.pack(z),h=grasp.pack(h),raw=grasp.pack(a)))
            mean,std=(np.tile(self.task[k],5).astype(np.float32) for k in ('action_mean','action_std'))
            tokens=(a.reshape(10)-mean)/std;outgoing=np.concatenate((h,tokens[None,None]),1)
            ae=self.module('AE',[outgoing]);self.event(dict(kind='predict_attempt',index=self.n));self.budget.predict_api()
            pred=self.module('predictor',[z,ae]);projected=self.module('lowproj',[pred.reshape(3,192)]).reshape(1,3,192)
            self.check();self.event(dict(kind='predict_return',index=self.n,output=grasp.pack(projected)))
            after=np.concatenate((z[:,1:],projected[:,-1:]),1);between=outgoing[:,1:]
            self.event(dict(kind='forward_return',index=self.n,z=grasp.pack(after),h=grasp.pack(between)))
            self.n+=1;return after,between
        return self.budget.call('predict',packet)


def saved_grasp(task,slot,b):
    """Reconstruct scalar loss, chart derivative, Adam/noise and paid IO.

    Gradient witness is source-consistency evidence, not independent CUDA AD.
    """
    budget=b.budget;c=b.c;u=lambda d,s,dtype=np.float32:grasp.unpack(d,s,np.dtype(dtype))
    x=np.zeros(30,np.float64);snapshot=x.copy();goal=np.asarray(task['goal'],np.float32)[None]
    root=task['root_context'];e=c.get('initialized')
    need(set(e)=={'kind','seed','fraction','x','v'} and e['seed']==grasp.seed(slot),'fixed init seed/fields')
    f=u(e['fraction'],(1,2,1));common.rounding(f,np.asarray([1/3,2/3],np.float32).reshape(1,2,1),8)
    equal(e['x'],grasp.pack(x),'zero init');v=u(e['v'],(1,2,192))
    need(np.array_equal(v,np.asarray(root['latent_history'],np.float32)[-1,None,None]+f*(goal[:,None]-np.asarray(root['latent_history'],np.float32)[-1,None,None])), 'fixed virtual init')
    mx=common.adam_reference.initial_memory(x);mv=common.adam_reference.initial_memory(v)
    export=None;failure=None;iterator=iter(common.phases());force_replay=False
    while True:
        if force_replay:phase,k,j='replay',None,None
        else:
            try:phase,k,j=next(iterator)
            except StopIteration:phase,k,j='replay',None,None
        label='replay' if phase=='replay' else f'joint:{k}' if phase=='joint' else f'sync:{k}:{j}'
        if not budget.admit(label):
            if budget.ledger.stop=='resource_cap' and phase!='replay' and budget.ledger.committed:force_replay=True;continue
            break
        try:
            if phase=='replay':
                b.event(dict(kind='actions_frozen',x=grasp.pack(snapshot),commands=grasp.pack(grasp.chart(snapshot).reshape(15,2))))
                _,whole=budget.call('replay',lambda:b.serial(snapshot,'whole'),part='whole')
                _,seg=budget.call('replay',lambda:b.serial(snapshot,'segmented',True),part='segmented')
                equal(whole,seg,'complete saved segmented replay')
                w=dict(logits=snapshot.tolist(),contexts=whole);accepted=budget.commit()
                b.event(dict(kind='export_complete',witness_sha256=old.digest(w),accepted=accepted))
                if accepted:export=w
                break
            if phase=='sync' and j==0:mx=common.adam_reference.initial_memory(x)
            b.event(dict(kind='optimization_attempt',phase=phase,iteration=k,sync_step=j,x=grasp.pack(x),v=grasp.pack(v)))
            preds=b.soft(v,x) if phase=='joint' else [b.serial(x,'sync')[0]]
            def score_expected():
                mse=lambda a,bb:np.mean((a-bb)**2,dtype=np.float32)
                if phase!='joint':return (mse(preds[-1],goal),None,None)
                dyn=np.float32(0);dense=np.float32(0)
                for kk,pred in enumerate(preds):
                    dyn=np.float32(dyn+mse(pred,v[:,kk] if kk<2 else goal));dense=np.float32(dense+mse(pred,goal))
                return np.float32(dyn+dense),dyn,dense
            expected_loss=budget.call('score',score_expected,trajectories=1,scope='lifted_loss' if phase=='joint' else 'serial_terminal')
            holder={}
            def backward():
                loss=c.get('backward_attempt');need(set(loss)=={'kind','phase','iteration','sync_step','loss','dyn','dense'}
                    and (loss['phase'],loss['iteration'],loss['sync_step'])==(phase,k,j),'actual loss request')
                if phase=='joint':
                    common.rounding(u(loss['dyn'],()),expected_loss[1],256);common.rounding(u(loss['dense'],()),expected_loss[2],256)
                    common.rounding(u(loss['loss'],()),expected_loss[0],512)
                else:
                    need(loss['dyn'] is loss['dense'] is None,'sync loss fields');common.rounding(u(loss['loss'],()),expected_loss[0],256)
                rawgrad=c.get('chart_backward');need(set(rawgrad)=={'kind','raw_gradient'},'one chart gradient')
                gr=u(rawgrad['raw_gradient'],(3,5,2));b.check();grad=c.get('backward_return')
                need(set(grad)=={'kind','gx','gv'},'actual returned grads');gx=u(grad['gx'],(30,),np.float64)
                need(np.array_equal(gx,gr.reshape(30).astype(np.float64)*(1-np.tanh(x)**2)),'exact raw chart derivative')
                gv=None
                if phase=='joint':gv=u(grad['gv'],(1,2,192));common.rounding(gv,np.stack([2/192*(v[:,kk]-preds[kk]) for kk in range(2)],1),32)
                else:need(grad['gv'] is None,'no virtual sync gradient')
                holder.update(gx=gx,gv=gv)
            budget.call('backward',backward)
            def update():
                nonlocal x,v,mx,mv
                b.event(dict(kind='optimizer_attempt',phase=phase,iteration=k,sync_step=j));b.check()
                op=c.get('optimizer_return');need(set(op)=={'kind','x_before','v_before','x_after','v_after','proposal','x','v'},'full Adam evidence')
                equal(op['x_before'],mx,'action Adam state');proposal=u(op['proposal'],(30,),np.float64)
                common.check_adam(mx,op['x_after'],x,holder['gx'],proposal,.005 if phase=='joint' else .05);mx=op['x_after']
                xx=u(op['x'],(30,),np.float64);need(np.array_equal(xx,grasp.project(proposal)),'actual global30 projection')
                vv=u(op['v'],(1,2,192))
                if phase=='joint':
                    equal(op['v_before'],mv,'virtual Adam state');common.check_adam(mv,op['v_after'],v,holder['gv'],vv,.1);mv=op['v_after']
                else:need(op['v_before'] is op['v_after'] is None and np.array_equal(vv,v),'virtual unchanged')
                x,v=xx,vv
            budget.auxiliary('Adam_step',update)
            if phase=='joint':
                def noise():
                    nonlocal v
                    b.event(dict(kind='noise_attempt'));e=c.get('noise');need(set(e)=={'kind','noise','v'},'actual saved noise')
                    n=u(e['noise'],(1,2,192));nv=u(e['v'],(1,2,192));need(np.array_equal(nv,v+n),'exact noise add');v=nv
                budget.auxiliary('virtual_noise',noise)
            if phase=='sync' and j==9:
                b.event(dict(kind='joint_optimizer_reset',after=k+1,x=grasp.pack(x),v=grasp.pack(v)))
                mx=common.adam_reference.initial_memory(x);mv=common.adam_reference.initial_memory(v)
            if budget.commit():snapshot=x.copy()
            else:break
        except box.Refused as exc:
            need(budget.ledger.stop=='wall_cap','only observed cooperative wall partial is replayable')
            failure=dict(type=type(exc).__name__,message=str(exc));b.event(dict(kind='algorithm_exception',**failure));break
    need(b.module_index==len(b.row['neural']),'all actual module captures consumed')
    return dict(export=export,error=failure)


def check_sources():
    box.source_check()
    for p,h in PINS.items():need(sha(p)==h,'exact frozen code '+str(p))
    return {str(p):h for p,h in PINS.items()}


def build(protocol_prepared,protocol_sha):
    """Use the original exact task/model closure; never read outcomes."""
    check_sources();p=common.build(protocol_prepared,protocol_sha)
    common.join_tasks(p['slots'])
    p.update(schema=NAME+'.prepare',contract=contract(),root_arm_order=[list(x) for x in box.ORDER],
             parent_protocol=str(Path(protocol_prepared).resolve()),parent_protocol_sha256=protocol_sha)
    for f in own_files()+[ROOT/'tests'/('test_'+box.NAME+'.py'),ROOT/'innerdocs'/(box.NAME+'.md')]:
        p['bindings_sha256'][str(f.resolve())]=sha(f)
    p['bindings_sha256'].update({str(f):h for f,h in PINS.items()});old.parent.check_files(p)
    return p


def write_new(path,value,cap=8*1024**2):
    p=Path(path).absolute();b=box.raw(value)
    need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=cap,
         'exclusive own bounded metadata')
    need(shutil.disk_usage(p.parent).free>=FLOOR+len(b),'metadata floor')
    with p.open('xb') as f:need(f.write(b)==len(b),'complete metadata');f.flush();os.fsync(f.fileno())
    old.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p))


def verify(prepared,expected):
    need(sha(prepared)==expected and Path(prepared).stat().st_size<=8*1024**2,'external full prepare SHA/size')
    b=Path(prepared).read_bytes();p=json.loads(b);need(b==box.raw(p),'canonical prepare')
    equal(p,build(p['parent_protocol'],p['parent_protocol_sha256']),'complete preparation rebuild');return p


def run_cuda(prepared,expected,output,authorization):
    need(authorization==AUTH,'explicit MAIN HOST CUDA authorization before I/O')
    start=time.monotonic();p=verify(prepared,expected);verified=time.monotonic();store=Store(output,start=start)
    rows=[];errors=[];model=before=after=host=guard0=guard1=peak=None;tasks=None
    timings=dict(initial_prepare_verify_seconds=verified-start,host_setup_seconds=None,loader_seconds=None,
                 task_rebind_seconds=None,post_search_persistence_seconds=0.,final_identity_seconds=None,artifact_hash_seconds=None)
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        begun=time.monotonic();timings['host_setup_seconds']=begun-verified
        model,before=old.loader.loaded_model(p,torch,device,store,expected)
        torch.cuda.synchronize(device);timings['loader_seconds']=time.monotonic()-begun
        begun=time.monotonic();tasks=common.join_tasks(p['slots']);timings['task_rebind_seconds']=time.monotonic()-begun
        def clock():torch.cuda.synchronize(device);return time.monotonic()
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
            guard0=grasp.precision(torch)
            for i,(root,arm) in enumerate(box.ORDER):
                row=dict(index=i,root_slot=root,arm=arm,forwards=[],AD=[],neural=[]);rows.append(row)
                store.event(dict(kind='arm_start',index=i,root=root,arm=arm))
                emit=lambda e,i=i:store.event(dict(kind='arm_event',index=i,payload=e))
                budget=Tape(root,arm,emit,clock)
                try:
                    if arm=='grasp':
                        b=GraspBackend(model,device,tasks[root],emit,store,budget,row)
                        with b.observe():result=drive_grasp(tasks[root],root,b)
                    else:
                        b=HostBackend(model,device,torch,store,row,emit,budget)
                        with b.observe():result=drive_spectral(p['slots'][root],arm,b)
                except BaseException as exc:
                    result=dict(fatal=dict(type=type(exc).__name__,message=str(exc)))
                    errors.append(dict(stage='arm',index=i,type=type(exc).__name__,message=str(exc)))
                persisted=time.monotonic()
                row.update(result=result,accounting=budget.ledger.summary())
                store.put(f'arm{i}.json',row)
                store.event(dict(kind='arm_sealed',index=i,row_sha256=old.digest(row)))
                timings['post_search_persistence_seconds']+=time.monotonic()-persisted
                if errors:break
            guard1=grasp.precision(torch)
        peak=dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device))
        need(peak['allocated']<=4*1024**3 and peak['reserved']<=6*1024**3,'stage CUDA peak')
    except BaseException as exc:errors.append(dict(stage='execution',type=type(exc).__name__,message=str(exc)))
    finally:
        begun=time.monotonic()
        if model is not None:
            try:old.loader.enc.frozen(model);after=old.loader.enc.tensor_hash(model)
            except BaseException as exc:errors.append(dict(stage='model_final',type=type(exc).__name__,message=str(exc)))
        try:old.parent.check_files(p)
        except BaseException as exc:errors.append(dict(stage='source_final',type=type(exc).__name__,message=str(exc)))
        for row in rows:
            path=store.output/f'arm{row["index"]}.json'
            if not path.exists():
                try:store.put(path.name,row,final=True)
                except BaseException as exc:errors.append(dict(stage='partial_row',type=type(exc).__name__,message=str(exc)))
        store.close();timings['final_identity_seconds']=time.monotonic()-begun
        begun=time.monotonic();artifacts={f.name:sha(f) for f in store.output.iterdir()}
        timings['artifact_hash_seconds']=time.monotonic()-begun
        report=dict(schema=NAME+'.report',prepared=str(Path(prepared).resolve()),prepare_sha256=expected,
            contract=contract(),rows=[dict(index=r['index'],root_slot=r['root_slot'],arm=r['arm'],file=f'arm{r["index"]}.json') for r in rows],
            fixed_denominator=6,status='recorded' if len(rows)==6 and not errors else 'failed_partial',errors=errors,
            model_before=before,model_after=after,host=host,precision_before=guard0,precision_after=guard1,peak=peak,
            timings=timings,total_seconds=time.monotonic()-start,partitions=deepcopy(store.totals),
            total_seconds_scope='initial verify through artifact hashes, excludes final report write/fsync',artifact_sha256=artifacts)
        store.put('report.json',report,final=True)
    return dict(report=str(store.output/'report.json'),sha256=sha(store.output/'report.json'),errors=errors)


def consume(output,expected):
    """Full saved replay including cooperative wall prefixes; no synthetic scores."""
    output=Path(output).absolute();need(sha(output/'report.json')==expected,'actual external report SHA')
    reader=Reader(output);r=reader.read('report.json',8*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    need(r['schema']==NAME+'.report' and r['status']=='recorded' and r['errors']==[] and r['fixed_denominator']==6,
         'all six recorded; partial/failed output is preserved but not admitted')
    timings=r['timings'];need(set(timings)=={'initial_prepare_verify_seconds','host_setup_seconds','loader_seconds',
        'task_rebind_seconds','post_search_persistence_seconds','final_identity_seconds','artifact_hash_seconds'} and
        all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in timings.values()) and
        type(r['total_seconds']) in (int,float) and math.isfinite(r['total_seconds'])
        and sum(timings.values())<=r['total_seconds']<OUTER and
        r['total_seconds_scope']=='initial verify through artifact hashes, excludes final report write/fsync','explicit nonoverlapping full outer time scopes')
    equal(r['contract'],contract(),'entire contract')
    need(r['model_before']==r['model_after']==p['expected_model_hash'],'model frozen before/after')
    guard=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False)
    equal(r['precision_before'],guard,'entry precision');equal(r['precision_after'],guard,'exit precision')
    peak=r['peak'];need(set(peak)=={'allocated','reserved'} and type(peak['allocated']) is type(peak['reserved']) is int
        and 0<=peak['allocated']<=4*1024**3 and peak['allocated']<=peak['reserved']<=6*1024**3,'actual stage peak')
    names=set(r['artifact_sha256']);need({x.name for x in output.iterdir()}==names|{'report.json'},'exact artifact census')
    need(sum(x.stat().st_size for x in output.iterdir())<=CAP,'actual total output bound')
    totals=r['partitions'];need(set(totals)=={'array','journal','row','final'} and totals['final']==0
        and totals['array']==(output/'arrays.bin').stat().st_size<=byte_proof()['array_partition']
        and totals['journal']==(output/'events.jsonl').stat().st_size<=byte_proof()['journal_partition']
        and totals['row']==sum((output/n).stat().st_size for n in names-{'arrays.bin','events.jsonl'})<=byte_proof()['row_partition']
        and (output/'report.json').stat().st_size<=RESERVE,'exact actual partition census')
    for name,h in r['artifact_sha256'].items():need(Path(name).name==name and sha(output/name)==h,'each actual artifact SHA')
    equal(reader.read('binding.json',1024**2),dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'full execution binding')
    loader=reader.read('loader.json',8*1024**2);old.loader.validate_loader(loader,p,r['prepare_sha256'])
    cursor=Cursor(reader.events())
    for f in loader['files']:
        cursor.emit(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']))
        cursor.emit(dict(kind='consumed_buffer_frozen',receipt=f))
    need(len(r['rows'])==6,'complete denominator');tasks=common.join_tasks(p['slots']);accounts=[];plans=[];partials=[]
    previous_end=None
    for i,(root,arm) in enumerate(box.ORDER):
        meta=dict(index=i,root_slot=root,arm=arm,file=f'arm{i}.json');equal(r['rows'][i],meta,'fixed six order')
        row=reader.read(meta['file'],byte_proof()['row_partition'])
        need((row['index'],row['root_slot'],row['arm'])==(i,root,arm),'full saved row identity')
        cursor.emit(dict(kind='arm_start',index=i,root=root,arm=arm))
        class ArmCursor:
            def get(self,kind):
                e=cursor.get('arm_event');need(e['index']==i and set(e)=={'kind','index','payload'},'exact arm scope')
                v=e['payload'];need(v['kind']==kind,'exact arm event '+kind);return v
            def emit(self,e):equal(self.get(e['kind']),e,'exact own event chronology')
            def clock(self):
                e=self.get('clock');need(set(e)=={'kind','value'} and type(e['value']) in (int,float)
                    and math.isfinite(e['value']),'finite actual clock');return e['value']
        ac=ArmCursor();budget=SavedTape(root,arm,ac)
        if arm=='grasp':
            b=SavedGrasp(tasks[root],ac,budget,row);result=saved_grasp(tasks[root],root,b)
        else:
            b=SavedSpectral(row,budget,ac,reader);result=drive_spectral(p['slots'][root],arm,b);b.complete()
        equal(result,row['result'],'complete source mathematics, outcome and actions');summary=budget.ledger.summary()
        if arm=='grasp' and result['error'] is not None:
            need(result['error']['type']=='Refused' and summary['stop']=='wall_cap' and result['export'] is None,
                 'only authenticated cooperative wall failure retains partial GRASP')
        if arm!='grasp' and result['aborted_unit'] is not None:
            need(summary['stop']=='wall_cap' and result['export'] is None,
                 'only authenticated cooperative wall failure retains partial spectral unit')
        if (arm=='grasp' and result['error'] is not None) or (arm!='grasp' and result['aborted_unit'] is not None):partials.append(i)
        equal(summary,row['accounting'],'actual complete 27-axis/side/time ledger')
        need(previous_end is None or budget.ledger.start>=previous_end,'nonoverlapping arm scopes')
        previous_end=budget.ledger.last
        cursor.emit(dict(kind='arm_sealed',index=i,row_sha256=old.digest(row)))
        accounts.append(summary);plans.append(result['export'])
    cursor.complete();reader.complete_archive()
    need(reader.used==names|{'report.json'},'every artifact read, none unknown')
    old.parent.check_files(p)
    for name,h in r['artifact_sha256'].items():need(sha(output/name)==h,'end-of-consume stable artifact')
    need(sha(output/'report.json')==expected,'end-of-consume report stable')
    return dict(schema=NAME+'.consume',status='all6_joint_box_saved_verified_not_scientific_admission',
        report_sha256=expected,fixed_denominator=6,accounts=accounts,plans=plans,
        partial_arm_indices=partials,missing_plan_indices=[i for i,p in enumerate(plans) if p is None],
        natural_full_schedule_indices=[i for i,a in enumerate(accounts) if a['stop']=='natural_completion'],
        actual_cost_equality=False,equal_FLOPs=False,total_FLOPs=None,independent_AD_validation=False,
        training=0,new_neural_calls=0,new_physics_calls=0,physical_release=False,P1_admission=False,P2_mechanism=False,
        bindings=dict(p['bindings_sha256'],**{str(output/n):h for n,h in r['artifact_sha256'].items()},**{str(output/'report.json'):expected}))


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('spec','verify-sources','prepare','verify','run','consume'))
    for n in ('protocol-prepared','protocol-sha','prepared','prepare-sha','output','authorization','report-sha','receipt'):q.add_argument('--'+n)
    a=q.parse_args(argv)
    if a.command=='spec':v=contract()
    elif a.command=='verify-sources':v=check_sources()
    elif a.command=='prepare':v=write_new(a.output,build(a.protocol_prepared,a.protocol_sha))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(verified=True,contract=contract())
    elif a.command=='run':v=run_cuda(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:
        v=consume(a.output,a.report_sha)
        if a.receipt:v=write_new(a.receipt,v,128*1024**2)
    print(box.raw(v).decode());return 0


if __name__=='__main__':raise SystemExit(main())
