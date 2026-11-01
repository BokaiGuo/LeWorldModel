"""Bounded unified E3 tensor/event transport; not an experiment consumer."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np

M=1024**2
CAP=512*M
RESERVE=64*M
FLOOR=200*M
WALL=1800
LIMITS={'numerical_arrays':143130624,'numerical_events':173025600,'numerical_metadata':16*M,
        'physical_png':8*M,'physical_events':32*M,'physical_metadata':16*M,
        'encoding_arrays':8*M,'encoding_events':8*M}
ORIGINS={'numerical','physical','encoding'}
ENC_SHAPES={('pixels',(224,224,3),'|u1'),('preprocessed',(1,1,3,224,224),'<f4'),
            ('encoder_input',(1,3,224,224),'<f4'),('encoder_hidden',(1,257,192),'<f4'),
            ('projector_input',(1,192),'<f4'),('projector_output',(1,192),'<f4'),('latent',(192,),'<f4')}


def need(x,why):
    if not x:raise ValueError(why)
def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(raw(x)).hexdigest()
def decode(data):
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda x:need(False,'nonfinite JSON'))
def syncdir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def fsha(b):return hashlib.sha256(b).hexdigest()


class Store:
    def __init__(self,path,started=None,clock=time.monotonic):
        p=Path(path);need(p.is_absolute() and str(p.resolve())==str(p) and not p.exists() and not p.is_symlink()
                           and p.parent.is_dir(),'new canonical output')
        need(shutil.disk_usage(p.parent).free>=CAP+FLOOR,'whole cap plus floor before creation')
        p.mkdir();syncdir(p.parent);self.output=p;self.started=clock() if started is None else started;self.clock=clock
        self.used={k:0 for k in LIMITS};self.number=0;self.index=0;self.previous=None;self.pending={}
    def check(self,category,size):
        need(category in LIMITS and type(size) is int and size>=0,'known category')
        need(0<=self.clock()-self.started<=WALL,'inclusive wall before write')
        need(self.used[category]+size<=LIMITS[category] and sum(self.used.values())+size<=CAP-RESERVE
             and shutil.disk_usage(self.output).free>=FLOOR+size,'category, whole reserve and floor')
    def write(self,name,data,category,append=False):
        need(type(name) is str and Path(name).name==name and name not in ('','.','..') and type(data) is bytes,'flat owned bytes')
        p=self.output/name;need(not p.is_symlink() and (name=='events.jsonl' if append else not p.exists()),'exclusive owned file')
        self.check(category,len(data))
        if append:need(p.exists()==(self.index>0),'no external journal insertion')
        with p.open('ab' if append else 'xb') as f:
            need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
        self.used[category]+=len(data);syncdir(self.output)
    def array(self,origin,role,value):
        need(origin in ('numerical','encoding') and type(role) is str and role,'array role')
        a=np.asarray(value);need(a.dtype in (np.dtype('float32'),np.dtype('uint8')) and np.isfinite(a).all(),'array finite dtype')
        if origin=='numerical':need(a.dtype==np.float32 and a.size<=768,'numerical f32 maximum')
        else:need((role,a.shape,a.dtype.str) in ENC_SHAPES,'original encoder shape/role/dtype')
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);b=buf.getvalue()
        need(len(b)<=a.nbytes+4096,'bounded NPY header')
        name=f'a{self.number:06}.npy';self.write(name,b,origin+'_arrays');self.number+=1
        return dict(file=name,sha256=fsha(b),shape=list(a.shape),dtype=a.dtype.str,origin=origin,role=role)
    def event(self,origin,kind,resource,token,detail):
        need(origin in ORIGINS and kind in ('intent','return','failure','record') and type(resource) is str and resource
             and type(token) is int and token>=0 and type(detail) is dict,'unified event ABI')
        key=(origin,token)
        if kind=='intent':need(key not in self.pending and token==self.index,'unique intent')
        elif kind in ('return','failure'):need(self.pending.get(key)==resource,'actual matching pending operation')
        unsigned=dict(index=self.index,previous=self.previous,origin=origin,kind=kind,resource=resource,token=token,detail=deepcopy(detail))
        e=dict(unsigned,sha256=digest(unsigned));b=raw(e)+b'\n';need(len(b)<=16385,'event record maximum')
        self.write('events.jsonl',b,origin+'_events',append=True)
        self.index+=1;self.previous=e['sha256']
        if kind=='intent':self.pending[key]=resource
        elif kind in ('return','failure'):del self.pending[key]
        return e


def read_bytes(path,cap):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded regular file')
    with path.open('rb') as f:b=f.read(cap+1)
    need(len(b)<=cap,'bounded read');return b


def consume(output,journal_sha):
    """Audit transport and arrays only; no model/physical authority is implied."""
    p=Path(output);need(p.is_absolute() and str(p.resolve())==str(p) and p.is_dir() and not p.is_symlink(),'canonical output')
    files=list(p.iterdir());need(all(f.is_file() and not f.is_symlink() for f in files),'flat regular inventory')
    need(sum(f.stat().st_size for f in files)<=CAP,'whole output cap')
    b=read_bytes(p/'events.jsonl',LIMITS['numerical_events']+LIMITS['physical_events']+LIMITS['encoding_events'])
    need(fsha(b)==journal_sha and b.endswith(b'\n') and b'\n\n' not in b,'journal SHA/singleLF')
    previous=None;pending={};counts={};events=[];refs={};charged={k:0 for k in LIMITS}
    def collect(v):
        if type(v) is dict:
            if set(v)=={'file','sha256','shape','dtype','origin','role'}:
                name=v['file'];need(type(name) is str and Path(name).name==name and name.endswith('.npy'),'array filename')
                need(name not in refs or refs[name]==v,'consistent immutable array descriptor');refs[name]=v
            else:
                for x in v.values():collect(x)
        elif type(v) is list:
            for x in v:collect(x)
    for i,line in enumerate(b.splitlines()):
        need(len(line)<=16384,'record length');e=decode(line)
        need(type(e) is dict and set(e)=={'index','previous','origin','kind','resource','token','detail','sha256'}
             and type(e['index']) is int and e['index']==i and e['previous']==previous
             and e['sha256']==digest({k:v for k,v in e.items() if k!='sha256'}),'full chain')
        origin=e['origin'];kind=e['kind'];token=e['token'];resource=e['resource']
        need(origin in ORIGINS and kind in ('intent','return','failure','record') and type(token) is int and token>=0
             and type(resource) is str and resource and type(e['detail']) is dict,'event ABI')
        key=(origin,token);c=counts.setdefault(origin+':'+resource,dict(attempted=0,returned=0,failed=0))
        if kind=='intent':need(token==i and key not in pending,'intent identity');pending[key]=resource;c['attempted']+=1
        elif kind in ('return','failure'):
            need(pending.pop(key,None)==resource,'matching operation');c['returned' if kind=='return' else 'failed']+=1
        charged[origin+'_events']+=len(line)+1;collect(e['detail']);events.append(e);previous=e['sha256']
    for name,d in refs.items():
        origin=d['origin'];need(origin in ('encoding','numerical') and type(d['shape']) is list
            and all(type(n) is int and n>=0 for n in d['shape']) and d['dtype'] in ('<f4','|u1'),'array descriptor')
        if origin=='numerical':need(d['dtype']=='<f4' and np.prod(d['shape'],dtype=object)<=768,'numerical array limit')
        else:need((d['role'],tuple(d['shape']),d['dtype']) in ENC_SHAPES,'encoder shape')
        maxbytes=4096+int(np.prod(d['shape'],dtype=object))*np.dtype(d['dtype']).itemsize
        blob=read_bytes(p/name,maxbytes);need(fsha(blob)==d['sha256'],'array SHA')
        s=io.BytesIO(blob);version=np.lib.format.read_magic(s)
        need(version in ((1,0),(2,0)),'supported NPY version')
        shape,fortran,dtype=(np.lib.format.read_array_header_1_0(s) if version==(1,0)
                             else np.lib.format.read_array_header_2_0(s))
        need(list(shape)==d['shape'] and dtype.str==d['dtype'] and not fortran
             and len(blob)-s.tell()==int(np.prod(shape,dtype=object))*dtype.itemsize,'NPY header and payload before allocation')
        s.seek(0);a=np.load(s,allow_pickle=False)
        need(s.tell()==len(blob) and a.dtype.str==d['dtype'] and list(a.shape)==d['shape'] and np.isfinite(a).all(),'NPY exact EOF/shape/dtype/finite')
        charged[origin+'_arrays']+=len(blob)
    need({x.name for x in files}==set(refs)|{'events.jsonl'},'all-and-only referenced inventory')
    need(all(charged[k]<=LIMITS[k] for k in charged),'independent category limits')
    return dict(events=events,counts=counts,pending=pending,bytes=charged,
        transport_complete=not pending and all(x['failed']==0 for x in counts.values()),
        full_native_strict=False,full_upstream_rebuilt_here=False,scientific_admission=False)
