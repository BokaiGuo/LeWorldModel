"""PNG and terminal publication transport; not an E3 experiment strict."""
import io
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_e3_unified_transport_20261029_v1 as base

NAME='agent_e3_transport_publication_20261029_v1'
PNG_CAP=256*1024
PNG_COUNT=32
FINAL_CAP=base.RESERVE


def png_data(b):
    base.need(type(b) is bytes and 0<len(b)<=PNG_CAP,'bounded PNG bytes')
    with Image.open(io.BytesIO(b)) as im:
        base.need(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'original RGB PNG shape')
        pixels=np.asarray(im).copy()
    return base.fsha(pixels.tobytes())


class Store(base.Store):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.frames=0;self.final_bytes=0;self.sealed=False;self.failures=0
    def event(self,origin,kind,resource,token,detail):
        base.need(not self.sealed,'no events after publication')
        e=super().event(origin,kind,resource,token,detail)
        if kind=='failure':self.failures+=1
        return e
    def frame(self,b):
        base.need(not self.sealed and self.frames<PNG_COUNT,'bounded frame count before write')
        pixel_sha=png_data(b);name=f'frame{self.frames:02}.png'
        self.write(name,b,'physical_png');self.frames+=1
        return dict(file=name,sha256=base.fsha(b),pixel_sha256=pixel_sha,shape=[224,224,3],dtype='uint8')
    def final(self,name,value):
        base.need(name in ('report.json','completion.json','publication_overrun.json') and not (self.output/name).exists(),
                  'exclusive fixed final basename')
        b=base.raw(value);base.need(self.final_bytes+len(b)<=FINAL_CAP and sum(self.used.values())+self.final_bytes+len(b)<=base.CAP
            and shutil.disk_usage(self.output).free>=base.FLOOR+len(b),'reserve/cap/floor before final write')
        with (self.output/name).open('xb') as f:
            base.need(f.write(b)==len(b),'complete final write');f.flush();os.fsync(f.fileno())
        self.final_bytes+=len(b);base.syncdir(self.output);return base.fsha(b)
    def publish(self,status):
        base.need(not self.sealed and type(status) is str and status in ('completed','partial'),'terminal status')
        self.sealed=True
        pending=[dict(origin=k[0],token=k[1],resource=v) for k,v in sorted(self.pending.items())]
        journal=self.output/'events.jsonl'
        base.need(journal.is_file(),'nonempty journal required')
        report=dict(schema=NAME+'.report',status=status,pending=pending,tail=self.previous,
                    event_count=self.index,journal_sha256=base.fsha(base.read_bytes(journal,base.CAP)),
                    category_bytes=self.used,scientific_admission=False,full_native_strict=False)
        report_sha=self.final('report.json',report);elapsed=self.clock()-self.started
        complete=(status=='completed' and not pending and self.failures==0 and 0<=elapsed<=base.WALL)
        completion=dict(schema=NAME+'.completion',report_sha256=report_sha,elapsed_seconds=elapsed,
                        boundary='after_report_fsync_hash',complete=complete)
        completion_sha=self.final('completion.json',completion)
        if self.clock()-self.started>base.WALL:
            self.final('publication_overrun.json',dict(reason='completion_fsync_after_deadline',complete=False))
            complete=False
        return dict(report_sha256=report_sha,completion_sha256=completion_sha,complete=complete)


def consume(output,report_sha,completion_sha):
    """Independent transport reconstruction, not native or numeric strict."""
    p=Path(output);base.need(p.is_absolute() and str(p.resolve())==str(p) and p.is_dir() and not p.is_symlink(),'canonical output')
    files=list(p.iterdir());base.need(all(x.is_file() and not x.is_symlink() for x in files)
        and sum(x.stat().st_size for x in files)<=base.CAP,'flat bounded census')
    rb=base.read_bytes(p/'report.json',FINAL_CAP);cb=base.read_bytes(p/'completion.json',FINAL_CAP)
    base.need(base.fsha(rb)==report_sha and base.fsha(cb)==completion_sha,'external final SHAs')
    report=base.decode(rb);completion=base.decode(cb)
    base.need(report.get('schema')==NAME+'.report' and completion.get('schema')==NAME+'.completion'
        and completion.get('report_sha256')==report_sha and completion.get('boundary')=='after_report_fsync_hash'
        and report.get('scientific_admission') is False and report.get('full_native_strict') is False,
        'new transport-only schema')
    journal=base.read_bytes(p/'events.jsonl',base.CAP);base.need(base.fsha(journal)==report['journal_sha256']
        and journal.endswith(b'\n') and b'\n\n' not in journal,'journal byte seal')
    previous=None;pending={};refs={};counts={};charged={k:0 for k in base.LIMITS};events=[]
    def collect(x):
        if type(x) is dict:
            if set(x)=={'file','sha256','shape','dtype','origin','role'} or set(x)=={'file','sha256','pixel_sha256','shape','dtype'}:
                name=x['file'];base.need(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat descriptor')
                base.need(name not in refs or refs[name]==x,'one immutable descriptor');refs[name]=x
            else:
                for y in x.values():collect(y)
        elif type(x) is list:
            for y in x:collect(y)
    for i,line in enumerate(journal.splitlines()):
        base.need(len(line)<=16384,'event bound');e=base.decode(line)
        base.need(type(e) is dict and set(e)=={'index','previous','origin','kind','resource','token','detail','sha256'}
            and type(e['index']) is int and e['index']==i and e['previous']==previous
            and e['sha256']==base.digest({k:v for k,v in e.items() if k!='sha256'}),'event chain')
        origin=e['origin'];kind=e['kind'];token=e['token'];resource=e['resource']
        base.need(origin in base.ORIGINS and kind in ('intent','return','failure','record')
            and type(token) is int and token>=0 and type(resource) is str and bool(resource)
            and type(e['detail']) is dict,'event ABI')
        key=(origin,token);c=counts.setdefault(origin+':'+resource,dict(attempted=0,returned=0,failed=0))
        if kind=='intent':base.need(token==i and key not in pending,'unique intent');pending[key]=resource;c['attempted']+=1
        elif kind in ('return','failure'):
            base.need(pending.pop(key,None)==resource,'event lifecycle');c['returned' if kind=='return' else 'failed']+=1
        charged[origin+'_events']+=len(line)+1;collect(e['detail']);events.append(e);previous=e['sha256']
    base.need(previous==report['tail'] and len(events)==report['event_count'],'terminal journal join')
    for name,d in refs.items():
        if 'pixel_sha256' in d:
            base.need(name.endswith('.png') and d['shape']==[224,224,3] and d['dtype']=='uint8','PNG descriptor')
            b=base.read_bytes(p/name,PNG_CAP);base.need(base.fsha(b)==d['sha256'] and png_data(b)==d['pixel_sha256'],'actual PNG bytes/pixels')
            charged['physical_png']+=len(b);continue
        origin=d['origin'];shape=d['shape'];dtype=d['dtype'];role=d['role']
        base.need(origin in ('encoding','numerical') and type(shape) is list and all(type(x) is int and x>=0 for x in shape)
            and dtype in ('<f4','|u1') and name.endswith('.npy'),'array descriptor')
        if origin=='numerical':base.need(dtype=='<f4' and np.prod(shape,dtype=object)<=768,'numeric bound')
        else:base.need((role,tuple(shape),dtype) in base.ENC_SHAPES,'encoder fixed shape')
        size=int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize;b=base.read_bytes(p/name,size+4096)
        base.need(base.fsha(b)==d['sha256'],'array SHA');s=io.BytesIO(b);v=np.lib.format.read_magic(s)
        base.need(v in ((1,0),(2,0)),'NPY version');sh,fort,dt=(np.lib.format.read_array_header_1_0(s) if v==(1,0) else np.lib.format.read_array_header_2_0(s))
        base.need(list(sh)==shape and dt.str==dtype and not fort and len(b)-s.tell()==size,'header/payload before load')
        s.seek(0);a=np.load(s,allow_pickle=False);base.need(s.tell()==len(b) and np.isfinite(a).all(),'array EOF/finite')
        charged[origin+'_arrays']+=len(b)
    expected=set(refs)|{'events.jsonl','report.json','completion.json'}
    base.need({x.name for x in files}==expected,'all-and-only referenced inventory, no overrun marker')
    base.need(charged==report['category_bytes'] and all(charged[k]<=base.LIMITS[k] for k in charged),'independent category bytes')
    expected_pending=[dict(origin=k[0],token=k[1],resource=v) for k,v in sorted(pending.items())]
    base.need(report['pending']==expected_pending,'pending state exact')
    elapsed=completion.get('elapsed_seconds');base.need(type(elapsed) in (int,float) and np.isfinite(elapsed) and 0<=elapsed<=base.WALL,'inclusive wall receipt')
    complete=(report['status']=='completed' and not pending and not any(c['failed'] for c in counts.values()))
    base.need(completion['complete'] is complete,'actual operation failure prevents completion')
    return dict(schema=NAME+'.consume',complete=complete,counts=counts,full_native_strict=False,
                scientific_admission=False,new_neural_calls=0,new_physics_calls=0)
