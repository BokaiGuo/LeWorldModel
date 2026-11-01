"""Bounded single-LF durable operation journal and independent saved audit.

Does not certify native events, inputs, model/encoder IO, or a complete episode.
"""
import hashlib
import json
import os
from pathlib import Path

MAX_BYTES=32*1024**2
MAX_RECORD=16384


def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(raw(x)).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)


class Ledger:
    def __init__(self,path,cap=MAX_BYTES):
        self.path=Path(path);need(self.path.is_absolute() and not self.path.exists() and not self.path.is_symlink(),'exclusive path')
        need(type(cap) is int and 0<cap<=MAX_BYTES,'fixed cap')
        self.file=self.path.open('xb');self.cap=cap;self.used=0;self.previous=None;self.index=0;self.pending={};self.closed=False
    def append(self,kind,ident,resource,detail):
        need(not self.closed and kind in ('intent','returned','failed') and type(ident) is int and ident>=0
             and type(resource) is str and bool(resource) and type(detail) is dict,'operation ABI')
        if kind=='intent':need(ident==self.index and ident not in self.pending,'unique intent id')
        else:need(self.pending.get(ident)==resource,'matching pending intent')
        unsigned=dict(index=self.index,previous=self.previous,kind=kind,id=ident,resource=resource,detail=detail)
        record=dict(unsigned,sha256=digest(unsigned));data=raw(record)+b'\n'
        need(len(data)<=MAX_RECORD and self.used+len(data)<=self.cap,'write-before bounded journal')
        need(self.file.write(data)==len(data),'complete event write');self.file.flush();os.fsync(self.file.fileno())
        self.used+=len(data);self.previous=record['sha256'];self.index+=1
        if kind=='intent':self.pending[ident]=resource
        else:del self.pending[ident]
        return ident
    def intent(self,resource,detail):return self.append('intent',self.index,resource,detail)
    def returned(self,ident,resource,detail):self.append('returned',ident,resource,detail)
    def failed(self,ident,resource,detail):self.append('failed',ident,resource,detail)
    def close(self):
        if not self.closed:self.file.close();self.closed=True


def audit(path,expected_sha,cap=MAX_BYTES):
    p=Path(path);need(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size<=cap<=MAX_BYTES,'bounded regular journal')
    with p.open('rb') as f:data=f.read(cap+1)
    need(len(data)<=cap and hashlib.sha256(data).hexdigest()==expected_sha,'exact saved SHA and bound')
    need(not data or data.endswith(b'\n') and b'\n\n' not in data,'exact single LF')
    previous=None;pending={};counts={};events=[]
    for i,line in enumerate(data.splitlines()):
        need(len(line)+1<=MAX_RECORD,'record bound')
        def pairs(xs):
            d={}
            for k,v in xs:need(k not in d,'duplicate key');d[k]=v
            return d
        e=json.loads(line,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite'))
        need(type(e) is dict and set(e)=={'index','previous','kind','id','resource','detail','sha256'}
             and type(e['index']) is int and e['index']==i and e['previous']==previous
             and digest({k:v for k,v in e.items() if k!='sha256'})==e['sha256'],'full journal chain')
        ident=e['id'];resource=e['resource'];kind=e['kind']
        need(type(ident) is int and ident>=0 and type(resource) is str and bool(resource)
             and type(e['detail']) is dict and kind in ('intent','returned','failed'),'event ABI')
        c=counts.setdefault(resource,dict(attempted=0,returned=0,failed=0))
        if kind=='intent':
            need(ident==i and ident not in pending,'unique intent');pending[ident]=resource;c['attempted']+=1
        else:
            need(pending.pop(ident,None)==resource,'return/failure matches intent');c[kind]+=1
        previous=e['sha256'];events.append(e)
    return dict(events=events,counts=counts,pending=pending,
                complete=not pending and all(c['failed']==0 for c in counts.values()),
                full_native_strict=False,scientific_admission=False)
