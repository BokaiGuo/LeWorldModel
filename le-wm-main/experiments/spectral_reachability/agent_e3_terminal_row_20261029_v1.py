"""Bounded durable terminal row records within the unified event journal.

Transport only: the row contents are not a native or scientific certificate.
"""
import base64
import hashlib
from experiments.spectral_reachability import agent_e3_unified_transport_20261029_v1 as u

MAX=16*1024**2
CHUNK=6000


def write(store, row):
    u.need(type(row) is dict and type(row.get('episode_id')) is str and bool(row['episode_id']), 'terminal row identity')
    data=u.raw(row);u.need(0<len(data)<=MAX, 'terminal row bounded')
    sha=hashlib.sha256(data).hexdigest();parts=[data[i:i+CHUNK] for i in range(0,len(data),CHUNK)]
    token=store.index
    store.event('physical','intent','terminal_row',token,dict(episode_id=row['episode_id'],bytes=len(data),sha256=sha,total=len(parts)))
    for i,part in enumerate(parts):
        store.event('physical','record','terminal_row_chunk',store.index,
                    dict(parent=token,index=i,total=len(parts),data=base64.b64encode(part).decode('ascii')))
    store.event('physical','return','terminal_row',token,dict(episode_id=row['episode_id'],sha256=sha))
    return sha


def read(events):
    u.need(type(events) is list, 'verified journal events')
    starts=[e for e in events if e['origin']=='physical' and e['resource']=='terminal_row' and e['kind']=='intent']
    u.need(len(starts)==1, 'one terminal row intent')
    start=starts[0];token=start['token'];d=start['detail']
    u.need(set(d)=={'episode_id','bytes','sha256','total'} and type(d['bytes']) is int and 0<d['bytes']<=MAX
           and type(d['total']) is int and d['total']==(d['bytes']+CHUNK-1)//CHUNK, 'bounded terminal row descriptor')
    chunks=[e for e in events if e['origin']=='physical' and e['resource']=='terminal_row_chunk']
    ends=[e for e in events if e['origin']=='physical' and e['resource']=='terminal_row' and e['kind']=='return']
    u.need(len(chunks)==d['total'] and len(ends)==1 and ends[0]['token']==token
           and ends[0]['detail']=={'episode_id':d['episode_id'],'sha256':d['sha256']}, 'terminal row completion')
    parts=[]
    for i,e in enumerate(chunks):
        x=e['detail']
        u.need(e['kind']=='record' and start['index']<e['index']<ends[0]['index']
               and set(x)=={'parent','index','total','data'} and x['parent']==token
               and type(x['index']) is int and x['index']==i and x['total']==d['total']
               and type(x['data']) is str and len(x['data'])<=8000, 'terminal chunk identity and order')
        part=base64.b64decode(x['data'],validate=True)
        u.need(len(part)==CHUNK if i<d['total']-1 else 0<len(part)<=CHUNK, 'terminal chunk length')
        parts.append(part)
    data=b''.join(parts)
    u.need(len(data)==d['bytes'] and hashlib.sha256(data).hexdigest()==d['sha256'], 'terminal row SHA')
    row=u.decode(data);u.need(type(row) is dict and row.get('episode_id')==d['episode_id'], 'terminal row identity')
    return row
