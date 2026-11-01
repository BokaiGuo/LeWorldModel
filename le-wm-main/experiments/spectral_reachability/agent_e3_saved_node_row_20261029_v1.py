"""Durable numeric node row inside unified event chain, independently replayed.

Only an intermediate numerical saved consumer, not whole E3 strict.
"""
import base64
import hashlib
from pathlib import Path
from experiments.spectral_reachability import agent_e3_host_transport_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_e3_feedback_node_20261029_v1 as node

MAX_ROW=16*1024**2
CHUNK=6000


def persist(store,row,boundary):
    bridge.transport.need(type(boundary) is int and boundary in (10,15),'own later boundary')
    data=bridge.transport.raw(row);bridge.transport.need(len(data)<=MAX_ROW,'bounded complete row')
    total=(len(data)+CHUNK-1)//CHUNK;h=hashlib.sha256(data).hexdigest()
    for i in range(total):
        detail=dict(boundary=boundary,index=i,total=total,row_sha256=h,
                    data=base64.b64encode(data[i*CHUNK:(i+1)*CHUNK]).decode('ascii'))
        store.event('numerical','record','host_row_chunk',store.index,detail)
    return dict(boundary=boundary,row_sha256=h,bytes=len(data),chunks=total)


def consume(body,output,journal_sha,boundary):
    """Transport census plus complete numeric saved replay, no external authority."""
    b=node.validate_body(body);bridge.transport.need(b['feedback_boundary']==boundary,'matching boundary')
    transport=bridge.transport.consume(Path(output),journal_sha);events=transport['events']
    chunks=[e for e in events if e['origin']=='numerical' and e['resource']=='host_row_chunk']
    bridge.transport.need(chunks and all(e['kind']=='record' and set(e['detail'])==
        {'boundary','index','total','row_sha256','data'} for e in chunks),'exact row chunk records')
    first=chunks[0]['detail'];total=first['total'];digest=first['row_sha256']
    bridge.transport.need(type(total) is int and total>0 and len(chunks)==total and total<=(MAX_ROW+CHUNK-1)//CHUNK,'bounded row chunks')
    parts=[]
    for i,e in enumerate(chunks):
        d=e['detail'];bridge.transport.need(type(d['index']) is int and d['index']==i and d['total']==total
            and type(d['boundary']) is int and d['boundary']==boundary and d['row_sha256']==digest
            and type(d['data']) is str and len(d['data'])<=4*((CHUNK+2)//3),'ordered row identity')
        data=base64.b64decode(d['data'],validate=True)
        bridge.transport.need(len(data)==CHUNK if i<total-1 else 0<len(data)<=CHUNK,'exact chunk lengths')
        parts.append(data)
    raw=b''.join(parts);bridge.transport.need(len(raw)<=MAX_ROW and hashlib.sha256(raw).hexdigest()==digest,'row bytes SHA')
    row=bridge.transport.decode(raw);bridge.transport.need(type(row) is dict and set(row)=={'forwards','AD','result'},'complete row schema')
    filtered=[e for e in events if not (e['origin']=='numerical' and e['resource']=='host_row_chunk')]
    expected=bridge.saved_native_events(filtered);replayed=[]
    saved=bridge.host.SavedBackend(b,row,Path(output),replayed.append)
    result=node.solve(b,saved,replayed.append);saved.complete()
    bridge.host.same(result,row['result'],'saved row reconstructed');bridge.host.same(replayed,expected,'exact native event order')
    axes=bridge.host.axes(replayed);bridge.host.same(axes['attempted'],bridge.host.numerical.axes27(b['horizon_packets']),'original resource axes')
    bridge.host.same(axes['completed'],axes['attempted'],'all numeric resources returned')
    return dict(row_sha256=digest,boundary=boundary,result=result,axes=axes,transport_complete=transport['transport_complete'],
                full_native_strict=False,source_model_rgb_authenticated=False,scientific_admission=False)
