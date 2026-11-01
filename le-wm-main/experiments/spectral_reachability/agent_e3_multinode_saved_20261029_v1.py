"""Multiple scoped HOST numeric nodes in one saved transport journal.

Numeric audit only: source/model/RGB and full episode are not certified here.
"""
import base64
import hashlib
from pathlib import Path
from experiments.spectral_reachability import agent_e3_host_transport_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_e3_feedback_node_20261029_v1 as node

MAX_ROW=16*1024**2
CHUNK=6000
need=bridge.transport.need


class NodeBridge(bridge.Bridge):
    def __init__(self,store,boundary):
        super().__init__(store);need(type(boundary) is int and boundary in (10,15),'fixed later boundary');self.boundary=boundary
        store.event('numerical','record','node_start',store.index,{'boundary':boundary})
    def event(self,value):
        need(type(value) is dict and type(value.get('kind')) is str,'HOST native event')
        self.store.event('numerical','record','node_native',self.store.index,
                         {'boundary':self.boundary,'native':self.expand(value)})
    def finish(self,row):
        data=bridge.transport.raw(row);need(0<len(data)<=MAX_ROW,'bounded row')
        count=(len(data)+CHUNK-1)//CHUNK;sha=hashlib.sha256(data).hexdigest()
        for i in range(count):
            self.store.event('numerical','record','node_row',self.store.index,dict(boundary=self.boundary,
                index=i,total=count,row_sha256=sha,data=base64.b64encode(data[i*CHUNK:(i+1)*CHUNK]).decode('ascii')))
        self.store.event('numerical','record','node_end',self.store.index,{'boundary':self.boundary,'row_sha256':sha})
        return sha


def consume(bodies,output,journal_sha):
    """Bounded directory census and replay of ordered complete later nodes."""
    need(type(bodies) is list and len(bodies) in (1,2),'one or two actual later nodes')
    bodies=[node.validate_body(b) for b in bodies];boundaries=[b['feedback_boundary'] for b in bodies]
    need(boundaries in ([10],[15],[10,15]),'fixed ordered boundaries')
    report=bridge.transport.consume(Path(output),journal_sha);events=report['events'];groups=[];current=None
    for e in events:
        if e['origin']!='numerical':continue
        need(e['kind']=='record' and e['resource'] in ('node_start','node_native','node_row','node_end'),'exact scoped numerical event')
        d=e['detail'];kind=e['resource']
        if kind=='node_start':
            need(current is None and set(d)=={'boundary'} and len(groups)<len(boundaries)
                 and d['boundary']==boundaries[len(groups)] and type(d['boundary']) is int,'exact node start order')
            current=dict(boundary=d['boundary'],native=[],chunks=[],ended=False)
        elif kind=='node_native':
            need(current is not None and not current['chunks'] and set(d)=={'boundary','native'}
                 and d['boundary']==current['boundary'],'native only before row')
            current['native'].append(dict(origin='numerical',kind='record',resource='host_native',detail={'native':d['native']}))
        elif kind=='node_row':
            need(current is not None and set(d)=={'boundary','index','total','row_sha256','data'}
                 and d['boundary']==current['boundary'],'row belongs to active node');current['chunks'].append(d)
        else:
            need(current is not None and set(d)=={'boundary','row_sha256'} and d['boundary']==current['boundary']
                 and current['chunks'] and all(x['row_sha256']==d['row_sha256'] for x in current['chunks']),'node end row seal')
            groups.append(current);current=None
    need(current is None and len(groups)==len(bodies),'all nodes sealed')
    results=[]
    for b,g in zip(bodies,groups,strict=True):
        chunks=g['chunks'];total=chunks[0]['total'];sha=chunks[0]['row_sha256']
        need(type(total) is int and total==len(chunks) and 0<total<=(MAX_ROW+CHUNK-1)//CHUNK,'bounded row count')
        pieces=[]
        for i,c in enumerate(chunks):
            need(type(c['index']) is int and c['index']==i and c['total']==total and c['row_sha256']==sha
                 and type(c['data']) is str and len(c['data'])<=4*((CHUNK+2)//3),'exact chunks')
            part=base64.b64decode(c['data'],validate=True)
            need(len(part)==CHUNK if i<total-1 else 0<len(part)<=CHUNK,'chunk bytes');pieces.append(part)
        data=b''.join(pieces);need(len(data)<=MAX_ROW and hashlib.sha256(data).hexdigest()==sha,'row SHA')
        row=bridge.transport.decode(data);need(type(row) is dict and set(row)=={'forwards','AD','result'},'exact row schema')
        expected=bridge.saved_native_events(g['native']);replayed=[]
        saved=bridge.host.SavedBackend(b,row,Path(output),replayed.append)
        value=node.solve(b,saved,replayed.append);saved.complete()
        bridge.host.same(value,row['result'],'saved numeric reconstruction');bridge.host.same(replayed,expected,'complete per-node chronology')
        axes=bridge.host.axes(replayed);bridge.host.same(axes['attempted'],bridge.host.numerical.axes27(b['horizon_packets']),'planned resources')
        bridge.host.same(axes['attempted'],axes['completed'],'returned resources')
        results.append(dict(boundary=g['boundary'],row_sha256=sha,result=value,axes=axes))
    return dict(nodes=results,transport_complete=report['transport_complete'],full_native_strict=False,
                source_model_rgb_authenticated=False,scientific_admission=False)
