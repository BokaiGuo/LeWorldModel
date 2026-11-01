"""Reconstruct numeric nodes from a published unified journal.

Bodies are caller-supplied; physical, encoder, source and model semantics are
not certified by this function. No neural model or physics is executed.
"""
import base64
import hashlib
from pathlib import Path
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub
from experiments.spectral_reachability import agent_e3_multinode_saved_20261029_v1 as m

need=pub.base.need


def consume(bodies, output, report_sha, completion_sha):
    transport=pub.consume(output,report_sha,completion_sha)
    need(transport['complete'], 'complete publication required')
    need(type(bodies) is list and len(bodies) in (1,2), 'fixed later node count')
    bodies=[m.node.validate_body(b) for b in bodies]
    boundaries=[b['feedback_boundary'] for b in bodies]
    need(boundaries in ([10],[15],[10,15]), 'ordered later nodes')
    p=Path(output)
    data=pub.base.read_bytes(p/'events.jsonl',pub.base.CAP)
    report_bytes=pub.base.read_bytes(p/'report.json',pub.FINAL_CAP)
    need(pub.base.fsha(report_bytes)==report_sha, 'report stable after publication audit')
    report=pub.base.decode(report_bytes)
    need(pub.base.fsha(data)==report['journal_sha256'], 'journal stable after publication audit')
    events=[pub.base.decode(line) for line in data.splitlines()]
    groups=[];current=None
    for e in events:
        if e['origin']!='numerical':continue
        need(e['kind']=='record' and e['resource'] in ('node_start','node_native','node_row','node_end'),
             'known numerical lifecycle')
        d=e['detail'];kind=e['resource']
        if kind=='node_start':
            need(current is None and set(d)=={'boundary'} and len(groups)<len(boundaries)
                 and type(d['boundary']) is int and d['boundary']==boundaries[len(groups)], 'node start')
            current=dict(boundary=d['boundary'],native=[],chunks=[])
        elif kind=='node_native':
            need(current is not None and not current['chunks'] and set(d)=={'boundary','native'}
                 and d['boundary']==current['boundary'], 'node native order')
            current['native'].append(dict(origin='numerical',kind='record',resource='host_native',detail={'native':d['native']}))
        elif kind=='node_row':
            need(current is not None and set(d)=={'boundary','index','total','row_sha256','data'}
                 and d['boundary']==current['boundary'], 'node row order')
            current['chunks'].append(d)
        else:
            need(current is not None and set(d)=={'boundary','row_sha256'} and d['boundary']==current['boundary']
                 and current['chunks'] and all(x['row_sha256']==d['row_sha256'] for x in current['chunks']), 'node end')
            groups.append(current);current=None
    need(current is None and len(groups)==len(bodies), 'all nodes sealed')
    result=[]
    for body,g in zip(bodies,groups,strict=True):
        chunks=g['chunks'];total=chunks[0]['total'];sha=chunks[0]['row_sha256']
        need(type(total) is int and total==len(chunks) and 0<total<=(m.MAX_ROW+m.CHUNK-1)//m.CHUNK, 'row count')
        pieces=[]
        for i,c in enumerate(chunks):
            need(type(c['index']) is int and c['index']==i and c['total']==total and c['row_sha256']==sha
                 and type(c['data']) is str and len(c['data'])<=4*((m.CHUNK+2)//3), 'row chunk identity')
            part=base64.b64decode(c['data'],validate=True)
            need(len(part)==m.CHUNK if i<total-1 else 0<len(part)<=m.CHUNK, 'row chunk size')
            pieces.append(part)
        data=b''.join(pieces)
        need(len(data)<=m.MAX_ROW and hashlib.sha256(data).hexdigest()==sha, 'row SHA')
        row=pub.base.decode(data)
        need(type(row) is dict and set(row)=={'forwards','AD','result'}, 'row schema')
        expected=m.bridge.saved_native_events(g['native']);replayed=[]
        saved=m.bridge.host.SavedBackend(body,row,p,replayed.append)
        value=m.node.solve(body,saved,replayed.append);saved.complete()
        m.bridge.host.same(value,row['result'],'saved numeric result')
        m.bridge.host.same(replayed,expected,'saved numeric chronology')
        axes=m.bridge.host.axes(replayed)
        m.bridge.host.same(axes['attempted'],m.bridge.host.numerical.axes27(body['horizon_packets']),'planned axes')
        m.bridge.host.same(axes['attempted'],axes['completed'],'returned axes')
        result.append(dict(boundary=g['boundary'],row_sha256=sha,result=value,axes=axes))
    return dict(nodes=result,transport_complete=True,full_native_strict=False,
                source_model_rgb_authenticated=False,scientific_admission=False)
