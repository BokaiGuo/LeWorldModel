"""Saved guard snapshot audit, never a full E3 semantic consumer."""
import base64
import hashlib
from pathlib import Path
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as publication
from experiments.spectral_reachability import agent_e3_model_source_guard_20261029_v1 as guard

need=guard.need


def consume(output,report_sha,completion_sha,bindings,required):
    p=Path(output);transport=publication.consume(p,report_sha,completion_sha)
    need(transport['complete'],'complete transport required')
    rb=publication.base.read_bytes(p/'report.json',publication.FINAL_CAP)
    need(publication.base.fsha(rb)==report_sha,'stable report')
    report=publication.base.decode(rb);journal=publication.base.read_bytes(p/'events.jsonl',publication.base.CAP)
    need(publication.base.fsha(journal)==report['journal_sha256'],'stable journal')
    events=[publication.base.decode(line) for line in journal.splitlines()]
    need(not any(e['resource']=='guard_failure' for e in events),'no guard failure')
    snapshots={};markers={}
    for stage in ('guard_before','guard_after'):
        end=[e for e in events if e['origin']=='physical' and e['kind']=='record' and e['resource']==stage]
        parts=[e for e in events if e['origin']=='physical' and e['kind']=='record'
               and e['resource']=='guard_snapshot_chunk' and e['detail'].get('stage')==stage]
        need(len(end)==1 and parts,'one complete guard snapshot per stage')
        marker=end[0];d=marker['detail']
        need(set(d)=={'sha256','total','bytes'} and type(d['total']) is int and d['total']==len(parts)
             and type(d['bytes']) is int and 0<d['bytes']<=16*1024**2, 'bounded snapshot descriptor')
        raw=[]
        for i,e in enumerate(parts):
            x=e['detail'];need(set(x)=={'stage','index','total','sha256','data'} and x['stage']==stage
                and type(x['index']) is int and x['index']==i and x['total']==d['total'] and x['sha256']==d['sha256']
                and e['index']<marker['index'] and type(x['data']) is str and len(x['data'])<=8000,'snapshot chunks')
            b=base64.b64decode(x['data'],validate=True)
            need(len(b)==6000 if i<len(parts)-1 else 0<len(b)<=6000,'snapshot chunk size');raw.append(b)
        data=b''.join(raw);need(len(data)==d['bytes'] and hashlib.sha256(data).hexdigest()==d['sha256'],'snapshot SHA')
        snapshots[stage]=publication.base.decode(data);markers[stage]=marker['index']
    clean=[e for e in events if e['origin']=='physical' and e['resource']=='native_cleanup_return']
    native=[e for e in events if e['origin']=='physical' and e['resource'].startswith('native_')
            and e['resource'] not in ('native_cleanup_return','native_cleanup_intent')]
    need(len(clean)==1 and native and markers['guard_before']<native[0]['index']<clean[0]['index']<markers['guard_after'],
         'actual snapshot/physical/cleanup chronology')
    guard.validate_sources(bindings,required)
    guard.validate_pair(snapshots['guard_before'],snapshots['guard_after'],guard.host.numerical.resolver.MODEL_SHA)
    for item in snapshots['guard_before']['classes']:
        need(type(item) is dict and set(item)=={'name','class_module','class_qualname','path','sha256'}
             and type(item['name']) is str and all(type(item[k]) is str and item[k] for k in
                 ('class_module','class_qualname','path','sha256')) and bindings.get(item['path'])==item['sha256'],
             'all actual class source identities sealed')
    names=[x['name'] for x in snapshots['guard_before']['classes']]
    need(len(set(names))==len(names),'unique class names')
    return dict(saved_model_source_guard_verified=True,full_native_strict=False,
                physical_rgb_semantics_verified=False,scientific_admission=False)
