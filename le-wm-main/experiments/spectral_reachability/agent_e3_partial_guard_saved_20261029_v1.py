"""Bounded saved reconstruction of partial E3 guard evidence, never full strict."""
import base64
import hashlib
from pathlib import Path

from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as launch
from experiments.spectral_reachability import agent_e3_continuous_saved_consumer_20261029_v1 as saved
from experiments.spectral_reachability import agent_e3_model_source_guard_20261029_v1 as guard

need=guard.need


def inspect(output,report_sha,completion_sha,authority_path,authority_sha):
    partial=saved.consume(output,report_sha,completion_sha,authority_path,authority_sha)
    need(partial['status']=='partial','partial publication only')
    binding=launch.read_authority(authority_path,authority_sha) if partial['authority_verified_here'] else None
    p=Path(output);report=launch.pub.base.read_bytes(p/'report.json',launch.pub.FINAL_CAP)
    need(launch.pub.base.fsha(report)==report_sha,'stable report')
    journal=launch.pub.base.read_bytes(p/'events.jsonl',launch.pub.base.CAP)
    need(launch.pub.base.fsha(journal)==launch.pub.base.decode(report)['journal_sha256'],'stable journal')
    events=[launch.pub.base.decode(line) for line in journal.splitlines()]
    snapshots={};indices={}
    for stage in ('guard_before','guard_after'):
        markers=[e for e in events if e['origin']=='physical' and e['resource']==stage]
        chunks=[e for e in events if e['origin']=='physical' and e['resource']=='guard_snapshot_chunk'
                and e['detail'].get('stage')==stage]
        need(len(markers)<=1,'unique guard marker')
        if not markers:
            need(not chunks,'incomplete guard chunks remain unverified');continue
        marker=markers[0];d=marker['detail']
        need(marker['kind']=='record' and set(d)=={'sha256','total','bytes'} and type(d['total']) is int
             and d['total']==len(chunks)>0 and type(d['bytes']) is int and 0<d['bytes']<=16*1024**2,
             'bounded guard descriptor')
        blocks=[];last=-1
        for i,e in enumerate(chunks):
            x=e['detail'];need(e['kind']=='record' and set(x)=={'stage','index','total','sha256','data'}
                and x['index']==i and x['total']==d['total'] and x['sha256']==d['sha256']
                and last<e['index']<marker['index'] and type(x['data']) is str and len(x['data'])<=8000,
                'ordered guard chunks')
            b=base64.b64decode(x['data'],validate=True)
            need(len(b)==6000 if i<len(chunks)-1 else 0<len(b)<=6000,'guard chunk size')
            blocks.append(b);last=e['index']
        raw=b''.join(blocks);need(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'],
                                  'guard snapshot hash')
        snapshots[stage]=launch.pub.base.decode(raw);indices[stage]=marker['index']
    need('guard_after' not in snapshots or 'guard_before' in snapshots,'no after without before')
    failure=[e for e in events if e['origin']=='physical' and e['resource']=='launcher_failure']
    need(len(failure)==1 and all(index<failure[0]['index'] for index in indices.values()),
         'guard snapshots before terminal failure record')
    pair_verified=False
    if len(snapshots)==2:
        need(binding is not None and indices['guard_before']<indices['guard_after'],'bound guard order')
        source=dict(binding['prepared']['bindings_sha256'])
        launch.host.source.merge(source,binding['additional_sources'])
        guard.validate_sources(source,[str(Path(launch.__file__).resolve())])
        guard.validate_pair(snapshots['guard_before'],snapshots['guard_after'],binding['prepared']['expected_model_hash'])
        classes=snapshots['guard_before']['classes'];names=[]
        for item in classes:
            need(type(item) is dict and set(item)=={'name','class_module','class_qualname','path','sha256'}
                 and all(type(item[k]) is str and item[k] for k in ('class_module','class_qualname','path','sha256'))
                 and type(item['name']) is str and source.get(item['path'])==item['sha256'],'sealed class source')
            names.append(item['name'])
        need(len(names)==len(set(names)),'unique model class names')
        cleanup=[e for e in events if e['origin']=='physical' and e['resource'] in
                 ('native_cleanup_return','native_cleanup_failure')]
        need(len(cleanup)<=1 and (not cleanup or indices['guard_before']<cleanup[0]['index']<indices['guard_after']),
             'guard encloses recorded cleanup outcome')
        pair_verified=True
    return dict(partial=partial,guard_stages=sorted(snapshots),guard_pair_verified=pair_verified,
                full_saved_strict=False,scientific_admission=False,
                limitation='recorded guard snapshots only; failed episode remains partial')
