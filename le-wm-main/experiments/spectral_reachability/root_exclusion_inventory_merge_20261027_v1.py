"""Merge frozen historical catalog into a compact resolver inventory.

Saved metadata only. Input-binding ancestry is never physical-root equality.
"""
from __future__ import annotations
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as r

NAME='root_exclusion_inventory_merge_20261027_v1'
ROOT=r.ROOT
CATALOG=ROOT/'innerdocs/moving_historical_authority_inventory_20261027_v1.json'
CATALOG_SHA='c425052a9a43e362eff77befc21e37c6a72fdc3fa455a6f5ac7c53b1c0487341'
BASELINE=ROOT/'results/root_exclusion_authority_resolution_20261027_v1_capture02.json'
BASELINE_SHA='032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00'
INVENTORY_CAP=200*1024
REPORT_CAP=4*1024**2
PARENT_CAP=128
BINDING_PREFIXES=(('bindings',),('plans_bindings',),('source_bindings',),('protocol','bindings'))
REQUIRE=r.v.require


def parse_small(raw,cap):
    REQUIRE(len(raw)<=cap,'small frozen metadata cap')
    def unique(items):
        d={}
        for k,v in items:REQUIRE(k not in d,'duplicate JSON key');d[k]=v
        return d
    x=json.loads(raw,object_pairs_hook=unique);r.v.canonical(x);return x


def load_bound(reader,path,sha,cap):
    return parse_small(reader.read(str(path),sha),cap)


def catalog_inventory(catalog,baseline,root=ROOT,expected_count=618):
    REQUIRE(catalog['schema']=='moving_historical_authority_inventory_20261027_v1.inventory','frozen catalog schema')
    REQUIRE(baseline['schema']==r.NAME+'.resolution_request','baseline schema')
    files=catalog['evidence_files'];REQUIRE(type(files) is list and len(files)==expected_count,'full catalog denominator')
    by_path={};ids=set();compact=[]
    for x in files:
        path=x['path'];r.safe_path(path,root);r.v.hash_value(x['sha256'])
        REQUIRE(path not in by_path and x['source_id'] not in ids,'unique catalog path/source_id')
        REQUIRE(type(x['bytes']) is int and 0<=x['bytes']<=r.FILE_CAP,'catalog file bytes')
        hint=x.get('family_hint','unclassified_requires_MAIN_scope')
        REQUIRE(type(hint) is str and 0<len(hint)<=100,'bounded family hint')
        by_path[path]=x;ids.add(x['source_id'])
        compact.append(dict(path=path,sha256=x['sha256'],family='hint_only:'+hint))
    baseline_paths=set()
    for x in baseline['history_file_requests']:
        REQUIRE(x['path'] not in baseline_paths,'duplicate baseline path');baseline_paths.add(x['path'])
        REQUIRE(x['path'] in by_path and x['sha256']==by_path[x['path']]['sha256'],'catalog/baseline exact path SHA')
    REQUIRE(sum(len(x['inherited_refs']) for x in baseline['history_file_requests'] if not x['root_identity_projections'])==614,'fixed614 missing-identity references')
    compact.sort(key=lambda x:x['path'])
    return dict(schema=r.NAME+'.inventory',files=compact),by_path


def tokens(pointer):
    REQUIRE(type(pointer) is str and pointer.startswith('/'),'JSON pointer required')
    return tuple(x.replace('~1','/').replace('~0','~') for x in pointer[1:].split('/'))


def is_sha(x):
    return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)


def binding_edges(projection,catalog):
    """Accept path-key/SHA, sibling path+SHA, or SHA-only in INPUT bindings."""
    fields={tokens(x['pointer']):x['value'] for x in projection}
    by_sha={}
    for path,item in catalog.items():by_sha.setdefault(item['sha256'],[]).append(path)
    edges=[];seen=set()
    for key,value in sorted(fields.items()):
        if not any(key[:len(p)]==p and len(key)>len(p) for p in BINDING_PREFIXES):continue
        pairs=[]
        if is_sha(value):
            if key[-1].startswith('/'):
                pairs.append((key[-1],value,'literal_path_key_and_sha'))
            elif key[-1] in ('sha256','file_sha256') and type(fields.get(key[:-1]+('path',))) is str:
                pairs.append((fields[key[:-1]+('path',)],value,'sibling_path_and_sha'))
            else:
                pairs.extend((p,value,'input_sha_matches_catalog') for p in sorted(by_sha.get(value,[])))
        for path,sha,basis in pairs:
            identity=(key,path,sha)
            if identity in seen:continue
            seen.add(identity);edges.append(dict(pointer=r.pointer(key),target_path=path,target_sha256=sha,basis=basis))
    return edges


def root_authority(projection):
    """Only original root/seed fields; no statuses, costs, selected outcomes."""
    out=[]
    for x in projection:
        t=tokens(x['pointer']);value=x['value']
        if t[-1] in ('root_seed','reset_seed','calibration_root_seed') and type(value) is int and value>=0:
            out.append(x)
        elif t[-1] in ('root_sha256','root_goal_sha256') and is_sha(value):out.append(x)
        elif len(t)>1 and t[-2] in ('roots','root_hashes') and is_sha(value):out.append(x)
    return out


def new_metadata(reader,path,sha,fmt='json'):
    raw=reader.read(path,sha)
    if fmt=='prepare_envelope':
        p=r.unpack_prepare(raw)
    else:p=r.project(BytesIO(raw),r.META_PATTERNS)
    return [dict(pointer=r.pointer(k),value=v) for k,v in sorted(p.leaves.items(),key=lambda x:str(x[0]))]


def resolve_parents(reader,catalog,baseline):
    cached={x['path']:dict(sha256=x['sha256'],projection=x['projection'],
        basis='frozen_capture02_metadata',error=None) for x in baseline['history_file_requests']}
    pending=[x for x in baseline['history_file_requests'] if not x['root_identity_projections']]
    # Capture the extra 138 explicit catalog items. Never scan or read by outcome.
    for path in sorted(set(catalog)-set(cached)):
        x=catalog[path]
        try:projection=new_metadata(reader,path,x['sha256'],x.get('format','json'));error=None
        except (ValueError,KeyError,OSError) as exc:projection=[];error=type(exc).__name__+': '+str(exc)
        cached[path]=dict(sha256=x['sha256'],projection=projection,basis='catalog_external_SHA_metadata',error=error)
    dispositions=[];parent_reads=set();edge_failures=[]
    for row in pending:
        start=row['path'];queue=[(start,[])];visited={start};found=[];unresolved=[]
        # Three binding hops maximum, deterministic breadth-first; no outcome rank.
        while queue:
            current,chain=queue.pop(0)
            if len(chain)>=3:continue
            node=cached[current]
            for edge in binding_edges(node['projection'],catalog):
                path=edge['target_path'];sha=edge['target_sha256']
                hop=dict(source_path=current,source_sha256=node['sha256'],**edge)
                try:
                    p=r.safe_path(path,reader.root)
                    REQUIRE(p.is_relative_to(reader.root/'results') and p.suffix=='.json','only explicit saved JSON parent authority')
                    if path in cached:REQUIRE(cached[path]['sha256']==sha,'binding target SHA conflicts with captured authority')
                    else:
                        REQUIRE(len(parent_reads)<PARENT_CAP,'parent file cap')
                        parent_reads.add(path)
                        projection=new_metadata(reader,path,sha)
                        cached[path]=dict(sha256=sha,projection=projection,basis='explicit_binding_external_SHA_metadata',error=None)
                    target=cached[path]
                    REQUIRE(target['error'] is None,'parent metadata unsupported: '+str(target['error']))
                    authority=root_authority(target['projection'])
                    if authority:
                        found.append(dict(parent_path=path,parent_sha256=sha,authority=authority,
                            binding_chain=chain+[hop],physical_equivalence=False))
                    if path not in visited:
                        visited.add(path);queue.append((path,chain+[hop]))
                except (ValueError,KeyError,OSError) as exc:
                    failure=dict(**hop,reason=type(exc).__name__+': '+str(exc))
                    unresolved.append(failure);edge_failures.append(failure)
        dispositions.append(dict(path=start,sha256=row['sha256'],original_references=row['inherited_refs'],
            status='input_parent_authority_located_identity_still_unmapped' if found else 'no_proven_input_parent_authority',
            parent_authorities=found,binding_failures=unresolved,
            original_identity_gap_resolved=False,physical_equivalence=False,
            remaining='own observed state/snapshot/RGB/rawhistory mapping and MAIN coverage disposition remain required'))
    return dispositions,cached,parent_reads


def build(reader,catalog_path=CATALOG,catalog_sha=CATALOG_SHA,baseline_path=BASELINE,baseline_sha=BASELINE_SHA):
    catalog=load_bound(reader,catalog_path,catalog_sha,2*1024**2)
    baseline=load_bound(reader,baseline_path,baseline_sha,2*1024**2)
    inventory,by_path=catalog_inventory(catalog,baseline,reader.root)
    dispositions,metadata,parent_reads=resolve_parents(reader,by_path,baseline)
    found=[x for x in dispositions if x['parent_authorities']]
    closure=[Path(__file__).resolve(),Path(r.__file__).resolve(),Path(r.v.__file__).resolve(),Path(r.v.lexical.__file__).resolve()]
    report=dict(schema=NAME+'.disposition',catalog_binding=dict(path=str(catalog_path),sha256=catalog_sha),
        baseline_binding=dict(path=str(baseline_path),sha256=baseline_sha),
        source_closure=[dict(path=str(p),sha256=r.v.file_sha(p)) for p in closure],
        inventory_content_sha256=hashlib.sha256(r.v.canonical(inventory)).hexdigest(),
        summary=dict(catalog_files=len(inventory['files']),catalog_declared_bytes=sum(x['bytes'] for x in by_path.values()),
            baseline_missing_identity_files=len(dispositions),baseline_missing_identity_references=sum(len(x['original_references']) for x in dispositions),
            files_with_parent_authority=len(found),references_with_parent_authority=sum(len(x['original_references']) for x in found),
            files_without_parent_authority=len(dispositions)-len(found),
            references_without_parent_authority=sum(len(x['original_references']) for x in dispositions if not x['parent_authorities']),
            physically_resolved_identity_references=0,extra_parent_files_attempted=len(parent_reads),bytes_read_unique=reader.used),
        metadata_errors=[dict(path=p,sha256=x['sha256'],reason=x['error']) for p,x in sorted(metadata.items()) if x['error']],
        missing_identity_dispositions=dispositions,
        extra_parent_evidence=[dict(path=p,**metadata[p]) for p in sorted(parent_reads) if p in metadata],
        budgets=dict(read_file_bytes=r.FILE_CAP,total_read_bytes=r.TOTAL_CAP,max_extra_parent_files=PARENT_CAP,max_binding_hops=3,
            inventory_bytes=INVENTORY_CAP,report_bytes=REPORT_CAP,disk_floor=r.FLOOR,neural=0,physics=0),
        family_hint_authority='routing_hint_only_not_coverage_approval',MAIN_coverage_approval=False,freshness_certified=False,P1_admission=False,
        remaining='All614 own identity gaps retained. Parent metadata is not physical identity. Historical reset bound disposition remains Arendt/MAIN authority.')
    REQUIRE(len(r.v.canonical(inventory))<=INVENTORY_CAP,'compact inventory cap')
    REQUIRE(len(r.v.canonical(report))<=REPORT_CAP,'compact disposition cap')
    return inventory,report


def save_pair(inventory_path,report_path,inventory,report):
    paths=[Path(inventory_path),Path(report_path)];data=[r.v.canonical(inventory),r.v.canonical(report)]
    REQUIRE(paths[0]!=paths[1],'distinct output paths')
    for p,d,cap in zip(paths,data,(INVENTORY_CAP,REPORT_CAP)):
        REQUIRE(not p.exists() and not p.is_symlink() and p.parent.is_dir() and p.parent.resolve()==p.parent,'NEW canonical output parent')
        REQUIRE(len(d)<=cap and shutil.disk_usage(p.parent).free>=r.FLOOR+sum(map(len,data)),'output cap and 200MiB reserve')
    results=[]
    for p,d in zip(paths,data):
        with p.open('xb') as f:f.write(d);f.flush();os.fsync(f.fileno())
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        results.append(dict(path=str(p),bytes=len(d),sha256=hashlib.sha256(d).hexdigest()))
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory-output',required=True);ap.add_argument('--disposition-output',required=True)
    a=ap.parse_args();inventory,report=build(r.Reader())
    print(json.dumps(dict(outputs=save_pair(a.inventory_output,a.disposition_output,inventory,report),summary=report['summary']),sort_keys=True))


if __name__=='__main__':main()
