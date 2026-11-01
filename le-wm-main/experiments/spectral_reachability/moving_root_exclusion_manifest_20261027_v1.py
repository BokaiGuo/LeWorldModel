"""Saved-only, explicit-inventory calibration identity/exclusion asset producer.

Never enumerate results, follow embedded paths, import models/environments,
decode outcome labels, or self-approve history coverage/freshness.
"""
from __future__ import annotations
import argparse
import base64
from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import zlib
import numpy as np
from experiments.spectral_reachability import p1_conditional_source_metadata_20261027_v1 as lexical

NAME='moving_root_exclusion_manifest_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT.parent/'docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md'
FLOOR=200*1024**2
FILE_CAP=64*1024**2
TOTAL_CAP=256*1024**2
OUTPUT_CAP=2*1024**2
REQUIRED_FAMILIES=('legacy_zero','original_four','constructive_zero','moving','overlay','feedback','reserved_prepares')
PROPOSED=[dict(source_slot=i,root_seed=203027101+i,reset_seed=203027201+i) for i in range(4)]
BASES=('p1_constructive_reachable_source_20261027_v1','p1_constructive_moving_history_source_20261027_v1',
       'p1_conditional_procedural_source_20261027_v1')
IDENTITY_FIELDS=('root_goal_sha256','root_state','root_state_sha256','root_snapshot_sha256','blind_sha256','blind_payload_sha256',
                 'history_pixels_sha256','raw_history_sha256','task_id','episode_id')


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def hash_value(h):require(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'SHA256 required')
def fields(v,s):require(type(v) is dict and set(v)==set(s.split()),'exact fields: '+s)
def integer(v):require(type(v) is int and 0<=v<2**63,'nonnegative literal seed/index')


def file_sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.resolve()==path,'explicit canonical regular nonsymlink file')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):h.update(block)
    return h.hexdigest()


def small_json(raw):
    require(len(raw)<=OUTPUT_CAP,'small request/approval')
    def unique(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    out=json.loads(raw,object_pairs_hook=unique);canonical(out);return out


def patterns(adapter):
    p=[('schema',)]
    if adapter=='legacy_index':
        return p+[('root_hashes','*'),('seed_families','*','seed'),('reset_spans','*','span','*'),
            ('unknown_sources','*','source'),('unknown_sources','*','reason'),('confirmation_authority','status')]
    if adapter=='task':
        return p+[(k,) for k in ('task_id','source_slot','blind_sha256','root_sha256','root_goal_sha256')]
    containers={'prepare':['slots'], 'sourcepack':['rows'], 'metadata':['records']}[adapter]
    for c in containers:
        for k in ('record_id','record_type','root_seed','reset_seed','root_sha256','calibration_task_id',*IDENTITY_FIELDS):
            p.append((c,'*',k))
        for k in ('maps_to','parent_refs','history_pixels_sha256','history_frame_pixel_sha256','root_state','reset_seed_span'):
            p.append((c,'*',k,'*'))
        for k in ('root_seed','reset_seed','index'):p.append((c,'*','slot',k))
    if adapter=='prepare':
        # Nested old exclusions are only a base index; their unknowns survive.
        p += [('exclusions','root_hashes','*'),('exclusions','seed_families','*','seed'),
              ('exclusions','reset_spans','*','span','*'),('exclusions','unknown_sources','*','source'),
              ('exclusions','unknown_sources','*','reason')]
    return p


class Projection(lexical.Projection):
    """Own allowlist, inherited lexical skipping only; never call old census."""
    def __init__(self,stream,allowed):
        super().__init__(stream,FILE_CAP);self.allowed=allowed;self.containers={}
    def walk(self,path=()):
        require(len(path)<=128,'metadata depth cap');self.whitespace()
        if not any(lexical.matches(p,path,True) for p in self.allowed):self.skip();return
        ch=self.peek()
        if ch==123:
            self.get();self.whitespace();seen=set();self.containers[path]=seen
            if self.peek()==125:self.get();return
            while True:
                self.whitespace();key=self.string(True);require(key not in seen,'duplicate metadata key')
                seen.add(key);require(len(seen)<=40000,'object key cap')
                self.whitespace();require(self.get()==58,'JSON colon')
                self.walk(path+(key,));self.whitespace();sep=self.get()
                if sep==125:break
                require(sep==44,'JSON object separator')
        elif ch==91:
            self.get();self.whitespace();i=0;self.containers[path]=[]
            if self.peek()==93:self.get();return
            while True:
                require(i<40000,'array entry cap');self.containers[path].append(i)
                self.walk(path+(i,));i+=1;self.whitespace();sep=self.get()
                if sep==93:break
                require(sep==44,'JSON array separator')
        elif any(lexical.matches(p,path) for p in self.allowed):
            if ch==34:v=self.string(True)
            else:
                token=bytearray()
                while self.peek() not in (None,32,9,10,13,44,93,125):
                    token.append(self.get());require(len(token)<1024,'metadata scalar cap')
                v=json.loads(token.decode('ascii'))
            require(v is None or type(v) in (str,int,float,bool),'metadata scalar only')
            require(len(self.leaves)<40000,'metadata leaf cap');self.leaves[path]=v
        else:self.skip()


def project(stream,adapter):
    p=Projection(stream,patterns(adapter));p.walk();p.whitespace()
    require(p.peek() is None,'trailing projected JSON');return p


def projected_file(item,read_bytes=None):
    """Only this descriptor's path. RAM provider is for tests, not CLI."""
    path=Path(item['path'])
    if read_bytes is None:
        require(path.stat().st_size<=FILE_CAP,'source file cap')
        require(file_sha(path)==item['sha256'],'external evidence SHA')
        raw=None;size=path.stat().st_size
    else:
        raw=read_bytes(str(path));size=len(raw)
        require(size<=FILE_CAP and hashlib.sha256(raw).hexdigest()==item['sha256'],'external RAM evidence SHA')
    fmt=item['format']
    if fmt=='prepare_envelope':
        data=raw if raw is not None else path.read_bytes()
        require(len(data)<=256*1024,'small prepared envelope')
        e=small_json(data);fields(e,'schema plan_sha256 gzip_base64')
        require(e['schema'] in [b+'.envelope' for b in BASES] and item['adapter']=='prepare','known prepare envelope')
        packed=base64.b64decode(e['gzip_base64'],validate=True)
        d=zlib.decompressobj(31);decoded=d.decompress(packed,4*1024**2+1)
        require(d.eof and not d.unused_data and not d.unconsumed_tail and len(decoded)<=4*1024**2,'single bounded prepare gzip')
        # Existing producers emit canonical sort_keys JSON plus exactly one LF.
        hash_value(e['plan_sha256'])
        require(decoded.endswith(b'\n') and hashlib.sha256(decoded[:-1]).hexdigest()==e['plan_sha256'],'original canonical prepared content SHA')
        p=project(BytesIO(decoded),'prepare')
    else:
        stream=BytesIO(raw) if raw is not None else path.open('rb')
        try:
            if fmt=='gzip_json':
                with gzip.GzipFile(fileobj=stream) as expanded:p=project(expanded,item['adapter'])
            else:p=project(stream,item['adapter'])
        finally:stream.close()
    if read_bytes is None:require(file_sha(path)==item['sha256'],'evidence final TOCTOU')
    return p,size


def values(p,prefix):
    a=[(k[len(prefix)],v) for k,v in p.leaves.items() if k[:len(prefix)]==prefix and len(k)==len(prefix)+1]
    a.sort();return [v for _,v in a]


def state_identity(state):
    s=np.asarray(state,dtype='<f8');require(s.shape==(7,) and np.isfinite(s).all(),'finite exact state7')
    pose=s[:5].copy();pose[4]=np.remainder(pose[4],2*np.pi);pose[pose==0]=0.
    return dict(root_state_sha256=hashlib.sha256(s.tobytes()).hexdigest(),
        physical_pose_key=hashlib.sha256(pose.astype('<f8').tobytes()).hexdigest(),physical_pose=pose.tolist())


def same_pose(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return bool(np.all(np.abs(a[:4]-b[:4])<=1e-9) and abs(np.arctan2(np.sin(a[4]-b[4]),np.cos(a[4]-b[4])))<=1e-12)


def validate_request(request):
    fields(request,'schema cutoff_utc project_root protocol evidence_files expected_source_families declared_unresolved_sources approval_file')
    require(request['schema']==NAME+'.request','own request schema')
    require(type(request['cutoff_utc']) is str,'explicit UTC cutoff string')
    offset=datetime.fromisoformat(request['cutoff_utc'].replace('Z','+00:00')).utcoffset()
    require(offset is not None and offset.total_seconds()==0,'UTC cutoff with timezone')
    require(str(Path(request['project_root']).resolve())==str(ROOT),'explicit Python project root')
    fields(request['protocol'],'path sha256');hash_value(request['protocol']['sha256'])
    require(request['protocol']['path']==str(PROTOCOL),'exact protocol authority path')
    require(type(request['evidence_files']) is list and 0<len(request['evidence_files'])<=512,'explicit1..512 evidence files')
    ids=set();paths=set()
    for e in request['evidence_files']:
        fields(e,'source_id family path sha256 format adapter')
        require(type(e['source_id']) is str and 0<len(e['source_id'])<128 and '#' not in e['source_id'] and e['source_id'] not in ids,'unique bounded source id')
        require(type(e['family']) is str and 0<len(e['family'])<128,'explicit family')
        p=Path(e['path']);require(p.is_absolute() and p.is_relative_to(ROOT) and '..' not in p.parts and e['path'] not in paths,'unique explicit project evidence path')
        require(not any(c in e['path'] for c in '*?[]'),'no glob metacharacters in explicit evidence path')
        hash_value(e['sha256']);require(e['format'] in ('json','gzip_json','prepare_envelope'),'explicit encoding')
        require(e['adapter'] in ('prepare','sourcepack','task','metadata','legacy_index'),'known metadata projection adapter')
        ids.add(e['source_id']);paths.add(e['path'])
    for name in ('expected_source_families','declared_unresolved_sources'):
        require(type(request[name]) is list and all(type(x) is str and 0<len(x)<=1000 for x in request[name]),'explicit bounded scope list')
    if request['approval_file'] is not None:
        fields(request['approval_file'],'path sha256');hash_value(request['approval_file']['sha256'])
        require(Path(request['approval_file']['path']).is_absolute(),'external approval path')
    return deepcopy(request)


def build(request, *, read_bytes=None):
    """Deterministic assets, not scientific admission. RAM provider never scans."""
    q=validate_request(request);unknown=[];evidence=[];records=[];seeds={'root':set(),'reset':set()};spans=[];seed_authority=[];root_goals=set();total=0;expanded_total=0
    def unresolved(source,reason):unknown.append(dict(source_id=source,reason=reason))
    def read_small(desc):
        if read_bytes is None:require(Path(desc['path']).stat().st_size<=OUTPUT_CAP,'small authority file cap')
        raw=read_bytes(desc['path']) if read_bytes else Path(desc['path']).read_bytes()
        require(len(raw)<=OUTPUT_CAP and hashlib.sha256(raw).hexdigest()==desc['sha256'],'explicit small authority SHA')
        if read_bytes is None:require(file_sha(Path(desc['path']))==desc['sha256'],'authority TOCTOU')
        return raw
    protocol_raw=read_small(q['protocol'])
    require(b'moving_fresh_source_exclusions.v1' in protocol_raw,'fresh-root protocol schema authority')
    for x in q['declared_unresolved_sources']:unresolved('MAIN_declared',x)
    for family in sorted(set(REQUIRED_FAMILIES)|set(q['expected_source_families'])):
        if family not in {e['family'] for e in q['evidence_files']}:unresolved(family,'source_family_not_in_explicit_inventory')
    def add_seed(kind,v,ref):
        try:integer(v);seeds[kind].add(v);seed_authority.append(dict(source_ref=ref,kind=kind,seed=v))
        except ValueError:unresolved(ref,'invalid_'+kind+'_seed')
    def add_span(v,ref):
        try:
            require(len(v)==2,'half-open span');integer(v[0]);integer(v[1]);require(v[1]>v[0],'positive half-open span')
            spans.append(tuple(v))
            seed_authority.append(dict(source_ref=ref,kind='reset_span',span=list(v)))
        except (ValueError,TypeError):unresolved(ref,'invalid_half_open_reset_span')
    for e in sorted(q['evidence_files'],key=lambda e:e['source_id']):
        ref=e['source_id'];ev=deepcopy(e);ev.update(projected=False,projected_schema=None,metadata_scope=e['adapter']+'_identity_only_no_outcomes');evidence.append(ev)
        if total>=TOTAL_CAP or expanded_total>=TOTAL_CAP:
            unresolved(ref,'aggregate_metadata_budget_exhausted_not_opened');continue
        try:
            n=len(read_bytes(e['path'])) if read_bytes else Path(e['path']).stat().st_size
            total+=n;require(total<=TOTAL_CAP,'aggregate256MiB original file cap')
            p,_=projected_file(e,read_bytes);expanded_total+=p.bytes
            require(expanded_total<=TOTAL_CAP,'aggregate256MiB expanded metadata cap')
            adapter=e['adapter'];schema=p.leaves.get(('schema',));ev.update(projected=True,projected_schema=schema)
            valid=(schema in [b+'.prepare' for b in BASES]+['p1_conditional_procedural_source_20261027_v1.plan'] if adapter=='prepare' else
                schema in [b+'.sourcepack' for b in BASES] if adapter=='sourcepack' else
                schema==NAME+'.identity_metadata' if adapter=='metadata' else
                schema=='p1_conditional_source_metadata_20261027_v1.authority_evidence' if adapter=='legacy_index' else
                schema in ('p1_constructive_moving_history_encode_cuda_20261027_v1.task_export',
                           'p1_constructive_blind_encode_cuda_20261027_v1.task_export'))
            require(valid,'unsupported projected schema')
            if adapter=='legacy_index' or adapter=='prepare':
                base=() if adapter=='legacy_index' else ('exclusions',)
                for h in values(p,base+('root_hashes',)):hash_value(h);root_goals.add(h)
                for k,v in p.leaves.items():
                    if k[:len(base)+1]==base+('seed_families',) and k[-1]=='seed':add_seed('root',v,ref)
                for k in p.containers:
                    if len(k)==len(base)+3 and k[:len(base)+1]==base+('reset_spans',) and k[-1]=='span':add_span(values(p,k),ref)
                for k in p.containers:
                    if len(k)==len(base)+2 and k[:len(base)+1]==base+('unknown_sources',):
                        unresolved(ref,'inherited_unknown:'+str(p.leaves.get(k+('source',)))+':'+str(p.leaves.get(k+('reason',))))
                if adapter=='legacy_index':
                    unresolved(ref,'legacy_index_coverage_unknown_and_root_state_mapping_incomplete');continue
            container={'prepare':'slots','sourcepack':'rows','metadata':'records','task':None}[adapter]
            bases=[()] if container is None else [(container,i) for i in p.containers.get((container,),[])]
            require(bases,'no_identity_records_in_container')
            for b in bases:
                get=lambda k:p.leaves.get(b+(k,))
                record_id=get('record_id') if adapter=='metadata' else 'task' if adapter=='task' else '/'+str(container)+'/'+str(b[-1])
                require(type(record_id) is str and record_id and '#' not in record_id,'stable literal record id')
                rid=ref+'#'+record_id
                r=dict(record_ref=rid,source_id=ref,family=e['family'],evidence_sha256=e['sha256'],
                    record_type='reservation' if adapter=='prepare' else get('record_type') or 'observed',
                    root_seed=get('root_seed') if get('root_seed') is not None else p.leaves.get(b+('slot','root_seed')),
                    reset_seed=get('reset_seed') if get('reset_seed') is not None else p.leaves.get(b+('slot','reset_seed')),
                    root_goal_sha256=get('root_goal_sha256') or get('root_sha256'), task_id=get('task_id') or get('calibration_task_id'),
                    blind_sha256=get('blind_sha256') or get('blind_payload_sha256'),
                    episode_id=get('episode_id'),root_state=values(p,b+('root_state',)) or None,
                    root_state_sha256=get('root_state_sha256'),root_snapshot_sha256=get('root_snapshot_sha256'),
                    history_pixels_sha256=values(p,b+('history_pixels_sha256',)) or values(p,b+('history_frame_pixel_sha256',)) or None,
                    raw_history_sha256=get('raw_history_sha256'),maps_to=values(p,b+('maps_to',)),parent_refs=values(p,b+('parent_refs',)))
                require(r['record_type'] in ('reservation','observed','derived'),'explicit observed/reservation record type')
                for kind in seeds:
                    if r[kind+'_seed'] is not None:add_seed(kind,r[kind+'_seed'],rid)
                span=values(p,b+('reset_seed_span',))
                r['reset_seed_span']=span or None
                if span:add_span(span,rid)
                if r['root_goal_sha256'] is not None:hash_value(r['root_goal_sha256']);root_goals.add(r['root_goal_sha256'])
                for k in ('task_id','blind_sha256','root_state_sha256','root_snapshot_sha256','raw_history_sha256'):
                    if r[k] is not None:hash_value(r[k])
                if r['root_state'] is not None:
                    identity=state_identity(r['root_state'])
                    require(r['root_state_sha256'] in (None,identity['root_state_sha256']),'state7 bits/hash conflict')
                    r.update(identity)
                if r['history_pixels_sha256'] is not None:
                    require(len(r['history_pixels_sha256'])==3,'three history pixel SHA')
                    for h in r['history_pixels_sha256']:hash_value(h)
                records.append(r)
        except (ValueError,TypeError,KeyError,OSError,EOFError,UnicodeError,zlib.error) as exc:
            unresolved(ref,'projection_or_identity_failure:'+type(exc).__name__+':'+str(exc)[:240])
    require(len(records)<=4096,'bounded root identity record count')
    indexed={r['record_ref']:r for r in records}
    if len(indexed)!=len(records):unresolved('inventory','duplicate_record_ref')
    # MAIN metadata bridge assertions never erase a file failure/legacy unknown.
    mapped={}
    for r in records:
        for target in r['maps_to']:
            if target not in indexed:unresolved(r['record_ref'],'mapping_target_not_in_explicit_inventory:'+str(target));continue
            if target in mapped:unresolved(target,'ambiguous_identity_bridge');continue
            t=indexed[target]
            if not any(t[k] is not None and t[k]==r[k] for k in ('root_goal_sha256','task_id','episode_id','blind_sha256')):
                unresolved(target,'identity_bridge_has_no_matching_root_task_episode_key');continue
            if any(t.get(k) is not None and r.get(k) is not None and t[k]!=r[k] for k in ('root_goal_sha256','task_id','episode_id','blind_sha256','root_seed','reset_seed',
                    'root_state_sha256','root_snapshot_sha256','history_pixels_sha256','raw_history_sha256')):
                unresolved(target,'identity_bridge_conflict');continue
            mapped[target]=r
    observed=[]
    for r in records:
        for parent in r['parent_refs']:
            if parent not in indexed:unresolved(r['record_ref'],'unmapped_lineage_parent:'+str(parent))
        if r['record_type']=='reservation':
            if r['root_seed'] is None or (r['reset_seed'] is None and r['reset_seed_span'] is None):unresolved(r['record_ref'],'reservation_root_seed_and_reset_seed_or_span_missing')
            continue
        item=deepcopy(r);bridge=mapped.get(r['record_ref'])
        if bridge is not None:
            for k in ('root_state','root_state_sha256','physical_pose','physical_pose_key','root_snapshot_sha256','history_pixels_sha256','raw_history_sha256','root_seed','reset_seed'):
                if item.get(k) is None:item[k]=deepcopy(bridge.get(k))
            item['mapping_evidence']=bridge['record_ref']
        missing=[k for k in ('root_state','root_state_sha256','physical_pose_key','root_snapshot_sha256','history_pixels_sha256','raw_history_sha256','root_seed','reset_seed') if item.get(k) is None]
        if missing:unresolved(r['record_ref'],'observed_root_identity_missing:'+','.join(missing))
        observed.append(item)
    parent=list(range(len(observed)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    aliases=[]
    for i,a in enumerate(observed):
        for j in range(i):
            b=observed[j];why=[]
            for k in ('root_goal_sha256','root_state_sha256','physical_pose_key','root_snapshot_sha256','history_pixels_sha256'):
                if a.get(k) is not None and a.get(k)==b.get(k):why.append(k)
            if a.get('physical_pose') is not None and b.get('physical_pose') is not None and same_pose(a['physical_pose'],b['physical_pose']):why.append('physical_pose_numeric_alias')
            if ('root_goal_sha256' in why or 'root_snapshot_sha256' in why) and a.get('physical_pose') is not None and b.get('physical_pose') is not None and not same_pose(a['physical_pose'],b['physical_pose']):
                unresolved(a['record_ref'],'shared_authority_hash_conflicts_with_physical_pose:'+b['record_ref'])
            if why:parent[find(i)]=find(j);aliases.append(dict(a=a['record_ref'],b=b['record_ref'],reasons=why))
    groups={}
    for i,r in enumerate(observed):groups.setdefault(find(i),[]).append(r)
    roots=[]
    for members in groups.values():
        keys=sorted({r['physical_pose_key'] for r in members if r.get('physical_pose_key')})
        roots.append(dict(stable_root_id=('pose:'+keys[0]) if keys else 'unmapped:'+digest(sorted(r['record_ref'] for r in members)),
            member_refs=sorted(r['record_ref'] for r in members),physical_pose_keys=keys,
            root_goal_sha256s=sorted({r['root_goal_sha256'] for r in members if r['root_goal_sha256']})))
    candidate=[]
    for s in PROPOSED:
        conflicts=[]
        for kind in ('root','reset'):
            v=s[kind+'_seed']
            if v in seeds['root'] or v in seeds['reset']:conflicts.append(dict(candidate_field=kind+'_seed',reason='used_or_reserved_seed',seed=v))
            for lo,hi in sorted(set(spans)):
                if lo<=v<hi:conflicts.append(dict(candidate_field=kind+'_seed',reason='reserved_reset_span',span=[lo,hi]))
        candidate.append(dict(**s,conflicts=conflicts,independence_certified=False))
    content=dict(schema='moving_fresh_source_exclusions.v1',producer=NAME,cutoff_utc=q['cutoff_utc'],project_root=q['project_root'],
        protocol=deepcopy(q['protocol']),evidence_files=evidence,covered_source_families=sorted({e['family'] for e in evidence if e['projected']}),
        used_or_reserved_root_seeds=sorted(seeds['root']),used_or_reserved_reset_seeds=sorted(seeds['reset']),
        seed_authority=sorted({digest(x):x for x in seed_authority}.values(),key=lambda x:canonical(x)),
        reserved_reset_spans=[list(x) for x in sorted(set(spans))],root_goal_sha256s=sorted(root_goals),
        observed_roots=sorted(observed,key=lambda r:r['record_ref']),identity_records=sorted(records,key=lambda r:r['record_ref']),
        stable_roots=sorted(roots,key=lambda r:r['stable_root_id']),aliases=sorted(aliases,key=lambda r:(r['a'],r['b'])),
        proposed_seed_checks=candidate,unresolved_sources=sorted({digest(x):x for x in unknown}.values(),key=lambda x:(x['source_id'],x['reason'])),
        inventory_sha256=digest(sorted(q['evidence_files'],key=lambda e:e['source_id'])),
        semantics=dict(numpy_version=np.__version__,physical_state_hash='little-endian float64 state7; signed-zero bits preserved',
            pose='state[:4] + numpy.remainder(angle,2pi); +0 canonical; float64',
            alias_xy_tolerance_pixels=1e-9,alias_wrapped_angle_tolerance_rad=1e-12,
            root_goal_hash='original authority hash, not redefined',placement_reset_zero='shared explicit placement detail; requires MAIN ledger scope',
            outcome_values_decoded=0,recursive_scan=False,scientific_independence_certified=False),
        source_sha256={str(Path(__file__).resolve()):file_sha(Path(__file__).resolve()),str(Path(lexical.__file__).resolve()):file_sha(Path(lexical.__file__).resolve())},
        P1_admission=False,freshness_certified=False)
    content_hash=digest(content)
    approval=dict(authority='MAIN',approved=False,coverage_scope=[],external_file=q['approval_file'])
    approval_errors=[]
    if q['approval_file']:
        try:
            a=small_json(read_small(q['approval_file']));fields(a,'schema authority approved coverage_scope cutoff_utc manifest_content_sha256')
            require(a['schema']==NAME+'.MAIN_coverage_approval' and a['authority']=='MAIN' and a['approved'] is True,'external MAIN approval only')
            require(a['cutoff_utc']==q['cutoff_utc'] and a['manifest_content_sha256']==content_hash,'approval exact content/cutoff')
            require(type(a['coverage_scope']) is list and set(a['coverage_scope'])>=set(REQUIRED_FAMILIES)|set(q['expected_source_families']),'MAIN complete declared scope')
            approval.update(approved=True,coverage_scope=sorted(set(a['coverage_scope'])))
        except (ValueError,TypeError,KeyError,OSError) as exc:approval_errors.append(str(exc)[:240])
    blocked=bool(content['unresolved_sources'] or not approval['approved'] or approval_errors or any(r['conflicts'] for r in candidate))
    result=dict(content,coverage_content_sha256=content_hash,approval=approval,approval_errors=approval_errors,
        source_run_preconditions_met=not blocked,status='blocked_unknown_unapproved_or_seed_conflict' if blocked else 'coverage_approved_pending_new_root_identity_checks',
        request_sha256=digest(q),projected_file_bytes=total,expanded_metadata_bytes=expanded_total)
    require(len(canonical(result))<=OUTPUT_CAP,'manifest2MiB cap')
    return result


def write_new(output,result):
    path=Path(output);raw=canonical(result)
    require(path.is_absolute() and not path.exists() and not path.is_symlink() and len(raw)<=OUTPUT_CAP,'NEW bounded output only')
    require(shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'200MiB disk reserve')
    with path.open('xb') as stream:require(stream.write(raw)==len(raw),'short manifest write');stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=file_sha(path),bytes=len(raw),status=result['status'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--request-sha256',required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();hash_value(a.request_sha256)
    require(a.request.stat().st_size<=OUTPUT_CAP,'small request file cap')
    require(file_sha(a.request.resolve())==a.request_sha256,'external request SHA')
    result=build(small_json(a.request.read_bytes()))
    require(file_sha(a.request.resolve())==a.request_sha256,'request final TOCTOU')
    print(canonical(write_new(a.output,result) if a.output else result).decode(),end='')


if __name__=='__main__':main()
