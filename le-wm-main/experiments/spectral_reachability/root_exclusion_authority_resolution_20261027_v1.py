"""Explicit historical metadata request producer; never certifies freshness.

Only pinned prepares, their literal unknown-source paths, and caller-supplied
inventory descriptors are opened. Outcome payloads are lexically skipped.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import numpy as np
from experiments.spectral_reachability import moving_root_exclusion_manifest_20261027_v1 as v

NAME = 'root_exclusion_authority_resolution_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
FILE_CAP = 128*1024**2
TOTAL_CAP = 512*1024**2
OUTPUT_CAP = 8*1024**2
FLOOR = 200*1024**2
PINNED = (
 ('reachable_prepare','results/p1_constructive_reachable_source_20261027_v1_prepare01.json','98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086'),
 ('reachable_source','results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json','9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9'),
 ('moving_prepare','results/p1_constructive_moving_history_source_20261027_v1_prepare01.json','c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9'),
 ('moving_source','results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json','c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'),
 ('moving_task00','results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz','a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b'),
 ('moving_task01','results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz','101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74'))
IDENTITY = ('schema','environment','env_name','env_id','root_seed','reset_seed','calibration_root_seed',
 'root_count','root_index','source_slot','root_sha256','root_goal_sha256','task_id','episode_id',
 'blind_sha256','blind_payload_sha256','root_protocol','root_oversample_factor')
CONTAINERS = ((),('protocol',),('plan',),('args',),('source',),('metadata',),('protocol','outcome'),
 ('roots','*'),('rows','*'),('slots','*'),('calibration_roots','*'),('protocol','roots','*'),
 ('protocol','packets','*'),('plan','slots','*'),('rows','*','slot'),('slots','*','metadata'))
META_PATTERNS = [c+(k,) for c in CONTAINERS for k in IDENTITY] + [
 ('root_hashes','*'),('roots','*'),('protocol','roots','*'),
 ('slots','*','reset_seed_span','*'),('plan','slots','*','reset_seed_span','*')]
META_PATTERNS += [prefix+('*',)*depth for prefix in (('bindings',),('plans_bindings',),
    ('protocol','bindings'),('source_bindings',)) for depth in (1,2,3)]


def project(stream, patterns):
    p = v.Projection(stream, patterns); p.limit = FILE_CAP
    p.walk(); p.whitespace(); v.require(p.peek() is None, 'trailing JSON')
    return p


def pointer(path):
    return '/'+('/'.join(str(x).replace('~','~0').replace('/','~1') for x in path))


def safe_path(path, root=ROOT):
    p=Path(path)
    v.require(p.is_absolute() and '..' not in p.parts and p.resolve()==p and p.is_relative_to(root), 'literal project path required')
    return p


class Reader:
    def __init__(self, root=ROOT, provider=None):
        self.root=Path(root); self.provider=provider; self.used=0; self.files={}
    def read(self, path, expected=None):
        p=safe_path(path,self.root)
        if str(p) in self.files:
            entry=self.files[str(p)]
            if expected is not None:v.require(entry['sha256']==expected,'conflicting external SHA')
            raw=self.provider(str(p)) if self.provider else p.read_bytes()
            v.require(hashlib.sha256(raw).hexdigest()==entry['sha256'],'evidence changed on reread')
            return raw
        size=len(self.provider(str(p))) if self.provider else p.stat().st_size
        v.require(size<=FILE_CAP and self.used+size<=TOTAL_CAP,'metadata read budget')
        raw=self.provider(str(p)) if self.provider else p.read_bytes()
        v.require(len(raw)==size,'size changed')
        h=hashlib.sha256(raw).hexdigest()
        if expected is not None:v.hash_value(expected); v.require(h==expected,'external SHA mismatch: '+str(p))
        self.used+=size; self.files[str(p)]=dict(path=str(p),sha256=h,bytes=size)
        return raw


def unpack_prepare(raw):
    v.require(len(raw)<=256*1024,'prepare cap'); e=v.small_json(raw)
    v.fields(e,'schema plan_sha256 gzip_base64')
    v.require(e['schema'] in [b+'.envelope' for b in v.BASES],'known prepare envelope')
    import zlib
    z=zlib.decompressobj(31); d=z.decompress(base64.b64decode(e['gzip_base64'],validate=True),4*1024**2+1)
    v.require(z.eof and not z.unconsumed_tail and not z.unused_data and len(d)<=4*1024**2,'bounded single prepare gzip')
    v.require(d.endswith(b'\n') and hashlib.sha256(d[:-1]).hexdigest()==e['plan_sha256'],'original prepare semantic SHA')
    return project(BytesIO(d),v.patterns('prepare')+ [('exclusions','files','*')])


def inherited(prepares):
    refs={}
    for source_id,p in prepares:
        for path,val in p.leaves.items():
            if path[:2]==('exclusions','unknown_sources') and path[-1]=='source':
                expected=p.leaves.get(('exclusions','files',val))
                refs.setdefault(val,[]).append(dict(parent=source_id,pointer=pointer(path),
                    original_reason=p.leaves.get(path[:-1]+('reason',)),historical_sha256=expected))
    return refs


def capture_history(reader, refs, extra=()):
    entries=[]
    inventory={str(x['path']):x for x in extra}
    v.require(len(inventory)==len(extra),'duplicate inventory path')
    for path in sorted(set(refs)|set(inventory)):
        row=dict(path=path,inherited_refs=refs.get(path,[]),projection=[],blockers=[],
            inventory_family=inventory.get(path,{}).get('family'))
        attested=sorted(set(x['historical_sha256'] for x in row['inherited_refs'] if x['historical_sha256']))
        if path in inventory:attested.append(inventory[path]['sha256'])
        try:
            v.require(len(set(attested))<=1,'conflicting historical SHA')
            raw=reader.read(path,attested[0] if attested else None)
            row.update(reader.files[path]);row['hash_authority']='external_attested' if attested else 'first_capture_not_historical_attestation'
            p=project(BytesIO(raw),META_PATTERNS)
            row['top_level_keys']=sorted(p.containers.get((),[]))
            row['projection']=[dict(pointer=pointer(k),value=val) for k,val in sorted(p.leaves.items(),key=lambda x:str(x[0]))]
            if not any(k[-1] in ('root_seed','reset_seed','root_sha256','root_goal_sha256') for k in p.leaves):
                row['blockers'].append('no_direct_root_seed_projection; explicit_parent_mapping_required')
            if any(x['original_reason']=='historical_reset_attempt_bound_not_attested' for x in row['inherited_refs']):
                row['blockers'].append('historical_reset_attempt_bound_not_attested; root_seed_is_not_reset_attempt_ledger')
            row['status']='metadata_captured_not_coverage_certified'
        except (OSError,ValueError,KeyError) as exc:
            row['status']='blocked';row['blockers'].append(type(exc).__name__+': '+str(exc))
        entries.append(row)
    return entries


def link_captured_metadata(entries):
    """Input-binding edges only, never infer physical equality from lineage."""
    by_path={x['path']:x for x in entries};by_sha={}
    for x in entries:
        if 'sha256' in x:by_sha.setdefault(x['sha256'],[]).append(x['path'])
    for entry in entries:
        links=[]
        for field in entry['projection']:
            value=field['value']
            if type(value) is not str:continue
            targets=([value] if value in by_path else [])+by_sha.get(value,[])
            for target in sorted(set(targets)):
                if target!=entry['path']:
                    links.append(dict(pointer=field['pointer'],target_path=target,
                        target_sha256=by_path[target].get('sha256'),
                        relation='explicit_input_binding_not_physical_root_equivalence'))
        entry['input_lineage_links']=links
        entry['root_identity_projections']=[x for x in entry['projection'] if x['pointer'].split('/')[-1] in
            ('root_seed','reset_seed','root_sha256','root_goal_sha256','task_id','episode_id','blind_sha256')]


def stable_projections(entries):
    roots={};seeds=[]
    for entry in entries:
        for item in entry['projection']:
            tokens=item['pointer'].split('/');value=item['value']
            evidence=dict(path=entry['path'],file_sha256=entry.get('sha256'),pointer=item['pointer'])
            if tokens[-1] in ('root_seed','reset_seed','calibration_root_seed'):
                v.integer(value);seeds.append(dict(kind=tokens[-1],seed=value,**evidence))
            is_root=(tokens[-1] in ('root_sha256','root_goal_sha256') or
                (len(tokens)>2 and tokens[-2] in ('root_hashes','roots')))
            if is_root and type(value) is str and len(value)==64:
                v.hash_value(value);roots.setdefault(value,[]).append(evidence)
    return dict(root_goal_groups=[dict(root_goal_sha256=h,stable_metadata_id='root_goal:'+h,
        evidence=roots[h],physical_root_equivalence_proved=False) for h in sorted(roots)],
        seed_projections=seeds,interpretation='Root-goal identity only; different goals may share the same physical root. Seed projections do not infer reset attempts or root index assignments.')


def tree(p,prefix):
    """Rebuild only the explicitly projected metadata subtree."""
    if prefix in p.leaves:return p.leaves[prefix]
    children=p.containers.get(prefix)
    v.require(children is not None,'missing metadata subtree '+str(prefix))
    if isinstance(children,list):return [tree(p,prefix+(i,)) for i in children]
    return {k:tree(p,prefix+(k,)) for k in sorted(children)}


SNAPSHOT=[('snapshot','state','*'),('snapshot','state_float64_bits'),
 ('snapshot','block_velocity','*'),('snapshot','block_angular_velocity'),('snapshot','block_angle'),
 ('snapshot','bodies','*','*'),('snapshot','bodies','*','*','*')]
HEADER=[(k,) for k in ('kind','index','previous_sha256','event_sha256','phase','step','resource')]+[('slot',),('slot','index')]


def journal_roots(raw,tail):
    observations={};actions={};previous=None;active=None;count=0
    for count,line in enumerate(BytesIO(raw),1):
        v.require(len(line)<=2*1024**2,'journal line cap')
        h=project(BytesIO(line),HEADER);g=lambda key:h.leaves.get((key,))
        v.require(g('index')==count-1 and g('previous_sha256')==previous,'journal order/chain')
        unhashed,n=re.subn(rb'"event_sha256": "[0-9a-f]{64}", ',b'',line.rstrip(b'\n'))
        v.require(n==1 and hashlib.sha256(unhashed).hexdigest()==g('event_sha256'),'raw canonical journal event SHA')
        previous=g('event_sha256');step=g('step');phase=g('phase')
        if g('kind')=='fresh_observation' and phase=='replay0' and step in (0,5,10):
            p=project(BytesIO(line),HEADER+SNAPSHOT+[('state','*'),('pixel_sha256',),('file_sha256',),('png',)])
            active=g('slot'); key=(active,step);v.require(key not in observations,'duplicate root checkpoint')
            observations[key]=dict(state=v.values(p,('state',)),snapshot=tree(p,('snapshot',)),
                pixel_sha256=p.leaves[('pixel_sha256',)],png_sha256=p.leaves[('file_sha256',)],
                png=p.leaves[('png',)],event_sha256=previous,journal_index=count-1)
        if phase=='replay0' and type(step) is int and 1<=step<=10 and g('resource')=='public_step':
            slot=g('slot') if g('slot') is not None else active
            if g('kind')=='operation_attempted':
                p=project(BytesIO(line),[('raw_action','*')]);a=v.values(p,('raw_action',))
                v.require(len(a)==2 and (slot,step) not in actions,'unique raw action')
                actions[slot,step]=dict(raw=a,completed=False)
            if g('kind')=='operation_completed':
                v.require((slot,step) in actions,'completion without attempt');actions[slot,step]['completed']=True
    v.require(previous==tail and count>0,'sourcepack journal tail')
    return observations,actions


def physical_bridges(reader,source_id,path,source_p,tasks):
    tail=source_p.leaves[('journal_chain_tail',)]
    journal=path.parent/'journal.jsonl';raw=reader.read(str(journal))
    obs,actions=journal_roots(raw,tail);records=[]
    for slot in (0,1):
        prefix=('rows',slot);row=lambda k:source_p.leaves.get(prefix+(k,))
        checkpoints=[obs[slot,s] for s in (0,5,10)]
        a=np.asarray([actions[slot,s]['raw'] for s in range(1,11)],dtype='<f4')
        v.require(all(actions[slot,s]['completed'] for s in range(1,11)) and a.shape==(10,2) and np.isfinite(a).all(),'complete raw10')
        pixels=[x['pixel_sha256'] for x in checkpoints];ah=hashlib.sha256(a.tobytes()).hexdigest()
        declared=v.values(source_p,prefix+('history_frame_pixel_sha256',))
        if declared:v.require(declared==pixels and row('raw_history_sha256')==ah,'sourcepack history binding')
        record=dict(record_id=source_id+'_root'+str(slot),record_type='observed',root_state=checkpoints[-1]['state'],
            root_goal_sha256=row('root_sha256'),root_seed=source_p.leaves[prefix+('slot','root_seed')],
            reset_seed=source_p.leaves[prefix+('slot','reset_seed')],blind_sha256=row('blind_payload_sha256'),
            root_snapshot_sha256=hashlib.sha256(json.dumps(checkpoints[-1]['snapshot'],sort_keys=True,allow_nan=False).encode()).hexdigest(),
            history_pixels_sha256=pixels,raw_history_sha256=ah,parent_refs=[],maps_to=[source_id+'#/rows/'+str(slot)])
        record.update(v.state_identity(record['root_state']))
        for task_id,tp in tasks:
            if tp.leaves.get(('blind_sha256',))==record['blind_sha256']:
                v.require(tp.leaves.get(('source_slot',))==slot,'task source slot mismatch')
                record['maps_to'].append(task_id+'#task');record['task_id']=tp.leaves[('task_id',)]
        records.append(dict(metadata=record,evidence=dict(journal=dict(reader.files[str(journal)]),
            journal_tail=tail,checkpoints=checkpoints,raw_history=a.tolist(),raw_history_basis='journal_attempt_and_completed',
            snapshot_algorithm='sha256(json.dumps(sort_keys=True,allow_nan=False).encode()); exported bodies, not solver cache')))
    return records


def generate(reader,inventory=()):
    prepares=[];sources=[];tasks=[];initial=[]
    for sid,rel,h in PINNED:
        path=reader.root/rel;raw=reader.read(str(path),h)
        adapter='prepare' if sid.endswith('prepare') else ('task' if 'task' in sid else 'sourcepack')
        if adapter=='prepare':p=unpack_prepare(raw);prepares.append((sid,p))
        else:
            patterns=v.patterns(adapter)+[('journal_chain_tail',)]
            p=project(BytesIO(gzip.decompress(raw) if adapter=='task' else raw),patterns)
            if adapter=='task':tasks.append((sid,p))
            else:sources.append((sid,path,p))
        initial.append(dict(source_id=sid,**reader.files[str(path)],adapter=adapter,
            family='reserved_prepares' if adapter=='prepare' else ('moving' if sid.startswith('moving') else 'constructive_zero'),
            format='prepare_envelope' if adapter=='prepare' else ('gzip_json' if adapter=='task' else 'json')))
    refs=inherited(prepares);entries=capture_history(reader,refs,inventory);link_captured_metadata(entries);bridges=[];failures=[]
    for sid,path,p in sources:
        try:bridges.extend(physical_bridges(reader,sid,path,p,tasks))
        except (ValueError,KeyError,OSError) as exc:failures.append(dict(source=sid,reason=type(exc).__name__+': '+str(exc)))
    authority=[]
    for family in ('legacy_zero','original_four','overlay','feedback'):
        supplied=[x for x in inventory if x.get('family')==family]
        authority.append(dict(family=family,inventory_paths=[x['path'] for x in supplied],
            status='inventory_received_requires_MAIN_scope_approval' if supplied else 'blocked_inventory_authority_not_received',
            remaining='all attempted/reserved/unselected sources and reset attempt coverage require explicit authority; filenames do not prove family completeness'))
    protocol=reader.root.parent/'docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md'
    return dict(schema=NAME+'.resolution_request',protocol_binding=dict(path=str(protocol),sha256=v.file_sha(protocol)),source_closure=[dict(path=str(Path(__file__).resolve()),sha256=v.file_sha(Path(__file__).resolve())),
        dict(path=str(Path(v.__file__).resolve()),sha256=v.file_sha(Path(v.__file__).resolve())),
        dict(path=str(Path(v.lexical.__file__).resolve()),sha256=v.file_sha(Path(v.lexical.__file__).resolve()))],
        pinned_evidence=initial,inherited_reference_count=sum(map(len,refs.values())),unique_inherited_paths=len(refs),
        history_file_requests=entries,physical_identity_bridges=bridges,physical_mapping_errors=failures,authority_requests=authority,
        stable_identity_projections=stable_projections(entries),
        bridge_payload=dict(schema=v.NAME+'.identity_metadata',records=[b['metadata'] for b in bridges]),
        summary=dict(captured_files=sum('sha256' in x for x in entries),metadata_projected_files=sum(bool(x['projection']) for x in entries),
            inherited_refs_with_projection=sum(len(x['inherited_refs']) for x in entries if x['projection']),
            inherited_refs_without_direct_identity=sum(len(x['inherited_refs']) for x in entries if not x['root_identity_projections']),
            inherited_reset_bound_unattested_refs=sum(z['original_reason']=='historical_reset_attempt_bound_not_attested' for x in entries for z in x['inherited_refs']),
            files_with_direct_root_identity=sum(bool(x['root_identity_projections']) for x in entries),
            files_with_explicit_input_links=sum(bool(x['input_lineage_links']) for x in entries),
            mapped_original_records=sum(len(b['metadata']['maps_to']) for b in bridges),bytes_read_unique=reader.used),
        budgets=dict(file_bytes=FILE_CAP,total_bytes=TOTAL_CAP,output_bytes=OUTPUT_CAP,disk_floor=FLOOR,neural=0,physics=0),
        MAIN_coverage_approval=False,freshness_certified=False,P1_admission=False,
        note='Metadata acquisition is not historical coverage adjudication. Original v1 unknowns remain immutable; no override or self-approval.')


def write_new(path,result):
    raw=v.canonical(result);p=Path(path)
    v.require(len(raw)<=OUTPUT_CAP and not p.exists() and p.parent.is_dir() and not p.is_symlink(),'NEW small output only')
    v.require(shutil.disk_usage(p.parent).free>=FLOOR+len(raw),'200MiB reserve')
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory');ap.add_argument('--inventory-sha256');ap.add_argument('--output')
    args=ap.parse_args();inventory=[]
    if args.inventory:
        v.require(args.inventory_sha256 is not None,'external inventory SHA required')
        p=safe_path(args.inventory);v.require(v.file_sha(p)==args.inventory_sha256,'inventory SHA')
        payload=v.small_json(p.read_bytes());v.fields(payload,'schema files')
        v.require(payload['schema']==NAME+'.inventory','inventory schema')
        inventory=payload['files'];v.require(type(inventory) is list and len(inventory)<=1024,'inventory cap')
        for x in inventory:v.fields(x,'path sha256 family');v.hash_value(x['sha256']);safe_path(x['path'])
    else:v.require(args.inventory_sha256 is None,'inventory required with SHA')
    result=generate(Reader(),inventory)
    result['inventory_binding']=dict(path=args.inventory,sha256=args.inventory_sha256)
    if args.output:print(json.dumps(dict(path=args.output,sha256=write_new(args.output,result),summary=result['summary']),sort_keys=True))
    else:print(v.canonical(result).decode(),end='')


if __name__=='__main__':main()
