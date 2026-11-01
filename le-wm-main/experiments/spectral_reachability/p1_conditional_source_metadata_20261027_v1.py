"""Metadata-only historical source resolver. NEVER interprets future values.

Report bytes are streamed for SHA and syntactic skipping. Only explicitly
whitelisted source identities/input bindings are decoded. This is NOT a
self-issued globally complete historical-source authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

NAME='p1_conditional_source_metadata_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
SKIP_MARKERS=re.compile(rb'["\[\]{}]')
PATTERNS=[
    ('schema',),('plan','schema'),
    ('plan','slots','*','root_seed'),('plan','slots','*','reset_seed_span','*'),
    ('slots','*','root_seed'),('slots','*','reset_seed_span','*'),
    ('slots','*','metadata','root_sha256'),
    ('root_seed',),('calibration_root_seed',),('root_count',),('root_sha256',),
    ('protocol','root_seed'),('protocol','calibration_root_seed'),
    ('protocol','root_count'),('protocol','calibration_roots'),
    ('protocol','root_oversample_factor'),('protocol','root_protocol'),
    ('protocol','outcome','root_seed'),('protocol','outcome','root_count'),
    ('protocol','outcome','oversample_factor'),
    ('root_hashes','*'),('roots','*'),('roots','*','root_sha256'),
    ('rows','*','root_sha256'),('calibration_roots','*','root_sha256'),
    ('protocol','roots','*'),('protocol','packets','*','root_sha256'),
    ('bindings','*'),('upstream_p0','path'),('upstream_p0','sha256'),
]


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def is_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def matches(pattern,path,prefix=False):
    return (len(path)<=len(pattern) if prefix else len(path)==len(pattern)) and all(a=='*' or a==b for a,b in zip(pattern,path))


class Projection:
    """Small streaming JSON projector; skipped scalar values are NOT decoded.

    Non-target subtrees are brace/string scanned, not json.loads'ed. Limits
    bound file bytes/depth/metadata leaves, without retaining trajectories.
    """
    def __init__(self,stream,limit):
        self.stream=stream;self.limit=limit;self.sha=hashlib.sha256();self.buffer=b'';self.index=0;self.bytes=0;self.leaves={}
    def peek(self):
        if self.index==len(self.buffer):
            self.buffer=self.stream.read(65536);self.index=0;self.bytes+=len(self.buffer)
            require(self.bytes<=self.limit,'metadata projection byte cap exceeded')
            self.sha.update(self.buffer)
        return self.buffer[self.index] if self.index<len(self.buffer) else None
    def get(self):
        value=self.peek()
        if value is not None:self.index+=1
        return value
    def whitespace(self):
        while self.peek() in (32,9,10,13):self.get()
    def string(self,decode):
        require(self.get()==34,'JSON string expected')
        raw=bytearray(b'"') if decode else None
        escaped=False
        while True:
            ch=self.get();require(ch is not None,'unterminated JSON string')
            if raw is not None:
                raw.append(ch);require(len(raw)<=1_000_000,'metadata string cap')
            if escaped:escaped=False
            elif ch==92:escaped=True
            elif ch==34:break
        return json.loads(raw.decode('utf8')) if raw is not None else None
    def skip(self):
        self.whitespace();ch=self.peek();require(ch is not None,'missing JSON value')
        if ch==34:self.string(False);return
        if ch in (123,91):
            stack=[self.get()]
            while stack:
                ch=self.peek();require(ch is not None,'unterminated skipped JSON subtree')
                # Jump over opaque runs of numbers/whitespace/field text in C;
                # do not decode them and do not retain them as Python values.
                marker=SKIP_MARKERS.search(self.buffer,self.index)
                if marker is None:self.index=len(self.buffer);continue
                self.index=marker.start();ch=self.peek()
                if ch==34:self.string(False);continue
                ch=self.get()
                if ch in (123,91):stack.append(ch);require(len(stack)<=256,'JSON depth cap')
                elif ch in (125,93):
                    require(stack.pop()==(123 if ch==125 else 91),'mismatched skipped JSON container')
            return
        length=0
        while self.peek() not in (None,32,9,10,13,44,93,125):self.get();length+=1
        require(length>0,'missing skipped JSON scalar')
    def walk(self,path=()):
        require(len(path)<=256,'JSON depth cap');self.whitespace()
        if not any(matches(p,path,True) for p in PATTERNS):self.skip();return
        ch=self.peek()
        if ch==123:
            self.get();self.whitespace();seen=set()
            if self.peek()==125:self.get();return
            while True:
                self.whitespace();key=self.string(True)
                require(key not in seen,'duplicate JSON metadata key');seen.add(key)
                self.whitespace();require(self.get()==58,'JSON colon expected')
                self.walk(path+(key,));self.whitespace();sep=self.get()
                if sep==125:break
                require(sep==44,'JSON object separator expected')
        elif ch==91:
            self.get();self.whitespace();i=0
            if self.peek()==93:self.get();return
            while True:
                self.walk(path+(i,));i+=1;self.whitespace();sep=self.get()
                if sep==93:break
                require(sep==44,'JSON array separator expected')
        elif any(matches(p,path) for p in PATTERNS):
            if ch==34:value=self.string(True)
            else:
                token=bytearray()
                while self.peek() not in (None,32,9,10,13,44,93,125):
                    token.append(self.get());require(len(token)<=1024,'metadata scalar cap')
                value=json.loads(token.decode('ascii'))
            require(value is None or type(value) in (str,int,bool,float),'metadata scalar only')
            require(len(self.leaves)<40000,'metadata leaf cap');self.leaves[path]=value
        else:self.skip()


def project_metadata(path,byte_limit=64*1024**2):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'regular immutable metadata container required')
    with path.open('rb') as stream:
        parser=Projection(stream,byte_limit);parser.walk();parser.whitespace()
        require(parser.peek() is None,'trailing metadata-container content')
    return dict(sha256=parser.sha.hexdigest(),leaves=parser.leaves,bytes=parser.bytes)


def resolve_history(project=PROJECT):
    project=Path(project).resolve();results=project/'results'
    directories=sorted(x for x in results.iterdir() if x.is_dir() and
        (x.name.startswith('pusht') or (x/'sourcepack.json').is_file() or (x/'plan.json').is_file()))
    require(len(directories)<=4096,'explicit metadata directory census cap')
    # Top-level prepared requests reserve the entire fixed pool, including
    # rejected/unselected proposals. Their filenames need not start pusht.
    files=[*results.glob('*request*.json'),*results.glob('*plan*.json')]
    for directory in directories:
        candidates=[directory/'report.json',directory/'protocol.json',directory/'sourcepack.json',directory/'plan.json',*sorted(directory.glob('root_*.json')),
                    *sorted(directory.glob('*source*/*args.json'))]
        files.extend(x for x in candidates if x.is_file())
    files=sorted(set(files));require(len(files)<=4096,'explicit metadata file census cap')
    result=dict(schema=NAME+'.authority_evidence',files={},root_hashes=[],seed_families=[],reset_spans=[],
                evidence=[],unknown_sources=[],non_origin_directories=[],
                confirmation_authority=dict(status='unknown',self_certified_complete=False,
                    scope='Whitelisted source identity metadata; omitted/nested histories and unbound historical reset attempts are NOT certified.'),
                physical_future_values_decoded=0,new_neural_calls=0,new_resets=0,new_public_steps=0)
    def unknown(source,reason):result['unknown_sources'].append(dict(source=str(source),reason=reason))
    records={};budget=0;root_seeds={};known_roots=set();directory_records={}
    for path in files:
        try:
            record=project_metadata(path);budget+=record['bytes']
            require(budget<=256*1024**2,'total metadata byte scan cap')
        except (ValueError,UnicodeError) as exc:
            unknown(path,'unparseable_source_metadata_container:'+type(exc).__name__);continue
        result['files'][str(path)]=record['sha256'];leaves=record['leaves']
        roots={v for k,v in leaves.items() if (k[-1]=='root_sha256' or 'roots' in k or 'root_hashes' in k) and is_hash(v)}
        own_plan='p1_conditional_procedural_source_20261027_v1.plan'
        prefix=('plan',) if leaves.get(('plan','schema'))==own_plan else ()
        is_own=leaves.get(prefix+('schema',))==own_plan
        if is_own:
            for key,value in leaves.items():
                if key[:len(prefix)+1]!=prefix+('slots',) or key[-1]!='root_seed':continue
                require(type(value) is int and value>=0,'invalid procedural parent seed metadata')
                base=key[:-1]+('reset_seed_span',)
                span=[leaves.get(base+(i,)) for i in range(2)]
                require(all(type(x) is int and x>=0 for x in span) and span[1]>span[0], 'invalid procedural reset span metadata')
                result['seed_families'].append(dict(seed=value,usage='used_or_planned_input',evidence=record['sha256']))
                result['reset_spans'].append(dict(span=span,usage='used_or_planned_input',evidence=record['sha256']))
                for root in roots:root_seeds.setdefault(root,set()).add(value)
                result['evidence'].append(dict(kind='bounded_procedural_plan_reservation',source=str(path),
                    seed=value,reset_seed_span=span,metadata_sha256=record['sha256']))
        known_roots.update(roots)
        seed=next((leaves[k] for k in (('protocol','root_seed'),('protocol','calibration_root_seed'),('root_seed',),('calibration_root_seed',))
                   if k in leaves and type(leaves[k]) is int and leaves[k]>=0),None)
        count=next((leaves[k] for k in (('protocol','root_count'),('protocol','calibration_roots'),('root_count',))
                    if k in leaves and type(leaves[k]) is int and leaves[k]>0),None)
        record.update(roots=roots,seed=seed);records[str(path)]=record
        directory=next((x for x in directories if path.is_relative_to(x)),None)
        if directory is not None:directory_records.setdefault(str(directory),[]).append(str(path))
        if seed is not None:
            result['seed_families'].append(dict(seed=seed,usage='used_or_planned_input',evidence=record['sha256']))
            for root in roots:root_seeds.setdefault(root,set()).add(seed)
            result['evidence'].append(dict(kind='explicit_source_seed',source=str(path),seed=seed,root_hashes=sorted(roots),metadata_sha256=record['sha256']))
            if count is not None:
                # Protect a deliberately loose CURRENT-factory envelope, not
                # a certification of the missing HISTORICAL runtime closure.
                N=count*int(leaves.get(('protocol','root_oversample_factor'),4))
                bound=20*max(8*N,N+16)
                result['reset_spans'].append(dict(span=[seed,seed+bound],usage='used_or_planned_input',evidence=record['sha256']))
            unknown(path,'historical_reset_attempt_bound_not_attested')
        reserved=leaves.get(('protocol','outcome','root_seed'))
        if type(reserved) is int and reserved>=0:
            result['seed_families'].append(dict(seed=reserved,usage='reserved_not_opened',evidence=record['sha256']))
            n=leaves.get(('protocol','outcome','root_count'))
            if type(n) is int and n>0:
                result['reset_spans'].append(dict(span=[reserved,reserved+20*max(8*n*4,n*4+16)],usage='reserved_not_opened',evidence=record['sha256']))
    # Explicit hash-bound input edges, grouped with the output metadata from
    # the SAME run directory. Merely mentioning an ancestor is insufficient:
    # observed child root identities must also be its subset.
    for directory,names in directory_records.items():
        child_roots=set().union(*(records[n]['roots'] for n in names))
        for name in names:
            for key,value in records[name]['leaves'].items():
                if len(key)!=2 or key[0]!='bindings' or not is_hash(value):continue
                reference=str(Path(key[1]).resolve())
                if reference not in records or not records[reference]['roots']:continue
                parent=records[reference]
                if value!=parent['sha256']:unknown(name,'input_binding_hash_mismatch');continue
                if child_roots and child_roots<=parent['roots']:
                    result['evidence'].append(dict(kind='root_set_inherited',source=directory,parent=reference,
                        parent_metadata_sha256=value,root_hashes=sorted(child_roots),input_metadata_sha256=records[name]['sha256']))
    for root in sorted(known_roots):
        if root not in root_seeds:unknown(root,'root_has_no_seed_provenance')
    # Empty/analysis-only directory NAMES are not source evidence and do NOT
    # create permanent unknown-origin objects. Coverage scope stays explicit.
    for directory in directories:
        names=directory_records.get(str(directory),[])
        if not any(records[n]['roots'] or records[n]['seed'] is not None for n in names):
            result['non_origin_directories'].append(str(directory))
    result['root_hashes']=sorted(known_roots)
    result['census']=dict(directories=len(directories),metadata_files=len(files),projected_bytes=budget,
                          directory_inventory_sha256=digest([str(p) for p in directories]))
    return result


def as_exclusions(authority):
    """No fake complete flag: use known exclusions for CALIBRATION ONLY."""
    generator_name='p1_conditional_procedural_source_20261027_v1'
    require(authority['schema']==NAME+'.authority_evidence' and authority['physical_future_values_decoded']==0,'own metadata-only evidence required')
    bound_files=dict(authority['files']);bound_files['metadata_authority_content_id']=digest(authority)
    source=Path(__file__).resolve();bound_files[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    return dict(schema=generator_name+'.exclusions',environment='swm/PushT-v1',files=bound_files,
                root_hashes=authority['root_hashes'],seed_families=authority['seed_families'],reset_spans=authority['reset_spans'],
                unknown_sources=authority['unknown_sources'],coverage=dict(status='unknown',authority_sha256=None,
                scope=authority['confirmation_authority']['scope']),physical_future_labels_read=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',type=Path,default=PROJECT)
    parser.add_argument('--output',type=Path);parser.add_argument('--as-exclusions',action='store_true')
    args=parser.parse_args();result=resolve_history(args.project_root)
    if args.as_exclusions:result=as_exclusions(result)
    raw=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(raw)
    print(raw)


if __name__=='__main__':main()
