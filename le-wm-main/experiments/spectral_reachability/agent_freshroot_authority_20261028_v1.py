"""Bounded evidence-request/response syntax and byte binding, NOT authority approval.

No discovery, environment/model imports, reset, collection or shell execution.
Only the pinned prior catalog and explicit response assets can be read.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

NAME = 'agent_freshroot_authority_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT/'results/agent_freshroots_resume_20261027_coverage.json'
CATALOG_SHA = '00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb'
CAP = 4*1024**2
ASSET_CAP = 32*1024**2
TOTAL_CAP = 128*1024**2
ASSET_COUNT = 128


def need(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(pairs):
        result={}
        for k,v in pairs:
            need(k not in result,'duplicate key');result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:need(False,'nonfinite JSON'))


def fields(v, keys):
    need(type(v) is dict and set(v)==set(keys.split()),'exact fields: '+keys)


def hash_value(v):
    need(type(v) is str and re.fullmatch('[a-f0-9]{64}',v),'SHA256')


def text(v):
    need(type(v) is str and 0<len(v)<=4096,'nonempty bounded text')


def integer(v):
    need(type(v) is int and 0<=v<=10**9,'nonnegative integer')


class Reader:
    def __init__(self, provider=None):
        self.provider=provider;self.used=0;self.cache={}

    def read(self, path, expected, cap=ASSET_CAP):
        hash_value(expected);p=Path(path)
        need(p.is_absolute() and '..' not in p.parts,'literal absolute path')
        if path in self.cache:
            raw=self.cache[path];need(sha(raw)==expected,'conflicting asset SHA');return raw
        need(len(self.cache)<ASSET_COUNT,'asset count cap')
        if self.provider is None:
            need(p.resolve()==p and p.is_file() and not p.is_symlink(),'regular canonical file')
            need(p.stat().st_size<=cap,'asset byte cap');raw=p.read_bytes()
        else: raw=self.provider(path)
        need(type(raw) is bytes and len(raw)<=cap,'asset byte cap')
        need(self.used+len(raw)<=TOTAL_CAP,'total byte cap')
        need(sha(raw)==expected,'asset SHA mismatch')
        self.used+=len(raw);self.cache[path]=raw;return raw


def requests(catalog):
    need(len(catalog['sources'])==8,'eight sources')
    runs=catalog['recovery']['dispatches']
    need(len(runs)==10,'ten known dispatches')
    result=[]
    for s in catalog['sources']:
        ds=[d for d in runs if d['source_id']==s['id']]
        need(ds and len(s['references'])==2,'two inherited references')
        result.append(dict(source_id=s['id'],report=dict(path=s['path'],sha256=s['sha256']),
            inherited_refs=[dict(request_id=s['id']+'/'+r['parent'],parent=r['parent'])
                            for r in s['references']],
            dispatches=[{k:d[k] for k in ('call_id','command','producer','log','line')}
                        for d in ds],
            required_routes=['exhaustive_ledger OR loaded_closure_and_exhaustive_envelope'],
            origin_review_required=True))
    need([s['source_id'] for s in result]==list('ABCDEFGH'),'fixed source order')
    return dict(schema=NAME+'.request',catalog_sha256=CATALOG_SHA,sources=result,
        inherited_reference_count=16,known_dispatch_count=10,
        coverage_complete=False,certified_fresh_candidates=0,P1_admitted=False)


def pointer(v,p):
    need(type(p) is str and (p=='' or p.startswith('/')),'JSON pointer')
    if p=='':return v
    for part in p[1:].split('/'):
        need(re.search(r'~(?![01])',part) is None,'pointer escape')
        key=part.replace('~1','/').replace('~0','~')
        if type(v) is list:
            need(re.fullmatch('0|[1-9][0-9]*',key) is not None,'pointer index')
            need(int(key)<len(v),'pointer index range');v=v[int(key)]
        else:
            need(type(v) is dict and key in v,'pointer member');v=v[key]
    return v


def citation(ref, reader, expected=None):
    fields(ref,'path sha256 pointer');text(ref['path']);hash_value(ref['sha256'])
    raw=reader.read(ref['path'],ref['sha256'],CAP)
    value=pointer(parse(raw),ref['pointer'])
    if expected is not None:need(canonical(value)==canonical(expected),'citation payload mismatch')
    return value


def load_roles(dispatch):
    roles={'producer','factory_1','factory_2','factory_3','factory_4','environment','wrappers'}
    if dispatch['producer'].endswith('preflight_pusht_d2p_p1_reconfiguration.py'):
        roles.add('p0_dependency')
    if dispatch['command'].startswith('bash '):roles.add('shell_launcher')
    return roles


def assertion(a, dispatch, report_sha, reader):
    fields(a,'call_id report_sha256 route attempts attempts_complete restarts_complete reset_evidence loaded')
    need(a['call_id']==dispatch['call_id'] and a['report_sha256']==report_sha,'run/report join')
    need(a['attempts_complete'] is True and a['restarts_complete'] is True,'attempt/restart declaration required')
    integer(a['attempts'])
    need(a['route'] in ('exhaustive_ledger','loaded_closure_and_exhaustive_envelope'),'route')
    reset=citation(a['reset_evidence'],reader)
    if a['route']=='exhaustive_ledger':
        fields(reset,'call_id attempts')
        need(reset['call_id']==a['call_id'] and type(reset['attempts']) is list and
             len(reset['attempts'])==a['attempts'],'ledger count/run')
        for i,row in enumerate(reset['attempts']):
            fields(row,'index reset_kind disposition identity_sha256')
            need(type(row['index']) is int and row['index']==i,'contiguous attempt index')
            need(row['reset_kind'] in ('random','explicit_state','goal_render','replay'),'reset kind')
            need(row['disposition'] in ('selected','unselected','rejected','failed','replay'),'disposition')
            # Null is retained for failed attempts without an observed root.
            if row['identity_sha256'] is not None:hash_value(row['identity_sha256'])
            else:need(row['disposition']=='failed','missing observed identity')
        need(a['loaded']==[],'ledger route does not silently assert loaded bytes')
    else:
        fields(reset,'call_id upper_bound derivation')
        need(reset['call_id']==a['call_id'],'envelope run')
        integer(reset['upper_bound']);text(reset['derivation'])
        need(a['attempts']<=reset['upper_bound'],'envelope upper bound')
        need(type(a['loaded']) is list,'loaded list')
        roles=[]
        for item in a['loaded']:
            fields(item,'role path sha256');text(item['role']);roles.append(item['role'])
            reader.read(item['path'],item['sha256'])
        need(len(set(roles))==len(roles) and set(roles)==load_roles(dispatch),'exact loaded closure roles')
    # Completeness and original-load provenance are declarations pending MAIN's
    # origin review, not theorems proved by a self-authored JSON file.


def validate(response, request, reader):
    fields(response,'schema request_sha256 sources historical_scope')
    need(response['schema']==NAME+'.response','response schema')
    need(response['request_sha256']==sha(canonical(request)),'request semantic SHA')
    need(type(response['sources']) is list and len(response['sources'])<=8,'source response cap')
    known={r['source_id']:r for r in request['sources']};seen=set();rows=[]
    for source in response['sources']:
        fields(source,'source_id report_sha256 runs unresolved_reason')
        sid=source['source_id'];need(type(sid) is str and sid in known and sid not in seen,'source identity')
        seen.add(sid);r=known[sid]
        need(source['report_sha256']==r['report']['sha256'],'original report SHA')
        expected={d['call_id']:d for d in r['dispatches']}
        need(type(source['runs']) is list and len(source['runs'])<=len(expected),'run count')
        calls=set()
        for run in source['runs']:
            fields(run,'assertion origin_citation')
            a=run['assertion'];need(type(a) is dict,'assertion object')
            cid=a.get('call_id');need(type(cid) is str and cid in expected and cid not in calls,'dispatch identity')
            calls.add(cid)
            # The exact assertion must appear at the explicitly SHA-bound origin
            # pointer; its historical provenance still requires external review.
            citation(run['origin_citation'],reader,a)
            assertion(a,expected[cid],source['report_sha256'],reader)
        complete=calls==set(expected)
        if complete:need(source['unresolved_reason'] is None,'complete submission reason must be null')
        else:text(source['unresolved_reason'])
        rows.append(dict(source_id=sid,submitted_dispatches=len(calls),
                         missing_dispatches=sorted(set(expected)-calls),
                         status='origin_review_required' if complete else 'partial'))
    scope=response['historical_scope']
    if scope is not None:
        fields(scope,'machines worktrees time_interval unsaved_failures origin_citation')
        for key in ('machines','worktrees'):
            need(type(scope[key]) is list and 0<len(scope[key])<=128,'scope list')
            for item in scope[key]:text(item)
        text(scope['time_interval']);text(scope['unsaved_failures'])
        payload={k:v for k,v in scope.items() if k!='origin_citation'}
        citation(scope['origin_citation'],reader,payload)
    return dict(schema=NAME+'.validation',status='syntax_and_bound_bytes_valid_not_authority',
        sources=rows,missing_sources=sorted(set(known)-seen),
        historical_scope_submitted=scope is not None,origin_review_required=True,
        reset_blockers_cleared=0,certified_fresh_candidates=0,P1_admitted=False,
        assets_read=len(reader.cache),asset_bytes=reader.used)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('request','validate'))
    p.add_argument('--response');p.add_argument('--response-sha256')
    args=p.parse_args();reader=Reader()
    req=requests(parse(reader.read(str(CATALOG),CATALOG_SHA,CAP)))
    if args.mode=='request':out=req
    else:
        need(args.response is not None and args.response_sha256 is not None,'explicit response path and SHA')
        response=parse(reader.read(args.response,args.response_sha256,CAP))
        out=validate(response,req,reader)
    raw=canonical(out);need(len(raw)<=CAP,'output cap');print(raw.decode())


if __name__=='__main__':main()
