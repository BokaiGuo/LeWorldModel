"""Bounded read-only HF metadata route; mirror claims never certify a scaler."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import time
import urllib.error
import urllib.parse
import urllib.request

TAG='agent_hilewm_normalizer_asset_route_20261029_v2'
ROOT=Path(__file__).resolve().parents[2]
REPO='quentinll/lewm-pusht'
HOSTS=('hf-mirror.com','huggingface.co')
BODY_CAP=1024*1024
OUTPUT_CAP=16*1024*1024
FLOOR=200*1024*1024
REQUEST_CAP=6
WALL=150
V1=ROOT/'innerdocs/agent_hilewm_normalizer_asset_route_20261029_v1.md'
V1_SHA='5eff899edacb0a4925d54b4f420a32e0e8b556e4ef94852733c2bca203547619'


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(body):return hashlib.sha256(body).hexdigest()
def encode(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def utc():return datetime.now(timezone.utc).isoformat()


def parse(body):
    need(len(body)<=BODY_CAP,'JSON cap')
    def pairs(xs):
        out={}
        for k,v in xs:need(k not in out,'duplicate metadata key');out[k]=v
        return out
    return json.loads(body,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))


def allowed(url,host):
    u=urllib.parse.urlsplit(url)
    need(host in HOSTS and u.scheme=='https' and u.netloc==host and not u.fragment and not u.username,'HTTPS fixed host/no credentials')
    prefix='/api/'
    need(u.path in (prefix+'datasets/'+REPO,prefix+'models/'+REPO) or
        re.fullmatch(r'/api/(datasets|models)/'+REPO+r'/tree/[0-9a-f]{40}',u.path),'API metadata only; no resolve/payload paths')
    need(len(url)<=4096,'URL bound')
    return url


def member_name(name):
    need(isinstance(name,str) and 0<len(name)<=1024 and not name.startswith('/') and '\\' not in name
        and all(part not in ('','.','..') for part in name.split('/')) and all(ord(c)>=32 for c in name),'safe declared repo member')
    return name


def tree_summary(rows):
    need(isinstance(rows,list) and len(rows)<=1000,'bounded tree page list')
    out=[]
    for row in rows:
        need(isinstance(row,dict) and row.get('type') in ('file','directory'),'tree entry shape')
        name=member_name(row['path'])
        if row['type']=='file':
            size=row.get('size');need(type(size) is int and size>=0,'declared file size, unknown not zero')
            out.append(dict(path=name,size=size,oid=row.get('oid'),lfs=row.get('lfs'),
                metadata_candidate=bool(re.search(r'dataset_info|process|normaliz|scaler|statistic|stats|readme|config',name,re.I)),
                authority='unverified_provider_claim'))
    need(len({r['path'] for r in out})==len(out),'duplicate tree paths')
    return out


def next_page(link,current,host):
    if not link:return None
    matches=re.findall(r'<([^>]+)>;\s*rel="next"',link)
    need(len(matches)<=1,'unambiguous next link')
    if not matches:return None
    u=urllib.parse.urljoin(current,matches[0]);allowed(u,host)
    need(urllib.parse.urlsplit(u).path==urllib.parse.urlsplit(current).path,'same pinned revision pagination')
    return u


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def fetch(url,host):
    allowed(url,host);request=urllib.request.Request(url,headers={'User-Agent':TAG,'Accept':'application/json','Accept-Encoding':'identity'})
    row=dict(url=url,method='GET',started_utc=utc(),status=None,error=None,headers={});body=b''
    try:
        opener=urllib.request.build_opener(NoRedirect)
        try:response=opener.open(request,timeout=20)
        except urllib.error.HTTPError as error:response=error
        with response as r:
            row['status']=r.code
            row['headers']={k.lower():v for k,v in r.headers.items() if k.lower() in
                ('content-length','content-type','content-encoding','etag','last-modified','link','date','x-repo-commit')}
            body=r.read(BODY_CAP+1)
            if len(body)>BODY_CAP:
                body=body[:BODY_CAP];row['error']='body exceeds cap; saved truncated prefix, not complete metadata'
            declared=row['headers'].get('content-length')
            if declared is not None and row['error'] is None:
                need(int(declared)==len(body),'HTTP length mismatch')
            need(row['headers'].get('content-encoding','identity').lower()=='identity','no implicit decompression')
    except Exception as e:row['error']=type(e).__name__+': '+str(e)[:1500]
    row.update(finished_utc=utc(),body_bytes=len(body),body_sha256=sha(body))
    return row,body


class Output:
    def __init__(self,path):
        self.path=Path(path)
        need(self.path.is_absolute() and self.path.resolve()==self.path and self.path.parent==ROOT/'results'
            and self.path.name.startswith(TAG+'_'),'new exclusive own-stem results child')
        need(shutil.disk_usage(self.path.parent).free>=FLOOR+OUTPUT_CAP,'output cap+floor');self.path.mkdir();self.used=0
    def put(self,name,body):
        need(PurePosixPath(name).name==name and len(name)>0 and self.used+len(body)<=OUTPUT_CAP and
            shutil.disk_usage(self.path).free>=FLOOR+len(body),'output path/cap/floor')
        with (self.path/name).open('xb') as f:need(f.write(body)==len(body),'full exclusive write')
        self.used+=len(body)


def collect(output,host,*,network_authorized=False):
    need(network_authorized is True,'explicit read-only network authorization before I/O')
    need(host in HOSTS,'fixed official/mirror host')
    need(sha(V1.read_bytes())==V1_SHA,'frozen source route v1')
    store=Output(output);start=time.monotonic()
    report=dict(schema=TAG+'.report',host=host,provider='official' if host=='huggingface.co' else 'mirror',
        source_sha256=sha(Path(__file__).read_bytes()),v1_sha256=V1_SHA,started_utc=utc(),requests=[],repositories=[],errors=[],
        raw_inverse_authority=None,physical_release=False,neural_calls=0,physics_calls=0,stats_fitted=False,
        author_frozen_scaler_obtained=False,official_equivalence=False,exhaustive=False)
    def request(url):
        need(len(report['requests'])<REQUEST_CAP and time.monotonic()-start<WALL,'request/wall budget')
        row,body=fetch(url,host);row['body_file']=f'response_{len(report["requests"]):02d}.bin'
        store.put(row['body_file'],body);report['requests'].append(row)
        need(row['status']==200 and row['error'] is None,'metadata unavailable; preserve HTTP/error, no bypass/retry')
        return row,parse(body)
    def deadline(sig,frame):raise TimeoutError('whole metadata round wall limit')
    old_handler=signal.signal(signal.SIGALRM,deadline);signal.alarm(WALL)
    try:
        for kind in ('datasets','models'):
            entry=dict(kind=kind,repo=REPO,candidate_revision=None,revision_authority='unknown',files=[],
                metadata_candidates=[],tree_complete=False,error=None)
            report['repositories'].append(entry)
            try:
                row,data=request(f'https://{host}/api/{kind}/{REPO}?blobs=true')
                need(isinstance(data,dict) and data.get('id')==REPO and
                    isinstance(data.get('sha'),str) and re.fullmatch('[0-9a-f]{40}',data['sha']),'actual declared repo/revision')
                entry.update(candidate_revision=data['sha'],revision_authority='provider_claim_not_scaler_authentication',
                    cardData=data.get('cardData'),siblings=data.get('siblings'))
                url=f'https://{host}/api/{kind}/{REPO}/tree/{data["sha"]}?recursive=true&expand=false&limit=1000'
                seen=set()
                while url:
                    need(url not in seen,'pagination cycle');seen.add(url)
                    row,data=request(url);entry['files'].extend(tree_summary(data))
                    need(len({r['path'] for r in entry['files']})==len(entry['files']),'duplicate cross-page file')
                    url=next_page(row['headers'].get('link'),url,host)
                entry['tree_complete']=True
            except Exception as e:entry['error']=type(e).__name__+': '+str(e)[:1500]
            entry['metadata_candidates']=[r for r in entry['files'] if r['metadata_candidate']]
    except Exception as e:report['errors'].append(type(e).__name__+': '+str(e)[:1500])
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        report.update(finished_utc=utc(),elapsed_seconds=time.monotonic()-start,
            status='metadata_recorded_not_scaler_certified' if any(r['tree_complete'] for r in report['repositories']) else 'metadata_blocked_unknown',
            saved_response_bytes=store.used)
        store.put('report.json',encode(report))
    return report,dict(report=str(store.path/'report.json'),sha256=sha((store.path/'report.json').read_bytes()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',choices=HOSTS,default='hf-mirror.com');parser.add_argument('--output',required=True)
    parser.add_argument('--network-authorized',action='store_true');args=parser.parse_args(argv)
    report,result=collect(args.output,args.host,network_authorized=args.network_authorized)
    print(encode(dict(result,status=report['status'],raw_inverse_authority=None)).decode(),end='')
    return 0 if any(r['tree_complete'] for r in report['repositories']) else 2


if __name__=='__main__':raise SystemExit(main())
