"""Persist the entire fixed-eight producer.consume result with a strict hash closure.

Saved validation only, never producer prepare/run, neural execution or physics.
NEW directory is an irreversible reservation: no deletion, overwrite or retry.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil

NAME='agent_CEM8_release_20261028_v1'
PRODUCER_NAME='agent_moving_CEM8_20261028_v1'
PARENT_NAME='capability_suffix_factorial_cuda_MAIN_20261027_v2'
ROOT=Path(__file__).resolve().parents[2]
PRODUCER_FILE=Path(__file__).with_name(PRODUCER_NAME+'.py')
PRODUCER_SHA='6dcb425c13ea8cade14bf23c97ecfe87b004b7804d155c5ef06282045e99c251'
PARENT_PREP=ROOT/'results/capability_suffix_factorial_cuda_MAIN_20261027_v2_prepare_MAIN01.json'
PARENT_SHA='330412c675844d7edff23ade45012cf4ef4920f64bd47a31a0c49e6872f673f5'
OWN_FILES=(Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))
ORDER=[(s,c,a) for s in (0,1) for c in ('off','on') for a in ('line_search_GN','full10_conditional_D2P')]
CAP,FLOOR,ERROR_RESERVE=4*1024**2,200*1024**2,16*1024
REPORT_CAP,PREP_CAP,PARENT_CAP,MAX_HASHES=32*1024**2,2*1024**2,8*1024**2,20000


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def hash_value(h):
    require(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'literal SHA256 required')


def literal(path):
    value=str(path);p=Path(value)
    require(p.is_absolute() and str(p)==value and '..' not in p.parts,'literal absolute path, no aliases')
    return value


def decode(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(v):raise ValueError('nonfinite JSON '+v)
    out=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad);canonical(out);return out


def syncdir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class LocalIO:
    """No unresolved globs, symlink aliases, directory traversal or model imports."""
    def regular(self,path):
        p=Path(literal(path));require(p.is_file() and not p.is_symlink() and p.resolve()==p,'regular nonsymlink input');return p
    def read(self,path,cap):
        p=self.regular(path);require(p.stat().st_size<=cap,'bounded saved JSON')
        with p.open('rb') as f:raw=f.read(cap+1)
        require(len(raw)<=cap,'bounded saved read');return raw
    def sha(self,path):
        p=self.regular(path);h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        return h.hexdigest()
    def names(self,path):
        p=Path(literal(path));require(p.is_dir() and p.resolve()==p,'literal input directory')
        names=set()
        for f in p.iterdir():
            require(f.is_file() and not f.is_symlink(),'flat regular artifact census');names.add(f.name)
            require(len(names)<=MAX_HASHES,'bounded artifact count')
        return names
    def claim(self,path):
        p=Path(literal(path));require(p.parent.is_dir() and p.parent.resolve()==p.parent and
            not p.exists() and not p.is_symlink(),'exclusive NEW directory with actual parent')
        require(shutil.disk_usage(p.parent).free>=FLOOR+CAP,'4MiB release reserve +200MiB disk floor')
        p.mkdir(mode=0o700);syncdir(p.parent)
    def write(self,path,raw):
        p=Path(literal(path));require(p.parent.is_dir() and p.parent.resolve()==p.parent and not p.is_symlink(),'actual reserved parent')
        used=sum(f.stat().st_size for f in p.parent.iterdir())
        require(used+len(raw)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(raw),'4MiB total release cap/disk floor')
        with p.open('xb') as f:
            require(f.write(raw)==len(raw),'full durable write');f.flush();os.fsync(f.fileno())
        syncdir(p.parent);return digest(raw)


def arguments(directory,report_sha,prepared,prepare_sha,output=None):
    hash_value(report_sha);hash_value(prepare_sha)
    d=literal(directory);p=literal(prepared)
    if output is not None:
        o=Path(literal(output));require(not o.is_relative_to(Path(d)),'release must be outside original producer directory')
    return dict(directory=d,report_sha=report_sha,prepared=p,prepare_sha=prepare_sha)


def snapshot(args,io):
    """Own118 + inherited complete factorial inputs + all actual output artifacts.

    Hash files by streaming; do not copy report, prepare, tensors or weights.
    Parent full binding table is explicit metadata, not recursive discovery.
    """
    args=arguments(**args);directory=Path(args['directory']);rp=str(directory/'report.json')
    require(io.sha(PRODUCER_FILE)==PRODUCER_SHA,'frozen producer source SHA')
    raw=io.read(rp,REPORT_CAP);require(digest(raw)==args['report_sha'],'external actual report SHA')
    r=decode(raw)
    require(r['schema']==PRODUCER_NAME+'.report' and r['prepared']==args['prepared'] and r['prepare_sha256']==args['prepare_sha'],
        'actual producer report/prepare ABI binding')
    raw=io.read(args['prepared'],PREP_CAP);require(digest(raw)==args['prepare_sha'],'external actual prepare SHA')
    p=decode(raw);require(p['schema']==PRODUCER_NAME+'.prepare' and type(p['bindings']) is dict,'own source-bound prepare')
    require(p['bindings'].get(str(PRODUCER_FILE))==PRODUCER_SHA and p['bindings'].get(str(PARENT_PREP))==PARENT_SHA,
        'own producer and inherited prepare frozen inputs')
    raw=io.read(PARENT_PREP,PARENT_CAP);require(digest(raw)==PARENT_SHA,'frozen inherited factorial prepare SHA')
    parent=decode(raw);require(parent['schema']==PARENT_NAME+'.prepare' and type(parent['bindings']) is dict,'full inherited binding table')
    hashes={}
    def add(path,h):
        path=literal(path);hash_value(h)
        require(path not in hashes or hashes[path]==h,'path/hash conflict fails closed')
        hashes[path]=h;require(len(hashes)<=MAX_HASHES,'bounded complete input hash inventory')
    for table in (p['bindings'],parent['bindings']):
        for path,h in table.items():add(path,h)
    add(PRODUCER_FILE,PRODUCER_SHA);add(PARENT_PREP,PARENT_SHA)
    add(args['prepared'],args['prepare_sha']);add(rp,args['report_sha'])
    artifacts=r['artifact_sha256'];require(type(artifacts) is dict and len(artifacts)<=MAX_HASHES,'explicit actual artifact seals')
    for name,h in artifacts.items():
        require(type(name) is str and Path(name).name==name and name not in ('','.','..','report.json'),'flat nonself artifact')
        add(directory/name,h)
    require(io.names(directory)==set(artifacts)|{'report.json'},'complete actual artifact census, no extra/absent output')
    binding=decode(io.read(directory/'binding.json',64*1024))
    require(binding==dict(prepared=args['prepared'],prepare_sha256=args['prepare_sha']),'actual execution binding')
    for path in OWN_FILES:add(path,io.sha(path))
    for path,h in hashes.items():require(io.sha(path)==h,'source/input/artifact SHA mismatch: '+path)
    return hashes


def load_producer():
    return importlib.import_module('experiments.spectral_reachability.'+PRODUCER_NAME)


def checked_consume(args,consumer):
    # The CLI has no alternate-provider option. Default is the true frozen function.
    result=(load_producer().consume if consumer is None else consumer)(args['directory'],args['report_sha'])
    raw=canonical(result);require(len(raw)<CAP-ERROR_RESERVE,'entire untruncated producer result fits cap')
    require(result['schema']==PRODUCER_NAME+'.consume' and result['status']=='teacher8_CEM_commands_verified_not_admitted'
        and result['errors']==[] and len(result['plans'])==8 and
        all(result[k] is False for k in ('P1_admission','P3_admission','equal_total_compute')) and
        type(result['new_neural_calls']) is int and type(result['new_physics_calls']) is int and
        result['new_neural_calls']==result['new_physics_calls']==0,'complete saved consume result, never prepare/CUDA admission')
    for i,(plan,key) in enumerate(zip(result['plans'],ORDER,strict=True)):
        require(type(plan['root_slot']) is int and type(plan['source_slot']) is int and type(plan['anchor']) is int and
            (plan['root_slot'],plan['condition'],plan['prefix_arm'])==key and plan['source_slot']==key[0] and
            plan['anchor']==i and plan['whole_run_report_sha256']==args['report_sha'],'all eight actual teacher anchors/report seals')
    return deepcopy(result)


def make_receipt(args,request_sha,hashes,result):
    return dict(schema=NAME+'.receipt',status='strict_CEM8_release_persisted_not_admitted',arguments=args,
        producer_source=dict(file=str(PRODUCER_FILE),sha256=PRODUCER_SHA),request_sha256=request_sha,
        input_sha256=hashes,input_count=len(hashes),producer_result=result,producer_result_sha256=digest(canonical(result)),
        verification_kind='actual producer.consume on saved artifacts; not new CUDA execution',
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,P3_admission=False,equal_total_compute=False)


def run(directory,report_sha,prepared,prepare_sha,output,*,io=None,consumer=None):
    args=arguments(directory,report_sha,prepared,prepare_sha,output);output=literal(output)
    io=LocalIO() if io is None else io
    io.claim(output)  # Atomic directory claim precedes expensive consume; never removed.
    stage='request'
    try:
        request=dict(schema=NAME+'.request',arguments=args,producer_source_sha256=PRODUCER_SHA)
        request_sha=io.write(Path(output)/'request.json',canonical(request))
        stage='preconsume_inputs';before=snapshot(args,io)
        stage='true_producer_consume';result=checked_consume(args,consumer)
        stage='postconsume_inputs';after=snapshot(args,io);require(before==after,'source/input closure changed during consume')
        receipt=make_receipt(args,request_sha,before,result)
        raw=canonical(receipt)
        require(len(raw)+len(canonical(request))<=CAP-ERROR_RESERVE,'entire receipt plus request fits4MiB; never truncate')
        stage='durable_receipt';path=str(Path(output)/'receipt.json');h=io.write(path,raw)
        require(io.sha(path)==h and io.names(output)=={'request.json','receipt.json'},'durable receipt/census readback')
        return dict(status=receipt['status'],file=path,sha256=h,plans=len(result['plans']),
            producer_status=result['status'],input_count=len(before),new_neural_calls=0,new_physics_calls=0,P1_admission=False,P3_admission=False)
    except BaseException as exc:
        try:
            io.write(Path(output)/'error.json',canonical(dict(schema=NAME+'.error',status='failed_no_release',stage=stage,
                type=type(exc).__name__,message=str(exc)[:2048],P1_admission=False,P3_admission=False)))
        except BaseException as log_error:exc.add_note('failure receipt write: '+str(log_error)[:1000])
        raise


def verify(output,receipt_sha,*,io=None,consumer=None):
    hash_value(receipt_sha);output=literal(output);io=LocalIO() if io is None else io
    require(io.names(output)=={'request.json','receipt.json'},'complete successful release only, no failed/extra files')
    raw=io.read(Path(output)/'receipt.json',CAP);require(digest(raw)==receipt_sha,'external actual receipt SHA')
    receipt=decode(raw);require(receipt['schema']==NAME+'.receipt' and receipt['status']=='strict_CEM8_release_persisted_not_admitted'
        and receipt['P1_admission'] is receipt['P3_admission'] is receipt['equal_total_compute'] is False,'unadmitted strict receipt schema')
    raw_request=io.read(Path(output)/'request.json',64*1024);require(digest(raw_request)==receipt['request_sha256'],'original durable request SHA')
    request=decode(raw_request);args=arguments(**receipt['arguments'],output=output)
    require(request==dict(schema=NAME+'.request',arguments=args,producer_source_sha256=PRODUCER_SHA) and
        receipt['producer_source']==dict(file=str(PRODUCER_FILE),sha256=PRODUCER_SHA),'same actual invocation and frozen producer')
    require(receipt['input_count']==len(receipt['input_sha256']) and snapshot(args,io)==receipt['input_sha256'],'all saved input/artifact bindings unchanged')
    result=checked_consume(args,consumer)
    require(result==receipt['producer_result'] and digest(canonical(result))==receipt['producer_result_sha256'],
        'entire actual consume return equals persistent receipt')
    require(canonical(receipt)==canonical(make_receipt(args,digest(raw_request),receipt['input_sha256'],result)),
        'entire release envelope exactly reconstructed, not fabricated execution counters')
    require(snapshot(args,io)==receipt['input_sha256'] and io.sha(Path(output)/'receipt.json')==receipt_sha
        and io.read(Path(output)/'request.json',64*1024)==raw_request and io.names(output)=={'request.json','receipt.json'},'final revalidation closure drift')
    return dict(status='release_reverified_not_admitted',sha256=receipt_sha,plans=8,new_neural_calls=0,new_physics_calls=0,P1_admission=False,P3_admission=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('release')
    for k in ('input','report-sha','prepared','prepare-sha','output'):q.add_argument('--'+k,required=True)
    v=sub.add_parser('verify');v.add_argument('--output',required=True);v.add_argument('--receipt-sha',required=True)
    a=p.parse_args(argv)
    value=verify(a.output,a.receipt_sha) if a.command=='verify' else run(a.input,a.report_sha,a.prepared,a.prepare_sha,a.output)
    print(canonical(value).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())
