"""Observed B4/B8 direct v2: storage-only lossless gzip extension of sealed v1.

No event/array filtering, downcasting, trajectory thinning, or resource changes.
MAIN owns actual CUDA. Imports, prepare and verify never execute neural/physics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import os
from pathlib import Path
import shutil
import zlib

from experiments.spectral_reachability import p1_observed_direct_cuda_20261027_v1 as original

NAME='p1_observed_direct_cuda_20261027_v2'
ROOT=original.ROOT
screen,observed,io=original.screen,original.observed,original.io
require,sha,read,digest=original.require,original.sha,original.read,original.digest
FLOOR=200*1024**2
CAP=24*1024**2
FINAL=512*1024
METHODS=original.METHODS
TOTAL=deepcopy(original.TOTAL)
STORAGE=dict(codec='gzip/zlib level6; original canonical JSON bytes unchanged',
    journal='Z_SYNC_FLUSH + flush + fsync EVERY original event; Z_FINISH footer',
    reports='entire original objects gzip, no field/precision/array removal',
    output_cap_bytes=CAP,disk_floor_bytes=FLOOR,finalization_reserve_bytes=FINAL,
    cap_basis='predeclared 24MiB; full four-arm dense B8 mock must fit with >=2x margin',
    compression_ratio_is_not_a_universal_guarantee=True,
    on_overflow='fail closed with partial receipts, never reduce resources or raise cap')


def used_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())


class Store:
    """One shared quota; partial gzip contains all previously fsynced events."""
    def __init__(self,output):
        self.output=Path(output);self.streams={};self.broken=set()
        require(not self.output.exists(),'exclusive NEW output')
        require(shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'224MiB required:24MiB output +200MiB reserve')
        self.output.mkdir()

    def write(self,path,raw,*,append=False,final=False):
        path=Path(path)
        require(path.is_relative_to(self.output) and not path.is_symlink(),'local exclusive output')
        require(used_bytes(self.output)+len(raw)<=CAP-(0 if final else FINAL),'24MiB complete output cap/finalization reserve')
        require(shutil.disk_usage(self.output).free>=FLOOR+len(raw),'200MiB write floor')
        with path.open('ab' if append else 'xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())

    def put(self,name,value,*,final=False):
        raw=(observed.source.canonical(value)+'\n').encode()
        self.write(self.output/name,gzip.compress(raw,compresslevel=6,mtime=0) if name.endswith('.gz') else raw,final=final)

    def journal(self,directory):
        directory=self.output/directory
        require(directory.parent==self.output,'flat component directory')
        directory.mkdir(exist_ok=False)
        path=directory/'direct_journal.jsonl.gz'
        comp=zlib.compressobj(6,zlib.DEFLATED,31)
        self.write(path,b'');self.streams[path]=comp
        def emit(event):
            require(path not in self.broken,'failed stream cannot resume')
            raw=(observed.source.canonical(event)+'\n').encode()
            try:self.write(path,comp.compress(raw)+comp.flush(zlib.Z_SYNC_FLUSH),append=True)
            except BaseException:
                self.broken.add(path)
                raise
        return emit,path

    def finish(self,path):
        comp=self.streams.pop(path)
        require(path not in self.broken,'partial stream after prior write failure; no false footer')
        self.write(path,comp.flush(zlib.Z_FINISH),append=True,final=True)

    def close(self):
        errors=[]
        for path in list(self.streams):
            try:self.finish(path)
            except BaseException as exc:errors.append(io.error('journal_finalize',exc))
        return errors


def read_gzip(path):
    """Strict complete gzip object; partials must use explicit recover_events."""
    return io.io.decode(gzip.decompress(Path(path).read_bytes()))


def recover_events(path):
    """Return only complete JSON lines of durable prefix, with EOF flag.

    Not a completed experiment verifier. CRC/data corruption still raises.
    """
    decoder=zlib.decompressobj(31)
    raw=decoder.decompress(Path(path).read_bytes())
    lines=raw.split(b'\n')
    return [io.io.decode(line) for line in lines[:-1]],decoder.eof


def build(**inputs):
    p=original.build(**inputs)
    p['schema']=NAME+'.prepare'
    p['output_cap_bytes']=CAP
    p['storage']=deepcopy(STORAGE)
    for path in [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:
        p['source_sha256'][str(path)]=sha(path)
    # The original driver, tests/doc, core and native encoder remain bound.
    require(p['methods']==[original.config(m) for m in METHODS] and p['direct_resources']==TOTAL,
            'v1 complete scientific/resource protocol unchanged')
    return p


def prepare(output,**inputs):
    p=build(**inputs);path=Path(output)
    raw=(observed.source.canonical(p)+'\n').encode()
    require(len(raw)<256*1024,'small hash-only prepare')
    require(shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'prepare-only 200MiB floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return p


def verify(prepared,expected_sha):
    require(sha(prepared)==expected_sha,'external v2 prepare SHA')
    p=read(prepared)
    require(p['schema']==NAME+'.prepare' and p==build(**p['inputs']),'full v2 protocol/source reconstruction')
    return p


def run(prepared,expected_sha,output):
    p=verify(prepared,expected_sha);store=Store(output);output=store.output
    errors=[];model=None;before=after=None;host=None;tasks=[];components=[];direct=[];image_events=[];source_ok=False;torch=None
    def fail(phase,exc):errors.append(dict(phase=phase,type=type(exc).__name__,message=str(exc)))
    image_comp=zlib.compressobj(6,zlib.DEFLATED,31)
    image_path=output/'image.events.jsonl.gz';store.write(image_path,b'');store.streams[image_path]=image_comp
    def image_event(event):
        image_events.append(deepcopy(event))
        require(image_path not in store.broken,'image journal already failed')
        raw=(observed.source.canonical(event)+'\n').encode()
        try:store.write(image_path,image_comp.compress(raw)+image_comp.flush(zlib.Z_SYNC_FLUSH),append=True)
        except BaseException:
            store.broken.add(image_path)
            raise
    try:
        data=original.material(p['inputs']['pack'],p['inputs']['source_sha'],p['inputs']['source_prepare'],p['inputs']['source_prepare_sha'])
        store.put('source_audit.json',data['summary'])
        import torch as runtime
        torch=runtime
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=False
        model=load_compatible_pretrained(p['model']['policy'],device=device).eval().requires_grad_(False)
        io.frozen(model);before=io.tensor_hash(model)
        encoder=screen.native_image_encoder(model,device,original.native_transform(),image_event)
        weights=next(h for path,h in p['model']['bindings'].items() if Path(path).name=='weights.pt')
        model_config=next(h for path,h in p['model']['bindings'].items() if Path(path).name=='config.json')
        base=original.template(data,digest(p),weights,model_config)
        for index,packet in enumerate(data['packets']):
            encoded=original.encode_observed(packet,base,encoder);tasks.append(encoded)
            store.put(f'task_{index}.json.gz',encoded)
        store.finish(image_path)
        with sdpa_kernel(SDPBackend.MATH):
            for index,encoded in enumerate(tasks):
                for c in p['methods']:
                    name=f'root_{index}_{c["method"]}'
                    sink,path=store.journal(name)
                    result=screen.run_direct(encoded['task'],c,screen.NativeCuda(model,device,encoded['task']),sink)
                    components.append(result)
                    row=dict(root_sha256=encoded['root_sha256'],method=c['method'],status=result['status'],error=result['error'])
                    direct.append(row)
                    store.put(name+'/report.json.gz',result)
                    store.finish(path)
                    row.update(report_path=str((output/name/'report.json.gz').resolve()),report_sha256=sha(output/name/'report.json.gz'),
                               journal_path=str(path.resolve()),journal_sha256=sha(path))
                    require(result['status']=='direct_component_completed_not_admitted' and result['error'] is None and not result['secondary_errors'],
                            'direct component failure; no retries')
    except BaseException as exc:fail('execution',exc)
    finally:
        if torch is not None:
            try:torch.cuda.synchronize()
            except BaseException as exc:fail('final_synchronize',exc)
        if model is not None:
            try:io.frozen(model);after=io.tensor_hash(model)
            except BaseException as exc:fail('final_model',exc)
        try:verify(prepared,expected_sha);source_ok=True
        except BaseException as exc:fail('final_source',exc)
        errors.extend(store.close())
        counts={}
        for resource in TOTAL:
            events=[e for c in components for e in c['extended_journal'] if e['resource']==resource]
            counts[resource]=dict(attempted=sum(e['samples'] for e in events),completed=sum(e['samples'] for e in events if e['completed']))
        image_counts={kind:dict(attempted=sum(e.get('samples',0) for e in image_events if e['kind']=='native_image_attempted' and e['resource']==kind),
            completed=sum(e['counts'][kind] for e in image_events if e['kind']=='native_image_completed')) for kind in ('encoder','projector')}
        budget_ok=(counts=={k:dict(attempted=n,completed=n) for k,n in TOTAL.items()} and
                   image_counts=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')})
        complete=not errors and budget_ok and len(tasks)==2 and len(components)==4 and source_ok and before is not None and before==after
        report=dict(schema=NAME+'.report',status='observed_direct_calibration_completed' if complete else 'partial_or_failed',
            errors=errors,host=host,prepare_sha256=expected_sha,source_verified=source_ok,source_audit=p['source_audit'],
            actual_source_acquisition_resources=p['source_audit']['actual_acquisition_counts'],source_acquisition_previously_paid=True,
            new_physical_steps=0,training=False,actual_direct_resources=counts,actual_image_resources=image_counts,
            exact_budget_verified=budget_ok,direct=direct,encoded_tasks=len(tasks),model_hash_before=before,
            model_hash_after=after,model_unchanged=before is not None and before==after,independent_sample_increment=0,
            global_freshness='unknown',P1_admission=False,model_task_eligible=False,gaps=p['gaps'],storage=STORAGE)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:
            fail('final_report',exc);report['status']='partial_or_failed'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('sourcepack','source-prepare','prepared','output'):parser.add_argument('--'+name,type=Path)
    for name in ('source-sha','source-prepare-sha','prepare-sha'):parser.add_argument('--'+name)
    a=parser.parse_args()
    if a.command=='prepare':
        prepare(a.output,pack=a.sourcepack,source_sha=a.source_sha,source_prepare=a.source_prepare,source_prepare_sha=a.source_prepare_sha)
        print(sha(a.output))
    elif a.command=='verify':print(verify(a.prepared,a.prepare_sha)['schema'])
    else:
        r=run(a.prepared,a.prepare_sha,a.output);print(observed.source.canonical(r))
        if r['status']!='observed_direct_calibration_completed':raise SystemExit(2)


if __name__=='__main__':main()
