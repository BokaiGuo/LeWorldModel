"""Fail-closed saved transport for future cp10; no authorized CUDA adapter yet."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p3_dynamic_cp10_physical_inputs_20261029_v1 as physical
from experiments.spectral_reachability import agent_p3_future_cp10_native_transcript_20261029_v1 as native

NAME='agent_p3_dynamic_cp10_cuda_transport_20261029_v1'
CAP=64*1024**2
FINAL=8*1024**2
need=physical.need


def canonical(value): return (json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()


class Store:
    """Create-only flat durable transport; failure reserve is never refunded."""
    def __init__(self,path):
        self.path=Path(path)
        need(self.path.is_absolute() and self.path.resolve()==self.path and not self.path.is_symlink()
             and self.path.name.startswith(NAME+'_'), 'own absolute new directory')
        self.path.mkdir();self.bytes=0;self.hashes={}
        self.sync_dir()

    def sync_dir(self):
        fd=os.open(self.path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(fd)
        finally:os.close(fd)

    def write(self,name,body,final=False):
        need(type(name) is str and Path(name).name==name and name not in self.hashes,'unique flat name')
        need(type(body) is bytes and self.bytes+len(body)<=CAP-(0 if final else FINAL),'output cap/reserve')
        path=self.path/name
        with path.open('xb') as f:
            need(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())
        self.bytes+=len(body);h=hashlib.sha256(body).hexdigest();self.hashes[name]=h;self.sync_dir()
        return h


def verify_transcript(prepared,report,events,load):
    """Semantic validator only; caller must authenticate every byte before load."""
    return native.validate_native(prepared,report,events,load)


def verify_saved(prepared,report,directory,expected_hashes):
    """Authenticate an explicit complete file census before validating native IO.

    Report, prepared object and census are caller-supplied and not authorities;
    an outer complete producer receipt must bind them before use.
    """
    root=Path(directory)
    need(root.is_absolute() and root.resolve()==root and root.is_dir() and not root.is_symlink(),'canonical directory')
    need(type(expected_hashes) is dict and len(expected_hashes)<=100,'bounded expected census')
    entries=list(root.iterdir())
    need(len(entries)==len(expected_hashes) and sum(x.stat().st_size for x in entries)<=CAP and
         all(x.is_file() and not x.is_symlink() for x in entries),'all files regular and bounded')
    need({x.name for x in entries}==set(expected_hashes),'no missing or orphan files')
    saved={}
    for x in entries:
        body=x.read_bytes();need(hashlib.sha256(body).hexdigest()==expected_hashes[x.name],'file SHA '+x.name)
        saved[x.name]=body
    need(report['event_count']==52,'complete fixed native events')
    seal=native.native.controls.native.decode(saved['native_seal.json'])
    need(seal=={k:report[k] for k in ('rows','modules','guards','blocked','errors')} and
         hashlib.sha256(saved['native_seal.json']).hexdigest()==report['native_seal_sha256'], 'native seal matches report')
    events=[native.native.controls.native.decode(saved[f'event_{i:04}.json']) for i in range(52)]
    load=lambda desc:native.native.bridge.rgb.load_array(desc,lambda name:saved[name])
    result=verify_transcript(prepared,report,events,load)
    return dict(schema=NAME+'.saved_transcript',native_validation=result,files=len(saved),
                producer_receipt_verified=False,physical_authority_verified=False,full_saved_consumer=False,
                execution_authorized=False,scientific_admission=False)


def run_cuda(specs,groups,output):
    """Do not import torch or create output while independent authority is absent."""
    source=physical.assemble(specs,groups,'future_candidate')
    need(source['source_authenticity_independently_verified'] is True and source['execution_authorized'] is True,
         'BLOCKED: independent future authority and reviewed CUDA backend unavailable')
    # Unreachable until a separately reviewed authorization mechanism is implemented.
    raise ValueError('BLOCKED: future CUDA model backend and full consumer not implemented')


def record_fixture(path,items,fail_at=None):
    """Test-only bytes transport. It does not simulate CUDA or prove any call."""
    store=Store(path)
    try:
        for i,(name,body) in enumerate(items):
            if i==fail_at: raise RuntimeError('fixture failure')
            store.write(name,body)
        return dict(status='fixture_only',hashes=store.hashes,execution_authorized=False)
    except BaseException as exc:
        store.write('partial.json',canonical(dict(status='partial_fixture',error=type(exc).__name__,
            persisted_hashes=dict(store.hashes),execution_authorized=False)),final=True)
        raise


def main():
    ap=argparse.ArgumentParser(description='No runnable production CUDA interface exists yet.')
    ap.add_argument('command',choices=('status',));ap.parse_args()
    print(json.dumps(dict(schema=NAME+'.status',status='blocked_no_future_authority_or_cuda_backend',
                          execution_authorized=False,full_saved_consumer=False)))


if __name__=='__main__':main()
