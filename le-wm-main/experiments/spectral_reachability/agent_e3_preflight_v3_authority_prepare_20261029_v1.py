"""Create v3 preflight authority from an existing identity authority, saved-only."""
import argparse
import hashlib
import os
from pathlib import Path
from experiments.spectral_reachability import agent_e3_preflight_v2_authority_prepare_20261029_v1 as previous
from experiments.spectral_reachability import agent_e3_encoder_cuda_preflight_20261029_v3 as v3

host = v3.host
need = host.need


def build(source, sha):
    result = previous.build(source, sha)
    for module in (v3, v3.chunked):
        path=str(Path(module.__file__).resolve()); digest=host.launch.host.sha(path)
        existing=result['additional_sources'].get(path)
        need(existing is None or existing==digest,'conflicting v3 source pin')
        result['additional_sources'][path]=digest
    binding=host.authority(source,sha)
    v3.sources(dict(binding,additional_sources=result['additional_sources']))
    return result


def verify(source, sha, output, output_sha):
    need(host.launch.host.read(output,output_sha)==build(source,sha),'v3 authority reconstruction')
    v3.sources(host.authority(output,output_sha))
    return dict(verified=True,sha256=output_sha,scientific_admission=False,full_upstream_strict_rebuilt=False)


def prepare(source, sha, output):
    value=build(source,sha); p=Path(output)
    need(p.is_absolute() and str(p.resolve())==str(p) and p.parent.is_dir() and not p.exists()
         and not p.is_symlink(),'new canonical output')
    data=host.launch.host.raw(value);need(len(data)<=host.launch.host.META,'metadata bound')
    with p.open('xb') as f:
        need(f.write(data)==len(data),'complete authority write');f.flush();os.fsync(f.fileno())
    host.launch.pub.base.syncdir(p.parent); digest=hashlib.sha256(data).hexdigest()
    verify(source,sha,str(p),digest)
    return dict(output=str(p),sha256=digest,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','verify'):
        x=s.add_parser(name)
        for arg in ('source-authority','source-sha256','output'): x.add_argument('--'+arg,required=True)
        if name=='verify': x.add_argument('--output-sha256',required=True)
    a=p.parse_args()
    result=prepare(a.source_authority,a.source_sha256,a.output) if a.command=='prepare' else verify(
        a.source_authority,a.source_sha256,a.output,a.output_sha256)
    print(host.launch.host.raw(result).decode())


if __name__=='__main__': main()
