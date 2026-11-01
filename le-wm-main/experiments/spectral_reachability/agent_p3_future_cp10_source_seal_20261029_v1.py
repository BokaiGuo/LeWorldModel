"""Durable candidate bridge handoff: inherited receipts, no production authorization."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p3_future_cp10_receipt_join_20261029_v1 as join

NAME='agent_p3_future_cp10_source_seal_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CAP=32*1024*1024
PINS={
    'agent_p3_future_cp10_receipt_join_20261029_v1': '7e08bdaf617fa7ebe6d56cfffc624149cdadbd7a56ab2caca20065e290b28172',
    'agent_p3_cp10_saved_source_closure_20261029_v1': 'a43e51257ec22ff3e69be2c2c0a9a8ef82d0494644dd4afe341c0a947c8afe03',
    'agent_p3_future_cp10_task_projection_20261029_v1': 'd2d2fb356b8187ff1d0f5f68e5f6a1d4bf910a3e9a0e50e3ab7f501dba2dc331',
    'agent_p3_same_time_capability_probe_cuda_20261029_v1': 'dac75c998e1a5ba5b1e8ab562cbaf02c04662844d884c3ae637ccbed3dbf25b4',
}
need=join.need


def encode(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def sha(path):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and not p.is_symlink() and p.is_file(), 'canonical closure file')
    need(p.stat().st_size<=join.files.MAX_FILE,'source bound')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(refs, manifest_ref):
    manifest=join.read(manifest_ref)
    need(type(manifest) is dict and set(manifest)=={'schema','producer_source','files'} and
         manifest['schema']==NAME+'.producer_source_manifest', 'source manifest schema')
    need(manifest['producer_source']==refs['source'], 'producer source manifest identity')
    listed=manifest['files'];need(type(listed) is list and 1<=len(listed)<=256,'bounded source closure')
    paths=[]
    for ref in listed:
        need(type(ref) is dict and set(ref)=={'path','sha256'}, 'source descriptor')
        need(sha(ref['path'])==ref['sha256'],'source bytes')
        paths.append(ref['path'])
    need(paths==sorted(set(paths)) and refs['source']['path'] in paths, 'sorted unique includes producer')
    need(next(x for x in listed if x['path']==refs['source']['path'])==refs['source'], 'exact producer source')
    prepare=join.read(refs['prepare'])
    need(prepare['producer_source_manifest']==manifest_ref,'prepare binds source manifest')
    receipt=join.read(refs['receipt'])
    need(receipt['bindings'][manifest_ref['path']]==manifest_ref['sha256'], 'receipt binds source manifest')
    for ref in listed:
        need(receipt['bindings'][ref['path']]==ref['sha256'], 'receipt binds each source')
    own={}
    for name,expected in PINS.items():
        path=ROOT/'experiments/spectral_reachability'/(name+'.py')
        need(sha(path)==expected,'reviewed component source SHA');own[str(path)]=expected
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        own[str(p)]=sha(p)
    candidate=join.join(refs)
    return dict(schema=NAME+'.seal',status='candidate_only_not_an_execution_authority',inputs=refs,
        source_manifest=manifest_ref,producer_source_files=listed,reviewed_component_sources=own,
        task_projection=candidate['task_projection'], source_class='declared_future_not_verified_fresh',
        source_authority_verified=False,producer_fullstrict_inherited_not_rebuilt=True,
        physical_fullstrict_inherited_not_rebuilt=True,source_closure_completeness_independently_verified=False,
        trusted_timestamp_proven=False,fresh_roots_certified=False,execution_authorized=False,scientific_admission=False)


def output_path(path):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and not p.is_symlink() and p.name.startswith(NAME+'_'), 'own canonical output')
    return p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('seal','verify'))
    for key in ('prepare','report','completion','receipt','source','manifest','output'):
        ap.add_argument('--'+key,required=True);ap.add_argument('--'+key+'-sha',required=key!='output')
    a=ap.parse_args(); refs={k:dict(path=getattr(a,k),sha256=getattr(a,k+'_sha')) for k in ('prepare','report','completion','receipt','source')}
    v=build(refs,dict(path=a.manifest,sha256=a.manifest_sha));body=encode(v);need(len(body)<=CAP,'seal cap');out=output_path(a.output)
    if a.command=='verify':
        need(a.output_sha is not None,'external seal SHA');need(join.files.data(out,a.output_sha,CAP)==body,'whole seal recomputation')
    else:
        need(a.output_sha is None,'no preexisting output hash')
        with out.open('xb') as f:
            need(f.write(body)==len(body),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(out.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(fd)
        finally:os.close(fd)
    print(json.dumps(dict(schema=NAME+'.'+a.command,sha256=hashlib.sha256(body).hexdigest(),execution_authorized=False)))


if __name__=='__main__':main()
