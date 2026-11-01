"""Fixed roster cell04 different-goal authority preparation; no CUDA/physics."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as identity
from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as startup

NAME='agent_e3_cell04_feedback_prepare_20261029_v1'
ROOT=host.launch.module_census.ROOT
BASE=str(ROOT/'results/agent_e3_observed_authority_20261029_v1_MAIN02.json')
BASE_SHA='ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9'
CELL=4
TASK='r0.g1.h1.dDxy'
SHA={'report':'fcc89ad01053d5f956a69290851acb321b9e161eb387b7e49c83dceb6b25a1c4',
     'completion':'03a09104111071d088a1636ac672792e70af632c17312c72ce7fba6ae85d16d4',
     'receipt':'436356d7060a08910f4ab4f2f4757a204f94b734c9300c55d12ff53610f06787'}
need=host.need


def control5(prior):
    refs=deepcopy(prior['control5'])
    for key in ('report','completion'):
        refs[key]=dict(path=str(ROOT/'results/agent_e3_gd_host_cuda_20261029_v1_cell04_MAIN01'/f'{key}.json'),sha256=SHA[key])
    refs['receipt']=dict(path=str(ROOT/'results/agent_e3_gd_host_cuda_20261029_v1_consume_cell04_MAIN01.json'),sha256=SHA['receipt'])
    return refs


def combine(prior,fresh):
    need(prior['cell']==0 and fresh['cell']==CELL and prior['schema']==fresh['schema'] and
         prior['resolver']==fresh['resolver'] and prior['control5']['prepare']==fresh['control5']['prepare'], 'fixed common prepare/resolver')
    need(fresh['control5']==control5(prior), 'fixed cell04 control5 refs')
    additions=dict(prior['additional_sources'])
    for path,digest in fresh['additional_sources'].items():
        need(path in additions and additions[path]==digest,'strict inherited source pins: '+path)
    return dict(schema=prior['schema'],cell=CELL,control5=deepcopy(fresh['control5']),
                resolver=deepcopy(prior['resolver']),additional_sources=additions)


def build():
    host.authority(BASE,BASE_SHA)
    prior=host.launch.host.read(BASE,BASE_SHA)
    fresh=identity.build(control5(prior),prior['resolver'],CELL)
    value=combine(prior,fresh)
    need(host.launch.host.sha(BASE)==BASE_SHA,'stable base authority')
    return value


def verify(path,sha):
    expected=build()
    need(host.launch.host.read(path,sha)==expected,'whole cell04 authority reconstruction')
    binding=host.authority(path,sha)
    need(binding['cell']==CELL and host.launch.host.sha(path)==sha and host.launch.host.sha(BASE)==BASE_SHA,'final strict source and cell')
    return dict(verified=True,cell=CELL,task=TASK,freshness=False,scientific_admission=False)


def prepare(path):
    p=Path(path);need(p.is_absolute() and p.parent==ROOT/'results' and p.name.startswith(NAME+'_'),'own exclusive result')
    digest=startup.write_new(path,build());verify(path,digest)
    return dict(path=path,sha256=digest,cell=CELL,task=TASK,freshness=False,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    x=s.add_parser('prepare');x.add_argument('--output',required=True)
    x=s.add_parser('verify');x.add_argument('--authority',required=True);x.add_argument('--authority-sha256',required=True)
    x=p.parse_args();result=prepare(x.output) if x.command=='prepare' else verify(x.authority,x.authority_sha256)
    print(host.launch.host.raw(result).decode())


if __name__=='__main__':main()
