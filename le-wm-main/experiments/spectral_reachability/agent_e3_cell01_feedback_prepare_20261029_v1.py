"""Prepare the fixed cell01 known-root feedback authority; no CUDA or physics."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as identity
from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as startup

NAME='agent_e3_cell01_feedback_prepare_20261029_v1'
ROOT=host.launch.module_census.ROOT
BASE=str(ROOT/'results/agent_e3_observed_authority_20261029_v1_MAIN02.json')
BASE_SHA='ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9'
CELL=1
SHA={'report':'e19b63330f6204fa149a733f81500e60f8d57a1fcda04722ffe704cbd87dce02',
     'completion':'d4d337b94f18a7e8d170ff9c73f9b8ccbe94f128df89029d5eb0f59419587271',
     'receipt':'f80fd02ba0158f40ba51f936117f323be4a46bc1e2a017359346f4102afc16de'}
need=host.need


def control5(prior):
    refs=deepcopy(prior['control5'])
    for name in ('report','completion'):
        refs[name]=dict(path=str(ROOT/'results/agent_e3_gd_host_cuda_20261029_v1_cell01_MAIN01'/f'{name}.json'),sha256=SHA[name])
    refs['receipt']=dict(path=str(ROOT/'results/agent_e3_gd_host_cuda_20261029_v1_consume_cell01_MAIN01.json'),sha256=SHA['receipt'])
    return refs


def combine(prior, fresh):
    need(prior['cell']==0 and fresh['cell']==CELL and prior['schema']==fresh['schema'] and
         prior['resolver']==fresh['resolver'] and prior['control5']['prepare']==fresh['control5']['prepare'], 'same fixed upstream prepare and resolver')
    need(fresh['control5']==control5(prior), 'fixed cell01 control5 refs')
    additions=dict(prior['additional_sources'])
    for path,digest in fresh['additional_sources'].items():
        need(path in additions and additions[path]==digest, 'strict existing source pins: '+path)
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
    value=build()
    need(host.launch.host.read(path,sha)==value,'exact cell01 authority reconstruction')
    binding=host.authority(path,sha)
    need(binding['cell']==CELL and host.launch.host.sha(path)==sha and host.launch.host.sha(BASE)==BASE_SHA,'final strict authority')
    return dict(verified=True,cell=CELL,task='r0.g0.h1.dDx',freshness=False,scientific_admission=False)


def prepare(path):
    p=Path(path)
    need(p.is_absolute() and p.parent==ROOT/'results' and p.name.startswith(NAME+'_'), 'own exclusive result')
    sha=startup.write_new(path,build()); verify(path,sha)
    return dict(path=path,sha256=sha,cell=CELL,task='r0.g0.h1.dDx',freshness=False,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    x=s.add_parser('prepare');x.add_argument('--output',required=True)
    x=s.add_parser('verify');x.add_argument('--authority',required=True);x.add_argument('--authority-sha256',required=True)
    x=p.parse_args()
    r=prepare(x.output) if x.command=='prepare' else verify(x.authority,x.authority_sha256)
    print(host.launch.host.raw(r).decode())


if __name__=='__main__':main()
