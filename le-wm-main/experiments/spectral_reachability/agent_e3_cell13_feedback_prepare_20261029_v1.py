"""Prepare fixed-roster cell13 feedback authority; no model or physics execution."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as identity
from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as startup

NAME = 'agent_e3_cell13_feedback_prepare_20261029_v1'
ROOT = host.launch.module_census.ROOT
BASE = str(ROOT / 'results/agent_e3_observed_authority_20261029_v1_MAIN02.json')
BASE_SHA = 'ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9'
CELL = 13
TASK = 'r1.g1.h1.dDx'
SHA = {'report': '9eda5740c65b2dad6b7ec8e17f41cd0215b97d92a38bcf4953b97384d4d5396d',
       'completion': '44c5d4f2b380244218bfec0e114b06a94aabd988372487629bca19c05b7bfaee',
       'receipt': '14b797ac5a781472d3aa5d473075104410b89893e9f971d101c41c3a5e4968cd'}
need = host.need


def build():
    host.authority(BASE, BASE_SHA)
    prior = host.launch.host.read(BASE, BASE_SHA)
    refs = deepcopy(prior['control5'])
    for key in ('report', 'completion'):
        refs[key] = dict(path=str(ROOT / f'results/agent_e3_gd_host_cuda_20261029_v1_cell13_MAIN01/{key}.json'), sha256=SHA[key])
    refs['receipt'] = dict(path=str(ROOT / 'results/agent_e3_gd_host_cuda_20261029_v1_consume_cell13_MAIN01.json'), sha256=SHA['receipt'])
    fresh = identity.build(refs, prior['resolver'], CELL)
    need(prior['cell'] == 0 and fresh['cell'] == CELL and prior['schema'] == fresh['schema'] and
         prior['resolver'] == fresh['resolver'] and prior['control5']['prepare'] == fresh['control5']['prepare'] and
         fresh['control5'] == refs, 'fixed common prepare/resolver and cell13 control5')
    additions = dict(prior['additional_sources'])
    for path, digest in fresh['additional_sources'].items():
        need(path in additions and additions[path] == digest, 'strict inherited source pins: ' + path)
    need(host.launch.host.sha(BASE) == BASE_SHA, 'stable base authority')
    return dict(schema=prior['schema'], cell=CELL, control5=deepcopy(refs),
                resolver=deepcopy(prior['resolver']), additional_sources=additions)


def verify(path, sha):
    need(host.launch.host.read(path, sha) == build(), 'whole cell13 authority reconstruction')
    binding = host.authority(path, sha)
    need(binding['cell'] == CELL and host.launch.host.sha(path) == sha and
         host.launch.host.sha(BASE) == BASE_SHA, 'final strict source and cell')
    return dict(verified=True, cell=CELL, task=TASK, freshness=False, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True)
    p = sub.add_parser('verify'); p.add_argument('--authority', required=True); p.add_argument('--authority-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        path = Path(args.output)
        need(path.is_absolute() and path.parent == ROOT / 'results' and path.name.startswith(NAME + '_'), 'own exclusive result')
        digest = startup.write_new(str(path), build())
        verify(str(path), digest)
        result = dict(path=str(path), sha256=digest, cell=CELL, task=TASK, freshness=False, scientific_admission=False)
    else:
        result = verify(args.authority, args.authority_sha256)
    print(host.launch.host.raw(result).decode())


if __name__ == '__main__':
    main()
