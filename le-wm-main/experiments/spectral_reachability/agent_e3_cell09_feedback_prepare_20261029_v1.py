"""Prepare fixed-roster root1/G0 cell09 feedback authority; no model or physics."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as identity
from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as startup

NAME = 'agent_e3_cell09_feedback_prepare_20261029_v1'
ROOT = host.launch.module_census.ROOT
BASE = str(ROOT / 'results/agent_e3_observed_authority_20261029_v1_MAIN02.json')
BASE_SHA = 'ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9'
CELL = 9
TASK = 'r1.g0.h1.dDx'
SHA = {'report': '7927d05c97ad8f46e96679ed15acd53778e04a4b2b807be2cdd6307c6d4c57e7',
       'completion': '84d1917667876d6d30104e2f59ae35bc7e8652afbb0779d26a590e2143388541',
       'receipt': 'dc6588fb3649225c73952f1743df4fcc4fde83a1bc5ff96e804a6836ccfc9583'}
need = host.need


def build():
    host.authority(BASE, BASE_SHA)
    prior = host.launch.host.read(BASE, BASE_SHA)
    refs = deepcopy(prior['control5'])
    for key in ('report', 'completion'):
        refs[key] = dict(path=str(ROOT / f'results/agent_e3_gd_host_cuda_20261029_v1_cell09_MAIN01/{key}.json'), sha256=SHA[key])
    refs['receipt'] = dict(path=str(ROOT / 'results/agent_e3_gd_host_cuda_20261029_v1_consume_cell09_MAIN01.json'), sha256=SHA['receipt'])
    fresh = identity.build(refs, prior['resolver'], CELL)
    need(prior['cell'] == 0 and fresh['cell'] == CELL and prior['schema'] == fresh['schema'] and
         prior['resolver'] == fresh['resolver'] and prior['control5']['prepare'] == fresh['control5']['prepare'] and
         fresh['control5'] == refs, 'fixed common prepare/resolver and cell09 control5')
    additions = dict(prior['additional_sources'])
    for path, digest in fresh['additional_sources'].items():
        need(path in additions and additions[path] == digest, 'strict inherited source pins: ' + path)
    need(host.launch.host.sha(BASE) == BASE_SHA, 'stable base authority')
    return dict(schema=prior['schema'], cell=CELL, control5=deepcopy(refs),
                resolver=deepcopy(prior['resolver']), additional_sources=additions)


def verify(path, sha):
    need(host.launch.host.read(path, sha) == build(), 'whole cell09 authority reconstruction')
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
