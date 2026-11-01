"""Cell02 fixed E3 feedback authority preparation; no GPU or physics."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as identity
from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as startup

NAME = 'e3_cell02_feedback_prepare_codex_20261029_v1'
ROOT = host.launch.module_census.ROOT
BASE = str(ROOT / 'results/agent_e3_observed_authority_20261029_v1_MAIN02.json')
BASE_SHA = 'ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9'
CELL = 2
TASK = 'r0.g0.h3.dDxy'
SHA = {
    'report': '55b1950367f13ac2678ebbeacce456f562158a48871dc29fdfa8b9ccaea6e2d4',
    'completion': 'a1eb224a28176281ab169b810d0b193ba49bc189f09bd8db53beb437b2dc51c3',
    'receipt': '0f1fafa74144db5c2e0e5a4a059d5c91d30260b7207ac99f02a1a07486174b97',
}
need = host.need


def references(prior):
    refs = deepcopy(prior['control5'])
    for key in ('report', 'completion'):
        refs[key] = dict(path=str(ROOT / f'results/agent_e3_gd_host_cuda_20261029_v1_cell02_MAIN01/{key}.json'), sha256=SHA[key])
    refs['receipt'] = dict(path=str(ROOT / 'results/agent_e3_gd_host_cuda_20261029_v1_consume_cell02_MAIN01.json'), sha256=SHA['receipt'])
    return refs


def combine(prior, fresh):
    refs = references(prior)
    need(prior['cell'] == 0 and fresh['cell'] == CELL and prior['schema'] == fresh['schema'] and
         prior['resolver'] == fresh['resolver'] and prior['control5']['prepare'] == fresh['control5']['prepare'] and
         fresh['control5'] == refs, 'fixed cell02 upstream identity')
    sources = dict(prior['additional_sources'])
    for path, digest in fresh['additional_sources'].items():
        need(path in sources and sources[path] == digest, 'strict inherited source pins: '+path)
    return dict(schema=prior['schema'], cell=CELL, control5=deepcopy(refs),
                resolver=deepcopy(prior['resolver']), additional_sources=sources)


def build():
    host.authority(BASE, BASE_SHA)
    prior = host.launch.host.read(BASE, BASE_SHA)
    fresh = identity.build(references(prior), prior['resolver'], CELL)
    result = combine(prior, fresh)
    need(host.launch.host.sha(BASE) == BASE_SHA, 'stable base authority')
    return result


def verify(path, sha):
    need(host.launch.host.read(path, sha) == build(), 'whole cell02 authority reconstruction')
    binding = host.authority(path, sha)
    need(binding['cell'] == CELL and host.launch.host.sha(path) == sha and
         host.launch.host.sha(BASE) == BASE_SHA, 'strict cell02 source binding')
    return dict(verified=True, cell=CELL, task=TASK, freshness=False, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True)
    p = sub.add_parser('verify'); p.add_argument('--authority', required=True); p.add_argument('--authority-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        path = Path(args.output)
        need(path.is_absolute() and path.parent == ROOT/'results' and path.name.startswith(NAME+'_'), 'own exclusive output')
        digest = startup.write_new(str(path), build())
        verify(str(path), digest)
        result = dict(path=str(path), sha256=digest, cell=CELL, task=TASK, freshness=False, scientific_admission=False)
    else:
        result = verify(args.authority, args.authority_sha256)
    print(host.launch.host.raw(result).decode())


if __name__ == '__main__':
    main()
