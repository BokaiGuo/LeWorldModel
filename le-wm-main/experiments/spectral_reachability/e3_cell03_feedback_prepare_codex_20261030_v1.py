"""Prepare and independently verify the fixed cell03 feedback authority (CPU only)."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import e3_cell02_feedback_prepare_codex_20261029_v1 as base

host = base.host
identity = base.identity
startup = base.startup
need = base.need
ROOT = base.ROOT
NAME = 'e3_cell03_feedback_prepare_codex_20261030_v1'
CELL = 3
TASK = 'r0.g0.h3.dDx'
SHA = {
    'report': '38c97762714f14d9caa41d5633042054f391aa9223b6f07f7b4efdae5492ad93',
    'completion': 'e4262011310e3a9a74f0ba4c62e8cd0bae44e55a9e818db29fabc385d8afccca',
    'receipt': '253ed296df3d812b0b4f01ca7df31c99dd024f8b2166454c578eef072be649e3',
}


def references(prior):
    refs = deepcopy(prior['control5'])
    for key in ('report', 'completion'):
        refs[key] = dict(path=str(ROOT / f'results/agent_e3_gd_host_cuda_20261029_v1_cell03_MAIN01/{key}.json'), sha256=SHA[key])
    refs['receipt'] = dict(path=str(ROOT / 'results/agent_e3_gd_host_cuda_20261029_v1_consume_cell03_MAIN01.json'), sha256=SHA['receipt'])
    return refs


def combine(prior, fresh):
    refs = references(prior)
    need(prior['cell'] == 0 and fresh['cell'] == CELL and prior['schema'] == fresh['schema'] and
         prior['resolver'] == fresh['resolver'] and prior['control5']['prepare'] == fresh['control5']['prepare'] and
         fresh['control5'] == refs, 'fixed cell03 upstream identity')
    sources = dict(prior['additional_sources'])
    for path, digest in fresh['additional_sources'].items():
        need(path in sources and sources[path] == digest, 'strict inherited source pins: ' + path)
    return dict(schema=prior['schema'], cell=CELL, control5=deepcopy(refs),
                resolver=deepcopy(prior['resolver']), additional_sources=sources)


def build():
    host.authority(base.BASE, base.BASE_SHA)
    prior = host.launch.host.read(base.BASE, base.BASE_SHA)
    fresh = identity.build(references(prior), prior['resolver'], CELL)
    result = combine(prior, fresh)
    need(host.launch.host.sha(base.BASE) == base.BASE_SHA, 'stable base authority')
    return result


def verify(path, sha):
    need(host.launch.host.read(path, sha) == build(), 'whole cell03 authority reconstruction')
    binding = host.authority(path, sha)
    need(binding['cell'] == CELL and host.launch.host.sha(path) == sha and
         host.launch.host.sha(base.BASE) == base.BASE_SHA, 'strict cell03 source binding')
    return dict(verified=True, cell=CELL, task=TASK, freshness=False, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True)
    p = sub.add_parser('verify'); p.add_argument('--authority', required=True); p.add_argument('--authority-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        path = Path(args.output)
        need(path.is_absolute() and path.parent == ROOT / 'results' and path.name.startswith(NAME + '_'), 'own exclusive output')
        digest = startup.write_new(str(path), build())
        verify(str(path), digest)
        result = dict(path=str(path), sha256=digest, cell=CELL, task=TASK, freshness=False, scientific_admission=False)
    else:
        result = verify(args.authority, args.authority_sha256)
    print(host.launch.host.raw(result).decode())


if __name__ == '__main__':
    main()
