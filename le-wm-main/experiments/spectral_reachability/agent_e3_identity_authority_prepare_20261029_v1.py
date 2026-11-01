"""Build a saved-only identity HOST authority from pinned control5 and resolver files."""
import argparse
import hashlib
import os
from pathlib import Path

from experiments.spectral_reachability import agent_e3_identity_host_20261029_v1 as identity_host

launch = identity_host.launch
host = launch.host
need = launch.need
base = launch.pub.base
NAME = 'agent_e3_identity_authority_prepare_20261029_v1'
KEYS = ('prepare', 'report', 'completion', 'receipt')


def build(control5, resolver, cell):
    need(type(control5) is dict and set(control5) == set(KEYS) and type(resolver) is dict
         and set(resolver) == {'path', 'sha256'} and type(cell) is int and 0 <= cell < 16,
         'explicit fixed authority inputs')
    for ref in list(control5.values()) + [resolver]:
        need(type(ref) is dict and set(ref) == {'path', 'sha256'} and type(ref['path']) is str
             and Path(ref['path']).is_absolute() and str(Path(ref['path']).resolve()) == ref['path']
             and type(ref['sha256']) is str and len(ref['sha256']) == 64
             and all(c in '0123456789abcdef' for c in ref['sha256']), 'canonical SHA reference')
    need(control5['prepare']['sha256'] == launch.CONTROL5_PREPARE_SHA
         and resolver['sha256'] == launch.RESOLVER_SHA, 'original fixed prepare and resolver')
    prepared = host.read(control5['prepare']['path'], control5['prepare']['sha256'])
    host.read(resolver['path'], resolver['sha256'])
    need(prepared['actual_resolver'] == resolver, 'resolver path and hash join')
    args = tuple(v for name in KEYS for v in (control5[name]['path'], control5[name]['sha256'])) + (cell,)
    value = launch.episode.old.authority.load(*args)
    need(value['cell'] == cell, 'original saved control5 receipt join')
    prior = identity_host.prior
    modules = (launch, launch.episode, launch.loader_bridge, launch.pub, launch.pub.base,
               launch.host, launch.reset_identity, launch.module_census, prior, prior.saved,
               prior.partial_guard, identity_host, identity_host.identity, identity_host.capacity)
    paths = {str(Path(module.__file__).resolve()) for module in modules}
    paths.add(str(Path(__file__).resolve()))
    additions = {path: host.sha(path) for path in sorted(paths)}
    sources = dict(prepared['bindings_sha256'])
    host.source.merge(sources, additions)
    launch.episode.old.guard.validate_sources(sources, sorted(paths))
    return dict(schema=launch.NAME+'.authority', cell=cell, control5=control5,
                resolver=resolver, additional_sources=additions)


def verify(control5, resolver, cell, output, output_sha):
    expected = build(control5, resolver, cell)
    need(host.read(output, output_sha) == expected, 'complete identity authority reconstruction')
    binding = identity_host.authority(output, output_sha)
    need(binding['cell'] == cell and host.sha(output) == output_sha, 'identity reader and stable authority')
    return dict(schema=NAME+'.verify', verified=True, sha256=output_sha, cell=cell,
                full_upstream_strict_rebuilt=False, scientific_admission=False)


def prepare(control5, resolver, cell, output):
    value = build(control5, resolver, cell)
    p = Path(output)
    need(p.is_absolute() and str(p.resolve()) == str(p) and p.parent.is_dir()
         and not p.exists() and not p.is_symlink(), 'new canonical output')
    data = host.raw(value)
    need(len(data) <= host.META, 'bounded authority')
    with p.open('xb') as f:
        need(f.write(data) == len(data), 'complete authority write')
        f.flush(); os.fsync(f.fileno())
    base.syncdir(p.parent)
    digest = hashlib.sha256(data).hexdigest()
    verify(control5, resolver, cell, str(p), digest)
    return dict(schema=NAME+'.prepare', path=str(p), sha256=digest, bytes=len(data),
                cell=cell, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'verify'):
        p = sub.add_parser(command)
        for label in (*KEYS, 'resolver'):
            p.add_argument('--'+label, required=True)
            p.add_argument('--'+label+'-sha256', required=True)
        p.add_argument('--cell', type=int, required=True)
        p.add_argument('--output', required=True)
        if command == 'verify': p.add_argument('--output-sha256', required=True)
    a = parser.parse_args()
    refs = {key: dict(path=getattr(a, key), sha256=getattr(a, key+'_sha256')) for key in KEYS}
    resolver = dict(path=a.resolver, sha256=a.resolver_sha256)
    result = (prepare(refs, resolver, a.cell, a.output) if a.command == 'prepare'
              else verify(refs, resolver, a.cell, a.output, a.output_sha256))
    print(host.raw(result).decode())


if __name__ == '__main__': main()
