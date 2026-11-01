"""Saved-only, exclusive creation of source-pinned chunked HOST authority."""
import argparse
import hashlib
import os
from pathlib import Path
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as new

host = new.launch.host
need = new.need


def build(source, sha):
    binding = new.host.authority(source, sha)
    value = host.read(source, sha)
    value = dict(value, additional_sources=dict(value['additional_sources']))
    for module in (new, new.chunks):
        path = str(Path(module.__file__).resolve()); digest = host.sha(path)
        old = value['additional_sources'].get(path)
        need(old is None or old == digest, 'conflicting source pin')
        value['additional_sources'][path] = digest
    path = str(Path(__file__).resolve()); digest = host.sha(path)
    old = value['additional_sources'].get(path)
    need(old is None or old == digest, 'conflicting authority source pin')
    value['additional_sources'][path] = digest
    sources = dict(binding['prepared']['bindings_sha256']); host.source.merge(sources, value['additional_sources'])
    new.launch.episode.old.guard.validate_sources(sources, list(value['additional_sources']))
    need(host.sha(source) == sha, 'stable source authority')
    return value


def verify(source, sha, output, output_sha):
    need(host.read(output, output_sha) == build(source, sha), 'exact authority reconstruction')
    new.authority(output, output_sha)
    return dict(verified=True, sha256=output_sha, scientific_admission=False, full_upstream_strict_rebuilt=False)


def prepare(source, sha, output):
    value = build(source, sha); p = Path(output)
    need(p.is_absolute() and str(p.resolve()) == str(p) and p.parent.is_dir() and not p.exists() and not p.is_symlink(), 'new canonical output')
    data = host.raw(value); need(len(data) <= host.META, 'bounded authority')
    with p.open('xb') as f:
        need(f.write(data) == len(data), 'complete authority write'); f.flush(); os.fsync(f.fileno())
    new.launch.pub.base.syncdir(p.parent); digest = hashlib.sha256(data).hexdigest()
    verify(source, sha, str(p), digest)
    return dict(output=str(p), sha256=digest, scientific_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'verify'):
        a = sub.add_parser(name)
        for opt in ('source-authority', 'source-sha256', 'output'): a.add_argument('--'+opt, required=True)
        if name == 'verify': a.add_argument('--output-sha256', required=True)
    a = p.parse_args()
    r = prepare(a.source_authority, a.source_sha256, a.output) if a.command == 'prepare' else verify(a.source_authority, a.source_sha256, a.output, a.output_sha256)
    print(host.raw(r).decode())


if __name__ == '__main__': main()
