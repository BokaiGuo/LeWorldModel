"""Extend a SHA-pinned E3 identity authority for the v2 preflight, without executing it."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from experiments.spectral_reachability import agent_e3_encoder_cuda_preflight_20261029_v2 as preflight

host = preflight.host
need = host.need
base = host.launch.pub.base
NAME = 'agent_e3_preflight_v2_authority_prepare_20261029_v1'


def build(source_path, source_sha):
    """Validate the old identity authority and produce its exact extended ABI in memory."""
    binding = host.authority(source_path, source_sha)
    original = host.launch.host.read(source_path, source_sha)
    result = deepcopy(original)
    additions = [str(Path(m.__file__).resolve()) for m in
                 (preflight, preflight.array_transport, preflight.capture, host.identity.encoder)]
    additions.append(str(Path(__file__).resolve()))
    for path in additions:
        sha = host.launch.host.sha(path)
        existing = result['additional_sources'].get(path)
        need(existing is None or existing == sha, 'conflicting source pin')
        old = binding['prepared']['bindings_sha256'].get(path)
        need(old is None or old == sha, 'conflicting prepared source pin')
        result['additional_sources'][path] = sha
    candidate_binding = dict(binding, additional_sources=result['additional_sources'])
    preflight.sources(candidate_binding)
    need(host.launch.host.sha(source_path) == source_sha, 'source authority changed')
    return result


def prepare(source_path, source_sha, output):
    value = build(source_path, source_sha)
    p = Path(output)
    need(p.is_absolute() and str(p.resolve()) == str(p) and p.parent.is_dir()
         and not p.exists() and not p.is_symlink() and str(p) != source_path, 'new canonical output')
    data = host.launch.host.raw(value)
    need(len(data) <= host.launch.host.META, 'authority metadata bound')
    with p.open('xb') as f:
        need(f.write(data) == len(data), 'complete authority write')
        f.flush(); os.fsync(f.fileno())
    base.syncdir(p.parent)
    sha = hashlib.sha256(data).hexdigest()
    verify(source_path, source_sha, str(p), sha)
    return dict(schema=NAME+'.prepare', output=str(p), sha256=sha, bytes=len(data), scientific_admission=False)


def verify(source_path, source_sha, output, output_sha):
    expected = build(source_path, source_sha)
    actual = host.launch.host.read(output, output_sha)
    need(actual == expected, 'complete authority reconstruction')
    binding = host.authority(output, output_sha)
    preflight.sources(binding)
    need(host.launch.host.sha(source_path) == source_sha and host.launch.host.sha(output) == output_sha,
         'authority files unchanged')
    return dict(schema=NAME+'.verify', verified=True, source_sha256=source_sha,
                output_sha256=output_sha, scientific_admission=False, full_upstream_strict_rebuilt=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'verify'):
        p = sub.add_parser(command)
        for option in ('source-authority', 'source-sha256', 'output'):
            p.add_argument('--'+option, required=True)
        if command == 'verify':
            p.add_argument('--output-sha256', required=True)
    args = parser.parse_args()
    result = (prepare(args.source_authority, args.source_sha256, args.output) if args.command == 'prepare'
              else verify(args.source_authority, args.source_sha256, args.output, args.output_sha256))
    print(host.launch.host.raw(result).decode())


if __name__ == '__main__':
    main()
