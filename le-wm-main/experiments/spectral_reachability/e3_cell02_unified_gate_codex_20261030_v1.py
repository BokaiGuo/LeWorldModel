"""Cell02 unified-gate audit and temporary-only concurrency protocol.

No production dispatch or production claim creation is implemented here:
legacy entrypoints can bypass any new cooperative gate.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from experiments.spectral_reachability import e3_cell02_atomic_scope_validation_codex_20261030_v1 as atomic
from experiments.spectral_reachability import e3_cell02_caps_dedup_codex_20261030_v1 as dedup

candidate = atomic.candidate
HERE = Path(__file__).resolve().parent
ENTRYPOINTS = (
    'e3_cell02_exclusive_candidate_codex_20261030_v1.py',
    'agent_e3_chunked_identity_host_20261029_v1.py',
    'agent_e3_identity_host_20261029_v1.py',
    'agent_e3_continuous_host_launcher_20261029_v1.py',
    'agent_e3_integrated_host_saved_20261029_v1.py',
    'agent_e3_gd_host_cuda_20261029_v1.py',
)
TEST_MARKER = 'e3_cell02_unified_gate_TEST_ONLY.claim'


def entrypoint_census():
    """Inventory known entrypoint bytes, not proof that this list is exhaustive."""
    result = {}
    for name in ENTRYPOINTS:
        path = HERE / name
        if not path.is_file() or path.is_symlink() or path != path.resolve():
            raise ValueError('entrypoint missing or noncanonical: ' + name)
        result[name] = candidate.file_sha(path)
    return result


def audit():
    """Bounded, read-only evidence; always blocked for production."""
    before = entrypoint_census()
    names = dedup.candidate_names(candidate.RESULTS)
    # lexists sees dangling symlinks and never opens a historical claim.
    fixed_claim = os.path.lexists(candidate.CLAIM)
    fixed_output = os.path.lexists(candidate.OUTPUT)
    if entrypoint_census() != before:
        raise ValueError('entrypoint source changed during audit')
    return {
        'schema': 'e3_cell02_unified_gate.audit.v1', 'cell': 2,
        'fixed_denominator': 16, 'known_entrypoint_sha256': before,
        'bounded_names': names, 'fixed_claim_present': fixed_claim,
        'fixed_output_present': fixed_output,
        'production_ready': False, 'production_authorized': False,
        'scientific_admission': False, 'full_saved_strict': False,
        'blockers': [
            'legacy entrypoints do not participate in a unified lock',
            'historical attempts outside bounded name scan not excluded',
            'other processes and hosts not excluded or fenced',
            'new gate and actual launch binding not sealed or approved',
            'final manifest, resource review and GPU coordination remain separate',
        ],
    }


def temporary_gate(directory, entrypoint, manifest_sha, known_upstream=()):
    """Exercise one permanent scope marker against a *closed test directory*.

    An unknown entry, including a broken symlink or another marker, blocks.
    This does not inspect or claim the real results tree. Caller-selected
    upstream names are safe only in a synthetic, independently controlled test.
    """
    path = Path(directory)
    root = Path('/tmp').resolve()
    if (not path.is_absolute() or path != path.resolve() or path == root or
            root not in path.parents or not path.is_dir()):
        raise ValueError('existing canonical /tmp subdirectory required')
    if entrypoint not in ENTRYPOINTS or not atomic.review.sha_valid(manifest_sha):
        raise ValueError('unknown entrypoint or manifest digest')
    allowed = set(known_upstream)
    if any(not isinstance(n, str) or n in ('', '.', '..', TEST_MARKER)
           or '/' in n for n in allowed):
        raise ValueError('invalid synthetic upstream allowlist')
    dfd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        def inventory():
            return set(os.listdir(dfd)) - allowed
        if inventory():
            raise FileExistsError('historical or unclassified test artifact')
        record = candidate.canonical({'scope': atomic.SCOPE, 'test_only': True,
                                      'entrypoint': entrypoint, 'manifest_sha256': manifest_sha,
                                      'pid': os.getpid()})
        fd = os.open(TEST_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=dfd)
        with os.fdopen(fd, 'wb') as stream:
            if stream.write(record) != len(record):
                raise OSError('short claim write')
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(dfd)
        if inventory() != {TEST_MARKER}:
            raise FileExistsError('test artifact appeared during claim')
        return {'test_scope_claimed': True, 'production_authorized': False}
    finally:
        os.close(dfd)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(candidate.canonical(audit()).decode())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({'error': str(exc), 'production_authorized': False}), file=sys.stderr)
    # An audit cannot authorize or launch production, even when observations clear.
    return 1


if __name__ == '__main__':
    sys.exit(main())
