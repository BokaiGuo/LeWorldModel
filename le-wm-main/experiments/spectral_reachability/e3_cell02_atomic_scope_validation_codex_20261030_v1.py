"""Cell02 read-only manifest validation and temporary-only atomic protocol experiment.

This module cannot launch production or acquire a production claim. The
temporary protocol is evidence about cooperating processes on one filesystem,
not an exclusion mechanism for existing entrypoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.spectral_reachability import e3_cell02_deployment_preflight_codex_20261030_v1 as preflight
from experiments.spectral_reachability import e3_cell02_exclusive_candidate_codex_20261030_v1 as candidate
from experiments.spectral_reachability import e3_cell02_final_review_codex_20261030_v1 as review

MARKER = 'e3_cell02_cooperative_scope.marker'
SCOPE = 'E3/cell02/all-cooperating-entrypoints'


def validate_manifest(path, digest):
    """Compare an externally pinned manifest to an isolated fresh draft.

    This is not the complete resource review or authorization. No environment
    variables are changed; the draft inherits this process's environment.
    """
    tracked = [Path(__file__).resolve(), Path(review.__file__).resolve(),
               Path(preflight.__file__).resolve(), Path(candidate.__file__).resolve(),
               Path(sys.executable).resolve(), candidate.AUTHORITY]
    before = {str(p): candidate.file_sha(p) for p in tracked}
    env = candidate.environment_sha()
    invocation = sys.executable
    data, approved = review.read_manifest(path, digest)
    fresh = json.loads(preflight.manifest_bytes())
    if approved != fresh:
        raise ValueError('manifest differs from isolated current draft')
    if review.read_manifest(path, digest)[0] != data:
        raise ValueError('manifest changed during validation')
    if (candidate.environment_sha() != env or sys.executable != invocation or
            {str(p): candidate.file_sha(p) for p in tracked} != before):
        raise ValueError('validator, authority, interpreter or environment drift')
    return {'cell': 2, 'scope': SCOPE, 'manifest_sha256': digest,
            'isolated_draft_matches': True, 'validator_sha256': before[str(tracked[0])],
            'production_ready': False, 'production_authorized': False,
            'scientific_admission': False, 'full_saved_strict': False}


def temporary_claim(directory, entrypoint, manifest_sha, output_label):
    """Test-only primitive: no removal, even if write/fsync fails.

    Directory must already exist under the system temporary directory. Callers
    must cooperate on the same directory; labels never affect the marker name.
    This primitive is deliberately not a production gate.
    """
    path = Path(directory)
    tmp = Path('/tmp').resolve()
    if (not path.is_absolute() or path != path.resolve() or path == tmp or
            tmp not in path.parents or not path.is_dir()):
        raise ValueError('canonical existing temporary subdirectory required')
    if not review.sha_valid(manifest_sha) or not entrypoint or not output_label:
        raise ValueError('invalid test metadata')
    record = candidate.canonical({'scope': SCOPE, 'entrypoint': entrypoint,
                                  'manifest_sha256': manifest_sha,
                                  'output_label': output_label, 'pid': os.getpid(),
                                  'test_only': True})
    dfd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=dfd)
        with os.fdopen(fd, 'wb') as stream:
            if stream.write(record) != len(record):
                raise OSError('short marker write')
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(dfd)
    finally:
        os.close(dfd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    try:
        print(candidate.canonical(validate_manifest(args.manifest, args.manifest_sha256)).decode())
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({'error': str(exc), 'production_authorized': False}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
