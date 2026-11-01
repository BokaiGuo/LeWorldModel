"""Read-only final cell02 manifest/resource review; never starts production."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.spectral_reachability import e3_cell02_deployment_preflight_codex_20261030_v1 as preflight
from experiments.spectral_reachability import e3_cell02_exclusive_candidate_codex_20261030_v1 as candidate

NAME = 'e3_cell02_final_review_codex_20261030_v1'
MAX_MANIFEST = 16*1024**2
BLOCKERS = [
    'independent manifest and checker source review and external SHA approval',
    'explicit cell02 production authorization and separate GPU coordination',
    'GPU selection, availability, exclusive use and memory capacity unverified',
    'cross-entrypoint and other-process duplicate attempt exclusion unverified',
    'cgroup and competing-workload memory, disk/inode reservations unverified',
    'filesystem write permissions, quota and fsync durability not pre-proved',
    'physical and numerical operation caps not independently summarized by this review',
    'native renderer and runtime behavior not exercised',
    'preflight-to-launch resource and claim race; no atomic resource reservation',
    'same-byte imports, transient modules and external native dependencies not attested',
    'terminal publication and independent saved acceptance not yet available',
    'scientific admission and full saved strict remain false',
]


def sha_valid(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def read_manifest(path, digest):
    p = Path(path)
    if (not p.is_absolute() or p.parent != preflight.MANIFEST_DIRECTORY
            or p.parent != p.parent.resolve() or not p.name.startswith('e3_cell02_manifest_')
            or p.suffix != '.json' or not p.is_file() or p.is_symlink()):
        raise ValueError('dedicated canonical cell02 manifest required')
    if not sha_valid(digest) or p.stat().st_size > MAX_MANIFEST:
        raise ValueError('bounded externally pinned manifest required')
    data = p.read_bytes()
    if len(data) > MAX_MANIFEST or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('manifest SHA mismatch')
    parsed = json.loads(data)
    if data not in (candidate.canonical(parsed), candidate.canonical(parsed)+b'\n'):
        raise ValueError('noncanonical manifest')
    return data, parsed


def local_checks(observed):
    return {'fixed_claim_absent': observed['claim_exists'] is False,
            'fixed_output_absent': observed['output_exists'] is False,
            'cuda_not_explicitly_disabled': observed['cuda_explicitly_disabled'] is False,
            'disk_contract_minimum_met': observed['disk_minimum_met'] is True,
            'inode_available': type(observed['available_inodes']) is int and observed['available_inodes'] > 0}


def review(path, manifest_sha):
    own_sha = candidate.file_sha(Path(__file__).resolve())
    data, approved = read_manifest(path, manifest_sha)
    before = preflight.observe()
    before_checks = local_checks(before)
    if not all(before_checks.values()):
        raise ValueError('pre-draft local checks failed: '+','.join(k for k, v in before_checks.items() if not v))
    fresh = preflight.manifest_bytes()
    current = json.loads(fresh)
    if approved != current:
        raise ValueError('approved manifest differs from fresh launcher draft')
    after = preflight.observe()
    after_checks = local_checks(after)
    if not all(after_checks.values()):
        raise ValueError('post-draft local checks failed: '+','.join(k for k, v in after_checks.items() if not v))
    if (before['contract'] != after['contract'] or before['tool_sha256'] != after['tool_sha256']
            or before['transport_source_sha256'] != after['transport_source_sha256']
            or candidate.file_sha(Path(__file__).resolve()) != own_sha):
        raise ValueError('review source or contract drift')
    if read_manifest(path, manifest_sha)[0] != data:
        raise ValueError('manifest changed during review')
    return {'schema': NAME+'.review', 'cell': 2, 'manifest_path': str(path),
            'manifest_sha256': manifest_sha, 'fresh_draft_matches': True,
            'before': before, 'after': after, 'checks_before': before_checks,
            'checks_after': after_checks, 'reviewer_source_sha256': own_sha,
            'unresolved_blockers': BLOCKERS, 'production_ready': False,
            'production_authorized': False, 'scientific_admission': False,
            'full_saved_strict': False, 'gpu_probed': False, 'physical_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    try:
        print(candidate.canonical(review(args.manifest, args.manifest_sha256)).decode())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({'error': str(exc), 'production_ready': False,
                          'production_authorized': False, 'scientific_admission': False}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
