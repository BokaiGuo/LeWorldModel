"""Read-only cell02 deployment observations and exclusive manifest creation.

No CUDA probing, model loading, physics, production launch, or claim creation.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from experiments.spectral_reachability import e3_cell02_exclusive_candidate_codex_20261030_v1 as candidate
from experiments.spectral_reachability import agent_e3_unified_transport_20261029_v1 as transport

MODULE = 'experiments.spectral_reachability.e3_cell02_exclusive_candidate_codex_20261030_v1'
MANIFEST_DIRECTORY = candidate.ROOT/'deployment_manifests_e3_cell02'


def memory_available():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                parts = line.split()
                if len(parts) == 3 and parts[2] == 'kB':
                    return int(parts[1]) * 1024
    except OSError:
        pass
    return None


def limit(name):
    soft, hard = resource.getrlimit(name)
    return [None if x == resource.RLIM_INFINITY else x for x in (soft, hard)]


def observe():
    """Point-in-time observations only; never claim GPU or native readiness."""
    candidate.check_environment()
    parent = candidate.OUTPUT.parent
    if not parent.is_dir() or parent != parent.resolve():
        raise ValueError('canonical existing output parent required')
    disk = shutil.disk_usage(parent)
    fs = os.statvfs(parent)
    claim_exists = os.path.lexists(candidate.CLAIM)
    output_exists = os.path.lexists(candidate.OUTPUT)
    disabled = os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')
    return {
        'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cell': 2,
        'claim_exists': claim_exists,
        'output_exists': output_exists,
        'cuda_explicitly_disabled': disabled,
        'disk_free_bytes': disk.free,
        'disk_minimum_bytes': transport.CAP + transport.FLOOR,
        'disk_minimum_met': disk.free >= transport.CAP + transport.FLOOR,
        'available_inodes': fs.f_favail,
        'host_mem_available_bytes': memory_available(),
        'rlimit_address_space_bytes': limit(resource.RLIMIT_AS),
        'rlimit_open_files': limit(resource.RLIMIT_NOFILE),
        'cpu_affinity_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'tool_sha256': candidate.file_sha(Path(__file__).resolve()),
        'transport_source_sha256': candidate.file_sha(Path(transport.__file__).resolve()),
        'contract': {'output_cap_bytes': transport.CAP, 'final_reserve_bytes': transport.RESERVE,
                     'disk_floor_bytes': transport.FLOOR, 'wall_seconds': transport.WALL,
                     'category_caps_bytes': dict(transport.LIMITS)},
        'unverified': ['GPU availability and exclusive coordination', 'GPU memory',
                       'cgroup memory and competing workloads', 'native renderer and physical runtime',
                       'cross-entrypoint duplicate attempts', 'independent authority and manifest approval'],
        'production_authorized': False, 'scientific_admission': False,
    }


def manifest_bytes(timeout=180):
    # Fresh interpreter matches the launcher's own -m entrypoint. Importing the
    # launcher in this process must not contaminate its parent-module census.
    proc = subprocess.run([sys.executable, '-B', '-s', '-m', MODULE, 'draft'],
                          cwd=candidate.ROOT, capture_output=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError('candidate draft failed: ' + proc.stderr.decode(errors='replace')[-2000:])
    data = proc.stdout
    value = json.loads(data)
    if data != candidate.canonical(value) + b'\n':
        raise ValueError('candidate draft not canonical newline-terminated JSON')
    if value.get('schema') != candidate.SCHEMA or value.get('cell') != 2:
        raise ValueError('wrong draft identity')
    if value.get('authority_sha256') != candidate.AUTHORITY_SHA:
        raise ValueError('wrong authority digest')
    return data


def create_exclusive(path, data):
    path = Path(path)
    if not path.is_absolute() or path.parent != path.parent.resolve() or not path.parent.is_dir():
        raise ValueError('manifest parent must be an existing canonical absolute directory')
    if path.parent != MANIFEST_DIRECTORY:
        raise ValueError('manifest must use the dedicated cell02 manifest directory')
    if not path.name.startswith('e3_cell02_manifest_') or path.suffix != '.json':
        raise ValueError('manifest basename must start e3_cell02_manifest_ and end .json')
    if path == candidate.CLAIM or path == candidate.OUTPUT or path == candidate.AUTHORITY:
        raise ValueError('reserved production path')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short manifest write')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def generate(path):
    before = observe()
    def admissible(o):
        return (not o['claim_exists'] and not o['output_exists'] and
                not o['cuda_explicitly_disabled'] and o['disk_minimum_met'] and
                o['available_inodes'] > 0)
    if not admissible(before):
        raise ValueError('local preflight blocked; no manifest created')
    start = time.monotonic()
    data = manifest_bytes()
    after = observe()
    if not admissible(after):
        raise ValueError('local preflight changed; no manifest created')
    create_exclusive(path, data)
    return {'schema': 'e3_cell02_deployment_preflight_v1', 'before': before, 'after': after,
            'draft_elapsed_seconds': time.monotonic() - start,
            'manifest_path': str(path), 'manifest_sha256': hashlib.sha256(data).hexdigest(),
            'manifest_bytes': len(data), 'production_authorized': False,
            'scientific_admission': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('observe')
    p = sub.add_parser('generate')
    p.add_argument('--manifest-out', required=True)
    args = parser.parse_args()
    try:
        result = observe() if args.command == 'observe' else generate(args.manifest_out)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, RuntimeError) as exc:
        print(json.dumps({'error': str(exc), 'production_authorized': False,
                          'manifest_may_exist_after_write_failure': args.command == 'generate'}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
