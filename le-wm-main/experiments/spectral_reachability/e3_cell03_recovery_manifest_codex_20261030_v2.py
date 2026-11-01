"""CPU-only candidate recovery manifest. No production entry is enabled."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from experiments.spectral_reachability import e3_cell03_recovery_acceptance_codex_20261030_v1 as prior
from experiments.spectral_reachability import e3_cell03_host_cpu_precheck_codex_20261030_v1 as precheck
from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock

ROOT = prior.ROOT
ATTEMPT = 'CELL03_RECOVERY01_PROPOSED'
LOCK = ROOT / 'results/e3_p2_host_cuda_cooperative.lock'
MANIFEST = ROOT / 'results/e3_cell03_recovery01_manifest_codex_20261030_v2.json'


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def loaded_project_files():
    """Snapshot loaded module files, not transient imports or system libraries."""
    result = {}
    for name, module in sorted(list(sys.modules.items())):
        filename = getattr(module, '__file__', None)
        if not isinstance(filename, str):
            continue
        path = Path(filename)
        if not path.is_absolute():
            continue
        resolved = path.resolve()
        if ROOT not in resolved.parents:
            continue
        if path != resolved or not path.is_file():
            raise ValueError('noncanonical project module: ' + str(path))
        result[name] = {'path': str(path), 'sha256': prior.file_sha(path)}
    if not result:
        raise ValueError('empty module inventory')
    return result


def build():
    previous = prior.proposal()
    precheck.check(str(prior.parent.AUTHORITY), prior.parent.AUTHORITY_SHA)
    binding = precheck.host.authority(str(prior.parent.AUTHORITY), prior.parent.AUTHORITY_SHA)
    sources = dict(binding['prepared']['bindings_sha256'])
    precheck.host.launch.host.source.merge(sources, binding['additional_sources'])
    precheck.host.launch.episode.old.guard.validate_sources(sources, list(sources))
    executable = Path(sys.executable).resolve(strict=True)
    return {
        'schema': 'e3_cell03_recovery_manifest_candidate_v2',
        'attempt': ATTEMPT,
        'predecessor_proposal': previous,
        'runner_source_bindings': sources,
        'parent_loaded_project_files': loaded_project_files(),
        'launcher_sha256': prior.file_sha(Path(__file__).resolve()),
        'lock_module_sha256': prior.file_sha(Path(lock.__file__).resolve()),
        'python_executable': str(executable),
        'python_sha256': prior.file_sha(executable),
        'python_version': sys.version,
        'cwd': str(Path.cwd().resolve()),
        'environment_sha256': digest(raw(dict(os.environ))),
        'lock_path': str(LOCK),
        'source_scope': 'authority bindings and parent loaded project files snapshot',
        'transient_imports_or_system_native_libraries_proved': False,
        'approved': False,
        'production_enabled': False,
    }


def write_exclusive(path, body):
    data = raw(body)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.command == 'run':
        raise SystemExit('blocked: no reviewed launcher closure and no P2 GPU coordination')
    if Path.cwd().resolve() != ROOT:
        raise SystemExit('prepare must run from project root')
    body = build()
    print(json.dumps({'manifest': str(MANIFEST), 'sha256': write_exclusive(MANIFEST, body),
                      'approved': False, 'production_enabled': False}, sort_keys=True))


if __name__ == '__main__':
    main()
