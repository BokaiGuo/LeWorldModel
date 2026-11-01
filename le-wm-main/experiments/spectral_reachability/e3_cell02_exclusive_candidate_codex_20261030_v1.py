"""Cell02 candidate launcher: draft is read-only; run needs an externally pinned manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/gbk/code/LeWorldModel/le-wm-main')
RESULTS = ROOT/'results'
AUTHORITY = RESULTS/'e3_cell02_feedback_prepare_codex_20261029_v1_authority_LOCAL01.json'
AUTHORITY_SHA = '596014e1c53b4db19ba3dedc40877d814883e9f07b3b794aec80d7a28d387518'
CLAIM = RESULTS/'e3_cell02_exclusive_candidate_codex_20261030_v1_MAIN01.claim'
OUTPUT = RESULTS/'agent_e3_chunked_identity_host_20261029_v1_CELL02_MAIN01'
CONFIRM = 'CELL02_HOST_CUDA_COORDINATED_AND_AUTHORIZED'
SCHEMA = 'e3_cell02_exclusive_candidate_v1'

# Each child is a clean interpreter; neither imports this candidate launcher.
PRECHECK = r"""
import hashlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from experiments.spectral_reachability import e3_non03_host_readiness_codex_20261030_v1 as check
check.cell02()
root = pathlib.Path(sys.argv[1])
sources = {}
for name, mod in sorted(list(sys.modules.items())):
    filename = getattr(mod, '__file__', None)
    if not isinstance(filename, str) or not pathlib.Path(filename).is_absolute(): continue
    p = pathlib.Path(filename)
    q = p.resolve()
    if root not in q.parents: continue
    if p != q or not q.is_file() or q.is_symlink(): raise ValueError('noncanonical module: '+str(p))
    sources[str(q)] = hashlib.sha256(q.read_bytes()).hexdigest()
print(json.dumps(sources, sort_keys=True))
"""
CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as runner
result = runner.run(sys.argv[2], sys.argv[3], sys.argv[4], runner.launch.AUTH)
print(json.dumps(result, sort_keys=True), flush=True)
sys.exit(0 if result.get('complete') is True else 1)
"""
AUDIT = r"""
import hashlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as p
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as runner
out = pathlib.Path(sys.argv[2])
sha = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
rsha, csha = sha(out/'report.json'), sha(out/'completion.json')
result = p.consume(str(out), rsha, csha)
report = json.loads((out/'report.json').read_bytes())
events = [json.loads(line) for line in (out/'events.jsonl').read_bytes().splitlines()]
bind = [e for e in events if e['resource'] == 'launcher_authority_verified']
end = [e for e in events if e['resource'] == 'launcher_guarded_return']
fail = [e for e in events if e['resource'] == 'launcher_failure']
if not (result['complete'] is True and report['status'] == 'completed' and not fail
        and len(bind) == len(end) == 1 and bind[0]['detail']['cell'] == 2
        and bind[0]['detail']['manifest_sha256'] == sys.argv[3]
        and bind[0]['detail']['manifest_path'] == sys.argv[4]
        and end[0]['detail']['cell'] == 2):
    raise ValueError('completed transport and cell02 terminal required')
binding = runner.authority(sys.argv[4], sys.argv[3])
if binding['cell'] != 2:
    raise ValueError('cell02 authority required')
sources = dict(binding['prepared']['bindings_sha256'])
runner.launch.host.source.merge(sources, binding['additional_sources'])
census = runner.launch.module_census.saved(events, sources)
print(json.dumps({'report_sha256': rsha, 'completion_sha256': csha,
                  'transport_complete': True, 'census': census,
                  'scientific_admission': False}, sort_keys=True))
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def module_sources():
    sources = {}
    for mod in list(sys.modules.values()):
        name = getattr(mod, '__file__', None)
        if not isinstance(name, str) or not Path(name).is_absolute():
            continue
        path = Path(name)
        resolved = path.resolve()
        if ROOT not in resolved.parents:
            continue
        if path != resolved or not resolved.is_file() or resolved.is_symlink():
            raise ValueError('noncanonical parent module')
        sources[str(resolved)] = file_sha(resolved)
    if not sources:
        raise ValueError('empty parent source census')
    return sources


def check_environment():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('fixed project working directory required')
    if any(os.environ.get(key) for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP')):
        raise ValueError('uncontrolled Python import environment')
    if os.environ.get('PYTHONNOUSERSITE') != '1' or os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise ValueError('disable user site and bytecode')


def environment_sha():
    # Only the digest is published; no environment values are placed in the manifest.
    return sha(canonical(dict(os.environ)))


def precheck_sources():
    proc = subprocess.run([sys.executable, '-B', '-s', '-c', PRECHECK, str(ROOT)],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError('CPU precheck child failed: ' + proc.stderr[-2000:])
    result = json.loads(proc.stdout)
    if type(result) is not dict or not result:
        raise ValueError('invalid precheck census')
    return result


def draft():
    check_environment()
    if file_sha(AUTHORITY) != AUTHORITY_SHA:
        raise ValueError('authority drift')
    before = module_sources()
    sources = precheck_sources()
    if module_sources() != before:
        raise ValueError('parent census drift during precheck')
    executable = Path(sys.executable).resolve()
    return {'schema': SCHEMA, 'cell': 2, 'authority': str(AUTHORITY),
            'authority_sha256': AUTHORITY_SHA, 'claim': str(CLAIM), 'output': str(OUTPUT),
            'root': str(ROOT), 'python_invocation': sys.executable, 'python_executable': str(executable),
            'python_sha256': file_sha(executable), 'environment_sha256': environment_sha(),
            'parent_sources': before, 'precheck_sources': sources,
            'precheck_code_sha256': sha(PRECHECK.encode()), 'child_code_sha256': sha(CHILD.encode()),
            'audit_code_sha256': sha(AUDIT.encode()),
            'production_authorization': 'external_decision_required', 'scientific_admission': False}


def parse_manifest(path, expected_sha):
    data = Path(path).read_bytes()
    if sha(data) != expected_sha:
        raise ValueError('externally pinned manifest hash mismatch')
    manifest = json.loads(data)
    if data not in (canonical(manifest), canonical(manifest) + b'\n'):
        raise ValueError('manifest must be canonical JSON')
    return manifest


def claim_once(path, record):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        data = canonical(record)
        if stream.write(data) != len(data):
            raise OSError('short claim write')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_current(approved):
    if (module_sources() != approved['parent_sources'] or
        environment_sha() != approved['environment_sha256'] or
        sys.executable != approved['python_invocation'] or
        str(Path(sys.executable).resolve()) != approved['python_executable'] or
        file_sha(Path(sys.executable).resolve()) != approved['python_sha256']):
        raise ValueError('parent source, interpreter or environment changed')
    for name, expected in approved['precheck_sources'].items():
        p = Path(name)
        if not p.is_absolute() or p != p.resolve() or not p.is_file() or p.is_symlink() or file_sha(p) != expected:
            raise ValueError('precheck source changed: ' + name)


def run(manifest_path, manifest_sha, confirmation):
    if confirmation != CONFIRM:
        raise ValueError('explicit coordination and authorization confirmation required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        raise ValueError('CUDA explicitly disabled')
    approved = parse_manifest(manifest_path, manifest_sha)
    if approved != draft():
        raise ValueError('approved manifest differs from current sources or environment')
    if os.path.lexists(OUTPUT):
        raise FileExistsError('fixed output already exists')
    claim_once(CLAIM, {'schema': SCHEMA+'.claim', 'manifest_sha256': manifest_sha,
                       'authority_sha256': AUTHORITY_SHA, 'output': str(OUTPUT),
                       'pid': os.getpid(), 'meaning': 'attempt claimed, outcome unknown'})
    verify_current(approved)
    child = subprocess.run([sys.executable, '-B', '-s', '-c', CHILD, str(ROOT),
                            str(AUTHORITY), AUTHORITY_SHA, str(OUTPUT)], cwd=ROOT, check=False)
    if child.returncode:
        raise SystemExit(child.returncode if child.returncode > 0 else 128-child.returncode)
    verify_current(approved)
    audit = subprocess.run([sys.executable, '-B', '-s', '-c', AUDIT, str(ROOT), str(OUTPUT),
                            AUTHORITY_SHA, str(AUTHORITY)], cwd=ROOT, check=False)
    if audit.returncode:
        raise SystemExit(audit.returncode if audit.returncode > 0 else 128-audit.returncode)
    verify_current(approved)
    return {'transport_complete': True, 'scientific_admission': False, 'full_saved_strict': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('draft')
    p = sub.add_parser('run')
    p.add_argument('--manifest', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--confirm', required=True)
    args = parser.parse_args()
    if args.command == 'draft':
        print(canonical(draft()).decode())
    else:
        print(json.dumps(run(args.manifest, args.manifest_sha256, args.confirm), sort_keys=True))


if __name__ == '__main__':
    main()
