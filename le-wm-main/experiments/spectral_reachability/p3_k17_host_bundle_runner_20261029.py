"""K17 bounded host runner with pinned prepare and explicit run confirmation.

Prepare/consume do not create an environment. Physical run is explicit and is
not a production or scientific authorization.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p3_c3_k17_bounded_store_20261029_v1 as bounds
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as meter
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_native_store_20261029_v1 as native_store
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import p3_k17_live_source_budget_seam_20261029 as seam
from experiments.spectral_reachability import p3_k17_saved_source_budget_consumer_20261029 as consumer
from experiments.spectral_reachability import p3_k17_saved_native_guard_frames_20261029 as strict
from experiments.spectral_reachability import p3_k17_native_outer_scoped_20261029 as scoped_outer

MAX_METADATA = 2 * 1024 * 1024
CONFIRM = 'P3_K17_KNOWN2_SIX_SLOT_FIRST5_PHYSICAL_ONLY'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bundle_codes():
    parent = Path(__file__).resolve().parent
    names = ('p3_k17_host_bundle_runner_20261029.py', 'p3_k17_saved_source_budget_consumer_20261029.py',
             'p3_k17_saved_native_guard_frames_20261029.py', 'p3_k17_native_outer_scoped_20261029.py')
    return {name: digest(consumer.file_bytes(parent/name)) for name in names}


def write_new(path, value, bundle):
    raw = transport.canonical(value) + b'\n'
    if len(raw) > MAX_METADATA:
        raise ValueError('bundle metadata limit')
    entries = list(Path(bundle).rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('bundle symlink')
    size = sum(p.stat().st_size for p in entries if p.is_file())
    if size + len(raw) > bounds.CAP or shutil.disk_usage(bundle).free < bounds.FLOOR + len(raw):
        raise ValueError('bundle cap or floor')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short write')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _native_factory():
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401 -- native registration
    return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224,
                    with_target=False, render_action=False)


def environment_evidence(env):
    body = env.unwrapped
    outer.h3.validate_env(body)
    if body.render_action is not False or body.render_size != 224 or body.render_mode != 'rgb_array':
        raise ValueError('native render contract')
    cls = type(body)
    path = inspect.getsourcefile(cls)
    if not path:
        raise ValueError('native source file missing')
    file = Path(path)
    if file.is_symlink() or not file.is_file():
        raise ValueError('native source file invalid')
    spec = getattr(env, 'spec', None)
    return {'env_id': getattr(spec, 'id', None), 'class_module': cls.__module__,
            'class_qualname': cls.__qualname__, 'class_source_sha256': digest(file.read_bytes()),
            'render_mode': body.render_mode, 'render_size': body.render_size,
            'render_action': body.render_action}


def prepare():
    """Read pinned saved inputs and native source without constructing env."""
    native = outer.h3.moving
    path = Path(native.ENV)
    if path.is_symlink() or not path.is_file() or digest(path.read_bytes()) != native.ENV_SHA:
        raise ValueError('original native environment source mismatch')
    data, rows = outer.material()
    return {'schema': 'p3_k17_host_prepare.v1',
            'source_files_sha256': consumer.sources(), 'guard_source_sha256': strict.source_files(),
            'bundle_code_sha256': bundle_codes(), 'source_sha256': seam.source_digest(data, rows),
            'caps': dict(meter.CAPS), 'wall_seconds': meter.WALL, 'order': [list(x) for x in outer.ORDER],
            'environment': {'env_id': 'swm/PushT-v1', 'class_module': 'stable_worldmodel.envs.pusht.env',
                            'class_qualname': 'PushT', 'class_source_sha256': native.ENV_SHA,
                            'render_mode': 'rgb_array', 'render_size': 224, 'render_action': False},
            'scope': CONFIRM, 'physical_authorized': False, 'scientific_pass': False}


def _admission(authority, authority_sha256, confirmation):
    if (confirmation != CONFIRM or authority.get('schema') != 'p3_k17_host_prepare.v1' or
        authority.get('scope') != CONFIRM or authority.get('physical_authorized') is not False or
        authority.get('scientific_pass') is not False or digest(transport.canonical(authority)) != authority_sha256):
        raise ValueError('explicit P3 scope and prepared authority required')


def run_host(directory, authority, authority_sha256, confirmation):
    """Explicit research-only native execution, subject to fixed source/budget."""
    return _execute_admitted(directory, authority, authority_sha256, _native_factory, confirmation=confirmation)


def _execute_admitted(directory, authority, authority_sha256, factory, *, confirmation=None):
    """Execution core; every caller must supply the fixed explicit scope."""
    _admission(authority, authority_sha256, confirmation)
    path = Path(directory)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ValueError('exclusive bundle path')
    if digest(transport.canonical(authority)) != authority_sha256:
        raise ValueError('authority SHA mismatch')
    if (authority.get('source_files_sha256') != consumer.sources() or
        authority.get('guard_source_sha256') != strict.source_files() or
        authority.get('bundle_code_sha256') != bundle_codes() or
        authority.get('caps') != meter.CAPS or authority.get('wall_seconds') != meter.WALL or
        authority.get('order') != [list(x) for x in outer.ORDER]):
        raise ValueError('authority source/budget contract')
    path.mkdir()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        native = native_store.NativeStore(path/'native')
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        write_new(path/'failure.json', {'schema': 'p3_k17_pre_store_failure.v1', 'error': failure}, path)
        raise
    environments = []
    try:
        bound = seam.SourceBoundStore.from_saved(native)
        if bound.source_sha256 != authority.get('source_sha256'):
            raise ValueError('authority data source mismatch')
        native.inner.put('execution_binding.json', {'authority_sha256': authority_sha256,
                         'source_sha256': bound.source_sha256, 'source_files_sha256': consumer.sources(),
                         'bundle_code_sha256': bundle_codes(),
                         'guard_source_sha256': strict.source_files(),
                         'caps': dict(meter.CAPS), 'wall_seconds': meter.WALL,
                         'expected_environment': authority['environment'],
                         'production_authorized': False})

        def validate_instance(env):
            evidence = environment_evidence(env)
            environments.append(evidence)
            if evidence != authority['environment']:
                raise ValueError('native environment pin mismatch')

        result = scoped_outer.execute_material(bound.data, bound.rows, bound, factory,
                                                validate_instance=validate_instance)
        if (len(environments) != result['counts']['constructor']['completed'] or
                any(e != authority['environment'] for e in environments)):
            raise ValueError('native environment pin mismatch')
        if consumer.sources() != authority['source_files_sha256'] or bundle_codes() != authority['bundle_code_sha256']:
            raise ValueError('source changed during execution')
        report = {'schema': 'p3_k17_host_report.v1', 'outer': result,
                  'seam': bound.summary(), 'environment_evidence': environments,
                  'authority_sha256': authority_sha256,
                  'production_authorized': False, 'scientific_pass': False}
        native.inner.put('report.json', report)
        journal_sha = digest((native.path/'journal.jsonl').read_bytes())
        audit = consumer.consume(native.path, journal_sha, authority['source_files_sha256'])
        guard_audit = strict.audit(native.path, journal_sha, authority['guard_source_sha256'], authority['source_sha256'])
        for axis, values in audit['counts'].items():
            if values['attempted'] != result['counts'][axis]['attempted'] or values['completed'] != result['counts'][axis]['completed']:
                raise ValueError('saved/live count mismatch')
        for row, saved in zip(result['rows'], audit['branches']):
            if row['index'] != saved['index'] or row['status'] != saved['status']:
                raise ValueError('saved/live branch mismatch')
        native.inner.put('completion.json', {'schema': 'p3_k17_native_audit.v1',
                         'journal_sha256': journal_sha, 'audit': audit, 'guard_frames_audit': guard_audit,
                         'production_authorized': False})
        failure = None
        partial = bool(result['errors'] or bound.fault or not audit['six_slot_event_coverage'] or
                       guard_audit['guards_recomputed'] != 6 or guard_audit['frames_checked'] != 306)
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        try:
            native.inner.fail({'schema': 'p3_k17_bundle_failure.v1', 'error': failure})
        except BaseException:
            pass
    files = {}
    for item in sorted(native.path.iterdir()):
        if not item.is_file() or item.is_symlink():
            raise ValueError('unexpected native bundle entry')
        files['native/'+item.name] = digest(item.read_bytes())
    manifest = {'schema': 'p3_k17_host_manifest.v1', 'files': files,
                'authority_sha256': authority_sha256, 'failure': failure,
                'production_authorized': False}
    write_new(path/'manifest.json', manifest, path)
    completion = {'schema': 'p3_k17_host_completion.v1',
                  'manifest_sha256': digest((path/'manifest.json').read_bytes()),
                  'status': 'failed' if failure else ('saved_incomplete' if partial else 'saved_engineering_only'),
                  'production_authorized': False, 'scientific_pass': False}
    write_new(path/'completion.json', completion, path)
    return {'completion_sha256': digest((path/'completion.json').read_bytes()), 'completion': completion}


def _canonical_file(path):
    raw = consumer.file_bytes(path)
    value = json.loads(raw)
    if raw != transport.canonical(value) + b'\n':
        raise ValueError('noncanonical bundle JSON')
    return value


def consume_bundle(directory, completion_sha256, authority, authority_sha256):
    """Verify externally pinned bundle and rerun saved consumer when available."""
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('bundle directory')
    if digest(transport.canonical(authority)) != authority_sha256:
        raise ValueError('external authority SHA')
    if (authority.get('source_files_sha256') != consumer.sources() or
        authority.get('guard_source_sha256') != strict.source_files() or
        authority.get('bundle_code_sha256') != bundle_codes() or authority.get('caps') != meter.CAPS or
        authority.get('wall_seconds') != meter.WALL or authority.get('order') != [list(x) for x in outer.ORDER]):
        raise ValueError('external authority source/budget')
    if digest(consumer.file_bytes(path/'completion.json')) != completion_sha256:
        raise ValueError('external completion SHA')
    completion = _canonical_file(path/'completion.json')
    if (set(completion) != {'schema', 'manifest_sha256', 'status', 'production_authorized', 'scientific_pass'} or
        completion['schema'] != 'p3_k17_host_completion.v1' or
        completion['status'] not in ('failed', 'saved_incomplete', 'saved_engineering_only') or
        completion['production_authorized'] is not False or completion['scientific_pass'] is not False):
        raise ValueError('completion contract')
    if digest(consumer.file_bytes(path/'manifest.json')) != completion['manifest_sha256']:
        raise ValueError('manifest SHA')
    manifest = _canonical_file(path/'manifest.json')
    if (set(manifest) != {'schema', 'files', 'authority_sha256', 'failure', 'production_authorized'} or
        manifest['schema'] != 'p3_k17_host_manifest.v1' or manifest['authority_sha256'] != authority_sha256 or
        manifest['production_authorized'] is not False or
        (completion['status'] == 'failed') != (manifest['failure'] is not None)):
        raise ValueError('manifest contract')
    native = path/'native'
    if native.is_symlink() or not native.is_dir():
        raise ValueError('native directory')
    entries = list(native.iterdir())
    if any(p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError('unexpected native entry')
    files = {'native/'+p.name: digest(p.read_bytes()) for p in entries}
    if files != manifest['files'] or set(p.name for p in path.iterdir()) != {'native', 'manifest.json', 'completion.json'}:
        raise ValueError('bundle inventory mismatch')
    audit = None
    if completion['status'] != 'failed':
        binding = _canonical_file(native/'execution_binding.json')
        saved = _canonical_file(native/'completion.json')
        report = _canonical_file(native/'report.json')
        if (binding['authority_sha256'] != authority_sha256 or binding['source_sha256'] != authority['source_sha256'] or
            binding['source_files_sha256'] != authority['source_files_sha256'] or binding['caps'] != meter.CAPS or
            binding['bundle_code_sha256'] != authority['bundle_code_sha256'] or bundle_codes() != authority['bundle_code_sha256'] or
            binding['guard_source_sha256'] != authority['guard_source_sha256'] or
            binding['wall_seconds'] != meter.WALL or binding['expected_environment'] != authority['environment'] or
            binding['production_authorized'] is not False or saved['schema'] != 'p3_k17_native_audit.v1' or
            saved['production_authorized'] is not False or report['production_authorized'] is not False or
            report['scientific_pass'] is not False or report['authority_sha256'] != authority_sha256):
            raise ValueError('saved binding/report')
        if any(e != authority['environment'] for e in report['environment_evidence']):
            raise ValueError('saved environment binding')
        audit = consumer.consume(native, saved['journal_sha256'], authority['source_files_sha256'])
        if audit != saved['audit']:
            raise ValueError('saved consumer mismatch')
        guard_audit = strict.audit(native, saved['journal_sha256'], authority['guard_source_sha256'], authority['source_sha256'])
        if guard_audit != saved['guard_frames_audit']:
            raise ValueError('saved guard/frame consumer mismatch')
        if (audit['source_sha256'] != authority['source_sha256'] or report['outer']['fixed_denominator'] != 6 or
            report['outer']['production_authorized'] is not False or report['outer']['physical_release'] is not False):
            raise ValueError('saved source/outer contract')
        rows = report['outer']['rows']
        if len(rows) != 6 or any(row['index'] != i for i, row in enumerate(rows)):
            raise ValueError('saved row roster')
        if any(rows[i]['status'] != b['status'] for i, b in enumerate(audit['branches'])) or any(
                row['status'] != 'not_started' for row in rows[len(audit['branches']):]):
            raise ValueError('saved row mismatch')
        incomplete = bool(report['outer']['errors'] or report['seam']['fault'] or not audit['six_slot_event_coverage'] or
                          guard_audit['guards_recomputed'] != 6 or guard_audit['frames_checked'] != 306)
        if (completion['status'] == 'saved_incomplete') != incomplete:
            raise ValueError('completion status mismatch')
        if len(report['environment_evidence']) != audit['counts']['constructor']['completed']:
            raise ValueError('environment count mismatch')
        for axis, values in audit['counts'].items():
            if (values['attempted'], values['completed']) != (report['outer']['counts'][axis]['attempted'], report['outer']['counts'][axis]['completed']):
                raise ValueError('saved count mismatch')
    return {'status': completion['status'], 'audit': audit,
            'guard_frames_audit': guard_audit if audit is not None else None, 'production_authorized': False,
            'native_provenance_verified': False, 'scientific_pass': False}


def _prepared(path, file_sha):
    raw = consumer.file_bytes(path)
    if digest(raw) != file_sha:
        raise ValueError('prepared external file SHA')
    value = json.loads(raw)
    if raw != transport.canonical(value) + b'\n':
        raise ValueError('prepared canonical encoding')
    return value, digest(transport.canonical(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', required=True)
    for name in ('run', 'consume'):
        p = sub.add_parser(name)
        p.add_argument('--prepared', required=True)
        p.add_argument('--prepare-sha256', required=True)
        p.add_argument('--bundle', required=True)
        if name == 'run':
            p.add_argument('--confirm', required=True)
        else:
            p.add_argument('--completion-sha256', required=True)
            p.add_argument('--receipt', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        authority = prepare()
        output = Path(args.output)
        raw = transport.canonical(authority) + b'\n'
        with output.open('xb') as stream:
            if stream.write(raw) != len(raw):
                raise OSError('short write')
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        result = {'prepare_sha256': digest(raw), 'authority_sha256': digest(transport.canonical(authority)),
                  'physical_executed': False}
    else:
        authority, anchor = _prepared(args.prepared, args.prepare_sha256)
        if args.command == 'run':
            result = run_host(args.bundle, authority, anchor, args.confirm)
        else:
            result = consume_bundle(args.bundle, args.completion_sha256, authority, anchor)
            output = Path(args.receipt)
            raw = transport.canonical(result) + b'\n'
            with output.open('xb') as stream:
                if stream.write(raw) != len(raw):
                    raise OSError('short write')
                stream.flush()
                os.fsync(stream.fileno())
            fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            result = {'receipt_sha256': digest(raw), 'status': result['status']}
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
