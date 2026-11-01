"""Exact inherited-pygame origin repair; unchanged v2 rendering/strict kernel."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

from experiments.spectral_reachability import agent_p5_renderer_factorial_20261028_v2 as native

NAME = 'agent_p5_renderer_factorial_20261028_v3'
ROOT = native.ROOT
require, sha, read, write = native.require, native.sha, native.read_json, native.write_new
SOURCE_SHA = 'e1fdcd06e058570970431f8b7716dfcc8e92f082970d414ecef1d12c9fd74d4b'
PREP = ROOT/'results/agent_p5_renderer_factorial_20261028_v2_prepare_MAIN01.json'
PREP_SHA = 'a52479877a62b776d2f857b7cd39848931726307ef52d178704a05af109fb406'
FAILURE = ROOT/'results/agent_p5_renderer_factorial_20261028_v2_render_MAIN01/report.json'
FAILURE_SHA = 'a0ad0280f11c8754018da4496115e0852c7b038c112a334771b7a2068e1bb751'


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md']


def build():
    require(sha(native.__file__) == SOURCE_SHA, 'frozen original renderer kernel')
    p = read(PREP, PREP_SHA)
    native.verify(p)
    failure = read(FAILURE, FAILURE_SHA)
    require(failure['global_error'] == 'ValueError: original environment and overlay provenance'
            and failure['meter'] is None and all(not r['shots'] for r in failure['rows']),
            'actual pre-dispatch engineering failure, never a physical negative')
    runtime = read(p['runtime_manifest'])
    record = runtime['inherited']['pygame']
    require(record == dict(site=str(ROOT/'.venv/lib/python3.12/site-packages'), version='2.6.1'),
            'original inherited pygame distribution, no arbitrary search path')
    origins = dict(environment=str(ROOT/native.parent.ENV),
                   pymunk=str(ROOT/'.jepa-physics-site/pymunk/__init__.py'),
                   pygame=str(Path(record['site'])/'pygame/__init__.py'))
    bindings = {**p['bindings'], str(PREP): PREP_SHA, str(FAILURE): FAILURE_SHA}
    bindings.update({path: sha(path) for path in origins.values()})
    bindings.update({str(path): sha(path) for path in own_files()})
    return dict(schema=NAME+'.prepare', native_manifest=p, native_prepare_sha256=PREP_SHA,
                origins=origins, pygame_distribution=record, bindings=bindings,
                repair='pygame is inherited from sealed base_site; pymunk remains exact overlay',
                protocol_unchanged=True, scientific_admission=False, neural_calls=0)


def verify(path, expected):
    p = read(path, expected)
    require(all(sha(f) == h for f, h in p['bindings'].items()), 'complete source/input origin closure')
    require(p == build(), 'complete deterministic repair prepare')
    return p


def check_origins(p, environment, pymunk, pygame, probe):
    observed = dict(environment=str(Path(environment).resolve()),
                    pymunk=str(Path(pymunk).resolve()), pygame=str(Path(pygame).resolve()))
    require(observed == p['origins'], 'exact bound module origins, not arbitrary base/overlay modules')
    require(probe['versions']['pygame'] == p['pygame_distribution'], 'sealed pygame distribution receipt')
    for kind, path in observed.items():
        imported_sha = (probe['native_source_sha256'] if kind == 'environment'
                        else probe['imported_dependency_hashes'].get(path))
        require(imported_sha == p['bindings'][path]
                and sha(path) == p['bindings'][path], 'actual imported file SHA and predeclared origin')


def load_runtime(p):
    require(os.environ.get('PYTHONPATH') == str(ROOT/'.jepa-physics-site')
            and os.environ.get('PYTHONNOUSERSITE') == '1', 'original isolated physics overlay')
    require(not any(k == 'torch' or k.startswith('torch.') for k in sys.modules), 'no neural imports')
    base = importlib.import_module('experiments.spectral_reachability.p5_jepa_factual_pilot_20261027_v1')
    require(sha(base.__file__) == native.parent.PIN[native.parent.PILOT], 'original public-state reader')
    runtime, old = base.load_runtime(ROOT)
    probe = runtime.probe(ROOT, old, Path(p['native_manifest']['runtime_manifest']))
    require(probe['status'] == 'native_import_ready_physics_not_executed', 'original full runtime probe')
    source = Path(p['origins']['environment'])
    spec = importlib.util.spec_from_file_location(NAME+'_native', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import pymunk
    import pygame
    check_origins(p, module.PushTEnv.__init__.__code__.co_filename, pymunk.__file__, pygame.__file__, probe)
    require(not any(k == 'torch' or k.startswith('torch.') for k in sys.modules), 'no neural imports after native loader')
    return module.PushTEnv, pymunk, pygame, base, probe


def output_path(path):
    path = Path(path).absolute()
    require(path.resolve() == path and path.parent == ROOT/'results'
            and path.name.startswith(NAME+'_') and not path.exists(), 'exclusive own results child')
    return path


def launch_binding(manifest, manifest_sha, out):
    return dict(schema=NAME+'.launch', manifest=str(Path(manifest).absolute()), manifest_sha256=manifest_sha,
                output=str(Path(out).absolute()), kernel_source_sha256=SOURCE_SHA, neural_calls=0)


def run(manifest, manifest_sha, out, authorized):
    require(authorized, 'explicit MAIN static-render authorization before any runtime access')
    p = verify(manifest, manifest_sha)
    out = output_path(out)
    write(Path(str(out)+'.launch.json'), launch_binding(manifest, manifest_sha, out))
    require(signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
            and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'no existing signal timer')
    def timeout(*args):
        raise TimeoutError('600s static render cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(native.WALL)
    try:
        return native.execute(p['native_manifest'], manifest_sha, out,
                              lambda: load_runtime(p), lambda: verify(manifest, manifest_sha))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


def consume(manifest, expected, report, report_sha):
    p = verify(manifest, expected)
    report = Path(report).absolute()
    require(read(Path(str(report.parent)+'.launch.json')) == launch_binding(manifest, expected, report.parent),
            'exact new runner execution/prepare/output edge')
    receipt = native.consume(p['native_manifest'], expected, report, report_sha)
    probe = read(report.parent/'runtime_probe.json')
    check_origins(p, *[p['origins'][k] for k in ('environment', 'pymunk', 'pygame')], probe)
    return dict(schema=NAME+'.consume', status=receipt['status'], native_consumer=receipt,
                manifest_sha256=expected, report_sha256=report_sha, origins=p['origins'],
                protocol_unchanged=True, neural_calls=0, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--report-sha256')
    parser.add_argument('--main-authorized', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare':
        output_path(a.manifest)
        write(a.manifest, build())
        result = dict(status='prepared', sha256=sha(a.manifest))
    elif a.command == 'verify':
        verify(a.manifest, a.manifest_sha256)
        result = dict(status='verified')
    elif a.command == 'run':
        require(a.output is not None, 'output required')
        result = run(a.manifest, a.manifest_sha256, a.output, a.main_authorized)
    else:
        require(a.output and a.report and a.report_sha256, 'receipt and original report required')
        result = consume(a.manifest, a.manifest_sha256, a.report, a.report_sha256)
        output_path(a.output)
        write(a.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'sha256') if k in result}))
    return 2 if result['status'] == 'static_render_failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
