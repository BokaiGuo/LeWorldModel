"""CPU startup probe, independently sealed closure and pre-CUDA HOST entry."""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host

launch = host.launch
base = launch.pub.base
need = host.need
NAME = 'agent_e3_startup_closure_20261029_v1'
ROOT = launch.module_census.ROOT


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def inventory(modules=None, root=ROOT):
    modules = sys.modules if modules is None else modules
    entries = []; hashes = {}
    for name, mod in sorted(list(modules.items())):
        path = getattr(mod, '__file__', None)
        if not isinstance(name, str) or not isinstance(path, str) or not Path(path).is_absolute(): continue
        p = Path(path); resolved = p.resolve()
        if root not in resolved.parents: continue
        need(p == resolved and p.is_file() and not p.is_symlink(), 'canonical loaded module')
        key = str(p)
        if key not in hashes: hashes[key] = file_hash(p)
        entries.append(dict(module=name, path=key, sha256=hashes[key]))
    need(entries, 'nonempty startup inventory')
    return entries


def probe(binding):
    """Actual configuration construction and checkpoint read, CPU only, no forward."""
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from hydra.utils import instantiate
    from experiments import checkpoint_compat as compat
    from experiments.gpu_preflight import require_cuda_allocation  # import only
    del require_cuda_allocation
    prepared = binding['prepared']; blobs = {}
    for name in ('config.json', 'weights.pt'):
        p = Path(prepared['policy']) / name
        need(p.is_file() and not p.is_symlink(), 'regular model input')
        data = p.read_bytes()
        need(hashlib.sha256(data).hexdigest() == prepared['bindings_sha256'][str(p)], 'sealed model input')
        blobs[name] = data
    model = instantiate(json.loads(blobs['config.json']))
    state = torch.load(io.BytesIO(blobs['weights.pt']), map_location='cpu', weights_only=True)
    mapped, _ = compat.remap_legacy_vit_keys(state)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    need(not missing and not unexpected, 'exact CPU checkpoint keys')
    need(all(p.device.type == 'cpu' for p in model.parameters()), 'CPU-only model')
    entries = inventory()
    del model, state, mapped, blobs
    gc.collect()
    return entries


def write_new(path, value):
    p = Path(path)
    need(p.is_absolute() and str(p.resolve()) == str(p) and p.parent.is_dir() and not p.exists() and not p.is_symlink(), 'exclusive canonical output')
    data = launch.host.raw(value); need(len(data) <= launch.host.META, 'bounded metadata')
    with p.open('xb') as f:
        need(f.write(data) == len(data), 'complete write'); f.flush(); os.fsync(f.fileno())
    base.syncdir(p.parent)
    return hashlib.sha256(data).hexdigest()


def collect(source, sha, output):
    binding = host.authority(source, sha)
    entries = probe(binding)
    need(launch.host.sha(source) == sha, 'source authority unchanged')
    value = dict(schema=NAME+'.closure', source=dict(path=source, sha256=sha),
                 python=str(Path(sys.executable).resolve()), entries=entries,
                 scope='cpu_config_checkpoint_loaded_project_modules', cuda_executed=False, physics_executed=False)
    digest = write_new(output, value)
    return dict(path=output, sha256=digest, module_count=len(entries), unique_paths=len({x['path'] for x in entries}), scientific_admission=False)


def read_closure(path, sha, source, source_sha):
    value = launch.host.read(path, sha)
    need(type(value) is dict and set(value) == {'schema','source','python','entries','scope','cuda_executed','physics_executed'}
         and value['schema'] == NAME+'.closure' and value['source'] == dict(path=source, sha256=source_sha)
         and value['python'] == str(Path(sys.executable).resolve()) and value['scope'] == 'cpu_config_checkpoint_loaded_project_modules'
         and value['cuda_executed'] is False and value['physics_executed'] is False, 'closure schema and source')
    entries = value['entries']; need(type(entries) is list and bool(entries), 'closure entries')
    names = []; pins = {}
    for x in entries:
        need(type(x) is dict and set(x) == {'module','path','sha256'} and type(x['module']) is str and x['module'], 'entry schema')
        p = Path(x['path']); need(p.is_absolute() and p == p.resolve() and ROOT in p.parents and p.is_file() and not p.is_symlink(), 'entry path')
        need(type(x['sha256']) is str and len(x['sha256']) == 64, 'entry hash')
        old = pins.setdefault(str(p), x['sha256']); need(old == x['sha256'], 'consistent path hash')
        names.append(x['module'])
    need(names == sorted(set(names)), 'ordered unique modules')
    for p, expected in pins.items(): need(file_hash(p) == expected, 'closure source drift: '+p)
    return value, pins


def build(source, source_sha, closure, closure_sha):
    binding = host.authority(source, source_sha)
    _, pins = read_closure(closure, closure_sha, source, source_sha)
    value = launch.host.read(source, source_sha)
    value = dict(value, additional_sources=dict(value['additional_sources']))
    for p, digest in pins.items():
        prior = binding['prepared']['bindings_sha256'].get(p, value['additional_sources'].get(p))
        need(prior is None or prior == digest, 'conflicting source pin')
        if p not in binding['prepared']['bindings_sha256']: value['additional_sources'][p] = digest
    p = str(Path(__file__).resolve()); digest = file_hash(p)
    prior = binding['prepared']['bindings_sha256'].get(p, value['additional_sources'].get(p))
    need(prior is None or prior == digest, 'probe source conflict')
    if p not in binding['prepared']['bindings_sha256']: value['additional_sources'][p] = digest
    need(launch.host.sha(source) == source_sha and launch.host.sha(closure) == closure_sha, 'stable authority inputs')
    return value


def prepare(source, source_sha, closure, closure_sha, output):
    value = build(source, source_sha, closure, closure_sha)
    sha = write_new(output, value)
    verify(source, source_sha, closure, closure_sha, output, sha)
    return dict(path=output, sha256=sha, scientific_admission=False)


def verify(source, source_sha, closure, closure_sha, output, output_sha):
    need(launch.host.read(output, output_sha) == build(source, source_sha, closure, closure_sha), 'authority reconstruction')
    host.authority(output, output_sha)
    return dict(verified=True, scientific_admission=False)


def precheck(source, source_sha, closure, closure_sha, authority, authority_sha):
    verify(source, source_sha, closure, closure_sha, authority, authority_sha)
    sealed, pins = read_closure(closure, closure_sha, source, source_sha)
    actual = probe(host.authority(source, source_sha))
    modules = {x['module']: (x['path'], x['sha256']) for x in sealed['entries']}
    for entry in actual:
        need(pins.get(entry['path']) == entry['sha256'] and
             modules.get(entry['module']) == (entry['path'], entry['sha256']),
             'new or changed startup dependency: '+entry['path'])
    verify(source, source_sha, closure, closure_sha, authority, authority_sha)
    return dict(verified=True, startup_modules=len(actual), cuda_executed=False, physics_executed=False,
                scientific_admission=False)


def run(source, source_sha, closure, closure_sha, authority, authority_sha, output, authorization):
    need(authorization == launch.AUTH, 'explicit HOST CUDA authorization')
    precheck(source, source_sha, closure, closure_sha, authority, authority_sha)
    return host.run(authority, authority_sha, output, authorization)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    for command in ('collect','prepare','verify','precheck','run'):
        x = sub.add_parser(command)
        for opt in ('source','source-sha256'): x.add_argument('--'+opt, required=True)
        if command != 'precheck': x.add_argument('--output', required=True)
        if command != 'collect':
            for opt in ('closure','closure-sha256'): x.add_argument('--'+opt, required=True)
        if command == 'verify': x.add_argument('--output-sha256', required=True)
        if command in ('precheck', 'run'):
            for opt in ('authority','authority-sha256'): x.add_argument('--'+opt, required=True)
        if command == 'run': x.add_argument('--authorization', required=True)
    a = p.parse_args()
    if a.command == 'collect': result = collect(a.source, a.source_sha256, a.output)
    elif a.command == 'prepare': result = prepare(a.source, a.source_sha256, a.closure, a.closure_sha256, a.output)
    elif a.command == 'verify': result = verify(a.source, a.source_sha256, a.closure, a.closure_sha256, a.output, a.output_sha256)
    elif a.command == 'precheck': result = precheck(a.source, a.source_sha256, a.closure, a.closure_sha256, a.authority, a.authority_sha256)
    else: result = run(a.source, a.source_sha256, a.closure, a.closure_sha256, a.authority, a.authority_sha256, a.output, a.authorization)
    print(launch.host.raw(result).decode())


if __name__ == '__main__': main()
