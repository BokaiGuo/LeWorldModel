"""Candidate E3 same-environment launcher. Not yet a full saved strict."""
import argparse
import os
from pathlib import Path
import time
import traceback

from experiments.spectral_reachability import agent_e3_authorized_episode_20261029_v2 as episode
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_launcher_loader_bridge_20261029_v1 as loader_bridge
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub
from experiments.spectral_reachability import agent_e3_reset_identity_20261029_v1 as reset_identity
from experiments.spectral_reachability import agent_e3_loaded_module_census_20261029_v1 as module_census

NAME = 'agent_e3_continuous_host_launcher_20261029_v1'
AUTH = episode.AUTH
need = pub.base.need
CONTROL5_PREPARE_SHA = 'ea8e4feae44e31876852c960db03ce4613184c55fc0745ef8f890f70485f8425'
RESOLVER_SHA = 'b44699e6a10bb4c257ed8872dc86b99a37508d7022850f8b4fcd257bcd232b26'


def _record(store, resource, detail):
    store.event('physical', 'record', resource, store.index, detail)


def _run_lifecycle(output, started, authority, perform):
    """Single publication and failure boundary; perform is a fixture seam.

    Authority is a zero-argument callable and is deliberately evaluated after
    the caller's monotonic start. It must not create a native environment.
    """
    store = pub.Store(output, started=started)
    _record(store, 'launcher_started', {'schema': NAME, 'scientific_admission': False})
    error = None
    try:
        binding = authority()
        store.check('physical_events', 0)
        _record(store, 'launcher_authority_verified', {'cell': binding['cell'],
                'manifest_path': binding['manifest_path'], 'manifest_sha256': binding['manifest_sha']})
        row = perform(binding, store)
        need(type(row) is dict and row.get('status') in
             ('completed20', 'native_stopped', 'native_stopped_warmup',
              'native_stopped_prefix', 'native_initial_success_control5'), 'terminal episode status')
        _record(store, 'launcher_guarded_return', {'cell': binding['cell'], 'status': row['status']})
    except BaseException as exc:
        error = exc
        details = dict(type=type(exc).__name__, message=str(exc)[:2048],
                       traceback=traceback.format_exc()[-8192:])
        try:
            _record(store, 'launcher_failure', details)
        except BaseException as secondary:
            exc.add_note('launcher failure event: ' + repr(secondary))
    try:
        result = store.publish('partial' if error else 'completed')
    except BaseException as secondary:
        if error is not None:
            error.add_note('publication failure: ' + repr(secondary))
            raise error
        raise
    if error is not None:
        return dict(publication=result, complete=False, error_type=type(error).__name__)
    return dict(publication=result, complete=result['complete'], error_type=None)


def read_authority(path, sha):
    """Read externally SHA-pinned run manifest, then original saved authorities."""
    p = host.read(path, sha)
    need(type(p) is dict and set(p) == {'schema','cell','control5','resolver','additional_sources'}
         and p['schema'] == NAME+'.authority' and type(p['cell']) is int and 0 <= p['cell'] < 16,
         'fixed authority manifest ABI')
    c = p['control5']; r = p['resolver']
    need(type(c) is dict and set(c) == {'prepare','report','completion','receipt'}
         and type(r) is dict and set(r) == {'path','sha256'}, 'explicit authority references')
    for item in list(c.values())+[r]:
        need(type(item) is dict and set(item) == {'path','sha256'} and type(item['path']) is str
             and Path(item['path']).is_absolute() and str(Path(item['path']).resolve()) == item['path']
             and type(item['sha256']) is str and len(item['sha256']) == 64
             and all(x in '0123456789abcdef' for x in item['sha256']), 'canonical pinned reference')
    args = tuple(v for name in ('prepare','report','completion','receipt')
                 for v in (c[name]['path'], c[name]['sha256'])) + (p['cell'],)
    prepared = host.read(args[0], args[1])
    need(args[1] == CONTROL5_PREPARE_SHA and r['sha256'] == RESOLVER_SHA
         and prepared['actual_resolver'] == r, 'exact MAIN control5 and resolver authorities')
    sources = dict(prepared['bindings_sha256'])
    host.source.merge(sources, p['additional_sources'])
    required = [str(Path(x.__file__).resolve()) for x in
                (episode, loader_bridge, pub, pub.base, host, reset_identity, module_census)] + [str(Path(__file__).resolve())]
    episode.old.guard.validate_sources(sources, required)
    value = episode.old.authority.load(*args)
    need(value['cell'] == p['cell'], 'authority cell identity')
    return dict(cell=p['cell'], args=args, resolver=r, additional_sources=p['additional_sources'],
                prepared=prepared, prepare_sha=args[1], manifest_path=str(Path(path)), manifest_sha=sha)


def _host_perform(binding, store):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from experiments.gpu_preflight import require_cuda_allocation
    gpu = require_cuda_allocation('cuda:0')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _record(store, 'launcher_gpu_preflight', {'preflight': gpu})
    model, model_sha = host.primitives.native.loader.loaded_model(
        binding['prepared'], torch, device, loader_bridge.LoaderBridge(store), binding['prepare_sha'])
    need(model_sha == binding['prepared']['expected_model_hash'], 'frozen model hash')
    row = dict(episode_id='cell%02d' % binding['cell'], warmup_returned=0,
               control_returned=0, executed_raw=[], PNG=0, checkpoints=[], status='started')
    sources=dict(binding['prepared']['bindings_sha256'])
    host.source.merge(sources,binding['additional_sources'])
    module_census.record(store,'before',module_census.snapshot(sources))
    primary=None
    try:
        result = reset_identity.run(binding['args'], binding['resolver']['path'], binding['resolver']['sha256'],
                                    model, torch, device, store, row, binding['additional_sources'], AUTH)
    except BaseException as exc:
        primary=exc
        raise
    finally:
        try:module_census.record(store,'after',module_census.snapshot(sources))
        except BaseException as exc:
            if primary is None:raise
            primary.add_note('post-cleanup module census: '+repr(exc))
    need(host.sha(binding['manifest_path']) == binding['manifest_sha'], 'authority manifest unchanged')
    return result


def run(authority_path, authority_sha, output, authorization):
    need(authorization == AUTH, 'explicit HOST CUDA authorization')
    started = time.monotonic()
    return _run_lifecycle(output, started, lambda: read_authority(authority_path, authority_sha), _host_perform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    print(host.raw(run(args.authority, args.authority_sha256, args.output, args.authorization)).decode())


if __name__ == '__main__':
    main()
