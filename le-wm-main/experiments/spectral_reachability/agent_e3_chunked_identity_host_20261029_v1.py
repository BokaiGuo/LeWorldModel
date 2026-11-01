"""Chunk-bound E3 HOST candidate. No execution occurs on import."""
import argparse
import os
from pathlib import Path
import time

from experiments.spectral_reachability import agent_e3_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_identity_chunk_transport_20261029_v1 as chunks

NAME = 'agent_e3_chunked_identity_host_20261029_v1'
need = host.need
launch = host.launch


def authority(path, sha):
    binding = host.authority(path, sha)
    sources = dict(binding['prepared']['bindings_sha256'])
    launch.host.source.merge(sources, binding['additional_sources'])
    launch.episode.old.guard.validate_sources(sources, [str(Path(m.__file__).resolve()) for m in (host, chunks)] + [str(Path(__file__).resolve())])
    return binding


def perform(binding, store):
    launch._record(store, 'chunked_identity_host_started', dict(schema=NAME))
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from experiments.gpu_preflight import require_cuda_allocation
    gpu = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    launch._record(store, 'launcher_gpu_preflight', {'preflight': gpu})
    loaded = chunks.load_bound(binding['prepared'], torch, device, store, binding['prepare_sha'])
    row = dict(episode_id='cell%02d' % binding['cell'], warmup_returned=0, control_returned=0,
               executed_raw=[], PNG=0, checkpoints=[], status='started')
    sources = dict(binding['prepared']['bindings_sha256']); launch.host.source.merge(sources, binding['additional_sources'])
    launch.module_census.record(store, 'before', launch.module_census.snapshot(sources))
    primary = None
    try:
        result = host.guarded_episode(binding, loaded, torch, device, store, row)
    except BaseException as exc:
        primary = exc; raise
    finally:
        try: launch.module_census.record(store, 'after', launch.module_census.snapshot(sources))
        except BaseException as exc:
            if primary is None: raise
            primary.add_note('post-cleanup module census: ' + repr(exc))
    need(launch.host.sha(binding['manifest_path']) == binding['manifest_sha'], 'authority unchanged')
    return result


def run(authority_path, authority_sha, output, authorization):
    need(authorization == launch.AUTH, 'explicit HOST CUDA authorization')
    return launch._run_lifecycle(output, time.monotonic(), lambda: authority(authority_path, authority_sha), perform)


def inspect(output, report_sha, completion_sha, authority_path, authority_sha):
    base = host.prior.inspect(output, report_sha, completion_sha, authority_path, authority_sha)
    binding = authority(authority_path, authority_sha)
    p = Path(output); b = launch.pub.base
    report = b.read_bytes(p/'report.json', launch.pub.FINAL_CAP)
    journal = b.read_bytes(p/'events.jsonl', b.CAP)
    need(b.fsha(report) == report_sha and b.fsha(journal) == b.decode(report)['journal_sha256'], 'stable journal')
    events = [b.decode(line) for line in journal.splitlines()]
    markers = [e for e in events if e['origin'] == 'physical' and e['resource'] == 'chunked_identity_host_started']
    need(len(markers) == 1 and markers[0]['kind'] == 'record' and markers[0]['detail'] == dict(schema=NAME), 'chunked host marker')
    loaders = [e for e in events if e['origin'] == 'physical' and e['resource'] == 'loader_receipt']
    cs = [e for e in events if e['origin'] == 'physical' and e['resource'] == 'preflight_binding_chunk']
    seals = [e for e in events if e['origin'] == 'physical' and e['resource'] == 'preflight_binding_seal']
    need(not loaders or markers[0]['index'] < loaders[0]['index'], 'host marker precedes loader')
    need(not cs or (len(loaders) == 1 and loaders[0]['index'] < cs[0]['index']), 'chunks follow loader')
    rebuilt = chunks.reconstruct(events, allow_partial=base['status'] == 'partial')
    if rebuilt is None:
        need(base['status'] == 'partial', 'unsealed binding only partial')
        return dict(schema=NAME+'.inspect', base=base, identity=None, binding_chunks_complete=False,
                    full_saved_strict=False, scientific_admission=False)
    need(len(seals) == 1, 'one binding seal')
    recorded = host.identity.verify(rebuilt, binding['prepared'], binding['prepare_sha'])
    physical = [x['boundary'] for x in base['encoding_ledger']['boundaries']]
    actual = [x['boundary'] for x in recorded['encodes']]
    if recorded['pending'] is not None: actual.append(recorded['pending']['boundary'])
    if base['status'] == 'partial':
        need(actual == physical[:len(actual)], 'partial identity boundary prefix')
    else:
        need(actual == physical and recorded['pending'] is None and
             all(x['outcome'] == 'return' for x in recorded['encodes']), 'successful boundaries bound')
    return dict(schema=NAME+'.inspect', base=base, identity=recorded, binding_chunks_complete=True,
                full_saved_strict=False, scientific_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    for name in ('run', 'inspect'):
        a = sub.add_parser(name)
        for opt in ('authority', 'authority-sha256', 'output'): a.add_argument('--'+opt, required=True)
        if name == 'run': a.add_argument('--authorization', required=True)
        else:
            a.add_argument('--report-sha256', required=True); a.add_argument('--completion-sha256', required=True)
    a = p.parse_args()
    result = run(a.authority, a.authority_sha256, a.output, a.authorization) if a.command == 'run' else inspect(
        a.output, a.report_sha256, a.completion_sha256, a.authority, a.authority_sha256)
    print(launch.host.raw(result).decode())


if __name__ == '__main__': main()
