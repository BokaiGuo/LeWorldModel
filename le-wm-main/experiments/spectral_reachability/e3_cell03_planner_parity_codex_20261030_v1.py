"""Independent CUDA recomputation of the two fixed cell03 saved planning nodes."""
import argparse
import base64
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback

import numpy as np

from experiments.spectral_reachability import e3_cell03_cuda_parity_codex_20261030_v2 as base
from experiments.spectral_reachability import e3_cell03_encoder_parity_inputs_codex_20261030_v1 as inputs
from experiments.spectral_reachability import agent_e3_actual_feedback_join_20261029_v2 as join
from experiments.spectral_reachability import agent_e3_multinode_saved_20261029_v1 as nodes
from experiments.spectral_reachability import agent_e3_unified_transport_20261029_v1 as transport

ROOT = inputs.ROOT
MANIFEST = ROOT/'e3_cell03_planner_parity_codex_20261030_v1_manifest.json'
OUTPUT = ROOT/'e3_cell03_planner_parity_codex_20261030_v1_RUN01'
MODULE = 'experiments.spectral_reachability.e3_cell03_planner_parity_codex_20261030_v1'
CONFIRM = 'E3_CELL03_PLANNER_PARITY_COOPERATIVE'
WALL = 900


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reconstruct():
    prepared = inputs.inputs()
    host = inputs.host
    binding = host.authority(str(inputs.AUTHORITY), inputs.AUTHORITY_SHA)
    launch = host.launch
    fixed = launch.episode.old.authority.load(*binding['args'])
    setup = launch.episode.old.setup_authority.resolve(binding['resolver']['path'],
             binding['resolver']['sha256'], 3, fixed['body'])
    journal = (inputs.OUTPUT/'events.jsonl').read_bytes()
    report = json.loads((inputs.OUTPUT/'report.json').read_bytes())
    need(hashlib.sha256(journal).hexdigest() == report['journal_sha256'], 'source journal drift')
    events = [json.loads(x) for x in journal.splitlines()]
    body = deepcopy(fixed['body'])
    prior = np.asarray(setup['prefix5'], np.float32)
    action = np.asarray(fixed['first5'], np.float32)
    result = []
    for row in prepared['rows']:
        boundary = row['boundary']
        def one(resource):
            found = [e for e in events if e['resource'] == resource and
                     e['detail'].get('boundary') == boundary]
            need(len(found) == 1, 'unique '+resource)
            return found[0]
        ei, er = one('native_e3_encode_intent'), one('native_e3_encode_return')
        ni, nr = one('native_e3_node_intent'), one('native_e3_node_return')
        need(ei['index'] < er['index'] < ni['index'] < nr['index'], 'node chronology')
        frame = row['frame']
        png = (inputs.OUTPUT/frame['file']).read_bytes()
        latent = inputs.checked_array(inputs.OUTPUT, row['arrays']['latent'])
        context = deepcopy(body['root_context'])
        for k in ('latent_history', 'action_history'):
            context[k] = np.asarray(context[k], np.float32)
        context = join.joined_context(context, prior, action, action, latent,
                    np.asarray(body['action_mean'], np.float32), np.asarray(body['action_std'], np.float32),
                    expected_rgb_sha=frame['sha256'], rgb_bytes=png, encoding_input_sha=frame['sha256'],
                    episode_id='cell03', physical_episode_id='cell03', encoder_episode_id='cell03',
                    start=boundary-5, end=boundary, physical_end=boundary,
                    encoding_boundary=boundary, stopped=False)
        body['root_context'] = dict(context, latent_history=context['latent_history'].tolist(),
                                    action_history=context['action_history'].tolist())
        body['feedback_boundary'] = boundary
        digest = nodes.bridge.host.digest(body)
        need(digest == ni['detail']['body_sha256'] == nr['detail']['body_sha256'], 'fixed node body')
        chunks = [e['detail'] for e in events if e['resource'] == 'node_row'
                  and e['detail']['boundary'] == boundary]
        need(chunks and [x['index'] for x in chunks] == list(range(len(chunks))) and
             all(x['total'] == len(chunks) and x['row_sha256'] == chunks[0]['row_sha256'] for x in chunks),
             'fixed node chunks')
        data = b''.join(base64.b64decode(x['data'], validate=True) for x in chunks)
        need(hashlib.sha256(data).hexdigest() == chunks[0]['row_sha256'], 'node row SHA')
        saved = json.loads(data)
        need(np.asarray(saved['result']['first5'], np.float32).tobytes().hex() == nr['detail']['raw_bits'],
             'saved node action')
        result.append(dict(boundary=boundary, body=deepcopy(body), body_sha256=digest,
                           original_row_sha256=chunks[0]['row_sha256'], original_result=saved['result'],
                           original_action_bits=nr['detail']['raw_bits']))
        prior = action.copy()
        action = np.frombuffer(bytes.fromhex(nr['detail']['raw_bits']), dtype='<f4').copy().reshape(5, 2)
    need([r['boundary'] for r in result] == [10, 15], 'fixed two nodes')
    return result


def prepare():
    rows = reconstruct()
    return dict(schema='e3_cell03_planner_parity_v1', rows=rows, authority_sha256=inputs.AUTHORITY_SHA,
                original_report_sha256=inputs.REPORT_SHA, original_completion_sha256=inputs.COMPLETION_SHA,
                launcher_sha256=base.sha(__file__), sources=base.loaded_project_sources(),
                python_sha256=base.sha(Path(sys.executable).resolve()), lock=str(base.LOCK),
                output=str(OUTPUT), wall_seconds=WALL, transport_cap_bytes=transport.CAP,
                per_node_iterations=32, prediction_packets_cap='68 times horizon packets',
                allocated_peak_cap=3*1024**3, reserved_peak_cap=4*1024**3,
                tolerance='none: exact result JSON and float32 action bytes',
                scientific_admission=False)


def verify(expected):
    data = MANIFEST.read_bytes()
    need(hashlib.sha256(data).hexdigest() == expected, 'manifest SHA')
    saved = json.loads(data)
    need(saved == prepare(), 'manifest or source drift')
    return saved


def execute(expected):
    base.write_new(OUTPUT/'child.claim', base.raw(dict(manifest_sha256=expected, pid=os.getpid())))
    sink = base.Sink(OUTPUT)
    try:
        manifest = verify(expected)
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        gpu = require_cuda_allocation('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        binding = inputs.host.authority(str(inputs.AUTHORITY), inputs.AUTHORITY_SHA)
        launch = inputs.host.launch
        sources = dict(binding['prepared']['bindings_sha256'])
        launch.host.source.merge(sources, binding['additional_sources'])
        sources = base.merge_sources(sources, manifest['sources'])
        model, model_hash = launch.host.primitives.native.loader.loaded_model(
            binding['prepared'], torch, device, sink, binding['prepare_sha'])
        guard = launch.episode.old.guard
        census_before = launch.module_census.snapshot(sources)
        store = transport.Store(OUTPUT/'numeric')
        results = []
        with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            before = guard.snapshot(model, torch, device, sources, [str(Path(__file__).resolve())])
            for entry in manifest['rows']:
                boundary = entry['boundary']
                sink.event(dict(stage='node_intent', boundary=boundary, body_sha256=entry['body_sha256']))
                bridge = nodes.NodeBridge(store, boundary)
                record = dict(forwards=[], AD=[], result=None)
                backend = nodes.bridge.host.HostBackend(model, device, torch, bridge, entry['body'], record)
                with backend.observe():
                    record['result'] = nodes.node.solve(entry['body'], backend, bridge.event)
                row_sha = bridge.finish(record)
                actual = record['result']
                exact = actual == entry['original_result']
                bits = np.asarray(actual['first5'], np.float32).tobytes().hex()
                results.append(dict(boundary=boundary, row_sha256=row_sha, result_exact=exact,
                                    action_exact=bits == entry['original_action_bits'], action_bits=bits))
                sink.event(dict(stage='node_return', **results[-1]))
            after = guard.snapshot(model, torch, device, sources, [str(Path(__file__).resolve())])
        guard.validate_pair(before, after, binding['prepared']['expected_model_hash'])
        census_after = launch.module_census.snapshot(sources)
        event_sha = base.sha(store.output/'events.jsonl')
        audited = transport.consume(store.output, event_sha)
        need(audited['transport_complete'], 'numeric transport incomplete')
        sink.put('report.json', dict(status='completed', manifest_sha256=expected, gpu=gpu,
                 model_hash=model_hash, rows=results, numeric_journal_sha256=event_sha,
                 numeric_counts=audited['counts'], numeric_bytes=audited['bytes'],
                 census_before=len(census_before), census_after=len(census_after),
                 full_saved_strict=False, scientific_admission=False))
    except BaseException as exc:
        try:
            sink.event(dict(stage='failure', type=type(exc).__name__, message=str(exc),
                            traceback=traceback.format_exc()[-8192:]))
            sink.put('failure.json', dict(status='failed', type=type(exc).__name__, message=str(exc)))
        except BaseException:
            pass
        raise


def run(expected, confirm):
    need(confirm == CONFIRM and os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'),
         'coordinated CUDA authorization')
    verify(expected)
    fd = os.open(base.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'lock inode')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        now = os.stat(base.LOCK, follow_symlinks=False)
        need((st.st_dev, st.st_ino) == (now.st_dev, now.st_ino), 'lock replaced')
        verify(expected)
        os.mkdir(OUTPUT, 0o700)
        base.syncdir(ROOT)
        start = time.monotonic()
        try:
            child = subprocess.run([sys.executable, '-B', '-m', MODULE, 'child', '--sha256', expected,
                                    '--lock-fd', str(fd)], cwd=ROOT.parent, pass_fds=(fd,),
                                   timeout=WALL, check=False)
            result = dict(child_returncode=child.returncode, elapsed_seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            result = dict(child_returncode=None, elapsed_seconds=time.monotonic()-start, timeout=True)
        base.write_new(OUTPUT/'parent.json', base.raw(result))
        need(result['child_returncode'] == 0, 'child failed or timed out')
        return result
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'run', 'child'))
    p.add_argument('--sha256')
    p.add_argument('--confirm')
    p.add_argument('--lock-fd', type=int)
    a = p.parse_args()
    if a.mode == 'prepare':
        data = base.raw(prepare())
        base.write_new(MANIFEST, data)
        base.syncdir(ROOT)
        print(hashlib.sha256(data).hexdigest())
    elif a.mode == 'run':
        print(json.dumps(run(a.sha256, a.confirm)))
    else:
        need(a.lock_fd is not None, 'inherited lock required')
        held, current = os.fstat(a.lock_fd), os.stat(base.LOCK, follow_symlinks=False)
        need((held.st_dev, held.st_ino) == (current.st_dev, current.st_ino), 'wrong lock')
        fcntl.flock(a.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(a.sha256)


if __name__ == '__main__':
    main()
