"""Bounded exploratory CUDA replay of two saved cell03 encoder boundaries."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import stat

import numpy as np

from experiments.spectral_reachability import e3_cell03_encoder_parity_inputs_codex_20261030_v1 as inputs
from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock

ROOT = inputs.ROOT.parent
MANIFEST = inputs.ROOT / 'e3_cell03_cuda_parity_codex_20261030_v1_manifest.json'
OUTPUT = inputs.ROOT / 'e3_cell03_cuda_parity_codex_20261030_v1_RUN01'
LOCK = inputs.ROOT / 'e3_p2_host_cuda_cooperative.lock'
WALL = 600
OUTPUT_CAP = 16 * 1024 * 1024
CONFIRM = 'E3_CELL03_CUDA_PARITY_COOPERATIVE'
MODULE = 'experiments.spectral_reachability.e3_cell03_cuda_parity_codex_20261030_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_new(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        if f.write(data) != len(data):
            raise OSError('short write')
        f.flush()
        os.fsync(f.fileno())


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare():
    rows = inputs.inputs()
    return dict(schema='e3_cell03_cuda_parity_manifest_v1',
                original_report=inputs.REPORT_SHA, original_completion=inputs.COMPLETION_SHA,
                authority_sha256=inputs.AUTHORITY_SHA, boundaries=[10, 15],
                input_descriptors=rows['rows'], launcher_sha256=sha(__file__),
                input_helper_sha256=sha(inputs.__file__), lock_sha256=sha(lock.__file__),
                python_path=sys.executable, python_binary_sha256=sha(Path(sys.executable).resolve()),
                output=str(OUTPUT), lock=str(LOCK), wall_seconds=WALL,
                output_cap_bytes=OUTPUT_CAP, encoder_calls_cap=2, projector_calls_cap=2,
                allocated_peak_cap=3*1024**3, reserved_peak_cap=4*1024**3,
                comparison='float32 exact bytes and max absolute difference; no threshold',
                snapshot_scope='authority and loaded project module snapshots, not system closure',
                scientific_admission=False)


def verify_manifest(expected):
    data = MANIFEST.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('manifest SHA mismatch')
    saved = json.loads(data)
    if saved != prepare():
        raise ValueError('manifest or source drift')
    return saved


class Sink:
    def __init__(self, output):
        self.output = output
        self.used = 0
        self.event_path = output / 'events.jsonl'

    def event(self, value):
        data = raw(value)
        self.used += len(data)
        if self.used > OUTPUT_CAP:
            raise ValueError('output cap')
        with self.event_path.open('ab') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    def put(self, name, value):
        data = raw(value)
        self.used += len(data)
        if self.used > OUTPUT_CAP:
            raise ValueError('output cap')
        write_new(self.output / name, data)

    def array(self, name, value):
        from io import BytesIO
        stream = BytesIO()
        np.save(stream, value, allow_pickle=False)
        data = stream.getvalue()
        self.used += len(data)
        if self.used > OUTPUT_CAP:
            raise ValueError('output cap')
        write_new(self.output / name, data)
        return hashlib.sha256(data).hexdigest()


def calculate(model, torch, device, x):
    captures = {}
    handles = []
    def pre(module, args):
        if 'projector_input' in captures:
            raise ValueError('projector called twice')
        captures['projector_input'] = args[0].detach().cpu().numpy().copy()
    def encoder_post(module, args, result):
        if 'encoder_hidden' in captures:
            raise ValueError('encoder called twice')
        if hasattr(result, 'last_hidden_state'):
            result = result.last_hidden_state
        captures['encoder_hidden'] = result.detach().cpu().numpy().copy()
    def projector_post(module, args, result):
        if 'projector_output' in captures:
            raise ValueError('projector called twice')
        captures['projector_output'] = result.detach().cpu().numpy().copy()
    try:
        handles.append(model.encoder.register_forward_hook(encoder_post))
        handles.append(model.projector.register_forward_pre_hook(pre))
        handles.append(model.projector.register_forward_hook(projector_post))
        z = model.encode({'pixels': x})['emb']
        captures['latent'] = z[0, 0].detach().cpu().numpy().copy()
        return captures
    finally:
        for handle in reversed(handles):
            handle.remove()


def execute(out, expected):
    write_new(out / 'child.claim', raw(dict(manifest_sha256=expected, pid=os.getpid())))
    syncdir(out)
    sink = Sink(out)
    try:
        manifest = verify_manifest(expected)
        sink.event(dict(stage='verified', manifest_sha256=expected))
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        gpu = require_cuda_allocation('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        host = inputs.host
        binding = host.authority(str(inputs.AUTHORITY), inputs.AUTHORITY_SHA)
        sources = dict(binding['prepared']['bindings_sha256'])
        host.launch.host.source.merge(sources, binding['additional_sources'])
        for path, digest in ((Path(__file__).resolve(), manifest['launcher_sha256']),
                             (Path(inputs.__file__).resolve(), manifest['input_helper_sha256']),
                             (Path(lock.__file__).resolve(), manifest['lock_sha256'])):
            sources[str(path)] = digest
        model, before_hash = host.launch.host.primitives.native.loader.loaded_model(
            binding['prepared'], torch, device, sink, binding['prepare_sha'])
        census_before = host.launch.module_census.snapshot(sources)
        guard = host.launch.episode.old.guard
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            before = guard.snapshot(model, torch, device, sources, [str(Path(__file__).resolve())])
            rows = []
            for entry in manifest['input_descriptors']:
                boundary = entry['boundary']
                sink.event(dict(stage='encode_intent', boundary=boundary))
                arr = inputs.checked_array(inputs.OUTPUT, entry['arrays']['preprocessed'])
                actual = calculate(model, torch, device, torch.from_numpy(arr).to(device))
                torch.cuda.synchronize(device)
                expected_arrays = {r: inputs.checked_array(inputs.OUTPUT, entry['arrays'][r]) for r in inputs.ROLES}
                comparison = inputs.compare(expected_arrays, actual)
                artifacts = {r: sink.array(f'boundary{boundary}_{r}.npy', actual[r]) for r in inputs.ROLES}
                rows.append(dict(boundary=boundary, comparison=comparison, actual_sha256=artifacts))
                sink.event(dict(stage='encode_return', boundary=boundary))
            after = guard.snapshot(model, torch, device, sources, [str(Path(__file__).resolve())])
        guard.validate_pair(before, after, binding['prepared']['expected_model_hash'])
        census_after = host.launch.module_census.snapshot(sources)
        sink.put('report.json', dict(status='completed', manifest_sha256=expected, gpu=gpu,
                                    model_hash=before_hash, rows=rows,
                                    census_before=len(census_before), census_after=len(census_after),
                                    full_saved_strict=False, scientific_admission=False))
    except BaseException as exc:
        try:
            sink.event(dict(stage='failure', type=type(exc).__name__, message=str(exc),
                            traceback=traceback.format_exc()[-8192:]))
            sink.put('failure.json', dict(status='failed', error_type=type(exc).__name__, message=str(exc)))
        except BaseException:
            pass
        raise


def run(expected, confirmation):
    if confirmation != CONFIRM:
        raise ValueError('explicit coordination required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        raise ValueError('CUDA disabled')
    verify_manifest(expected)
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError('invalid lock file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = os.stat(LOCK, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (st.st_dev, st.st_ino):
            raise ValueError('lock replaced')
        verify_manifest(expected)
        os.mkdir(OUTPUT, 0o700)
        syncdir(OUTPUT.parent)
        started = time.monotonic()
        try:
            child = subprocess.run([sys.executable, '-B', '-m', MODULE, 'child',
                                    '--sha256', expected, '--lock-fd', str(fd)], cwd=ROOT,
                                   timeout=WALL, check=False, pass_fds=(fd,))
            result = dict(child_returncode=child.returncode, elapsed_seconds=time.monotonic()-started)
        except subprocess.TimeoutExpired:
            result = dict(child_returncode=None, elapsed_seconds=time.monotonic()-started, timeout=True)
        write_new(OUTPUT / 'parent.json', raw(result))
        syncdir(OUTPUT)
        if result['child_returncode'] != 0:
            raise RuntimeError('child failed or timed out')
        return result
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run', 'child'])
    p.add_argument('--sha256')
    p.add_argument('--confirm')
    p.add_argument('--lock-fd', type=int)
    a = p.parse_args()
    if a.mode == 'prepare':
        body = prepare()
        data = raw(body)
        write_new(MANIFEST, data)
        syncdir(MANIFEST.parent)
        print(json.dumps(dict(manifest=str(MANIFEST), sha256=hashlib.sha256(data).hexdigest())))
    elif a.mode == 'run':
        print(json.dumps(run(a.sha256, a.confirm)))
    else:
        if a.lock_fd is None:
            raise ValueError('child requires inherited cooperative lock descriptor')
        held, current = os.fstat(a.lock_fd), os.stat(LOCK, follow_symlinks=False)
        if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError('wrong lock descriptor')
        fcntl.flock(a.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(OUTPUT, a.sha256)


if __name__ == '__main__':
    main()
