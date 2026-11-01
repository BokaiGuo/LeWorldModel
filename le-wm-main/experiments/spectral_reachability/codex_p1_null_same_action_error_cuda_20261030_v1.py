"""Fixed16 saved prefix-null physical trajectories: same-action CUDA prediction errors.

No physical execution, training, optimization, outcome selection, or CPU model
fallback. Prepare is CPU/read-only except its exclusive manifest output.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from experiments.spectral_reachability import codex_p1_prefix_null_saved_consumer_20261030_v2 as consume
from experiments.spectral_reachability import p1_backtrack_cuda_replay_20261030 as replay

NAME = 'codex_p1_null_same_action_error_cuda_20261030_v1'
ROOT = replay.ROOT
RUN = ROOT / 'results/codex_p1_prefix_null_runner_20261030_v2_run_MAIN01'
REPORT = RUN / 'report.json'
REPORT_SHA = '751344ac69ac078322592707dcd89eb534ee777e63128b3a8cd6298251953551'
PREPARE_SHA = '9344fb56183fbbfeb69520b8161c1d77000c334e241755d1ad92f2810427bef3'
require = consume.runner.need
sha = consume.runner.v1.base.sha
native = consume.runner.v1.base.native


def rgb(item):
    path = Path(item['path'])
    require(path.is_file() and not path.is_symlink() and path.resolve() == path and
            sha(path) == item['file_sha256'], 'RGB file identity')
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (224, 224), 'RGB mode or dimensions')
        pixels = np.asarray(image).copy()
    require(hashlib.sha256(pixels.tobytes()).hexdigest() == item['pixel_sha256'], 'RGB pixel identity')
    return pixels


def action_inputs(row):
    raw_bytes = bytes.fromhex(row['raw15_float32_hex'])
    require(len(raw_bytes) == 120 and hashlib.sha256(raw_bytes).hexdigest() == row['raw15_sha256'],
            'physical action bytes')
    raw = np.frombuffer(raw_bytes, dtype='<f4').copy().reshape(3, 10)
    mean, std = (np.asarray(row[k], dtype=np.float32) for k in ('action_mean', 'action_std'))
    history = np.asarray(row['action_history'], dtype=np.float32)
    z = np.asarray(row['model_history'], dtype=np.float32)
    warm = np.frombuffer(bytes.fromhex(row['warmup_raw_float32_hex']), dtype='<f4')
    require(mean.shape == std.shape == (2,) and history.shape == (2, 10) and z.shape == (3, 192)
            and warm.shape == (20,) and np.isfinite(raw).all() and np.abs(raw).max() <= 1
            and all(np.isfinite(x).all() for x in (mean, std, history, z, warm))
            and (std > 0).all(), 'action or history dimensions')
    require(np.array_equal((warm.reshape(2, 10) - np.tile(mean, 5)) / np.tile(std, 5), history),
            'warmup model action history mismatch')
    return raw, mean, std, history, z


def contexts(source, actual):
    z = np.asarray(source, np.float32)
    a = np.asarray(actual, np.float32)
    require(z.shape == (3, 192) and a.shape == (4, 192) and
            np.isfinite(z).all() and np.isfinite(a).all(), 'latent contexts')
    return np.stack((np.stack((z[1], a[0], a[1])), np.stack((a[0], a[1], a[2]))))


def action_contexts(history, tokens):
    h, t = np.asarray(history, np.float32), np.asarray(tokens, np.float32)
    require(h.shape == (2, 10) and t.shape == (3, 10) and
            np.isfinite(h).all() and np.isfinite(t).all(), 'action context dimensions')
    result = []
    for token in t:
        outgoing = np.concatenate((h, token[None]), axis=0)
        result.append(outgoing)
        h = outgoing[1:]
    return np.stack(result)


def compare_latents(actual, opens, refreshed):
    a, o, r = (np.asarray(x, np.float32) for x in (actual, opens, refreshed))
    require(a.shape == (4, 192) and o.shape == (3, 192) and r.shape == (2, 192)
            and all(np.isfinite(x).all() for x in (a, o, r)), 'comparison dimensions')
    comparisons = []
    for k, step in enumerate((5, 10, 15)):
        target, pred = a[k+1], o[k]
        diff = pred.astype(np.float64)-target.astype(np.float64)
        item = {'step': step, 'actual_latent': target.tolist(), 'open_prediction': pred.tolist(),
                'open_mse': float(np.mean(diff**2)), 'open_l2': float(np.linalg.norm(diff))}
        if k:
            pred = r[k-1]
            diff = pred.astype(np.float64)-target.astype(np.float64)
            item.update(refreshed_prediction=pred.tolist(), refreshed_mse=float(np.mean(diff**2)),
                        refreshed_l2=float(np.linalg.norm(diff)))
        else:
            item.update(refreshed_prediction=None, refreshed_mse=None, refreshed_l2=None)
        comparisons.append(item)
    return comparisons


def material():
    require(sha(REPORT) == REPORT_SHA, 'fixed physical report SHA')
    checked = consume.consume(RUN, REPORT_SHA)
    require(checked['status'] == 'saved_checks_passed_exploratory_only' and
            len(checked['rows']) == 16 and all(not r['missing_checkpoints'] for r in checked['rows']),
            'saved physical acceptance')
    report = native.read(REPORT, cap=64 * 1024**2)
    binding = report['execution_binding']
    require(binding['prepare_sha256'] == PREPARE_SHA, 'physical prepare identity')
    physical, _ = consume.runner.verify(binding['prepared'], PREPARE_SHA)
    _, _, model_report, original = replay.sources()
    for path, expected in ((replay.INPUT, replay.INPUT_SHA),
                           (replay.one.source.REPORT, replay.one.source.REPORT_SHA),
                           (replay.one.source.RECEIPT, replay.one.source.RECEIPT_SHA),
                           (Path(model_report['prepared']), model_report['prepare_sha256'])):
        require(sha(path) == expected, 'model input source pin')
    bindings = dict(physical['bindings'])
    for key, value in original['bindings'].items():
        require(key not in bindings or bindings[key] == value, 'source binding collision')
        bindings[key] = value
    paths = (REPORT, RUN / 'execution_binding.json', RUN / 'journal.jsonl',
             Path(binding['prepared']), Path(__file__).resolve(),
             Path(consume.__file__).resolve(), Path(replay.__file__).resolve(),
             replay.INPUT, replay.one.source.REPORT, replay.one.source.RECEIPT,
             Path(model_report['prepared']), ROOT/'experiments/checkpoint_compat.py',
             ROOT/'experiments/gpu_preflight.py',
             ROOT/'experiments/run_pusht_counterfactual.py',
             ROOT/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py')
    for path in paths:
        value = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == value, 'binding collision')
        bindings[str(path)] = value
    rows = []
    require(len(report['rows']) == len(physical['plans']) == 16, 'fixed sixteen')
    for index, (row, plan) in enumerate(zip(report['rows'], physical['plans'], strict=True)):
        task_row = physical['task_rows'][plan['slot']]
        task = original['requests'][plan['slot']]['task']
        require(row['index'] == plan['index'] == index and row['plan'] == plan and
                row['task_id'] == plan['task_id'] == task_row['task_id'] == task['binding']['task_id'] and
                task['binding']['root_sha256'] == task_row['model_root_sha256'] and
                task['binding']['blind_sha256'] == task_row['blind_sha256'] and
                row['root_sha256'] == plan['physical_root_sha256'] == task_row['native_root_sha256'] and
                row['condition'] == plan['condition'] == task_row['condition'] and
                row['role'] == plan['role'] and row['status'] in ('horizon_completed', 'native_stopped'),
                'row, physical and model identities')
        require([cp['step'] for cp in row['checkpoints']] == [0, 5, 10, 15], 'checkpoint coverage')
        images = []
        for cp in row['checkpoints']:
            path = native.paired.flat(RUN, cp['png'])
            item = dict(step=cp['step'], path=str(path), file_sha256=cp['file_sha256'],
                        pixel_sha256=cp['pixel_sha256'])
            rgb(item)
            require(str(path) not in bindings or bindings[str(path)] == cp['file_sha256'],
                    'RGB binding collision')
            bindings[str(path)] = cp['file_sha256']
            images.append(item)
        item = dict(index=index, slot=plan['slot'], root_slot=plan['root_slot'],
                    condition=plan['condition'], arm=plan['arm'], role=plan['role'],
                    task_id=plan['task_id'], duplicate_action_task_group=plan['duplicate_action_task_group'],
                    physical_root_sha256=plan['physical_root_sha256'], images=images,
                    raw15_float32_hex=plan['raw15_float32_hex'], raw15_sha256=plan['raw15_sha256'],
                    warmup_raw_float32_hex=task_row['warmup_raw_float32_hex'],
                    model_history=task['root_context']['latent_history'],
                    action_history=task['root_context']['action_history'],
                    action_mean=task['action_mean'], action_std=task['action_std'],
                    native_status=row['status'])
        action_inputs(item)
        rows.append(item)
    for path, digest in bindings.items():
        require(sha(path) == digest, 'source changed: ' + path)
    require(sha(REPORT) == REPORT_SHA, 'physical report changed')
    return dict(schema=NAME+'.prepare', report_sha256=REPORT_SHA, physical_prepare_sha256=PREPARE_SHA,
                model_hash=model_report['model_before'], policy=original['policy'], rows=rows,
                bindings=bindings, fixed_rows=16, image_encoder_calls=64, predictor_calls=80,
                action_encoder_calls=80, physical_execution=False, training=False, P1_admission=False,
                independent_confirmation=False, freshness=False)


def exclusive_path(path, kind):
    p = Path(path)
    require(p.is_absolute() and p.parent == ROOT / 'results' and p.parent == p.parent.resolve() and
            p.name.startswith(NAME+'_'+kind+'_') and not os.path.lexists(p), 'exclusive output path')
    return p


def write_exclusive(path, payload):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        require(stream.write(payload) == len(payload), 'short write')
        stream.flush()
        os.fsync(stream.fileno())
    native.paired.fsync_dir(path.parent)


def encode_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def prepare(path):
    path = exclusive_path(path, 'prepare')
    payload = encode_json(material())
    write_exclusive(path, payload)
    return hashlib.sha256(payload).hexdigest()


def run(prepared_path, expected, output):
    native.actual_hash(expected)
    prepared_path = Path(prepared_path)
    require(prepared_path.is_absolute() and prepared_path.parent == ROOT/'results' and
            prepared_path.name.startswith(NAME+'_prepare_') and prepared_path.is_file() and
            not prepared_path.is_symlink() and sha(prepared_path) == expected, 'prepare SHA or path')
    prepared = native.read(prepared_path, cap=64*1024**2)
    require(prepared == material() and sha(prepared_path) == expected, 'prepared or sources drift')
    out = exclusive_path(output, 'run')
    # An empty marker is deliberately retained on any failure; no silent retry.
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        native.paired.fsync_dir(out.parent)
        result = cuda_predictions(prepared)
        for path, digest in prepared['bindings'].items():
            require(sha(path) == digest, 'source changed during run: '+path)
        require(sha(prepared_path) == expected, 'prepare changed during run')
        result.update(prepared=str(prepared_path), prepare_sha256=expected)
        payload = encode_json(result)
        with os.fdopen(fd, 'wb') as stream:
            fd = None
            require(stream.write(payload) == len(payload), 'short report write')
            stream.flush()
            os.fsync(stream.fileno())
        native.paired.fsync_dir(out.parent)
        return hashlib.sha256(payload).hexdigest()
    finally:
        if fd is not None:
            os.close(fd)


def cuda_predictions(p):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    require(os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'), 'CUDA explicitly disabled')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    device_info = require_cuda_allocation('cuda:0')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
    before = replay.one.host.signal.model_hash(model)
    require(before == p['model_hash'] and all(x.device.type == 'cuda'
            for x in list(model.parameters()) + list(model.buffers())),
            'frozen model identity and CUDA placement')
    for module in model.modules():
        require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'], 'unbound model class')
    counts = dict(encoder=0, projector=0, predictor=0, action_encoder=0)
    handles = []
    for name in counts:
        def hook(_module, _args, key=name):
            counts[key] += 1
        handles.append(getattr(model, name).register_forward_pre_hook(hook))
    output = []
    try:
        transform = p0.image_transform()
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            for row in p['rows']:
                raw, mean, std, old_h, old_z = action_inputs(row)
                actual = []
                for item in row['images']:
                    x = p0._image_tensor(rgb(item), transform=transform, device=device)
                    z = model.encode({'pixels': x})['emb']
                    require(z.device.type == 'cuda' and z.dtype == torch.float32 and
                            tuple(z.shape) == (1, 1, 192) and bool(torch.isfinite(z).all()), 'encoded RGB')
                    actual.append(z[0, 0].cpu().numpy().copy())
                actual = np.stack(actual)
                token = (torch.tensor(raw, device=device) -
                         torch.tensor(np.tile(mean, 5), device=device)) / torch.tensor(np.tile(std, 5), device=device)
                tokens = token.cpu().numpy()
                expected = (raw - np.tile(mean, 5)) / np.tile(std, 5)
                require(np.allclose(tokens, expected, rtol=1e-6, atol=1e-6), 'CUDA action normalization')
                expected_contexts = action_contexts(old_h, tokens)
                z = torch.tensor(old_z, device=device)[None]
                h = torch.tensor(old_h, device=device)[None]
                opens, action_histories = [], []
                for k in range(3):
                    outgoing = torch.cat((h, token[k][None, None]), dim=1)
                    require(np.array_equal(outgoing[0].cpu().numpy(), expected_contexts[k]),
                            'action history ordering')
                    end = model.predict(z, model.action_encoder(outgoing))[:, -1:]
                    require(end.device.type == 'cuda' and tuple(end.shape) == (1, 1, 192)
                            and bool(torch.isfinite(end).all()), 'open prediction')
                    action_histories.append(outgoing[0].cpu().tolist())
                    opens.append(end[0, 0].cpu().numpy().copy())
                    z, h = torch.cat((z[:, 1:], end), dim=1), outgoing[:, 1:]
                refreshed = []
                for k, context in enumerate(contexts(old_z, actual)):
                    h = torch.tensor(action_histories[k+1], dtype=torch.float32, device=device)[None]
                    z = torch.tensor(context, device=device)[None]
                    end = model.predict(z, model.action_encoder(h))[:, -1]
                    require(end.device.type == 'cuda' and tuple(end.shape) == (1, 192)
                            and bool(torch.isfinite(end).all()), 'refreshed prediction')
                    refreshed.append(end[0].cpu().numpy().copy())
                output.append(dict(index=row['index'], task_id=row['task_id'], role=row['role'],
                                   slot=row['slot'], root_slot=row['root_slot'], condition=row['condition'],
                                   arm=row['arm'], physical_root_sha256=row['physical_root_sha256'],
                                   raw15_sha256=row['raw15_sha256'], images=row['images'],
                                   native_status=row['native_status'],
                                   duplicate_action_task_group=row['duplicate_action_task_group'],
                                   actual_root_latent=actual[0].tolist(), normalized_tokens=tokens.tolist(),
                                   open_action_contexts=action_histories,
                                   comparisons=compare_latents(actual, opens, refreshed)))
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    after = replay.one.host.signal.model_hash(model)
    require(before == after and counts == dict(encoder=64, projector=64, predictor=80, action_encoder=80),
            'model hash or neural call count')
    return dict(schema=NAME+'.report', rows=output, fixed_rows=16, model_before=before, model_after=after,
                host=device_info, calls=counts, training=False, physical_execution=False, P1_admission=False,
                independent_confirmation=False, freshness=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepared')
    parser.add_argument('--prepare-sha')
    args = parser.parse_args()
    try:
        print('Fixed 16 saved trajectories; same raw actions; open and RGB-refreshed CUDA predictions.', flush=True)
        digest = prepare(args.output) if args.command == 'prepare' else run(
            args.prepared, args.prepare_sha, args.output)
        print(json.dumps({'output': args.output, 'sha256': digest, 'P1_admission': False}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(json.dumps({'status': 'blocked_or_failed', 'reason': str(exc), 'P1_admission': False}),
              file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
