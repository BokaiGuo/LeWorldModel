"""Construct a 24-row planner-only input bundle from fixed H10 source evidence.

No model inference or physical execution. This is not process isolation: a
launcher must additionally restrict file access before invoking a planner.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import codex_p1_three_prefix_h10_runner_20261030 as h10
from experiments.spectral_reachability import p1_backtrack_cuda_replay_20261030 as replay

ROOT = h10.ROOT
OUTPUT = ROOT/'results/codex_p1_three_prefix_blind_inputs_20261030_v1.json'
KEYS = {'index', 'logical_pair', 'role', 'prefix_raw_f32_hex', 'cp5_png_base64',
        'cp5_pixel_sha256', 'root_latent_history', 'root_action_history',
        'goal_latent', 'action_mean', 'action_std'}


def need(value, message):
    if not value:
        raise ValueError(message)


def vector(value, shape):
    a = np.asarray(value, np.float32)
    need(a.shape == shape and np.isfinite(a).all(), 'finite tensor shape')
    return a


def validate_row(row):
    need(type(row) is dict and set(row) == KEYS, 'exact planner whitelist')
    need(type(row['index']) is int and 0 <= row['index'] < 24 and
         row['logical_pair'] == row['index']//3 and
         row['role'] == ('selected', 'zero', 'greedy')[row['index']%3], 'roster identity')
    raw = bytes.fromhex(row['prefix_raw_f32_hex'])
    need(len(raw) == 40, 'prefix length')
    a = np.frombuffer(raw, dtype='<f4').reshape(5, 2)
    need(np.isfinite(a).all() and (abs(a) <= 1).all(), 'raw action bounds')
    vector(row['root_latent_history'], (3, 192))
    vector(row['root_action_history'], (2, 10))
    vector(row['goal_latent'], (192,))
    vector(row['action_mean'], (2,))
    need((vector(row['action_std'], (2,)) > 0).all(), 'positive std')
    try:
        data = base64.b64decode(row['cp5_png_base64'], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224, 224),
                 'RGB ABI')
            pixels = np.asarray(image)
    except (ValueError, OSError) as exc:
        raise ValueError('invalid image') from exc
    need(hashlib.sha256(pixels.tobytes()).hexdigest() == row['cp5_pixel_sha256'], 'pixel SHA')
    return row


def make_row(anchor, task, warm_raw, png):
    checkpoint = anchor['checkpoint']
    need(hashlib.sha256(png).hexdigest() == checkpoint['file_sha256'], 'checkpoint PNG SHA')
    raw = vector(warm_raw, (10, 2))
    mean, std = vector(task['action_mean'], (2,)), vector(task['action_std'], (2,))
    need((std > 0).all(), 'normalizer')
    history = vector(task['root_context']['action_history'], (2, 10))
    expected = ((raw-mean)/std).reshape(2, 10)
    need(expected.tobytes() == history.tobytes(), 'warmup action-history normalization mismatch')
    need(task['binding']['task_id'] == anchor['task_id'], 'fixed task identity')
    return validate_row(dict(index=anchor['index'], logical_pair=anchor['logical_pair'], role=anchor['role'],
                prefix_raw_f32_hex=anchor['prefix5_float32_hex'],
                cp5_png_base64=base64.b64encode(png).decode('ascii'),
                cp5_pixel_sha256=checkpoint['pixel_sha256'],
                root_latent_history=task['root_context']['latent_history'],
                root_action_history=task['root_context']['action_history'],
                goal_latent=task['goal'], action_mean=task['action_mean'], action_std=task['action_std']))


def build():
    prepared = h10.design.verify(h10.INPUT, h10.INPUT_SHA)
    _, _, _, old = replay.sources()
    source_path = ROOT/'results/p1_backtrack_physical_source_join_20261030_MAIN02.json'
    source_bytes = source_path.read_bytes()
    need(hashlib.sha256(source_bytes).hexdigest() ==
         'cb82a39b8a464f16b7e9f71cd9d072f2d48c0887bf36d57c91ca50baf0f2af84', 'source join SHA')
    source_tasks = json.loads(source_bytes)['task_rows']
    need(len(prepared['anchors']) == 24 and len(old['requests']) == 4, 'full fixed roster')
    bindings = prepared['bindings']
    rows = []
    for i, anchor in enumerate(prepared['anchors']):
        need(anchor['index'] == i and anchor['logical_pair'] == i//3, 'ordered roster')
        slot = 2*anchor['root_slot'] + (anchor['condition'] == 'on')
        request = old['requests'][slot]
        need((request['root_slot'], request['condition']) ==
             (anchor['root_slot'], anchor['condition']), 'source request identity')
        overlay = 0 if anchor['condition'] == 'off' else 1
        events = [x['receipt'] for x in anchor['trace'] if x['receipt']['phase'] == 'plan' and
                  x['receipt']['step'] == 5 and x['receipt']['overlay_index'] == overlay]
        need(len(events) == 1, 'unique cp5 observation')
        receipt = events[0]
        checkpoint = anchor['checkpoint']
        need(all(receipt[k] == checkpoint[k] for k in ('png', 'file_sha256', 'pixel_sha256', 'state', 'snapshot')),
             'source checkpoint identity')
        matches = [Path(path) for path, digest in bindings.items() if
                   Path(path).name == checkpoint['png'] and digest == checkpoint['file_sha256']]
        need(len(matches) == 1 and matches[0].is_file() and not matches[0].is_symlink(),
             'unique bound cp5 image')
        warm = bytes.fromhex(next(x for x in source_tasks
            if x['root_slot'] == anchor['root_slot'] and x['condition'] == anchor['condition'])['warmup_raw_float32_hex'])
        need(len(warm) == 80, 'warm history bytes')
        rows.append(make_row(anchor, request['task'], np.frombuffer(warm, '<f4').reshape(10, 2),
                             matches[0].read_bytes()))
    need(len(rows) == 24, 'full denominator')
    return dict(schema='codex_p1_three_prefix_blind_inputs_v1', rows=rows,
                fixed_roster=24, planner_file_isolation_established=False,
                GPU_run=False, physical_run=False, P1_admission=False)


if __name__ == '__main__':
    body = build()
    data = (json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        need(stream.write(data) == len(data), 'complete output')
        stream.flush()
        os.fsync(stream.fileno())
    print(hashlib.sha256(data).hexdigest())
