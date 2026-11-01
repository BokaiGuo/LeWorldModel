import copy
import hashlib
import io
import json

import numpy as np
from PIL import Image
import pytest
from test_agent_p3_c3_k17_live_guard_20261029_v1 import fixture as guard_fixture
from experiments.spectral_reachability import agent_p3_c3_k17_native_store_20261029_v1 as store
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_physical_gate_20261029_v1 as gate
from experiments.spectral_reachability import agent_p3_c3_k17_replay_guard_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import p3_k17_saved_native_guard_frames_20261029 as strict
from experiments.spectral_reachability import p3_k17_live_source_budget_seam_20261029 as seam


def file_sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def recording(tmp_path, monkeypatch):
    identity, task, ref, live = guard_fixture()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    raw = buffer.getvalue()
    pixels = outer.h3.moving.image_sha(image)
    fsha = hashlib.sha256(raw).hexdigest()
    for frame in live['warm_frames'] + [live['goal_frame']] + ref['warm_trace'] + [ref['goal_frame']]:
        frame['receipt']['file_sha256'] = fsha
        frame['receipt']['pixel_sha256'] = pixels
    item = dict(identity, root_task_sha256=bridge.digest(task), first5_raw_f32_hex=np.zeros((5, 2), '<f4').tobytes().hex(),
                replays=[{'token_f32_hex': np.zeros(10, '<f4').tobytes().hex()} for _ in range(2)])
    rows = [(dict(item, slot=s, root=r, mode=m), task, ref) for s, r, m in outer.ORDER]
    monkeypatch.setattr(outer, 'material', lambda: ({}, rows))
    monkeypatch.setattr(outer.h3.moving, 'guard', lambda *args: None)
    native = store.NativeStore(tmp_path/'native')
    native.emit('operation_dispatch_intent', receipt={'resource': 'constructor'})
    where = {'branch': 0, 'slot': 3, 'root_slot': 0, 'mode': 'fixedH1'}
    native.emit('placement_recipe', **where, reset_identity=ref['reset_identity'])
    for n, frame in enumerate(live['warm_frames']):
        if n and n % 3 == 0:
            op = live['warm_returns'][n//3-1]
            native.emit('returned_transition', **where, phase='warmup', step=op['step'], actual_dtype=op['actual_dtype'],
                        requested_bits=op['requested_f32_hex'], actual_bits=op['actual_f32_hex'],
                        terminated=False, truncated=False, state=live['root_state'], snapshot=live['bound_goal']['snapshot'])
        frame['receipt']['png'] = f'{n}.png'
        native.inner.png(f'{n}.png', raw)
        native.emit('fresh_observation', **where, **frame)
    goal = live['bound_goal']
    native.emit('goal_attached', **where, goal=live['goal_state'], snapshot=goal['snapshot'],
                goal_pose=goal['goal_pose'], goal_rgb_sha256=goal['goal_rgb_sha256'])
    goal0 = live['goal_frame']
    for k in range(3):
        f = copy.deepcopy(goal0)
        f['receipt'].update(png=f'{33+k}.png', overlay_index=k, condition='on' if k == 1 else 'off')
        native.inner.png(f'{33+k}.png', raw)
        native.emit('fresh_observation', **where, **f)
        if k == 0: live['goal_frame'] = copy.deepcopy(f)
    proof = gate.capture_check(item, task, ref, live, native.path)
    native.emit('k17_guard', accepted=True, proof=proof)
    return native.path, seam.source_digest({}, rows)


def reseal(path, change):
    journal = path/'journal.jsonl'
    events = [json.loads(line) for line in journal.read_bytes().splitlines()]
    change(events)
    previous, result = None, []
    for i, event in enumerate(events):
        body = dict(index=i, previous=previous, kind=event['kind'], payload=event['payload'])
        previous = hashlib.sha256(transport.canonical(body)).hexdigest()
        result.append(transport.canonical(dict(body, sha256=previous)))
    journal.write_bytes(b'\n'.join(result)+b'\n')


def test_reconstructs_real_guard_from_saved_events_and_png(recording):
    path, material = recording
    result = strict.audit(path, file_sha(path/'journal.jsonl'), strict.source_files(), material)
    assert result['guards_recomputed'] == 1 and result['frames_checked'] == 36
    assert not result['native_provenance_verified'] and not result['scientific_pass']


@pytest.mark.parametrize('change', ['frame_order', 'proof', 'placement', 'goal'])
def test_rechained_semantic_forgery_fails(recording, change):
    recording, material = recording
    def mutate(events):
        if change == 'frame_order':
            e = next(e for e in events if e['kind'] == 'fresh_observation')
            e['payload']['receipt']['step'] = 4
        if change == 'proof':
            e = next(e for e in events if e['kind'] == 'k17_guard')
            e['payload']['proof']['history']['accepted'] = False
        if change == 'placement':
            next(e for e in events if e['kind'] == 'placement_recipe')['payload']['reset_identity']['seed'] = 99
        if change == 'goal':
            next(e for e in events if e['kind'] == 'goal_attached')['payload']['goal'][0] = -9
    reseal(recording, mutate)
    with pytest.raises(ValueError): strict.audit(recording, file_sha(recording/'journal.jsonl'), strict.source_files(), material)


def test_corrupt_frame_and_external_source_pin_fail(recording):
    recording, material = recording
    pins = strict.source_files()
    bad = dict(pins)
    bad[next(iter(bad))] = '0'*64
    with pytest.raises(ValueError, match='source pins'): strict.audit(recording, file_sha(recording/'journal.jsonl'), bad, material)
    (recording/'0.png').write_bytes(b'not a PNG')
    with pytest.raises(ValueError, match='frame file SHA'): strict.audit(recording, file_sha(recording/'journal.jsonl'), pins, material)


def test_resealed_file_cannot_fake_pixel_digest(recording):
    path, material = recording
    output = io.BytesIO()
    Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8)).save(output, format='PNG')
    (path/'0.png').write_bytes(output.getvalue())
    def change(events):
        e = next(e for e in events if e['kind'] == 'fresh_observation')
        e['payload']['receipt']['file_sha256'] = file_sha(path/'0.png')
    reseal(path, change)
    with pytest.raises(ValueError, match='pixel SHA'):
        strict.audit(path, file_sha(path/'journal.jsonl'), strict.source_files(), material)
