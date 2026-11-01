"""Metadata/fake-array tests only: never open the reserved native sources."""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p4_fresh_source_20261028_v2 as producer


def test_fixed_slots_and_non_overridable_history_certificate_requirement():
    slots = producer.slots()
    assert [s['root_seed'] for s in slots] == [203027101, 203027102, 203027103, 203027104]
    assert [s['reset_seed'] for s in slots] == [203027201, 203027202, 203027203, 203027204]
    with pytest.raises(ValueError, match='external historical certificate'):
        producer.history_certificate(None, None)


def test_documented_calibration_prohibition_is_not_a_runtime_gate():
    doc = (producer.ROOT / 'innerdocs/agent_p4_fresh_source_20261028_v2.md').read_text()
    assert '不执行四个预留根，包括calibration_only' in doc
    # Document the gap without invoking build or treating the mode as permission.
    source = inspect.getsource(producer.build)
    assert "mode in ('calibration_only','fresh_candidate')" in source
    assert "mode == 'fresh_candidate'" in source


def test_snapshot_and_rgb3_identity_are_computed_on_supplied_observation():
    state = [1., 2., 3., 4., 0., 0., 0.]
    snapshot = {'state': state, 'bodies': {'agent': {'x': 1}, 'block': {'x': 2}}}
    pixels = ['1' * 64, '2' * 64, '3' * 64]
    got = producer.root_identity({'state': state, 'snapshot': snapshot}, [{'pixel_sha256': h} for h in pixels])
    assert got['root_state'] == state
    assert got['root_state_sha256'] == hashlib.sha256(np.asarray(state, dtype='<f8').tobytes()).hexdigest()
    assert got['root_snapshot_sha256'] == producer.moving.source.digest(snapshot)
    assert got['history_pixels_sha256'] == pixels
    assert got['raw_history_sha256'] == producer.moving.image_sha(producer.moving.warmup_commands())


def test_pre_witness_gate_rejects_pose_alias_including_velocity_and_wrap():
    base = {'root_state': [1., 2., 3., 4., 0., 0., 0.], 'physical_pose': [1., 2., 3., 4., 0.],
            'root_state_sha256': 'a', 'root_snapshot_sha256': 'b', 'history_pixels_sha256': ['c']}
    same_pose = {'root_state': [1., 2., 3., 4., 2 * np.pi, 1., 2.]}
    assert producer.duplicate(base, same_pose)


def test_physics_authorization_checked_before_prepared_input_or_factory(monkeypatch):
    monkeypatch.setattr(producer, 'load_prepared', lambda *args: pytest.fail('read prepared before authorization'))
    with pytest.raises(ValueError, match='MAIN-only source physics'):
        producer.run('/not/read', '0' * 64, '/not/created', authorization='')
