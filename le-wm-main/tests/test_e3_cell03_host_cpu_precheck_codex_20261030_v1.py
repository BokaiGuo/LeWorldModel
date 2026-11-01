import numpy as np
import pytest

from experiments.spectral_reachability import e3_cell03_host_cpu_precheck_codex_20261030_v1 as m


def fixtures(monkeypatch):
    body = dict(root_sha256=m.ROOT_SHA, horizon_packets=3, domain={'name': 'Dx'}, feedback_boundary=5)
    action = np.zeros((5, 2), dtype=np.float32)
    binding = dict(cell=3, args=(), resolver={'path': 'r', 'sha256': 's'})
    control = dict(body=body, first5=action)
    setup = dict(original_root_sha=m.ROOT_SHA, expected_cp5=dict(step=5, pixel_sha256=m.CP5_PIXEL_SHA),
                 prefix5=np.zeros((5, 2), dtype=np.float32))
    monkeypatch.setattr(m.host, 'authority', lambda *_: binding)
    monkeypatch.setattr(m.host.launch.episode.old.authority, 'load', lambda *_: control)
    monkeypatch.setattr(m.host.launch.episode.old.setup_authority, 'resolve', lambda *_: setup)
    monkeypatch.setattr(m.host.launch.host, 'sha', lambda _: 'sha')
    return binding, control, setup


def test_success_is_saved_only(monkeypatch):
    fixtures(monkeypatch)
    result = m.check('/authority', 'sha')
    assert result['saved_setup_inputs_joined'] and not result['runtime_replay']
    assert result['physics_calls'] == result['neural_calls'] == 0 and not result['scientific_admission']


@pytest.mark.parametrize('fault', ['cell', 'root', 'horizon', 'domain', 'boundary', 'negative_zero', 'cp5', 'sha'])
def test_reject_changed_binding(monkeypatch, fault):
    binding, control, setup = fixtures(monkeypatch)
    body = control['body']
    if fault == 'cell': binding['cell'] = 2
    if fault == 'root': body['root_sha256'] = 'bad'
    if fault == 'horizon': body['horizon_packets'] = 1
    if fault == 'domain': body['domain']['name'] = 'Dxy'
    if fault == 'boundary': body['feedback_boundary'] = 10
    if fault == 'negative_zero': control['first5'][0, 1] = -0.0
    if fault == 'cp5': setup['expected_cp5']['pixel_sha256'] = 'bad'
    if fault == 'sha': monkeypatch.setattr(m.host.launch.host, 'sha', lambda _: 'changed')
    with pytest.raises(ValueError):
        m.check('/authority', 'sha')
