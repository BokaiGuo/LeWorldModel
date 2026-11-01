"""Pure NumPy/mock regression for sealed live v2; never GPU/physics."""
from copy import deepcopy
import math
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p4_feedback_live_cuda_20261027_v2 as m
from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
from test_p4_feedback_live_cuda_20261027_v1 import protocol, setup, MockPhysical


@pytest.mark.parametrize('delta', [0., .2, math.pi, 2*math.pi+.1, 2*math.pi+.5, 4*math.pi+math.pi, -4*math.pi-.5])
def test_independent_wrapped_multi_turn_diagnostic_preserves_native(delta):
    state = np.zeros(7); goal = np.zeros(7); state[4] = delta
    native = labels(state, goal); before = deepcopy(native)
    result = m.wrapped_diagnostic(state, goal, native)
    assert native == before
    assert 0 <= result['wrapped_angle_error'] <= math.pi
    expected = abs((delta+math.pi) % (2*math.pi)-math.pi)
    assert result['wrapped_angle_error'] == pytest.approx(expected, abs=1e-14)
    assert result['wrapped_official_success'] == (expected < math.pi/9)
    assert result['native_reported_success'] == native['official_success']
    assert result['controls_actions'] is result['changes_goal_Q'] is result['P1_admission'] is result['P4_admission'] is False
    if abs(delta) > 2*math.pi and expected > math.pi/9:
        assert native['angle_error'] < 0 and native['official_success'] is True
        assert result['wrapped_official_success'] is False and result['native_wrapped_success_agree'] is False


def test_wrapped_seam_and_position_thresholds():
    state = np.zeros(7); goal = np.zeros(7); state[4], goal[4] = -math.pi+.01, math.pi-.01
    d = m.wrapped_diagnostic(state, goal, labels(state, goal)); assert d['wrapped_angle_error'] == pytest.approx(.02)
    state[0] = 25
    d = m.wrapped_diagnostic(state, goal, labels(state, goal))
    assert not d['wrapped_official_success'] and d['wrapped_object_only_success']
    state[2] = 25
    assert not m.wrapped_diagnostic(state, goal, labels(state, goal))['wrapped_object_only_success']


@pytest.mark.parametrize('state', [np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)])
def test_invalid_terminal_state_fails_closed(state):
    with pytest.raises(ValueError): m.wrapped_diagnostic(state, np.zeros(7), dict(angle_error=0, official_success=True))


def test_terminal_physical_adapter_names_native_labels_explicitly(monkeypatch):
    state = np.zeros(7); state[4] = 4*math.pi+.5; goal = np.zeros(7)
    original = dict(state=state.tolist(), goal=goal.tolist(), official=labels(state, goal), pose=dict(angle_distance=.5))
    monkeypatch.setattr(m.v1.Physical, 'evaluate', lambda self: deepcopy(original))
    physical = object.__new__(m.Physical); result = physical.evaluate()
    assert 'official' not in result and result['official_native'] == original['official']
    assert result['official_native']['official_success'] and not result['wrapped_diagnostic']['wrapped_official_success']
    assert result['pose'] == original['pose']


@pytest.mark.parametrize('has_first,write_fails,close_fails', [(True, False, True), (True, True, True), (False, False, True), (False, False, False)])
def test_journal_finalization_never_overwrites_first_error(has_first, write_fails, close_fails):
    first = dict(stage='mock_execution', message='ORIGINAL_FAILURE'); errors = [deepcopy(first)] if has_first else []
    class Journal:
        def __call__(self, event):
            if write_fails: raise OSError('journal write failed')
        def close(self):
            if close_fails: raise OSError('journal close failed')
    complete = m.finish_journal(Journal(), not has_first, errors)
    assert complete == (not has_first and not write_fails and not close_fails)
    if has_first: assert errors[0] == first
    assert sum(e['stage'] == 'journal_close' for e in errors) == int(close_fails)
    assert sum(e['stage'] == 'final_journal' for e in errors) == int(write_fails)


def test_run_early_failure_plus_journal_close_failure_still_writes_receipt(tmp_path, monkeypatch):
    # Abort before ANY torch/gym import by simulating a preexisting process alarm.
    p = protocol(); p['semantics'] = m.semantics(); p['response_receipt_sha256'] = 'mock_response'
    p['model_bindings'] = {}; monkeypatch.setattr(m, 'verify', lambda *args: p)
    monkeypatch.setattr(m.signal, 'getitimer', lambda *args: (1., 0.))
    original = m.v1.response.Journal
    class CloseFailure(original):
        def close(self):
            super().close(); raise OSError('mock close fault')
    monkeypatch.setattr(m.v1.response, 'Journal', CloseFailure)
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='incomplete live run'): m.run(tmp_path / 'mock-prepared', 'mock-sha', output)
    r = m.io.read(output / 'receipt.json')
    assert r['status'] == 'partial_or_failed' and r['first_error'] == r['errors'][0]
    assert r['journal_close_failed'] is True
    assert [e['stage'] for e in r['errors']] == ['live_execution', 'journal_close']
    assert r['first_error']['message'] == 'MAIN dedicated process; no existing alarm'
    assert all(c['attempted'] == c['completed'] == 0 for c in r['counts'].values())
    assert sum(r['physics_attempts'].values()) == 0


def test_model_predict_API_not_internal_predictor_entry_semantics(setup):
    _, _, ledger, _, _ = setup; seen = []
    tensor = lambda shape: SimpleNamespace(shape=shape, device=SimpleNamespace(type='cuda'))
    predictor = SimpleNamespace(forward=lambda *args: seen.append('internal'))
    model = SimpleNamespace(predictor=predictor, action_encoder=SimpleNamespace(forward=lambda *args: None),
        encoder=SimpleNamespace(forward=lambda *args: None), projector=SimpleNamespace(forward=lambda *args: None), encode=lambda *args: None)
    def predict(*args): predictor.forward(); predictor.forward(); return 'mock'
    model.predict = predict
    engine = lambda *args: None
    t = SimpleNamespace(autograd=SimpleNamespace(backward=engine, grad=engine, _engine_run_backward=engine,
        graph=SimpleNamespace(_engine_run_backward=engine), functional=SimpleNamespace(jacobian=engine)))
    hooks = m.io.Hooks(); m.v1.install_hooks(t, model, ledger, hooks)
    try:
        ledger.slot('imagined', 5, 'nominal'); model.action_encoder.forward(tensor((1, 3, 10)))
        model.predict(tensor((1, 3, 192)), None)
        assert ledger.counts()['predict'] == dict(attempted=1, completed=1) and seen == ['internal', 'internal']
        assert m.semantics()['internal_predictor_entries_instrumented'] is False
    finally: hooks.restore([])
    assert model.predict is predict


@pytest.mark.parametrize('perturbation,passes', [(1e-5, True), (4e-5, False)])
def test_overlap_tolerance_is_consistency_not_exact_observed_equality(setup, perturbation, passes):
    p, store, ledger, backend, _ = setup; physical = MockPhysical(ledger, store); rows = []
    original = backend.encode
    def encode(frames):
        values = original(frames)
        if ledger.boundary == 5: values[0, 0] += np.float32(perturbation)
        return values
    backend.encode = encode
    if passes:
        m.v1.workflow(p, backend, physical, ledger, store, rows)
        observed = np.load(store.path / 'actual.step5.observed.npy', allow_pickle=False)
        assert observed[0, 0] != np.float32(2.)
        assert len(ledger.rounds) == 6 and sum(ledger.physics[k] for k in ('warmup_steps', 'selected_steps')) == 70
    else:
        with pytest.raises(ValueError, match='overlapping'): m.v1.workflow(p, backend, physical, ledger, store, rows)
        assert ledger.counts()['jacobian_api']['attempted'] == 0


def test_parent_caps_may_not_be_lowered_or_changed(monkeypatch):
    p = protocol(); monkeypatch.setattr(m.v1, 'verify', lambda *args: p)
    reduced = deepcopy(p['limits']); reduced['wall_seconds'] = 300
    with pytest.raises(ValueError, match='caps unchanged'): m.material(reduced)


def test_prepare_missing_parent_never_fabricates_directory(tmp_path, monkeypatch):
    def missing(*args): raise ValueError('missing real parent source')
    # Isolate source refusal from independently tested, fluctuating disk gate.
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda *args: SimpleNamespace(free=m.RESERVE+m.PREPARE_CAP+m.OUTPUT_CAP+1024**2))
    monkeypatch.setattr(m.v1, 'verify', missing)
    output = tmp_path / 'new'
    with pytest.raises(ValueError, match='missing real parent'): m.prepare(output, protocol()['limits'])
    assert not output.exists()


def test_prepare_disk_headroom_checked_before_material(tmp_path, monkeypatch):
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda *args: SimpleNamespace(free=m.RESERVE+m.PREPARE_CAP+m.OUTPUT_CAP-1))
    with pytest.raises(ValueError, match='headroom'): m.prepare(tmp_path / 'new', protocol()['limits'])
    assert not (tmp_path / 'new').exists()


def test_parent_source_files_unchanged_and_increment_is_small():
    assert m.io.sha(m.v1.__file__) == 'c878a8cd6e1c7a5d645221e299993da5cd8e3609c30038ca1a50e4faaecdcaeb'
    assert m.io.sha(m.PARENT / 'prepare.json') == m.PARENT_SHA
    files = m.revision_sources()
    assert len(files) == 3 and all(m.NAME in f.name for f in files)
    assert sum(f.stat().st_size for f in files) < 256*1024
