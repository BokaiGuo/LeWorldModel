"""Offline/RAM tests; no model, scaler fit, simulator or network."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.spectral_reachability import agent_e1_hilewm_time_scaler_contract_20261029_v1 as m


def test_actual_source_semantics_and_d50_rows():
    data = m.pinned_sources()
    proof = m.source_proof(data)
    assert proof['first_target'] == 823
    assert proof['action_inverse'] > proof['replan_condition']
    rows = m.matrix(data['matrix'])
    assert len(rows) == 8
    assert rows[0]['high_samples_per_solve'] == 60000
    assert rows[0]['low_samples_per_solve'] == 27000
    assert sum(r['config_note'] == 'hiStaged' for r in rows) == 1


@pytest.mark.parametrize('key,old,new', [
    ('policy', 'self._z_subgoal = pred[:, 0, 0, :]', 'self._z_subgoal = pred[:, 0, -1, :]'),
    ('policy', 'self.process["action"].inverse_transform(action_np)', 'self.process["action"].transform(action_np)'),
    ('eval', 'processor.fit(col_data)', 'processor.fit(col_data[:1])'),
    ('sampling', 'high=max_available_span + 1', 'high=max_available_span'),
    ('model', 'def rollout_high(self, z_init: torch.Tensor, latent_actions: torch.Tensor)',
     'def rollout_high(self, z_init: torch.Tensor, latent_actions: torch.Tensor, horizon=50)'),
])
def test_semantic_drift_rejected_even_before_hash(key, old, new):
    data = m.pinned_sources()
    assert old.encode() in data[key]
    data[key] = data[key].replace(old.encode(), new.encode())
    with pytest.raises(ValueError):
        m.source_proof(data)


def test_pin_rejects_changed_source(tmp_path):
    rel, _ = m.SOURCES['eval']
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_text('not native')
    with pytest.raises(ValueError, match='SHA mismatch'):
        m.pinned_sources(tmp_path)


def test_reference_budget_is_conditional_and_complete_scope():
    row = m.matrix(m.pinned_sources()['matrix'])[0]
    cap = m.reference_ceiling(row)
    assert cap['episodes'] == 6
    assert cap['raw_control_calls'] == 600
    assert cap['high_solver_invocation_cap'] == cap['low_solver_invocation_cap'] == 120
    assert cap['high_configured_candidate_sequences_cap'] == 7_200_000
    assert cap['low_configured_candidate_sequences_cap'] == 3_240_000
    assert cap['winner_replay_module_calls'] is None
    assert cap['physics_substeps'] is None


def packets():
    shared = {k: 'same' for k in m.SHARED}
    shared.update(goal_frame_offset=50, eval_budget=100)
    return [dict(root_slot=r, arm=a, shared=copy.deepcopy(shared)) for r in (0, 1) for a in m.ARMS]


def test_three_arm_identity_is_structural_not_certification():
    p = packets()
    assert all(set(x['fields'].values()) == {'match'} for x in m.compare_packets(p))
    p[1]['shared']['history_bytes_sha256'] = None
    p[2]['shared']['goal_bytes_sha256'] = 'other'
    out = m.compare_packets(p)
    assert out[0]['fields']['history_bytes_sha256'] == 'unknown'
    assert out[0]['fields']['goal_bytes_sha256'] == 'mismatch'


@pytest.mark.parametrize('mutation', ['reverse', 'drop', 'h10', 'budget'])
def test_reject_wrong_denominator_order_or_time(mutation):
    p = packets()
    if mutation == 'reverse': p.reverse()
    if mutation == 'drop': p.pop()
    if mutation == 'h10': p[0]['shared']['goal_frame_offset'] = 10
    if mutation == 'budget': p[0]['shared']['eval_budget'] = 50
    with pytest.raises(ValueError): m.compare_packets(p)


def test_missing_is_not_zero_or_match():
    p = packets()
    for x in p: x['shared'] = {}
    assert all(set(x['fields'].values()) == {'unknown'} for x in m.compare_packets(p))


def test_json_duplicate_and_nonfinite():
    for text in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        with pytest.raises(ValueError): m.decode(text)


def test_real_tempdisk_prepare_verify_and_tamper(tmp_path):
    base = [sys.executable, '-B', '-m', 'experiments.spectral_reachability.'+m.NAME]
    output = tmp_path/'prepare.json'
    run = subprocess.run(base+['prepare', '--output', str(output)], cwd=m.ROOT, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = json.loads(output.read_bytes())
    assert manifest['semantics']['original_scaler_values'] is None
    assert manifest['prospective']['physical_release'] is False
    verify = subprocess.run(base+['verify', '--manifest', str(output), '--manifest-sha256', result['sha256']], cwd=m.ROOT, capture_output=True, text=True)
    assert verify.returncode == 0, verify.stderr
    again = subprocess.run(base+['prepare', '--output', str(output)], cwd=m.ROOT, capture_output=True)
    assert again.returncode != 0
    manifest['prospective']['physical_release'] = True
    output.write_bytes(m.encoded(manifest))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    verify = subprocess.run(base+['verify', '--manifest', str(output), '--manifest-sha256', digest], cwd=m.ROOT, capture_output=True)
    assert verify.returncode != 0


def test_cli_future_abi_no_dispatch(tmp_path):
    path = tmp_path/'packets.json'
    path.write_bytes(m.encoded(packets()))
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    cmd = [sys.executable, '-B', '-m', 'experiments.spectral_reachability.'+m.NAME,
           'check-abi', '--packets', str(path), '--packets-sha256', h]
    r = subprocess.run(cmd, cwd=m.ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    x = json.loads(r.stdout)
    assert x['identity_evidence_authenticated'] is False and x['dispatch_allowed'] is False
