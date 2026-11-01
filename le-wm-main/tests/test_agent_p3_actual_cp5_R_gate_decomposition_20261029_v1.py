"""RAM-only real-shape own-schema fixtures; never read production reports."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_actual_cp5_R_gate_decomposition_20261029_v1 as m


def native_row(index='zero', immediate=1., reach=4.):
    # Exact squares make scalar fixtures transparent; full actual native shapes.
    return dict(index=index, status='complete',
                postcontext=dict(latent_history=[[0.]*192, [0.]*192, [immediate**.5]*192],
                                 action_history=[[0.]*10, [0.]*10]),
                signal=dict(nominal=[reach**.5]*192, jacobian=[[0.]*10 for _ in range(192)]),
                solve=dict(params=[0.]*10, linear_terminal_cost=reach, ridge_penalty=0., own_reach_value=reach),
                immediate=immediate, reach=reach)


def fixture():
    flags = dict.fromkeys(m.FLAGS, False)
    sources = {'/RAM/installed_class.py': 'a'*64}
    task = dict(goal=[0.]*192, projection=[[float(i == j) for j in range(192)] for i in range(192)],
                action_mean=[0., 0.], action_std=[1., 1.],
                root_context=dict(latent_history=[[0.]*192 for _ in range(3)], action_history=[[0.]*10 for _ in range(2)]),
                binding={k: 'b'*64 for k in ('root_sha256', 'blind_sha256', 'model_sha256', 'goal_sha256',
                        'projection_sha256', 'normalizer_sha256', 'action_domain_sha256')})
    tasks, pros = [], []
    for root in range(2):
        t = deepcopy(task); t['binding']['task_id'] = f'actual-{root}'
        old = deepcopy(t); old['binding']['task_id'] = f'original-{root}'
        pro = dict(source_slot=root, original_task=old, original_task_sha256=m.digest(old), local_task_sha256=m.digest(t),
                   actual_cp5_absolute_steps=[5, 10, 15], actual_cp5_control_steps=[-5, 0, 5],
                   original_observed_absolute_steps=[0, 5, 10], original_control_steps=[-10, -5, 0],
                   branch_witnesses=[dict(episode_id=f'r{root}-e{i}') for i in range(4)],
                   raw10=[[0., 0.] for _ in range(10)], prefix5=[[0., 0.] for _ in range(5)])
        tasks.append(t); pros.append(pro)
    roster = [dict(slot=i, root=r, horizon=h) for i, (r, h) in enumerate(((0, 1), (0, 3), (1, 1), (1, 3)))]
    cp5 = dict(path=str(m.CP5), sha256=m.CP5_SHA)
    prepared = str(m.ROOT/'results'/m.AUTH['prepared'][0])
    p = dict(schema=m.PRODUCER+'.prepare', tasks=tasks, provenance=pros, sources=sources, roster=roster,
             cp5_tasks=cp5, rules=dict(root_context_role=m.ROLE, actual_cp5_start=True), expected_model_hash='model', **flags)
    r = dict(schema=m.PRODUCER+'.report', status='four_slots_settled_not_admitted', errors=[],
             prepared=prepared, prepare_sha256=m.AUTH['prepared'][1], sources_verified=True,
             source_manifest_sha256=m.digest(sources), model_before=dict(original_state_dict='model'),
             model_after=dict(original_state_dict='model'), classes_before=[dict(path='/RAM/installed_class.py', sha256='a'*64)],
             classes_after=[dict(path='/RAM/installed_class.py', sha256='a'*64)], rows=[], **flags)
    c = dict(schema=m.PRODUCER+'.consume', status='actual_cp5_R_model_deliveries_verified_NOT_physical',
             denominator=4, prepared=prepared, prepare_sha256=m.AUTH['prepared'][1], report_sha256=m.AUTH['report'][1],
             cp5_tasks=cp5, protocol_sha256=m.PROTOCOL_SHA, root_context_role=m.ROLE, arms=[], **flags)
    for identity in roster:
        root = identity['root']
        candidates = [dict(native_row(i), eligible=False) for i in range(8)]
        g = dict(control=native_row(), rows=candidates, eligible_indices=[], status='no_witness')
        result = dict(schema=m.PRODUCER+'.group', backend_kind='HOST_CUDA', root_context_role=m.ROLE,
                      actual_cp5_start=True, horizon=identity['horizon'], errors=[], task_sha256=m.digest(tasks[root]),
                      provenance_sha256=m.digest(pros[root]), source_closure_sha256=m.digest(sources),
                      R_groups=[g], resource_receipt=dict(attempted={'predict': 38}), eligible_order=[],
                      status='no_witness', model_delivery_verified=False, delivery=None, **flags)
        r['rows'].append(dict(**identity, status='no_witness', result=result))
        c['arms'].append(dict(**identity, status='no_witness', complete=True, resources=deepcopy(result['resource_receipt']),
                              eligible_order=[], model_delivery_verified=False, no_witness=True, delivery=None, **flags))
    # JSON roundtrip is intentional: never hide a real saved-schema mismatch.
    return json.loads(json.dumps([p, r, c]))


def test_actual_shape_all36_all32_and_real_receipt_flag():
    p, r, c = fixture()
    assert 'C3' not in c and c['full_C3_demonstrated'] is False
    out = m.explain(p, r, c)
    assert (out['groups_count'], out['items_count'], out['candidates_count']) == (4, 36, 32)
    assert len(out['groups']) == 4
    for g in out['groups']:
        assert len(g['items']) == 9
        assert g['items'][0]['gate_applies'] is False
        assert g['items'][0]['report_origin_pointer'] == f"/rows/{g['slot']}/result/R_groups/0/control"
        assert g['items'][8]['report_origin_pointer'] == f"/rows/{g['slot']}/result/R_groups/0/rows/7"
        assert all(len(x['rejection_factors']) == 2 for x in g['items'][1:])
        assert g['provenance']['actual_cp5_absolute_steps'] == [5, 10, 15]
    assert out['upstream_fullstrict_rebuilt_here'] is False
    assert 'NOT a second' in out['arithmetic_reuse']


def test_all_rejection_combinations_and_gain_drift():
    g = fixture()[1]['rows'][0]['result']['R_groups'][0]
    # (immediate, reach): both fail / immediate fail / reach fail / neither fail.
    cases = [(1., 4.), (1., 1.), (4., 4.), (4., 1.)]
    g['rows'] = [dict(native_row(i, *cases[i % 4]), eligible=i % 4 == 3) for i in range(8)]
    g['eligible_indices'] = [3, 7]; g['status'] = 'complete'
    out = m.decompose(g, [0.]*192)
    assert [len(x['rejection_factors']) for x in out['items'][1:5]] == [2, 1, 1, 0]
    assert out['counts'] == dict(nongreedy=4, better_reach=4, eligible=2)
    for item in out['items']:
        d = item['delta']
        assert d['reach'] == pytest.approx(d['immediate']+d['nominal_drift']-d['bounded_linear_gain'])


def test_exact_gate_does_not_use_arithmetic_tolerance():
    g = fixture()[1]['rows'][0]['result']['R_groups'][0]
    g['rows'][0]['immediate'] += 1e-13
    g['rows'][0]['reach'] -= 1e-13
    g['rows'][0]['eligible'] = True; g['eligible_indices'] = [0]; g['status'] = 'complete'
    out = m.decompose(g, [0.]*192)
    assert out['items'][1]['eligible'] is True


def test_nonzero_bounded_params_full_j_and_ridge_terms():
    row = native_row()
    row['signal']['jacobian'] = [[-.25]+[0.]*9 for _ in range(192)]
    row['solve']['params'] = [.5]+[0.]*9
    linear = 1.875**2
    penalty = .002*.25/192
    row['solve'].update(linear_terminal_cost=linear, ridge_penalty=penalty,
                        own_reach_value=linear+penalty)
    row['reach'] = linear+penalty
    s = m.component(row, [0.]*192)
    assert s['nominal_drift'] == 3.
    assert s['bounded_linear_gain'] == pytest.approx(4.-linear-penalty)
    row['solve']['ridge_penalty'] = 0.
    with pytest.raises(ValueError): m.component(row, [0.]*192)


@pytest.mark.parametrize('field', ['schema', 'status', 'C3', 'missing_full_C3', 'denominator', 'reportsha',
    'preparesha', 'prepared', 'cp5', 'protocol', 'role', 'arms', 'rows', 'slot', 'complete', 'resources',
    'task', 'provenance', 'sourceclosure', 'sourceverified', 'model', 'class', 'backend', 'resultflag',
    'rootcontext', 'actioncontext', 'control0', 'branches', 'raw', 'normalizer', 'projection',
    'candidate', 'zero', 'gate', 'eligibleorder', 'delivery', 'jacobian', 'nominal', 'nonfinite', 'scalar'])
def test_fail_closed_mutations(field):
    p, r, c = fixture(); a = c['arms'][0]; row = r['rows'][0]; v = row['result']; g = v['R_groups'][0]
    if field == 'schema': c['schema'] = 'old.consume'
    elif field == 'status': c['status'] = 'all4_native_R_groups_verified_not_admitted'
    elif field == 'C3': c['full_C3_demonstrated'] = True; c['C3'] = False
    elif field == 'missing_full_C3': del c['full_C3_demonstrated']; c['C3'] = False
    elif field == 'denominator': c['denominator'] = 3
    elif field == 'reportsha': c['report_sha256'] = '0'*64
    elif field == 'preparesha': r['prepare_sha256'] = '0'*64
    elif field == 'prepared': c['prepared'] = 'placeholder'
    elif field == 'cp5': c['cp5_tasks'] = {}
    elif field == 'protocol': c['protocol_sha256'] = '0'*64
    elif field == 'role': c['root_context_role'] = 'control0'
    elif field == 'arms': c['arms'].pop()
    elif field == 'rows': r['rows'].reverse()
    elif field == 'slot': row['slot'] = True
    elif field == 'complete': a['complete'] = False
    elif field == 'resources': a['resources'] = {}
    elif field == 'task': v['task_sha256'] = '0'*64
    elif field == 'provenance': v['provenance_sha256'] = '0'*64
    elif field == 'sourceclosure': v['source_closure_sha256'] = '0'*64
    elif field == 'sourceverified': r['sources_verified'] = False
    elif field == 'model': r['model_after'] = {}
    elif field == 'class': r['classes_before'][0]['sha256'] = '0'*64
    elif field == 'backend': v['backend_kind'] = 'RAM_FIXTURE'
    elif field == 'resultflag': v['full_C3_demonstrated'] = True
    elif field == 'rootcontext': p['tasks'][0]['root_context']['latent_history'].pop()
    elif field == 'actioncontext': p['tasks'][0]['root_context']['action_history'].pop()
    elif field == 'control0': p['provenance'][0]['actual_cp5_absolute_steps'] = [0, 5, 10]
    elif field == 'branches': p['provenance'][0]['branch_witnesses'].pop()
    elif field == 'raw': p['provenance'][0]['raw10'][9][0] = 2.
    elif field == 'normalizer': p['tasks'][0]['action_std'][0] = 0.
    elif field == 'projection': p['tasks'][0]['projection'][0][0] = 2.
    elif field == 'candidate': g['rows'].pop()
    elif field == 'zero': g['control']['index'] = 0
    elif field == 'gate': g['rows'][0]['eligible'] = True
    elif field == 'eligibleorder': a['eligible_order'] = [0]
    elif field == 'delivery': v['delivery'] = {'fake': True}; a['delivery'] = {'fake': True}
    elif field == 'jacobian': g['rows'][0]['signal']['jacobian'][0].pop()
    elif field == 'nominal': g['rows'][0]['signal']['nominal'].pop()
    elif field == 'nonfinite': g['rows'][0]['signal']['nominal'][0] = float('nan')
    elif field == 'scalar': g['rows'][0]['reach'] = 3.
    with pytest.raises((ValueError, KeyError)):
        m.explain(p, r, c)


def test_read_pinned_and_complete_closure(tmp_path):
    p = tmp_path/'data.json'; p.write_bytes(b'{"x":1}')
    assert m.read_pinned(p, m.sha(p)) == {'x': 1}
    m.check_sources({str(p): m.sha(p)})
    with pytest.raises(ValueError): m.read_pinned(p, '0'*64)
    with pytest.raises(ValueError): m.check_sources({str(p): '0'*64})
    with pytest.raises(ValueError): m.check_sources({})
    p.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError): m.read_pinned(p, m.sha(p))
    p.write_bytes(b'{"x":NaN}')
    with pytest.raises(ValueError): m.read_pinned(p, m.sha(p))


def test_cli_full_result_verify_and_exclusive_write(tmp_path, monkeypatch, capsys):
    result = m.explain(*fixture())
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(m, 'analyze', lambda: deepcopy(result))
    output = tmp_path/'results'/(m.NAME+'_fixture.json')
    m.main(['analyze', '--output', str(output)])
    m.main(['verify', '--output', str(output)])
    assert json.loads(output.read_bytes())['items_count'] == 36
    with pytest.raises(FileExistsError): m.main(['analyze', '--output', str(output)])
    changed = json.loads(output.read_bytes()); changed['groups'][3]['items'][8]['delta']['reach'] = 10.
    output.write_bytes(m.arithmetic.encode(changed))
    with pytest.raises(ValueError): m.main(['verify', '--output', str(output)])
    with pytest.raises(ValueError): m.main(['analyze', '--output', str(tmp_path/'bad.json')])


def test_cold_import_and_help_no_neural_producer():
    code = ('import sys; from experiments.spectral_reachability import '+m.NAME+' as m; '
            'assert not any(x in sys.modules for x in ("torch","numpy","pickle")); '
            'assert not any(x.endswith(m.PRODUCER) for x in sys.modules); m.main(["--help"])')
    run = subprocess.run([sys.executable, '-c', code], cwd=m.ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'analyze' in run.stdout and 'verify' in run.stdout


def test_digest_is_producer_compact_not_output_newline():
    assert m.canonical({'b': 1, 'a': 2}) == b'{"a":2,"b":1}'
    assert m.digest({'a': 1}) != m.hashlib.sha256(m.arithmetic.encode({'a': 1})).hexdigest()
