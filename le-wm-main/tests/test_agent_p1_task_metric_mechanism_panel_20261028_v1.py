"""Offline RAM arithmetic/metadata regressions; no model or environment construction."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_task_metric_mechanism_panel_20261028_v1 as m


def metrics(distance):
    # Same original whole-body wrapper formula; outline is not task goal.
    s = [distance, 0., 0., 0., 0., 0., 0.]
    return m.base.task_metrics(s, [0.]*7, [0.]*3, [256., 256., 0.], distance < 20, distance, -distance)


def rows():
    result = []
    for i in range(40):
        task, rem = divmod(i, 10); role, word = divmod(rem, 5)
        row = dict(index=i, root_slot=task//2, condition=('off', 'on')[task%2], role=('Z', 'S')[role],
            library_index=word, library_label=m.teacher.LIBRARY[word], anchor_id=str(i//5), task_id=str(task),
            goal192=[0.]*192, goal_binding=dict(task=task), suffix_raw_bits=str(word),
            physical_goal=[0.]*7, outline_pose=[256., 256., 0.], physical_metrics=metrics(30.+word))
        row.update(m.base.rank.scalar_terms(np.full(192, word+1.), None if i == 12 else np.full(192, 5.-word), np.zeros(192)))
        boundaries = {str(t): dict(absolute_step=t, goal=[0.]*7, outline_pose=[256., 256., 0.],
            metrics=metrics(d)) for t, d in ((10, 40.), (15, 50.+word), (20, 30.+word))}
        result.append(m.row_panel(row, boundaries))
    return result


def test_all_fixed_pairs_no_success_filter_or_missing_drop():
    report = m.assemble(rows())
    assert len(report['rows']) == 40 and len(report['task_groups']) == 4
    assert [len(g['pairs']) for g in report['task_groups']] == [45]*4
    assert [len(g['pairs']) for g in report['role_groups']] == [10]*8
    assert len(report['same_word_ZS_pairs']) == 20
    assert report['task_all_pair_summary']['predicted_Q_to_actual_Q']['unknown'] == 9
    assert report['within_role_summary']['predicted_Q_to_actual_Q']['unknown'] == 4
    assert report['same_word_ZS_summary']['predicted_Q_to_actual_Q']['unknown'] == 1
    assert report['rows'][12]['actual_Q'] is None
    assert report['rows'][12]['boundary_trace']['20']['metrics']['native_distance'] == 32.


def test_success_ties_are_not_reversals_and_coverage_unknown():
    report = m.assemble(rows())
    stats = report['task_all_pair_summary']['physical_metrics']
    assert stats['task_success']['predicted_Q_to_physical']['ties'] == 180
    assert stats['task_success']['actual_Q_to_physical']['ties'] == 171
    assert stats['task_success']['actual_Q_to_physical']['reversal_count'] == 0
    assert stats['native_coverage']['predicted_Q_to_physical']['unknown'] == 180
    assert stats['object_only_coverage']['actual_Q_to_physical']['unknown'] == 180


def test_observed_worse_then_better_is_not_identified_non_greedy_or_capability():
    row = rows()[0]; p = row['trajectory_patterns']['native_distance']
    assert p['prefix_worsens'] and p['suffix_improves'] and p['observed_worse_then_better']
    assert p['truly_physical_non_greedy'] is p['capability_change'] is None
    assert row['trajectory_patterns']['native_coverage']['observed_worse_then_better'] is None
    assert row['segment_changes']['prefix']['native_distance']['raw_end_minus_start'] == 10.
    assert row['segment_changes']['suffix']['native_distance']['raw_end_minus_start'] == -20.


def test_pair_axes_and_orientations_no_combined_score():
    r = rows(); p = m.pair(r[0], r[1])
    assert p['predicted_Q_to_actual_Q']['reversal'] is True
    assert p['physical_metrics']['native_distance']['raw_delta'] == -1.
    assert p['physical_metrics']['native_distance']['actual_Q_to_physical']['reversal'] is True
    assert p['physical_metrics']['task_success']['orientation'] == -1
    assert set(p['physical_metrics']) == set(m.base.METRICS)
    assert p['physical_metrics']['native_distance']['segment_comparisons']['suffix']['oriented_change_delta'] == 0.


@pytest.mark.parametrize('change', ['drop', 'reorder', 'fill12', 'missing_other', 'task', 'goal', 'outline', 'condition', 'anchor', 'suffix'])
def test_roster_or_illegal_pair_mutations_rejected(change):
    r = rows()
    if change == 'drop': r.pop()
    elif change == 'reorder': r[0], r[1] = r[1], r[0]
    elif change == 'fill12': r[12]['actual_Q'] = 0.
    elif change == 'missing_other': r[0]['actual_Q'] = None
    elif change == 'task': r[1]['task_id'] = 'different'
    elif change == 'goal': r[1]['physical_goal'][0] = 1.
    elif change == 'outline': r[1]['outline_pose'][0] = 0.
    elif change == 'condition': r[1]['condition'] = 'on'
    elif change == 'anchor': r[1]['anchor_id'] = 'different'
    else: r[5]['suffix_raw_bits'] = 'different'
    with pytest.raises(ValueError): m.assemble(r)


@pytest.mark.parametrize('change', ['missing', 'goal', 'outline', 'step', 'terminal', 'nan'])
def test_boundary_mutations_rejected(change):
    row = rows()[0]; b = deepcopy(row['boundary_trace'])
    if change == 'missing': del b['10']
    elif change == 'goal': b['15']['goal'][0] = 1.
    elif change == 'outline': b['10']['outline_pose'][0] = 1.
    elif change == 'step': b['10']['absolute_step'] = 9
    elif change == 'terminal': b['20']['metrics']['native_distance'] = 1.
    else: b['15']['metrics']['native_distance'] = float('nan')
    row = {k: v for k, v in row.items() if k not in ('boundary_trace', 'segment_changes', 'trajectory_patterns', 'causal_mechanism_identified')}
    with pytest.raises(ValueError): m.row_panel(row, b)


@pytest.mark.parametrize('a,b,label,reversal', [(0.,0.,'both_tie',False), (0.,1.,'left_tie_only',False),
    (1.,0.,'right_tie_only',False), (1.,-1.,'discordant',True), (1.,2.,'concordant',False), (None,1.,'unknown',None)])
def test_exact_sign_classifications(a, b, label, reversal):
    c = m.base.classify(a, b)
    assert c['classification'] == label and c['reversal'] is reversal
    assert m.tally([c])['fixed'] == 1


def test_tiny_nonzero_not_relaxed_to_tie():
    assert m.base.classify(1e-300, -1e-300)['reversal'] is True


def test_whole_report_verify_rejects_tampering_even_rehashed(tmp_path):
    p = tmp_path/'panel.json'; a = dict(schema=m.NAME+'.analysis', sample=1)
    p.write_bytes(m.canonical(a))
    assert m.verify(p, m.sha(p), a)['verified']
    p.write_bytes(m.canonical(dict(a, sample=2)))
    with pytest.raises(ValueError): m.verify(p, m.sha(p), a)
    with pytest.raises(ValueError): m.verify(p, '0'*64, a)


def test_full_panel_and_protocol_json_roundtrip(tmp_path):
    a = dict(schema=m.NAME+'.analysis', protocol=dict(segments=m.SEGMENTS), **m.assemble(rows()))
    p = tmp_path/'panel.json'; p.write_bytes(m.canonical(a))
    assert json.loads(p.read_text()) == a
    assert m.verify(p, m.sha(p), a)['verified']


def test_exclusive_write_and_no_overwrite(tmp_path, monkeypatch):
    (tmp_path/'results').mkdir(); monkeypatch.setattr(m, 'ROOT', tmp_path)
    p = tmp_path/'results'/(m.NAME+'_test.json')
    m.write(p, dict(ok=True))
    assert json.loads(p.read_text()) == dict(ok=True)
    with pytest.raises(FileExistsError): m.write(p, dict(ok=False))
    with pytest.raises(ValueError): m.write(tmp_path/'results/MAIN.json', {})


def test_cross_role_different_word_all_included_but_not_causal():
    r = rows(); p = m.pair(r[0], r[6])
    assert p['prefix_contexts_differ'] and not p['same_suffix_word'] and not p['matched_suffix_bits']
    assert not p['causal_interpretation']
    with pytest.raises(ValueError): m.pair(r[0], r[10])


def test_stage_metrics_not_replaced_by_endpoint():
    r = rows(); p = m.pair(r[0], r[1])['physical_metrics']['native_distance']
    assert p['lower_is_better_delta'] == -1.
    assert p['segment_comparisons']['suffix']['oriented_change_delta'] == 0.
    assert r[0]['boundary_trace']['10']['metrics']['native_distance'] == 40.
    assert r[0]['boundary_trace']['15']['metrics']['native_distance'] == 50.


def test_finite_unknown_not_zero():
    assert m.delta(None, 1.) is None
    assert m.delta(1., None) is None
    with pytest.raises(ValueError): m.delta(float('inf'), 1.)


@pytest.mark.parametrize('mutation', [None, 'file_sha', 'dtype', 'shape', 'position', 'goal_postbind', 'reward'])
def test_real_info_schema_prebind_goal_and_strict_checks(monkeypatch, mutation):
    arrays = dict(goal_state=np.array([10.,0.,0.,0.,0.,0.,0.]), goal_pose=np.array([256.,256.,0.]),
        block_pose=np.zeros(3), pos_agent=np.array([40.,0.]), vel_agent=np.zeros(2))
    tree = {k: dict(array=k, dtype=a.dtype.str, shape=list(a.shape)) for k, a in arrays.items()}
    event = dict(phase='warmup', step=10, state=[40.,0.,0.,0.,0.,0.,0.], native_success=False,
        native_distance=30., reward=-30., terminated=False, truncated=False, event_sha256='e'*64,
        info=dict(file='info.npz', sha256='a'*64, tree=tree))
    row = rows()[0]
    if mutation == 'file_sha': event['info']['sha256'] = 'b'*64
    elif mutation == 'dtype': tree['goal_state']['dtype'] = '<f4'
    elif mutation == 'shape': tree['goal_state']['shape'] = [6]
    elif mutation == 'position': arrays['pos_agent'][0] = 41.
    elif mutation == 'goal_postbind': event.update(phase='prefix', step=15)
    elif mutation == 'reward': event['reward'] = 0.
    class Archive:
        def __enter__(self): return arrays
        def __exit__(self, *args): return False
    monkeypatch.setattr(m.np, 'load', lambda *args, **kwargs: Archive())
    monkeypatch.setattr(m.teacher, 'flat', lambda *args: Path('/ram/info.npz'))
    monkeypatch.setattr(m, 'sha', lambda *args: 'a'*64)
    if mutation:
        with pytest.raises(ValueError): m.read_boundary(event, row, Path('/ram'), {})
    else:
        receipts = {}; out = m.read_boundary(event, row, Path('/ram'), receipts)
        assert out['returned_native_metrics']['native_distance'] == 30.
        assert out['metrics']['native_distance'] == 40.
        assert out['native_reward'] == -30. and out['returned_goal'] != out['goal']
        assert out['warmup_return_precedes_task_goal_bind'] is True and len(receipts) == 5
