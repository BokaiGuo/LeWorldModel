"""RAM-only arithmetic and witness mutation tests. Never execute a model/env."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_teacher_library40_history_localization_20261028_v1 as m


def lane():
    warm = np.concatenate((np.tile(np.asarray([.1, 0.], '<f4'), (5, 1)), np.tile(np.asarray([0., .1], '<f4'), (5, 1))))
    prefix = np.arange(10, dtype='<f4').reshape(5, 2)/32
    suffix = -prefix.copy()
    raw = np.concatenate((warm[5:], prefix))
    blind = dict(raw_history=raw.tolist(), context=dict(action_history=m.teacher.kernel.normalized_history(raw).tolist()))
    events = []
    for phase, first, commands in [('warmup', 1, warm), ('prefix', 11, prefix), ('suffix', 1, suffix)]:
        for j, a in enumerate(commands):
            events.append(dict(kind='raw_return', job_digest='job', phase=phase, step=first+j, raw_bits=m.bits(a),
                               actual_bits=m.bits(a), actual_dtype='<f4', event_sha256=str(len(events))))
    return warm, prefix, suffix, blind, events


def test_exact_history_and_distinct_index_conventions():
    warm, prefix, suffix, blind, events = lane()
    h = m.history_contract(warm, prefix, suffix, blind)
    assert h['teacher_latent_absolute_steps'] == [5, 10, 15]
    assert h['predictor_action_absolute_steps'] == [list(range(6, 11)), list(range(11, 16)), list(range(16, 21))]
    assert h['suffix_journal_steps'] == [1, 2, 3, 4, 5]
    assert h['suffix_absolute_steps'] == [16, 17, 18, 19, 20]
    assert len(m.raw_trace(events, 'job', warm, prefix, suffix)) == 20


@pytest.mark.parametrize('mutation', ['old_warm', 'shift_raw', 'prefix_swap', 'old_tokens', 'zero_warm', 'shape'])
def test_history_mutations_fail_closed(mutation):
    warm, prefix, suffix, blind, _ = lane()
    if mutation == 'old_warm': blind['raw_history'][:5] = warm[:5].tolist()
    elif mutation == 'shift_raw': blind['raw_history'] = np.roll(blind['raw_history'], 1, axis=0).tolist()
    elif mutation == 'prefix_swap': prefix = prefix[::-1].copy()
    elif mutation == 'old_tokens': blind['context']['action_history'] = m.teacher.kernel.normalized_history(warm).tolist()
    elif mutation == 'zero_warm': warm[:] = 0
    else: suffix = suffix[:-1]
    with pytest.raises(ValueError): m.history_contract(warm, prefix, suffix, blind)


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'phase', 'offbyone', 'actual', 'raw', 'dtype', 'job', 'signzero', 'reorder'])
def test_actual_return_mutations_fail_closed(mutation):
    warm, prefix, suffix, _, events = lane()
    if mutation == 'drop': events.pop()
    elif mutation == 'duplicate': events.append(deepcopy(events[-1]))
    elif mutation == 'phase': events[-1]['phase'] = 'prefix'
    elif mutation == 'offbyone': events[-1]['step'] = 20
    elif mutation == 'actual': events[-1]['actual_bits'] = m.bits([0., 0.])
    elif mutation == 'raw': events[-1]['raw_bits'] = m.bits([0., 0.])
    elif mutation == 'dtype': events[-1]['actual_dtype'] = '<f8'
    elif mutation == 'job': events[-1]['job_digest'] = 'other'
    elif mutation == 'signzero': events[0]['actual_bits'] = m.bits([.1, -0.])
    else: events[0], events[1] = events[1], events[0]
    with pytest.raises(ValueError): m.raw_trace(events, 'job', warm, prefix, suffix)


def target():
    goal = [0.125]*192
    context = dict(x='same saved context')
    binding = dict(pixel_sha256='goalpng', state=[1.]*7, pose=[256., 256., .785])
    task = dict(task_id='task', task=dict(root_context=context, goal=goal, projection=np.eye(192).tolist(), goal_source_sha256='goalpng'),
        task_contract=dict(root_slot=0, condition='off', blind_sha256='blind', pixel_sha256=['a','b','c','goalpng'], renderer_only=True, goal_pose_edited=False))
    anchor = dict(task_id='task', blind_sha256='blind', root_context=context, goal=goal)
    blind = dict(goal=goal, q=np.eye(192).tolist())
    row = dict(task_id='task', blind_sha256='blind', root_slot=0, condition='off', goal192=goal, goal_binding=binding)
    plan = dict(task_id='task', blind_sha256='blind')
    obs = dict(task_id='task', blind_sha256='blind', goal=binding['state'], goal_pose=binding['pose'], goal_pixel_sha256='goalpng')
    return deepcopy((task, anchor, blind, row, plan, obs))


def test_target_preserves_native_goal_vs_outline():
    out = m.target_contract(*target())
    assert out['native_goal_state'] == [1.]*7
    assert out['native_outline_pose'] == [256., 256., .785]
    assert out['outline_is_taskgoal'] is False


@pytest.mark.parametrize('mutation', ['task', 'condition', 'blind', 'latent_goal', 'native_goal', 'pose', 'pixels', 'Q', 'renderer', 'edited'])
def test_target_mutations_fail_closed(mutation):
    task, anchor, blind, row, plan, obs = target()
    if mutation == 'task': plan['task_id'] = 'other'
    elif mutation == 'condition': row['condition'] = 'on'
    elif mutation == 'blind': plan['blind_sha256'] = 'other'
    elif mutation == 'latent_goal': row['goal192'] = [1.]*192
    elif mutation == 'native_goal': obs['goal'] = [2.]*7
    elif mutation == 'pose': obs['goal_pose'] = [0.]*3
    elif mutation == 'pixels': obs['goal_pixel_sha256'] = 'other'
    elif mutation == 'Q': blind['q'][0][0] = 2
    elif mutation == 'renderer': task['task_contract']['renderer_only'] = False
    else: task['task_contract']['goal_pose_edited'] = True
    with pytest.raises(ValueError): m.target_contract(task, anchor, blind, row, plan, obs)


def analysis():
    rows = []
    for i in range(40):
        task, rem = divmod(i, 10); role, word = divmod(rem, 5)
        p, a, goal = np.full(192, word+1.), np.full(192, 5.-word), np.zeros(192)
        r = dict(index=i, root_slot=task//2, condition=('off','on')[task%2], role=('Z','S')[role],
            library_index=word, library_label=m.teacher.LIBRARY[word], anchor_id=str(i//5), goal192=goal.tolist(),
            goal_binding=dict(task=task), suffix_raw_bits=m.bits(np.full((5,2), word)))
        r.update(m.rank.scalar_terms(p, None if i == 12 else a, goal)); rows.append(r)
    return m.rank.assemble(rows)


def checks_for(a):
    return [dict({k:r[k] for k in ('index','root_slot','condition','role','library_label','actual_Q','predicted_Q')},
                 input_binding_status='saved_exact') for r in a['rows']]


def test_full_denominators_and_unknown_preserved_no_winner():
    a = analysis(); c = checks_for(a); out = m.locate(a, c)
    assert out['fixed_slots'] == 40 and len(out['groups']) == 8
    assert out['within_role_summary']['available_pairs'] == 76
    assert out['within_role_summary']['unknown_pairs'] == 4
    assert out['same_word_ZS_summary']['unknown_pairs'] == 1
    assert out['rank_rows'] == dict(fixed=40, known=35, unknown=5)
    assert out['rows'] == c and 'winner' not in out
    assert all(p['role'] in ('Z','S') and p['input_binding_verified'] for p in out['reversals'])


@pytest.mark.parametrize('mutation', ['drop_check', 'duplicate_check', 'drop_row', 'fill_missing', 'fake_summary', 'fake_pair', 'unverified', 'cross_condition_check'])
def test_localization_mutations_rejected(mutation):
    a = analysis(); checks = checks_for(a)
    if mutation == 'drop_check': checks.pop()
    elif mutation == 'duplicate_check': checks[-1]['index'] = 38
    elif mutation == 'drop_row': a['rows'].pop()
    elif mutation == 'fill_missing': a['rows'][12]['actual_Q'] = 0.
    elif mutation == 'fake_summary': a['within_role_summary']['observed_reversals'] = 0
    elif mutation == 'fake_pair': a['groups'][0]['pairs'][0]['reversal'] = False
    elif mutation == 'unverified': checks[0]['input_binding_status'] = 'pending'
    else: checks[0]['condition'] = 'on'
    with pytest.raises(ValueError): m.locate(a, checks)


def test_signed_zero_and_shape_are_not_equal():
    with pytest.raises(ValueError): m.equal_bits([0.], [-0.], 'signed zero')
    with pytest.raises(ValueError): m.equal_bits([[1.,2.]], [1.,2.], 'shape')
